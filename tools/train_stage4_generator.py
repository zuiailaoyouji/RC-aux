#!/usr/bin/env python3
"""Train the Stage 4 high-level generator for at least three seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage4_generator import (
    GeneratorTrajectoryDataset,
    HighLevelSubgoalGenerator,
    TAU_MODEL_STEPS,
    assert_protocol_consistency,
    atomic_torch_save,
    build_generator_sample_refs,
    compute_residual_statistics,
    generator_checkpoint,
    generator_loss,
    load_checkpoint_protocol,
    load_latent_cache,
    load_stage2_protocol,
    split_cached_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen-backbone Stage 4 latent generator."
    )
    parser.add_argument(
        "--latent-cache",
        type=Path,
        default=Path("outputs/stage3_progress_latents.pt"),
    )
    parser.add_argument(
        "--stage2-report",
        type=Path,
        default=Path("outputs/rc_filter_only_tau3.json"),
    )
    parser.add_argument(
        "--progress-ranker",
        type=Path,
        default=Path("outputs/stage3_progress_ranker.pt"),
    )
    parser.add_argument("--seeds", default="3072,3073,3074")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/stage4_generator_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage4_generator_training.json"),
    )
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if len(seeds) < 3:
        raise ValueError("the formal Stage 4 experiment requires at least 3 seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("generator seeds must be unique")
    return seeds


def validate_args(args: argparse.Namespace) -> list[int]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for name in ("batch_size", "max_epochs", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("optimizer and gradient clipping values are invalid")
    return parse_seeds(args.seeds)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_epoch(
    model: HighLevelSubgoalGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "smooth_l1": 0.0, "cosine_loss": 0.0}
    examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        predicted, _ = model(
            batch["history_latents"],
            batch["history_padding_mask"],
            batch["goal_latent"],
        )
        loss, smooth_l1, cosine = generator_loss(
            predicted, batch["target_latent"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = predicted.size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["smooth_l1"] += float(smooth_l1.detach()) * batch_size
        totals["cosine_loss"] += float(cosine.detach()) * batch_size
        examples += batch_size
    return {name: value / examples for name, value in totals.items()}


@torch.inference_mode()
def validate_epoch(
    model: HighLevelSubgoalGenerator,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "smooth_l1": 0.0, "cosine_loss": 0.0}
    examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        predicted, _ = model(
            batch["history_latents"],
            batch["history_padding_mask"],
            batch["goal_latent"],
        )
        loss, smooth_l1, cosine = generator_loss(
            predicted, batch["target_latent"]
        )
        batch_size = predicted.size(0)
        totals["loss"] += float(loss) * batch_size
        totals["smooth_l1"] += float(smooth_l1) * batch_size
        totals["cosine_loss"] += float(cosine) * batch_size
        examples += batch_size
    return {name: value / examples for name, value in totals.items()}


def make_loader(
    dataset: GeneratorTrajectoryDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def train_seed(
    *,
    seed: int,
    train_dataset: GeneratorTrajectoryDataset,
    validation_dataset: GeneratorTrajectoryDataset,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = HighLevelSubgoalGenerator().to(device)
    model.set_residual_normalizer(residual_mean, residual_std)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=args.num_workers,
    )

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_path = checkpoint_dir / f"stage4_generator_seed{seed}.pt"
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        validation_metrics = validate_epoch(
            model,
            validation_loader,
            device=device,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"seed={seed} epoch={epoch}/{args.max_epochs} "
            f"train_smooth_l1={train_metrics['smooth_l1']:.6f} "
            f"validation_smooth_l1={validation_metrics['smooth_l1']:.6f}",
            flush=True,
        )
        if validation_metrics["smooth_l1"] < best_validation:
            best_validation = validation_metrics["smooth_l1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                generator_checkpoint(
                    model,
                    seed=seed,
                    epoch=epoch,
                    validation_smooth_l1=best_validation,
                    metadata=metadata,
                ),
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_smooth_l1": best_validation,
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.max_epochs,
        "history": history,
    }


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, output)


def main() -> int:
    args = parse_args()
    seeds = validate_args(args)
    stage2_protocol = load_stage2_protocol(args.stage2_report)
    stage3_protocol = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(
        stage2_protocol,
        stage3_protocol,
        checkpoint_label="Stage 3 progress ranker",
    )
    cache = load_latent_cache(args.latent_cache)
    train_episodes, validation_episodes, test_episodes = split_cached_episodes(
        cache
    )
    train_refs = build_generator_sample_refs(train_episodes)
    validation_refs = build_generator_sample_refs(validation_episodes)
    test_refs = build_generator_sample_refs(test_episodes)
    train_dataset = GeneratorTrajectoryDataset(train_episodes, train_refs)
    validation_dataset = GeneratorTrajectoryDataset(
        validation_episodes, validation_refs
    )
    residual_mean, residual_std = compute_residual_statistics(train_dataset)
    metadata = {
        "latent_cache": str(args.latent_cache.expanduser().resolve()),
        "latent_cache_policy": cache["metadata"].get("policy"),
        "tau_model_steps": TAU_MODEL_STEPS,
        "eta_r": stage2_protocol["eta_r"],
        "eta_r_source": stage2_protocol["report_path"],
        "stage3_progress_ranker": stage3_protocol["checkpoint_path"],
        "train_episode_range": {"start_inclusive": 0, "end_exclusive": 4000},
        "validation_episode_range": {
            "start_inclusive": 4000,
            "end_exclusive": 5000,
        },
        "test_episode_range": {
            "start_inclusive": 5000,
            "end_exclusive": 10000,
        },
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "test_episode_count": len(test_episodes),
        "train_sample_count": len(train_refs),
        "validation_sample_count": len(validation_refs),
        "test_sample_count": len(test_refs),
        "episode_overlap": 0,
        "history_actions_used": False,
        "future_information_used_as_input": False,
        "loss_terms": ["smooth_l1", "0.1 * cosine_distance"],
        "rc_or_progress_loss_used": False,
    }
    seed_results = []
    for seed in seeds:
        seed_results.append(
            train_seed(
                seed=seed,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                residual_mean=residual_mean,
                residual_std=residual_std,
                args=args,
                metadata=metadata,
            )
        )
        atomic_write_json(
            {
                "stage4_generator_training_complete": False,
                "protocol": metadata,
                "seeds_requested": seeds,
                "seed_results": seed_results,
            },
            args.output,
        )

    report = {
        "stage4_generator_training_complete": len(seed_results) == len(seeds),
        "frozen_modules": ["E_theta", "F_theta", "R_phi", "D_psi"],
        "protocol": metadata,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "grad_clip": args.grad_clip,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "checkpoint_selection": "minimum validation SmoothL1",
        },
        "residual_normalization": {
            "fit_split": "train_only",
            "mean_norm": float(residual_mean.norm()),
            "std_min": float(residual_std.min()),
            "std_mean": float(residual_std.mean()),
            "std_max": float(residual_std.max()),
        },
        "seeds_requested": seeds,
        "seed_results": seed_results,
    }
    atomic_write_json(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"report_path: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
