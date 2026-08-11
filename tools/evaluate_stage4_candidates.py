#!/usr/bin/env python3
"""ARCHIVED: evaluate the retired Stage 4 R_local + D_psi selector."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, TWOROOM_PROFILE
from legacy_stage4_protocol import (
    assert_protocol_consistency,
    load_checkpoint_protocol,
    load_stage2_protocol,
    select_candidate_index,
)
from stage4_generator import (
    GeneratorTrajectoryDataset,
    TAU_MODEL_STEPS,
    build_generator_sample_refs,
    distribution_summary,
    load_generator_checkpoint,
    load_latent_cache,
    sample_subgoal_candidates,
    split_cached_episodes,
)
from tools.train_progress_ranker_stage3 import ProgressRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 4 generation and RC plus D_psi selection."
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("outputs/stage4_generator_training.json"),
    )
    parser.add_argument(
        "--stage2-report",
        type=Path,
        default=Path("outputs/rc_filter_only_tau3.json"),
    )
    parser.add_argument(
        "--latent-cache",
        type=Path,
        default=Path("outputs/stage3_progress_latents.pt"),
    )
    parser.add_argument(
        "--progress-ranker",
        type=Path,
        default=Path("outputs/stage3_progress_ranker.pt"),
    )
    parser.add_argument("--policy", default="tworoom_rcaux/rcaux_tworoom")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/sxw/work/datasets/stable-wm"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260809)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage4_candidate_validation.json"),
    )
    return parser.parse_args()


def load_progress_ranker(
    path: Path,
    *,
    device: torch.device,
) -> tuple[ProgressRanker, dict[str, Any]]:
    checkpoint = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    model = ProgressRanker(
        int(checkpoint["latent_dim"]), tuple(checkpoint["hidden_dims"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def checkpoint_paths(report_path: Path) -> list[Path]:
    report = json.loads(report_path.expanduser().resolve().read_text())
    if report.get("stage4_generator_training_complete") is not True:
        raise ValueError("Stage 4 generator training report is incomplete")
    paths = [Path(item["checkpoint"]) for item in report["seed_results"]]
    if len(paths) < 3:
        raise ValueError("candidate validation requires at least three seeds")
    return paths


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    synchronize(device)
    started = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, time.perf_counter() - started


def pairwise_diversity(candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    l2_values = []
    cosine_values = []
    for candidate_set in candidates:
        first, second = torch.triu_indices(
            candidate_set.size(0), candidate_set.size(0), offset=1
        )
        differences = candidate_set[first] - candidate_set[second]
        l2_values.append(differences.norm(dim=-1).mean())
        normalized = F.normalize(candidate_set, dim=-1)
        cosine_values.append(
            (1.0 - (normalized[first] * normalized[second]).sum(dim=-1)).mean()
        )
    return torch.stack(l2_values), torch.stack(cosine_values)


def append(values: list[float], tensor: torch.Tensor) -> None:
    values.extend(float(value) for value in tensor.detach().cpu().reshape(-1))


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    adapter: RCAuxAdapter,
    progress_ranker: ProgressRanker,
    *,
    num_candidates: int,
    sample_seed: int,
    eta_r: float,
    device: torch.device,
) -> dict[str, Any]:
    generator, checkpoint = load_generator_checkpoint(
        checkpoint_path, device=device
    )
    generator.requires_grad_(False)
    torch.manual_seed(sample_seed + int(checkpoint["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed + int(checkpoint["seed"]))

    values: dict[str, list[float]] = {
        name: []
        for name in (
            "deterministic_l2",
            "deterministic_cosine",
            "best32_l2",
            "best32_cosine",
            "pairwise_l2_diversity",
            "pairwise_cosine_diversity",
            "generated_residual_norm",
            "gt_residual_norm",
            "all_candidate_rc",
            "gt_rc",
            "best_candidate_rc",
            "selected_rc",
            "selected_progress",
            "selected_pairwise_l2_diversity",
            "selected_pairwise_cosine_diversity",
            "generated_best_minus_gt_rc",
            "direct_goal_rc",
            "direct_goal_progress",
        )
    }
    nrc_zero = nrc_one = nrc_multiple = 0
    source_count = candidate_count = rc_pass_count = feasible_count = 0
    generated_coverage = full_coverage = no_candidate = 0
    direct_goal_eligible = direct_goal_selected = no_progress_without_rc = 0
    timing = {"generator": 0.0, "rc": 0.0, "d_psi": 0.0}

    for batch in loader:
        history = batch["history_latents"].to(device)
        mask = batch["history_padding_mask"].to(device)
        goal = batch["goal_latent"].to(device)
        current = batch["current_latent"].to(device)
        target = batch["target_latent"].to(device)
        batch_size = current.size(0)

        deterministic, det_time = timed(
            device,
            lambda: sample_subgoal_candidates(
                generator,
                history,
                mask,
                goal,
                num_candidates=1,
                stochastic=False,
            )[:, 0],
        )
        candidates, sample_time = timed(
            device,
            lambda: sample_subgoal_candidates(
                generator,
                history,
                mask,
                goal,
                num_candidates=num_candidates,
                stochastic=True,
            ),
        )
        timing["generator"] += det_time + sample_time

        deterministic_l2 = (deterministic - target).norm(dim=-1)
        deterministic_cosine = F.cosine_similarity(deterministic, target, dim=-1)
        candidate_l2 = (candidates - target.unsqueeze(1)).norm(dim=-1)
        best_indices = candidate_l2.argmin(dim=1)
        row = torch.arange(batch_size, device=device)
        best = candidates[row, best_indices]
        pair_l2, pair_cosine = pairwise_diversity(candidates)
        residual_norm = (candidates - current.unsqueeze(1)).norm(dim=-1)
        gt_residual_norm = (target - current).norm(dim=-1)

        rc, rc_time = timed(
            device,
            lambda: adapter.reachability(
                current,
                candidates,
                horizon_model_steps=TAU_MODEL_STEPS,
            ),
        )
        gt_rc, gt_rc_time = timed(
            device,
            lambda: adapter.reachability(
                current,
                target,
                horizon_model_steps=TAU_MODEL_STEPS,
            ),
        )
        direct_goal_rc, direct_goal_rc_time = timed(
            device,
            lambda: adapter.reachability(
                current,
                goal,
                horizon_model_steps=TAU_MODEL_STEPS,
            ),
        )
        timing["rc"] += rc_time + gt_rc_time + direct_goal_rc_time

        def score_progress():
            source_score = progress_ranker(current, goal)
            candidate_score = progress_ranker(
                candidates.reshape(-1, candidates.size(-1)),
                goal.repeat_interleave(num_candidates, dim=0),
            ).reshape(batch_size, num_candidates)
            direct_goal_score = progress_ranker(goal, goal)
            return source_score, candidate_score, direct_goal_score

        (source_score, candidate_score, direct_goal_score), d_time = timed(
            device, score_progress
        )
        timing["d_psi"] += d_time
        progress = source_score.unsqueeze(1) - candidate_score
        rc_pass = rc >= eta_r
        feasible_rows = []
        generated_feasible_rows = []
        selected_indices = []
        no_rc_rows = []
        for row_index in range(batch_size):
            combined_score = torch.cat(
                [
                    candidate_score[row_index],
                    direct_goal_score[row_index : row_index + 1],
                ]
            )
            combined_rc = torch.cat(
                [rc[row_index], direct_goal_rc[row_index : row_index + 1]]
            )
            full_selection = select_candidate_index(
                combined_score,
                source_score[row_index],
                rc_scores=combined_rc,
                eta_r=eta_r,
            )
            progress_only = select_candidate_index(
                candidate_score[row_index],
                source_score[row_index],
                rc_scores=None,
            )
            feasible_rows.append(full_selection["feasible"])
            generated_feasible_rows.append(full_selection["feasible"][:-1])
            selected_indices.append(full_selection["selected_index"])
            no_rc_rows.append(progress_only["feasible"])
        feasible = torch.stack(feasible_rows)
        generated_feasible = torch.stack(generated_feasible_rows)
        no_rc_feasible = torch.stack(no_rc_rows)
        nrc = rc_pass.sum(dim=1)
        nrc_zero += int((nrc == 0).sum())
        nrc_one += int((nrc == 1).sum())
        nrc_multiple += int((nrc >= 2).sum())
        no_progress_without_rc += int((~no_rc_feasible.any(dim=1)).sum())

        covered = feasible.any(dim=1)
        generated_covered = generated_feasible.any(dim=1)
        direct_eligible_mask = feasible[:, -1]
        selected_index_tensor = torch.tensor(
            [index if index is not None else 0 for index in selected_indices],
            device=device,
        )
        selected_is_direct = covered & (selected_index_tensor == num_candidates)
        candidate_selected_index = selected_index_tensor.clamp_max(num_candidates - 1)
        selected_rc = rc[row, candidate_selected_index]
        selected_progress = progress[row, candidate_selected_index]
        selected_rc = torch.where(selected_is_direct, direct_goal_rc, selected_rc)
        direct_progress = source_score - direct_goal_score
        selected_progress = torch.where(
            selected_is_direct, direct_progress, selected_progress
        )

        source_count += batch_size
        candidate_count += batch_size * num_candidates
        rc_pass_count += int(rc_pass.sum())
        feasible_count += int(generated_feasible.sum())
        generated_coverage += int(generated_covered.sum())
        full_coverage += int(covered.sum())
        no_candidate += int((~covered).sum())
        direct_goal_eligible += int(direct_eligible_mask.sum())
        direct_goal_selected += int(selected_is_direct.sum())

        append(values["deterministic_l2"], deterministic_l2)
        append(values["deterministic_cosine"], deterministic_cosine)
        append(values["best32_l2"], candidate_l2[row, best_indices])
        append(values["best32_cosine"], F.cosine_similarity(best, target, dim=-1))
        append(values["pairwise_l2_diversity"], pair_l2)
        append(values["pairwise_cosine_diversity"], pair_cosine)
        append(values["generated_residual_norm"], residual_norm)
        append(values["gt_residual_norm"], gt_residual_norm)
        append(values["all_candidate_rc"], rc)
        append(values["gt_rc"], gt_rc)
        append(values["direct_goal_rc"], direct_goal_rc)
        append(values["direct_goal_progress"], direct_progress)
        append(values["best_candidate_rc"], rc.max(dim=1).values)
        append(
            values["generated_best_minus_gt_rc"],
            rc.max(dim=1).values - gt_rc,
        )
        if bool(covered.any()):
            append(values["selected_rc"], selected_rc[covered])
            append(values["selected_progress"], selected_progress[covered])
            append(values["selected_pairwise_l2_diversity"], pair_l2[covered])
            append(
                values["selected_pairwise_cosine_diversity"], pair_cosine[covered]
            )

    return {
        "seed": int(checkpoint["seed"]),
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "deterministic": {
            "gt_l2_error": distribution_summary(values["deterministic_l2"]),
            "gt_cosine_similarity": distribution_summary(
                values["deterministic_cosine"]
            ),
        },
        "best_of_32_gt_oracle_diagnostic": {
            "gt_l2_error": distribution_summary(values["best32_l2"]),
            "gt_cosine_similarity": distribution_summary(values["best32_cosine"]),
        },
        "diversity": {
            "pairwise_l2": distribution_summary(values["pairwise_l2_diversity"]),
            "pairwise_cosine_distance": distribution_summary(
                values["pairwise_cosine_diversity"]
            ),
            "generated_residual_norm": distribution_summary(
                values["generated_residual_norm"]
            ),
            "gt_residual_norm": distribution_summary(values["gt_residual_norm"]),
        },
        "filtering": {
            "source_count": source_count,
            "candidate_count": candidate_count,
            "candidate_rc_pass_rate": rc_pass_count / candidate_count,
            "candidate_rc_and_progress_pass_rate": feasible_count / candidate_count,
            "sources_n_rc_zero_rate": nrc_zero / source_count,
            "sources_n_rc_one_rate": nrc_one / source_count,
            "sources_n_rc_at_least_two_rate": nrc_multiple / source_count,
            "generated_rc_and_progress_coverage": (
                generated_coverage / source_count
            ),
            "rc_and_progress_coverage_with_direct_goal": (
                full_coverage / source_count
            ),
            "no_candidate_rate": no_candidate / source_count,
            "direct_goal_eligible_rate": direct_goal_eligible / source_count,
            "direct_goal_selected_rate": direct_goal_selected / source_count,
            "direct_goal_rc_score": distribution_summary(
                values["direct_goal_rc"]
            ),
            "direct_goal_d_psi_progress": distribution_summary(
                values["direct_goal_progress"]
            ),
            "no_progress_candidate_without_rc_rate": (
                no_progress_without_rc / source_count
            ),
            "all_candidate_rc_score": distribution_summary(
                values["all_candidate_rc"]
            ),
            "selected_rc_score": (
                distribution_summary(values["selected_rc"])
                if values["selected_rc"]
                else None
            ),
            "selected_d_psi_progress": (
                distribution_summary(values["selected_progress"])
                if values["selected_progress"]
                else None
            ),
        },
        "generated_target_rc_shift": {
            "interpretation": (
                "RC-score distribution difference between generated candidates "
                "and true z_t_plus_3 only; this is not an RC calibration claim"
            ),
            "gt_rc_score": distribution_summary(values["gt_rc"]),
            "gt_rc_pass_rate": float(np.mean(np.asarray(values["gt_rc"]) >= eta_r)),
            "best_generated_rc_score": distribution_summary(
                values["best_candidate_rc"]
            ),
            "best_generated_minus_gt_rc": distribution_summary(
                values["generated_best_minus_gt_rc"]
            ),
        },
        "timing_seconds": {
            **timing,
            "generator_per_source": timing["generator"] / source_count,
            "rc_per_source": timing["rc"] / source_count,
            "d_psi_per_source": timing["d_psi"] / source_count,
        },
    }


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, output)


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.num_candidates != 32:
        raise ValueError("the formal Stage 4 protocol requires N=32")
    if args.batch_size < 1 or args.max_test_samples < 0:
        raise ValueError("invalid evaluation size")
    device = torch.device(args.device)
    stage2_protocol = load_stage2_protocol(args.stage2_report)
    stage3_protocol = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(
        stage2_protocol,
        stage3_protocol,
        checkpoint_label="Stage 3 progress ranker",
    )
    cache = load_latent_cache(args.latent_cache)
    _, _, test_episodes = split_cached_episodes(cache)
    refs = build_generator_sample_refs(test_episodes)
    dataset: Any = GeneratorTrajectoryDataset(test_episodes, refs)
    if args.max_test_samples:
        if args.max_test_samples > len(dataset):
            raise ValueError("max-test-samples exceeds the test sample count")
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(
            rng.choice(len(dataset), args.max_test_samples, replace=False)
        )
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=args.cache_dir.expanduser().resolve(),
        device=device,
        use_reachability_cost=False,
    )
    adapter.model.interpolate_pos_encoding = True
    progress_ranker, _ = load_progress_ranker(args.progress_ranker, device=device)
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("E_theta, F_theta, and R_phi must remain frozen")
    if any(parameter.requires_grad for parameter in progress_ranker.parameters()):
        raise RuntimeError("D_psi must remain frozen")

    results = []
    for path in checkpoint_paths(args.training_report):
        stage4_protocol = load_checkpoint_protocol(path, stage=4)
        assert_protocol_consistency(
            stage2_protocol,
            stage4_protocol,
            checkpoint_label=f"Stage 4 generator {path}",
        )
        result = evaluate_checkpoint(
            path,
            loader,
            adapter,
            progress_ranker,
            num_candidates=args.num_candidates,
            sample_seed=args.sample_seed,
            eta_r=stage2_protocol["eta_r"],
            device=device,
        )
        results.append(result)
        atomic_write_json(
            {
                "stage4_candidate_validation_complete": False,
                "seed_results": results,
            },
            args.output,
        )

    report = {
        "stage4_candidate_validation_complete": True,
        "frozen_modules": ["E_theta", "F_theta", "R_phi", "D_psi"],
        "protocol": {
            "test_episode_range": {
                "start_inclusive": 5000,
                "end_exclusive": 10000,
            },
            "test_episode_overlap_with_train_or_validation": 0,
            "tau_model_steps": TAU_MODEL_STEPS,
            "eta_r": stage2_protocol["eta_r"],
            "eta_r_source": stage2_protocol["report_path"],
            "num_candidates": args.num_candidates,
            "stochasticity": "generator dropout only",
            "selection": (
                "32 generated candidates plus direct z_T precheck; RC pass, "
                "positive D_psi progress, minimum D_psi"
            ),
            "gt_used_for_selection": False,
        },
        "seed_results": results,
    }
    atomic_write_json(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"report_path: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
