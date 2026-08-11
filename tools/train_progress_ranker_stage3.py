#!/usr/bin/env python3
"""ARCHIVED: train and validate the retired Stage 3 D_psi ranker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, TWOROOM_PROFILE


class ProgressRanker(nn.Module):
    """Map ``[g, z_goal, g - z_goal]`` to a scalar remaining-cost score."""

    def __init__(self, latent_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        if latent_dim <= 0 or not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("latent_dim and hidden_dims must be positive")

        dimensions = (3 * latent_dim, *hidden_dims, 1)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if latent.shape != goal.shape:
            raise ValueError("latent and goal must have identical shapes")
        if latent.ndim != 2 or latent.size(-1) != self.latent_dim:
            raise ValueError(
                f"latent and goal must be [B,{self.latent_dim}]"
            )
        features = torch.cat([latent, goal, latent - goal], dim=-1)
        return self.network(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train D_psi only from successful trajectory ordering, then "
            "validate it after the fixed Stage 2 RC filter."
        )
    )
    parser.add_argument("--policy", default="tworoom_rcaux/rcaux_tworoom")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/sxw/work/datasets/stable-wm"),
    )
    parser.add_argument("--dataset", default="tworoom.h5")
    parser.add_argument("--device", default="cuda")
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
        "--model-output",
        type=Path,
        default=Path("outputs/stage3_progress_ranker.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage3_progress_ranker.json"),
    )
    parser.add_argument("--rebuild-latent-cache", action="store_true")
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-test-episodes", type=int, default=0)
    parser.add_argument("--max-test-sources-per-episode", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--reachability-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pairs-per-epoch", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--regression-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--min-test-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--require-validation",
        action="store_true",
        help="Return status 1 after writing outputs if Stage 3 does not pass.",
    )
    return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    try:
        hidden_dims = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("hidden-dims must be comma-separated integers") from exc
    if not hidden_dims or min(hidden_dims) <= 0:
        raise ValueError("hidden-dims must contain positive integers")
    return hidden_dims


def validate_args(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for name in (
        "encode_batch_size",
        "reachability_batch_size",
        "epochs",
        "pairs_per_epoch",
        "batch_size",
        "bootstrap_samples",
        "min_test_queries",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in (
        "max_train_episodes",
        "max_test_episodes",
        "max_test_sources_per_episode",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative")
    if args.ranking_margin < 0 or args.regression_weight < 0:
        raise ValueError("ranking-margin and regression-weight must be nonnegative")
    parse_hidden_dims(args.hidden_dims)


def load_stage2_protocol(path: Path) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    report = json.loads(report_path.read_text())
    if report.get("stage2_rc_filter_validated") is not True:
        raise ValueError("the Stage 2 report must have a validated RC filter")

    protocol = report.get("protocol", {})
    tau = int(protocol.get("tau_model_steps", -1))
    if tau != 3:
        raise ValueError("Stage 3 requires the Stage 2 tau_model_steps=3")
    eta_r = float(report.get("threshold", {}).get("eta_r", float("nan")))
    if not np.isfinite(eta_r) or not 0.0 <= eta_r <= 1.0:
        raise ValueError("the Stage 2 report does not contain a valid eta_R")

    train_range = tuple(protocol.get("calibration_episode_range", ()))
    test_range = tuple(protocol.get("test_episode_range", ()))
    if len(train_range) != 2 or len(test_range) != 2:
        raise ValueError("the Stage 2 report must define both episode ranges")
    train_start, train_stop = map(int, train_range)
    test_start, test_stop = map(int, test_range)
    if not (0 <= train_start < train_stop <= test_start < test_stop):
        raise ValueError("Stage 2 train/test episode ranges must be disjoint")

    return {
        "report_path": str(report_path),
        "eta_r": eta_r,
        "tau_model_steps": tau,
        "threshold_source": report["threshold"].get("source"),
        "train_episode_range": [train_start, train_stop],
        "test_episode_range": [test_start, test_stop],
    }


def successful_episode_indices(
    handle: h5py.File,
    episode_range: tuple[int, int] | list[int],
) -> list[int]:
    start, stop = map(int, episode_range)
    if stop > len(handle["ep_offset"]):
        raise ValueError("episode range exceeds the dataset")
    successful = []
    for episode_index in range(start, stop):
        offset = int(handle["ep_offset"][episode_index])
        length = int(handle["ep_len"][episode_index])
        if length < 2:
            continue
        terminal_row = offset + length - 1
        if bool(handle["terminated"][terminal_row]):
            successful.append(episode_index)
    return successful


def limit_episode_indices(
    episode_indices: list[int],
    maximum: int,
    *,
    seed: int,
) -> list[int]:
    if maximum == 0 or maximum >= len(episode_indices):
        return episode_indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(episode_indices, size=maximum, replace=False)
    return sorted(int(value) for value in selected)


def episode_sample_rows(
    offset: int,
    length: int,
    *,
    model_step_env_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonterminal model-step rows plus the exact terminal goal row."""

    if length < 2 or model_step_env_steps <= 0:
        raise ValueError("episode length and model-step size must be positive")
    terminal_row = offset + length - 1
    rows = np.arange(
        offset,
        terminal_row,
        model_step_env_steps,
        dtype=np.int64,
    )
    rows = np.concatenate([rows, np.asarray([terminal_row], dtype=np.int64)])
    remaining = np.ceil(
        (terminal_row - rows) / model_step_env_steps
    ).astype(np.int64)
    return rows, remaining


def latent_cache_metadata(
    *,
    dataset_path: Path,
    policy: str,
    train_episode_indices: list[int],
    test_episode_indices: list[int],
) -> dict[str, Any]:
    dataset_stat = dataset_path.stat()
    return {
        "format_version": 1,
        "dataset_path": str(dataset_path),
        "dataset_size": int(dataset_stat.st_size),
        "dataset_mtime_ns": int(dataset_stat.st_mtime_ns),
        "policy": policy,
        "model_step_env_steps": TWOROOM_PROFILE.model_step_env_steps,
        "train_episode_indices": train_episode_indices,
        "test_episode_indices": test_episode_indices,
    }


def encode_episodes(
    adapter: RCAuxAdapter,
    handle: h5py.File,
    episode_indices: list[int],
    *,
    batch_size: int,
    split_name: str,
) -> list[dict[str, Any]]:
    episode_specs = []
    for episode_index in episode_indices:
        rows, remaining = episode_sample_rows(
            int(handle["ep_offset"][episode_index]),
            int(handle["ep_len"][episode_index]),
            model_step_env_steps=TWOROOM_PROFILE.model_step_env_steps,
        )
        if len(rows) < 3:
            continue
        episode_specs.append((episode_index, rows, remaining))

    if not episode_specs:
        raise RuntimeError(f"no usable successful episodes in {split_name}")
    flat_rows = np.concatenate([spec[1] for spec in episode_specs])
    latent_batches = []
    for start in range(0, len(flat_rows), batch_size):
        stop = min(start + batch_size, len(flat_rows))
        images = np.asarray(handle["pixels"][flat_rows[start:stop]])
        latents = adapter.encode_observation(images)[:, -1].detach().cpu()
        latent_batches.append(latents)
        print(
            f"encode split={split_name} frames={stop}/{len(flat_rows)}",
            flush=True,
        )
    flat_latents = torch.cat(latent_batches, dim=0).to(torch.float32)

    episodes = []
    cursor = 0
    for episode_index, rows, remaining in episode_specs:
        count = len(rows)
        episodes.append(
            {
                "episode_index": int(episode_index),
                "rows": torch.from_numpy(rows.copy()),
                "remaining_model_steps": torch.from_numpy(remaining.copy()),
                "latents": flat_latents[cursor : cursor + count].clone(),
            }
        )
        cursor += count
    if cursor != len(flat_latents):
        raise RuntimeError("encoded trajectory reconstruction failed")
    return episodes


def atomic_torch_save(value: Any, path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)


def load_or_encode_latents(
    adapter: RCAuxAdapter,
    dataset_path: Path,
    train_episode_indices: list[int],
    test_episode_indices: list[int],
    *,
    policy: str,
    cache_path: Path,
    batch_size: int,
    rebuild: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    metadata = latent_cache_metadata(
        dataset_path=dataset_path,
        policy=policy,
        train_episode_indices=train_episode_indices,
        test_episode_indices=test_episode_indices,
    )
    resolved_cache = cache_path.expanduser().resolve()
    if resolved_cache.exists() and not rebuild:
        cached = torch.load(
            resolved_cache,
            map_location="cpu",
            weights_only=False,
        )
        if cached.get("metadata") != metadata:
            raise ValueError(
                "latent cache metadata does not match this run; use "
                "--rebuild-latent-cache"
            )
        return cached["train"], cached["test"], True

    with h5py.File(dataset_path, "r") as handle:
        train = encode_episodes(
            adapter,
            handle,
            train_episode_indices,
            batch_size=batch_size,
            split_name="train",
        )
        test = encode_episodes(
            adapter,
            handle,
            test_episode_indices,
            batch_size=batch_size,
            split_name="test",
        )
    atomic_torch_save(
        {"metadata": metadata, "train": train, "test": test},
        resolved_cache,
    )
    return train, test, False


def sample_pair_batch(
    episodes: list[dict[str, Any]],
    batch_size: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    episode_choices = rng.integers(0, len(episodes), size=batch_size)
    earlier = []
    later = []
    goals = []
    earlier_targets = []
    later_targets = []
    for episode_choice in episode_choices:
        episode = episodes[int(episode_choice)]
        terminal_index = len(episode["latents"]) - 1
        if terminal_index < 2:
            raise RuntimeError("training episode has fewer than two states")
        i = int(rng.integers(0, terminal_index - 1))
        j = int(rng.integers(i + 1, terminal_index))
        scale = float(episode["remaining_model_steps"][0])
        earlier.append(episode["latents"][i])
        later.append(episode["latents"][j])
        goals.append(episode["latents"][-1])
        earlier_targets.append(
            float(episode["remaining_model_steps"][i]) / scale
        )
        later_targets.append(
            float(episode["remaining_model_steps"][j]) / scale
        )
    return (
        torch.stack(earlier).to(device),
        torch.stack(later).to(device),
        torch.stack(goals).to(device),
        torch.tensor(earlier_targets, dtype=torch.float32, device=device),
        torch.tensor(later_targets, dtype=torch.float32, device=device),
    )


def train_ranker(
    ranker: ProgressRanker,
    episodes: list[dict[str, Any]],
    *,
    epochs: int,
    pairs_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    ranking_margin: float,
    regression_weight: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    history = []
    ranker.train()
    for epoch in range(epochs):
        totals = {"loss": 0.0, "ranking_loss": 0.0, "regression_loss": 0.0}
        examples = 0
        while examples < pairs_per_epoch:
            current_batch = min(batch_size, pairs_per_epoch - examples)
            earlier, later, goals, earlier_target, later_target = (
                sample_pair_batch(
                    episodes,
                    current_batch,
                    rng=rng,
                    device=device,
                )
            )
            earlier_score = ranker(earlier, goals)
            later_score = ranker(later, goals)
            ranking_loss = F.margin_ranking_loss(
                earlier_score,
                later_score,
                torch.ones_like(earlier_score),
                margin=ranking_margin,
            )
            regression_loss = 0.5 * (
                F.mse_loss(earlier_score, earlier_target)
                + F.mse_loss(later_score, later_target)
            )
            loss = ranking_loss + regression_weight * regression_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss) * current_batch
            totals["ranking_loss"] += float(ranking_loss) * current_batch
            totals["regression_loss"] += float(regression_loss) * current_batch
            examples += current_batch
        record = {
            "epoch": epoch + 1,
            **{name: value / examples for name, value in totals.items()},
        }
        history.append(record)
        print(
            f"epoch={epoch + 1}/{epochs} loss={record['loss']:.6f} "
            f"ranking={record['ranking_loss']:.6f} "
            f"regression={record['regression_loss']:.6f}",
            flush=True,
        )
    ranker.eval()
    return history


@torch.inference_mode()
def score_episode_states(
    ranker: ProgressRanker,
    episodes: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    all_scores = []
    for episode in episodes:
        latents = episode["latents"]
        goal = latents[-1]
        batches = []
        for start in range(0, len(latents), batch_size):
            values = latents[start : start + batch_size].to(device)
            goals = goal.expand(len(values), -1).to(device)
            batches.append(ranker(values, goals).cpu())
        all_scores.append(torch.cat(batches).numpy())
    return all_scores


def evenly_limited(values: list[int], maximum: int) -> list[int]:
    if maximum == 0 or len(values) <= maximum:
        return values
    positions = np.linspace(0, len(values) - 1, num=maximum)
    return [values[int(round(position))] for position in positions]


def build_test_queries(
    episodes: list[dict[str, Any]],
    *,
    max_sources_per_episode: int,
) -> list[dict[str, Any]]:
    queries = []
    for episode_list_index, episode in enumerate(episodes):
        terminal_index = len(episode["latents"]) - 1
        source_indices = evenly_limited(
            list(range(max(0, terminal_index - 2))),
            max_sources_per_episode,
        )
        for source_index in source_indices:
            candidate_indices = list(range(source_index + 1, terminal_index))
            if len(candidate_indices) < 2:
                continue
            queries.append(
                {
                    "episode_list_index": episode_list_index,
                    "source_index": source_index,
                    "candidate_indices": candidate_indices,
                    "rc_scores": np.empty(len(candidate_indices), dtype=np.float32),
                }
            )
    return queries


@torch.inference_mode()
def score_query_reachability(
    adapter: RCAuxAdapter,
    episodes: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    tau_model_steps: int,
    batch_size: int,
) -> None:
    flat_pairs = [
        (query_index, candidate_position)
        for query_index, query in enumerate(queries)
        for candidate_position in range(len(query["candidate_indices"]))
    ]
    for start in range(0, len(flat_pairs), batch_size):
        chunk = flat_pairs[start : start + batch_size]
        sources = []
        targets = []
        for query_index, candidate_position in chunk:
            query = queries[query_index]
            episode = episodes[query["episode_list_index"]]
            sources.append(episode["latents"][query["source_index"]])
            candidate_index = query["candidate_indices"][candidate_position]
            targets.append(episode["latents"][candidate_index])
        probabilities = adapter.reachability(
            torch.stack(sources),
            torch.stack(targets),
            horizon_model_steps=tau_model_steps,
        ).detach().cpu().numpy()
        for probability, (query_index, candidate_position) in zip(
            probabilities, chunk
        ):
            queries[query_index]["rc_scores"][candidate_position] = probability
        print(
            f"reachability pairs={min(start + batch_size, len(flat_pairs))}/"
            f"{len(flat_pairs)}",
            flush=True,
        )


def ranking_query_statistics(
    *,
    source_score: float,
    candidate_scores: np.ndarray,
    remaining_model_steps: np.ndarray,
) -> dict[str, Any]:
    """Score one RC-passed set without censoring it by predicted progress."""

    scores = np.asarray(candidate_scores, dtype=np.float64)
    remaining = np.asarray(remaining_model_steps, dtype=np.int64)
    if scores.ndim != 1 or remaining.shape != scores.shape or len(scores) < 2:
        raise ValueError("a ranking query requires at least two candidates")
    if len(np.unique(remaining)) != len(remaining):
        raise ValueError("Oracle remaining model steps must be unique")

    first, second = np.triu_indices(len(scores), k=1)
    true_sign = np.sign(remaining[first] - remaining[second])
    predicted_sign = np.sign(scores[first] - scores[second])
    correct = predicted_sign == true_sign
    ties = predicted_sign == 0
    pairwise_credit = float(np.sum(correct) + 0.5 * np.sum(ties))
    pair_count = int(len(first))

    correlation = float(spearmanr(scores, remaining).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0

    positive_progress = source_score - scores > 0.0
    selected_index = None
    if bool(np.any(positive_progress)):
        eligible = np.flatnonzero(positive_progress)
        selected_index = int(eligible[np.argmin(scores[eligible])])
    oracle_index = int(np.argmin(remaining))
    return {
        "pairwise_credit": pairwise_credit,
        "pair_count": pair_count,
        "spearman": correlation,
        "top1_correct": float(selected_index == oracle_index),
        "top1_chance": 1.0 / len(scores),
        "selected_index": selected_index,
        "oracle_index": oracle_index,
        "positive_progress_count": int(np.sum(positive_progress)),
    }


def aggregate_episode_metrics(
    episode_records: list[dict[str, float]],
    sampled_indices: np.ndarray | None = None,
) -> dict[str, float]:
    if sampled_indices is None:
        selected = episode_records
    else:
        selected = [episode_records[int(index)] for index in sampled_indices]
    pair_count = sum(record["pair_count"] for record in selected)
    query_count = sum(record["query_count"] for record in selected)
    if pair_count <= 0 or query_count <= 0:
        raise ValueError("test records contain no evaluable ranking queries")
    return {
        "pairwise_accuracy": sum(
            record["pairwise_credit"] for record in selected
        )
        / pair_count,
        "spearman": sum(record["spearman_sum"] for record in selected)
        / query_count,
        "top1_accuracy": sum(record["top1_correct"] for record in selected)
        / query_count,
        "top1_chance_accuracy": sum(
            record["top1_chance"] for record in selected
        )
        / query_count,
    }


def bootstrap_ranking_metrics(
    episode_records: list[dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = {
        "pairwise_accuracy": [],
        "spearman": [],
        "top1_accuracy": [],
        "top1_advantage_over_chance": [],
    }
    for _ in range(samples):
        indices = rng.integers(0, len(episode_records), len(episode_records))
        metrics = aggregate_episode_metrics(episode_records, indices)
        values["pairwise_accuracy"].append(metrics["pairwise_accuracy"])
        values["spearman"].append(metrics["spearman"])
        values["top1_accuracy"].append(metrics["top1_accuracy"])
        values["top1_advantage_over_chance"].append(
            metrics["top1_accuracy"] - metrics["top1_chance_accuracy"]
        )
    return {
        name: {
            "lower_95": float(np.quantile(measurements, 0.025)),
            "upper_95": float(np.quantile(measurements, 0.975)),
        }
        for name, measurements in values.items()
    }


def evaluate_ranker(
    ranker: ProgressRanker,
    adapter: RCAuxAdapter,
    episodes: list[dict[str, Any]],
    *,
    eta_r: float,
    tau_model_steps: int,
    score_batch_size: int,
    reachability_batch_size: int,
    max_sources_per_episode: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    scores_by_episode = score_episode_states(
        ranker,
        episodes,
        batch_size=score_batch_size,
        device=device,
    )
    queries = build_test_queries(
        episodes,
        max_sources_per_episode=max_sources_per_episode,
    )
    score_query_reachability(
        adapter,
        episodes,
        queries,
        tau_model_steps=tau_model_steps,
        batch_size=reachability_batch_size,
    )

    episode_records = [
        {
            "pairwise_credit": 0.0,
            "pair_count": 0.0,
            "spearman_sum": 0.0,
            "query_count": 0.0,
            "top1_correct": 0.0,
            "top1_chance": 0.0,
            "no_positive_progress": 0.0,
        }
        for _ in episodes
    ]
    rc_candidate_count = 0
    rc_pass_count = 0
    insufficient_rc_queries = 0
    for query in queries:
        episode_index = query["episode_list_index"]
        episode = episodes[episode_index]
        pass_mask = query["rc_scores"] >= eta_r
        rc_candidate_count += len(pass_mask)
        rc_pass_count += int(np.sum(pass_mask))
        if int(np.sum(pass_mask)) < 2:
            insufficient_rc_queries += 1
            continue

        candidate_indices = np.asarray(query["candidate_indices"])[pass_mask]
        episode_scores = scores_by_episode[episode_index]
        statistics = ranking_query_statistics(
            source_score=float(episode_scores[query["source_index"]]),
            candidate_scores=episode_scores[candidate_indices],
            remaining_model_steps=(
                episode["remaining_model_steps"][candidate_indices].numpy()
            ),
        )
        record = episode_records[episode_index]
        record["pairwise_credit"] += statistics["pairwise_credit"]
        record["pair_count"] += statistics["pair_count"]
        record["spearman_sum"] += statistics["spearman"]
        record["query_count"] += 1
        record["top1_correct"] += statistics["top1_correct"]
        record["top1_chance"] += statistics["top1_chance"]
        record["no_positive_progress"] += float(
            statistics["positive_progress_count"] == 0
        )

    evaluable_records = [
        record for record in episode_records if record["query_count"] > 0
    ]
    point = aggregate_episode_metrics(evaluable_records)
    confidence = bootstrap_ranking_metrics(
        evaluable_records,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    query_count = int(sum(record["query_count"] for record in evaluable_records))
    no_positive = int(
        sum(record["no_positive_progress"] for record in evaluable_records)
    )
    return {
        "ranking_metrics": {
            "pairwise_accuracy": {
                "value": point["pairwise_accuracy"],
                **confidence["pairwise_accuracy"],
            },
            "spearman": {
                "value": point["spearman"],
                **confidence["spearman"],
            },
            "top1_accuracy": {
                "value": point["top1_accuracy"],
                **confidence["top1_accuracy"],
                "chance_accuracy": point["top1_chance_accuracy"],
                "advantage_over_chance_lower_95": confidence[
                    "top1_advantage_over_chance"
                ]["lower_95"],
                "advantage_over_chance_upper_95": confidence[
                    "top1_advantage_over_chance"
                ]["upper_95"],
            },
        },
        "evaluation_counts": {
            "successful_test_episodes": len(episodes),
            "episodes_with_evaluable_queries": len(evaluable_records),
            "source_queries_before_rc": len(queries),
            "source_queries_with_at_least_two_rc_passes": query_count,
            "source_queries_with_fewer_than_two_rc_passes": (
                insufficient_rc_queries
            ),
            "source_queries_with_no_positive_progress": no_positive,
            "candidate_queries_before_rc": rc_candidate_count,
            "candidate_queries_passing_rc": rc_pass_count,
        },
    }


def atomic_write_json(report: dict[str, Any], path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)


def main() -> int:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    protocol = load_stage2_protocol(args.stage2_report)
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_path = cache_dir / args.dataset
    with h5py.File(dataset_path, "r") as handle:
        train_episode_indices = successful_episode_indices(
            handle, protocol["train_episode_range"]
        )
        test_episode_indices = successful_episode_indices(
            handle, protocol["test_episode_range"]
        )
    train_episode_indices = limit_episode_indices(
        train_episode_indices,
        args.max_train_episodes,
        seed=args.seed,
    )
    test_episode_indices = limit_episode_indices(
        test_episode_indices,
        args.max_test_episodes,
        seed=args.seed + 1,
    )
    if set(train_episode_indices) & set(test_episode_indices):
        raise RuntimeError("train and test episode IDs overlap")

    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=cache_dir,
        device=args.device,
        use_reachability_cost=False,
    )
    adapter.model.interpolate_pos_encoding = True
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("RC-aux Encoder and reachability model must be frozen")

    train_episodes, test_episodes, reused_cache = load_or_encode_latents(
        adapter,
        dataset_path,
        train_episode_indices,
        test_episode_indices,
        policy=args.policy,
        cache_path=args.latent_cache,
        batch_size=args.encode_batch_size,
        rebuild=args.rebuild_latent_cache,
    )
    latent_dim = int(train_episodes[0]["latents"].size(-1))
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = torch.device(args.device)
    ranker = ProgressRanker(latent_dim, hidden_dims).to(device)
    training_history = train_ranker(
        ranker,
        train_episodes,
        epochs=args.epochs,
        pairs_per_epoch=args.pairs_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ranking_margin=args.ranking_margin,
        regression_weight=args.regression_weight,
        seed=args.seed,
        device=device,
    )
    evaluation = evaluate_ranker(
        ranker,
        adapter,
        test_episodes,
        eta_r=protocol["eta_r"],
        tau_model_steps=protocol["tau_model_steps"],
        score_batch_size=args.batch_size,
        reachability_batch_size=args.reachability_batch_size,
        max_sources_per_episode=args.max_test_sources_per_episode,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed + 2,
        device=device,
    )

    metrics = evaluation["ranking_metrics"]
    counts = evaluation["evaluation_counts"]
    validation_criteria = {
        "minimum_test_queries_met": (
            counts["source_queries_with_at_least_two_rc_passes"]
            >= args.min_test_queries
        ),
        "pairwise_accuracy_ci_above_chance": (
            metrics["pairwise_accuracy"]["lower_95"] > 0.5
        ),
        "spearman_ci_above_zero": metrics["spearman"]["lower_95"] > 0.0,
        "top1_accuracy_ci_above_chance": (
            metrics["top1_accuracy"]["advantage_over_chance_lower_95"] > 0.0
        ),
    }
    validated = all(validation_criteria.values())
    model_output = args.model_output.expanduser().resolve()
    atomic_torch_save(
        {
            "format_version": 1,
            "model_type": "ProgressRanker",
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in ranker.state_dict().items()
            },
            "policy": args.policy,
            "eta_r": protocol["eta_r"],
            "tau_model_steps": protocol["tau_model_steps"],
            "training": {
                "ranking_margin": args.ranking_margin,
                "regression_weight": args.regression_weight,
                "epochs": args.epochs,
                "pairs_per_epoch": args.pairs_per_epoch,
                "seed": args.seed,
            },
        },
        model_output,
    )

    report = {
        "stage3_progress_ranker_validated": validated,
        "ready_to_train_high_level_generator": validated,
        "validation_criteria": validation_criteria,
        "policy": args.policy,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "stage2": protocol,
        "protocol": {
            "train_test_split_unit": "episode",
            "successful_trajectories_only": True,
            "train_episode_indices": train_episode_indices,
            "test_episode_indices": test_episode_indices,
            "train_test_episode_overlap": 0,
            "trajectory_sampling_env_steps": (
                TWOROOM_PROFILE.model_step_env_steps
            ),
            "terminal_image_is_goal": True,
            "training_pairs": "same episode i < j < T",
            "ranker_input": "[g, z_G, g-z_G]",
            "tau_model_steps": protocol["tau_model_steps"],
            "eta_r": protocol["eta_r"],
            "eta_r_recalibrated": False,
            "test_candidates": (
                "future nonterminal model-step samples from the same episode"
            ),
            "pairwise_and_spearman_set": "RC-passed candidates",
            "top1_selector": "positive progress then minimum D_psi",
            "oracle": "trajectory order and remaining model steps only",
        },
        "training": {
            "successful_train_episodes": len(train_episodes),
            "latent_dim": latent_dim,
            "hidden_dims": list(hidden_dims),
            "epochs": args.epochs,
            "pairs_per_epoch": args.pairs_per_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ranking_margin": args.ranking_margin,
            "regression_weight": args.regression_weight,
            "history": training_history,
        },
        "test": evaluation,
        "artifacts": {
            "model_checkpoint": str(model_output),
            "latent_cache": str(args.latent_cache.expanduser().resolve()),
            "latent_cache_reused": reused_cache,
        },
    }
    atomic_write_json(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"model_path: {model_output}")
    print(f"report_path: {args.output.expanduser().resolve()}")
    if args.require_validation and not validated:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
