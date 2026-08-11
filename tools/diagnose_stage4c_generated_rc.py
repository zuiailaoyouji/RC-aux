#!/usr/bin/env python3
"""ARCHIVED: diagnose the retired high-level local-RC gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, TWOROOM_PROFILE
from legacy_stage4_protocol import (
    assert_protocol_consistency,
    load_checkpoint_protocol,
    load_stage2_protocol,
)
from stage4_generator import (
    GeneratorTrajectoryDataset,
    TAU_MODEL_STEPS,
    build_generator_sample_refs,
    load_generator_checkpoint,
    load_latent_cache,
    sample_subgoal_candidates,
    split_cached_episodes,
)
from tools.evaluate_stage4_candidates import checkpoint_paths, load_progress_ranker


FORMAL_SEEDS = (3072, 3073, 3074)
NUM_CANDIDATES = 32
KNN_K = 5
DROPOUT_SEED = 20260809
THRESHOLDS = tuple(sorted({*[index / 10 for index in range(10)], 0.5295726657}))
REFERENCE_CATEGORIES = (
    "within_budget_witnessed",
    "same_trajectory_strict_over_budget",
    "cross_trajectory_strict_over_budget",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4C generated-target RC behavior diagnostic only."
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
    parser.add_argument("--dataset", default="tworoom.h5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator-seeds", default="3072,3073,3074")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--knn-query-chunk", type=int, default=256)
    parser.add_argument("--knn-bank-chunk", type=int, default=8192)
    parser.add_argument("--dropout-seed", type=int, default=DROPOUT_SEED)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the fixed split/checkpoint protocol without running models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage4c_generated_target_rc.json"),
    )
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("generator-seeds must be a nonempty unique list")
    if any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"generator-seeds must be selected from {FORMAL_SEEDS}")
    if len(seeds) not in (1, 3) or (len(seeds) == 1 and seeds[0] != 3072):
        raise ValueError("only the formal three seeds or seed 3072 pilot are allowed")
    return FORMAL_SEEDS if len(seeds) == 3 else seeds


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, output)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)


def distribution(values: Any) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def aggregate_distributions(
    summaries: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        statistic: mean_std([float(summary[statistic]) for summary in summaries])
        for statistic in ("mean", "p05", "p25", "median", "p75", "p95", "p99")
    }


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@torch.inference_mode()
def exact_knn_mean_distance(
    queries: torch.Tensor,
    bank: torch.Tensor,
    *,
    k: int = KNN_K,
    query_chunk: int = 256,
    bank_chunk: int = 8192,
) -> torch.Tensor:
    """Return exact mean Euclidean distance to the k nearest bank entries."""

    if queries.ndim != 2 or bank.ndim != 2 or queries.size(1) != bank.size(1):
        raise ValueError("queries and bank must be [N,D] with the same D")
    if not 1 <= k <= bank.size(0):
        raise ValueError("k must be in [1, bank size]")
    if query_chunk < 1 or bank_chunk < 1:
        raise ValueError("kNN chunk sizes must be positive")
    outputs = []
    bank_norm = bank.square().sum(dim=1)
    for query_start in range(0, queries.size(0), query_chunk):
        query = queries[query_start : query_start + query_chunk]
        query_norm = query.square().sum(dim=1, keepdim=True)
        best = torch.full(
            (query.size(0), k),
            torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        for bank_start in range(0, bank.size(0), bank_chunk):
            bank_part = bank[bank_start : bank_start + bank_chunk]
            squared = (
                query_norm
                + bank_norm[bank_start : bank_start + bank_chunk].unsqueeze(0)
                - 2.0 * query @ bank_part.T
            ).clamp_min_(0.0)
            best = torch.topk(
                torch.cat([best, squared], dim=1),
                k=k,
                dim=1,
                largest=False,
            ).values
        outputs.append(best.sqrt().mean(dim=1).cpu())
    return torch.cat(outputs)


def candidate_progress_ranks(progress: torch.Tensor) -> torch.Tensor:
    if progress.ndim != 2:
        raise ValueError("progress must be [S,N]")
    order = progress.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    values = torch.arange(1, progress.size(1) + 1, device=progress.device)
    ranks.scatter_(1, order, values.unsqueeze(0).expand_as(order))
    return ranks


def selected_indices(
    progress: torch.Tensor,
    rc_score: torch.Tensor,
    threshold: float | None,
) -> torch.Tensor:
    feasible = progress > 0.0
    if threshold is not None:
        feasible &= rc_score >= threshold
    masked = progress.masked_fill(~feasible, -torch.inf)
    indices = masked.argmax(dim=1)
    return torch.where(feasible.any(dim=1), indices, -torch.ones_like(indices))


def selected_values(values: torch.Tensor, indices: torch.Tensor) -> np.ndarray:
    covered = indices >= 0
    if not bool(covered.any()):
        return np.asarray([], dtype=np.float64)
    rows = torch.arange(values.size(0))[covered]
    return values[rows, indices[covered]].cpu().numpy()


def threshold_diagnostic(
    rc_score: torch.Tensor,
    progress: torch.Tensor,
    residual_norm: torch.Tensor,
    manifold_distance: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    rc_pass = rc_score >= threshold
    positive = progress > 0.0
    count = rc_pass.sum(dim=1)
    no_rc_index = selected_indices(progress, rc_score, None)
    rc_index = selected_indices(progress, rc_score, threshold)
    source_has_progress = no_rc_index >= 0
    rows = torch.arange(progress.size(0))[source_has_progress]
    best_rejected = torch.zeros(progress.size(0), dtype=torch.bool)
    best_rejected[source_has_progress] = ~rc_pass[
        rows, no_rc_index[source_has_progress]
    ]
    return {
        "eta": float(threshold),
        "candidate_rc_pass_rate": float(rc_pass.float().mean()),
        "sources_n_rc_zero_rate": float((count == 0).float().mean()),
        "sources_n_rc_one_rate": float((count == 1).float().mean()),
        "sources_n_rc_at_least_two_rate": float((count >= 2).float().mean()),
        "rc_positive_progress_coverage": float(rc_index.ge(0).float().mean()),
        "positive_progress_source_rate": float(source_has_progress.float().mean()),
        "highest_progress_candidate_rejection_rate": (
            float(best_rejected[source_has_progress].float().mean())
            if bool(source_has_progress.any())
            else None
        ),
        "selected_source_count": int(rc_index.ge(0).sum()),
        "selected_progress": distribution(selected_values(progress, rc_index)),
        "selected_residual_norm": distribution(
            selected_values(residual_norm, rc_index)
        ),
        "selected_manifold_distance": distribution(
            selected_values(manifold_distance, rc_index)
        ),
    }


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def rowwise_correlation(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    x = x.to(torch.float64)
    y = y.to(torch.float64)

    def coefficients(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)
        denominator = first.square().sum(dim=1).sqrt() * second.square().sum(
            dim=1
        ).sqrt()
        return (first * second).sum(dim=1) / denominator.clamp_min(1.0e-12)

    x_order = x.argsort(dim=1)
    y_order = y.argsort(dim=1)
    x_rank = torch.empty_like(x_order)
    y_rank = torch.empty_like(y_order)
    rank = torch.arange(x.size(1)).unsqueeze(0).expand_as(x_order)
    x_rank.scatter_(1, x_order, rank)
    y_rank.scatter_(1, y_order, rank)
    return {
        "pearson_across_32_within_each_source": distribution(coefficients(x, y)),
        "spearman_across_32_within_each_source": distribution(
            coefficients(x_rank.to(torch.float64), y_rank.to(torch.float64))
        ),
    }


def selection_comparison(
    artifact: dict[str, Any], eta_r: float
) -> dict[str, Any]:
    rc = artifact["rc_score"]
    progress = artifact["d_psi_progress"]
    residual = artifact["residual_norm"]
    manifold = artifact["manifold_distance"]
    no_rc_index = selected_indices(progress, rc, None)
    rc_index = selected_indices(progress, rc, eta_r)
    has_progress = no_rc_index >= 0
    rows = torch.arange(progress.size(0))[has_progress]
    rejected = torch.zeros(progress.size(0), dtype=torch.bool)
    rejected[has_progress] = rc[rows, no_rc_index[has_progress]] < eta_r
    replacement = rejected & (rc_index >= 0)
    lost = rejected & (rc_index < 0)

    changes = {}
    for name, values in (
        ("d_psi_progress", progress),
        ("residual_norm", residual),
        ("rc_score", rc),
        ("manifold_distance", manifold),
    ):
        replacement_rows = torch.arange(values.size(0))[replacement]
        delta = (
            values[replacement_rows, rc_index[replacement]]
            - values[replacement_rows, no_rc_index[replacement]]
        )
        changes[f"rc_selected_minus_no_rc_selected_{name}"] = distribution(delta)

    return {
        "positive_progress_source_count": int(has_progress.sum()),
        "no_rc_selected": {
            "coverage": float(has_progress.float().mean()),
            "d_psi_progress": distribution(selected_values(progress, no_rc_index)),
            "residual_norm": distribution(selected_values(residual, no_rc_index)),
            "rc_score": distribution(selected_values(rc, no_rc_index)),
            "manifold_distance": distribution(
                selected_values(manifold, no_rc_index)
            ),
        },
        "rc_selected": {
            "coverage": float(rc_index.ge(0).float().mean()),
            "d_psi_progress": distribution(selected_values(progress, rc_index)),
            "residual_norm": distribution(selected_values(residual, rc_index)),
            "rc_score": distribution(selected_values(rc, rc_index)),
            "manifold_distance": distribution(selected_values(manifold, rc_index)),
        },
        "highest_progress_candidate_rejected_count": int(rejected.sum()),
        "highest_progress_candidate_rejection_rate": float(
            rejected[has_progress].float().mean()
        ),
        "rejected_with_rc_replacement_count": int(replacement.sum()),
        "rejected_without_rc_replacement_count": int(lost.sum()),
        "changes_after_rejection_when_replacement_exists": changes,
    }


def summarize_seed(
    artifact: dict[str, Any],
    *,
    eta_r: float,
    real_manifold: np.ndarray,
) -> dict[str, Any]:
    rc = artifact["rc_score"]
    progress = artifact["d_psi_progress"]
    residual = artifact["residual_norm"]
    manifold = artifact["manifold_distance"]
    real_p95 = float(np.quantile(real_manifold, 0.95))
    real_summary = distribution(real_manifold)
    relationships = {}
    for name, values in (
        ("latent_residual_norm", residual),
        ("manifold_distance", manifold),
        ("d_psi_progress", progress),
    ):
        relationships[f"rc_vs_{name}"] = {
            "candidate_level_descriptive": correlation(rc.numpy(), values.numpy()),
            **rowwise_correlation(rc, values),
        }
    direct_eligible = artifact["direct_goal_eligible"]
    return {
        "seed": artifact["metadata"]["generator_seed"],
        "checkpoint": artifact["metadata"]["checkpoint"],
        "source_count": int(rc.size(0)),
        "candidate_count": int(rc.numel()),
        "candidate_distributions": {
            "rc_score": distribution(rc),
            "d_psi_progress": distribution(progress),
            "latent_residual_norm": distribution(residual),
            "manifold_distance": distribution(manifold),
            "progress_rank": distribution(artifact["d_psi_rank"]),
            "generated_manifold_above_real_p95_rate": float(
                (manifold > real_p95).float().mean()
            ),
            "generated_to_real_manifold_mean_ratio": float(
                manifold.float().mean() / real_summary["mean"]
            ),
            "generated_to_real_manifold_median_ratio": float(
                manifold.median() / real_summary["median"]
            ),
        },
        "relationships": relationships,
        "selection_at_stage2_eta": selection_comparison(artifact, eta_r),
        "direct_goal_states": {
            "eligible_count": int(direct_eligible.sum()),
            "eligible_rate": float(direct_eligible.float().mean()),
            "rc_score": distribution(artifact["direct_goal_rc_score"]),
            "d_psi_progress": distribution(
                artifact["direct_goal_d_psi_progress"]
            ),
        },
        "threshold_sensitivity": [
            threshold_diagnostic(rc, progress, residual, manifold, threshold)
            for threshold in THRESHOLDS
        ],
    }


def artifact_path(output: Path, seed: int) -> Path:
    output = output.expanduser().resolve()
    return output.with_name(f"{output.stem}_seed{seed}_candidates.pt")


@torch.inference_mode()
def evaluate_seed(
    checkpoint_path: Path,
    loader: DataLoader,
    adapter: RCAuxAdapter,
    progress_ranker,
    bank: torch.Tensor,
    *,
    bank_sha256: str,
    eta_r: float,
    dropout_seed: int,
    query_chunk: int,
    bank_chunk: int,
    device: torch.device,
) -> dict[str, Any]:
    generator, checkpoint = load_generator_checkpoint(checkpoint_path, device=device)
    generator.requires_grad_(False)
    seed = int(checkpoint["seed"])
    torch.manual_seed(dropout_seed + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dropout_seed + seed)
    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "episode_index",
            "source_row",
            "current_latent",
            "goal_latent",
            "candidate_latents",
            "rc_score",
            "d_psi_progress",
            "residual_norm",
            "manifold_distance",
            "d_psi_rank",
            "direct_goal_rc_score",
            "direct_goal_d_psi_progress",
        )
    }
    for batch_index, batch in enumerate(loader):
        history = batch["history_latents"].to(device)
        mask = batch["history_padding_mask"].to(device)
        current = batch["current_latent"].to(device)
        goal = batch["goal_latent"].to(device)
        candidates = sample_subgoal_candidates(
            generator,
            history,
            mask,
            goal,
            num_candidates=NUM_CANDIDATES,
            stochastic=True,
        )
        rc = adapter.reachability(
            current,
            candidates,
            horizon_model_steps=TAU_MODEL_STEPS,
        )
        direct_rc = adapter.reachability(
            current,
            goal,
            horizon_model_steps=TAU_MODEL_STEPS,
        )
        source_score = progress_ranker(current, goal)
        candidate_score = progress_ranker(
            candidates.flatten(0, 1),
            goal.repeat_interleave(NUM_CANDIDATES, dim=0),
        ).reshape(current.size(0), NUM_CANDIDATES)
        direct_score = progress_ranker(goal, goal)
        progress = source_score.unsqueeze(1) - candidate_score
        residual = (candidates - current.unsqueeze(1)).norm(dim=-1)
        manifold = exact_knn_mean_distance(
            candidates.flatten(0, 1),
            bank,
            query_chunk=query_chunk,
            bank_chunk=bank_chunk,
        ).reshape(current.size(0), NUM_CANDIDATES)
        tensors = {
            "episode_index": batch["episode_index"],
            "source_row": batch["source_row"],
            "current_latent": current,
            "goal_latent": goal,
            "candidate_latents": candidates,
            "rc_score": rc,
            "d_psi_progress": progress,
            "residual_norm": residual,
            "manifold_distance": manifold,
            "d_psi_rank": candidate_progress_ranks(progress),
            "direct_goal_rc_score": direct_rc,
            "direct_goal_d_psi_progress": source_score - direct_score,
        }
        for name, value in tensors.items():
            collected[name].append(torch.as_tensor(value).detach().cpu())
        print(
            f"seed={seed} batch={batch_index + 1}/{len(loader)} "
            f"sources={sum(len(item) for item in collected['source_row'])}",
            flush=True,
        )
    artifact = {name: torch.cat(values) for name, values in collected.items()}
    artifact["rc_pass"] = artifact["rc_score"] >= eta_r
    artifact["direct_goal_rc_pass"] = artifact["direct_goal_rc_score"] >= eta_r
    artifact["direct_goal_progress_pass"] = (
        artifact["direct_goal_d_psi_progress"] > 0.0
    )
    artifact["direct_goal_eligible"] = (
        artifact["direct_goal_rc_pass"] & artifact["direct_goal_progress_pass"]
    )
    artifact["no_rc_selected_index"] = selected_indices(
        artifact["d_psi_progress"], artifact["rc_score"], None
    )
    artifact["rc_selected_index"] = selected_indices(
        artifact["d_psi_progress"], artifact["rc_score"], eta_r
    )
    artifact["metadata"] = {
        "format_version": 1,
        "stage": "4C",
        "generator_seed": seed,
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "dropout_seed": dropout_seed + seed,
        "batch_size": loader.batch_size,
        "source_count": len(loader.dataset),
        "source_episode_range": [4000, 5000],
        "candidate_count_per_source": NUM_CANDIDATES,
        "eta_r": eta_r,
        "knn_k": KNN_K,
        "knn_space": "raw unnormalized RC-aux 192D latent",
        "knn_metric": "Euclidean",
        "knn_aggregation": "mean distance to five nearest train-bank latents",
        "latent_bank_episode_range": [0, 4000],
        "latent_bank_size": len(bank),
        "latent_bank_sha256": bank_sha256,
    }
    return artifact


@torch.inference_mode()
def encode_rows(
    adapter: RCAuxAdapter,
    handle: h5py.File,
    rows: list[int],
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    unique = sorted(set(rows))
    encoded = {}
    for start in range(0, len(unique), batch_size):
        row_batch = unique[start : start + batch_size]
        images = np.asarray(handle["pixels"][row_batch])
        latents = adapter.encode_observation(images)[:, -1].cpu()
        encoded.update(zip(row_batch, latents))
    return encoded


def stage12_reference(
    report_path: Path,
    adapter: RCAuxAdapter,
    bank: torch.Tensor,
    dataset_path: Path,
    *,
    batch_size: int,
    query_chunk: int,
    bank_chunk: int,
) -> dict[str, Any]:
    report = json.loads(report_path.expanduser().resolve().read_text())
    trials = report["calibration_trials"] + report["test_trials"]
    rows = [int(trial["source_row"]) for trial in trials]
    rows.extend(
        int(candidate["row"])
        for trial in trials
        for candidate in trial["candidates"]
    )
    with h5py.File(dataset_path, "r") as handle:
        encoded = encode_rows(adapter, handle, rows, batch_size=batch_size)
    sources = []
    targets = []
    categories = []
    reported_rc = []
    physical_distance = []
    for trial in trials:
        source = encoded[int(trial["source_row"])]
        for candidate in trial["candidates"]:
            sources.append(source)
            targets.append(encoded[int(candidate["row"])])
            categories.append(candidate["category"])
            reported_rc.append(float(candidate["local_rc_score"]))
            physical_distance.append(float(candidate["source_distance"]))
    source = torch.stack(sources).to(bank.device)
    target = torch.stack(targets).to(bank.device)
    rc = adapter.reachability(source, target, horizon_model_steps=TAU_MODEL_STEPS).cpu()
    manifold = exact_knn_mean_distance(
        target,
        bank,
        query_chunk=query_chunk,
        bank_chunk=bank_chunk,
    )
    residual = (target - source).norm(dim=1).cpu()
    reported = torch.tensor(reported_rc)
    records = {
        "rc_score": rc,
        "latent_residual_norm": residual,
        "manifold_distance": manifold,
        "physical_source_distance_diagnostic": torch.tensor(physical_distance),
    }
    groups = {}
    category_array = np.asarray(categories)
    for category in REFERENCE_CATEGORIES:
        mask = torch.from_numpy(category_array == category)
        groups[category] = {
            "distributions": {
                name: distribution(values[mask]) for name, values in records.items()
            },
            "relationships": {
                "rc_vs_latent_residual_norm": correlation(
                    rc[mask].numpy(), residual[mask].numpy()
                ),
                "rc_vs_manifold_distance": correlation(
                    rc[mask].numpy(), manifold[mask].numpy()
                ),
            },
        }
    strict_mask = torch.from_numpy(category_array != REFERENCE_CATEGORIES[0])
    groups["strict_over_budget_or_unreachable_combined"] = {
        "distributions": {
            name: distribution(values[strict_mask]) for name, values in records.items()
        },
        "relationships": {
            "rc_vs_latent_residual_norm": correlation(
                rc[strict_mask].numpy(), residual[strict_mask].numpy()
            ),
            "rc_vs_manifold_distance": correlation(
                rc[strict_mask].numpy(), manifold[strict_mask].numpy()
            ),
        },
    }
    return {
        "trial_count": len(trials),
        "target_count": len(targets),
        "label_source": (
            "Stage 1/2 witnessed-within-budget and strict lower-bound categories; "
            "no generated-target labels are introduced"
        ),
        "recomputed_vs_stage2_reported_rc_max_abs_difference": float(
            (rc - reported).abs().max()
        ),
        "groups": groups,
    }


def aggregate_seeds(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "candidate_rc_mean": lambda report: report["candidate_distributions"][
            "rc_score"
        ]["mean"],
        "candidate_progress_mean": lambda report: report[
            "candidate_distributions"
        ]["d_psi_progress"]["mean"],
        "candidate_residual_norm_mean": lambda report: report[
            "candidate_distributions"
        ]["latent_residual_norm"]["mean"],
        "candidate_manifold_distance_mean": lambda report: report[
            "candidate_distributions"
        ]["manifold_distance"]["mean"],
        "generated_manifold_above_real_p95_rate": lambda report: report[
            "candidate_distributions"
        ]["generated_manifold_above_real_p95_rate"],
        "highest_progress_rejection_rate": lambda report: report[
            "selection_at_stage2_eta"
        ]["highest_progress_candidate_rejection_rate"],
        "rc_positive_progress_coverage": lambda report: report[
            "selection_at_stage2_eta"
        ]["rc_selected"]["coverage"],
        "direct_goal_eligible_rate": lambda report: report["direct_goal_states"][
            "eligible_rate"
        ],
    }
    correlation_paths = {}
    for relation in (
        "rc_vs_latent_residual_norm",
        "rc_vs_manifold_distance",
        "rc_vs_d_psi_progress",
    ):
        for coefficient in ("pearson", "spearman"):
            correlation_paths[f"{relation}_{coefficient}"] = (
                lambda report, relation=relation, coefficient=coefficient: report[
                    "relationships"
                ][relation]["candidate_level_descriptive"][coefficient]
            )
    scalars = {
        name: mean_std([float(extract(report)) for report in seed_reports])
        for name, extract in {**scalar_paths, **correlation_paths}.items()
    }
    candidate_distribution_mean_std = {}
    for name in (
        "rc_score",
        "d_psi_progress",
        "latent_residual_norm",
        "manifold_distance",
    ):
        candidate_distribution_mean_std[name] = aggregate_distributions(
            [report["candidate_distributions"][name] for report in seed_reports]
        )
    threshold_rows = []
    for row_index, threshold in enumerate(THRESHOLDS):
        rows = [report["threshold_sensitivity"][row_index] for report in seed_reports]
        threshold_rows.append(
            {
                "eta": float(threshold),
                **{
                    name: mean_std([float(row[name]) for row in rows])
                    for name in (
                        "candidate_rc_pass_rate",
                        "sources_n_rc_zero_rate",
                        "sources_n_rc_one_rate",
                        "sources_n_rc_at_least_two_rate",
                        "rc_positive_progress_coverage",
                        "highest_progress_candidate_rejection_rate",
                    )
                },
                "selected_progress_mean_std": aggregate_distributions(
                    [row["selected_progress"] for row in rows]
                ),
                "selected_residual_norm_mean_std": aggregate_distributions(
                    [row["selected_residual_norm"] for row in rows]
                ),
                "selected_manifold_distance_mean_std": aggregate_distributions(
                    [row["selected_manifold_distance"] for row in rows]
                ),
            }
        )
    return {
        "aggregation_unit": "generator seed; candidates are never pooled across seeds",
        "generator_seed_count": len(seed_reports),
        "scalar_mean_std_across_generator_seeds": scalars,
        "candidate_distribution_statistics_mean_std_across_generator_seeds": (
            candidate_distribution_mean_std
        ),
        "threshold_sensitivity_mean_std_across_generator_seeds": threshold_rows,
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.generator_seeds)
    if (
        not args.validate_only
        and args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but is unavailable")
    if min(args.batch_size, args.knn_query_chunk, args.knn_bank_chunk) < 1:
        raise ValueError("batch and kNN chunk sizes must be positive")
    stage2 = load_stage2_protocol(args.stage2_report)
    if not np.isclose(stage2["eta_r"], 0.5295726657, atol=1.0e-9, rtol=0.0):
        raise ValueError("Stage 4C requires eta_R=0.5295726657 from Stage 2")
    stage3 = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(stage2, stage3, checkpoint_label="Stage 3 D_psi")

    cache = load_latent_cache(args.latent_cache)
    train_episodes, validation_episodes, _ = split_cached_episodes(cache)
    train_bank_cpu = torch.cat(
        [episode["latents"].to(torch.float32) for episode in train_episodes]
    )
    validation_real = torch.cat(
        [episode["latents"].to(torch.float32) for episode in validation_episodes]
    )
    validation_refs = build_generator_sample_refs(validation_episodes)
    validation_dataset = GeneratorTrajectoryDataset(validation_episodes, validation_refs)
    expected_episode_index = torch.tensor(
        [
            int(validation_dataset[index]["episode_index"])
            for index in range(len(validation_dataset))
        ]
    )
    expected_source_row = torch.tensor(
        [
            int(validation_dataset[index]["source_row"])
            for index in range(len(validation_dataset))
        ]
    )
    available = {}
    for path in checkpoint_paths(args.training_report):
        checkpoint = torch.load(
            path.expanduser().resolve(), map_location="cpu", weights_only=False
        )
        available[int(checkpoint["seed"])] = path
    missing_seeds = [seed for seed in seeds if seed not in available]
    if missing_seeds:
        raise ValueError(f"training report is missing generator seeds {missing_seeds}")
    if args.validate_only:
        print(
            f"validated {len(validation_episodes)} validation episodes, "
            f"{len(validation_dataset)} sources, {len(train_bank_cpu)} bank latents, "
            f"and generator seeds {list(seeds)}"
        )
        return 0
    loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    bank = train_bank_cpu.to(device)
    bank_sha256 = tensor_sha256(train_bank_cpu)
    real_manifold = exact_knn_mean_distance(
        validation_real.to(device),
        bank,
        query_chunk=args.knn_query_chunk,
        bank_chunk=args.knn_bank_chunk,
    ).numpy()

    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=args.cache_dir.expanduser().resolve(),
        device=device,
    )
    adapter.model.interpolate_pos_encoding = True
    progress_ranker, _ = load_progress_ranker(args.progress_ranker, device=device)
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("E_theta, F_theta, and R_phi must remain frozen")
    if any(parameter.requires_grad for parameter in progress_ranker.parameters()):
        raise RuntimeError("D_psi must remain frozen")

    dataset_path = args.cache_dir.expanduser().resolve() / args.dataset
    references = stage12_reference(
        args.stage2_report,
        adapter,
        bank,
        dataset_path,
        batch_size=args.batch_size,
        query_chunk=args.knn_query_chunk,
        bank_chunk=args.knn_bank_chunk,
    )
    seed_reports = []
    artifact_paths = []
    for seed in seeds:
        checkpoint_path = available.get(seed)
        if checkpoint_path is None:
            raise ValueError(f"training report does not contain generator seed {seed}")
        stage4 = load_checkpoint_protocol(checkpoint_path, stage=4)
        assert_protocol_consistency(
            stage2, stage4, checkpoint_label=f"Stage 4 generator seed {seed}"
        )
        path = artifact_path(args.output, seed)
        if path.exists():
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            metadata = artifact.get("metadata", {})
            if (
                metadata.get("generator_seed") != seed
                or not np.isclose(metadata.get("eta_r", -1), stage2["eta_r"])
                or metadata.get("knn_k") != KNN_K
                or metadata.get("dropout_seed") != args.dropout_seed + seed
                or metadata.get("batch_size") != args.batch_size
                or metadata.get("source_count") != len(validation_dataset)
                or metadata.get("latent_bank_size") != len(train_bank_cpu)
                or metadata.get("latent_bank_sha256") != bank_sha256
                or Path(metadata.get("checkpoint", "")).expanduser().resolve()
                != checkpoint_path.expanduser().resolve()
                or not isinstance(artifact.get("episode_index"), torch.Tensor)
                or not torch.equal(
                    artifact.get("episode_index"), expected_episode_index
                )
                or not isinstance(artifact.get("source_row"), torch.Tensor)
                or not torch.equal(artifact.get("source_row"), expected_source_row)
            ):
                raise ValueError(f"existing artifact has incompatible metadata: {path}")
        else:
            artifact = evaluate_seed(
                checkpoint_path,
                loader,
                adapter,
                progress_ranker,
                bank,
                bank_sha256=bank_sha256,
                eta_r=stage2["eta_r"],
                dropout_seed=args.dropout_seed,
                query_chunk=args.knn_query_chunk,
                bank_chunk=args.knn_bank_chunk,
                device=device,
            )
            atomic_torch_save(artifact, path)
        artifact_paths.append(str(path))
        seed_reports.append(
            summarize_seed(artifact, eta_r=stage2["eta_r"], real_manifold=real_manifold)
        )
        atomic_write_json(
            {
                "stage4c_complete": False,
                "completed_seeds": [report["seed"] for report in seed_reports],
                "seed_results": seed_reports,
            },
            args.output,
        )

    report = {
        "stage4c_complete": seeds == FORMAL_SEEDS,
        "pilot_only": seeds == (3072,),
        "protocol": {
            "diagnostic_only": True,
            "modules_trained_or_modified": False,
            "generated_target_ground_truth_reachability_labels_used": False,
            "generated_target_auc_or_accuracy_reported": False,
            "threshold_recalibrated": False,
            "source_episode_range": [4000, 5000],
            "source_episode_count": len(validation_episodes),
            "source_state_count": len(validation_dataset),
            "source_episodes_successful_only": True,
            "generator_seeds": list(seeds),
            "formal_generator_seeds": list(FORMAL_SEEDS),
            "candidates_per_source": NUM_CANDIDATES,
            "dropout_seed_rule": "20260809 + generator_seed, once before ordered loader",
            "eta_r": stage2["eta_r"],
            "tau_model_steps": TAU_MODEL_STEPS,
            "latent_bank_episode_range": [0, 4000],
            "latent_bank_successful_only": True,
            "latent_bank_episode_count": len(train_episodes),
            "latent_bank_size": len(train_bank_cpu),
            "validation_real_latent_count": len(validation_real),
            "test_bank_used": False,
            "test_bank_participates_in_any_decision": False,
            "knn_k": KNN_K,
            "knn_space": "raw unnormalized RC-aux 192D latent",
            "knn_metric": "Euclidean",
            "knn_aggregation": "mean of five nearest-neighbor distances",
            "thresholds": list(THRESHOLDS),
            "direct_goal_excluded_from_generated_candidate_threshold_scan": True,
            "candidate_artifacts": artifact_paths,
        },
        "real_to_real_manifold_reference": distribution(real_manifold),
        "stage1_stage2_real_target_references": references,
        "seed_results": seed_reports,
        "across_generator_seeds": aggregate_seeds(seed_reports),
        "interpretation_constraints": [
            "Generated targets have no ground-truth reachability labels here.",
            "Do not interpret this report as generated-target calibration evidence.",
            "Do not select or recalibrate eta_R from this report.",
            "Do not retrain R_phi from this report.",
            "Test episodes [5000,10000) are untouched by Stage 4C.",
        ],
    }
    atomic_write_json(report, args.output)
    print(f"report_path: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
