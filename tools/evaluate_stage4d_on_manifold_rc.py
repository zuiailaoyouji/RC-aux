#!/usr/bin/env python3
"""ARCHIVED: evaluate the retired local-RC gate under a 1-NN control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, RCAuxPlannerConfig, TWOROOM_PROFILE
from legacy_stage4_protocol import (
    assert_protocol_consistency,
    load_checkpoint_protocol,
    load_stage2_protocol,
)
from stage4_generator import (
    MODEL_STEP_ENV_STEPS,
    TAU_MODEL_STEPS,
    distribution_summary,
    load_generator_checkpoint,
    load_latent_cache,
)
from tools.diagnose_stage4c_generated_rc import (
    KNN_K,
    exact_knn_mean_distance,
    tensor_sha256,
)
from tools.evaluate_stage4_candidates import checkpoint_paths, load_progress_ranker
from tools.evaluate_stage4_closed_loop import (
    ABLATION_DIRECT_ONLY_RC,
    TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
    atomic_write_json,
    bootstrap_mean_95_ci,
    load_environment_record,
    run_rollout,
    summarize_method,
)


FORMAL_SEEDS = (3072, 3073, 3074)
NUM_CANDIDATES = 32
EXPECTED_VALIDATION_EPISODES = 411
NO_RC_METHOD = "projected32_dpsi_direct_no_rc"
RC_METHOD = "projected32_rc_dpsi_direct"
METHODS = (NO_RC_METHOD, RC_METHOD)
SELECTOR_METHOD = {
    NO_RC_METHOD: ABLATION_DIRECT_ONLY_RC,
    RC_METHOD: "stochastic32_rc_dpsi",
}

DEFAULT_DROPOUT_SEED = 20260809
DEFAULT_CEM_SEED = 4200
DEFAULT_ENV_SEED = 42
DEFAULT_NUM_SAMPLES = 300
DEFAULT_CEM_ITERATIONS = 30
DEFAULT_TOPK = 30
DEFAULT_FLAT_HORIZON = 5
DEFAULT_REACHABILITY_COST_WEIGHT = 0.85
DEFAULT_MAX_REPLAY_PIXEL_DIFF = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="On-manifold matched control for the frozen Stage 4 RC selector."
    )
    parser.add_argument(
        "--stage4c-report",
        type=Path,
        default=Path("outputs/stage4c_generated_target_rc.json"),
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
    parser.add_argument("--offline-batch-size", type=int, default=256)
    parser.add_argument("--knn-query-chunk", type=int, default=256)
    parser.add_argument("--knn-bank-chunk", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--cem-iterations", type=int, default=DEFAULT_CEM_ITERATIONS
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--cem-seed", type=int, default=DEFAULT_CEM_SEED)
    parser.add_argument("--env-seed", type=int, default=DEFAULT_ENV_SEED)
    parser.add_argument(
        "--dropout-seed", type=int, default=DEFAULT_DROPOUT_SEED
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--flat-planning-horizon-model-steps",
        type=int,
        default=DEFAULT_FLAT_HORIZON,
    )
    parser.add_argument(
        "--reachability-cost-weight",
        type=float,
        default=DEFAULT_REACHABILITY_COST_WEIGHT,
    )
    parser.add_argument(
        "--max-replay-pixel-diff",
        type=int,
        default=DEFAULT_MAX_REPLAY_PIXEL_DIFF,
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="finish offline projected-candidate control and skip CEM rollouts",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate fixed inputs without loading frozen models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage4d_on_manifold_rc.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text())


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if seeds not in ((3072,), FORMAL_SEEDS):
        raise ValueError("generator-seeds must be 3072 pilot or 3072,3073,3074")
    return seeds


def train_validation_episodes_only(
    cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split only cache['train']; never inspect cache['test'] in this control."""

    first_half = cache["train"]
    train = [
        episode for episode in first_half if int(episode["episode_index"]) < 4000
    ]
    validation = [
        episode
        for episode in first_half
        if 4000 <= int(episode["episode_index"]) < 5000
    ]
    if not train or not validation:
        raise ValueError("train and validation successful episode splits are required")
    train_ids = {int(episode["episode_index"]) for episode in train}
    validation_ids = {int(episode["episode_index"]) for episode in validation}
    if train_ids & validation_ids:
        raise ValueError("train and validation episode splits overlap")
    return train, validation


def distribution(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    return distribution_summary(array.reshape(-1).tolist())


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


class ExactRealLatentProjector:
    """Project every query to an actual raw-space 1-NN bank row."""

    def __init__(
        self,
        bank: torch.Tensor,
        *,
        query_chunk: int = 256,
        bank_chunk: int = 8192,
    ) -> None:
        if bank.ndim != 2:
            raise ValueError("latent bank must be [M,D]")
        if min(query_chunk, bank_chunk) < 1:
            raise ValueError("projection chunk sizes must be positive")
        self.bank = bank
        self.bank_norm = bank.square().sum(dim=1)
        self.query_chunk = query_chunk
        self.bank_chunk = bank_chunk

    @torch.inference_mode()
    def indices(self, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 2 or queries.size(1) != self.bank.size(1):
            raise ValueError("projection queries must be [Q,D]")
        queries = queries.to(self.bank.device)
        outputs = []
        for query_start in range(0, len(queries), self.query_chunk):
            query = queries[query_start : query_start + self.query_chunk]
            query_norm = query.square().sum(dim=1)
            best_distance = torch.full(
                (len(query),), torch.inf, device=query.device
            )
            best_index = torch.zeros(
                len(query), dtype=torch.long, device=query.device
            )
            for bank_start in range(0, len(self.bank), self.bank_chunk):
                bank_part = self.bank[bank_start : bank_start + self.bank_chunk]
                squared = (
                    query_norm.unsqueeze(1)
                    + self.bank_norm[
                        bank_start : bank_start + self.bank_chunk
                    ].unsqueeze(0)
                    - 2.0 * query @ bank_part.T
                ).clamp_min_(0.0)
                part_distance, part_index = squared.min(dim=1)
                update = part_distance < best_distance
                best_distance[update] = part_distance[update]
                best_index[update] = part_index[update] + bank_start
            outputs.append(best_index)
        return torch.cat(outputs)

    @torch.inference_mode()
    def __call__(self, queries: torch.Tensor) -> torch.Tensor:
        original_shape = queries.shape
        flat = queries.reshape(-1, original_shape[-1])
        projected = self.bank[self.indices(flat)]
        return projected.reshape(original_shape)


def artifact_path_from_report(report: dict[str, Any], seed: int) -> Path:
    candidates = report["protocol"]["candidate_artifacts"]
    matches = [
        Path(path) for path in candidates if f"seed{seed}_" in Path(path).name
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage 4C report does not identify seed {seed} artifact")
    return matches[0].expanduser().resolve()


def validate_stage4c_report(
    report: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    eta_r: float,
    bank_sha256: str,
    expected_episode_indices: set[int],
) -> None:
    if seeds == FORMAL_SEEDS and report.get("stage4c_complete") is not True:
        raise ValueError("formal control requires a complete Stage 4C report")
    protocol = report["protocol"]
    expected = {
        "source_episode_range": [4000, 5000],
        "source_episode_count": EXPECTED_VALIDATION_EPISODES,
        "source_episodes_successful_only": True,
        "candidates_per_source": NUM_CANDIDATES,
        "latent_bank_episode_range": [0, 4000],
        "latent_bank_successful_only": True,
        "test_bank_used": False,
        "knn_space": "raw unnormalized RC-aux 192D latent",
        "knn_metric": "Euclidean",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"Stage 4C {key}={protocol.get(key)!r}, expected {value!r}"
            )
    if not np.isclose(protocol["eta_r"], eta_r, atol=1.0e-12, rtol=0.0):
        raise ValueError("Stage 4C eta_R differs from Stage 2")
    for seed in seeds:
        artifact = torch.load(
            artifact_path_from_report(report, seed),
            map_location="cpu",
            weights_only=False,
        )
        metadata = artifact["metadata"]
        if metadata["generator_seed"] != seed:
            raise ValueError(f"candidate artifact seed mismatch for {seed}")
        if metadata["latent_bank_sha256"] != bank_sha256:
            raise ValueError(f"candidate artifact bank mismatch for seed {seed}")
        source_count = int(protocol["source_state_count"])
        expected_shape = (source_count, NUM_CANDIDATES, 192)
        if tuple(artifact["candidate_latents"].shape) != expected_shape:
            raise ValueError(
                f"seed {seed} candidate shape differs from {expected_shape}"
            )
        artifact_episode_indices = {
            int(value) for value in artifact["episode_index"].tolist()
        }
        if artifact_episode_indices != expected_episode_indices:
            raise ValueError(
                f"seed {seed} candidate artifact does not cover all validation episodes"
            )
        if metadata.get("dropout_seed") != DEFAULT_DROPOUT_SEED + seed:
            raise ValueError(f"seed {seed} Stage 4C dropout seed differs")


def select_with_direct_goal(
    progress: torch.Tensor,
    rc_score: torch.Tensor,
    direct_progress: torch.Tensor,
    direct_rc: torch.Tensor,
    *,
    eta_r: float,
    filter_generated_with_rc: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if progress.ndim != 2 or rc_score.shape != progress.shape:
        raise ValueError("candidate progress and RC must be [S,N]")
    generated_feasible = progress > 0.0
    if filter_generated_with_rc:
        generated_feasible &= rc_score >= eta_r
    direct_feasible = (direct_progress > 0.0) & (direct_rc >= eta_r)
    combined_progress = torch.cat([progress, direct_progress.unsqueeze(1)], dim=1)
    combined_feasible = torch.cat(
        [generated_feasible, direct_feasible.unsqueeze(1)], dim=1
    )
    masked = combined_progress.masked_fill(~combined_feasible, -torch.inf)
    selected = masked.argmax(dim=1)
    covered = combined_feasible.any(dim=1)
    selected = torch.where(covered, selected, -torch.ones_like(selected))
    return selected, covered, generated_feasible


def gather_selected(
    candidate_values: torch.Tensor,
    direct_values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    combined = torch.cat([candidate_values, direct_values.unsqueeze(1)], dim=1)
    covered = indices >= 0
    rows = torch.arange(len(indices), device=indices.device)[covered]
    return combined[rows, indices[covered]]


@torch.inference_mode()
def evaluate_projected_offline_seed(
    artifact: dict[str, Any],
    projector: ExactRealLatentProjector,
    adapter: RCAuxAdapter,
    progress_ranker,
    *,
    eta_r: float,
    batch_size: int,
    query_chunk: int,
    bank_chunk: int,
    device: torch.device,
    real_manifold_reference: dict[str, Any],
) -> dict[str, Any]:
    candidates_cpu = artifact["candidate_latents"].to(torch.float32)
    current_cpu = artifact["current_latent"].to(torch.float32)
    goal_cpu = artifact["goal_latent"].to(torch.float32)
    projected_parts = []
    projection_index_parts = []
    for start in range(0, len(candidates_cpu), batch_size):
        raw = candidates_cpu[start : start + batch_size].to(device)
        indices = projector.indices(raw.flatten(0, 1))
        projected_parts.append(projector.bank[indices].reshape_as(raw).cpu())
        projection_index_parts.append(indices.reshape(raw.shape[:2]).cpu())
    projected = torch.cat(projected_parts)
    projection_indices = torch.cat(projection_index_parts)
    exact_rows = projector.bank[projection_indices.to(device)].cpu()
    if not torch.equal(projected, exact_rows):
        raise RuntimeError("1-NN projection did not return exact latent-bank rows")

    rc_parts = []
    progress_parts = []
    direct_rc_parts = []
    direct_progress_parts = []
    for start in range(0, len(projected), batch_size):
        current = current_cpu[start : start + batch_size].to(device)
        goal = goal_cpu[start : start + batch_size].to(device)
        candidate = projected[start : start + batch_size].to(device)
        rc_parts.append(
            adapter.reachability(
                current, candidate, horizon_model_steps=TAU_MODEL_STEPS
            ).cpu()
        )
        source_score = progress_ranker(current, goal)
        candidate_score = progress_ranker(
            candidate.flatten(0, 1),
            goal.repeat_interleave(NUM_CANDIDATES, dim=0),
        ).reshape(len(current), NUM_CANDIDATES)
        progress_parts.append((source_score.unsqueeze(1) - candidate_score).cpu())
        direct_rc_parts.append(
            adapter.reachability(
                current, goal, horizon_model_steps=TAU_MODEL_STEPS
            ).cpu()
        )
        direct_progress_parts.append(
            (source_score - progress_ranker(goal, goal)).cpu()
        )
    rc = torch.cat(rc_parts)
    progress = torch.cat(progress_parts)
    direct_rc = torch.cat(direct_rc_parts)
    direct_progress = torch.cat(direct_progress_parts)
    residual = (projected - current_cpu.unsqueeze(1)).norm(dim=-1)
    direct_residual = (goal_cpu - current_cpu).norm(dim=-1)
    manifold = exact_knn_mean_distance(
        projected.flatten(0, 1).to(device),
        projector.bank,
        k=KNN_K,
        query_chunk=query_chunk,
        bank_chunk=bank_chunk,
    ).reshape_as(progress)
    direct_manifold = exact_knn_mean_distance(
        goal_cpu.to(device),
        projector.bank,
        k=KNN_K,
        query_chunk=query_chunk,
        bank_chunk=bank_chunk,
    )

    no_rc_index, no_rc_covered, no_rc_generated_feasible = (
        select_with_direct_goal(
            progress,
            rc,
            direct_progress,
            direct_rc,
            eta_r=eta_r,
            filter_generated_with_rc=False,
        )
    )
    rc_index, rc_covered, rc_generated_feasible = select_with_direct_goal(
        progress,
        rc,
        direct_progress,
        direct_rc,
        eta_r=eta_r,
        filter_generated_with_rc=True,
    )
    positive = progress > 0.0
    best_generated = progress.masked_fill(~positive, -torch.inf).argmax(dim=1)
    has_positive = positive.any(dim=1)
    rows = torch.arange(len(progress))[has_positive]
    highest_rejected = torch.zeros(len(progress), dtype=torch.bool)
    highest_rejected[has_positive] = (
        rc[rows, best_generated[has_positive]] < eta_r
    )
    rc_count = (rc >= eta_r).sum(dim=1)

    methods = {}
    for method, index, covered in (
        (NO_RC_METHOD, no_rc_index, no_rc_covered),
        (RC_METHOD, rc_index, rc_covered),
    ):
        methods[method] = {
            "coverage": float(covered.float().mean()),
            "fallback_rate": float((~covered).float().mean()),
            "direct_goal_selected_rate": float(
                ((index == NUM_CANDIDATES) & covered).float().mean()
            ),
            "selected_d_psi_progress": distribution(
                gather_selected(progress, direct_progress, index)
            ),
            "selected_residual_norm": distribution(
                gather_selected(residual, direct_residual, index)
            ),
            "selected_manifold_distance": distribution(
                gather_selected(manifold, direct_manifold, index)
            ),
        }

    manifold_summary = distribution(manifold)
    return {
        "generator_seed": int(artifact["metadata"]["generator_seed"]),
        "source_count": len(projected),
        "projected_candidate_count": int(projected.numel() // projected.size(-1)),
        "unique_projected_bank_rows": int(projection_indices.unique().numel()),
        "projection_index_sha256": hashlib.sha256(
            projection_indices.numpy().tobytes()
        ).hexdigest(),
        "projection_is_exact_bank_row": True,
        "real_to_real_manifold_reference": real_manifold_reference,
        "projected_candidate_manifold_distance": manifold_summary,
        "projected_to_real_manifold_mean_ratio": (
            manifold_summary["mean"] / real_manifold_reference["mean"]
        ),
        "projected_candidate_residual_norm": distribution(residual),
        "projected_candidate_rc_score": distribution(rc),
        "projected_candidate_d_psi_progress": distribution(progress),
        "candidate_rc_pass_rate": float((rc >= eta_r).float().mean()),
        "source_n_rc": {
            "zero": float((rc_count == 0).float().mean()),
            "one": float((rc_count == 1).float().mean()),
            "at_least_two": float((rc_count >= 2).float().mean()),
        },
        "rc_positive_progress_coverage": float(
            rc_generated_feasible.any(dim=1).float().mean()
        ),
        "positive_progress_coverage_no_rc": float(
            no_rc_generated_feasible.any(dim=1).float().mean()
        ),
        "highest_progress_projected_candidate_rejection_rate": float(
            highest_rejected[has_positive].float().mean()
        ),
        "direct_goal_eligible_rate": float(
            ((direct_rc >= eta_r) & (direct_progress > 0.0)).float().mean()
        ),
        "methods": methods,
    }


def aggregate_offline(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_paths: dict[str, Callable[[dict[str, Any]], float]] = {
        "candidate_rc_pass_rate": lambda item: item["candidate_rc_pass_rate"],
        "rc_positive_progress_coverage": (
            lambda item: item["rc_positive_progress_coverage"]
        ),
        "highest_progress_rejection_rate": (
            lambda item: item[
                "highest_progress_projected_candidate_rejection_rate"
            ]
        ),
        "projected_to_real_manifold_mean_ratio": (
            lambda item: item["projected_to_real_manifold_mean_ratio"]
        ),
        "source_n_rc_zero": lambda item: item["source_n_rc"]["zero"],
        "source_n_rc_one": lambda item: item["source_n_rc"]["one"],
        "source_n_rc_at_least_two": (
            lambda item: item["source_n_rc"]["at_least_two"]
        ),
        "direct_goal_eligible_rate": (
            lambda item: item["direct_goal_eligible_rate"]
        ),
    }
    result = {
        key: mean_std([extract(item) for item in seed_reports])
        for key, extract in scalar_paths.items()
    }
    result["methods"] = {}
    for method in METHODS:
        result["methods"][method] = {
            metric: mean_std(
                [item["methods"][method][metric] for item in seed_reports]
            )
            for metric in (
                "coverage",
                "fallback_rate",
                "direct_goal_selected_rate",
            )
        }
        for metric in (
            "selected_d_psi_progress",
            "selected_residual_norm",
            "selected_manifold_distance",
        ):
            result["methods"][method][metric] = {
                statistic: mean_std(
                    [
                        item["methods"][method][metric][statistic]
                        for item in seed_reports
                    ]
                )
                for statistic in ("mean", "median", "p95")
            }
    no_rc = result["methods"][NO_RC_METHOD]
    rc = result["methods"][RC_METHOD]
    result["rc_minus_no_rc_selected_mean"] = {
        metric: (
            rc[metric]["mean"]["mean"] - no_rc[metric]["mean"]["mean"]
        )
        for metric in (
            "selected_d_psi_progress",
            "selected_residual_norm",
            "selected_manifold_distance",
        )
    }
    return result


def paired_closed_loop_bootstrap(
    seed_results: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(seed_results) != 3:
        raise ValueError("formal RC decision requires exactly three generator seeds")
    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "task_success": lambda record: float(record["success"]),
        "horizon_censored_completion_env_steps": lambda record: float(
            record["completion_env_steps"]
            if record["completion_env_steps"] is not None
            else TWOROOM_OFFICIAL_MAX_EPISODE_STEPS
        ),
        "fallback_rate": lambda record: float(record["fallback_rate"]),
        "d_psi_realized_progress": lambda record: float(
            record["total_d_psi_realized_progress"]
        ),
        "candidate_coverage": lambda record: float(record["candidate_coverage"]),
    }
    by_method_episode: dict[str, dict[int, list[dict[str, Any]]]] = {
        method: {} for method in METHODS
    }
    all_records = []
    for seed_result in seed_results:
        all_records.extend(seed_result["rollouts"])
        for record in seed_result["rollouts"]:
            by_method_episode[record["method"]].setdefault(
                int(record["episode_index"]), []
            ).append(record)
    rng = np.random.default_rng(seed)
    intervals: dict[str, Any] = {}
    differences: dict[str, Any] = {}
    for metric, extract in extractors.items():
        per_method = {}
        for method in METHODS:
            per_method[method] = {
                episode: float(np.mean([extract(record) for record in records]))
                for episode, records in by_method_episode[method].items()
            }
        common = sorted(set(per_method[RC_METHOD]) & set(per_method[NO_RC_METHOD]))
        if len(common) != EXPECTED_VALIDATION_EPISODES:
            raise ValueError(
                f"paired bootstrap has {len(common)} episodes, expected 411"
            )
        for method in METHODS:
            intervals.setdefault(method, {})[metric] = bootstrap_mean_95_ci(
                np.asarray([per_method[method][episode] for episode in common]),
                samples=samples,
                rng=rng,
            )
        delta = np.asarray(
            [
                per_method[RC_METHOD][episode] - per_method[NO_RC_METHOD][episode]
                for episode in common
            ]
        )
        differences[metric] = bootstrap_mean_95_ci(
            delta, samples=samples, rng=rng
        )
        differences[metric]["difference"] = f"{RC_METHOD} minus {NO_RC_METHOD}"

    success = differences["task_success"]
    if success["mean"] > 0.0 and success["lower_95"] > 0.0:
        primary_decision = "RC_positive_CI; review_offline_tradeoff_before_retention"
    elif success["mean"] < 0.0 and success["upper_95"] < 0.0:
        primary_decision = "hard_RC_gate_harmful; stop_proposal_aware_alignment"
    else:
        primary_decision = "inconclusive; CI_crosses_or_touches_zero"
    return {
        "seed_aggregation": "average three generator seeds within each episode",
        "bootstrap_unit": "paired validation episode",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "method_episode_bootstrap_95_ci": intervals,
        "paired_rc_minus_no_rc_95_ci": differences,
        "primary_success_rate_decision": primary_decision,
        "retention_requires_secondary_offline_tradeoff_review": True,
        "method_summaries_all_seed_episode_rollouts": {
            method: summarize_method(
                [record for record in all_records if record["method"] == method]
            )
            for method in METHODS
        },
    }


def build_protocol(
    args: argparse.Namespace,
    *,
    seeds: tuple[int, ...],
    eta_r: float,
    bank_size: int,
    bank_sha256: str,
    episode_indices: list[int],
) -> dict[str, Any]:
    return {
        "experiment": "on-manifold matched control for frozen hard RC gate",
        "training_report": str(args.training_report.expanduser().resolve()),
        "stage2_report": str(args.stage2_report.expanduser().resolve()),
        "latent_cache": str(args.latent_cache.expanduser().resolve()),
        "progress_ranker": str(args.progress_ranker.expanduser().resolve()),
        "policy": args.policy,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "dataset": args.dataset,
        "modules_trained_or_modified": False,
        "eta_r_recalibrated": False,
        "eta_r": eta_r,
        "tau_model_steps": TAU_MODEL_STEPS,
        "model_step_env_steps": MODEL_STEP_ENV_STEPS,
        "source_episode_range": [4000, 5000],
        "source_successful_only": True,
        "source_episode_count": EXPECTED_VALIDATION_EPISODES,
        "episode_indices": episode_indices,
        "test_split_used": False,
        "generator_seeds": list(seeds),
        "num_candidates": NUM_CANDIDATES,
        "projection": "exact Euclidean 1-NN bank row; no averaging",
        "projection_space": "raw unnormalized RC-aux 192D latent",
        "projection_bank_episode_range": [0, 4000],
        "projection_bank_successful_only": True,
        "projection_bank_size": bank_size,
        "projection_bank_sha256": bank_sha256,
        "manifold_diagnostic": "raw Euclidean k=5 mean-kNN after projection",
        "methods": list(METHODS),
        "direct_goal_rule_matched": True,
        "fallback": "flat z_T for one model step, then retry high level",
        "episode_horizon_env_steps": TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
        "subgoal_fixed_within_segment": True,
        "subgoal_horizon_sequence": [3, 2, 1],
        "execution_horizon_model_steps": 1,
        "num_samples": args.num_samples,
        "cem_iterations": args.cem_iterations,
        "topk": args.topk,
        "cem_seed": args.cem_seed,
        "cem_seed_rule": "cem_seed + validation episode position, matched by method",
        "env_seed": args.env_seed,
        "dropout_seed": args.dropout_seed,
        "closed_loop_dropout_seed_rule": (
            "dropout_seed + generator_seed * 10000 + episode_index + "
            "high_level_attempt_index; reset immediately before generation"
        ),
        "offline_candidates": (
            "reuse exact Stage 4C per-seed artifacts and fixed dropout candidates"
        ),
        "flat_planning_horizon_model_steps": (
            args.flat_planning_horizon_model_steps
        ),
        "reachability_cost_weight": args.reachability_cost_weight,
        "max_replay_pixel_diff": args.max_replay_pixel_diff,
        "offline_batch_size": args.offline_batch_size,
        "knn_query_chunk": args.knn_query_chunk,
        "knn_bank_chunk": args.knn_bank_chunk,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "stage4c_report": str(args.stage4c_report.expanduser().resolve()),
    }


def partial_report(
    protocol: dict[str, Any],
    offline: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage4d_on_manifold_complete": False,
        "protocol": protocol,
        "offline_seed_results": offline,
        "offline_across_generator_seeds": (
            aggregate_offline(offline) if len(offline) == 3 else None
        ),
        "closed_loop_seed_results": seed_results,
        "completed_rollouts": sum(len(item["rollouts"]) for item in seed_results),
        "expected_rollouts": (
            len(protocol["generator_seeds"])
            * len(protocol["episode_indices"])
            * len(METHODS)
        ),
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.generator_seeds)
    if args.device.startswith("cuda") and not args.validate_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    if min(
        args.offline_batch_size,
        args.knn_query_chunk,
        args.knn_bank_chunk,
        args.bootstrap_samples,
    ) < 1:
        raise ValueError("batch, chunk, and bootstrap values must be positive")

    stage2 = load_stage2_protocol(args.stage2_report)
    if not np.isclose(stage2["eta_r"], 0.5295726657, atol=1.0e-9, rtol=0.0):
        raise ValueError("control requires fixed Stage 2 eta_R=0.5295726657")
    stage3 = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(stage2, stage3, checkpoint_label="Stage 3 D_psi")
    cache = load_latent_cache(args.latent_cache)
    train_episodes, validation_episodes = train_validation_episodes_only(cache)
    if len(validation_episodes) != EXPECTED_VALIDATION_EPISODES:
        raise ValueError(
            f"expected 411 successful validation episodes, got {len(validation_episodes)}"
        )
    train_bank_cpu = torch.cat(
        [episode["latents"].to(torch.float32) for episode in train_episodes]
    )
    bank_sha256 = tensor_sha256(train_bank_cpu)
    episode_indices = [
        int(episode["episode_index"]) for episode in validation_episodes
    ]
    stage4c_report = read_json(args.stage4c_report)
    validate_stage4c_report(
        stage4c_report,
        seeds=seeds,
        eta_r=stage2["eta_r"],
        bank_sha256=bank_sha256,
        expected_episode_indices=set(episode_indices),
    )
    checkpoints_by_seed = {}
    for path in checkpoint_paths(args.training_report):
        checkpoint = torch.load(
            path.expanduser().resolve(), map_location="cpu", weights_only=False
        )
        checkpoints_by_seed[int(checkpoint["seed"])] = path
    if any(seed not in checkpoints_by_seed for seed in seeds):
        raise ValueError("training report is missing a requested generator seed")
    for seed in seeds:
        stage4 = load_checkpoint_protocol(checkpoints_by_seed[seed], stage=4)
        assert_protocol_consistency(
            stage2, stage4, checkpoint_label=f"Stage 4 generator seed {seed}"
        )
    protocol = build_protocol(
        args,
        seeds=seeds,
        eta_r=stage2["eta_r"],
        bank_size=len(train_bank_cpu),
        bank_sha256=bank_sha256,
        episode_indices=episode_indices,
    )
    if args.validate_only:
        print(
            f"validated {len(validation_episodes)} validation episodes, "
            f"{len(train_bank_cpu)} train-bank latents, and seeds {list(seeds)}"
        )
        return 0

    output_path = args.output.expanduser().resolve()
    existing: dict[str, Any] = {}
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("protocol") != protocol:
            raise ValueError("existing output uses a different protocol")
        if existing.get("stage4d_on_manifold_complete") is True:
            print(f"report already complete: {output_path}")
            return 0
    offline_by_seed = {
        int(item["generator_seed"]): item
        for item in existing.get("offline_seed_results", [])
    }
    closed_by_seed = {
        int(item["generator_seed"]): item
        for item in existing.get("closed_loop_seed_results", [])
    }

    device = torch.device(args.device)
    bank = train_bank_cpu.to(device)
    projector = ExactRealLatentProjector(
        bank,
        query_chunk=args.knn_query_chunk,
        bank_chunk=args.knn_bank_chunk,
    )
    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=args.flat_planning_horizon_model_steps,
        execution_horizon_model_steps=1,
        num_samples=args.num_samples,
        n_steps=args.cem_iterations,
        topk=args.topk,
        seed=args.cem_seed,
        warm_start=True,
    )
    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=args.cache_dir.expanduser().resolve(),
        device=device,
        planner_config=planner_config,
        use_reachability_cost=True,
        reachability_cost_weight=args.reachability_cost_weight,
    )
    adapter.model.interpolate_pos_encoding = True
    progress_ranker, _ = load_progress_ranker(args.progress_ranker, device=device)
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("E_theta, F_theta, and R_phi must remain frozen")
    if any(parameter.requires_grad for parameter in progress_ranker.parameters()):
        raise RuntimeError("D_psi must remain frozen")

    offline_results = []
    for seed in seeds:
        if seed not in offline_by_seed:
            artifact = torch.load(
                artifact_path_from_report(stage4c_report, seed),
                map_location="cpu",
                weights_only=False,
            )
            offline_by_seed[seed] = evaluate_projected_offline_seed(
                artifact,
                projector,
                adapter,
                progress_ranker,
                eta_r=stage2["eta_r"],
                batch_size=args.offline_batch_size,
                query_chunk=args.knn_query_chunk,
                bank_chunk=args.knn_bank_chunk,
                device=device,
                real_manifold_reference=stage4c_report[
                    "real_to_real_manifold_reference"
                ],
            )
        offline_results.append(offline_by_seed[seed])
        atomic_write_json(
            partial_report(
                protocol,
                offline_results,
                [closed_by_seed[key] for key in seeds if key in closed_by_seed],
            ),
            args.output,
        )
    if args.offline_only:
        print(f"offline report_path: {output_path}")
        return 0

    dataset_path = args.cache_dir.expanduser().resolve() / args.dataset
    with h5py.File(dataset_path, "r") as handle:
        environment_records = [
            load_environment_record(handle, episode)
            for episode in validation_episodes
        ]
    closed_results = []
    for seed in seeds:
        checkpoint_path = checkpoints_by_seed[seed]
        generator, checkpoint = load_generator_checkpoint(
            checkpoint_path, device=device
        )
        generator.requires_grad_(False)
        if int(checkpoint["seed"]) != seed:
            raise ValueError("generator checkpoint seed mismatch")
        seed_result = closed_by_seed.get(
            seed,
            {
                "generator_seed": seed,
                "generator_checkpoint": str(
                    checkpoint_path.expanduser().resolve()
                ),
                "rollouts": [],
            },
        )
        closed_results.append(seed_result)
        completed = {
            (int(record["episode_index"]), record["method"])
            for record in seed_result["rollouts"]
        }
        if len(completed) != len(seed_result["rollouts"]):
            raise ValueError(f"duplicate saved rollout for generator seed {seed}")
        for episode_position, record in enumerate(environment_records):
            for output_method in METHODS:
                key = (int(record["episode_index"]), output_method)
                if key in completed:
                    continue
                result = run_rollout(
                    SELECTOR_METHOD[output_method],
                    generator,
                    progress_ranker,
                    adapter,
                    record,
                    num_candidates=NUM_CANDIDATES,
                    episode_horizon_env_steps=TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
                    flat_horizon=args.flat_planning_horizon_model_steps,
                    eta_r=stage2["eta_r"],
                    env_seed=args.env_seed,
                    cem_seed=args.cem_seed + episode_position,
                    dropout_seed=(
                        args.dropout_seed
                        + seed * 10000
                        + int(record["episode_index"])
                    ),
                    max_replay_pixel_diff=args.max_replay_pixel_diff,
                    device=device,
                    candidate_transform=projector,
                    reseed_dropout_each_high_level_attempt=True,
                )
                result["selector_method"] = result["method"]
                result["method"] = output_method
                result["candidate_projection"] = "exact_train_bank_1nn"
                seed_result["rollouts"].append(result)
                completed.add(key)
                atomic_write_json(
                    partial_report(protocol, offline_results, closed_results),
                    args.output,
                )
                print(
                    f"seed={seed} episode={record['episode_index']} "
                    f"method={output_method} success={result['success']} "
                    f"steps={result['env_steps']}",
                    flush=True,
                )
        seed_result["method_summaries"] = {
            method: summarize_method(
                [
                    record
                    for record in seed_result["rollouts"]
                    if record["method"] == method
                ]
            )
            for method in METHODS
        }
        atomic_write_json(
            partial_report(protocol, offline_results, closed_results),
            args.output,
        )

    expected_rollouts = len(validation_episodes) * len(METHODS)
    if any(len(item["rollouts"]) != expected_rollouts for item in closed_results):
        raise RuntimeError("closed-loop matched control is incomplete")
    if seeds != FORMAL_SEEDS:
        atomic_write_json(
            partial_report(protocol, offline_results, closed_results),
            args.output,
        )
        print(f"pilot report_path: {output_path}")
        return 0

    paired = paired_closed_loop_bootstrap(
        closed_results,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report = {
        "stage4d_on_manifold_complete": True,
        "protocol": protocol,
        "offline_seed_results": offline_results,
        "offline_across_generator_seeds": aggregate_offline(offline_results),
        "closed_loop_seed_results": closed_results,
        "paired_closed_loop_analysis": paired,
        "decision_rule": {
            "primary_metric": "Delta SR = SR_RC - SR_noRC",
            "retain_rc": (
                "Delta SR > 0 and paired-bootstrap lower 95% CI > 0, then only "
                "if offline tradeoff no longer shows substantial high-progress "
                "rejection with negligible improvement elsewhere"
            ),
            "reject_hard_gate": "Delta SR < 0 and upper 95% CI < 0",
            "inconclusive": "paired 95% CI includes or touches zero",
            "secondary_tradeoff_numeric_threshold_invented": False,
        },
        "interpretation_constraints": [
            "The test split [5000,10000) is not used.",
            "No model is trained or modified and eta_R is not recalibrated.",
            "A positive success CI is necessary but not sufficient to retain RC.",
            "No unrequested numeric threshold is imposed on the secondary tradeoff.",
        ],
    }
    atomic_write_json(report, args.output)
    print(f"report_path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
