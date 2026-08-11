#!/usr/bin/env python3
"""CONTROL ONLY: Stage 5B 1-NN/D_psi comparison, not deployment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401 - registers TwoRoom
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
    left_padded_history,
    load_generator_checkpoint,
    load_latent_cache,
    sample_subgoal_candidates,
)
from stage5_global_reachability import (
    GAMMA,
    HIDDEN_DIMS,
    LATENT_DIM,
    load_global_hitting_potential,
)
from tools.diagnose_stage4c_generated_rc import (
    KNN_K,
    exact_knn_mean_distance,
    tensor_sha256,
)
from tools.evaluate_stage4_candidates import checkpoint_paths, load_progress_ranker
from tools.evaluate_stage4_closed_loop import (
    TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
    atomic_write_json,
    bootstrap_mean_95_ci,
    execute_action_block,
    frame_error,
    load_environment_record,
    pairwise_l2,
    synchronize,
)
from tools.evaluate_stage4d_on_manifold_rc import (
    ExactRealLatentProjector,
    artifact_path_from_report,
    train_validation_episodes_only,
    validate_stage4c_report,
)
from tools.train_stage5a_global_reachability import (
    build_temporal_pair_buckets,
    score_temporal_pairs,
    temporal_validation_report,
)


FORMAL_SEEDS = (3072, 3073, 3074)
NUM_CANDIDATES = 32
EXPECTED_VALIDATION_EPISODES = 411
METHOD_A = "A_dpsi_baseline"
METHOD_B = "B_rg_rank_only"
METHOD_C = "C_rg_filter_rank"
METHODS = (METHOD_A, METHOD_B, METHOD_C)

DEFAULT_DROPOUT_SEED = 20260809
DEFAULT_CEM_SEED = 4200
DEFAULT_ENV_SEED = 42
DEFAULT_BOOTSTRAP_SEED = 20260815
DEFAULT_NUM_SAMPLES = 300
DEFAULT_CEM_ITERATIONS = 30
DEFAULT_TOPK = 30
DEFAULT_FLAT_HORIZON = 5
DEFAULT_REACHABILITY_COST_WEIGHT = 0.85
DEFAULT_MAX_REPLAY_PIXEL_DIFF = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5B frozen R_G filtering/ranking comparison."
    )
    parser.add_argument(
        "--stage5a-report",
        type=Path,
        default=Path("outputs/stage5a_global_reachability.json"),
    )
    parser.add_argument(
        "--stage5a-checkpoint-dir",
        type=Path,
        default=Path("outputs/stage5a_global_reachability_checkpoints"),
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
    parser.add_argument("--seeds", default="3072,3073,3074")
    parser.add_argument("--offline-batch-size", type=int, default=256)
    parser.add_argument("--audit-batch-size", type=int, default=8192)
    parser.add_argument("--knn-query-chunk", type=int, default=256)
    parser.add_argument("--knn-bank-chunk", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--cem-iterations", type=int, default=DEFAULT_CEM_ITERATIONS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--cem-seed", type=int, default=DEFAULT_CEM_SEED)
    parser.add_argument("--env-seed", type=int, default=DEFAULT_ENV_SEED)
    parser.add_argument("--dropout-seed", type=int, default=DEFAULT_DROPOUT_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
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
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage5b_hecrl_selector.json"),
    )
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if seeds not in ((3072,), FORMAL_SEEDS):
        raise ValueError("seeds must be 3072 pilot or 3072,3073,3074")
    return seeds


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text())


def distribution(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    return distribution_summary(array.reshape(-1).tolist())


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def stage5a_checkpoint_path(checkpoint_dir: Path, seed: int) -> Path:
    return checkpoint_dir.expanduser().resolve() / (
        f"stage5a_global_reachability_seed{seed}.pt"
    )


def validate_stage5a_artifacts(
    report: dict[str, Any],
    checkpoint_dir: Path,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    if seeds == FORMAL_SEEDS and report.get("stage5a_complete") is not True:
        raise ValueError("formal Stage 5B requires complete three-seed Stage 5A")
    protocol = report["protocol"]
    expected = {
        "train_episode_range": [0, 4000],
        "local_threshold_calibration_episode_range": [4000, 4500],
        "local_threshold_heldout_episode_range": [4500, 5000],
        "test_episode_range_5000_10000_used": False,
        "latent_dim": LATENT_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "gamma": GAMMA,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"Stage 5A {key}={protocol.get(key)!r}, expected {value!r}"
            )
    results = {int(item["seed"]): item for item in report["seed_results"]}
    if any(seed not in results for seed in seeds):
        raise ValueError("Stage 5A report is missing a requested seed")
    checkpoints: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = stage5a_checkpoint_path(checkpoint_dir, seed)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        result = results[seed]
        if Path(result["checkpoint"]).expanduser().resolve() != path:
            raise ValueError(f"Stage 5A report/checkpoint path mismatch for seed {seed}")
        if checkpoint.get("protocol") != protocol:
            raise ValueError(f"Stage 5A checkpoint protocol mismatch for seed {seed}")
        eta_report = float(result["local_tau3_validation"]["eta_3"])
        if checkpoint.get("stage") != "5A" or int(checkpoint["seed"]) != seed:
            raise ValueError(f"invalid Stage 5A checkpoint for seed {seed}")
        if not checkpoint.get("not_absolute_global_reachability", False):
            raise ValueError("R_G checkpoint has invalid reachability semantics")
        if not np.isclose(checkpoint["gamma"], GAMMA, atol=0.0, rtol=0.0):
            raise ValueError("Stage 5B forbids changing gamma")
        if not np.isclose(
            checkpoint["eta_3"], eta_report, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(f"seed {seed} eta_3 differs between report/checkpoint")
        checkpoints[seed] = {
            "path": path,
            "eta_3": eta_report,
            "checkpoint": checkpoint,
        }
    return checkpoints


def heldout_audit_episodes(
    validation_episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    heldout = [
        episode
        for episode in validation_episodes
        if 4500 <= int(episode["episode_index"]) < 5000
    ]
    if not heldout or any(int(item["episode_index"]) < 4500 for item in heldout):
        raise ValueError("held-out audit must use only [4500,5000)")
    return heldout


@torch.inference_mode()
def long_range_audit(
    model,
    heldout_episodes: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    buckets = build_temporal_pair_buckets(heldout_episodes)
    scored = score_temporal_pairs(
        model,
        heldout_episodes,
        buckets,
        batch_size=batch_size,
        device=device,
    )
    rg = temporal_validation_report(scored, gamma=GAMMA)["models"]["r_g"]
    bucket_means = {
        str(distance): rg["score_by_distance_bucket"][str(distance)]["mean"]
        for distance in range(1, 21)
    }
    audit = {
        "episode_range": [4500, 5000],
        "episode_count": len(heldout_episodes),
        "pair_count": int(len(scored["distance"])),
        "global_oriented_spearman": rg["global_oriented_spearman"],
        "same_source_pairwise_ordering": rg["same_source_ordering"][
            "pairwise_temporal_order_accuracy"
        ],
        "same_goal_pairwise_ordering": rg["same_goal_ordering"][
            "pairwise_temporal_order_accuracy"
        ],
        "bucket_means_d1_d20": bucket_means,
        "tail_d15_d20_linear_slope": rg["saturation_diagnostics"][
            "tail_d15_d20_linear_slope"
        ],
    }
    checks = {
        "all_metrics_finite": bool(
            np.isfinite(
                [
                    audit["global_oriented_spearman"],
                    audit["same_source_pairwise_ordering"],
                    audit["same_goal_pairwise_ordering"],
                    audit["tail_d15_d20_linear_slope"],
                    *bucket_means.values(),
                ]
            ).all()
        ),
        "oriented_spearman_positive": audit["global_oriented_spearman"] > 0.0,
        "same_source_above_chance": audit["same_source_pairwise_ordering"] > 0.5,
        "same_goal_above_chance": audit["same_goal_pairwise_ordering"] > 0.5,
        "tail_slope_has_expected_sign": audit["tail_d15_d20_linear_slope"] < 0.0,
    }
    audit["obvious_directional_collapse"] = not all(checks.values())
    audit["collapse_checks"] = checks
    audit["collapse_rule_is_directional_not_tuned"] = True
    return audit


def select_indices(
    method: str,
    local_rg: torch.Tensor,
    goal_rg: torch.Tensor,
    d_psi_progress: torch.Tensor,
    direct_eligible: torch.Tensor,
    *,
    eta_3: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select candidate indices, using N as direct goal and -1 as fallback."""

    if local_rg.ndim != 2 or goal_rg.shape != local_rg.shape:
        raise ValueError("R_G candidate scores must be [S,N]")
    if d_psi_progress.shape != local_rg.shape:
        raise ValueError("D_psi progress must match candidate scores")
    if direct_eligible.shape != local_rg.shape[:1]:
        raise ValueError("direct-goal eligibility must be [S]")
    if method == METHOD_A:
        feasible = d_psi_progress > 0.0
        candidate_index = d_psi_progress.masked_fill(~feasible, -torch.inf).argmax(1)
    elif method == METHOD_B:
        feasible = torch.ones_like(local_rg, dtype=torch.bool)
        candidate_index = goal_rg.argmax(1)
    elif method == METHOD_C:
        feasible = local_rg >= eta_3
        candidate_index = goal_rg.masked_fill(~feasible, -torch.inf).argmax(1)
    else:
        raise ValueError(f"unknown Stage 5B method: {method}")
    candidate_covered = feasible.any(dim=1)
    selected = torch.where(
        candidate_covered,
        candidate_index,
        -torch.ones_like(candidate_index),
    )
    selected = torch.where(
        direct_eligible,
        torch.full_like(selected, local_rg.size(1)),
        selected,
    )
    return selected, candidate_covered


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
def evaluate_offline_seed(
    artifact: dict[str, Any],
    projector: ExactRealLatentProjector,
    r_g,
    d_psi,
    *,
    eta_3: float,
    batch_size: int,
    query_chunk: int,
    bank_chunk: int,
    device: torch.device,
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
    if not torch.equal(
        projected,
        projector.bank[projection_indices.to(device)].cpu(),
    ):
        raise RuntimeError("projection did not return exact latent-bank rows")

    local_parts = []
    goal_score_parts = []
    progress_parts = []
    direct_rg_parts = []
    direct_goal_score_parts = []
    direct_progress_parts = []
    for start in range(0, len(projected), batch_size):
        current = current_cpu[start : start + batch_size].to(device)
        goal = goal_cpu[start : start + batch_size].to(device)
        candidate = projected[start : start + batch_size].to(device)
        count = len(current)
        flat_candidate = candidate.flatten(0, 1)
        repeated_current = current.repeat_interleave(NUM_CANDIDATES, dim=0)
        repeated_goal = goal.repeat_interleave(NUM_CANDIDATES, dim=0)
        local_parts.append(
            r_g(repeated_current, flat_candidate).reshape(count, NUM_CANDIDATES).cpu()
        )
        goal_score_parts.append(
            r_g(flat_candidate, repeated_goal).reshape(count, NUM_CANDIDATES).cpu()
        )
        source_dpsi = d_psi(current, goal)
        candidate_dpsi = d_psi(flat_candidate, repeated_goal).reshape(
            count, NUM_CANDIDATES
        )
        progress_parts.append((source_dpsi.unsqueeze(1) - candidate_dpsi).cpu())
        direct_rg_parts.append(r_g(current, goal).cpu())
        direct_goal_score_parts.append(r_g(goal, goal).cpu())
        direct_progress_parts.append((source_dpsi - d_psi(goal, goal)).cpu())
    local_rg = torch.cat(local_parts)
    goal_rg = torch.cat(goal_score_parts)
    d_psi_progress = torch.cat(progress_parts)
    direct_rg = torch.cat(direct_rg_parts)
    direct_goal_rg = torch.cat(direct_goal_score_parts)
    direct_progress = torch.cat(direct_progress_parts)
    direct_eligible = direct_rg >= eta_3
    residual = (projected - current_cpu.unsqueeze(1)).norm(dim=-1)
    direct_residual = (goal_cpu - current_cpu).norm(dim=-1)
    manifold = exact_knn_mean_distance(
        projected.flatten(0, 1).to(device),
        projector.bank,
        k=KNN_K,
        query_chunk=query_chunk,
        bank_chunk=bank_chunk,
    ).reshape_as(local_rg).cpu()
    direct_manifold = exact_knn_mean_distance(
        goal_cpu.to(device),
        projector.bank,
        k=KNN_K,
        query_chunk=query_chunk,
        bank_chunk=bank_chunk,
    ).cpu()

    methods = {}
    for method in METHODS:
        indices, candidate_covered = select_indices(
            method,
            local_rg,
            goal_rg,
            d_psi_progress,
            direct_eligible,
            eta_3=eta_3,
        )
        decision_covered = indices >= 0
        methods[method] = {
            "generated_candidate_coverage": float(candidate_covered.float().mean()),
            "decision_coverage_including_direct_goal": float(
                decision_covered.float().mean()
            ),
            "fallback_rate": float((~decision_covered).float().mean()),
            "direct_goal_eligible_rate": float(direct_eligible.float().mean()),
            "direct_goal_selected_rate": float(
                ((indices == NUM_CANDIDATES) & decision_covered).float().mean()
            ),
            "selected_r_g_source_to_subgoal": distribution(
                gather_selected(local_rg, direct_rg, indices)
            ),
            "selected_r_g_subgoal_to_goal": distribution(
                gather_selected(goal_rg, direct_goal_rg, indices)
            ),
            "selected_d_psi_progress": distribution(
                gather_selected(d_psi_progress, direct_progress, indices)
            ),
            "selected_residual_norm": distribution(
                gather_selected(residual, direct_residual, indices)
            ),
            "selected_mean_5nn_manifold_distance": distribution(
                gather_selected(manifold, direct_manifold, indices)
            ),
        }

    pass_mask = local_rg >= eta_3
    pass_count = pass_mask.sum(dim=1)
    best_b = goal_rg.argmax(dim=1)
    rows = torch.arange(len(goal_rg))
    best_b_rejected = ~pass_mask[rows, best_b]
    selector_invoked = ~direct_eligible
    return {
        "generator_seed": int(artifact["metadata"]["generator_seed"]),
        "r_g_seed": int(artifact["metadata"]["generator_seed"]),
        "eta_3": eta_3,
        "source_count": len(projected),
        "projected_candidate_count": int(projected.numel() // projected.size(-1)),
        "projection_is_exact_bank_row": True,
        "unique_projected_bank_rows": int(projection_indices.unique().numel()),
        "candidate_r_g_pass_rate_method_c": float(pass_mask.float().mean()),
        "source_n_pass_method_c": {
            "zero": float((pass_count == 0).float().mean()),
            "one": float((pass_count == 1).float().mean()),
            "at_least_two": float((pass_count >= 2).float().mean()),
        },
        "method_c_rejects_method_b_top_r_g_candidate_rate": float(
            best_b_rejected.float().mean()
        ),
        "method_c_rejects_method_b_top_r_g_candidate_rate_direct_ineligible": (
            float(best_b_rejected[selector_invoked].float().mean())
            if bool(selector_invoked.any())
            else None
        ),
        "methods": methods,
    }


@torch.inference_mode()
def choose_stage5b_subgoal(
    method: str,
    generator,
    r_g,
    d_psi,
    history: deque[torch.Tensor],
    goal: torch.Tensor,
    *,
    eta_3: float,
    num_candidates: int,
    device: torch.device,
    candidate_transform: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"unknown Stage 5B method: {method}")
    if num_candidates != NUM_CANDIDATES:
        raise ValueError("Stage 5B fixes 32 candidates")
    current = history[-1].to(device)
    goal = goal.to(device)
    diagnostics = {
        "candidate_count": NUM_CANDIDATES,
        "candidate_covered": False,
        "decision_covered": False,
        "r_g_pass_count": None,
        "direct_goal_eligible": False,
        "direct_goal_selected": False,
        "selected_r_g_source_to_subgoal": None,
        "selected_r_g_subgoal_to_goal": None,
        "selected_d_psi_progress": None,
        "pairwise_diversity": None,
        "timing_seconds": {
            "generator_and_projection": 0.0,
            "r_g": 0.0,
            "d_psi": 0.0,
        },
    }

    history_tensor, mask = left_padded_history(history)
    synchronize(device)
    started = time.perf_counter()
    candidates = sample_subgoal_candidates(
        generator,
        history_tensor.to(device),
        mask.to(device),
        goal.unsqueeze(0),
        num_candidates=NUM_CANDIDATES,
        stochastic=True,
    )[0]
    candidates = candidate_transform(candidates).to(
        device=device, dtype=candidates.dtype
    )
    if candidates.shape != (NUM_CANDIDATES, LATENT_DIM):
        raise ValueError("projected candidates must be [32,192]")
    synchronize(device)
    diagnostics["timing_seconds"]["generator_and_projection"] = (
        time.perf_counter() - started
    )
    diagnostics["pairwise_diversity"] = pairwise_l2(candidates)

    synchronize(device)
    started = time.perf_counter()
    local_rg = r_g(current.expand(NUM_CANDIDATES, -1), candidates)
    goal_rg = r_g(candidates, goal.expand(NUM_CANDIDATES, -1))
    direct_rg = r_g(current.unsqueeze(0), goal.unsqueeze(0))[0]
    direct_goal_rg = r_g(goal.unsqueeze(0), goal.unsqueeze(0))[0]
    synchronize(device)
    diagnostics["timing_seconds"]["r_g"] = time.perf_counter() - started

    synchronize(device)
    started = time.perf_counter()
    source_dpsi = d_psi(current.unsqueeze(0), goal.unsqueeze(0))[0]
    candidate_dpsi = d_psi(candidates, goal.expand(NUM_CANDIDATES, -1))
    direct_dpsi = d_psi(goal.unsqueeze(0), goal.unsqueeze(0))[0]
    d_psi_progress = source_dpsi - candidate_dpsi
    direct_progress = source_dpsi - direct_dpsi
    synchronize(device)
    diagnostics["timing_seconds"]["d_psi"] = time.perf_counter() - started

    direct_eligible = direct_rg >= eta_3
    indices, candidate_covered = select_indices(
        method,
        local_rg.unsqueeze(0),
        goal_rg.unsqueeze(0),
        d_psi_progress.unsqueeze(0),
        direct_eligible.unsqueeze(0),
        eta_3=eta_3,
    )
    selected_index = int(indices[0])
    diagnostics["candidate_covered"] = bool(candidate_covered[0])
    diagnostics["decision_covered"] = selected_index >= 0
    diagnostics["direct_goal_eligible"] = bool(direct_eligible)
    if method == METHOD_C:
        diagnostics["r_g_pass_count"] = int((local_rg >= eta_3).sum())
    if selected_index < 0:
        return None, diagnostics
    if selected_index == NUM_CANDIDATES:
        selected = goal
        diagnostics["direct_goal_selected"] = True
        diagnostics["selected_r_g_source_to_subgoal"] = float(direct_rg)
        diagnostics["selected_r_g_subgoal_to_goal"] = float(direct_goal_rg)
        diagnostics["selected_d_psi_progress"] = float(direct_progress)
    else:
        selected = candidates[selected_index]
        diagnostics["selected_r_g_source_to_subgoal"] = float(
            local_rg[selected_index]
        )
        diagnostics["selected_r_g_subgoal_to_goal"] = float(
            goal_rg[selected_index]
        )
        diagnostics["selected_d_psi_progress"] = float(
            d_psi_progress[selected_index]
        )
    return selected.detach(), diagnostics


@torch.inference_mode()
def run_stage5b_rollout(
    method: str,
    generator,
    r_g,
    d_psi,
    adapter: RCAuxAdapter,
    record: dict[str, Any],
    *,
    eta_3: float,
    episode_horizon_env_steps: int,
    flat_horizon: int,
    env_seed: int,
    cem_seed: int,
    dropout_seed: int,
    max_replay_pixel_diff: int,
    device: torch.device,
    candidate_transform: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        max_episode_steps=episode_horizon_env_steps,
        disable_env_checker=True,
    )
    torch.manual_seed(dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dropout_seed)
    adapter.reset_planner(seed=cem_seed)
    env_steps = 0
    success = False
    truncated = False
    fallback_model_steps = 0
    high_level_attempts = 0
    candidate_covered_attempts = 0
    decision_covered_attempts = 0
    direct_goal_eligible_attempts = 0
    direct_goal_selected_attempts = 0
    selected_local_rg: list[float] = []
    selected_goal_rg: list[float] = []
    selected_dpsi_progress: list[float] = []
    candidate_diversities: list[float] = []
    segment_records = []
    timing = {
        "generator_and_projection": 0.0,
        "r_g": 0.0,
        "d_psi": 0.0,
        "cem": 0.0,
    }

    try:
        env.reset(seed=env_seed)
        base_env = env.unwrapped
        base_env._set_state(record["source_state"])
        base_env._set_goal_state(record["task_target_state"])
        source_error = frame_error(record["source_image"], env.render())
        rendered_goal = (
            base_env._render_frame(
                agent_pos=torch.as_tensor(record["terminal_state"])
            )
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        goal_error = frame_error(record["goal_image"], rendered_goal)
        if max(source_error, goal_error) > max_replay_pixel_diff:
            raise RuntimeError("dataset/environment replay mismatch")

        goal = record["reference_latents"][-1].to(device)
        current = adapter.encode_observation(env.render())[:, -1][0]
        history: deque[torch.Tensor] = deque([current.cpu()], maxlen=3)

        while (
            env_steps < episode_horizon_env_steps
            and not success
            and not truncated
        ):
            high_level_attempts += 1
            attempt_seed = dropout_seed + high_level_attempts - 1
            torch.manual_seed(attempt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(attempt_seed)
            subgoal, diagnostics = choose_stage5b_subgoal(
                method,
                generator,
                r_g,
                d_psi,
                history,
                goal,
                eta_3=eta_3,
                num_candidates=NUM_CANDIDATES,
                device=device,
                candidate_transform=candidate_transform,
            )
            for key in ("generator_and_projection", "r_g", "d_psi"):
                timing[key] += diagnostics["timing_seconds"][key]
            candidate_covered_attempts += int(diagnostics["candidate_covered"])
            decision_covered_attempts += int(diagnostics["decision_covered"])
            direct_goal_eligible_attempts += int(
                diagnostics["direct_goal_eligible"]
            )
            direct_goal_selected_attempts += int(
                diagnostics["direct_goal_selected"]
            )
            if diagnostics["pairwise_diversity"] is not None:
                candidate_diversities.append(diagnostics["pairwise_diversity"])

            segment_start = env_steps
            start_latent = history[-1].to(device)
            synchronize(device)
            started = time.perf_counter()
            start_rg_goal = r_g(
                start_latent.unsqueeze(0), goal.unsqueeze(0)
            )[0]
            synchronize(device)
            timing["r_g"] += time.perf_counter() - started
            synchronize(device)
            started = time.perf_counter()
            start_dpsi = d_psi(
                start_latent.unsqueeze(0), goal.unsqueeze(0)
            )[0]
            synchronize(device)
            timing["d_psi"] += time.perf_counter() - started
            if subgoal is None:
                plan = adapter.plan_to_latent(
                    env.render(),
                    goal.unsqueeze(0),
                    planning_horizon_model_steps=flat_horizon,
                    execution_horizon_model_steps=1,
                    force_reset_warm_start=True,
                )
                timing["cem"] += plan.diagnostics.planning_time_seconds
                executed, success, truncated, _ = execute_action_block(
                    env,
                    plan.actions_to_execute_env_steps[0],
                    remaining_budget=episode_horizon_env_steps - env_steps,
                )
                env_steps += executed
                fallback_model_steps += 1
                current = adapter.encode_observation(env.render())[:, -1][0]
                history.append(current.cpu())
                fallback = True
            else:
                selected_local_rg.append(
                    diagnostics["selected_r_g_source_to_subgoal"]
                )
                selected_goal_rg.append(
                    diagnostics["selected_r_g_subgoal_to_goal"]
                )
                selected_dpsi_progress.append(
                    diagnostics["selected_d_psi_progress"]
                )
                for h_rem in range(TAU_MODEL_STEPS, 0, -1):
                    if (
                        env_steps >= episode_horizon_env_steps
                        or success
                        or truncated
                    ):
                        break
                    plan = adapter.plan_to_latent(
                        env.render(),
                        subgoal.unsqueeze(0),
                        planning_horizon_model_steps=h_rem,
                        execution_horizon_model_steps=1,
                        force_reset_warm_start=h_rem == TAU_MODEL_STEPS,
                    )
                    timing["cem"] += plan.diagnostics.planning_time_seconds
                    executed, success, truncated, _ = execute_action_block(
                        env,
                        plan.actions_to_execute_env_steps[0],
                        remaining_budget=episode_horizon_env_steps - env_steps,
                    )
                    env_steps += executed
                    current = adapter.encode_observation(env.render())[:, -1][0]
                    history.append(current.cpu())
                fallback = False

            end_latent = history[-1].to(device)
            synchronize(device)
            started = time.perf_counter()
            end_rg_goal = r_g(end_latent.unsqueeze(0), goal.unsqueeze(0))[0]
            synchronize(device)
            timing["r_g"] += time.perf_counter() - started
            synchronize(device)
            started = time.perf_counter()
            end_dpsi = d_psi(end_latent.unsqueeze(0), goal.unsqueeze(0))[0]
            synchronize(device)
            timing["d_psi"] += time.perf_counter() - started
            segment_records.append(
                {
                    "env_step_start": segment_start,
                    "env_step_end": env_steps,
                    "r_g_goal_progress": float(end_rg_goal - start_rg_goal),
                    "d_psi_progress": float(start_dpsi - end_dpsi),
                    "fallback": fallback,
                }
            )
    finally:
        env.close()

    return {
        "episode_index": int(record["episode_index"]),
        "method": method,
        "success": bool(success),
        "truncated": bool(truncated),
        "env_steps": env_steps,
        "completion_env_steps": env_steps if success else None,
        "horizon_censored_completion_env_steps": (
            env_steps if success else episode_horizon_env_steps
        ),
        "high_level_attempts": high_level_attempts,
        "candidate_covered_attempts": candidate_covered_attempts,
        "decision_covered_attempts": decision_covered_attempts,
        "candidate_coverage": (
            candidate_covered_attempts / high_level_attempts
            if high_level_attempts
            else 0.0
        ),
        "decision_coverage_including_direct_goal": (
            decision_covered_attempts / high_level_attempts
            if high_level_attempts
            else 0.0
        ),
        "fallback_model_steps": fallback_model_steps,
        "fallback_rate": (
            fallback_model_steps / high_level_attempts
            if high_level_attempts
            else 0.0
        ),
        "direct_goal_eligible_attempts": direct_goal_eligible_attempts,
        "direct_goal_selected_attempts": direct_goal_selected_attempts,
        "direct_goal_eligible_rate": (
            direct_goal_eligible_attempts / high_level_attempts
            if high_level_attempts
            else 0.0
        ),
        "direct_goal_selected_rate": (
            direct_goal_selected_attempts / high_level_attempts
            if high_level_attempts
            else 0.0
        ),
        "selected_r_g_source_to_subgoal": selected_local_rg,
        "selected_r_g_subgoal_to_goal": selected_goal_rg,
        "selected_d_psi_progress": selected_dpsi_progress,
        "candidate_pairwise_diversities": candidate_diversities,
        "segments": segment_records,
        "total_r_g_goal_progress": sum(
            segment["r_g_goal_progress"] for segment in segment_records
        ),
        "total_d_psi_realized_progress": sum(
            segment["d_psi_progress"] for segment in segment_records
        ),
        "timing_seconds": timing,
    }


def summarize_rollouts(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    attempts = sum(record["high_level_attempts"] for record in records)
    return {
        "episode_seed_rollout_count": len(records),
        "success_rate": len(successes) / len(records),
        "completion_env_steps_success_only": (
            distribution([record["completion_env_steps"] for record in successes])
            if successes
            else None
        ),
        "horizon_censored_completion_env_steps": distribution(
            [record["horizon_censored_completion_env_steps"] for record in records]
        ),
        "fallback_rate": (
            sum(record["fallback_model_steps"] for record in records) / attempts
            if attempts
            else 0.0
        ),
        "candidate_coverage": (
            sum(record["candidate_covered_attempts"] for record in records) / attempts
            if attempts
            else 0.0
        ),
        "decision_coverage_including_direct_goal": (
            sum(record["decision_covered_attempts"] for record in records) / attempts
            if attempts
            else 0.0
        ),
        "direct_goal_eligible_rate": (
            sum(record["direct_goal_eligible_attempts"] for record in records)
            / attempts
            if attempts
            else 0.0
        ),
        "direct_goal_selected_rate": (
            sum(record["direct_goal_selected_attempts"] for record in records)
            / attempts
            if attempts
            else 0.0
        ),
        "total_r_g_goal_progress": distribution(
            [record["total_r_g_goal_progress"] for record in records]
        ),
        "total_d_psi_realized_progress": distribution(
            [record["total_d_psi_realized_progress"] for record in records]
        ),
        "segment_r_g_goal_progress": distribution(
            [
                segment["r_g_goal_progress"]
                for record in records
                for segment in record["segments"]
            ]
        ),
        "segment_d_psi_progress": distribution(
            [
                segment["d_psi_progress"]
                for record in records
                for segment in record["segments"]
            ]
        ),
        "timing_seconds_total": {
            key: sum(record["timing_seconds"][key] for record in records)
            for key in ("generator_and_projection", "r_g", "d_psi", "cem")
        },
    }


def aggregate_offline(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    direct_ineligible_rejection = [
        item[
            "method_c_rejects_method_b_top_r_g_candidate_rate_direct_ineligible"
        ]
        for item in seed_reports
        if item[
            "method_c_rejects_method_b_top_r_g_candidate_rate_direct_ineligible"
        ]
        is not None
    ]
    result = {
        "candidate_r_g_pass_rate_method_c": mean_std(
            [item["candidate_r_g_pass_rate_method_c"] for item in seed_reports]
        ),
        "method_c_rejects_method_b_top_r_g_candidate_rate": mean_std(
            [
                item["method_c_rejects_method_b_top_r_g_candidate_rate"]
                for item in seed_reports
            ]
        ),
        "method_c_rejects_method_b_top_r_g_candidate_rate_direct_ineligible": (
            mean_std(direct_ineligible_rejection)
            if direct_ineligible_rejection
            else None
        ),
        "source_n_pass_method_c": {
            key: mean_std(
                [item["source_n_pass_method_c"][key] for item in seed_reports]
            )
            for key in ("zero", "one", "at_least_two")
        },
        "methods": {},
    }
    for method in METHODS:
        method_result = {}
        for metric in (
            "generated_candidate_coverage",
            "decision_coverage_including_direct_goal",
            "fallback_rate",
            "direct_goal_eligible_rate",
            "direct_goal_selected_rate",
        ):
            method_result[metric] = mean_std(
                [item["methods"][method][metric] for item in seed_reports]
            )
        for metric in (
            "selected_r_g_source_to_subgoal",
            "selected_r_g_subgoal_to_goal",
            "selected_d_psi_progress",
            "selected_residual_norm",
            "selected_mean_5nn_manifold_distance",
        ):
            method_result[metric] = {
                statistic: mean_std(
                    [
                        item["methods"][method][metric][statistic]
                        for item in seed_reports
                    ]
                )
                for statistic in ("mean", "median", "p95")
            }
        result["methods"][method] = method_result
    return result


def aggregate_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(audits),
        "any_obvious_directional_collapse": any(
            item["obvious_directional_collapse"] for item in audits
        ),
        "global_oriented_spearman": mean_std(
            [item["global_oriented_spearman"] for item in audits]
        ),
        "same_source_pairwise_ordering": mean_std(
            [item["same_source_pairwise_ordering"] for item in audits]
        ),
        "same_goal_pairwise_ordering": mean_std(
            [item["same_goal_pairwise_ordering"] for item in audits]
        ),
        "tail_d15_d20_linear_slope": mean_std(
            [item["tail_d15_d20_linear_slope"] for item in audits]
        ),
        "bucket_means_d1_d20": {
            str(distance): mean_std(
                [item["bucket_means_d1_d20"][str(distance)] for item in audits]
            )
            for distance in range(1, 21)
        },
    }


def stage5b_decision(success_deltas: dict[str, dict[str, float]]) -> str:
    delta_filter = success_deltas["Delta_filter_SR_C_minus_SR_B"]
    delta_rank = success_deltas["Delta_rank_SR_B_minus_SR_A"]
    delta_full = success_deltas["Delta_full_SR_C_minus_SR_A"]
    if any(
        interval["lower_95"] <= 0.0 <= interval["upper_95"]
        for interval in (delta_filter, delta_rank, delta_full)
    ):
        return "inconclusive"
    if delta_filter["lower_95"] > 0.0 and delta_full["lower_95"] > 0.0:
        return "unified_R_G_filtering_and_ranking_supported"
    if delta_rank["upper_95"] < 0.0:
        return "retain_D_psi_route"
    if delta_filter["upper_95"] < 0.0 and delta_rank["upper_95"] >= 0.0:
        return "retain_R_G_ranking_remove_high_level_hard_gate"
    return "inconclusive"


def paired_closed_loop_analysis(
    seed_results: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(seed_results) != 3:
        raise ValueError("formal Stage 5B bootstrap requires three paired seeds")
    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "task_success": lambda item: float(item["success"]),
        "horizon_censored_completion_env_steps": lambda item: float(
            item["horizon_censored_completion_env_steps"]
        ),
        "fallback_rate": lambda item: float(item["fallback_rate"]),
        "candidate_coverage": lambda item: float(item["candidate_coverage"]),
        "direct_goal_eligible_rate": lambda item: float(
            item["direct_goal_eligible_rate"]
        ),
        "direct_goal_selected_rate": lambda item: float(
            item["direct_goal_selected_rate"]
        ),
        "r_g_realized_progress": lambda item: float(
            item["total_r_g_goal_progress"]
        ),
        "d_psi_realized_progress": lambda item: float(
            item["total_d_psi_realized_progress"]
        ),
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
    for method in METHODS:
        if len(by_method_episode[method]) != EXPECTED_VALIDATION_EPISODES:
            raise ValueError(f"method {method} does not cover all 411 episodes")
        if any(len(records) != 3 for records in by_method_episode[method].values()):
            raise ValueError("each episode/method must contain exactly three seeds")

    rng = np.random.default_rng(seed)
    per_episode: dict[str, dict[str, dict[int, float]]] = {
        method: {} for method in METHODS
    }
    intervals: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    for method in METHODS:
        for metric, extract in extractors.items():
            values = {
                episode: float(np.mean([extract(item) for item in records]))
                for episode, records in by_method_episode[method].items()
            }
            per_episode[method][metric] = values
            intervals[method][metric] = bootstrap_mean_95_ci(
                np.asarray([values[index] for index in sorted(values)]),
                samples=samples,
                rng=rng,
            )

    comparisons = {
        "Delta_filter": (METHOD_C, METHOD_B),
        "Delta_rank": (METHOD_B, METHOD_A),
        "Delta_full": (METHOD_C, METHOD_A),
    }
    paired_differences: dict[str, dict[str, Any]] = {}
    for label, (left, right) in comparisons.items():
        paired_differences[label] = {}
        for metric in extractors:
            left_values = per_episode[left][metric]
            right_values = per_episode[right][metric]
            common = sorted(set(left_values) & set(right_values))
            if len(common) != EXPECTED_VALIDATION_EPISODES:
                raise ValueError("paired comparison lost validation episodes")
            difference = np.asarray(
                [left_values[index] - right_values[index] for index in common]
            )
            interval = bootstrap_mean_95_ci(
                difference, samples=samples, rng=rng
            )
            interval["difference"] = f"{left} minus {right}"
            interval["paired_episode_count"] = len(common)
            paired_differences[label][metric] = interval
    canonical_success_deltas = {
        "Delta_filter_SR_C_minus_SR_B": paired_differences["Delta_filter"][
            "task_success"
        ],
        "Delta_rank_SR_B_minus_SR_A": paired_differences["Delta_rank"][
            "task_success"
        ],
        "Delta_full_SR_C_minus_SR_A": paired_differences["Delta_full"][
            "task_success"
        ],
    }
    return {
        "seed_aggregation": "average three paired seeds within each episode",
        "bootstrap_unit": "paired validation episode",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "method_episode_bootstrap_95_ci": intervals,
        "paired_differences_95_ci": paired_differences,
        "primary_success_rate_deltas": canonical_success_deltas,
        "decision": stage5b_decision(canonical_success_deltas),
        "method_summaries_all_seed_episode_rollouts": {
            method: summarize_rollouts(
                [item for item in all_records if item["method"] == method]
            )
            for method in METHODS
        },
    }


def validate_fixed_runtime(args: argparse.Namespace) -> None:
    positive = (
        args.offline_batch_size,
        args.audit_batch_size,
        args.knn_query_chunk,
        args.knn_bank_chunk,
    )
    if min(positive) < 1:
        raise ValueError("batch and kNN chunk sizes must be positive")
    fixed = {
        "num_samples": (args.num_samples, DEFAULT_NUM_SAMPLES),
        "cem_iterations": (args.cem_iterations, DEFAULT_CEM_ITERATIONS),
        "topk": (args.topk, DEFAULT_TOPK),
        "flat_planning_horizon_model_steps": (
            args.flat_planning_horizon_model_steps,
            DEFAULT_FLAT_HORIZON,
        ),
        "bootstrap_samples": (args.bootstrap_samples, 10000),
    }
    for name, (actual, expected) in fixed.items():
        if actual != expected:
            raise ValueError(f"Stage 5B fixes {name}={expected}")
    if not np.isclose(
        args.reachability_cost_weight,
        DEFAULT_REACHABILITY_COST_WEIGHT,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("Stage 5B fixes low-level reachability-cost weight=0.85")
    if args.device.startswith("cuda") and not args.validate_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")


def build_protocol(
    args: argparse.Namespace,
    *,
    seeds: tuple[int, ...],
    eta_by_seed: dict[int, float],
    r_g_paths: dict[int, Path],
    generator_paths: dict[int, Path],
    bank_size: int,
    bank_sha256: str,
    episode_indices: list[int],
) -> dict[str, Any]:
    return {
        "stage": "5B",
        "experiment": "frozen HECRL-style R_G selector comparison",
        "modules_trained_or_modified": False,
        "modules_frozen": [
            "RC-aux Encoder",
            "RC-aux world model",
            "R_local(z,g,h)",
            "RC-LeWM",
            "Stage 4 generator",
            "D_psi",
            "three Stage 5A R_G heads",
        ],
        "stage5a_report": str(args.stage5a_report.expanduser().resolve()),
        "stage4c_report": str(args.stage4c_report.expanduser().resolve()),
        "training_report": str(args.training_report.expanduser().resolve()),
        "latent_cache": str(args.latent_cache.expanduser().resolve()),
        "progress_ranker": str(args.progress_ranker.expanduser().resolve()),
        "policy": args.policy,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "dataset": args.dataset,
        "generator_r_g_seed_pairs": [[seed, seed] for seed in seeds],
        "generator_checkpoints": {
            str(seed): str(generator_paths[seed].expanduser().resolve())
            for seed in seeds
        },
        "r_g_checkpoints": {
            str(seed): str(r_g_paths[seed].expanduser().resolve()) for seed in seeds
        },
        "eta_3_by_seed": {str(seed): eta_by_seed[seed] for seed in seeds},
        "eta_3_averaged": False,
        "eta_3_recalibrated": False,
        "gamma": GAMMA,
        "gamma_changed": False,
        "long_range_audit_episode_range": [4500, 5000],
        "long_range_audit_not_used_for_early_stopping": True,
        "source_episode_range": [4000, 5000],
        "source_successful_only": True,
        "source_episode_count": EXPECTED_VALIDATION_EPISODES,
        "episode_indices": episode_indices,
        "test_split_5000_10000_used": False,
        "num_candidates": NUM_CANDIDATES,
        "offline_candidates": "exact Stage 4C artifacts for the paired seed",
        "closed_loop_dropout_seed_rule": (
            "dropout_seed + generator_seed * 10000 + episode_index + "
            "high_level_attempt_index; reset immediately before generation"
        ),
        "dropout_seed": args.dropout_seed,
        "projection": "exact Euclidean 1-NN actual bank row; no averaging",
        "projection_space": "raw unnormalized RC-aux 192D latent",
        "projection_bank_episode_range": [0, 4000],
        "projection_bank_successful_only": True,
        "projection_bank_size": bank_size,
        "projection_bank_sha256": bank_sha256,
        "manifold_diagnostic": "raw Euclidean mean distance to 5 nearest bank rows",
        "methods": {
            METHOD_A: (
                "positive D_psi progress then minimum D_psi(g,z_G)"
            ),
            METHOD_B: "argmax R_G(g,z_G), without current-to-candidate gate",
            METHOD_C: (
                "R_G(z_t,g)>=seed eta_3 then argmax R_G(g,z_G)"
            ),
        },
        "direct_goal_rule_all_methods": "select z_G first if R_G(z_t,z_G)>=eta_3(seed)",
        "fallback": "flat z_G plan for one model step then retry high level",
        "episode_horizon_env_steps": TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
        "failed_completion_steps_censored_at": TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
        "subgoal_fixed_within_segment": True,
        "subgoal_horizon_sequence": [3, 2, 1],
        "execution_horizon_model_steps": 1,
        "model_step_env_steps": MODEL_STEP_ENV_STEPS,
        "history_max_model_steps": 3,
        "flat_fallback_horizon_model_steps": DEFAULT_FLAT_HORIZON,
        "num_samples": args.num_samples,
        "cem_iterations": args.cem_iterations,
        "topk": args.topk,
        "reachability_cost_weight": args.reachability_cost_weight,
        "cem_seed": args.cem_seed,
        "cem_seed_rule": "cem_seed + validation episode position, matched by method",
        "env_seed": args.env_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "episode after averaging three paired seeds",
        "selector_or_protocol_tuned_from_result": False,
    }


def partial_report(
    protocol: dict[str, Any],
    audits: list[dict[str, Any]],
    offline: list[dict[str, Any]],
    closed: list[dict[str, Any]],
    *,
    stopped_after_audit: bool = False,
) -> dict[str, Any]:
    return {
        "stage5b_complete": False,
        "stopped_after_long_range_audit": stopped_after_audit,
        "protocol": protocol,
        "long_range_audit_by_seed": audits,
        "long_range_audit_across_seeds": (
            aggregate_audits(audits) if audits else None
        ),
        "offline_seed_results": offline,
        "offline_across_seeds": (
            aggregate_offline(offline) if len(offline) == 3 else None
        ),
        "closed_loop_seed_results": closed,
        "completed_rollouts": sum(len(item["rollouts"]) for item in closed),
        "expected_rollouts": (
            len(protocol["generator_r_g_seed_pairs"])
            * len(protocol["episode_indices"])
            * len(METHODS)
        ),
    }


def main() -> int:
    args = parse_args()
    validate_fixed_runtime(args)
    seeds = parse_seeds(args.seeds)
    cache = load_latent_cache(args.latent_cache)
    train_episodes, validation_episodes = train_validation_episodes_only(cache)
    if len(validation_episodes) != EXPECTED_VALIDATION_EPISODES:
        raise ValueError(
            f"expected 411 successful validation episodes, got {len(validation_episodes)}"
        )
    heldout_episodes = heldout_audit_episodes(validation_episodes)
    train_bank_cpu = torch.cat(
        [episode["latents"].to(torch.float32) for episode in train_episodes]
    )
    bank_sha256 = tensor_sha256(train_bank_cpu)
    episode_indices = [
        int(episode["episode_index"]) for episode in validation_episodes
    ]

    stage5a_report = read_json(args.stage5a_report)
    r_g_artifacts = validate_stage5a_artifacts(
        stage5a_report, args.stage5a_checkpoint_dir, seeds
    )
    stage2 = load_stage2_protocol(args.stage2_report)
    stage3 = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(stage2, stage3, checkpoint_label="Stage 3 D_psi")
    generator_paths_by_seed = {}
    for path in checkpoint_paths(args.training_report):
        checkpoint = torch.load(
            path.expanduser().resolve(), map_location="cpu", weights_only=False
        )
        generator_paths_by_seed[int(checkpoint["seed"])] = path
    if any(seed not in generator_paths_by_seed for seed in seeds):
        raise ValueError("generator report is missing a paired seed")
    for seed in seeds:
        stage4 = load_checkpoint_protocol(generator_paths_by_seed[seed], stage=4)
        assert_protocol_consistency(
            stage2, stage4, checkpoint_label=f"Stage 4 generator seed {seed}"
        )

    stage4c_report = read_json(args.stage4c_report)
    validate_stage4c_report(
        stage4c_report,
        seeds=seeds,
        eta_r=stage2["eta_r"],
        bank_sha256=bank_sha256,
        expected_episode_indices=set(episode_indices),
    )
    protocol = build_protocol(
        args,
        seeds=seeds,
        eta_by_seed={seed: r_g_artifacts[seed]["eta_3"] for seed in seeds},
        r_g_paths={seed: r_g_artifacts[seed]["path"] for seed in seeds},
        generator_paths=generator_paths_by_seed,
        bank_size=len(train_bank_cpu),
        bank_sha256=bank_sha256,
        episode_indices=episode_indices,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validation_successful_episodes": len(validation_episodes),
                    "heldout_audit_successful_episodes": len(heldout_episodes),
                    "train_bank_latents": len(train_bank_cpu),
                    "seed_pairs": [[seed, seed] for seed in seeds],
                    "eta_3_by_seed": protocol["eta_3_by_seed"],
                    "test_split_used": False,
                },
                indent=2,
            )
        )
        return 0

    output_path = args.output.expanduser().resolve()
    existing: dict[str, Any] = {}
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("protocol") != protocol:
            raise ValueError("existing Stage 5B output uses a different protocol")
        if existing.get("stage5b_complete") is True:
            print(f"report already complete: {output_path}")
            return 0

    device = torch.device(args.device)
    d_psi, _ = load_progress_ranker(args.progress_ranker, device=device)
    d_psi.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in d_psi.parameters()):
        raise RuntimeError("D_psi must remain frozen")

    audit_by_seed = {
        int(item["r_g_seed"]): item
        for item in existing.get("long_range_audit_by_seed", [])
    }
    r_g_models = {}
    audits = []
    for seed in seeds:
        model, checkpoint = load_global_hitting_potential(
            r_g_artifacts[seed]["path"], device=device
        )
        model.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("R_G must remain frozen")
        if int(checkpoint["seed"]) != seed:
            raise ValueError("R_G checkpoint seed mismatch")
        r_g_models[seed] = model
        if seed not in audit_by_seed:
            audit = long_range_audit(
                model,
                heldout_episodes,
                batch_size=args.audit_batch_size,
                device=device,
            )
            audit["r_g_seed"] = seed
            audit["checkpoint"] = str(r_g_artifacts[seed]["path"])
            audit_by_seed[seed] = audit
        audits.append(audit_by_seed[seed])

    audit_summary = aggregate_audits(audits)
    if audit_summary["any_obvious_directional_collapse"]:
        atomic_write_json(
            partial_report(
                protocol, audits, [], [], stopped_after_audit=True
            ),
            args.output,
        )
        print(f"Stage 5B stopped after directional audit: {output_path}")
        return 0
    if args.audit_only:
        atomic_write_json(partial_report(protocol, audits, [], []), args.output)
        print(f"audit report_path: {output_path}")
        return 0

    bank = train_bank_cpu.to(device)
    projector = ExactRealLatentProjector(
        bank,
        query_chunk=args.knn_query_chunk,
        bank_chunk=args.knn_bank_chunk,
    )
    offline_by_seed = {
        int(item["generator_seed"]): item
        for item in existing.get("offline_seed_results", [])
    }
    offline_results = []
    for seed in seeds:
        if seed not in offline_by_seed:
            artifact = torch.load(
                artifact_path_from_report(stage4c_report, seed),
                map_location="cpu",
                weights_only=False,
            )
            offline_by_seed[seed] = evaluate_offline_seed(
                artifact,
                projector,
                r_g_models[seed],
                d_psi,
                eta_3=r_g_artifacts[seed]["eta_3"],
                batch_size=args.offline_batch_size,
                query_chunk=args.knn_query_chunk,
                bank_chunk=args.knn_bank_chunk,
                device=device,
            )
        offline_results.append(offline_by_seed[seed])
        atomic_write_json(
            partial_report(protocol, audits, offline_results, []), args.output
        )
    if args.offline_only:
        print(f"offline report_path: {output_path}")
        return 0

    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=DEFAULT_FLAT_HORIZON,
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
    adapter.model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("Encoder, world model, and R_local must remain frozen")

    dataset_path = args.cache_dir.expanduser().resolve() / args.dataset
    with h5py.File(dataset_path, "r") as handle:
        environment_records = [
            load_environment_record(handle, episode)
            for episode in validation_episodes
        ]
    closed_by_seed = {
        int(item["generator_seed"]): item
        for item in existing.get("closed_loop_seed_results", [])
    }
    closed_results = []
    for seed in seeds:
        generator, generator_checkpoint = load_generator_checkpoint(
            generator_paths_by_seed[seed], device=device
        )
        generator.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in generator.parameters()):
            raise RuntimeError("generator must remain frozen")
        if int(generator_checkpoint["seed"]) != seed:
            raise ValueError("generator checkpoint seed mismatch")
        seed_result = closed_by_seed.get(
            seed,
            {
                "generator_seed": seed,
                "r_g_seed": seed,
                "eta_3": r_g_artifacts[seed]["eta_3"],
                "generator_checkpoint": str(
                    generator_paths_by_seed[seed].expanduser().resolve()
                ),
                "r_g_checkpoint": str(r_g_artifacts[seed]["path"]),
                "rollouts": [],
            },
        )
        if not np.isclose(
            seed_result["eta_3"],
            r_g_artifacts[seed]["eta_3"],
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("saved rollout seed uses a different eta_3")
        closed_results.append(seed_result)
        completed = {
            (int(item["episode_index"]), item["method"])
            for item in seed_result["rollouts"]
        }
        if len(completed) != len(seed_result["rollouts"]):
            raise ValueError(f"duplicate saved rollout for seed {seed}")
        for episode_position, record in enumerate(environment_records):
            for method in METHODS:
                key = (int(record["episode_index"]), method)
                if key in completed:
                    continue
                result = run_stage5b_rollout(
                    method,
                    generator,
                    r_g_models[seed],
                    d_psi,
                    adapter,
                    record,
                    eta_3=r_g_artifacts[seed]["eta_3"],
                    episode_horizon_env_steps=TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
                    flat_horizon=DEFAULT_FLAT_HORIZON,
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
                )
                result["generator_seed"] = seed
                result["r_g_seed"] = seed
                result["eta_3"] = r_g_artifacts[seed]["eta_3"]
                result["candidate_projection"] = "exact_train_bank_1nn"
                seed_result["rollouts"].append(result)
                completed.add(key)
                atomic_write_json(
                    partial_report(
                        protocol,
                        audits,
                        offline_results,
                        closed_results,
                    ),
                    args.output,
                )
                print(
                    f"seed={seed} episode={record['episode_index']} "
                    f"method={method} success={result['success']} "
                    f"steps={result['env_steps']}",
                    flush=True,
                )
        seed_result["method_summaries"] = {
            method: summarize_rollouts(
                [item for item in seed_result["rollouts"] if item["method"] == method]
            )
            for method in METHODS
        }
        atomic_write_json(
            partial_report(protocol, audits, offline_results, closed_results),
            args.output,
        )

    expected = len(validation_episodes) * len(METHODS)
    if any(len(item["rollouts"]) != expected for item in closed_results):
        raise RuntimeError("Stage 5B closed-loop comparison is incomplete")
    if seeds != FORMAL_SEEDS:
        print(f"pilot report_path: {output_path}")
        return 0

    paired = paired_closed_loop_analysis(
        closed_results,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report = {
        "stage5b_complete": True,
        "protocol": protocol,
        "long_range_audit_by_seed": audits,
        "long_range_audit_across_seeds": audit_summary,
        "offline_seed_results": offline_results,
        "offline_across_seeds": aggregate_offline(offline_results),
        "closed_loop_seed_results": closed_results,
        "paired_closed_loop_analysis": paired,
        "decision_rule": {
            "unified_supported": (
                "Delta_filter and Delta_full success-rate 95% CI lower bounds > 0"
            ),
            "rank_only": (
                "Delta_filter upper bound < 0 and Delta_rank is not "
                "significantly below zero"
            ),
            "retain_d_psi": "Delta_rank success-rate 95% CI upper bound < 0",
            "otherwise": "inconclusive",
            "strict_inequalities": True,
        },
        "interpretation_constraints": [
            "No model is trained or modified in Stage 5B.",
            "Every R_G uses its own frozen Stage 5A eta_3.",
            "No eta_3, gamma, projection, candidate count, or selector is tuned.",
            "The [5000,10000) test split is not accessed.",
            "R_G remains a discounted witnessed hitting-time potential, not absolute global reachability.",
        ],
    }
    atomic_write_json(report, args.output)
    print(f"report_path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
