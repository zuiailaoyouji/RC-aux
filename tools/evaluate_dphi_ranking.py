#!/usr/bin/env python3
"""Stage 3: validate D_phi ranking independently inside feasible sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401 - registers TwoRoom
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, RCAuxPlannerConfig, TWOROOM_PROFILE
from tools.evaluate_rc_filter_only import (
    WITHIN_BUDGET,
    bootstrap_mean_ci,
    execute_candidate,
)


HORIZONS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate D_phi only as a ranker inside Oracle-feasible and "
            "RC-passed candidate sets from a completed Stage 2 report."
        )
    )
    parser.add_argument(
        "--stage2-report",
        type=Path,
        default=Path("outputs/rc_filter_only_tau3.json"),
    )
    parser.add_argument("--policy", default="tworoom_rcaux/rcaux_tworoom")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/sxw/work/datasets/stable-wm"),
    )
    parser.add_argument("--dataset", default="tworoom.h5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feasible-success-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-ranking-trials", type=int, default=30)
    parser.add_argument("--planner-repeats", type=int)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--reachability-cost-weight", type=float, default=0.85)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--planner-seed", type=int, default=4200)
    parser.add_argument("--success-radius", type=float, default=16.0)
    parser.add_argument("--max-replay-pixel-diff", type=int, default=1)
    parser.add_argument("--door-grid-samples", type=int, default=513)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dphi_ranking_tau3.json"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0.0 <= args.feasible_success_rate <= 1.0:
        raise ValueError("feasible-success-rate must be in [0, 1]")
    if args.min_ranking_trials <= 0:
        raise ValueError("min-ranking-trials must be positive")
    if args.planner_repeats is not None and args.planner_repeats <= 0:
        raise ValueError("planner-repeats must be positive")
    if args.door_grid_samples < 3:
        raise ValueError("door-grid-samples must be at least 3")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if not 0 < args.topk <= args.num_samples:
        raise ValueError("topk must be in [1, num-samples]")
    if not np.isclose(args.success_radius, 16.0):
        raise ValueError("TwoRoom has a fixed environment success radius of 16")


class TwoRoomObstacleCost:
    """Obstacle-aware continuous path cost using TwoRoom collision geometry."""

    def __init__(self, *, seed: int, success_radius: float, door_samples: int):
        self.env = gym.make(
            "swm/TwoRoom-v1",
            render_mode="rgb_array",
            disable_env_checker=True,
        )
        self.env.reset(seed=seed)
        self.base = self.env.unwrapped
        self.success_radius = float(success_radius)
        self.door_samples = int(door_samples)

    def close(self) -> None:
        self.env.close()

    def goal_image(self, goal_state: np.ndarray) -> np.ndarray:
        self.base._set_goal_state(goal_state)
        return (
            self.base._render_frame(agent_pos=torch.as_tensor(goal_state))
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

    def _side(self, state: np.ndarray) -> bool:
        axis = 0 if self.base.wall_axis == 1 else 1
        return bool(state[axis] < self.base.WALL_CENTER)

    def path_length(self, state: np.ndarray, goal: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        direct = float(np.linalg.norm(state - goal))
        if direct <= self.success_radius:
            return 0.0
        if self._side(state) == self._side(goal):
            return max(0.0, direct - self.success_radius)

        center = float(self.base.WALL_CENTER)
        half = self.base.wall_thickness // 2
        agent_radius = float(
            self.base.variation_space["agent"]["radius"].value.item()
        )
        near_boundary = center - half - agent_radius
        far_boundary = center + half + agent_radius
        state_on_lower_side = self._side(state)
        if not state_on_lower_side:
            near_boundary, far_boundary = far_boundary, near_boundary

        door_margin = 1.75
        best = float("inf")
        for index in range(self.base.num_doors):
            door_center = float(self.base.door_positions[index])
            door_size = float(self.base.door_sizes[index])
            if door_size < 1.1 * agent_radius:
                continue
            coordinates = np.linspace(
                door_center - door_size - door_margin,
                door_center + door_size + door_margin,
                self.door_samples,
            )
            if self.base.wall_axis == 1:
                near = np.stack(
                    [np.full_like(coordinates, near_boundary), coordinates],
                    axis=-1,
                )
                far = np.stack(
                    [np.full_like(coordinates, far_boundary), coordinates],
                    axis=-1,
                )
            else:
                near = np.stack(
                    [coordinates, np.full_like(coordinates, near_boundary)],
                    axis=-1,
                )
                far = np.stack(
                    [coordinates, np.full_like(coordinates, far_boundary)],
                    axis=-1,
                )
            first = np.linalg.norm(near - state[None], axis=-1)
            passage = np.linalg.norm(far - near, axis=-1)
            final = np.maximum(
                0.0,
                np.linalg.norm(goal[None] - far, axis=-1)
                - self.success_radius,
            )
            best = min(best, float(np.min(first + passage + final)))
        return best

    def env_step_cost(self, state: np.ndarray, goal: np.ndarray) -> float:
        speed = float(
            self.base.variation_space["agent"]["speed"].value.item()
        )
        return self.path_length(state, goal) / speed


def reachability_time(
    adapter: RCAuxAdapter,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.stack(
        [
            adapter.reachability(
                source,
                target,
                horizon_model_steps=horizon,
            )
            for horizon in HORIZONS
        ],
        dim=-1,
    )
    return probabilities, (1.0 - probabilities).sum(dim=-1)


def pairwise_accuracy(candidates: list[dict[str, Any]]) -> float | None:
    outcomes = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index]
            right = candidates[right_index]
            true_delta = (
                left["target_obstacle_cost_env_steps"]
                - right["target_obstacle_cost_env_steps"]
            )
            if np.isclose(true_delta, 0.0):
                continue
            predicted_delta = left["dphi"] - right["dphi"]
            if np.isclose(predicted_delta, 0.0):
                outcomes.append(0.5)
            else:
                outcomes.append(float(predicted_delta * true_delta > 0.0))
    return float(np.mean(outcomes)) if outcomes else None


def mean_spearman(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    predicted = [candidate["dphi"] for candidate in candidates]
    target = [
        candidate["target_obstacle_cost_env_steps"]
        for candidate in candidates
    ]
    if np.allclose(predicted, predicted[0]) or np.allclose(target, target[0]):
        return None
    return float(spearmanr(predicted, target).statistic)


def selected_record(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "category": candidate["category"],
        "dphi": candidate["dphi"],
        "latent_l2": candidate["latent_l2"],
        "target_obstacle_cost_env_steps": candidate[
            "target_obstacle_cost_env_steps"
        ],
        "stage2_success_rate": candidate["stage2_success_rate"],
        "stage3_success_rate": candidate["stage3_success_rate"],
        "mean_executed_obstacle_progress_env_steps": candidate[
            "mean_executed_obstacle_progress_env_steps"
        ],
        "trajectory_remaining_env_steps": candidate[
            "trajectory_remaining_env_steps"
        ],
    }


def evaluate_candidate_set(
    candidates: list[dict[str, Any]],
    *,
    source_cost: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    dphi = min(candidates, key=lambda item: (item["dphi"], item["candidate_id"]))
    latent_l2 = min(
        candidates,
        key=lambda item: (item["latent_l2"], item["candidate_id"]),
    )
    oracle = min(
        candidates,
        key=lambda item: (
            item["target_obstacle_cost_env_steps"],
            item["candidate_id"],
        ),
    )
    random_target_cost = float(
        np.mean(
            [item["target_obstacle_cost_env_steps"] for item in candidates]
        )
    )
    random_executed_progress = float(
        np.mean(
            [
                item["mean_executed_obstacle_progress_env_steps"]
                for item in candidates
            ]
        )
    )
    random_success = float(
        np.mean([item["stage3_success_rate"] for item in candidates])
    )
    trajectory_values = [
        item["trajectory_remaining_env_steps"]
        for item in candidates
        if item["trajectory_remaining_env_steps"] is not None
    ]
    return {
        "candidate_count": len(candidates),
        "ranking_eligible": len(candidates) >= 2,
        "pairwise_accuracy": pairwise_accuracy(candidates),
        "spearman": mean_spearman(candidates),
        "random_uniform": {
            "target_obstacle_cost_env_steps": random_target_cost,
            "target_obstacle_progress_env_steps": source_cost
            - random_target_cost,
            "executed_obstacle_progress_env_steps": random_executed_progress,
            "success_rate": random_success,
            "trajectory_remaining_env_steps": (
                float(np.mean(trajectory_values))
                if trajectory_values
                else None
            ),
        },
        "dphi": selected_record(dphi),
        "latent_l2": selected_record(latent_l2),
        "oracle_obstacle_cost": selected_record(oracle),
        "dphi_top1_matches_oracle": (
            dphi["candidate_id"] == oracle["candidate_id"]
        ),
        "dphi_target_cost_regret_env_steps": (
            dphi["target_obstacle_cost_env_steps"]
            - oracle["target_obstacle_cost_env_steps"]
        ),
    }


def summarize_candidate_set(
    trials: list[dict[str, Any]],
    *,
    set_key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    evaluations = [trial[set_key] for trial in trials if trial[set_key] is not None]
    ranking = [item for item in evaluations if item["ranking_eligible"]]

    def values(function):
        return [float(function(item)) for item in ranking]

    random_minus_dphi_cost = values(
        lambda item: item["random_uniform"]["target_obstacle_cost_env_steps"]
        - item["dphi"]["target_obstacle_cost_env_steps"]
    )
    l2_minus_dphi_cost = values(
        lambda item: item["latent_l2"]["target_obstacle_cost_env_steps"]
        - item["dphi"]["target_obstacle_cost_env_steps"]
    )
    dphi_minus_random_execution = values(
        lambda item: item["dphi"][
            "mean_executed_obstacle_progress_env_steps"
        ]
        - item["random_uniform"]["executed_obstacle_progress_env_steps"]
    )
    dphi_minus_l2_execution = values(
        lambda item: item["dphi"][
            "mean_executed_obstacle_progress_env_steps"
        ]
        - item["latent_l2"][
            "mean_executed_obstacle_progress_env_steps"
        ]
    )
    pairwise_advantage = [
        item["pairwise_accuracy"] - 0.5
        for item in ranking
        if item["pairwise_accuracy"] is not None
    ]
    spearman_values = [
        item["spearman"]
        for item in ranking
        if item["spearman"] is not None
    ]

    selector_means = {}
    for selector in ("random_uniform", "dphi", "latent_l2", "oracle_obstacle_cost"):
        if selector == "random_uniform":
            target_cost = [
                item[selector]["target_obstacle_cost_env_steps"]
                for item in evaluations
            ]
            executed_progress = [
                item[selector]["executed_obstacle_progress_env_steps"]
                for item in evaluations
            ]
            success = [item[selector]["success_rate"] for item in evaluations]
        else:
            target_cost = [
                item[selector]["target_obstacle_cost_env_steps"]
                for item in evaluations
            ]
            executed_progress = [
                item[selector]["mean_executed_obstacle_progress_env_steps"]
                for item in evaluations
            ]
            success = [item[selector]["stage3_success_rate"] for item in evaluations]
        selector_means[selector] = {
            "mean_target_obstacle_cost_env_steps": float(np.mean(target_cost)),
            "mean_executed_obstacle_progress_env_steps": float(
                np.mean(executed_progress)
            ),
            "mean_success_rate": float(np.mean(success)),
        }

    return {
        "covered_trial_count": len(evaluations),
        "ranking_trial_count": len(ranking),
        "mean_pairwise_accuracy": (
            float(np.mean([value + 0.5 for value in pairwise_advantage]))
            if pairwise_advantage
            else None
        ),
        "mean_within_trial_spearman": (
            float(np.mean(spearman_values)) if spearman_values else None
        ),
        "dphi_top1_oracle_match_rate": (
            float(np.mean([item["dphi_top1_matches_oracle"] for item in ranking]))
            if ranking
            else None
        ),
        "mean_dphi_target_cost_regret_env_steps": (
            float(
                np.mean(
                    [item["dphi_target_cost_regret_env_steps"] for item in ranking]
                )
            )
            if ranking
            else None
        ),
        "selector_means": selector_means,
        "paired_cis": {
            "random_minus_dphi_target_cost": bootstrap_mean_ci(
                random_minus_dphi_cost,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "latent_l2_minus_dphi_target_cost": bootstrap_mean_ci(
                l2_minus_dphi_cost,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 1,
            ),
            "dphi_minus_random_executed_progress": bootstrap_mean_ci(
                dphi_minus_random_execution,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 2,
            ),
            "dphi_minus_latent_l2_executed_progress": bootstrap_mean_ci(
                dphi_minus_l2_execution,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 3,
            ),
            "pairwise_accuracy_minus_chance": bootstrap_mean_ci(
                pairwise_advantage,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 4,
            ),
        },
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    path = output.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    path.write_text(payload + "\n")
    print(payload)
    print(f"report_path: {path}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    stage2_path = args.stage2_report.expanduser().resolve()
    stage2 = json.loads(stage2_path.read_text())
    if not stage2.get("stage2_rc_filter_validated", False):
        raise RuntimeError("Stage 2 report has not validated the RC filter")
    if stage2["protocol"].get("uses_dphi_or_final_goal_progress"):
        raise RuntimeError("Stage 2 report is not filter-only")
    if stage2["protocol"]["tau_model_steps"] != 3:
        raise RuntimeError("Stage 3 requires the tau=3 Stage 2 protocol")

    repeats = args.planner_repeats
    if repeats is None:
        repeats = int(stage2["protocol"]["planner_repeats_per_candidate"])
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_path = cache_dir / args.dataset
    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=3,
        execution_horizon_model_steps=1,
        num_samples=args.num_samples,
        n_steps=args.cem_iterations,
        topk=args.topk,
        seed=args.planner_seed,
        warm_start=True,
    )
    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=cache_dir,
        device=args.device,
        planner_config=planner_config,
        use_reachability_cost=True,
        reachability_cost_weight=args.reachability_cost_weight,
    )
    adapter.model.interpolate_pos_encoding = True
    if adapter.max_reachability_horizon_model_steps < max(HORIZONS):
        raise RuntimeError("checkpoint does not support RC horizons 1..5")

    obstacle_cost = TwoRoomObstacleCost(
        seed=args.env_seed,
        success_radius=args.success_radius,
        door_samples=args.door_grid_samples,
    )
    trials = []
    rerun_differences = []
    try:
        with h5py.File(dataset_path, "r") as handle:
            offsets = handle["ep_offset"]
            lengths = handle["ep_len"]
            for trial_index, stage2_trial in enumerate(stage2["test_trials"]):
                source_row = int(stage2_trial["source_row"])
                source_state = np.asarray(
                    stage2_trial["source_state"], dtype=np.float32
                )
                source_image = np.asarray(handle["pixels"][source_row])
                final_goal_state = np.asarray(
                    handle["pos_target"][source_row], dtype=np.float32
                )
                goal_image = obstacle_cost.goal_image(final_goal_state)
                source_latent = adapter.encode_observation(source_image)[:, -1]
                goal_latent = adapter.encode_observation(goal_image)[:, -1]
                source_curve, source_dphi = reachability_time(
                    adapter,
                    source_latent,
                    goal_latent,
                )
                source_cost = obstacle_cost.env_step_cost(
                    source_state,
                    final_goal_state,
                )

                images = np.stack(
                    [
                        np.asarray(handle["pixels"][candidate["row"]])
                        for candidate in stage2_trial["candidates"]
                    ]
                )
                latents = adapter.encode_observation(images)[:, -1]
                goal_targets = goal_latent.expand(latents.size(0), -1)
                curves, dphi_values = reachability_time(
                    adapter,
                    latents,
                    goal_targets,
                )
                latent_l2 = (latents - goal_targets).square().sum(dim=-1)

                records = []
                for candidate_index, stage2_candidate in enumerate(
                    stage2_trial["candidates"]
                ):
                    candidate_state = np.asarray(
                        stage2_candidate["state"], dtype=np.float32
                    )
                    stage2_success = float(
                        stage2_candidate["actual_success_rate"]
                    )
                    oracle_feasible = (
                        stage2_success >= args.feasible_success_rate
                    )
                    rc_passed = bool(stage2_candidate["passed_rc"])
                    should_execute = oracle_feasible or rc_passed
                    executions = []
                    if should_execute:
                        for repeat in range(repeats):
                            execution = execute_candidate(
                                adapter,
                                source_state=source_state,
                                source_image=source_image,
                                target_state=candidate_state,
                                target_image=images[candidate_index],
                                target_latent=(
                                    latents[candidate_index : candidate_index + 1]
                                ),
                                tau_model_steps=3,
                                env_seed=args.env_seed,
                                planner_seed=(
                                    args.planner_seed
                                    + trial_index * 100
                                    + repeat
                                ),
                                success_radius=args.success_radius,
                                max_replay_pixel_diff=(
                                    args.max_replay_pixel_diff
                                ),
                            )
                            final_state = np.asarray(
                                execution["final_state"], dtype=np.float32
                            )
                            final_cost = obstacle_cost.env_step_cost(
                                final_state,
                                final_goal_state,
                            )
                            execution["final_obstacle_cost_env_steps"] = (
                                final_cost
                            )
                            execution["obstacle_progress_env_steps"] = (
                                source_cost - final_cost
                            )
                            execution["repeat_index"] = repeat
                            executions.append(execution)

                    stage3_success = (
                        float(
                            np.mean(
                                [item["actual_success"] for item in executions]
                            )
                        )
                        if executions
                        else None
                    )
                    if stage3_success is not None:
                        rerun_differences.append(
                            abs(stage3_success - stage2_success)
                        )
                    episode_index = int(stage2_candidate["episode_index"])
                    if episode_index == int(stage2_trial["episode_index"]):
                        terminal_row = (
                            int(offsets[episode_index])
                            + int(lengths[episode_index])
                            - 1
                        )
                        trajectory_remaining = terminal_row - int(
                            stage2_candidate["row"]
                        )
                    else:
                        trajectory_remaining = None
                    records.append(
                        {
                            "candidate_id": stage2_candidate["candidate_id"],
                            "category": stage2_candidate["category"],
                            "row": int(stage2_candidate["row"]),
                            "episode_index": episode_index,
                            "state": candidate_state.tolist(),
                            "rc_passed": rc_passed,
                            "oracle_feasible": oracle_feasible,
                            "local_rc_score": stage2_candidate[
                                "local_rc_score"
                            ],
                            "stage2_success_rate": stage2_success,
                            "stage3_success_rate": stage3_success,
                            "goal_reachability_curve": (
                                curves[candidate_index].detach().cpu().tolist()
                            ),
                            "dphi": float(dphi_values[candidate_index]),
                            "predicted_progress": float(
                                source_dphi.item()
                                - dphi_values[candidate_index].item()
                            ),
                            "latent_l2": float(latent_l2[candidate_index]),
                            "target_obstacle_cost_env_steps": (
                                obstacle_cost.env_step_cost(
                                    candidate_state,
                                    final_goal_state,
                                )
                            ),
                            "trajectory_remaining_env_steps": (
                                trajectory_remaining
                            ),
                            "mean_executed_obstacle_progress_env_steps": (
                                float(
                                    np.mean(
                                        [
                                            item[
                                                "obstacle_progress_env_steps"
                                            ]
                                            for item in executions
                                        ]
                                    )
                                )
                                if executions
                                else None
                            ),
                            "executions": executions,
                        }
                    )

                oracle_candidates = [
                    item for item in records if item["oracle_feasible"]
                ]
                rc_candidates = [item for item in records if item["rc_passed"]]
                trial = {
                    "trial_index": trial_index,
                    "episode_index": int(stage2_trial["episode_index"]),
                    "source_row": source_row,
                    "source_state": source_state.tolist(),
                    "final_goal_state": final_goal_state.tolist(),
                    "source_goal_reachability_curve": (
                        source_curve[0].detach().cpu().tolist()
                    ),
                    "source_dphi": float(source_dphi.item()),
                    "source_obstacle_cost_env_steps": source_cost,
                    "oracle_feasible": evaluate_candidate_set(
                        oracle_candidates,
                        source_cost=source_cost,
                    ),
                    "rc_passed": evaluate_candidate_set(
                        rc_candidates,
                        source_cost=source_cost,
                    ),
                    "candidates": records,
                }
                trials.append(trial)
                print(
                    f"trial={trial_index} episode={trial['episode_index']} "
                    f"oracle_feasible={len(oracle_candidates)} "
                    f"rc_passed={len(rc_candidates)}",
                    flush=True,
                )
    finally:
        obstacle_cost.close()

    oracle_summary = summarize_candidate_set(
        trials,
        set_key="oracle_feasible",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rc_summary = summarize_candidate_set(
        trials,
        set_key="rc_passed",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 100,
    )
    oracle_cis = oracle_summary["paired_cis"]

    def lower_positive(name: str) -> bool:
        interval = oracle_cis[name]
        return interval is not None and interval["lower_95"] > 0.0

    validation_criteria = {
        "enough_oracle_feasible_ranking_trials": bool(
            oracle_summary["ranking_trial_count"] >= args.min_ranking_trials
        ),
        "stage3_rerun_matches_stage2_success_rates": bool(
            rerun_differences and max(rerun_differences) <= 1.0e-12
        ),
        "dphi_pairwise_accuracy_ci_above_chance": lower_positive(
            "pairwise_accuracy_minus_chance"
        ),
        "dphi_target_cost_beats_random_ci": lower_positive(
            "random_minus_dphi_target_cost"
        ),
        "dphi_executed_progress_beats_random_ci": lower_positive(
            "dphi_minus_random_executed_progress"
        ),
        "dphi_target_cost_beats_latent_l2_ci": lower_positive(
            "latent_l2_minus_dphi_target_cost"
        ),
        "dphi_executed_progress_beats_latent_l2_ci": lower_positive(
            "dphi_minus_latent_l2_executed_progress"
        ),
    }
    dphi_validated = all(validation_criteria.values())
    report = {
        "stage2_report_path": str(stage2_path),
        "stage2_rc_filter_validated": True,
        "stage3_dphi_ranking_validated": dphi_validated,
        "ready_for_combined_stage4_evaluation": dphi_validated,
        "ready_to_train_high_level_generator": False,
        "validation_criteria": validation_criteria,
        "protocol": {
            "candidate_set_primary": (
                "Stage 2 actual_success_rate >= feasible_success_rate"
            ),
            "candidate_set_secondary": "Stage 2 RC-passed candidates",
            "feasible_success_rate": args.feasible_success_rate,
            "dphi_definition": "sum_h(1-R_phi(candidate, final_goal, h))",
            "horizons_model_steps": list(HORIZONS),
            "final_goal_definition": (
                "official TwoRoom target image rendered with agent at pos_target"
            ),
            "independent_target_cost": (
                "continuous obstacle-aware path through valid door regions, "
                "wall collision boundaries, agent radius, success radius, "
                "and environment speed"
            ),
            "trajectory_remaining_steps_reported": True,
            "planner_repeats_per_candidate": repeats,
            "shared_stage2_cem_seeds": True,
            "comparators": [
                "uniform_random",
                "latent_l2",
                "oracle_obstacle_cost",
            ],
        },
        "rerun_success_rate_max_abs_difference": max(rerun_differences),
        "oracle_feasible_summary": oracle_summary,
        "rc_passed_summary": rc_summary,
        "trials": trials,
    }
    write_report(report, args.output)
    return int(not dphi_validated)


if __name__ == "__main__":
    raise SystemExit(main())
