#!/usr/bin/env python3
"""Evaluate RC filtering independently of a learned subgoal generator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401 - registers TwoRoom
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, RCAuxPlannerConfig, TWOROOM_PROFILE


WITHIN_BUDGET = "within_budget"
OVER_BUDGET = "same_trajectory_over_budget"
CROSS_UNREACHABLE = "cross_trajectory_unreachable"
STRATEGIES = ("rc_then_progress", "random", "progress_only")
REACHABILITY_TIME_HORIZONS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build real-trajectory latent Oracle pools, score them with RC, "
            "and compare three selection mechanisms using real execution."
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
    parser.add_argument("--tau-model-steps", type=int, default=3)
    parser.add_argument("--eta-r", type=float, default=0.5)
    parser.add_argument(
        "--eta-r-grid",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--start-episode-index", type=int, default=0)
    parser.add_argument("--t-env-step", type=int, default=0)
    parser.add_argument("--cross-candidates", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--reachability-cost-weight", type=float, default=0.85)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--planner-seed", type=int, default=4200)
    parser.add_argument("--selection-seed", type=int, default=20260807)
    parser.add_argument("--success-radius", type=float, default=16.0)
    parser.add_argument("--max-replay-pixel-diff", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rc_oracle_filter_tau3.json"),
    )
    return parser.parse_args()


def parse_threshold_grid(value: str, eta_r: float) -> list[float]:
    thresholds = {float(item.strip()) for item in value.split(",") if item.strip()}
    thresholds.add(float(eta_r))
    if any(item < 0.0 or item > 1.0 for item in thresholds):
        raise ValueError("all eta-r-grid values must be in [0, 1]")
    return sorted(thresholds)


def frame_error(reference: np.ndarray, rendered: np.ndarray) -> dict[str, Any]:
    difference = np.abs(reference.astype(np.int16) - rendered.astype(np.int16))
    return {
        "exact_equal": bool(np.array_equal(reference, rendered)),
        "max_abs_pixel_diff": int(difference.max(initial=0)),
        "mean_abs_pixel_diff": float(difference.mean()),
    }


def environment_speed(seed: int) -> float:
    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        disable_env_checker=True,
    )
    try:
        env.reset(seed=seed)
        return float(env.unwrapped.variation_space["agent"]["speed"].value.item())
    finally:
        env.close()


def lower_bound_env_steps(
    source_state: np.ndarray,
    target_state: np.ndarray,
    *,
    speed: float,
    success_radius: float,
) -> int:
    distance = float(np.linalg.norm(source_state - target_state))
    maximum_step_displacement = speed * np.sqrt(TWOROOM_PROFILE.action_dim)
    required_displacement = max(0.0, distance - success_radius)
    return int(np.ceil(required_displacement / maximum_step_displacement))


def reachability_time_estimate(
    adapter: RCAuxAdapter,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    horizons: tuple[int, ...] = REACHABILITY_TIME_HORIZONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RC probabilities by horizon and their truncated time integral."""

    probabilities = torch.stack(
        [
            adapter.reachability(
                source,
                target,
                horizon_model_steps=horizon,
            )
            for horizon in horizons
        ],
        dim=-1,
    )
    truncated_time = (1.0 - probabilities).sum(dim=-1)
    return probabilities, truncated_time


def candidate_from_row(
    handle: h5py.File,
    *,
    row: int,
    episode_index: int,
    category: str,
    candidate_id: str,
    source_state: np.ndarray,
    speed: float,
    success_radius: float,
    temporal_delta_model_steps: int | None,
) -> dict[str, Any]:
    state = np.asarray(handle["proprio"][row], dtype=np.float32)
    distance = float(np.linalg.norm(source_state - state))
    return {
        "candidate_id": candidate_id,
        "category": category,
        "row": int(row),
        "episode_index": int(episode_index),
        "temporal_delta_model_steps": temporal_delta_model_steps,
        "image": np.asarray(handle["pixels"][row]),
        "state": state,
        "source_distance": distance,
        "lower_bound_env_steps": lower_bound_env_steps(
            source_state,
            state,
            speed=speed,
            success_radius=success_radius,
        ),
    }


def build_candidate_pool(
    handle: h5py.File,
    *,
    episode_index: int,
    t_env_step: int,
    tau_model_steps: int,
    cross_candidates: int,
    speed: float,
    success_radius: float,
) -> dict[str, Any] | None:
    episode_offsets = handle["ep_offset"]
    episode_lengths = handle["ep_len"]
    episode_start = int(episode_offsets[episode_index])
    episode_length = int(episode_lengths[episode_index])
    terminal_row = episode_start + episode_length - 1
    if not bool(handle["terminated"][terminal_row]):
        return None
    block_env_steps = TWOROOM_PROFILE.model_step_env_steps
    candidate_window_model_steps = tau_model_steps + 2
    candidate_window_env_step = (
        t_env_step + candidate_window_model_steps * block_env_steps
    )
    if candidate_window_env_step >= episode_length:
        return None

    source_row = episode_start + t_env_step
    source_state = np.asarray(handle["proprio"][source_row], dtype=np.float32)
    source_image = np.asarray(handle["pixels"][source_row])
    candidates: list[dict[str, Any]] = []

    for delta in range(1, candidate_window_model_steps + 1):
        row = source_row + delta * block_env_steps
        category = WITHIN_BUDGET if delta <= tau_model_steps else OVER_BUDGET
        candidate = candidate_from_row(
            handle,
            row=row,
            episode_index=episode_index,
            category=category,
            candidate_id=f"same_ep_delta_{delta}",
            source_state=source_state,
            speed=speed,
            success_radius=success_radius,
            temporal_delta_model_steps=delta,
        )
        if candidate["source_distance"] >= success_radius:
            candidates.append(candidate)

    categories = {candidate["category"] for candidate in candidates}
    if WITHIN_BUDGET not in categories or OVER_BUDGET not in categories:
        return None

    budget_env_steps = tau_model_steps * block_env_steps
    episode_count = len(episode_offsets)
    for shift in range(1, episode_count):
        cross_episode = (episode_index + shift) % episode_count
        cross_length = int(episode_lengths[cross_episode])
        if t_env_step >= cross_length:
            continue
        row = int(episode_offsets[cross_episode]) + t_env_step
        candidate = candidate_from_row(
            handle,
            row=row,
            episode_index=cross_episode,
            category=CROSS_UNREACHABLE,
            candidate_id=f"cross_ep_{cross_episode}",
            source_state=source_state,
            speed=speed,
            success_radius=success_radius,
            temporal_delta_model_steps=None,
        )
        if candidate["lower_bound_env_steps"] > budget_env_steps:
            candidates.append(candidate)
        if sum(c["category"] == CROSS_UNREACHABLE for c in candidates) >= cross_candidates:
            break

    if sum(c["category"] == CROSS_UNREACHABLE for c in candidates) < cross_candidates:
        return None

    return {
        "episode_index": episode_index,
        "source_row": source_row,
        "source_state": source_state,
        "source_image": source_image,
        "final_goal_row": terminal_row,
        "final_goal_state": np.asarray(
            handle["proprio"][terminal_row], dtype=np.float32
        ),
        "final_goal_image": np.asarray(handle["pixels"][terminal_row]),
        "candidates": candidates,
    }


def execute_latent_subgoal(
    adapter: RCAuxAdapter,
    *,
    source_state: np.ndarray,
    source_image: np.ndarray,
    target_state: np.ndarray,
    target_image: np.ndarray,
    target_latent: torch.Tensor,
    final_goal_state: np.ndarray,
    final_goal_latent: torch.Tensor,
    tau_model_steps: int,
    env_seed: int,
    planner_seed: int,
    success_radius: float,
    max_replay_pixel_diff: int,
) -> dict[str, Any]:
    budget_env_steps = tau_model_steps * TWOROOM_PROFILE.model_step_env_steps
    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        max_episode_steps=budget_env_steps,
        disable_env_checker=True,
    )
    positions: list[list[float]] = []
    distances: list[float] = []
    h_plan_sequence: list[int] = []
    reachability_sequence: list[float] = []
    warm_start_sequence: list[str] = []
    success = False
    truncated = False
    env_steps = 0
    final_observation = source_image
    try:
        env.reset(seed=env_seed)
        base_env = env.unwrapped
        base_env._set_state(source_state)
        base_env._set_goal_state(target_state)
        rendered_source = env.render()
        rendered_target = (
            base_env._render_frame(agent_pos=torch.as_tensor(target_state))
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        source_replay = frame_error(source_image, rendered_source)
        target_replay = frame_error(target_image, rendered_target)
        replay_valid = (
            source_replay["max_abs_pixel_diff"] <= max_replay_pixel_diff
            and target_replay["max_abs_pixel_diff"] <= max_replay_pixel_diff
        )
        if not replay_valid:
            raise RuntimeError(
                "Dataset/environment replay mismatch exceeds pixel tolerance"
            )

        initial_distance = float(np.linalg.norm(source_state - target_state))
        if initial_distance < success_radius:
            raise RuntimeError("Trivial candidate starts inside success radius")
        positions.append(source_state.tolist())
        distances.append(initial_distance)
        adapter.reset_planner(seed=planner_seed)

        for executed_model_steps in range(tau_model_steps):
            h_rem_model_steps = tau_model_steps - executed_model_steps
            current_image = env.render()
            current_latent = adapter.encode_observation(current_image)[:, -1]
            reachability = adapter.reachability(
                current_latent,
                target_latent,
                horizon_model_steps=h_rem_model_steps,
            )
            plan = adapter.plan_to_latent(
                current_image,
                target_latent,
                planning_horizon_model_steps=h_rem_model_steps,
                execution_horizon_model_steps=1,
            )
            h_plan_sequence.append(
                plan.diagnostics.planning_horizon_model_steps
            )
            reachability_sequence.append(float(reachability.item()))
            warm_start_sequence.append(plan.diagnostics.warm_start_source)

            for action in plan.actions_to_execute_env_steps[0]:
                _, _, terminated, step_truncated, info = env.step(action)
                env_steps += 1
                success = success or bool(terminated)
                truncated = truncated or bool(step_truncated)
                position = np.asarray(info["proprio"], dtype=np.float32)
                positions.append(position.tolist())
                distances.append(float(info["distance_to_target"]))
                if success or truncated:
                    break
            if success or truncated:
                break
        final_observation = env.render()
    finally:
        env.close()

    final_state = np.asarray(positions[-1], dtype=np.float32)
    final_latent = adapter.encode_observation(final_observation)[:, -1]
    initial_goal_curve, initial_goal_time = reachability_time_estimate(
        adapter,
        adapter.encode_observation(source_image)[:, -1],
        final_goal_latent,
    )
    final_goal_curve, final_goal_time = reachability_time_estimate(
        adapter,
        final_latent,
        final_goal_latent,
    )
    initial_goal_distance = float(np.linalg.norm(source_state - final_goal_state))
    final_goal_distance = float(np.linalg.norm(final_state - final_goal_state))

    return {
        "actual_success": bool(success),
        "truncated": bool(truncated),
        "env_steps_executed": env_steps,
        "initial_distance": distances[0],
        "final_distance": distances[-1],
        "minimum_distance": min(distances),
        "distance_reduction": distances[0] - distances[-1],
        "h_plan_sequence": h_plan_sequence,
        "reachability_sequence": reachability_sequence,
        "warm_start_sequence": warm_start_sequence,
        "positions": positions,
        "final_task_initial_physical_distance": initial_goal_distance,
        "final_task_final_physical_distance": final_goal_distance,
        "final_task_physical_progress": (
            initial_goal_distance - final_goal_distance
        ),
        "final_task_initial_reachability_curve": (
            initial_goal_curve[0].detach().cpu().tolist()
        ),
        "final_task_final_reachability_curve": (
            final_goal_curve[0].detach().cpu().tolist()
        ),
        "final_task_initial_truncated_time": float(initial_goal_time.item()),
        "final_task_final_truncated_time": float(final_goal_time.item()),
        "final_task_reachability_time_progress": float(
            initial_goal_time.item() - final_goal_time.item()
        ),
    }


def choose_candidates(
    candidates: list[dict[str, Any]],
    *,
    eta_r: float,
    random_index: int,
) -> dict[str, int | None]:
    remaining_time = np.asarray(
        [candidate["truncated_reachability_time"] for candidate in candidates]
    )
    predicted_progress = np.asarray(
        [candidate["predicted_progress"] for candidate in candidates]
    )
    local_rc_scores = np.asarray(
        [candidate["local_rc_score"] for candidate in candidates]
    )
    positive_progress = np.flatnonzero(predicted_progress > 0.0)
    feasible = np.flatnonzero(
        (local_rc_scores >= eta_r) & (predicted_progress > 0.0)
    )
    return {
        "rc_then_progress": (
            int(feasible[np.argmin(remaining_time[feasible])])
            if feasible.size
            else None
        ),
        "random": int(random_index),
        "progress_only": (
            int(positive_progress[np.argmin(remaining_time[positive_progress])])
            if positive_progress.size
            else None
        ),
    }


def strategy_summary(trials: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    selected = [trial["selection"][strategy] for trial in trials]
    successes = [
        bool(item["actual_success"]) if item is not None else False
        for item in selected
    ]
    categories = Counter(
        item["category"] for item in selected if item is not None
    )
    selected_count = sum(item is not None for item in selected)
    unconditional_physical_progress = [
        item["final_task_physical_progress"] if item is not None else 0.0
        for item in selected
    ]
    unconditional_rc_time_progress = [
        item["final_task_reachability_time_progress"]
        if item is not None
        else 0.0
        for item in selected
    ]
    return {
        "trial_count": len(trials),
        "selected_count": selected_count,
        "no_feasible_count": len(trials) - selected_count,
        "actual_completion_count": int(sum(successes)),
        "actual_completion_rate": float(np.mean(successes)) if successes else 0.0,
        "conditional_completion_rate": (
            float(
                np.mean(
                    [item["actual_success"] for item in selected if item is not None]
                )
            )
            if selected_count
            else 0.0
        ),
        "selected_categories": dict(categories),
        "mean_final_task_physical_progress": float(
            np.mean(unconditional_physical_progress)
        ),
        "mean_final_task_reachability_time_progress": float(
            np.mean(unconditional_rc_time_progress)
        ),
        "mean_selected_predicted_progress": (
            float(
                np.mean(
                    [
                        item["predicted_progress"]
                        for item in selected
                        if item is not None
                    ]
                )
            )
            if selected_count
            else None
        ),
    }


def paired_comparison(
    trials: list[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, int]:
    outcomes = []
    for trial in trials:
        left_item = trial["selection"][left]
        right_item = trial["selection"][right]
        left_success = bool(
            left_item is not None and left_item["actual_success"]
        )
        right_success = bool(
            right_item is not None and right_item["actual_success"]
        )
        outcomes.append((left_success, right_success))
    return {
        "left_wins": sum(
            left_success and not right_success
            for left_success, right_success in outcomes
        ),
        "right_wins": sum(
            right_success and not left_success
            for left_success, right_success in outcomes
        ),
        "ties": sum(
            left_success == right_success
            for left_success, right_success in outcomes
        ),
    }


def classification_summary(records: list[dict[str, Any]], eta_r: float) -> dict[str, Any]:
    labels = np.asarray([record["actual_success"] for record in records], dtype=bool)
    scores = np.asarray(
        [record["local_rc_score"] for record in records],
        dtype=np.float64,
    )
    predictions = scores >= eta_r
    true_positive = int(np.sum(predictions & labels))
    false_positive = int(np.sum(predictions & ~labels))
    true_negative = int(np.sum(~predictions & ~labels))
    false_negative = int(np.sum(~predictions & labels))
    both_classes = np.unique(labels).size == 2
    calibration_bins = []
    for lower in np.linspace(0.0, 0.8, 5):
        upper = lower + 0.2
        in_bin = (scores >= lower) & (
            scores <= upper if upper >= 1.0 else scores < upper
        )
        calibration_bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(in_bin.sum()),
                "actual_completion_rate": (
                    float(labels[in_bin].mean()) if in_bin.any() else None
                ),
            }
        )
    return {
        "eta_r": eta_r,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "specificity": true_negative / max(1, true_negative + false_positive),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, scores)) if labels.any() else None
        ),
        "mean_score_success": float(scores[labels].mean()) if labels.any() else None,
        "mean_score_failure": float(scores[~labels].mean()) if (~labels).any() else None,
        "calibration_bins": calibration_bins,
    }


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.tau_model_steps != 3:
        raise ValueError("this controlled experiment requires tau-model-steps=3")
    if not 0.0 <= args.eta_r <= 1.0:
        raise ValueError("eta-r must be in [0, 1]")
    if args.num_trials <= 0:
        raise ValueError("num-trials must be positive")
    if args.cross_candidates <= 0:
        raise ValueError("cross-candidates must be positive")
    if args.num_samples <= 0 or args.cem_iterations <= 0:
        raise ValueError("CEM sample and iteration counts must be positive")
    if not 0 < args.topk <= args.num_samples:
        raise ValueError("topk must be in [1, num-samples]")
    thresholds = parse_threshold_grid(args.eta_r_grid, args.eta_r)
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_path = cache_dir / args.dataset
    speed = environment_speed(args.env_seed)

    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=args.tau_model_steps,
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
    if adapter.max_reachability_horizon_model_steps < max(
        REACHABILITY_TIME_HORIZONS
    ):
        raise RuntimeError(
            "checkpoint reachability head does not support horizons 1..5"
        )

    trials: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    with h5py.File(dataset_path, "r") as handle:
        episode_count = len(handle["ep_offset"])
        for episode_index in range(args.start_episode_index, episode_count):
            if len(trials) >= args.num_trials:
                break
            pool = build_candidate_pool(
                handle,
                episode_index=episode_index,
                t_env_step=args.t_env_step,
                tau_model_steps=args.tau_model_steps,
                cross_candidates=args.cross_candidates,
                speed=speed,
                success_radius=args.success_radius,
            )
            if pool is None:
                continue

            source_latent = adapter.encode_observation(pool["source_image"])[:, -1]
            candidate_images = np.stack(
                [candidate["image"] for candidate in pool["candidates"]]
            )
            candidate_latents = adapter.encode_observation(candidate_images)[:, -1]
            final_goal_latent = adapter.encode_observation(
                pool["final_goal_image"]
            )[:, -1]
            reachability = adapter.reachability(
                source_latent,
                candidate_latents.unsqueeze(0),
                horizon_model_steps=args.tau_model_steps,
                return_diagnostics=True,
            )
            goal_targets = final_goal_latent.expand(
                candidate_latents.size(0), -1
            )
            candidate_goal_curves, candidate_goal_times = (
                reachability_time_estimate(
                    adapter,
                    candidate_latents,
                    goal_targets,
                )
            )
            source_goal_curve, source_goal_time = reachability_time_estimate(
                adapter,
                source_latent,
                final_goal_latent,
            )

            trial_index = len(trials)
            print(
                f"trial={trial_index} episode={episode_index} "
                f"candidates={len(pool['candidates'])}",
                flush=True,
            )
            for candidate_index, candidate in enumerate(pool["candidates"]):
                candidate["local_rc_score"] = float(
                    reachability.probabilities[0, candidate_index]
                )
                candidate["local_rc_logit"] = float(
                    reachability.logits[0, candidate_index]
                )
                candidate["goal_reachability_curve"] = (
                    candidate_goal_curves[candidate_index]
                    .detach()
                    .cpu()
                    .tolist()
                )
                candidate["truncated_reachability_time"] = float(
                    candidate_goal_times[candidate_index]
                )
                candidate["source_goal_reachability_curve"] = (
                    source_goal_curve[0].detach().cpu().tolist()
                )
                candidate["source_goal_truncated_reachability_time"] = float(
                    source_goal_time.item()
                )
                candidate["predicted_progress"] = float(
                    source_goal_time.item()
                    - candidate_goal_times[candidate_index].item()
                )
                candidate["passed_local_rc"] = (
                    candidate["local_rc_score"] >= args.eta_r
                )
                candidate["passed_positive_progress"] = (
                    candidate["predicted_progress"] > 0.0
                )
                candidate["passed_joint_filter"] = (
                    candidate["passed_local_rc"]
                    and candidate["passed_positive_progress"]
                )
                execution = execute_latent_subgoal(
                    adapter,
                    source_state=pool["source_state"],
                    source_image=pool["source_image"],
                    target_state=candidate["state"],
                    target_image=candidate["image"],
                    target_latent=candidate_latents[candidate_index : candidate_index + 1],
                    final_goal_state=pool["final_goal_state"],
                    final_goal_latent=final_goal_latent,
                    tau_model_steps=args.tau_model_steps,
                    env_seed=args.env_seed,
                    planner_seed=args.planner_seed + trial_index,
                    success_radius=args.success_radius,
                    max_replay_pixel_diff=args.max_replay_pixel_diff,
                )
                candidate.update(execution)
                print(
                    f"  {candidate['candidate_id']} category={candidate['category']} "
                    f"r_local={candidate['local_rc_score']:.4f} "
                    f"D_goal={candidate['truncated_reachability_time']:.4f} "
                    f"progress={candidate['predicted_progress']:.4f} "
                    f"pass={candidate['passed_joint_filter']} "
                    f"success={candidate['actual_success']} "
                    f"task_progress="
                    f"{candidate['final_task_reachability_time_progress']:.4f}",
                    flush=True,
                )

            rng = np.random.default_rng(args.selection_seed + trial_index)
            selected_indices = choose_candidates(
                pool["candidates"],
                eta_r=args.eta_r,
                random_index=int(rng.integers(len(pool["candidates"]))),
            )
            selection = {
                strategy: (
                    pool["candidates"][index] if index is not None else None
                )
                for strategy, index in selected_indices.items()
            }
            serializable_candidates = []
            for candidate in pool["candidates"]:
                record = {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"image", "state"}
                }
                record["state"] = candidate["state"].tolist()
                record["trial_index"] = trial_index
                candidate_records.append(record)
                serializable_candidates.append(record)
            trials.append(
                {
                    "trial_index": trial_index,
                    "episode_index": episode_index,
                    "source_row": pool["source_row"],
                    "source_state": pool["source_state"].tolist(),
                    "final_goal_row": pool["final_goal_row"],
                    "final_goal_state": pool["final_goal_state"].tolist(),
                    "candidates": serializable_candidates,
                    "selection": {
                        strategy: (
                            {
                                "candidate_id": item["candidate_id"],
                                "category": item["category"],
                                "local_rc_score": item["local_rc_score"],
                                "truncated_reachability_time": item[
                                    "truncated_reachability_time"
                                ],
                                "predicted_progress": item[
                                    "predicted_progress"
                                ],
                                "actual_success": item["actual_success"],
                                "final_task_physical_progress": item[
                                    "final_task_physical_progress"
                                ],
                                "final_task_reachability_time_progress": item[
                                    "final_task_reachability_time_progress"
                                ],
                            }
                            if item is not None
                            else None
                        )
                        for strategy, item in selection.items()
                    },
                }
            )

    if len(trials) != args.num_trials:
        raise RuntimeError(
            f"constructed only {len(trials)} valid trials, requested {args.num_trials}"
        )

    category_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        category_records[record["category"]].append(record)
    category_summary = {
        category: {
            "candidate_count": len(records),
            "local_rc_pass_rate": float(
                np.mean([r["passed_local_rc"] for r in records])
            ),
            "positive_progress_pass_rate": float(
                np.mean([r["passed_positive_progress"] for r in records])
            ),
            "joint_filter_pass_rate": float(
                np.mean([r["passed_joint_filter"] for r in records])
            ),
            "actual_completion_rate": float(
                np.mean([r["actual_success"] for r in records])
            ),
            "mean_local_rc_score": float(
                np.mean([r["local_rc_score"] for r in records])
            ),
            "mean_predicted_progress": float(
                np.mean([r["predicted_progress"] for r in records])
            ),
            "mean_final_task_reachability_time_progress": float(
                np.mean(
                    [
                        r["final_task_reachability_time_progress"]
                        for r in records
                    ]
                )
            ),
        }
        for category, records in category_records.items()
    }
    strategy_summaries = {
        strategy: strategy_summary(trials, strategy) for strategy in STRATEGIES
    }
    random_expected_rate = float(
        np.mean(
            [
                np.mean([candidate["actual_success"] for candidate in trial["candidates"]])
                for trial in trials
            ]
        )
    )
    sweep = {}
    for threshold in thresholds:
        sweep_trials = []
        for trial in trials:
            candidates = trial["candidates"]
            feasible = [
                candidate
                for candidate in candidates
                if candidate["local_rc_score"] >= threshold
                and candidate["predicted_progress"] > 0.0
            ]
            selected = (
                min(
                    feasible,
                    key=lambda item: item["truncated_reachability_time"],
                )
                if feasible
                else None
            )
            sweep_trials.append(selected)
        sweep[str(threshold)] = {
            "actual_completion_rate": float(
                np.mean(
                    [
                        bool(item and item["actual_success"])
                        for item in sweep_trials
                    ]
                )
            ),
            "actual_completion_count": int(
                sum(
                    bool(item and item["actual_success"])
                    for item in sweep_trials
                )
            ),
            "mean_final_task_reachability_time_progress": float(
                np.mean(
                    [
                        item["final_task_reachability_time_progress"]
                        if item is not None
                        else 0.0
                        for item in sweep_trials
                    ]
                )
            ),
        }

    rc_rate = strategy_summaries["rc_then_progress"]["actual_completion_rate"]
    random_rate = strategy_summaries["random"]["actual_completion_rate"]
    progress_rate = strategy_summaries["progress_only"]["actual_completion_rate"]
    progress_key = "mean_final_task_physical_progress"
    rc_task_progress = strategy_summaries["rc_then_progress"][progress_key]
    random_task_progress = strategy_summaries["random"][progress_key]
    unfiltered_task_progress = strategy_summaries["progress_only"][progress_key]
    correspondence = classification_summary(candidate_records, args.eta_r)
    completion_superior = bool(
        rc_rate > random_rate and rc_rate > progress_rate
    )
    final_task_progress_superior = bool(
        rc_task_progress > random_task_progress
        and rc_task_progress > unfiltered_task_progress
    )
    rc_success_correspondence_valid = bool(
        correspondence["roc_auc"] is not None
        and correspondence["roc_auc"] > 0.5
    )
    mechanism_validated = bool(
        completion_superior
        and final_task_progress_superior
        and rc_success_correspondence_valid
    )
    report = {
        "rc_filter_mechanism_validated": mechanism_validated,
        "ready_to_train_high_level_generator": mechanism_validated,
        "validation_criteria": {
            "completion_rate_strictly_beats_both_baselines": (
                completion_superior
            ),
            "final_task_physical_progress_strictly_beats_both_baselines": (
                final_task_progress_superior
            ),
            "local_rc_score_roc_auc_above_chance": (
                rc_success_correspondence_valid
            ),
        },
        "policy": args.policy,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "protocol": {
            "tau_model_steps": args.tau_model_steps,
            "budget_env_steps": (
                args.tau_model_steps * TWOROOM_PROFILE.model_step_env_steps
            ),
            "eta_r": args.eta_r,
            "eta_r_selection": "pre_fixed_before_execution",
            "reachability_time_horizons_model_steps": list(
                REACHABILITY_TIME_HORIZONS
            ),
            "reachability_time_estimator": (
                "sum_h(1 - R_phi(source, target, h)) with delta_h=1 "
                "model step"
            ),
            "predicted_progress_definition": (
                "D_phi(source, final_goal) - D_phi(candidate, final_goal)"
            ),
            "selection_filters": [
                "local_rc_score >= eta_r",
                "predicted_progress > 0",
            ],
            "final_goal_definition": (
                "terminal observation from the same successfully terminated "
                "dataset trajectory"
            ),
            "over_budget_category_definition": (
                "same-trajectory temporal offset exceeds tau; this is not a "
                "shortest-path lower-bound claim"
            ),
            "success_label_definition": (
                "real closed-loop environment completion within tau model steps"
            ),
            "candidate_categories": [
                WITHIN_BUDGET,
                OVER_BUDGET,
                CROSS_UNREACHABLE,
            ],
            "low_level_h_plan_sequence": [3, 2, 1],
            "execution_horizon_model_steps": 1,
            "num_samples": args.num_samples,
            "cem_iterations": args.cem_iterations,
            "topk": args.topk,
            "planner_seed_per_trial": args.planner_seed,
        },
        "candidate_statistics": {
            "candidate_count": len(candidate_records),
            "overall_local_rc_pass_rate": float(
                np.mean(
                    [record["passed_local_rc"] for record in candidate_records]
                )
            ),
            "overall_positive_progress_pass_rate": float(
                np.mean(
                    [
                        record["passed_positive_progress"]
                        for record in candidate_records
                    ]
                )
            ),
            "overall_joint_filter_pass_rate": float(
                np.mean(
                    [record["passed_joint_filter"] for record in candidate_records]
                )
            ),
            "overall_actual_completion_rate": float(
                np.mean([record["actual_success"] for record in candidate_records])
            ),
            "by_category": category_summary,
        },
        "rc_score_vs_actual_success": correspondence,
        "selection_strategies": strategy_summaries,
        "random_expected_completion_rate": random_expected_rate,
        "paired_comparisons": {
            "rc_vs_random": paired_comparison(
                trials, "rc_then_progress", "random"
            ),
            "rc_vs_progress_only": paired_comparison(
                trials, "rc_then_progress", "progress_only"
            ),
        },
        "eta_r_sweep": sweep,
        "trials": trials,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2)
    output_path.write_text(report_json + "\n")
    print(report_json)
    print(f"report_path: {output_path}")
    return int(not mechanism_validated)


if __name__ == "__main__":
    raise SystemExit(main())
