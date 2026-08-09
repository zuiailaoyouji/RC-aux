#!/usr/bin/env python3
"""Validate strict Oracle pools and RC filtering without a progress ranker."""

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


WITHIN_BUDGET = "within_budget_witnessed"
SAME_TRAJECTORY_STRICT = "same_trajectory_strict_over_budget"
CROSS_TRAJECTORY_STRICT = "cross_trajectory_strict_over_budget"
CATEGORIES = (
    WITHIN_BUDGET,
    SAME_TRAJECTORY_STRICT,
    CROSS_TRAJECTORY_STRICT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: build strict balanced Oracle pools. Stage 2: evaluate "
            "only the local RC filter with disjoint calibration/test episodes."
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
        "--pool-only",
        action="store_true",
        help="Validate and report Stage 1 pools without loading the checkpoint.",
    )
    parser.add_argument("--tau-model-steps", type=int, default=3)
    parser.add_argument("--t-env-step", type=int, default=0)
    parser.add_argument("--candidates-per-category", type=int, default=2)
    parser.add_argument("--calibration-trials", type=int, default=10)
    parser.add_argument("--test-trials", type=int, default=50)
    parser.add_argument("--planner-repeats", type=int, default=3)
    parser.add_argument(
        "--eta-r",
        type=float,
        help=(
            "Pre-fixed threshold. By default it is selected only from the "
            "calibration split under the precision/FPR constraints."
        ),
    )
    parser.add_argument("--max-calibration-fpr", type=float, default=0.05)
    parser.add_argument("--min-calibration-precision", type=float, default=0.95)
    parser.add_argument("--min-test-trial-coverage", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--reachability-cost-weight", type=float, default=0.85)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--planner-seed", type=int, default=4200)
    parser.add_argument("--success-radius", type=float, default=16.0)
    parser.add_argument("--max-replay-pixel-diff", type=int, default=1)
    parser.add_argument("--witness-state-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rc_filter_only_tau3.json"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not args.pool_only
        and args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.tau_model_steps != 3:
        raise ValueError("this controlled experiment requires tau-model-steps=3")
    if args.t_env_step < 0:
        raise ValueError("t-env-step must be nonnegative")
    if not np.isclose(args.success_radius, 16.0):
        raise ValueError("TwoRoom has a fixed environment success radius of 16")
    for name in (
        "candidates_per_category",
        "calibration_trials",
        "test_trials",
        "planner_repeats",
        "bootstrap_samples",
        "num_samples",
        "cem_iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0 < args.topk <= args.num_samples:
        raise ValueError("topk must be in [1, num-samples]")
    for name in (
        "max_calibration_fpr",
        "min_calibration_precision",
        "min_test_trial_coverage",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")
    if args.eta_r is not None and not 0.0 <= args.eta_r <= 1.0:
        raise ValueError("eta-r must be in [0, 1]")


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
    maximum_displacement = speed * np.sqrt(TWOROOM_PROFILE.action_dim)
    required_displacement = max(0.0, distance - success_radius)
    return int(np.ceil(required_displacement / maximum_displacement))


def evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) < count:
        return []
    indices = np.linspace(0, len(items) - 1, num=count)
    return [items[int(round(index))] for index in indices]


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
    return {
        "candidate_id": candidate_id,
        "category": category,
        "row": int(row),
        "episode_index": int(episode_index),
        "temporal_delta_model_steps": temporal_delta_model_steps,
        "image": np.asarray(handle["pixels"][row]),
        "state": state,
        "source_distance": float(np.linalg.norm(source_state - state)),
        "lower_bound_env_steps": lower_bound_env_steps(
            source_state,
            state,
            speed=speed,
            success_radius=success_radius,
        ),
    }


def build_strict_candidate_pool(
    handle: h5py.File,
    *,
    episode_index: int,
    episode_range: range,
    t_env_step: int,
    tau_model_steps: int,
    candidates_per_category: int,
    speed: float,
    success_radius: float,
) -> dict[str, Any] | None:
    offsets = handle["ep_offset"]
    lengths = handle["ep_len"]
    episode_start = int(offsets[episode_index])
    episode_length = int(lengths[episode_index])
    terminal_row = episode_start + episode_length - 1
    if not bool(handle["terminated"][terminal_row]):
        return None
    if t_env_step >= episode_length:
        return None

    block_env_steps = TWOROOM_PROFILE.model_step_env_steps
    budget_env_steps = tau_model_steps * block_env_steps
    source_row = episode_start + t_env_step
    source_state = np.asarray(handle["proprio"][source_row], dtype=np.float32)
    source_image = np.asarray(handle["pixels"][source_row])
    maximum_delta = (episode_length - 1 - t_env_step) // block_env_steps

    within: list[dict[str, Any]] = []
    for delta in range(1, min(tau_model_steps, maximum_delta) + 1):
        candidate = candidate_from_row(
            handle,
            row=source_row + delta * block_env_steps,
            episode_index=episode_index,
            category=WITHIN_BUDGET,
            candidate_id=f"within_delta_{delta}",
            source_state=source_state,
            speed=speed,
            success_radius=success_radius,
            temporal_delta_model_steps=delta,
        )
        if candidate["source_distance"] >= success_radius:
            candidate["witness_action_start_row"] = source_row
            candidate["witness_action_end_row_exclusive"] = candidate["row"]
            within.append(candidate)

    strict_same: list[dict[str, Any]] = []
    for delta in range(tau_model_steps + 1, maximum_delta + 1):
        candidate = candidate_from_row(
            handle,
            row=source_row + delta * block_env_steps,
            episode_index=episode_index,
            category=SAME_TRAJECTORY_STRICT,
            candidate_id=f"strict_same_delta_{delta}",
            source_state=source_state,
            speed=speed,
            success_radius=success_radius,
            temporal_delta_model_steps=delta,
        )
        if candidate["lower_bound_env_steps"] > budget_env_steps:
            strict_same.append(candidate)

    cross: list[dict[str, Any]] = []
    split_episodes = list(episode_range)
    source_position = episode_index - episode_range.start
    for shift in range(1, len(split_episodes)):
        cross_position = (source_position + shift) % len(split_episodes)
        cross_episode = split_episodes[cross_position]
        cross_length = int(lengths[cross_episode])
        if t_env_step >= cross_length:
            continue
        row = int(offsets[cross_episode]) + t_env_step
        candidate = candidate_from_row(
            handle,
            row=row,
            episode_index=cross_episode,
            category=CROSS_TRAJECTORY_STRICT,
            candidate_id=f"strict_cross_ep_{cross_episode}",
            source_state=source_state,
            speed=speed,
            success_radius=success_radius,
            temporal_delta_model_steps=None,
        )
        if candidate["lower_bound_env_steps"] > budget_env_steps:
            cross.append(candidate)
        if len(cross) >= candidates_per_category:
            break

    within = evenly_spaced(within, candidates_per_category)
    strict_same = evenly_spaced(strict_same, candidates_per_category)
    cross = evenly_spaced(cross, candidates_per_category)
    if not within or not strict_same or not cross:
        return None

    candidates = within + strict_same + cross
    return {
        "episode_index": int(episode_index),
        "source_row": int(source_row),
        "source_state": source_state,
        "source_image": source_image,
        "original_target_state": np.asarray(
            handle["pos_target"][source_row], dtype=np.float32
        ),
        "candidates": candidates,
    }


def verify_within_budget_witness(
    handle: h5py.File,
    pool: dict[str, Any],
    candidate: dict[str, Any],
    *,
    env_seed: int,
    state_tolerance: float,
    max_replay_pixel_diff: int,
) -> dict[str, Any]:
    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        disable_env_checker=True,
    )
    try:
        env.reset(seed=env_seed)
        base_env = env.unwrapped
        base_env._set_state(pool["source_state"])
        base_env._set_goal_state(pool["original_target_state"])
        source_replay = frame_error(pool["source_image"], env.render())
        action_start = candidate["witness_action_start_row"]
        action_end = candidate["witness_action_end_row_exclusive"]
        final_state = pool["source_state"]
        for row in range(action_start, action_end):
            _, _, _, _, info = env.step(
                np.asarray(handle["action"][row], dtype=np.float32)
            )
            final_state = np.asarray(info["proprio"], dtype=np.float32)
        target_replay = frame_error(candidate["image"], env.render())
    finally:
        env.close()

    final_state_error = float(np.linalg.norm(final_state - candidate["state"]))
    valid = bool(
        action_end - action_start
        <= 3 * TWOROOM_PROFILE.model_step_env_steps
        and final_state_error <= state_tolerance
        and source_replay["max_abs_pixel_diff"] <= max_replay_pixel_diff
        and target_replay["max_abs_pixel_diff"] <= max_replay_pixel_diff
    )
    return {
        "valid": valid,
        "action_count_env_steps": int(action_end - action_start),
        "final_state_error": final_state_error,
        "source_replay": source_replay,
        "target_replay": target_replay,
    }


def collect_pools(
    handle: h5py.File,
    *,
    episode_range: range,
    requested_trials: int,
    args: argparse.Namespace,
    speed: float,
) -> list[dict[str, Any]]:
    pools = []
    budget_env_steps = (
        args.tau_model_steps * TWOROOM_PROFILE.model_step_env_steps
    )
    for episode_index in episode_range:
        if len(pools) >= requested_trials:
            break
        pool = build_strict_candidate_pool(
            handle,
            episode_index=episode_index,
            episode_range=episode_range,
            t_env_step=args.t_env_step,
            tau_model_steps=args.tau_model_steps,
            candidates_per_category=args.candidates_per_category,
            speed=speed,
            success_radius=args.success_radius,
        )
        if pool is None:
            continue
        witnesses = []
        for candidate in pool["candidates"]:
            if candidate["category"] == WITHIN_BUDGET:
                witness = verify_within_budget_witness(
                    handle,
                    pool,
                    candidate,
                    env_seed=args.env_seed,
                    state_tolerance=args.witness_state_tolerance,
                    max_replay_pixel_diff=args.max_replay_pixel_diff,
                )
                candidate["trajectory_witness"] = witness
                witnesses.append(witness["valid"])
            else:
                candidate["strict_over_budget_proof"] = bool(
                    candidate["lower_bound_env_steps"] > budget_env_steps
                )
        if not all(witnesses):
            continue
        pools.append(pool)
    if len(pools) != requested_trials:
        raise RuntimeError(
            f"constructed only {len(pools)} strict pools, requested "
            f"{requested_trials}, from episode range "
            f"[{episode_range.start}, {episode_range.stop})"
        )
    return pools


def execute_candidate(
    adapter: RCAuxAdapter,
    *,
    source_state: np.ndarray,
    source_image: np.ndarray,
    target_state: np.ndarray,
    target_image: np.ndarray,
    target_latent: torch.Tensor,
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
    success = False
    truncated = False
    env_steps = 0
    h_plan_sequence = []
    online_rc_sequence = []
    distances = []
    final_state = np.asarray(source_state, dtype=np.float32)
    try:
        env.reset(seed=env_seed)
        base_env = env.unwrapped
        base_env._set_state(source_state)
        base_env._set_goal_state(target_state)
        source_replay = frame_error(source_image, env.render())
        rendered_target = (
            base_env._render_frame(agent_pos=torch.as_tensor(target_state))
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        target_replay = frame_error(target_image, rendered_target)
        if (
            source_replay["max_abs_pixel_diff"] > max_replay_pixel_diff
            or target_replay["max_abs_pixel_diff"] > max_replay_pixel_diff
        ):
            raise RuntimeError("dataset/environment replay mismatch")
        initial_distance = float(np.linalg.norm(source_state - target_state))
        if initial_distance < success_radius:
            raise RuntimeError("candidate starts inside the success radius")
        distances.append(initial_distance)
        adapter.reset_planner(seed=planner_seed)

        for executed_model_steps in range(tau_model_steps):
            h_rem = tau_model_steps - executed_model_steps
            current_image = env.render()
            current_latent = adapter.encode_observation(current_image)[:, -1]
            online_rc = adapter.reachability(
                current_latent,
                target_latent,
                horizon_model_steps=h_rem,
            )
            plan = adapter.plan_to_latent(
                current_image,
                target_latent,
                planning_horizon_model_steps=h_rem,
                execution_horizon_model_steps=1,
            )
            h_plan_sequence.append(
                plan.diagnostics.planning_horizon_model_steps
            )
            online_rc_sequence.append(float(online_rc.item()))
            for action in plan.actions_to_execute_env_steps[0]:
                _, _, terminated, step_truncated, info = env.step(action)
                env_steps += 1
                success = success or bool(terminated)
                truncated = truncated or bool(step_truncated)
                final_state = np.asarray(info["proprio"], dtype=np.float32)
                distances.append(float(info["distance_to_target"]))
                if success or truncated:
                    break
            if success or truncated:
                break
    finally:
        env.close()

    return {
        "actual_success": bool(success),
        "truncated": bool(truncated),
        "env_steps_executed": int(env_steps),
        "initial_distance": distances[0],
        "final_distance": distances[-1],
        "minimum_distance": min(distances),
        "final_state": final_state.tolist(),
        "h_plan_sequence": h_plan_sequence,
        "online_rc_sequence": online_rc_sequence,
    }


def evaluate_pools(
    adapter: RCAuxAdapter,
    pools: list[dict[str, Any]],
    *,
    split_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    trials = []
    for trial_index, pool in enumerate(pools):
        source_latent = adapter.encode_observation(pool["source_image"])[:, -1]
        candidate_images = np.stack(
            [candidate["image"] for candidate in pool["candidates"]]
        )
        candidate_latents = adapter.encode_observation(candidate_images)[:, -1]
        rc = adapter.reachability(
            source_latent,
            candidate_latents.unsqueeze(0),
            horizon_model_steps=args.tau_model_steps,
            return_diagnostics=True,
        )
        print(
            f"split={split_name} trial={trial_index} "
            f"episode={pool['episode_index']}",
            flush=True,
        )
        records = []
        for candidate_index, candidate in enumerate(pool["candidates"]):
            executions = []
            for repeat in range(args.planner_repeats):
                execution = execute_candidate(
                    adapter,
                    source_state=pool["source_state"],
                    source_image=pool["source_image"],
                    target_state=candidate["state"],
                    target_image=candidate["image"],
                    target_latent=(
                        candidate_latents[candidate_index : candidate_index + 1]
                    ),
                    tau_model_steps=args.tau_model_steps,
                    env_seed=args.env_seed,
                    planner_seed=(
                        args.planner_seed + trial_index * 100 + repeat
                    ),
                    success_radius=args.success_radius,
                    max_replay_pixel_diff=args.max_replay_pixel_diff,
                )
                execution["repeat_index"] = repeat
                executions.append(execution)
            record = {
                key: value
                for key, value in candidate.items()
                if key not in {"image", "state"}
            }
            record.update(
                {
                    "state": candidate["state"].tolist(),
                    "local_rc_score": float(
                        rc.probabilities[0, candidate_index]
                    ),
                    "local_rc_logit": float(rc.logits[0, candidate_index]),
                    "executions": executions,
                    "actual_success_rate": float(
                        np.mean([item["actual_success"] for item in executions])
                    ),
                }
            )
            records.append(record)
            print(
                f"  {record['candidate_id']} category={record['category']} "
                f"lower_bound={record['lower_bound_env_steps']} "
                f"r={record['local_rc_score']:.4f} "
                f"success_rate={record['actual_success_rate']:.3f}",
                flush=True,
            )
        trials.append(
            {
                "trial_index": trial_index,
                "episode_index": pool["episode_index"],
                "source_row": pool["source_row"],
                "source_state": pool["source_state"].tolist(),
                "candidates": records,
            }
        )
    return trials


def execution_arrays(
    trials: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for trial in trials:
        for candidate in trial["candidates"]:
            for execution in candidate["executions"]:
                scores.append(candidate["local_rc_score"])
                labels.append(execution["actual_success"])
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = scores >= threshold
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    predicted_positive = tp + fp
    both_classes = np.unique(labels).size == 2
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / predicted_positive if predicted_positive else None,
        "recall": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "specificity": tn / max(1, tn + fp),
        "execution_coverage": float(predictions.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, scores))
            if labels.any()
            else None
        ),
    }


def select_calibration_threshold(
    trials: list[dict[str, Any]],
    *,
    max_fpr: float,
    min_precision: float,
) -> tuple[float, bool, list[dict[str, Any]]]:
    scores, labels = execution_arrays(trials)
    thresholds = sorted({0.0, 1.0, *scores.tolist()})
    rows = [binary_metrics(scores, labels, threshold) for threshold in thresholds]
    feasible = [
        row
        for row in rows
        if row["precision"] is not None
        and row["precision"] >= min_precision
        and row["false_positive_rate"] <= max_fpr
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["recall"],
                row["execution_coverage"],
                -row["threshold"],
            ),
        )
        constraints_met = True
    else:
        nonempty = [row for row in rows if row["precision"] is not None]
        selected = max(
            nonempty,
            key=lambda row: (
                row["precision"],
                row["recall"],
                -row["false_positive_rate"],
            ),
        )
        constraints_met = False
    return float(selected["threshold"]), constraints_met, rows


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def summarize_split(
    trials: list[dict[str, Any]],
    *,
    threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    scores, labels = execution_arrays(trials)
    classification = binary_metrics(scores, labels, threshold)
    category_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_expected = []
    filtered_conditional = []
    filtered_unconditional = []
    paired_differences = []
    selected_category_counts = Counter()

    for trial in trials:
        candidates = trial["candidates"]
        for candidate in candidates:
            candidate["passed_rc"] = bool(
                candidate["local_rc_score"] >= threshold
            )
            category_records[candidate["category"]].append(candidate)
        trial_all = float(
            np.mean([candidate["actual_success_rate"] for candidate in candidates])
        )
        passed = [candidate for candidate in candidates if candidate["passed_rc"]]
        all_expected.append(trial_all)
        if passed:
            trial_filtered = float(
                np.mean(
                    [candidate["actual_success_rate"] for candidate in passed]
                )
            )
            filtered_conditional.append(trial_filtered)
            filtered_unconditional.append(trial_filtered)
            paired_differences.append(trial_filtered - trial_all)
            selected_category_counts.update(
                candidate["category"] for candidate in passed
            )
        else:
            filtered_unconditional.append(0.0)

    category_summary = {
        category: {
            "candidate_count": len(records),
            "rc_pass_rate": float(np.mean([r["passed_rc"] for r in records])),
            "mean_rc_score": float(
                np.mean([r["local_rc_score"] for r in records])
            ),
            "actual_completion_rate": float(
                np.mean([r["actual_success_rate"] for r in records])
            ),
        }
        for category, records in category_records.items()
    }
    trial_coverage = len(filtered_conditional) / len(trials)
    return {
        "classification": classification,
        "candidate_count": int(sum(len(t["candidates"]) for t in trials)),
        "trial_count": len(trials),
        "trial_coverage": float(trial_coverage),
        "abstained_trial_count": len(trials) - len(filtered_conditional),
        "by_category": category_summary,
        "uniform_all_pool": {
            "expected_completion_rate": float(np.mean(all_expected)),
        },
        "uniform_rc_pass_set": {
            "conditional_expected_completion_rate": (
                float(np.mean(filtered_conditional))
                if filtered_conditional
                else None
            ),
            "unconditional_expected_completion_rate": float(
                np.mean(filtered_unconditional)
            ),
            "passed_candidate_categories": dict(selected_category_counts),
        },
        "paired_conditional_improvement": bootstrap_mean_ci(
            paired_differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def risk_coverage_curve(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    curve = []
    for threshold in np.linspace(0.0, 1.0, 21):
        passed = [
            candidate
            for trial in trials
            for candidate in trial["candidates"]
            if candidate["local_rc_score"] >= threshold
        ]
        covered_trials = sum(
            any(
                candidate["local_rc_score"] >= threshold
                for candidate in trial["candidates"]
            )
            for trial in trials
        )
        curve.append(
            {
                "threshold": float(threshold),
                "candidate_coverage": len(passed)
                / sum(len(trial["candidates"]) for trial in trials),
                "trial_coverage": covered_trials / len(trials),
                "conditional_expected_completion_rate": (
                    float(
                        np.mean(
                            [item["actual_success_rate"] for item in passed]
                        )
                    )
                    if passed
                    else None
                ),
            }
        )
    return curve


def stage1_split_summary(
    pools: list[dict[str, Any]],
    *,
    budget_env_steps: int,
    witness_state_tolerance: float,
    max_replay_pixel_diff: int,
) -> dict[str, Any]:
    category_counts = Counter()
    strict_lower_bounds = []
    witness_errors = []
    witness_source_pixel_diffs = []
    witness_target_pixel_diffs = []
    serializable_pools = []
    for pool in pools:
        candidates = []
        for candidate in pool["candidates"]:
            category_counts[candidate["category"]] += 1
            if candidate["category"] == WITHIN_BUDGET:
                witness = candidate["trajectory_witness"]
                witness_errors.append(witness["final_state_error"])
                witness_source_pixel_diffs.append(
                    witness["source_replay"]["max_abs_pixel_diff"]
                )
                witness_target_pixel_diffs.append(
                    witness["target_replay"]["max_abs_pixel_diff"]
                )
            else:
                strict_lower_bounds.append(candidate["lower_bound_env_steps"])
            record = {
                key: value
                for key, value in candidate.items()
                if key not in {"image", "state"}
            }
            record["state"] = candidate["state"].tolist()
            candidates.append(record)
        serializable_pools.append(
            {
                "episode_index": pool["episode_index"],
                "source_row": pool["source_row"],
                "source_state": pool["source_state"].tolist(),
                "candidates": candidates,
            }
        )
    return {
        "trial_count": len(pools),
        "episode_indices": [pool["episode_index"] for pool in pools],
        "category_counts": dict(category_counts),
        "all_strict_negatives_exceed_budget": bool(
            strict_lower_bounds
            and min(strict_lower_bounds) > budget_env_steps
        ),
        "strict_negative_lower_bound_min": min(strict_lower_bounds),
        "strict_negative_lower_bound_max": max(strict_lower_bounds),
        "all_within_witnesses_valid": bool(
            witness_errors
            and max(witness_errors) <= witness_state_tolerance
            and max(witness_source_pixel_diffs) <= max_replay_pixel_diff
            and max(witness_target_pixel_diffs) <= max_replay_pixel_diff
        ),
        "within_witness_state_error_max": max(witness_errors),
        "within_witness_source_pixel_diff_max": max(
            witness_source_pixel_diffs
        ),
        "within_witness_target_pixel_diff_max": max(
            witness_target_pixel_diffs
        ),
        "pools": serializable_pools,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2)
    output_path.write_text(report_json + "\n")
    print(report_json)
    print(f"report_path: {output_path}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_path = cache_dir / args.dataset
    speed = environment_speed(args.env_seed)
    budget_env_steps = (
        args.tau_model_steps * TWOROOM_PROFILE.model_step_env_steps
    )

    with h5py.File(dataset_path, "r") as handle:
        episode_count = len(handle["ep_offset"])
        split_boundary = episode_count // 2
        calibration_range = range(0, split_boundary)
        test_range = range(split_boundary, episode_count)
        calibration_pools = collect_pools(
            handle,
            episode_range=calibration_range,
            requested_trials=args.calibration_trials,
            args=args,
            speed=speed,
        )
        test_pools = collect_pools(
            handle,
            episode_range=test_range,
            requested_trials=args.test_trials,
            args=args,
            speed=speed,
        )

    stage1 = {
        "strict_oracle_pool_validated": True,
        "budget_env_steps": budget_env_steps,
        "candidates_per_category": args.candidates_per_category,
        "balanced_candidates_per_pool": True,
        "calibration_episode_range": [0, split_boundary],
        "test_episode_range": [split_boundary, episode_count],
        "calibration": stage1_split_summary(
            calibration_pools,
            budget_env_steps=budget_env_steps,
            witness_state_tolerance=args.witness_state_tolerance,
            max_replay_pixel_diff=args.max_replay_pixel_diff,
        ),
        "test": stage1_split_summary(
            test_pools,
            budget_env_steps=budget_env_steps,
            witness_state_tolerance=args.witness_state_tolerance,
            max_replay_pixel_diff=args.max_replay_pixel_diff,
        ),
    }
    if args.pool_only:
        write_report(
            {
                "stage1_strict_oracle_pool_validated": True,
                "stage2_rc_filter_validated": None,
                "stage1": stage1,
            },
            args.output,
        )
        return 0

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

    calibration_trials = evaluate_pools(
        adapter,
        calibration_pools,
        split_name="calibration",
        args=args,
    )
    test_trials = evaluate_pools(
        adapter,
        test_pools,
        split_name="test",
        args=args,
    )

    if args.eta_r is None:
        eta_r, calibration_constraints_met, calibration_thresholds = (
            select_calibration_threshold(
                calibration_trials,
                max_fpr=args.max_calibration_fpr,
                min_precision=args.min_calibration_precision,
            )
        )
        threshold_source = "selected_on_disjoint_calibration_split"
    else:
        eta_r = args.eta_r
        scores, labels = execution_arrays(calibration_trials)
        selected_metrics = binary_metrics(scores, labels, eta_r)
        calibration_constraints_met = bool(
            selected_metrics["precision"] is not None
            and selected_metrics["precision"]
            >= args.min_calibration_precision
            and selected_metrics["false_positive_rate"]
            <= args.max_calibration_fpr
        )
        calibration_thresholds = [selected_metrics]
        threshold_source = "pre_fixed_command_line"

    calibration_summary = summarize_split(
        calibration_trials,
        threshold=eta_r,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    test_summary = summarize_split(
        test_trials,
        threshold=eta_r,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 1,
    )
    test_improvement = test_summary["paired_conditional_improvement"]
    test_classification = test_summary["classification"]
    validation_criteria = {
        "calibration_constraints_met": calibration_constraints_met,
        "test_rc_auc_above_chance": bool(
            test_classification["roc_auc"] is not None
            and test_classification["roc_auc"] > 0.5
        ),
        "test_fpr_within_calibrated_limit": bool(
            test_classification["false_positive_rate"]
            <= args.max_calibration_fpr
        ),
        "test_trial_coverage_at_least_minimum": bool(
            test_summary["trial_coverage"] >= args.min_test_trial_coverage
        ),
        "paired_completion_improvement_ci_above_zero": bool(
            test_improvement is not None
            and test_improvement["lower_95"] > 0.0
        ),
    }
    stage2_validated = all(validation_criteria.values())
    report = {
        "stage1_strict_oracle_pool_validated": True,
        "stage2_rc_filter_validated": stage2_validated,
        "ready_for_next_experiment_design": stage2_validated,
        "ready_to_train_high_level_generator": False,
        "stage1": stage1,
        "validation_criteria": validation_criteria,
        "policy": args.policy,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "protocol": {
            "tau_model_steps": args.tau_model_steps,
            "budget_env_steps": budget_env_steps,
            "model_step_env_steps": TWOROOM_PROFILE.model_step_env_steps,
            "candidate_categories": list(CATEGORIES),
            "candidates_per_category": args.candidates_per_category,
            "balanced_candidates_per_pool": True,
            "within_budget_definition": (
                "real trajectory action witness of at most 15 environment steps"
            ),
            "strict_negative_definition": (
                "collision-agnostic displacement lower bound exceeds 15 "
                "environment steps; obstacles can only increase required effort"
            ),
            "uses_progress_ranking_or_final_goal_progress": False,
            "selection_comparison": [
                "uniform_all_pool_expected_completion",
                "uniform_rc_pass_set_expected_completion",
            ],
            "calibration_episode_range": [0, split_boundary],
            "test_episode_range": [split_boundary, episode_count],
            "episode_disjoint_calibration_and_test": True,
            "calibration_trials": args.calibration_trials,
            "test_trials": args.test_trials,
            "planner_repeats_per_candidate": args.planner_repeats,
            "low_level_h_plan_sequence": [3, 2, 1],
            "execution_horizon_model_steps": 1,
        },
        "threshold": {
            "eta_r": eta_r,
            "source": threshold_source,
            "max_calibration_fpr": args.max_calibration_fpr,
            "min_calibration_precision": args.min_calibration_precision,
            "calibration_constraints_met": calibration_constraints_met,
            "calibration_threshold_table": calibration_thresholds,
        },
        "calibration": calibration_summary,
        "test": test_summary,
        "test_risk_coverage_curve": risk_coverage_curve(test_trials),
        "calibration_trials": calibration_trials,
        "test_trials": test_trials,
    }
    write_report(report, args.output)
    return int(not stage2_validated)


if __name__ == "__main__":
    raise SystemExit(main())
