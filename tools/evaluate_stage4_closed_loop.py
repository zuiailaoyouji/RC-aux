#!/usr/bin/env python3
"""Run the formal Stage 4 closed-loop comparison in TwoRoom."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401 - registers TwoRoom
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, RCAuxPlannerConfig, TWOROOM_PROFILE
from stage4_generator import (
    MODEL_STEP_ENV_STEPS,
    TAU_MODEL_STEPS,
    assert_protocol_consistency,
    distribution_summary,
    left_padded_history,
    load_checkpoint_protocol,
    load_generator_checkpoint,
    load_latent_cache,
    load_stage2_protocol,
    sample_subgoal_candidates,
    select_candidate_index,
    split_cached_episodes,
)
from tools.evaluate_stage4_candidates import (
    checkpoint_paths,
    load_progress_ranker,
)


METHODS = (
    "reference_trajectory_waypoint",
    "deterministic_single",
    "stochastic32_dpsi_no_rc",
    "stochastic32_rc_dpsi",
    "flat_rc_lewm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare five Stage 4 policies with matched episodes and CEM seeds."
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
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--eval-budget-env-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--cem-seed", type=int, default=4200)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--dropout-seed", type=int, default=20260809)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--flat-planning-horizon-model-steps", type=int, default=5)
    parser.add_argument("--reachability-cost-weight", type=float, default=0.85)
    parser.add_argument("--max-replay-pixel-diff", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage4_closed_loop.json"),
    )
    return parser.parse_args()


def frame_error(reference: np.ndarray, rendered: np.ndarray) -> int:
    difference = np.abs(reference.astype(np.int16) - rendered.astype(np.int16))
    return int(difference.max(initial=0))


def select_episode_records(
    episodes: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    if count > len(episodes):
        raise ValueError("num-episodes exceeds successful test episodes")
    positions = np.linspace(0, len(episodes) - 1, num=count)
    return [episodes[int(round(position))] for position in positions]


def demonstrations_within_budget(
    handle: h5py.File,
    episodes: list[dict[str, Any]],
    *,
    budget_env_steps: int,
) -> list[dict[str, Any]]:
    eligible = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        demonstration_env_steps = int(handle["ep_len"][episode_index]) - 1
        if demonstration_env_steps <= budget_env_steps:
            eligible.append(episode)
    return eligible


def load_environment_record(
    handle: h5py.File,
    episode: dict[str, Any],
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    start = int(handle["ep_offset"][episode_index])
    length = int(handle["ep_len"][episode_index])
    terminal = start + length - 1
    return {
        "episode_index": episode_index,
        "start_row": start,
        "terminal_row": terminal,
        "demonstration_env_steps": length - 1,
        "source_image": np.asarray(handle["pixels"][start]),
        "goal_image": np.asarray(handle["pixels"][terminal]),
        "source_state": np.asarray(handle["proprio"][start], dtype=np.float32),
        "terminal_state": np.asarray(
            handle["proprio"][terminal], dtype=np.float32
        ),
        "task_target_state": np.asarray(
            handle["pos_target"][start], dtype=np.float32
        ),
        "reference_rows": episode["rows"].numpy(),
        "reference_latents": episode["latents"].to(torch.float32),
    }


def reference_trajectory_waypoint(
    record: dict[str, Any],
    *,
    env_steps_executed: int,
) -> torch.Tensor:
    desired_row = record["start_row"] + env_steps_executed + (
        TAU_MODEL_STEPS * MODEL_STEP_ENV_STEPS
    )
    rows = record["reference_rows"]
    matches = np.flatnonzero(rows == desired_row)
    if len(matches):
        return record["reference_latents"][int(matches[0])]
    return record["reference_latents"][-1]


def pairwise_l2(candidates: torch.Tensor) -> float:
    if candidates.size(0) < 2:
        return 0.0
    return float(torch.pdist(candidates).mean())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def choose_subgoal(
    method: str,
    generator,
    progress_ranker,
    adapter: RCAuxAdapter,
    history: deque[torch.Tensor],
    goal: torch.Tensor,
    record: dict[str, Any],
    *,
    env_steps_executed: int,
    num_candidates: int,
    eta_r: float,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    current = history[-1].to(device)
    goal = goal.to(device)
    diagnostics = {
        "candidate_count": 0,
        "rc_pass_count": None,
        "rc_and_progress_count": None,
        "coverage": True,
        "selected_rc_score": None,
        "selected_d_psi_progress": None,
        "pairwise_diversity": None,
        "direct_goal_eligible": False,
        "direct_goal_selected": False,
        "timing_seconds": {"generator": 0.0, "rc": 0.0, "d_psi": 0.0},
    }
    if method == "reference_trajectory_waypoint":
        selected = reference_trajectory_waypoint(
            record, env_steps_executed=env_steps_executed
        ).to(device)
    elif method == "deterministic_single":
        history_tensor, mask = left_padded_history(history)
        synchronize(device)
        started = time.perf_counter()
        selected = sample_subgoal_candidates(
            generator,
            history_tensor.to(device),
            mask.to(device),
            goal.unsqueeze(0),
            num_candidates=1,
            stochastic=False,
        )[0, 0]
        synchronize(device)
        diagnostics["timing_seconds"]["generator"] = time.perf_counter() - started
        diagnostics["candidate_count"] = 1
    else:
        history_tensor, mask = left_padded_history(history)
        synchronize(device)
        started = time.perf_counter()
        candidates = sample_subgoal_candidates(
            generator,
            history_tensor.to(device),
            mask.to(device),
            goal.unsqueeze(0),
            num_candidates=num_candidates,
            stochastic=True,
        )[0]
        synchronize(device)
        diagnostics["timing_seconds"]["generator"] = time.perf_counter() - started
        diagnostics["candidate_count"] = num_candidates
        diagnostics["pairwise_diversity"] = pairwise_l2(candidates)

        synchronize(device)
        started = time.perf_counter()
        source_score = progress_ranker(current.unsqueeze(0), goal.unsqueeze(0))[0]
        candidate_score = progress_ranker(
            candidates, goal.expand(num_candidates, -1)
        )
        direct_goal_score = progress_ranker(
            goal.unsqueeze(0), goal.unsqueeze(0)
        )[0]
        progress = source_score - candidate_score
        synchronize(device)
        diagnostics["timing_seconds"]["d_psi"] = time.perf_counter() - started
        if method == "stochastic32_rc_dpsi":
            synchronize(device)
            started = time.perf_counter()
            rc = adapter.reachability(
                current.unsqueeze(0),
                candidates.unsqueeze(0),
                horizon_model_steps=TAU_MODEL_STEPS,
            )[0]
            direct_goal_rc = adapter.reachability(
                current.unsqueeze(0),
                goal.unsqueeze(0),
                horizon_model_steps=TAU_MODEL_STEPS,
            )[0]
            synchronize(device)
            diagnostics["timing_seconds"]["rc"] = time.perf_counter() - started
            combined_score = torch.cat(
                [candidate_score, direct_goal_score.unsqueeze(0)]
            )
            combined_rc = torch.cat([rc, direct_goal_rc.unsqueeze(0)])
            selection = select_candidate_index(
                combined_score,
                source_score,
                rc_scores=combined_rc,
                eta_r=eta_r,
            )
            feasible = selection["feasible"]
            diagnostics["rc_pass_count"] = int(selection["rc_pass"][:-1].sum())
            diagnostics["rc_and_progress_count"] = int(feasible[:-1].sum())
            diagnostics["direct_goal_eligible"] = bool(feasible[-1])
        else:
            rc = None
            selection = select_candidate_index(
                candidate_score, source_score, rc_scores=None
            )
            feasible = selection["feasible"]
            diagnostics["rc_and_progress_count"] = int(feasible.sum())
        if not bool(feasible.any()):
            diagnostics["coverage"] = False
            return None, diagnostics
        selected_index = int(selection["selected_index"])
        if rc is not None and selected_index == num_candidates:
            selected = goal
            diagnostics["direct_goal_selected"] = True
            diagnostics["selected_d_psi_progress"] = float(
                source_score - direct_goal_score
            )
            diagnostics["selected_rc_score"] = float(direct_goal_rc)
        else:
            selected = candidates[selected_index]
            diagnostics["selected_d_psi_progress"] = float(progress[selected_index])
            if rc is not None:
                diagnostics["selected_rc_score"] = float(rc[selected_index])

    if diagnostics["selected_rc_score"] is None:
        synchronize(device)
        started = time.perf_counter()
        selected_rc = adapter.reachability(
            current.unsqueeze(0),
            selected.unsqueeze(0),
            horizon_model_steps=TAU_MODEL_STEPS,
        )
        synchronize(device)
        diagnostics["timing_seconds"]["rc"] += time.perf_counter() - started
        diagnostics["selected_rc_score"] = float(selected_rc.item())
    if diagnostics["selected_d_psi_progress"] is None:
        synchronize(device)
        started = time.perf_counter()
        diagnostics["selected_d_psi_progress"] = float(
            progress_ranker(current.unsqueeze(0), goal.unsqueeze(0))[0]
            - progress_ranker(selected.unsqueeze(0), goal.unsqueeze(0))[0]
        )
        synchronize(device)
        diagnostics["timing_seconds"]["d_psi"] += time.perf_counter() - started
    return selected.detach(), diagnostics


def execute_action_block(
    env,
    actions: np.ndarray,
    *,
    remaining_budget: int,
) -> tuple[int, bool, bool, dict[str, Any]]:
    executed = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = {}
    for action in actions[:remaining_budget]:
        _, _, step_terminated, step_truncated, last_info = env.step(action)
        executed += 1
        terminated = terminated or bool(step_terminated)
        truncated = truncated or bool(step_truncated)
        if terminated or truncated:
            break
    return executed, terminated, truncated, last_info


@torch.inference_mode()
def run_rollout(
    method: str,
    generator,
    progress_ranker,
    adapter: RCAuxAdapter,
    record: dict[str, Any],
    *,
    num_candidates: int,
    eval_budget_env_steps: int,
    flat_horizon: int,
    eta_r: float,
    env_seed: int,
    cem_seed: int,
    dropout_seed: int,
    max_replay_pixel_diff: int,
    device: torch.device,
) -> dict[str, Any]:
    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        max_episode_steps=eval_budget_env_steps,
        disable_env_checker=True,
    )
    torch.manual_seed(dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dropout_seed)
    adapter.reset_planner(seed=cem_seed)
    env_steps = 0
    success = False
    truncated = False
    fallback_steps = 0
    high_level_attempts = 0
    covered_attempts = 0
    direct_goal_eligible_attempts = 0
    direct_goal_selected_attempts = 0
    selected_rc_scores = []
    selected_progress = []
    candidate_diversities = []
    segment_records = []
    timing = {"generator": 0.0, "rc": 0.0, "d_psi": 0.0, "cem": 0.0}

    try:
        env.reset(seed=env_seed)
        base_env = env.unwrapped
        base_env._set_state(record["source_state"])
        # Restore the dataset task for official termination. The high-level
        # goal remains E(o_T); this privileged state is never passed to a model.
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
        current_image = env.render()
        current = adapter.encode_observation(current_image)[:, -1][0]
        history: deque[torch.Tensor] = deque([current.cpu()], maxlen=3)
        initial_distance = float(
            np.linalg.norm(
                record["source_state"] - record["task_target_state"]
            )
        )
        current_distance = initial_distance

        while env_steps < eval_budget_env_steps and not success and not truncated:
            if method == "flat_rc_lewm":
                segment_start_distance = current_distance
                segment_start_latent = history[-1].to(device)
                segment_env_start = env_steps
                for _ in range(TAU_MODEL_STEPS):
                    if env_steps >= eval_budget_env_steps or success or truncated:
                        break
                    current_image = env.render()
                    plan = adapter.plan_to_latent(
                        current_image,
                        goal.unsqueeze(0),
                        planning_horizon_model_steps=flat_horizon,
                        execution_horizon_model_steps=1,
                    )
                    timing["cem"] += plan.diagnostics.planning_time_seconds
                    executed, success, truncated, info = execute_action_block(
                        env,
                        plan.actions_to_execute_env_steps[0],
                        remaining_budget=eval_budget_env_steps - env_steps,
                    )
                    env_steps += executed
                    if info:
                        current_distance = float(info["distance_to_target"])
                    current = adapter.encode_observation(env.render())[:, -1][0]
                    history.append(current.cpu())
                end_score = progress_ranker(
                    history[-1].to(device).unsqueeze(0), goal.unsqueeze(0)
                )[0]
                start_score = progress_ranker(
                    segment_start_latent.unsqueeze(0), goal.unsqueeze(0)
                )[0]
                segment_records.append(
                    {
                        "env_step_start": segment_env_start,
                        "env_step_end": env_steps,
                        "euclidean_target_distance_progress": (
                            segment_start_distance - current_distance
                        ),
                        "d_psi_progress": float(start_score - end_score),
                        "fallback": False,
                    }
                )
                continue

            high_level_attempts += 1
            subgoal, diagnostics = choose_subgoal(
                method,
                generator,
                progress_ranker,
                adapter,
                history,
                goal,
                record,
                env_steps_executed=env_steps,
                num_candidates=num_candidates,
                eta_r=eta_r,
                device=device,
            )
            for key in ("generator", "rc", "d_psi"):
                timing[key] += diagnostics["timing_seconds"][key]
            if diagnostics["pairwise_diversity"] is not None:
                candidate_diversities.append(diagnostics["pairwise_diversity"])
            direct_goal_eligible_attempts += int(
                diagnostics["direct_goal_eligible"]
            )
            direct_goal_selected_attempts += int(
                diagnostics["direct_goal_selected"]
            )

            if subgoal is None:
                current_image = env.render()
                plan = adapter.plan_to_latent(
                    current_image,
                    goal.unsqueeze(0),
                    planning_horizon_model_steps=flat_horizon,
                    execution_horizon_model_steps=1,
                    force_reset_warm_start=True,
                )
                timing["cem"] += plan.diagnostics.planning_time_seconds
                start_distance = current_distance
                start_latent = history[-1].to(device)
                executed, success, truncated, info = execute_action_block(
                    env,
                    plan.actions_to_execute_env_steps[0],
                    remaining_budget=eval_budget_env_steps - env_steps,
                )
                env_steps += executed
                fallback_steps += 1
                if info:
                    current_distance = float(info["distance_to_target"])
                current = adapter.encode_observation(env.render())[:, -1][0]
                history.append(current.cpu())
                end_score = progress_ranker(current.unsqueeze(0), goal.unsqueeze(0))[0]
                start_score = progress_ranker(
                    start_latent.unsqueeze(0), goal.unsqueeze(0)
                )[0]
                segment_records.append(
                    {
                        "env_step_start": env_steps - executed,
                        "env_step_end": env_steps,
                        "euclidean_target_distance_progress": (
                            start_distance - current_distance
                        ),
                        "d_psi_progress": float(start_score - end_score),
                        "fallback": True,
                    }
                )
                continue

            covered_attempts += 1
            selected_rc_scores.append(diagnostics["selected_rc_score"])
            selected_progress.append(diagnostics["selected_d_psi_progress"])
            segment_start = env_steps
            start_distance = current_distance
            start_latent = history[-1].to(device)
            for h_rem in range(TAU_MODEL_STEPS, 0, -1):
                if env_steps >= eval_budget_env_steps or success or truncated:
                    break
                current_image = env.render()
                plan = adapter.plan_to_latent(
                    current_image,
                    subgoal.unsqueeze(0),
                    planning_horizon_model_steps=h_rem,
                    execution_horizon_model_steps=1,
                    force_reset_warm_start=h_rem == TAU_MODEL_STEPS,
                )
                timing["cem"] += plan.diagnostics.planning_time_seconds
                executed, success, truncated, info = execute_action_block(
                    env,
                    plan.actions_to_execute_env_steps[0],
                    remaining_budget=eval_budget_env_steps - env_steps,
                )
                env_steps += executed
                if info:
                    current_distance = float(info["distance_to_target"])
                current = adapter.encode_observation(env.render())[:, -1][0]
                history.append(current.cpu())
            end_score = progress_ranker(current.unsqueeze(0), goal.unsqueeze(0))[0]
            start_score = progress_ranker(
                start_latent.unsqueeze(0), goal.unsqueeze(0)
            )[0]
            segment_records.append(
                {
                    "env_step_start": segment_start,
                    "env_step_end": env_steps,
                    "euclidean_target_distance_progress": (
                        start_distance - current_distance
                    ),
                    "d_psi_progress": float(start_score - end_score),
                    "fallback": False,
                }
            )
    finally:
        env.close()

    return {
        "episode_index": record["episode_index"],
        "method": method,
        "success": bool(success),
        "truncated": bool(truncated),
        "env_steps": env_steps,
        "completion_env_steps": env_steps if success else None,
        "initial_distance": initial_distance,
        "final_distance": current_distance,
        "total_euclidean_target_distance_progress_diagnostic": (
            initial_distance - current_distance
        ),
        "high_level_attempts": high_level_attempts,
        "covered_high_level_attempts": covered_attempts,
        "candidate_coverage": (
            covered_attempts / high_level_attempts if high_level_attempts else None
        ),
        "fallback_model_steps": fallback_steps,
        "fallback_rate": (
            fallback_steps / high_level_attempts if high_level_attempts else 0.0
        ),
        "selected_rc_scores": selected_rc_scores,
        "selected_d_psi_progress": selected_progress,
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
        "candidate_diversities": candidate_diversities,
        "segments": segment_records,
        "total_d_psi_realized_progress": sum(
            segment["d_psi_progress"] for segment in segment_records
        ),
        "timing_seconds": timing,
    }


def summarize_method(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    selected_rc = [
        value for record in records for value in record["selected_rc_scores"]
    ]
    selected_progress = [
        value for record in records for value in record["selected_d_psi_progress"]
    ]
    diversity = [
        value for record in records for value in record["candidate_diversities"]
    ]
    distance_progress = [
        segment["euclidean_target_distance_progress"]
        for record in records
        for segment in record["segments"]
    ]
    latent_progress = [
        segment["d_psi_progress"]
        for record in records
        for segment in record["segments"]
    ]
    attempts = sum(record["high_level_attempts"] for record in records)
    covered = sum(record["covered_high_level_attempts"] for record in records)
    fallback = sum(record["fallback_model_steps"] for record in records)
    return {
        "episode_count": len(records),
        "success_rate": len(successes) / len(records),
        "completion_env_steps": (
            distribution_summary(
                [record["completion_env_steps"] for record in successes]
            )
            if successes
            else None
        ),
        "fallback_rate": fallback / attempts if attempts else 0.0,
        "candidate_coverage": covered / attempts if attempts else None,
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
        "total_d_psi_realized_progress": distribution_summary(
            [record["total_d_psi_realized_progress"] for record in records]
        ),
        "selected_rc_score": distribution_summary(selected_rc) if selected_rc else None,
        "selected_d_psi_progress": (
            distribution_summary(selected_progress) if selected_progress else None
        ),
        "candidate_pairwise_diversity": (
            distribution_summary(diversity) if diversity else None
        ),
        "euclidean_target_distance_progress_diagnostic": (
            distribution_summary(distance_progress) if distance_progress else None
        ),
        "segment_d_psi_progress": (
            distribution_summary(latent_progress) if latent_progress else None
        ),
        "timing_seconds_total": {
            key: sum(record["timing_seconds"][key] for record in records)
            for key in ("generator", "rc", "d_psi", "cem")
        },
    }


def bootstrap_mean_95_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "bootstrap_unit": "episode",
    }


def across_seed_paired_bootstrap(
    seed_results: list[dict[str, Any]],
    *,
    eval_budget_env_steps: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(seed_results) < 3:
        raise ValueError("formal Stage 4 comparison requires at least 3 seeds")
    metric_specs = {
        "task_success": (lambda record: float(record["success"]), "higher_is_better"),
        "budgeted_completion_env_steps": (
            lambda record: float(
                record["completion_env_steps"]
                if record["completion_env_steps"] is not None
                else eval_budget_env_steps
            ),
            "lower_is_better",
        ),
        "d_psi_realized_progress": (
            lambda record: float(record["total_d_psi_realized_progress"]),
            "higher_is_better",
        ),
        "fallback_rate": (
            lambda record: float(record["fallback_rate"]),
            "lower_is_better",
        ),
        "candidate_coverage": (
            lambda record: (
                None
                if record["candidate_coverage"] is None
                else float(record["candidate_coverage"])
            ),
            "higher_is_better",
        ),
    }
    records_by_method_episode: dict[str, dict[int, list[dict[str, Any]]]] = {
        method: {} for method in METHODS
    }
    all_records = []
    for seed_result in seed_results:
        all_records.extend(seed_result["rollouts"])
        for record in seed_result["rollouts"]:
            records_by_method_episode[record["method"]].setdefault(
                int(record["episode_index"]), []
            ).append(record)

    rng = np.random.default_rng(seed)
    method_intervals: dict[str, Any] = {}
    episode_metric_values: dict[str, dict[str, dict[int, float]]] = {}
    for method in METHODS:
        episode_metric_values[method] = {}
        method_intervals[method] = {}
        for metric, (extract, direction) in metric_specs.items():
            per_episode = {}
            for episode_index, records in records_by_method_episode[method].items():
                values = [extract(record) for record in records]
                if all(value is not None for value in values):
                    per_episode[episode_index] = float(np.mean(values))
            episode_metric_values[method][metric] = per_episode
            if per_episode:
                interval = bootstrap_mean_95_ci(
                    np.asarray(list(per_episode.values())),
                    samples=samples,
                    rng=rng,
                )
                interval["direction"] = direction
                method_intervals[method][metric] = interval

    complete_method = "stochastic32_rc_dpsi"
    paired_comparisons = {}
    for baseline in METHODS:
        if baseline == complete_method:
            continue
        comparison = {}
        for metric, (_, direction) in metric_specs.items():
            complete_values = episode_metric_values[complete_method][metric]
            baseline_values = episode_metric_values[baseline][metric]
            common = sorted(set(complete_values) & set(baseline_values))
            if not common:
                continue
            differences = np.asarray(
                [complete_values[index] - baseline_values[index] for index in common]
            )
            interval = bootstrap_mean_95_ci(
                differences,
                samples=samples,
                rng=rng,
            )
            interval.update(
                {
                    "difference": "stochastic32_rc_dpsi minus baseline",
                    "direction": direction,
                    "paired_episode_count": len(common),
                    "generator_seeds_averaged_within_episode": len(seed_results),
                }
            )
            comparison[metric] = interval
        paired_comparisons[baseline] = comparison

    return {
        "generator_seed_count": len(seed_results),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "seed_aggregation": "average seeds within each episode before bootstrap",
        "method_summaries_all_seed_episode_rollouts": {
            method: summarize_method(
                [record for record in all_records if record["method"] == method]
            )
            for method in METHODS
        },
        "method_episode_bootstrap_95_ci": method_intervals,
        "paired_bootstrap_95_ci": paired_comparisons,
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
    if args.num_candidates != 32 or args.num_episodes < 1:
        raise ValueError("formal Stage 4 requires N=32 and positive episodes")
    if args.eval_budget_env_steps < 1:
        raise ValueError("evaluation budget must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    if not 1 <= args.topk <= args.num_samples:
        raise ValueError("topk must be in [1, num-samples]")

    stage2_protocol = load_stage2_protocol(args.stage2_report)
    stage3_protocol = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(
        stage2_protocol,
        stage3_protocol,
        checkpoint_label="Stage 3 progress ranker",
    )
    cache = load_latent_cache(args.latent_cache)
    _, _, test_episodes = split_cached_episodes(cache)
    dataset_path = args.cache_dir.expanduser().resolve() / args.dataset
    with h5py.File(dataset_path, "r") as handle:
        eligible_episodes = demonstrations_within_budget(
            handle,
            test_episodes,
            budget_env_steps=args.eval_budget_env_steps,
        )
        if args.num_episodes > len(eligible_episodes):
            raise ValueError(
                f"only {len(eligible_episodes)} test demonstrations finish within "
                f"the {args.eval_budget_env_steps}-step evaluation budget"
            )
        selected_episodes = select_episode_records(
            eligible_episodes, count=args.num_episodes
        )
        environment_records = [
            load_environment_record(handle, episode) for episode in selected_episodes
        ]

    device = torch.device(args.device)
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
    checkpoint_list = checkpoint_paths(args.training_report)
    all_seed_results = []
    for checkpoint_path in checkpoint_list:
        stage4_protocol = load_checkpoint_protocol(checkpoint_path, stage=4)
        assert_protocol_consistency(
            stage2_protocol,
            stage4_protocol,
            checkpoint_label=f"Stage 4 generator {checkpoint_path}",
        )
        generator, checkpoint = load_generator_checkpoint(
            checkpoint_path, device=device
        )
        generator.requires_grad_(False)
        rollout_records = []
        for episode_position, record in enumerate(environment_records):
            for method in METHODS:
                result = run_rollout(
                    method,
                    generator,
                    progress_ranker,
                    adapter,
                    record,
                    num_candidates=args.num_candidates,
                    eval_budget_env_steps=args.eval_budget_env_steps,
                    flat_horizon=args.flat_planning_horizon_model_steps,
                    eta_r=stage2_protocol["eta_r"],
                    env_seed=args.env_seed,
                    cem_seed=args.cem_seed + episode_position,
                    dropout_seed=(
                        args.dropout_seed
                        + int(checkpoint["seed"]) * 10000
                        + record["episode_index"]
                    ),
                    max_replay_pixel_diff=args.max_replay_pixel_diff,
                    device=device,
                )
                rollout_records.append(result)
                partial = {
                    "stage4_closed_loop_complete": False,
                    "current_seed": int(checkpoint["seed"]),
                    "completed_rollouts": len(rollout_records),
                    "completed_seed_results": all_seed_results,
                    "current_seed_rollouts": rollout_records,
                }
                atomic_write_json(partial, args.output)
                print(
                    f"seed={checkpoint['seed']} episode={record['episode_index']} "
                    f"method={method} success={result['success']} "
                    f"steps={result['env_steps']}",
                    flush=True,
                )
        summaries = {
            method: summarize_method(
                [record for record in rollout_records if record["method"] == method]
            )
            for method in METHODS
        }
        all_seed_results.append(
            {
                "generator_seed": int(checkpoint["seed"]),
                "generator_checkpoint": str(checkpoint_path),
                "method_summaries": summaries,
                "rollouts": rollout_records,
            }
        )

    across_seeds = across_seed_paired_bootstrap(
        all_seed_results,
        eval_budget_env_steps=args.eval_budget_env_steps,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report = {
        "stage4_closed_loop_complete": True,
        "protocol": {
            "test_episode_range": {
                "start_inclusive": 5000,
                "end_exclusive": 10000,
            },
            "episode_indices": [
                record["episode_index"] for record in environment_records
            ],
            "test_successful_demonstration_count": len(test_episodes),
            "demonstrations_within_evaluation_budget": len(eligible_episodes),
            "selected_demonstration_env_steps": [
                record["demonstration_env_steps"] for record in environment_records
            ],
            "episode_eligibility": (
                "successful test demonstration length is no greater than "
                "eval_budget_env_steps"
            ),
            "methods": list(METHODS),
            "tau_model_steps": TAU_MODEL_STEPS,
            "model_step_env_steps": MODEL_STEP_ENV_STEPS,
            "subgoal_horizon_sequence": [3, 2, 1],
            "execution_horizon_model_steps": 1,
            "subgoal_fixed_within_segment": True,
            "eta_r": stage2_protocol["eta_r"],
            "eta_r_source": stage2_protocol["report_path"],
            "num_candidates": args.num_candidates,
            "eval_budget_env_steps": args.eval_budget_env_steps,
            "flat_planning_horizon_model_steps": (
                args.flat_planning_horizon_model_steps
            ),
            "fallback": "flat z_T for one model step, then retry high level",
            "direct_goal_precheck": (
                "include z_T in the complete selector when RC passes and "
                "D_psi progress is positive"
            ),
            "high_level_goal": "z_T = E(o_T), the actual terminal observation",
            "final_success": "official TwoRoom termination at the dataset task target",
            "pos_target_model_input": False,
            "custom_subgoal_completion_radius_used": False,
            "matched_cem_seed_rule": "cem_seed + episode_position",
        },
        "seed_results": all_seed_results,
        "across_generator_seeds": across_seeds,
    }
    atomic_write_json(report, args.output)
    print(f"report_path: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
