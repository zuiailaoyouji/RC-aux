#!/usr/bin/env python3
"""Drive TwoRoom toward a real future observation encoded as a latent goal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import stable_worldmodel  # noqa: F401 - registers the TwoRoom environment
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, RCAuxPlannerConfig, TWOROOM_PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set g=E(o_{t+k}) from the real TwoRoom dataset and use only that "
            "latent goal for closed-loop environment control."
        )
    )
    parser.add_argument("--policy", default="tworoom_rcaux/rcaux_tworoom")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/sxw/work/datasets/stable-wm"),
    )
    parser.add_argument("--dataset", default="tworoom.h5")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--t-env-step", type=int, default=0)
    parser.add_argument("--future-k-env-steps", type=int, default=25)
    parser.add_argument("--eval-budget-env-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planning-horizon-model-steps", type=int, default=5)
    parser.add_argument("--execution-horizon-model-steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--reachability-cost-weight", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/latent_subgoal_oracle.json"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("outputs/latent_subgoal_oracle.mp4"),
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--max-replay-pixel-diff", type=int, default=1)
    parser.add_argument(
        "--require-replay-match",
        action="store_true",
        help="Fail if environment re-rendering exceeds the pixel tolerance.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit with status 1 unless the environment reaches o_{t+k}'s state.",
    )
    return parser.parse_args()


def load_oracle_pair(
    dataset_path: Path,
    episode_index: int,
    t_env_step: int,
    future_k_env_steps: int,
) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as handle:
        episode_count = len(handle["ep_offset"])
        if not 0 <= episode_index < episode_count:
            raise ValueError(
                f"episode-index must be in [0, {episode_count - 1}]"
            )
        episode_start = int(handle["ep_offset"][episode_index])
        episode_length = int(handle["ep_len"][episode_index])
        future_env_step = t_env_step + future_k_env_steps
        if t_env_step < 0 or future_k_env_steps < 1:
            raise ValueError("t-env-step must be nonnegative and k must be positive")
        if future_env_step >= episode_length:
            raise ValueError(
                f"t+k={future_env_step} exceeds episode length {episode_length}"
            )

        source_row = episode_start + t_env_step
        target_row = episode_start + future_env_step
        return {
            "episode_index": episode_index,
            "episode_start_row": episode_start,
            "episode_length": episode_length,
            "source_row": source_row,
            "target_row": target_row,
            "t_env_step": t_env_step,
            "future_k_env_steps": future_k_env_steps,
            "source_image": np.asarray(handle["pixels"][source_row]),
            "target_image": np.asarray(handle["pixels"][target_row]),
            "source_state": np.asarray(
                handle["proprio"][source_row], dtype=np.float32
            ),
            "target_state": np.asarray(
                handle["proprio"][target_row], dtype=np.float32
            ),
        }


def frame_comparison(dataset_frame: np.ndarray, rendered: np.ndarray) -> dict:
    difference = np.abs(
        dataset_frame.astype(np.int16) - rendered.astype(np.int16)
    )
    return {
        "exact_equal": bool(np.array_equal(dataset_frame, rendered)),
        "max_abs_pixel_diff": int(difference.max(initial=0)),
        "mean_abs_pixel_diff": float(difference.mean()),
    }


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_path = cache_dir / args.dataset
    oracle = load_oracle_pair(
        dataset_path,
        args.episode_index,
        args.t_env_step,
        args.future_k_env_steps,
    )

    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=args.planning_horizon_model_steps,
        execution_horizon_model_steps=args.execution_horizon_model_steps,
        num_samples=args.num_samples,
        n_steps=args.cem_iterations,
        topk=args.topk,
        seed=args.seed,
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
    goal_latent_btd = adapter.encode_observation(oracle["target_image"])
    goal_latent_bd = goal_latent_btd[:, -1]

    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        max_episode_steps=args.eval_budget_env_steps,
        disable_env_checker=True,
    )
    frames: list[np.ndarray] = []
    positions: list[list[float]] = []
    distances: list[float] = []
    executed_actions: list[list[float]] = []
    replans: list[dict[str, Any]] = []
    success = False
    truncated = False

    try:
        env.reset(seed=args.seed)
        base_env = env.unwrapped
        base_env._set_state(oracle["source_state"])
        base_env._set_goal_state(oracle["target_state"])

        source_render = env.render()
        target_render = (
            base_env._render_frame(
                agent_pos=torch.as_tensor(oracle["target_state"])
            )
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        source_replay = frame_comparison(
            oracle["source_image"], source_render
        )
        target_replay = frame_comparison(
            oracle["target_image"], target_render
        )
        replay_match = (
            source_replay["max_abs_pixel_diff"]
            <= args.max_replay_pixel_diff
            and target_replay["max_abs_pixel_diff"]
            <= args.max_replay_pixel_diff
        )
        if args.require_replay_match and not replay_match:
            raise RuntimeError(
                "Dataset frames exceed the environment replay pixel tolerance; "
                "the oracle rollout would not be controlled."
            )

        current_position = np.asarray(
            base_env.agent_position.cpu(), dtype=np.float32
        )
        current_distance = float(
            np.linalg.norm(current_position - oracle["target_state"])
        )
        positions.append(current_position.tolist())
        distances.append(current_distance)
        frames.append(np.hstack([source_render, oracle["target_image"]]))

        env_steps = 0
        while env_steps < args.eval_budget_env_steps and not success:
            current_image = env.render()
            current_latent = adapter.encode_observation(current_image)[:, -1]
            reachability = adapter.reachability(
                current_latent,
                goal_latent_bd,
                horizon_model_steps=(
                    adapter.max_reachability_horizon_model_steps
                ),
            )
            plan = adapter.plan_to_latent(current_image, goal_latent_bd)
            plan_record = plan.diagnostics.to_log_dict()
            plan_record["replan_index"] = len(replans)
            plan_record["env_step_before_execution"] = env_steps
            plan_record["distance_before_execution"] = current_distance
            plan_record["reachability_probability"] = float(
                reachability.item()
            )
            replans.append(plan_record)

            for action in plan.actions_to_execute_env_steps[0]:
                _, _, terminated, step_truncated, info = env.step(action)
                env_steps += 1
                success = success or bool(terminated)
                truncated = truncated or bool(step_truncated)
                current_position = np.asarray(
                    info["proprio"], dtype=np.float32
                )
                current_distance = float(info["distance_to_target"])
                positions.append(current_position.tolist())
                distances.append(current_distance)
                executed_actions.append(np.asarray(action).tolist())
                frames.append(
                    np.hstack([env.render(), oracle["target_image"]])
                )
                if success or truncated or env_steps >= args.eval_budget_env_steps:
                    break
            if truncated:
                break
    finally:
        env.close()

    initial_distance = distances[0]
    final_distance = distances[-1]
    initially_within_success_radius = initial_distance < 16.0
    oracle_validated = bool(
        replay_match and not initially_within_success_radius and success
    )
    report = {
        "oracle_validated": oracle_validated,
        "made_progress": bool(final_distance < initial_distance),
        "policy": args.policy,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "oracle": {
            "definition": "g = E(o_{t+k})",
            "episode_index": oracle["episode_index"],
            "source_row": oracle["source_row"],
            "target_row": oracle["target_row"],
            "t_env_step": oracle["t_env_step"],
            "future_k_env_steps": oracle["future_k_env_steps"],
            "source_state": oracle["source_state"].tolist(),
            "target_state": oracle["target_state"].tolist(),
            "goal_latent_shape": list(goal_latent_bd.shape),
            "goal_latent_dtype": str(goal_latent_bd.dtype),
        },
        "replay_validation": {
            "within_tolerance": replay_match,
            "max_replay_pixel_diff": args.max_replay_pixel_diff,
            "source": source_replay,
            "target": target_replay,
        },
        "planner": {
            "planning_horizon_model_steps": (
                args.planning_horizon_model_steps
            ),
            "execution_horizon_model_steps": (
                args.execution_horizon_model_steps
            ),
            "model_step_env_steps": TWOROOM_PROFILE.model_step_env_steps,
            "num_samples": args.num_samples,
            "cem_iterations": args.cem_iterations,
            "topk": args.topk,
            "seed": args.seed,
            "warm_start": True,
            "reachability_cost_weight": args.reachability_cost_weight,
        },
        "rollout": {
            "success": bool(success),
            "truncated": bool(truncated),
            "env_steps_executed": len(executed_actions),
            "replan_count": len(replans),
            "initially_within_success_radius": initially_within_success_radius,
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "minimum_distance": min(distances),
            "distance_reduction": initial_distance - final_distance,
            "positions": positions,
            "distances": distances,
            "executed_actions": executed_actions,
        },
        "replans": replans,
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, default=json_value)
    output_path.write_text(report_json + "\n")
    print(report_json)
    print(f"report_path: {output_path}")

    if args.video is not None:
        video_path = args.video.expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=args.video_fps)
        print(f"video_path: {video_path}")

    return int(args.require_success and not oracle_validated)


if __name__ == "__main__":
    raise SystemExit(main())
