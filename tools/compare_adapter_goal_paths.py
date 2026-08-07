#!/usr/bin/env python3
"""Compare RC-aux image-goal and latent-goal planning end to end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, RCAuxPlannerConfig, TWOROOM_PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two full CEM plans with a real checkpoint and compare "
            "o_G -> plan_to_image against E(o_G) -> plan_to_latent."
        )
    )
    parser.add_argument(
        "--policy",
        default="tworoom_rcaux/rcaux_tworoom",
        help="Official policy name or checkpoint path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/sxw/work/datasets/stable-wm"),
    )
    parser.add_argument("--dataset", default="tworoom.h5")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--observation-offset", type=int, default=0)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planning-horizon-model-steps", type=int, default=5)
    parser.add_argument("--execution-horizon-model-steps", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--reachability-cost-weight", type=float, default=0.85)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with status 1 unless every compared planner output is allclose.",
    )
    return parser.parse_args()


def load_images(
    dataset_path: Path,
    episode_index: int,
    observation_offset: int,
    goal_offset: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with h5py.File(dataset_path, "r") as handle:
        episode_count = len(handle["ep_offset"])
        if not 0 <= episode_index < episode_count:
            raise ValueError(
                f"episode-index must be in [0, {episode_count - 1}]"
            )
        episode_start = int(handle["ep_offset"][episode_index])
        episode_length = int(handle["ep_len"][episode_index])
        for name, offset in (
            ("observation-offset", observation_offset),
            ("goal-offset", goal_offset),
        ):
            if not 0 <= offset < episode_length:
                raise ValueError(
                    f"{name} must be in [0, {episode_length - 1}] for "
                    f"episode {episode_index}"
                )
        observation_row = episode_start + observation_offset
        goal_row = episode_start + goal_offset
        observation = np.asarray(handle["pixels"][observation_row])
        goal_image = np.asarray(handle["pixels"][goal_row])
    rows = {
        "episode_index": episode_index,
        "episode_start_row": episode_start,
        "observation_row": observation_row,
        "goal_row": goal_row,
    }
    return observation, goal_image, rows


def fit_official_action_scaler(cache_dir: Path) -> StandardScaler:
    dataset = swm.data.HDF5Dataset(
        TWOROOM_PROFILE.dataset_name,
        keys_to_cache=["action"],
        cache_dir=cache_dir,
    )
    actions = dataset.get_col_data("action")
    actions = actions[~np.isnan(actions).any(axis=1)]
    return StandardScaler().fit(actions)


def comparison(
    image_value: Any,
    latent_value: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    image_array = np.asarray(image_value)
    latent_array = np.asarray(latent_value)
    if image_array.shape != latent_array.shape:
        return {
            "image_shape": list(image_array.shape),
            "latent_shape": list(latent_array.shape),
            "exact_equal": False,
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    difference = np.abs(image_array - latent_array)
    return {
        "image_shape": list(image_array.shape),
        "latent_shape": list(latent_array.shape),
        "exact_equal": bool(np.array_equal(image_array, latent_array)),
        "allclose": bool(
            np.allclose(image_array, latent_array, atol=atol, rtol=rtol)
        ),
        "max_abs_diff": float(difference.max(initial=0.0)),
        "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
    }


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_path = cache_dir / args.dataset
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )

    observation, goal_image, rows = load_images(
        dataset_path,
        args.episode_index,
        args.observation_offset,
        args.goal_offset,
    )
    action_scaler = fit_official_action_scaler(cache_dir)
    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=args.planning_horizon_model_steps,
        execution_horizon_model_steps=args.execution_horizon_model_steps,
        num_samples=args.num_samples,
        n_steps=args.cem_iterations,
        topk=args.topk,
        seed=args.seed,
        warm_start=False,
    )

    model = swm.policy.AutoCostModel(args.policy, cache_dir=cache_dir)
    model.interpolate_pos_encoding = True
    common = {
        "profile": TWOROOM_PROFILE,
        "planner_config": planner_config,
        "device": args.device,
        "cache_dir": cache_dir,
        "action_scaler": action_scaler,
        "use_reachability_cost": True,
        "reachability_cost_weight": args.reachability_cost_weight,
    }
    image_adapter = RCAuxAdapter(model, **common)
    latent_adapter = RCAuxAdapter(model, **common)

    goal_latent = latent_adapter.encode_observation(goal_image)[:, -1]
    image_result = image_adapter.plan_to_image(observation, goal_image)
    latent_result = latent_adapter.plan_to_latent(observation, goal_latent)

    outputs = {
        "normalized_action_blocks": comparison(
            image_result.normalized_action_blocks.numpy(),
            latent_result.normalized_action_blocks.numpy(),
            atol=args.atol,
            rtol=args.rtol,
        ),
        "planned_actions_env_steps": comparison(
            image_result.planned_actions_env_steps,
            latent_result.planned_actions_env_steps,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "actions_to_execute_env_steps": comparison(
            image_result.actions_to_execute_env_steps,
            latent_result.actions_to_execute_env_steps,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "final_costs": comparison(
            image_result.diagnostics.final_costs,
            latent_result.diagnostics.final_costs,
            atol=args.atol,
            rtol=args.rtol,
        ),
    }
    equivalent = all(value["allclose"] for value in outputs.values())
    report = {
        "equivalent": equivalent,
        "policy": args.policy,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "rows": rows,
        "goal_image_shape": list(goal_image.shape),
        "goal_latent_shape": list(goal_latent.shape),
        "goal_latent_dtype": str(goal_latent.dtype),
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
            "warm_start": False,
            "reachability_cost_weight": args.reachability_cost_weight,
        },
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "outputs": outputs,
        "planning_time_seconds": {
            "image_goal": image_result.diagnostics.planning_time_seconds,
            "latent_goal": latent_result.diagnostics.planning_time_seconds,
        },
    }
    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json + "\n")
        print(f"report_path: {output_path}")
    return int(args.fail_on_mismatch and not equivalent)


if __name__ == "__main__":
    raise SystemExit(main())
