#!/usr/bin/env python3
"""ARCHIVED: reproduce Stage 4 local-RC/direct-goal ablations."""

from __future__ import annotations

import argparse
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
    load_generator_checkpoint,
    load_latent_cache,
    split_cached_episodes,
)
from tools.evaluate_stage4_candidates import checkpoint_paths, load_progress_ranker
from tools.evaluate_stage4_closed_loop import (
    ABLATION_DIRECT_ONLY_RC,
    ABLATION_METHODS,
    ABLATION_RC_NO_DIRECT,
    TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
    atomic_write_json,
    bootstrap_mean_95_ci,
    load_environment_record,
    run_rollout,
    summarize_method,
)


NO_RC_NO_DIRECT = "stochastic32_dpsi_no_rc"
RC_AND_DIRECT = "stochastic32_rc_dpsi"
FACTORIAL_METHODS = (
    NO_RC_NO_DIRECT,
    ABLATION_RC_NO_DIRECT,
    ABLATION_DIRECT_ONLY_RC,
    RC_AND_DIRECT,
)

# The original Stage 4 report predates explicit serialization of these values.
# Its documented command used the evaluator defaults below. This ablation fixes
# them instead of exposing overrides that could invalidate paired comparisons.
NUM_CANDIDATES = 32
NUM_SAMPLES = 300
CEM_ITERATIONS = 30
TOPK = 30
CEM_SEED = 4200
ENV_SEED = 42
DROPOUT_SEED = 20260809
FLAT_HORIZON_MODEL_STEPS = 5
REACHABILITY_COST_WEIGHT = 0.85
MAX_REPLAY_PIXEL_DIFF = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the two missing Stage 4 RC/direct-goal factorial cells "
            "on the exact episodes and checkpoints in the baseline report."
        )
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("outputs/stage4_closed_loop.json"),
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
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all paired inputs without loading RC-aux or running rollouts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage4_rc_direct_ablations.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text())


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"baseline {label} is {actual!r}, expected {expected!r}")


def validate_baseline(
    report: dict[str, Any],
    *,
    eta_r: float,
    checkpoints: list[Path],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    if report.get("stage4_closed_loop_complete") is not True:
        raise ValueError("baseline Stage 4 closed-loop report is incomplete")
    protocol = report["protocol"]
    require_equal(protocol["tau_model_steps"], TAU_MODEL_STEPS, "tau")
    require_equal(
        protocol["model_step_env_steps"],
        MODEL_STEP_ENV_STEPS,
        "model-step conversion",
    )
    require_equal(protocol["subgoal_horizon_sequence"], [3, 2, 1], "H_plan")
    require_equal(protocol["execution_horizon_model_steps"], 1, "H_exec")
    require_equal(protocol["subgoal_fixed_within_segment"], True, "fixed subgoal")
    require_equal(protocol["num_candidates"], NUM_CANDIDATES, "candidate count")
    require_equal(
        protocol["episode_horizon_env_steps"],
        TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
        "episode horizon",
    )
    require_equal(
        protocol["flat_planning_horizon_model_steps"],
        FLAT_HORIZON_MODEL_STEPS,
        "flat fallback horizon",
    )
    require_equal(
        protocol["matched_cem_seed_rule"],
        "cem_seed + episode_position",
        "CEM seed rule",
    )
    serialized_defaults = {
        "num_samples": NUM_SAMPLES,
        "cem_iterations": CEM_ITERATIONS,
        "topk": TOPK,
        "cem_seed": CEM_SEED,
        "env_seed": ENV_SEED,
        "dropout_seed": DROPOUT_SEED,
        "reachability_cost_weight": REACHABILITY_COST_WEIGHT,
        "max_replay_pixel_diff": MAX_REPLAY_PIXEL_DIFF,
    }
    for key, expected in serialized_defaults.items():
        if key in protocol:
            require_equal(protocol[key], expected, key)
    if not np.isclose(float(protocol["eta_r"]), eta_r, atol=1.0e-12, rtol=0.0):
        raise ValueError("baseline eta_R does not match the Stage 2 report")
    for method in (NO_RC_NO_DIRECT, RC_AND_DIRECT):
        if method not in protocol["methods"]:
            raise ValueError(f"baseline report is missing method {method}")

    episode_indices = [int(value) for value in protocol["episode_indices"]]
    if not 100 <= len(episode_indices) <= 200:
        raise ValueError("baseline must contain the formal 100-200 episode subset")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("baseline episode indices are not unique")

    seed_results = {
        int(seed_result["generator_seed"]): seed_result
        for seed_result in report["seed_results"]
    }
    if len(seed_results) < 3:
        raise ValueError("the ablation requires at least three generator seeds")
    expected_checkpoints = {path.expanduser().resolve() for path in checkpoints}
    baseline_checkpoints = {
        Path(result["generator_checkpoint"]).expanduser().resolve()
        for result in seed_results.values()
    }
    if baseline_checkpoints != expected_checkpoints:
        raise ValueError("baseline and training report checkpoint sets differ")

    expected_pairs = {
        (episode_index, method)
        for episode_index in episode_indices
        for method in (NO_RC_NO_DIRECT, RC_AND_DIRECT)
    }
    for seed, seed_result in seed_results.items():
        observed = [
            (int(record["episode_index"]), record["method"])
            for record in seed_result["rollouts"]
            if record["method"] in (NO_RC_NO_DIRECT, RC_AND_DIRECT)
        ]
        if len(observed) != len(set(observed)) or set(observed) != expected_pairs:
            raise ValueError(f"baseline seed {seed} lacks exact paired rollouts")
    return episode_indices, seed_results


def exact_test_episodes(
    test_episodes: list[dict[str, Any]], episode_indices: list[int]
) -> list[dict[str, Any]]:
    by_index = {int(item["episode_index"]): item for item in test_episodes}
    missing = [index for index in episode_indices if index not in by_index]
    if missing:
        raise ValueError(f"baseline episodes missing from latent cache: {missing[:5]}")
    return [by_index[index] for index in episode_indices]


def metric_specs(
    episode_horizon_env_steps: int,
) -> dict[str, tuple[Callable[[dict[str, Any]], float | None], str]]:
    return {
        "task_success": (
            lambda record: float(record["success"]),
            "higher_is_better",
        ),
        "horizon_censored_completion_env_steps": (
            lambda record: float(
                record["completion_env_steps"]
                if record["completion_env_steps"] is not None
                else episode_horizon_env_steps
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


def factorial_bootstrap(
    seed_results: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(seed_results) < 3:
        raise ValueError("factorial comparison requires at least three seeds")
    specs = metric_specs(TWOROOM_OFFICIAL_MAX_EPISODE_STEPS)
    records_by_method_episode: dict[str, dict[int, list[dict[str, Any]]]] = {
        method: {} for method in FACTORIAL_METHODS
    }
    all_records = []
    for seed_result in seed_results:
        all_records.extend(seed_result["rollouts"])
        for record in seed_result["rollouts"]:
            method = record["method"]
            records_by_method_episode[method].setdefault(
                int(record["episode_index"]), []
            ).append(record)

    seed_count = len(seed_results)
    episode_values: dict[str, dict[str, dict[int, float]]] = {
        method: {} for method in FACTORIAL_METHODS
    }
    intervals: dict[str, dict[str, Any]] = {
        method: {} for method in FACTORIAL_METHODS
    }
    rng = np.random.default_rng(seed)
    for method in FACTORIAL_METHODS:
        for metric, (extract, direction) in specs.items():
            per_episode = {}
            for episode_index, records in records_by_method_episode[method].items():
                if len(records) != seed_count:
                    raise ValueError(
                        f"{method} episode {episode_index} does not have "
                        f"exactly {seed_count} seed records"
                    )
                values = [extract(record) for record in records]
                if all(value is not None for value in values):
                    per_episode[episode_index] = float(np.mean(values))
            episode_values[method][metric] = per_episode
            if per_episode:
                interval = bootstrap_mean_95_ci(
                    np.asarray(list(per_episode.values())),
                    samples=samples,
                    rng=rng,
                )
                interval["direction"] = direction
                intervals[method][metric] = interval

    comparisons = {
        "generated_rc_effect_without_direct_goal": (
            ABLATION_RC_NO_DIRECT,
            NO_RC_NO_DIRECT,
        ),
        "generated_rc_effect_with_direct_goal": (
            RC_AND_DIRECT,
            ABLATION_DIRECT_ONLY_RC,
        ),
        "direct_goal_effect_without_generated_rc": (
            ABLATION_DIRECT_ONLY_RC,
            NO_RC_NO_DIRECT,
        ),
        "direct_goal_effect_with_generated_rc": (
            RC_AND_DIRECT,
            ABLATION_RC_NO_DIRECT,
        ),
    }
    paired = {}
    for label, (treatment, control) in comparisons.items():
        paired[label] = {}
        for metric, (_, direction) in specs.items():
            treatment_values = episode_values[treatment][metric]
            control_values = episode_values[control][metric]
            common = sorted(set(treatment_values) & set(control_values))
            if not common:
                continue
            differences = np.asarray(
                [
                    treatment_values[index] - control_values[index]
                    for index in common
                ]
            )
            interval = bootstrap_mean_95_ci(
                differences,
                samples=samples,
                rng=rng,
            )
            interval.update(
                {
                    "difference": f"{treatment} minus {control}",
                    "direction": direction,
                    "favorable_difference_sign": (
                        "positive" if direction == "higher_is_better" else "negative"
                    ),
                    "paired_episode_count": len(common),
                    "generator_seeds_averaged_within_episode": seed_count,
                }
            )
            paired[label][metric] = interval

    return {
        "factorial_cells": {
            NO_RC_NO_DIRECT: {
                "generated_candidate_rc_filter": False,
                "direct_goal_precheck": False,
                "source": "baseline_report",
            },
            ABLATION_RC_NO_DIRECT: {
                "generated_candidate_rc_filter": True,
                "direct_goal_precheck": False,
                "source": "ablation_run",
            },
            ABLATION_DIRECT_ONLY_RC: {
                "generated_candidate_rc_filter": False,
                "direct_goal_precheck": True,
                "source": "ablation_run",
            },
            RC_AND_DIRECT: {
                "generated_candidate_rc_filter": True,
                "direct_goal_precheck": True,
                "source": "baseline_report",
            },
        },
        "generator_seed_count": seed_count,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "seed_aggregation": "average seeds within episode before bootstrap",
        "method_summaries_all_seed_episode_rollouts": {
            method: summarize_method(
                [record for record in all_records if record["method"] == method]
            )
            for method in FACTORIAL_METHODS
        },
        "method_episode_bootstrap_95_ci": intervals,
        "paired_factorial_effects_95_ci": paired,
    }


def build_protocol(
    args: argparse.Namespace,
    *,
    episode_indices: list[int],
    eta_r: float,
    generator_seeds: list[int],
) -> dict[str, Any]:
    return {
        "baseline_report": str(args.baseline_report.expanduser().resolve()),
        "training_report": str(args.training_report.expanduser().resolve()),
        "stage2_report": str(args.stage2_report.expanduser().resolve()),
        "latent_cache": str(args.latent_cache.expanduser().resolve()),
        "progress_ranker": str(args.progress_ranker.expanduser().resolve()),
        "policy": args.policy,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "dataset": args.dataset,
        "episode_indices": episode_indices,
        "generator_seeds": generator_seeds,
        "new_methods": list(ABLATION_METHODS),
        "factorial_methods": list(FACTORIAL_METHODS),
        "tau_model_steps": TAU_MODEL_STEPS,
        "model_step_env_steps": MODEL_STEP_ENV_STEPS,
        "subgoal_horizon_sequence": [3, 2, 1],
        "execution_horizon_model_steps": 1,
        "subgoal_fixed_within_segment": True,
        "eta_r": eta_r,
        "num_candidates": NUM_CANDIDATES,
        "episode_horizon_env_steps": TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
        "num_samples": NUM_SAMPLES,
        "cem_iterations": CEM_ITERATIONS,
        "topk": TOPK,
        "cem_seed": CEM_SEED,
        "env_seed": ENV_SEED,
        "dropout_seed": DROPOUT_SEED,
        "flat_planning_horizon_model_steps": FLAT_HORIZON_MODEL_STEPS,
        "reachability_cost_weight": REACHABILITY_COST_WEIGHT,
        "max_replay_pixel_diff": MAX_REPLAY_PIXEL_DIFF,
        "cem_seed_rule": "cem_seed + episode_position",
        "dropout_seed_rule": (
            "dropout_seed + generator_seed * 10000 + episode_index"
        ),
        "fallback": "flat z_T for one model step, then retry high level",
        "legacy_baseline_parameter_source": (
            "documented evaluate_stage4_closed_loop.py defaults; the existing "
            "baseline report did not serialize numeric CEM/dropout/env settings"
        ),
    }


def resume_seed_results(
    output: Path,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    path = output.expanduser().resolve()
    if not path.exists():
        return [], False
    report = read_json(path)
    if report.get("protocol") != protocol:
        raise ValueError("existing ablation output uses a different protocol")
    if report.get("stage4_ablation_complete") is True:
        return report["seed_results"], True
    return report.get("seed_results", []), False


def partial_report(
    protocol: dict[str, Any], seed_results: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "stage4_ablation_complete": False,
        "protocol": protocol,
        "seed_results": seed_results,
        "completed_new_rollouts": sum(
            len(seed_result["rollouts"]) for seed_result in seed_results
        ),
        "expected_new_rollouts": (
            len(protocol["generator_seeds"])
            * len(protocol["episode_indices"])
            * len(ABLATION_METHODS)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")

    stage2_protocol = load_stage2_protocol(args.stage2_report)
    stage3_protocol = load_checkpoint_protocol(args.progress_ranker, stage=3)
    assert_protocol_consistency(
        stage2_protocol,
        stage3_protocol,
        checkpoint_label="Stage 3 progress ranker",
    )
    checkpoints = checkpoint_paths(args.training_report)
    baseline_report = read_json(args.baseline_report)
    episode_indices, baseline_by_seed = validate_baseline(
        baseline_report,
        eta_r=stage2_protocol["eta_r"],
        checkpoints=checkpoints,
    )

    checkpoint_seeds = []
    for path in checkpoints:
        checkpoint = torch.load(
            path.expanduser().resolve(), map_location="cpu", weights_only=False
        )
        checkpoint_seeds.append(int(checkpoint["seed"]))
    if set(checkpoint_seeds) != set(baseline_by_seed):
        raise ValueError("generator seeds differ between baseline and checkpoints")
    protocol = build_protocol(
        args,
        episode_indices=episode_indices,
        eta_r=stage2_protocol["eta_r"],
        generator_seeds=checkpoint_seeds,
    )
    saved_seed_results, already_complete = resume_seed_results(
        args.output, protocol
    )
    if already_complete:
        print(f"report already complete: {args.output.expanduser().resolve()}")
        return 0

    cache = load_latent_cache(args.latent_cache)
    _, _, test_episodes = split_cached_episodes(cache)
    selected_episodes = exact_test_episodes(test_episodes, episode_indices)
    dataset_path = args.cache_dir.expanduser().resolve() / args.dataset
    with h5py.File(dataset_path, "r") as handle:
        environment_records = [
            load_environment_record(handle, episode) for episode in selected_episodes
        ]
    if args.validate_only:
        print(
            f"validated {len(environment_records)} episodes and "
            f"{len(checkpoints)} generator checkpoints"
        )
        return 0

    device = torch.device(args.device)
    planner_config = RCAuxPlannerConfig(
        planning_horizon_model_steps=FLAT_HORIZON_MODEL_STEPS,
        execution_horizon_model_steps=1,
        num_samples=NUM_SAMPLES,
        n_steps=CEM_ITERATIONS,
        topk=TOPK,
        seed=CEM_SEED,
        warm_start=True,
    )
    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=args.cache_dir.expanduser().resolve(),
        device=device,
        planner_config=planner_config,
        use_reachability_cost=True,
        reachability_cost_weight=REACHABILITY_COST_WEIGHT,
    )
    adapter.model.interpolate_pos_encoding = True
    progress_ranker, _ = load_progress_ranker(args.progress_ranker, device=device)
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("E_theta, F_theta, and R_phi must remain frozen")
    if any(parameter.requires_grad for parameter in progress_ranker.parameters()):
        raise RuntimeError("D_psi must remain frozen")

    saved_by_seed = {
        int(item["generator_seed"]): item for item in saved_seed_results
    }
    seed_results = []
    for checkpoint_path, generator_seed in zip(checkpoints, checkpoint_seeds):
        saved = saved_by_seed.get(generator_seed, {})
        saved_checkpoint = saved.get("generator_checkpoint")
        if saved_checkpoint is not None and (
            Path(saved_checkpoint).expanduser().resolve()
            != checkpoint_path.expanduser().resolve()
        ):
            raise ValueError(f"saved seed {generator_seed} checkpoint differs")
        seed_results.append(
            {
                "generator_seed": generator_seed,
                "generator_checkpoint": str(checkpoint_path.expanduser().resolve()),
                "rollouts": list(saved.get("rollouts", [])),
            }
        )

    for checkpoint_path, seed_result in zip(checkpoints, seed_results):
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
        generator_seed = int(checkpoint["seed"])
        if generator_seed != int(seed_result["generator_seed"]):
            raise ValueError("checkpoint seed changed between validation and loading")
        rollout_records = seed_result["rollouts"]
        completed_keys = {
            (int(record["episode_index"]), record["method"])
            for record in rollout_records
        }
        expected_keys = {
            (episode_index, method)
            for episode_index in episode_indices
            for method in ABLATION_METHODS
        }
        if not completed_keys <= expected_keys:
            raise ValueError(f"saved seed {generator_seed} has unknown rollouts")
        if len(completed_keys) != len(rollout_records):
            raise ValueError(f"saved seed {generator_seed} has duplicate rollouts")
        for episode_position, record in enumerate(environment_records):
            for method in ABLATION_METHODS:
                key = (int(record["episode_index"]), method)
                if key in completed_keys:
                    continue
                result = run_rollout(
                    method,
                    generator,
                    progress_ranker,
                    adapter,
                    record,
                    num_candidates=NUM_CANDIDATES,
                    episode_horizon_env_steps=TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
                    flat_horizon=FLAT_HORIZON_MODEL_STEPS,
                    eta_r=stage2_protocol["eta_r"],
                    env_seed=ENV_SEED,
                    cem_seed=CEM_SEED + episode_position,
                    dropout_seed=(
                        DROPOUT_SEED
                        + generator_seed * 10000
                        + int(record["episode_index"])
                    ),
                    max_replay_pixel_diff=MAX_REPLAY_PIXEL_DIFF,
                    device=device,
                )
                rollout_records.append(result)
                completed_keys.add(key)
                atomic_write_json(partial_report(protocol, seed_results), args.output)
                print(
                    f"seed={generator_seed} episode={record['episode_index']} "
                    f"method={method} success={result['success']} "
                    f"steps={result['env_steps']}",
                    flush=True,
                )
        seed_result["method_summaries"] = {
            method: summarize_method(
                [record for record in rollout_records if record["method"] == method]
            )
            for method in ABLATION_METHODS
        }
        atomic_write_json(partial_report(protocol, seed_results), args.output)

    expected_per_seed = len(episode_indices) * len(ABLATION_METHODS)
    for seed_result in seed_results:
        if len(seed_result["rollouts"]) != expected_per_seed:
            raise RuntimeError(
                f"seed {seed_result['generator_seed']} has incomplete ablation results"
            )

    combined_seed_results = []
    for new_result in seed_results:
        generator_seed = int(new_result["generator_seed"])
        baseline_rollouts = [
            record
            for record in baseline_by_seed[generator_seed]["rollouts"]
            if record["method"] in (NO_RC_NO_DIRECT, RC_AND_DIRECT)
        ]
        combined_seed_results.append(
            {
                "generator_seed": generator_seed,
                "rollouts": baseline_rollouts + new_result["rollouts"],
            }
        )

    report = {
        "stage4_ablation_complete": True,
        "protocol": protocol,
        "seed_results": seed_results,
        "factorial_analysis": factorial_bootstrap(
            combined_seed_results,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
    }
    atomic_write_json(report, args.output)
    print(f"report_path: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
