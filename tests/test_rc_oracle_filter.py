import h5py
import numpy as np
import torch

from tools.evaluate_rc_oracle_filter import (
    CROSS_UNREACHABLE,
    OVER_BUDGET,
    WITHIN_BUDGET,
    build_candidate_pool,
    choose_candidates,
    paired_comparison,
    reachability_time_estimate,
    strategy_summary,
)


def test_candidate_pool_uses_successful_trajectory_terminal_goal(tmp_path):
    dataset_path = tmp_path / "oracle.h5"
    episode_length = 31
    episode_count = 4
    total = episode_length * episode_count
    proprio = np.zeros((total, 2), dtype=np.float32)
    pixels = np.zeros((total, 2, 2, 3), dtype=np.uint8)
    terminated = np.zeros(total, dtype=bool)
    for episode in range(episode_count):
        start = episode * episode_length
        proprio[start : start + episode_length, 0] = (
            episode * 100 + np.arange(episode_length) * 4
        )
        pixels[start : start + episode_length] = episode
        terminated[start + episode_length - 1] = episode != 1

    with h5py.File(dataset_path, "w") as handle:
        handle.create_dataset(
            "ep_offset",
            data=np.arange(episode_count) * episode_length,
        )
        handle.create_dataset(
            "ep_len",
            data=np.full(episode_count, episode_length),
        )
        handle.create_dataset("proprio", data=proprio)
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("terminated", data=terminated)

    with h5py.File(dataset_path, "r") as handle:
        pool = build_candidate_pool(
            handle,
            episode_index=0,
            t_env_step=0,
            tau_model_steps=3,
            cross_candidates=2,
            speed=1.0,
            success_radius=1.0,
        )
        failed_pool = build_candidate_pool(
            handle,
            episode_index=1,
            t_env_step=0,
            tau_model_steps=3,
            cross_candidates=2,
            speed=1.0,
            success_radius=1.0,
        )

    assert pool is not None
    assert pool["final_goal_row"] == episode_length - 1
    assert pool["final_goal_row"] != 5 * 5
    assert failed_pool is None
    categories = [candidate["category"] for candidate in pool["candidates"]]
    assert WITHIN_BUDGET in categories
    assert OVER_BUDGET in categories
    assert categories.count(CROSS_UNREACHABLE) == 2


def test_selection_strategies_are_derived_from_the_same_candidates():
    candidates = [
        {
            "local_rc_score": 0.8,
            "truncated_reachability_time": 5.0,
            "predicted_progress": 1.0,
        },
        {
            "local_rc_score": 0.4,
            "truncated_reachability_time": 1.0,
            "predicted_progress": 5.0,
        },
        {
            "local_rc_score": 0.7,
            "truncated_reachability_time": 3.0,
            "predicted_progress": 3.0,
        },
    ]

    selection = choose_candidates(candidates, eta_r=0.5, random_index=0)

    assert selection == {
        "rc_then_progress": 2,
        "random": 0,
        "progress_only": 1,
    }


def test_strategy_and_paired_statistics_treat_no_feasible_as_failure():
    trials = [
        {
            "selection": {
                "rc": None,
                "baseline": {
                    "category": WITHIN_BUDGET,
                    "actual_success": True,
                    "predicted_progress": 1.0,
                    "final_task_physical_progress": 2.0,
                    "final_task_reachability_time_progress": 0.5,
                },
            }
        },
        {
            "selection": {
                "rc": {
                    "category": WITHIN_BUDGET,
                    "actual_success": True,
                    "predicted_progress": 2.0,
                    "final_task_physical_progress": 3.0,
                    "final_task_reachability_time_progress": 1.0,
                },
                "baseline": {
                    "category": OVER_BUDGET,
                    "actual_success": False,
                    "predicted_progress": 3.0,
                    "final_task_physical_progress": -1.0,
                    "final_task_reachability_time_progress": -0.5,
                },
            }
        },
    ]

    summary = strategy_summary(trials, "rc")
    comparison = paired_comparison(trials, "rc", "baseline")

    assert summary["selected_count"] == 1
    assert summary["no_feasible_count"] == 1
    assert summary["actual_completion_rate"] == 0.5
    assert summary["mean_final_task_physical_progress"] == 1.5
    assert summary["mean_final_task_reachability_time_progress"] == 0.5
    assert comparison == {"left_wins": 1, "right_wins": 1, "ties": 0}


def test_reachability_time_estimate_integrates_one_minus_probability():
    class FakeAdapter:
        def reachability(self, source, target, horizon_model_steps):
            shape = np.broadcast_shapes(source.shape[:-1], target.shape[:-1])
            return torch.full(shape, horizon_model_steps / 10.0)

    probabilities, distance = reachability_time_estimate(
        FakeAdapter(),
        np.zeros((2, 4)),
        np.ones((2, 4)),
    )

    np.testing.assert_allclose(probabilities, [[0.1, 0.2, 0.3, 0.4, 0.5]] * 2)
    np.testing.assert_allclose(distance, [3.5, 3.5])
