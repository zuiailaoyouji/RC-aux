import h5py
import numpy as np

from tools.evaluate_rc_filter_only import (
    CROSS_TRAJECTORY_STRICT,
    SAME_TRAJECTORY_STRICT,
    WITHIN_BUDGET,
    build_strict_candidate_pool,
    select_calibration_threshold,
    summarize_split,
)


def test_strict_pool_is_balanced_and_negative_lower_bounds_exceed_budget(
    tmp_path,
):
    dataset_path = tmp_path / "strict_pool.h5"
    episode_length = 61
    episode_count = 4
    total = episode_length * episode_count
    proprio = np.zeros((total, 2), dtype=np.float32)
    pixels = np.zeros((total, 2, 2, 3), dtype=np.uint8)
    terminated = np.zeros(total, dtype=bool)
    pos_target = np.zeros((total, 2), dtype=np.float32)
    for episode in range(episode_count):
        start = episode * episode_length
        proprio[start : start + episode_length, 0] = (
            episode * 200 + np.arange(episode_length) * 4
        )
        terminated[start + episode_length - 1] = True

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
        handle.create_dataset("pos_target", data=pos_target)

    with h5py.File(dataset_path, "r") as handle:
        pool = build_strict_candidate_pool(
            handle,
            episode_index=0,
            episode_range=range(episode_count),
            t_env_step=0,
            tau_model_steps=3,
            candidates_per_category=2,
            speed=5.0,
            success_radius=16.0,
        )

    assert pool is not None
    categories = [candidate["category"] for candidate in pool["candidates"]]
    assert categories.count(WITHIN_BUDGET) == 2
    assert categories.count(SAME_TRAJECTORY_STRICT) == 2
    assert categories.count(CROSS_TRAJECTORY_STRICT) == 2
    for candidate in pool["candidates"]:
        if candidate["category"] == WITHIN_BUDGET:
            assert candidate["temporal_delta_model_steps"] <= 3
        else:
            assert candidate["lower_bound_env_steps"] > 15


def candidate(score, success, category=WITHIN_BUDGET):
    return {
        "category": category,
        "local_rc_score": score,
        "actual_success_rate": float(success),
        "executions": [{"actual_success": bool(success)}],
    }


def test_threshold_is_selected_from_calibration_outcomes_under_constraints():
    trials = [
        {
            "candidates": [
                candidate(0.9, True),
                candidate(0.7, True),
                candidate(0.4, False),
                candidate(0.1, False),
            ]
        }
    ]

    threshold, constraints_met, _ = select_calibration_threshold(
        trials,
        max_fpr=0.0,
        min_precision=1.0,
    )

    assert constraints_met
    assert threshold == 0.7


def test_rc_pass_set_uses_uniform_expected_completion_not_one_random_draw():
    trials = [
        {
            "candidates": [
                candidate(0.8, True),
                candidate(0.2, False, CROSS_TRAJECTORY_STRICT),
            ]
        },
        {
            "candidates": [
                candidate(0.9, True),
                candidate(0.1, False, SAME_TRAJECTORY_STRICT),
            ]
        },
    ]

    summary = summarize_split(
        trials,
        threshold=0.5,
        bootstrap_samples=1000,
        bootstrap_seed=7,
    )

    assert summary["uniform_all_pool"]["expected_completion_rate"] == 0.5
    assert (
        summary["uniform_rc_pass_set"][
            "conditional_expected_completion_rate"
        ]
        == 1.0
    )
    assert summary["trial_coverage"] == 1.0
    assert summary["paired_conditional_improvement"] == {
        "mean": 0.5,
        "lower_95": 0.5,
        "upper_95": 0.5,
    }
