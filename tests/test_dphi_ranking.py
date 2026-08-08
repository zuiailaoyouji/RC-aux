import numpy as np
import torch

from tools.evaluate_dphi_ranking import (
    TwoRoomObstacleCost,
    evaluate_candidate_set,
    pairwise_accuracy,
    reachability_time,
    summarize_candidate_set,
)


def ranking_candidate(
    candidate_id,
    *,
    dphi,
    latent_l2,
    target_cost,
    executed_progress,
):
    return {
        "candidate_id": candidate_id,
        "category": "within_budget_witnessed",
        "dphi": dphi,
        "latent_l2": latent_l2,
        "target_obstacle_cost_env_steps": target_cost,
        "stage2_success_rate": 1.0,
        "stage3_success_rate": 1.0,
        "mean_executed_obstacle_progress_env_steps": executed_progress,
        "trajectory_remaining_env_steps": target_cost,
    }


def test_obstacle_cost_requires_crossing_a_valid_door_region():
    cost = TwoRoomObstacleCost(
        seed=42,
        success_radius=16.0,
        door_samples=513,
    )
    try:
        source = np.array([40.0, 40.0])
        same_room = np.array([80.0, 40.0])
        blocked_cross_room = np.array([180.0, 180.0])

        same_cost = cost.path_length(source, same_room)
        blocked_cost = cost.path_length(source, blocked_cross_room)

        assert np.isclose(same_cost, 24.0)
        assert blocked_cost > (
            np.linalg.norm(source - blocked_cross_room) - 16.0
        )
    finally:
        cost.close()


def test_reachability_time_integrates_horizons_one_through_five():
    class FakeAdapter:
        def reachability(self, source, target, horizon_model_steps):
            return torch.full(
                source.shape[:-1],
                horizon_model_steps / 10.0,
            )

    probabilities, dphi = reachability_time(
        FakeAdapter(),
        torch.zeros(2, 4),
        torch.ones(2, 4),
    )

    torch.testing.assert_close(
        probabilities,
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]] * 2),
    )
    torch.testing.assert_close(dphi, torch.tensor([3.5, 3.5]))


def test_candidate_set_keeps_dphi_random_l2_and_oracle_separate():
    candidates = [
        ranking_candidate(
            "dphi_best",
            dphi=1.0,
            latent_l2=5.0,
            target_cost=10.0,
            executed_progress=20.0,
        ),
        ranking_candidate(
            "l2_best",
            dphi=2.0,
            latent_l2=1.0,
            target_cost=20.0,
            executed_progress=10.0,
        ),
    ]

    result = evaluate_candidate_set(candidates, source_cost=40.0)

    assert result is not None
    assert result["dphi"]["candidate_id"] == "dphi_best"
    assert result["latent_l2"]["candidate_id"] == "l2_best"
    assert result["oracle_obstacle_cost"]["candidate_id"] == "dphi_best"
    assert result["random_uniform"]["target_obstacle_cost_env_steps"] == 15.0
    assert result["pairwise_accuracy"] == 1.0
    assert pairwise_accuracy(candidates) == 1.0


def test_summary_uses_paired_trial_differences():
    candidates = [
        ranking_candidate(
            "best",
            dphi=1.0,
            latent_l2=2.0,
            target_cost=10.0,
            executed_progress=20.0,
        ),
        ranking_candidate(
            "worse",
            dphi=2.0,
            latent_l2=1.0,
            target_cost=20.0,
            executed_progress=10.0,
        ),
    ]
    evaluation = evaluate_candidate_set(candidates, source_cost=40.0)
    trials = [
        {"oracle_feasible": evaluation},
        {"oracle_feasible": evaluation},
    ]

    summary = summarize_candidate_set(
        trials,
        set_key="oracle_feasible",
        bootstrap_samples=1000,
        bootstrap_seed=9,
    )

    assert summary["ranking_trial_count"] == 2
    assert summary["mean_pairwise_accuracy"] == 1.0
    assert summary["dphi_top1_oracle_match_rate"] == 1.0
    assert summary["paired_cis"]["random_minus_dphi_target_cost"] == {
        "mean": 5.0,
        "lower_95": 5.0,
        "upper_95": 5.0,
    }
