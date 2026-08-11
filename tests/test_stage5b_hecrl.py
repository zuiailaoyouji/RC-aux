from argparse import Namespace

import pytest
import torch

from tools.evaluate_stage5b_hecrl import (
    DEFAULT_CEM_ITERATIONS,
    DEFAULT_FLAT_HORIZON,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_REACHABILITY_COST_WEIGHT,
    DEFAULT_TOPK,
    METHOD_A,
    METHOD_B,
    METHOD_C,
    evaluate_offline_seed,
    heldout_audit_episodes,
    select_indices,
    stage5b_decision,
    validate_fixed_runtime,
)
from tools.evaluate_stage4d_on_manifold_rc import ExactRealLatentProjector


def candidate_scores():
    local = torch.tensor([[0.8, 0.2, 0.7], [0.2, 0.3, 0.4]])
    goal = torch.tensor([[0.3, 0.9, 0.6], [0.8, 0.7, 0.6]])
    progress = torch.tensor([[0.1, 0.5, -0.2], [-0.1, -0.2, -0.3]])
    direct = torch.tensor([False, False])
    return local, goal, progress, direct


def test_three_selectors_follow_the_fixed_rules():
    local, goal, progress, direct = candidate_scores()
    selected_a, covered_a = select_indices(
        METHOD_A, local, goal, progress, direct, eta_3=0.5
    )
    selected_b, covered_b = select_indices(
        METHOD_B, local, goal, progress, direct, eta_3=0.5
    )
    selected_c, covered_c = select_indices(
        METHOD_C, local, goal, progress, direct, eta_3=0.5
    )

    assert selected_a.tolist() == [1, -1]
    assert covered_a.tolist() == [True, False]
    assert selected_b.tolist() == [1, 0]
    assert covered_b.tolist() == [True, True]
    assert selected_c.tolist() == [2, -1]
    assert covered_c.tolist() == [True, False]


@pytest.mark.parametrize("method", [METHOD_A, METHOD_B, METHOD_C])
def test_direct_goal_precheck_overrides_every_selector(method):
    local, goal, progress, _ = candidate_scores()
    direct = torch.tensor([True, True])
    selected, _ = select_indices(
        method, local, goal, progress, direct, eta_3=0.5
    )
    assert selected.tolist() == [3, 3]


def intervals(filter_ci, rank_ci, full_ci):
    return {
        "Delta_filter_SR_C_minus_SR_B": filter_ci,
        "Delta_rank_SR_B_minus_SR_A": rank_ci,
        "Delta_full_SR_C_minus_SR_A": full_ci,
    }


def ci(lower, upper):
    return {"lower_95": lower, "upper_95": upper}


def test_stage5b_decision_supports_unified_only_with_both_positive_lower_bounds():
    result = stage5b_decision(
        intervals(ci(0.01, 0.1), ci(0.01, 0.05), ci(0.02, 0.2))
    )
    assert result == "unified_R_G_filtering_and_ranking_supported"


def test_stage5b_decision_keeps_dpsi_when_rg_ranking_is_significantly_worse():
    result = stage5b_decision(
        intervals(ci(-0.1, -0.01), ci(-0.2, -0.01), ci(-0.2, -0.01))
    )
    assert result == "retain_D_psi_route"


def test_stage5b_decision_removes_gate_when_filter_hurts_but_rank_is_not_worse():
    result = stage5b_decision(
        intervals(ci(-0.1, -0.01), ci(0.01, 0.04), ci(-0.09, -0.01))
    )
    assert result == "retain_R_G_ranking_remove_high_level_hard_gate"


def test_stage5b_decision_is_inconclusive_when_key_intervals_cross_zero():
    result = stage5b_decision(
        intervals(ci(-0.01, 0.03), ci(-0.01, 0.03), ci(-0.01, 0.03))
    )
    assert result == "inconclusive"


def test_heldout_audit_excludes_early_stopping_episodes():
    episodes = [
        {"episode_index": 4499},
        {"episode_index": 4500},
        {"episode_index": 4999},
    ]
    assert [item["episode_index"] for item in heldout_audit_episodes(episodes)] == [
        4500,
        4999,
    ]


def runtime_args(**updates):
    values = {
        "offline_batch_size": 1,
        "audit_batch_size": 1,
        "knn_query_chunk": 1,
        "knn_bank_chunk": 1,
        "num_samples": DEFAULT_NUM_SAMPLES,
        "cem_iterations": DEFAULT_CEM_ITERATIONS,
        "topk": DEFAULT_TOPK,
        "flat_planning_horizon_model_steps": DEFAULT_FLAT_HORIZON,
        "bootstrap_samples": 10000,
        "reachability_cost_weight": DEFAULT_REACHABILITY_COST_WEIGHT,
        "device": "cpu",
        "validate_only": False,
    }
    values.update(updates)
    return Namespace(**values)


def test_fixed_runtime_rejects_protocol_tuning():
    validate_fixed_runtime(runtime_args())
    with pytest.raises(ValueError, match="num_samples"):
        validate_fixed_runtime(runtime_args(num_samples=299))
    with pytest.raises(ValueError, match="bootstrap_samples"):
        validate_fixed_runtime(runtime_args(bootstrap_samples=9999))


class DummyRg(torch.nn.Module):
    def forward(self, source, goal):
        return torch.sigmoid(1.0 - (source - goal).norm(dim=-1) / 10.0)


class DummyDpsi(torch.nn.Module):
    def forward(self, source, goal):
        return (source - goal).square().mean(dim=-1)


def test_offline_control_uses_exact_projection_and_reports_all_methods():
    torch.manual_seed(5)
    bank = 9.8 + 0.05 * torch.randn(40, 192)
    artifact = {
        "metadata": {"generator_seed": 3072},
        "candidate_latents": 9.8 + 0.1 * torch.randn(2, 32, 192),
        "current_latent": torch.full((2, 192), 10.0),
        "goal_latent": torch.zeros(2, 192),
    }
    report = evaluate_offline_seed(
        artifact,
        ExactRealLatentProjector(bank, query_chunk=16, bank_chunk=20),
        DummyRg(),
        DummyDpsi(),
        eta_3=0.5,
        batch_size=2,
        query_chunk=16,
        bank_chunk=20,
        device=torch.device("cpu"),
    )

    assert report["projection_is_exact_bank_row"] is True
    assert set(report["methods"]) == {METHOD_A, METHOD_B, METHOD_C}
    assert report["candidate_r_g_pass_rate_method_c"] == 1.0
    assert all(
        report["methods"][method]["decision_coverage_including_direct_goal"]
        == 1.0
        for method in (METHOD_A, METHOD_B, METHOD_C)
    )
