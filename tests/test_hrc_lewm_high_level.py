import pytest
import torch

from hrc_lewm_high_level import GlobalReachabilitySelector


class DummyGenerator(torch.nn.Module):
    latent_dim = 192
    max_history = 3
    tau_model_steps = 3

    def __init__(self, candidates):
        super().__init__()
        self.register_buffer("residual_mean", torch.zeros(192))
        self.candidates = candidates
        self.position = 0

    def forward(self, history, mask, goal, duration_model_steps=3):
        batch = history.size(0)
        values = self.candidates[self.position : self.position + batch]
        self.position += batch
        return values.to(history.device), torch.zeros_like(values)


class TablePotential(torch.nn.Module):
    latent_dim = 192

    def __init__(self):
        super().__init__()
        self.call_index = 0

    def forward(self, source, goal):
        self.call_index += 1
        if self.call_index == 1:
            return goal[:, 0]  # R_G(z_t, g_i)
        if self.call_index == 2:
            return source[:, 0]  # R_G(g_i, z_G)
        return goal[:, 0]  # R_G(z_t, z_G)


def make_selector(candidate_scores, *, eta=0.5):
    candidates = torch.zeros(len(candidate_scores), 192)
    candidates[:, 0] = torch.tensor(candidate_scores)
    return GlobalReachabilitySelector(
        DummyGenerator(candidates),
        TablePotential(),
        eta_3=eta,
        num_candidates=len(candidate_scores),
    )


def test_selector_uses_rg_filter_then_rg_goal_rank_on_raw_candidates():
    selector = make_selector([0.6, 0.9, 0.7])
    history = torch.zeros(1, 1, 192)
    history[:, :, 0] = 0.8
    goal = torch.zeros(1, 192)
    goal[:, 0] = 0.4

    result = selector.select(history, goal, stochastic=False)

    assert result.selected_index.tolist() == [1]
    assert result.selected_latent[0, 0] == torch.tensor(0.9)
    assert result.candidate_latents[0, :, 0].tolist() == pytest.approx(
        [0.6, 0.9, 0.7]
    )
    assert result.to_log_dict()["selector"] == "r_g_filter_rank"
    assert result.to_log_dict()["candidate_source"] == "raw_generator_output"


def test_direct_goal_uses_only_rg_threshold_and_overrides_candidates():
    selector = make_selector([0.6, 0.9])
    history = torch.zeros(1, 3, 192)
    goal = torch.zeros(1, 192)
    goal[:, 0] = 0.75

    result = selector.select(history, goal, stochastic=False)

    assert result.direct_goal_eligible.tolist() == [True]
    assert result.direct_goal_selected.tolist() == [True]
    assert result.selected_index.tolist() == [2]
    assert torch.equal(result.selected_latent, goal)


def test_no_surviving_candidate_requests_one_step_flat_goal_fallback():
    selector = make_selector([0.1, 0.2], eta=0.5)
    history = torch.zeros(1, 2, 192)
    goal = torch.zeros(1, 192)
    goal[:, 0] = 0.3

    result = selector.select(history, goal, stochastic=False)

    assert result.selected_index.tolist() == [-1]
    assert result.fallback_to_flat_goal.tolist() == [True]
    assert torch.equal(result.selected_latent, goal)


def test_selector_rejects_history_shape_outside_btd_convention():
    selector = make_selector([0.6])
    goal = torch.zeros(1, 192)
    with pytest.raises(ValueError, match=r"\[B,T,192\]"):
        selector.select(torch.zeros(1, 192), goal, stochastic=False)
