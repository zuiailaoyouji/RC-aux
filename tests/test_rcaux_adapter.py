import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rcaux_adapter import (
    RCAuxAdapter,
    RCAuxPlannerConfig,
    RCAuxProfile,
    ReachabilityDiagnostics,
)


class IdentityScaler:
    mean_ = np.zeros(2, dtype=np.float32)
    scale_ = np.ones(2, dtype=np.float32)

    def transform(self, value):
        return np.asarray(value, dtype=np.float32)

    def inverse_transform(self, value):
        return np.asarray(value, dtype=np.float32)


class FakeActionEncoder(nn.Module):
    def forward(self, actions):
        mean = actions.mean(dim=-1, keepdim=True)
        return mean.expand(*actions.shape[:-1], 4)


class FakeReachabilityHead(nn.Module):
    max_horizon = 5

    def forward(self, source, target, horizon):
        return -(source - target).square().sum(dim=-1) + horizon.float()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.predictor = SimpleNamespace(pos_embedding=torch.zeros(1, 3, 4))
        self.action_encoder = FakeActionEncoder()
        self.reachability_head = FakeReachabilityHead()
        self.image_goal_cost_calls = 0
        self.latent_goal_criterion_calls = 0
        self.last_latent_goal_shape = None

    def encode(self, info):
        pixels = info["pixels"]
        assert pixels.ndim == 5
        value = pixels.mean(dim=(-3, -2, -1), keepdim=False)
        return {**info, "emb": value.unsqueeze(-1).expand(-1, -1, 4)}

    def rollout_open_loop(
        self,
        emb_history,
        act_history,
        future_act_emb,
        horizon,
        history_size,
    ):
        current = emb_history[:, -1:]
        predictions = []
        for step in range(horizon):
            action = (
                act_history[:, -1:]
                if step == 0
                else future_act_emb[:, step - 1 : step]
            )
            current = current + action
            predictions.append(current)
        return torch.cat(predictions, dim=1)

    def rollout(self, info, action_candidates, history_size):
        batch, samples, model_steps = action_candidates.shape[:3]
        increments = action_candidates.mean(dim=-1, keepdim=True)
        increments = increments.expand(-1, -1, -1, 4).cumsum(dim=2)
        initial = torch.zeros(batch, samples, 1, 4)
        info["predicted_emb"] = torch.cat([initial, increments], dim=2)
        return info

    def criterion(self, info):
        self.latent_goal_criterion_calls += 1
        prediction = info["predicted_emb"][:, :, -1]
        goal = info["goal_emb"]
        self.last_latent_goal_shape = tuple(goal.shape)
        return (prediction - goal).square().sum(dim=-1)

    def score_reachability(self, source, target, horizon):
        return self.reachability_head(source, target, horizon)

    def get_cost(self, info, candidates):
        self.image_goal_cost_calls += 1
        assert info["pixels"].ndim == 6
        assert info["goal"].ndim == 6
        return candidates.square().mean(dim=(-2, -1))


TEST_PROFILE = RCAuxProfile(
    name="test",
    dataset_name="test",
    image_size=224,
    action_dim=2,
    model_step_env_steps=2,
    action_low=(-1.0, -1.0),
    action_high=(1.0, 1.0),
)


def make_adapter(**kwargs):
    config = RCAuxPlannerConfig(
        planning_horizon_model_steps=2,
        execution_horizon_model_steps=1,
        num_samples=4,
        n_steps=1,
        topk=2,
        seed=7,
    )
    return RCAuxAdapter(
        FakeModel(),
        profile=TEST_PROFILE,
        planner_config=config,
        device="cpu",
        action_scaler=IdentityScaler(),
        **kwargs,
    )


def test_encode_observation_returns_btd_latent():
    adapter = make_adapter()
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    latent = adapter.encode_observation(image)

    assert latent.shape == (1, 1, 4)
    assert latent.dtype == torch.float32


def test_predict_latents_uses_model_step_action_blocks_open_loop():
    adapter = make_adapter()
    history = torch.zeros(1, 1, 4)
    history_actions = torch.ones(1, 2, 2)
    future_actions = 2 * torch.ones(1, 2, 2)

    predicted = adapter.predict_latents(
        history,
        history_actions,
        future_actions,
        actions_normalized=True,
    )

    assert predicted.shape == (1, 2, 4)
    assert torch.allclose(predicted[:, 0], torch.ones(1, 4))
    assert torch.allclose(predicted[:, 1], 3 * torch.ones(1, 4))
    with pytest.raises(ValueError, match=r"\[B,T,4\]"):
        adapter.predict_latents(history[0], history_actions)


def test_reachability_supports_bd_and_bnd_candidate_queries():
    adapter = make_adapter()
    source = torch.zeros(2, 4)
    paired_target = torch.zeros(2, 4)
    candidate_targets = torch.zeros(2, 3, 4)

    paired = adapter.reachability(source, paired_target, torch.tensor([1, 2]))
    diagnostics = adapter.reachability(
        source,
        candidate_targets,
        torch.tensor([2, 3]),
        return_diagnostics=True,
    )

    assert paired.shape == (2,)
    assert isinstance(diagnostics, ReachabilityDiagnostics)
    assert diagnostics.probabilities.shape == (2, 3)
    assert diagnostics.logits.shape == (2, 3)
    assert diagnostics.horizons_model_steps.tolist() == [[2, 2, 2], [3, 3, 3]]
    assert diagnostics.source_shape == (2, 4)
    assert diagnostics.target_shape == (2, 3, 4)
    assert diagnostics.to_log_dict()["query_shape"] == [2, 3]
    json.dumps(diagnostics.to_log_dict())

    candidate_sources = torch.zeros(2, 3, 4)
    shared_targets = torch.ones(2, 4)
    reverse_broadcast = adapter.reachability(
        candidate_sources,
        shared_targets,
        torch.tensor([[1, 2, 3], [3, 2, 1]]),
    )
    assert reverse_broadcast.shape == (2, 3)


def test_latent_goal_is_primary_and_steps_are_explicit():
    adapter = make_adapter()
    observation = np.zeros((224, 224, 3), dtype=np.uint8)
    goal_latent = torch.zeros(1, 4)

    result = adapter.plan_to_latent(observation, goal_latent)

    assert adapter.model.latent_goal_criterion_calls > 0
    assert adapter.model.image_goal_cost_calls == 0
    assert adapter.model.last_latent_goal_shape == (1, 1, 4)
    assert result.planned_actions_env_steps.shape == (1, 4, 2)
    assert result.actions_to_execute_env_steps.shape == (1, 2, 2)
    assert result.normalized_action_blocks.shape == (1, 2, 4)
    assert result.diagnostics.goal_mode == "latent"
    assert result.diagnostics.planning_horizon_model_steps == 2
    assert result.diagnostics.execution_horizon_model_steps == 1
    assert result.diagnostics.planning_horizon_env_steps == 4
    assert result.diagnostics.execution_horizon_env_steps == 2
    assert result.diagnostics.to_log_dict()["profile_name"] == "test"
    json.dumps(result.diagnostics.to_log_dict())


def test_image_goal_path_is_retained_for_regression():
    adapter = make_adapter()
    observation = np.zeros((224, 224, 3), dtype=np.uint8)
    goal_image = np.full((224, 224, 3), 255, dtype=np.uint8)

    result = adapter.plan_to_image(observation, goal_image)

    assert adapter.model.image_goal_cost_calls > 0
    assert adapter.model.latent_goal_criterion_calls == 0
    assert result.diagnostics.goal_mode == "image"


def test_new_subgoal_resets_previous_plan_warm_start():
    adapter = make_adapter()
    observation = np.zeros((224, 224, 3), dtype=np.uint8)
    first_goal = torch.zeros(1, 4)
    second_goal = torch.ones(1, 4)

    first = adapter.plan_to_latent(observation, first_goal)
    repeated = adapter.plan_to_latent(observation, first_goal)
    changed = adapter.plan_to_latent(observation, second_goal)

    assert first.diagnostics.warm_start_source == "none"
    assert repeated.diagnostics.warm_start_source == "previous_plan"
    assert not repeated.diagnostics.warm_start_reset
    assert changed.diagnostics.warm_start_source == "none"
    assert changed.diagnostics.warm_start_reset
    assert changed.diagnostics.warm_start_reset_reason == "subgoal_changed"


def test_unified_plan_requires_exactly_one_goal_representation():
    adapter = make_adapter()
    observation = np.zeros((224, 224, 3), dtype=np.uint8)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    latent = torch.zeros(1, 4)

    with pytest.raises(ValueError, match="exactly one"):
        adapter.plan(observation)
    with pytest.raises(ValueError, match="exactly one"):
        adapter.plan(observation, goal_latent=latent, goal_image=image)
