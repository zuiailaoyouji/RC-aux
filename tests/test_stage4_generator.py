import json
from collections import deque

import numpy as np
import pytest
import torch
from torch import nn

from stage4_generator import (
    GeneratorTrajectoryDataset,
    HighLevelSubgoalGenerator,
    assert_protocol_consistency,
    build_generator_sample_refs,
    enable_generator_dropout_only,
    generator_loss,
    left_padded_history,
    load_checkpoint_protocol,
    load_stage2_protocol,
    sample_subgoal_candidates,
    select_candidate_index,
    split_cached_episodes,
)
from tools.evaluate_stage4_closed_loop import (
    METHODS,
    TWOROOM_OFFICIAL_MAX_EPISODE_STEPS,
    across_seed_paired_bootstrap,
    choose_subgoal,
    reference_trajectory_waypoint,
    select_episode_records,
)
from tools.train_stage4_generator import parse_seeds


def episode(episode_index, rows):
    return {
        "episode_index": episode_index,
        "rows": torch.tensor(rows, dtype=torch.long),
        "remaining_model_steps": torch.arange(len(rows) - 1, -1, -1),
        "latents": torch.arange(len(rows) * 192, dtype=torch.float32).reshape(
            len(rows), 192
        ),
    }


def test_stage4_episode_split_is_disjoint():
    cache = {
        "train": [episode(10, [0, 5, 10]), episode(4500, [20, 25, 30])],
        "test": [episode(6000, [40, 45, 50])],
    }

    train, validation, test = split_cached_episodes(cache)

    assert [item["episode_index"] for item in train] == [10]
    assert [item["episode_index"] for item in validation] == [4500]
    assert [item["episode_index"] for item in test] == [6000]


def test_sample_refs_require_exactly_three_complete_model_steps():
    episodes = [episode(10, [100, 105, 110, 115, 120, 123])]

    refs = build_generator_sample_refs(episodes)

    assert [(ref.source_index, ref.target_index) for ref in refs] == [(0, 3), (1, 4)]


def test_dataset_uses_only_past_latents_goal_and_exact_future_target():
    episodes = [episode(10, [100, 105, 110, 115, 120, 123])]
    refs = build_generator_sample_refs(episodes)
    dataset = GeneratorTrajectoryDataset(episodes, refs)

    first = dataset[0]
    second = dataset[1]

    assert first["history_padding_mask"].tolist() == [True, True, False]
    assert torch.equal(first["history_latents"][-1], episodes[0]["latents"][0])
    assert torch.equal(first["target_latent"], episodes[0]["latents"][3])
    assert second["history_padding_mask"].tolist() == [True, False, False]
    assert torch.equal(second["goal_latent"], episodes[0]["latents"][-1])
    assert second["target_row"] - second["source_row"] == 15


def test_generator_predicts_destandardized_residual_from_current_latent():
    model = HighLevelSubgoalGenerator(dropout=0.0)
    for parameter in model.parameters():
        parameter.data.zero_()
    mean = torch.linspace(-1.0, 1.0, 192)
    model.set_residual_normalizer(mean, torch.full((192,), 2.0))
    history = torch.zeros(2, 3, 192)
    history[:, -1] = 4.0
    mask = torch.tensor([[True, True, False], [True, False, False]])
    goal = torch.ones(2, 192)

    subgoal, normalized_residual = model(history, mask, goal)

    assert torch.equal(normalized_residual, torch.zeros_like(normalized_residual))
    assert torch.allclose(subgoal, history[:, -1] + mean)
    with pytest.raises(ValueError, match="fixed to 3"):
        model(history, mask, goal, duration_model_steps=2)


def test_generator_has_the_required_lightweight_transformer_architecture():
    model = HighLevelSubgoalGenerator()

    assert model.latent_dim == 192
    assert model.model_dim == 256
    assert model.num_heads == 4
    assert model.ffn_dim == 512
    assert model.num_layers == 2
    assert model.dropout_probability == pytest.approx(0.1)
    assert len(model.transformer.layers) == 2
    assert model.position_embedding.shape == (1, 6, 256)


def test_generator_dropout_only_does_not_enable_other_modules():
    model = HighLevelSubgoalGenerator()

    enable_generator_dropout_only(model)

    assert not model.training
    assert all(
        module.training
        for module in model.modules()
        if isinstance(module, nn.Dropout)
    )
    assert all(
        module.training
        for module in model.modules()
        if isinstance(module, nn.MultiheadAttention)
    )
    assert all(
        module.training
        for module in model.modules()
        if isinstance(module, nn.TransformerEncoderLayer)
    )
    assert all(
        not module.training
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )


def test_stochastic_sampling_uses_dropout_while_deterministic_is_stable():
    torch.manual_seed(7)
    model = HighLevelSubgoalGenerator()
    history = torch.randn(1, 3, 192)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    goal = torch.randn(1, 192)

    deterministic_a = sample_subgoal_candidates(
        model, history, mask, goal, num_candidates=2, stochastic=False
    )
    deterministic_b = sample_subgoal_candidates(
        model, history, mask, goal, num_candidates=2, stochastic=False
    )
    stochastic = sample_subgoal_candidates(
        model, history, mask, goal, num_candidates=2, stochastic=True
    )

    assert torch.equal(deterministic_a, deterministic_b)
    assert torch.allclose(
        deterministic_a[:, 0], deterministic_a[:, 1], atol=1.0e-6, rtol=1.0e-6
    )
    assert not torch.equal(stochastic[:, 0], stochastic[:, 1])
    assert not model.training


def test_candidate_selection_applies_rc_then_positive_progress_then_argmin():
    selected = select_candidate_index(
        torch.tensor([3.0, 1.0, 2.0]),
        torch.tensor(4.0),
        rc_scores=torch.tensor([0.9, 0.2, 0.8]),
        eta_r=0.5,
    )

    assert selected["rc_pass"].tolist() == [True, False, True]
    assert selected["progress_pass"].tolist() == [True, True, True]
    assert selected["selected_index"] == 2

    rejected = select_candidate_index(
        torch.tensor([5.0, 6.0]),
        torch.tensor(4.0),
        rc_scores=torch.tensor([0.9, 0.9]),
        eta_r=0.5,
    )
    assert rejected["selected_index"] is None
    with pytest.raises(ValueError, match="Stage 2 eta_r"):
        select_candidate_index(
            torch.tensor([1.0]),
            torch.tensor(2.0),
            rc_scores=torch.tensor([0.9]),
        )


def test_direct_goal_joins_rc_and_progress_selection_pool():
    candidate_scores = torch.tensor([3.0, 2.0])
    direct_goal_score = torch.tensor([1.0])
    selection = select_candidate_index(
        torch.cat([candidate_scores, direct_goal_score]),
        torch.tensor(4.0),
        rc_scores=torch.tensor([0.9, 0.8, 0.7]),
        eta_r=0.5,
    )

    assert selection["feasible"].tolist() == [True, True, True]
    assert selection["selected_index"] == len(candidate_scores)


def test_complete_selector_can_choose_direct_terminal_goal(monkeypatch):
    candidates = torch.zeros(1, 2, 192)
    candidates[0, 0, 0] = 3.0
    candidates[0, 1, 0] = 2.0

    def fake_sample(*args, **kwargs):
        return candidates

    class Ranker:
        def __call__(self, source, goal):
            return source[:, 0]

    class Adapter:
        def reachability(self, source, target, horizon_model_steps):
            return torch.full(target.shape[:-1], 0.9)

    monkeypatch.setattr(
        "tools.evaluate_stage4_closed_loop.sample_subgoal_candidates",
        fake_sample,
    )
    current = torch.zeros(192)
    current[0] = 4.0
    goal = torch.zeros(192)
    goal[0] = 1.0

    selected, diagnostics = choose_subgoal(
        "stochastic32_rc_dpsi",
        object(),
        Ranker(),
        Adapter(),
        deque([current], maxlen=3),
        goal,
        {},
        env_steps_executed=0,
        num_candidates=2,
        eta_r=0.5,
        device=torch.device("cpu"),
    )

    assert torch.equal(selected, goal)
    assert diagnostics["direct_goal_eligible"] is True
    assert diagnostics["direct_goal_selected"] is True


def test_stage2_threshold_must_match_stage3_and_stage4_metadata(tmp_path):
    stage2_path = tmp_path / "stage2.json"
    stage2_path.write_text(
        json.dumps(
            {
                "stage2_rc_filter_validated": True,
                "protocol": {"tau_model_steps": 3},
                "threshold": {"eta_r": 0.61, "source": "calibration"},
            }
        )
    )
    stage3_path = tmp_path / "stage3.pt"
    torch.save({"eta_r": 0.61, "tau_model_steps": 3}, stage3_path)
    stage4_path = tmp_path / "stage4.pt"
    torch.save(
        {"metadata": {"eta_r": 0.61, "tau_model_steps": 3}}, stage4_path
    )

    stage2 = load_stage2_protocol(stage2_path)
    assert_protocol_consistency(
        stage2,
        load_checkpoint_protocol(stage3_path, stage=3),
        checkpoint_label="Stage 3",
    )
    assert_protocol_consistency(
        stage2,
        load_checkpoint_protocol(stage4_path, stage=4),
        checkpoint_label="Stage 4",
    )

    mismatched = {"eta_r": 0.62, "tau_model_steps": 3}
    with pytest.raises(ValueError, match="does not match"):
        assert_protocol_consistency(
            stage2, mismatched, checkpoint_label="mismatched checkpoint"
        )


def test_closed_loop_episode_sampling_is_fixed_and_length_independent():
    episodes = [
        {"episode_index": index, "demonstration_env_steps": 20 + index * 50}
        for index in range(10)
    ]

    first = select_episode_records(episodes, count=6, seed=17)
    second = select_episode_records(episodes, count=6, seed=17)

    assert [item["episode_index"] for item in first] == [
        item["episode_index"] for item in second
    ]
    assert len(first) == 6
    assert max(item["demonstration_env_steps"] for item in first) > 100
    assert TWOROOM_OFFICIAL_MAX_EPISODE_STEPS == 100


def test_paired_bootstrap_aggregates_seeds_within_matched_episodes():
    seed_results = []
    for generator_seed in (1, 2, 3):
        rollouts = []
        for episode_index in (5001, 5002):
            for method in METHODS:
                complete = method == "stochastic32_rc_dpsi"
                rollouts.append(
                    {
                        "episode_index": episode_index,
                        "method": method,
                        "success": complete,
                        "completion_env_steps": 30 if complete else None,
                        "fallback_rate": 0.1 if complete else 0.2,
                        "candidate_coverage": (
                            None if method == "flat_rc_lewm" else 0.8
                        ),
                        "total_d_psi_realized_progress": 2.0 if complete else 1.0,
                        "selected_rc_scores": [],
                        "selected_d_psi_progress": [],
                        "candidate_diversities": [],
                        "segments": [
                            {
                                "euclidean_target_distance_progress": 0.0,
                                "d_psi_progress": 0.0,
                            }
                        ],
                        "high_level_attempts": 1,
                        "covered_high_level_attempts": int(complete),
                        "fallback_model_steps": int(not complete),
                        "direct_goal_eligible_attempts": 0,
                        "direct_goal_selected_attempts": 0,
                        "timing_seconds": {
                            "generator": 0.0,
                            "rc": 0.0,
                            "d_psi": 0.0,
                            "cem": 0.0,
                        },
                    }
                )
        seed_results.append(
            {"generator_seed": generator_seed, "rollouts": rollouts}
        )

    report = across_seed_paired_bootstrap(
        seed_results,
        episode_horizon_env_steps=100,
        samples=100,
        seed=9,
    )

    success_difference = report["paired_bootstrap_95_ci"][
        "reference_trajectory_waypoint"
    ]["task_success"]
    assert report["generator_seed_count"] == 3
    assert success_difference["paired_episode_count"] == 2
    assert success_difference["mean"] == pytest.approx(1.0)


def test_reference_waypoint_uses_demonstration_t_plus_three_or_terminal():
    record = {
        "start_row": 100,
        "reference_rows": np.asarray([100, 105, 110, 115, 120, 123]),
        "reference_latents": torch.arange(6 * 192).reshape(6, 192),
    }

    assert torch.equal(
        reference_trajectory_waypoint(record, env_steps_executed=0),
        record["reference_latents"][3],
    )
    assert torch.equal(
        reference_trajectory_waypoint(record, env_steps_executed=15),
        record["reference_latents"][-1],
    )


def test_generator_loss_matches_required_objective():
    predicted = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])

    loss, smooth_l1, cosine = generator_loss(predicted, target)

    assert torch.allclose(loss, smooth_l1 + 0.1 * cosine)
    assert cosine.item() == pytest.approx(1.0)


def test_formal_training_requires_three_unique_seeds():
    assert parse_seeds("1,2,3") == [1, 2, 3]
    with pytest.raises(ValueError, match="at least 3"):
        parse_seeds("1,2")
    with pytest.raises(ValueError, match="unique"):
        parse_seeds("1,1,2")


def test_left_padded_history_keeps_current_latent_last():
    first = torch.ones(192)
    current = torch.full((192,), 2.0)

    history, mask = left_padded_history([first, current])

    assert history.shape == (1, 3, 192)
    assert mask.tolist() == [[True, False, False]]
    assert torch.equal(history[0, -1], current)
