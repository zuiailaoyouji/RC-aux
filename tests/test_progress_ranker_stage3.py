import json

import numpy as np
import pytest
import torch

from tools.train_progress_ranker_stage3 import (
    ProgressRanker,
    aggregate_episode_metrics,
    build_test_queries,
    episode_sample_rows,
    load_stage2_protocol,
    ranking_query_statistics,
    sample_pair_batch,
)


def test_progress_ranker_accepts_only_matching_bd_latents():
    ranker = ProgressRanker(latent_dim=4, hidden_dims=(8, 4))
    latent = torch.randn(3, 4)
    goal = torch.randn(3, 4)

    assert ranker(latent, goal).shape == (3,)
    with pytest.raises(ValueError, match="identical shapes"):
        ranker(latent, goal[:2])


def test_episode_rows_use_model_steps_and_exact_terminal_goal():
    rows, remaining = episode_sample_rows(
        offset=100,
        length=13,
        model_step_env_steps=5,
    )

    assert rows.tolist() == [100, 105, 110, 112]
    assert remaining.tolist() == [3, 2, 1, 0]


def test_training_pairs_are_ordered_and_exclude_terminal_goal():
    episode = {
        "latents": torch.arange(20, dtype=torch.float32).reshape(5, 4),
        "remaining_model_steps": torch.tensor([4, 3, 2, 1, 0]),
    }

    earlier, later, goals, earlier_target, later_target = sample_pair_batch(
        [episode],
        64,
        rng=np.random.default_rng(4),
        device=torch.device("cpu"),
    )

    assert torch.all(earlier_target > later_target)
    assert torch.all(later_target > 0)
    assert torch.all(goals == episode["latents"][-1])
    assert not torch.any(torch.all(later == episode["latents"][-1], dim=1))


def test_test_candidates_are_future_nonterminal_states_only():
    episode = {
        "episode_index": 7,
        "latents": torch.zeros(6, 4),
        "remaining_model_steps": torch.tensor([5, 4, 3, 2, 1, 0]),
    }

    queries = build_test_queries([episode], max_sources_per_episode=0)

    assert [query["source_index"] for query in queries] == [0, 1, 2]
    assert queries[0]["candidate_indices"] == [1, 2, 3, 4]
    assert all(5 not in query["candidate_indices"] for query in queries)


def test_stage2_protocol_reuses_validated_threshold_and_disjoint_split(
    tmp_path,
):
    report_path = tmp_path / "stage2.json"
    report_path.write_text(
        json.dumps(
            {
                "stage2_rc_filter_validated": True,
                "threshold": {
                    "eta_r": 0.61,
                    "source": "selected_on_disjoint_calibration_split",
                },
                "protocol": {
                    "tau_model_steps": 3,
                    "calibration_episode_range": [0, 5],
                    "test_episode_range": [5, 10],
                },
            }
        )
    )

    protocol = load_stage2_protocol(report_path)

    assert protocol["eta_r"] == 0.61
    assert protocol["tau_model_steps"] == 3
    assert protocol["train_episode_range"] == [0, 5]
    assert protocol["test_episode_range"] == [5, 10]


def test_stage2_protocol_rejects_overlapping_episode_ranges(tmp_path):
    report_path = tmp_path / "stage2.json"
    report_path.write_text(
        json.dumps(
            {
                "stage2_rc_filter_validated": True,
                "threshold": {"eta_r": 0.5},
                "protocol": {
                    "tau_model_steps": 3,
                    "calibration_episode_range": [0, 6],
                    "test_episode_range": [5, 10],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="disjoint"):
        load_stage2_protocol(report_path)


def test_ranking_metrics_follow_trajectory_order_after_rc_filter():
    statistics = ranking_query_statistics(
        source_score=4.0,
        candidate_scores=np.asarray([3.0, 2.0, 1.0]),
        remaining_model_steps=np.asarray([3, 2, 1]),
    )

    assert statistics["pairwise_credit"] == 3.0
    assert statistics["pair_count"] == 3
    assert statistics["spearman"] == 1.0
    assert statistics["selected_index"] == 2
    assert statistics["oracle_index"] == 2
    assert statistics["top1_correct"] == 1.0


def test_positive_progress_filter_failure_counts_as_top1_failure():
    statistics = ranking_query_statistics(
        source_score=0.0,
        candidate_scores=np.asarray([3.0, 2.0, 1.0]),
        remaining_model_steps=np.asarray([3, 2, 1]),
    )

    assert statistics["pairwise_credit"] == 3.0
    assert statistics["spearman"] == 1.0
    assert statistics["positive_progress_count"] == 0
    assert statistics["selected_index"] is None
    assert statistics["top1_correct"] == 0.0


def test_episode_aggregation_reports_only_ranking_metrics():
    metrics = aggregate_episode_metrics(
        [
            {
                "pairwise_credit": 3.0,
                "pair_count": 4.0,
                "spearman_sum": 0.8,
                "query_count": 1.0,
                "top1_correct": 1.0,
                "top1_chance": 0.25,
            },
            {
                "pairwise_credit": 1.0,
                "pair_count": 2.0,
                "spearman_sum": 0.2,
                "query_count": 1.0,
                "top1_correct": 0.0,
                "top1_chance": 0.5,
            },
        ]
    )

    assert metrics == {
        "pairwise_accuracy": 4.0 / 6.0,
        "spearman": 0.5,
        "top1_accuracy": 0.5,
        "top1_chance_accuracy": 0.375,
    }
