import pytest
import torch

from tools.diagnose_stage4c_generated_rc import (
    candidate_progress_ranks,
    exact_knn_mean_distance,
    parse_seeds,
    rowwise_correlation,
    selected_indices,
    threshold_diagnostic,
)


def test_exact_knn_uses_mean_raw_euclidean_distance():
    bank = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    queries = torch.tensor([[0.0, 0.0], [2.0, 0.0]])

    distance = exact_knn_mean_distance(
        queries, bank, k=2, query_chunk=1, bank_chunk=2
    )

    assert torch.allclose(distance, torch.tensor([0.5, 1.0]))


def test_progress_rank_and_selectors_follow_maximum_positive_progress():
    progress = torch.tensor([[0.2, 0.8, -0.1]])
    rc_score = torch.tensor([[0.9, 0.2, 0.8]])

    assert candidate_progress_ranks(progress).tolist() == [[2, 1, 3]]
    assert selected_indices(progress, rc_score, None).tolist() == [1]
    assert selected_indices(progress, rc_score, 0.5).tolist() == [0]


def test_threshold_diagnostic_counts_rejected_best_and_missing_replacement():
    rc_score = torch.tensor([[0.1, 0.8, 0.9], [0.2, 0.3, 0.4]])
    progress = torch.tensor([[3.0, 2.0, -1.0], [1.0, 0.5, 0.2]])
    residual = torch.tensor([[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]])
    manifold = torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.3, 0.2]])

    result = threshold_diagnostic(
        rc_score, progress, residual, manifold, threshold=0.5
    )

    assert result["candidate_rc_pass_rate"] == pytest.approx(2.0 / 6.0)
    assert result["sources_n_rc_zero_rate"] == pytest.approx(0.5)
    assert result["sources_n_rc_at_least_two_rate"] == pytest.approx(0.5)
    assert result["rc_positive_progress_coverage"] == pytest.approx(0.5)
    assert result["highest_progress_candidate_rejection_rate"] == pytest.approx(1.0)
    assert result["selected_progress"]["mean"] == pytest.approx(2.0)


def test_rowwise_correlations_are_reported_per_source():
    first = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    second = torch.tensor([[2.0, 4.0, 6.0], [3.0, 2.0, 1.0]])

    result = rowwise_correlation(first, second)

    assert result["pearson_across_32_within_each_source"]["mean"] == pytest.approx(
        0.0
    )
    assert result["pearson_across_32_within_each_source"]["min"] == pytest.approx(
        -1.0
    )
    assert result["spearman_across_32_within_each_source"]["max"] == pytest.approx(
        1.0
    )


def test_seed_protocol_allows_only_formal_run_or_3072_pilot():
    assert parse_seeds("3074,3072,3073") == (3072, 3073, 3074)
    assert parse_seeds("3072") == (3072,)
    with pytest.raises(ValueError, match="formal three seeds"):
        parse_seeds("3072,3073")
