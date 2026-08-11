import numpy as np
import pytest
import torch

from stage5_global_reachability import (
    GlobalHittingTimePotential,
    temporal_ranking_loss,
)
from tools.train_stage5a_global_reachability import (
    build_temporal_pair_buckets,
    classification_metrics,
    pairwise_query_metrics,
    sample_balanced_pair_refs,
    select_local_threshold,
    temporal_splits,
)


def _episode(index, count):
    return {
        "episode_index": index,
        "latents": torch.arange(count * 192, dtype=torch.float32).reshape(
            count, 192
        ),
    }


def test_global_head_uses_raw_576d_features_and_sigmoid_scalar():
    model = GlobalHittingTimePotential()
    source = torch.randn(4, 192)
    goal = torch.randn(4, 192)

    score = model(source, goal)

    assert score.shape == (4,)
    assert torch.all((score > 0.0) & (score < 1.0))
    assert model.network[0].in_features == 576
    assert model.network[0].out_features == 256
    assert model.network[2].out_features == 128
    with pytest.raises(ValueError, match="identical"):
        model(source, goal[:3])


def test_temporal_ranking_loss_prefers_near_scores_above_far_scores():
    ordered = temporal_ranking_loss(
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.8, 0.7]),
        torch.tensor([0.1, 0.2]),
    )
    reversed_loss = temporal_ranking_loss(
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.8, 0.7]),
    )

    assert ordered < reversed_loss


def test_temporal_pair_buckets_use_exact_index_distance():
    buckets = build_temporal_pair_buckets([_episode(1, 21)])

    assert len(buckets[1]) == 20
    assert len(buckets[20]) == 1
    assert np.all(buckets[3][:, 2] - buckets[3][:, 1] == 3)


def test_epoch_sampler_is_exactly_balanced_across_distance_buckets():
    buckets = build_temporal_pair_buckets([_episode(1, 21)])
    refs, distances = sample_balanced_pair_refs(
        buckets, 200, rng=np.random.default_rng(7)
    )

    assert refs.shape == (200, 3)
    assert {distance: int(np.sum(distances == distance)) for distance in range(1, 21)} == {
        distance: 10 for distance in range(1, 21)
    }


def test_stage5a_split_never_accesses_test_half():
    class Cache(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("test split must remain untouched")
            return super().__getitem__(key)

    cache = Cache(
        train=[
            _episode(1, 21),
            _episode(4001, 21),
            _episode(4501, 21),
        ]
    )

    train, calibration, heldout = temporal_splits(cache)

    assert [item["episode_index"] for item in train] == [1]
    assert [item["episode_index"] for item in calibration] == [4001]
    assert [item["episode_index"] for item in heldout] == [4501]


def test_pairwise_metrics_support_potential_and_cost_orientations():
    refs = np.asarray(
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
        ]
    )
    distances = np.asarray([1, 2, 3])

    potential = pairwise_query_metrics(
        refs,
        distances,
        np.asarray([0.9, 0.7, 0.4]),
        group_by="source",
        higher_for_near=True,
    )
    cost = pairwise_query_metrics(
        refs,
        distances,
        np.asarray([0.1, 0.3, 0.8]),
        group_by="source",
        higher_for_near=False,
    )

    assert potential["pairwise_temporal_order_accuracy"] == 1.0
    assert cost["pairwise_temporal_order_accuracy"] == 1.0
    assert potential["oriented_per_query_spearman"]["mean"] == 1.0
    assert cost["oriented_per_query_spearman"]["mean"] == 1.0


def test_local_threshold_uses_calibration_constraints_and_reports_f1():
    scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = np.asarray([True, True, True, False, False, False])

    threshold, constraints_met, _ = select_local_threshold(
        scores,
        labels,
        max_fpr=0.05,
        min_precision=0.95,
    )
    metrics = classification_metrics(scores, labels, threshold)

    assert constraints_met is True
    assert threshold == pytest.approx(0.7)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
