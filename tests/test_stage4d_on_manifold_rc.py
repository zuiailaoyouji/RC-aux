from collections import deque

import torch

from tools.evaluate_stage4d_on_manifold_rc import (
    NO_RC_METHOD,
    RC_METHOD,
    ExactRealLatentProjector,
    paired_closed_loop_bootstrap,
    select_with_direct_goal,
    train_validation_episodes_only,
)
from tools.evaluate_stage4_closed_loop import choose_subgoal


def test_exact_projector_returns_real_bank_rows_not_neighbor_average():
    bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    queries = torch.tensor([[1.1, 0.0], [4.0, 0.0]])
    projector = ExactRealLatentProjector(bank, query_chunk=1, bank_chunk=2)

    indices = projector.indices(queries)
    projected = projector(queries)

    assert indices.tolist() == [1, 2]
    assert torch.equal(projected, bank[indices])
    assert projected[0].tolist() == [2.0, 0.0]
    assert projected[0].tolist() != [1.0, 0.0]


def test_projector_preserves_batched_candidate_shape():
    bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    candidates = torch.tensor([[[0.1, 0.0], [4.9, 0.0]]])

    projected = ExactRealLatentProjector(bank)(candidates)

    assert projected.shape == candidates.shape
    assert torch.equal(projected, torch.tensor([[[0.0, 0.0], [5.0, 0.0]]]))


def test_validation_split_does_not_access_test_entries():
    class Cache(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("test split must remain untouched")
            return super().__getitem__(key)

    cache = Cache(
        train=[
            {"episode_index": 10},
            {"episode_index": 4001},
        ]
    )

    train, validation = train_validation_episodes_only(cache)

    assert [item["episode_index"] for item in train] == [10]
    assert [item["episode_index"] for item in validation] == [4001]


def test_matched_selector_keeps_direct_goal_rule_in_both_branches():
    progress = torch.tensor([[3.0, 2.0], [3.0, 2.0]])
    rc = torch.tensor([[0.2, 0.8], [0.2, 0.3]])
    direct_progress = torch.tensor([2.5, 4.0])
    direct_rc = torch.tensor([0.9, 0.9])

    no_rc, no_rc_covered, _ = select_with_direct_goal(
        progress,
        rc,
        direct_progress,
        direct_rc,
        eta_r=0.5,
        filter_generated_with_rc=False,
    )
    with_rc, rc_covered, _ = select_with_direct_goal(
        progress,
        rc,
        direct_progress,
        direct_rc,
        eta_r=0.5,
        filter_generated_with_rc=True,
    )

    assert no_rc.tolist() == [0, 2]
    assert with_rc.tolist() == [2, 2]
    assert no_rc_covered.all()
    assert rc_covered.all()


def test_candidate_transform_runs_before_dpsi_and_rc(monkeypatch):
    raw = torch.zeros(1, 2, 192)

    def fake_sample(*args, **kwargs):
        return raw

    class Ranker:
        def __init__(self):
            self.seen = []

        def __call__(self, source, goal):
            self.seen.append(source.clone())
            return source[:, 0]

    class Adapter:
        def __init__(self):
            self.seen = []

        def reachability(self, source, target, horizon_model_steps):
            self.seen.append(target.clone())
            return torch.full(target.shape[:-1], 0.9)

    monkeypatch.setattr(
        "tools.evaluate_stage4_closed_loop.sample_subgoal_candidates",
        fake_sample,
    )
    current = torch.zeros(192)
    current[0] = 4.0
    goal = torch.zeros(192)
    ranker = Ranker()
    adapter = Adapter()

    selected, _ = choose_subgoal(
        "stochastic32_dpsi_no_rc",
        object(),
        ranker,
        adapter,
        deque([current], maxlen=3),
        goal,
        {},
        env_steps_executed=0,
        num_candidates=2,
        eta_r=0.5,
        device=torch.device("cpu"),
        candidate_transform=lambda candidates: candidates + 2.0,
    )

    assert selected[0].item() == 2.0
    assert any(
        tensor.shape == (2, 192) and torch.all(tensor == 2.0)
        for tensor in ranker.seen
    )
    assert torch.all(adapter.seen[-1] == 2.0)


def _rollout(episode, method, success, progress):
    return {
        "episode_index": episode,
        "method": method,
        "success": success,
        "completion_env_steps": 50 if success else None,
        "fallback_rate": 0.0,
        "candidate_coverage": 1.0,
        "total_d_psi_realized_progress": progress,
        "selected_rc_scores": [],
        "selected_d_psi_progress": [],
        "candidate_diversities": [],
        "segments": [],
        "high_level_attempts": 1,
        "covered_high_level_attempts": 1,
        "fallback_model_steps": 0,
        "direct_goal_eligible_attempts": 0,
        "direct_goal_selected_attempts": 0,
        "timing_seconds": {
            "generator": 0.0,
            "rc": 0.0,
            "d_psi": 0.0,
            "cem": 0.0,
        },
    }


def test_paired_bootstrap_averages_seeds_before_episode_resampling(monkeypatch):
    import tools.evaluate_stage4d_on_manifold_rc as module

    monkeypatch.setattr(module, "EXPECTED_VALIDATION_EPISODES", 2)
    seed_results = []
    for seed in (3072, 3073, 3074):
        seed_results.append(
            {
                "generator_seed": seed,
                "rollouts": [
                    _rollout(1, NO_RC_METHOD, False, 0.0),
                    _rollout(1, RC_METHOD, True, 1.0),
                    _rollout(2, NO_RC_METHOD, True, 1.0),
                    _rollout(2, RC_METHOD, True, 2.0),
                ],
            }
        )

    report = paired_closed_loop_bootstrap(seed_results, samples=100, seed=7)

    delta = report["paired_rc_minus_no_rc_95_ci"]["task_success"]
    assert delta["mean"] == 0.5
    assert report["seed_aggregation"].startswith("average three")
    assert report["primary_success_rate_decision"].startswith("inconclusive")
