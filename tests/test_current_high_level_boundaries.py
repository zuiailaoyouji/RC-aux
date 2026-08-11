import inspect
import sys

import hrc_lewm_high_level
import stage4_generator
from rcaux_adapter import RCAuxAdapter
from tools import train_stage4_generator, train_stage5a_global_reachability


def parsed_args(monkeypatch, module):
    monkeypatch.setattr(sys, "argv", [module.__file__])
    return module.parse_args()


def test_generator_training_has_no_stage2_or_stage3_runtime_dependency(monkeypatch):
    args = parsed_args(monkeypatch, train_stage4_generator)
    assert not hasattr(args, "stage2_report")
    assert not hasattr(args, "progress_ranker")
    source = inspect.getsource(train_stage4_generator)
    assert "load_stage2_protocol" not in source
    assert "load_progress_ranker" not in source
    generator_source = inspect.getsource(stage4_generator)
    assert "load_stage2_protocol" not in generator_source
    assert "select_candidate_index" not in generator_source


def test_stage5a_training_has_no_dpsi_runtime_dependency(monkeypatch):
    args = parsed_args(monkeypatch, train_stage5a_global_reachability)
    assert not hasattr(args, "progress_ranker")
    assert not hasattr(args, "progress_ranker_report")
    source = inspect.getsource(train_stage5a_global_reachability.main)
    assert "load_progress_ranker" not in source
    assert "d_psi" not in source


def test_deployment_selector_has_no_old_gate_ranker_or_projection_hook():
    source = inspect.getsource(hrc_lewm_high_level.GlobalReachabilitySelector)
    assert "candidate_transform" not in source
    assert "ExactRealLatentProjector" not in source
    assert "progress_ranker" not in source
    assert ".reachability(" not in source


def test_low_level_budget_conditioned_reachability_is_preserved():
    assert callable(RCAuxAdapter.reachability)
    assert "horizon_model_steps" in inspect.signature(
        RCAuxAdapter.reachability
    ).parameters
