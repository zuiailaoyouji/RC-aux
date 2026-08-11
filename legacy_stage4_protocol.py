"""Archived Stage 2-4 protocol and D_psi selector helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from stage4_generator import TAU_MODEL_STEPS


def load_stage2_protocol(path: Path) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    report = json.loads(report_path.read_text())
    if report.get("stage2_rc_filter_validated") is not True:
        raise ValueError("the Stage 2 report must contain a validated RC filter")
    protocol = report.get("protocol", {})
    tau_model_steps = int(protocol.get("tau_model_steps", -1))
    if tau_model_steps != TAU_MODEL_STEPS:
        raise ValueError(
            f"Stage 4 requires Stage 2 tau_model_steps={TAU_MODEL_STEPS}"
        )
    eta_r = float(report.get("threshold", {}).get("eta_r", float("nan")))
    if not math.isfinite(eta_r) or not 0.0 <= eta_r <= 1.0:
        raise ValueError("the Stage 2 report does not contain a valid eta_R")
    return {
        "report_path": str(report_path),
        "eta_r": eta_r,
        "tau_model_steps": tau_model_steps,
        "threshold_source": report.get("threshold", {}).get("source"),
    }


def load_checkpoint_protocol(path: Path, *, stage: int) -> dict[str, Any]:
    checkpoint_path = path.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if stage == 3:
        eta_r = checkpoint.get("eta_r")
        tau_model_steps = checkpoint.get("tau_model_steps")
    elif stage == 4:
        metadata = checkpoint.get("metadata", {})
        eta_r = metadata.get("eta_r")
        tau_model_steps = metadata.get("tau_model_steps")
    else:
        raise ValueError("checkpoint protocol stage must be 3 or 4")
    try:
        eta_r = float(eta_r)
        tau_model_steps = int(tau_model_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage {stage} checkpoint lacks RC protocol metadata"
        ) from exc
    return {
        "checkpoint_path": str(checkpoint_path),
        "eta_r": eta_r,
        "tau_model_steps": tau_model_steps,
    }


def assert_protocol_consistency(
    stage2_protocol: dict[str, Any],
    checkpoint_protocol: dict[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    expected_eta = float(stage2_protocol["eta_r"])
    actual_eta = float(checkpoint_protocol["eta_r"])
    if not math.isclose(expected_eta, actual_eta, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(
            f"{checkpoint_label} eta_R={actual_eta} does not match "
            f"Stage 2 eta_R={expected_eta}"
        )
    expected_tau = int(stage2_protocol["tau_model_steps"])
    actual_tau = int(checkpoint_protocol["tau_model_steps"])
    if actual_tau != expected_tau:
        raise ValueError(
            f"{checkpoint_label} tau={actual_tau} does not match "
            f"Stage 2 tau={expected_tau}"
        )


def select_candidate_index(
    candidate_d_psi: torch.Tensor,
    source_d_psi: torch.Tensor,
    *,
    rc_scores: torch.Tensor | None,
    eta_r: float | None = None,
) -> dict[str, torch.Tensor | int | None]:
    """Apply the archived progress threshold and optional local-RC gate."""

    scores = torch.as_tensor(candidate_d_psi, dtype=torch.float32)
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("candidate_d_psi must be a nonempty [N] tensor")
    source = torch.as_tensor(
        source_d_psi, dtype=torch.float32, device=scores.device
    )
    if source.numel() != 1:
        raise ValueError("source_d_psi must be scalar")
    progress = source.reshape(()) - scores
    progress_pass = progress > 0.0
    if rc_scores is None:
        rc_pass = torch.ones_like(progress_pass)
    else:
        if eta_r is None or not 0.0 <= eta_r <= 1.0:
            raise ValueError("a valid Stage 2 eta_r is required for RC filtering")
        rc = torch.as_tensor(rc_scores, dtype=torch.float32, device=scores.device)
        if rc.shape != scores.shape:
            raise ValueError("rc_scores must match candidate_d_psi shape")
        rc_pass = rc >= eta_r
    feasible = rc_pass & progress_pass
    selected_index = None
    if bool(feasible.any()):
        selected_index = int(scores.masked_fill(~feasible, float("inf")).argmin())
    return {
        "progress": progress,
        "progress_pass": progress_pass,
        "rc_pass": rc_pass,
        "feasible": feasible,
        "selected_index": selected_index,
    }


__all__ = [
    "assert_protocol_consistency",
    "load_checkpoint_protocol",
    "load_stage2_protocol",
    "select_candidate_index",
]
