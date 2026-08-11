"""Deployment high-level selector for HRC-LeWM.

The only high-level score is the frozen Stage 5A discounted witnessed
hitting-time potential R_G. The low-level budget-conditioned R_local remains
encapsulated in RC-LeWM and is intentionally not queried here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from stage4_generator import (
    LATENT_DIM,
    MAX_HISTORY,
    TAU_MODEL_STEPS,
    HighLevelSubgoalGenerator,
    load_generator_checkpoint,
    sample_subgoal_candidates,
)
from stage5_global_reachability import (
    GlobalHittingTimePotential,
    load_global_hitting_potential,
)


@dataclass(frozen=True)
class HighLevelSelection:
    """Batched selector output using the canonical [B,N,D] convention."""

    selected_latent: torch.Tensor
    selected_index: torch.Tensor
    direct_goal_eligible: torch.Tensor
    direct_goal_selected: torch.Tensor
    candidate_coverage: torch.Tensor
    fallback_to_flat_goal: torch.Tensor
    candidate_latents: torch.Tensor
    local_competence_scores: torch.Tensor
    goal_potential_scores: torch.Tensor
    direct_goal_score: torch.Tensor
    eta_3: float

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "batch_size": int(self.selected_latent.size(0)),
            "candidate_count": int(self.candidate_latents.size(1)),
            "eta_3": self.eta_3,
            "direct_goal_eligible_rate": float(
                self.direct_goal_eligible.float().mean()
            ),
            "direct_goal_selected_rate": float(
                self.direct_goal_selected.float().mean()
            ),
            "candidate_coverage": float(self.candidate_coverage.float().mean()),
            "fallback_rate": float(self.fallback_to_flat_goal.float().mean()),
            "candidate_pass_rate": float(
                (self.local_competence_scores >= self.eta_3).float().mean()
            ),
            "selected_index": self.selected_index.detach().cpu().tolist(),
            "selector": "r_g_filter_rank",
            "candidate_source": "raw_generator_output",
        }


class GlobalReachabilitySelector(nn.Module):
    """Generator -> R_G filter -> R_G rank deployment policy."""

    def __init__(
        self,
        generator: HighLevelSubgoalGenerator,
        global_potential: GlobalHittingTimePotential,
        *,
        eta_3: float,
        num_candidates: int = 32,
    ) -> None:
        super().__init__()
        if not 0.0 <= eta_3 <= 1.0:
            raise ValueError("eta_3 must be in [0,1]")
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if generator.latent_dim != LATENT_DIM:
            raise ValueError("generator latent dimension must be 192")
        if global_potential.latent_dim != LATENT_DIM:
            raise ValueError("R_G latent dimension must be 192")
        if generator.max_history != MAX_HISTORY:
            raise ValueError("generator history must be at most three tokens")
        if generator.tau_model_steps != TAU_MODEL_STEPS:
            raise ValueError("generator duration token must remain three model steps")
        self.generator = generator.eval().requires_grad_(False)
        self.global_potential = global_potential.eval().requires_grad_(False)
        self.eta_3 = float(eta_3)
        self.num_candidates = int(num_candidates)

    @property
    def device(self) -> torch.device:
        return self.generator.residual_mean.device

    @classmethod
    def from_checkpoints(
        cls,
        generator_path: str | Path,
        global_potential_path: str | Path,
        *,
        device: str | torch.device,
        num_candidates: int = 32,
    ) -> "GlobalReachabilitySelector":
        generator, generator_checkpoint = load_generator_checkpoint(
            Path(generator_path), device=device
        )
        global_potential, global_checkpoint = load_global_hitting_potential(
            Path(global_potential_path), device=torch.device(device)
        )
        generator_seed = int(generator_checkpoint["seed"])
        global_seed = int(global_checkpoint["seed"])
        if generator_seed != global_seed:
            raise ValueError(
                "deployment requires a paired generator/R_G checkpoint seed"
            )
        if global_checkpoint.get("not_absolute_global_reachability") is not True:
            raise ValueError("R_G checkpoint has invalid potential semantics")
        eta_3 = global_checkpoint.get("eta_3")
        if eta_3 is None:
            raise ValueError("R_G checkpoint does not contain its calibrated eta_3")
        return cls(
            generator,
            global_potential,
            eta_3=float(eta_3),
            num_candidates=num_candidates,
        )

    def _prepare_history(
        self,
        history_latents: torch.Tensor,
        history_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.as_tensor(
            history_latents, dtype=torch.float32, device=self.device
        )
        if history.ndim != 3 or history.size(-1) != LATENT_DIM:
            raise ValueError("history_latents must be [B,T,192]")
        if not 1 <= history.size(1) <= MAX_HISTORY:
            raise ValueError("history length T must satisfy 1<=T<=3")
        if history_padding_mask is not None and history.size(1) != MAX_HISTORY:
            raise ValueError("an explicit padding mask requires [B,3,192] history")
        if history.size(1) < MAX_HISTORY:
            missing = MAX_HISTORY - history.size(1)
            padding = torch.zeros(
                history.size(0), missing, LATENT_DIM, device=self.device
            )
            history = torch.cat([padding, history], dim=1)
            mask = torch.cat(
                [
                    torch.ones(
                        history.size(0), missing, dtype=torch.bool, device=self.device
                    ),
                    torch.zeros(
                        history.size(0), MAX_HISTORY - missing,
                        dtype=torch.bool, device=self.device,
                    ),
                ],
                dim=1,
            )
        elif history_padding_mask is None:
            mask = torch.zeros(
                history.size(0), MAX_HISTORY, dtype=torch.bool, device=self.device
            )
        else:
            mask = torch.as_tensor(
                history_padding_mask, dtype=torch.bool, device=self.device
            )
            if mask.shape != history.shape[:2]:
                raise ValueError("history_padding_mask must be [B,3]")
        if torch.any(mask[:, -1]):
            raise ValueError("the current latent cannot be padding")
        return history, mask

    @torch.inference_mode()
    def select(
        self,
        history_latents: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        history_padding_mask: torch.Tensor | None = None,
        stochastic: bool = True,
        dropout_seed: int | None = None,
    ) -> HighLevelSelection:
        history, mask = self._prepare_history(
            history_latents, history_padding_mask
        )
        goal = torch.as_tensor(
            goal_latent, dtype=torch.float32, device=self.device
        )
        if goal.shape != (history.size(0), LATENT_DIM):
            raise ValueError("goal_latent must be [B,192]")
        if dropout_seed is not None:
            torch.manual_seed(int(dropout_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(dropout_seed))

        # Candidates remain raw generator outputs without a transform hook.
        candidates = sample_subgoal_candidates(
            self.generator,
            history,
            mask,
            goal,
            num_candidates=self.num_candidates,
            stochastic=stochastic,
        )
        batch, count, latent_dim = candidates.shape
        current = history[:, -1]
        flat_candidates = candidates.reshape(batch * count, latent_dim)
        repeated_current = current.repeat_interleave(count, dim=0)
        repeated_goal = goal.repeat_interleave(count, dim=0)
        local_scores = self.global_potential(
            repeated_current, flat_candidates
        ).reshape(batch, count)
        goal_scores = self.global_potential(
            flat_candidates, repeated_goal
        ).reshape(batch, count)
        direct_score = self.global_potential(current, goal)

        direct_eligible = direct_score >= self.eta_3
        candidate_pass = local_scores >= self.eta_3
        candidate_coverage = candidate_pass.any(dim=1)
        ranked = goal_scores.masked_fill(~candidate_pass, -torch.inf).argmax(dim=1)
        selected_index = torch.where(
            candidate_coverage,
            ranked,
            -torch.ones_like(ranked),
        )
        selected_index = torch.where(
            direct_eligible,
            torch.full_like(selected_index, count),
            selected_index,
        )
        rows = torch.arange(batch, device=self.device)
        safe_index = selected_index.clamp(min=0, max=count - 1)
        selected = candidates[rows, safe_index]
        use_goal = direct_eligible | (selected_index < 0)
        selected = torch.where(use_goal.unsqueeze(1), goal, selected)
        return HighLevelSelection(
            selected_latent=selected,
            selected_index=selected_index,
            direct_goal_eligible=direct_eligible,
            direct_goal_selected=direct_eligible,
            candidate_coverage=candidate_coverage,
            fallback_to_flat_goal=(selected_index < 0),
            candidate_latents=candidates,
            local_competence_scores=local_scores,
            goal_potential_scores=goal_scores,
            direct_goal_score=direct_score,
            eta_3=self.eta_3,
        )


__all__ = [
    "GlobalReachabilitySelector",
    "HighLevelSelection",
]
