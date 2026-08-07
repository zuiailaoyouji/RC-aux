"""Stable low-level inference API for released RC-aux world models."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms


@dataclass(frozen=True)
class RCAuxProfile:
    """Environment-specific model and data conventions."""

    name: str
    dataset_name: str
    image_size: int
    action_dim: int
    model_step_env_steps: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.image_size < 1 or self.action_dim < 1 or self.model_step_env_steps < 1:
            raise ValueError("profile dimensions and step counts must be positive")
        if len(self.action_low) != self.action_dim:
            raise ValueError("action_low length must equal action_dim")
        if len(self.action_high) != self.action_dim:
            raise ValueError("action_high length must equal action_dim")

    @property
    def action_block_dim(self) -> int:
        return self.action_dim * self.model_step_env_steps


TWOROOM_PROFILE = RCAuxProfile(
    name="tworoom",
    dataset_name="tworoom",
    image_size=224,
    action_dim=2,
    model_step_env_steps=5,
    action_low=(-1.0, -1.0),
    action_high=(1.0, 1.0),
)


@dataclass(frozen=True)
class RCAuxPlannerConfig:
    """Planner settings with explicit model-step units."""

    planning_horizon_model_steps: int = 5
    execution_horizon_model_steps: int = 1
    num_samples: int = 300
    n_steps: int = 30
    topk: int = 30
    var_scale: float = 1.0
    batch_size: int = 1
    seed: int = 42
    warm_start: bool = True

    def __post_init__(self) -> None:
        if self.planning_horizon_model_steps < 1:
            raise ValueError("planning_horizon_model_steps must be positive")
        if not 1 <= self.execution_horizon_model_steps <= self.planning_horizon_model_steps:
            raise ValueError(
                "execution_horizon_model_steps must be in "
                "[1, planning_horizon_model_steps]"
            )
        if not 1 <= self.topk <= self.num_samples:
            raise ValueError("topk must be in [1, num_samples]")


@dataclass(frozen=True)
class ReachabilityDiagnostics:
    """Reachability outputs and metadata for experiment logging."""

    probabilities: torch.Tensor
    logits: torch.Tensor
    horizons_model_steps: torch.Tensor
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    probability_min: float
    probability_mean: float
    probability_max: float

    def to_log_dict(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostics, including per-query values."""

        return {
            "probabilities": self.probabilities.detach().cpu().tolist(),
            "logits": self.logits.detach().cpu().tolist(),
            "horizons_model_steps": (
                self.horizons_model_steps.detach().cpu().tolist()
            ),
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
            "query_shape": list(self.query_shape),
            "probability_min": self.probability_min,
            "probability_mean": self.probability_mean,
            "probability_max": self.probability_max,
        }


@dataclass(frozen=True)
class PlannerDiagnostics:
    """Planner metadata with explicit model-step and env-step units."""

    profile_name: str
    goal_mode: Literal["latent", "image"]
    goal_signature: str
    rollout_mode: Literal["rcaux_open_loop", "official_image_goal"]
    observation_history_model_steps: int
    history_action_blocks_model_steps: int | None
    predicted_future_latents_model_steps: int | None
    planning_horizon_model_steps: int
    execution_horizon_model_steps: int
    model_step_env_steps: int
    planning_horizon_env_steps: int
    execution_horizon_env_steps: int
    warm_start_source: Literal["none", "previous_plan", "explicit"]
    warm_start_reset: bool
    warm_start_reset_reason: str | None
    cem_num_samples: int
    cem_iterations: int
    cem_topk: int
    final_costs: np.ndarray
    final_cost_min: float
    final_cost_mean: float
    final_cost_max: float
    planning_time_seconds: float
    reachability_cost_enabled: bool
    reachability_cost_weight: float
    action_scaler_mean: tuple[float, ...]
    action_scaler_scale: tuple[float, ...]

    def to_log_dict(self) -> dict[str, Any]:
        """Return JSON-serializable planner diagnostics."""

        return {
            "profile_name": self.profile_name,
            "goal_mode": self.goal_mode,
            "goal_signature": self.goal_signature,
            "rollout_mode": self.rollout_mode,
            "observation_history_model_steps": (
                self.observation_history_model_steps
            ),
            "history_action_blocks_model_steps": (
                self.history_action_blocks_model_steps
            ),
            "predicted_future_latents_model_steps": (
                self.predicted_future_latents_model_steps
            ),
            "planning_horizon_model_steps": self.planning_horizon_model_steps,
            "execution_horizon_model_steps": self.execution_horizon_model_steps,
            "model_step_env_steps": self.model_step_env_steps,
            "planning_horizon_env_steps": self.planning_horizon_env_steps,
            "execution_horizon_env_steps": self.execution_horizon_env_steps,
            "warm_start_source": self.warm_start_source,
            "warm_start_reset": self.warm_start_reset,
            "warm_start_reset_reason": self.warm_start_reset_reason,
            "cem_num_samples": self.cem_num_samples,
            "cem_iterations": self.cem_iterations,
            "cem_topk": self.cem_topk,
            "final_costs": self.final_costs.tolist(),
            "final_cost_min": self.final_cost_min,
            "final_cost_mean": self.final_cost_mean,
            "final_cost_max": self.final_cost_max,
            "planning_time_seconds": self.planning_time_seconds,
            "reachability_cost_enabled": self.reachability_cost_enabled,
            "reachability_cost_weight": self.reachability_cost_weight,
            "action_scaler_mean": list(self.action_scaler_mean),
            "action_scaler_scale": list(self.action_scaler_scale),
        }


@dataclass(frozen=True)
class PlanResult:
    """Full planned trajectory plus the prefix that should be executed."""

    planned_actions_env_steps: np.ndarray
    actions_to_execute_env_steps: np.ndarray
    normalized_action_blocks: torch.Tensor
    diagnostics: PlannerDiagnostics


class _AdapterPlanningCost:
    """Route image goals and latent goals through their respective cost paths."""

    def __init__(self, adapter: "RCAuxAdapter") -> None:
        self.adapter = adapter

    def get_cost(
        self,
        info_dict: dict[str, torch.Tensor],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if "goal_latent" in info_dict:
            return self.adapter._cost_from_latent_goal(info_dict, action_candidates)
        return self.adapter.model.get_cost(info_dict, action_candidates)


class RCAuxAdapter:
    """Unified inference boundary around an RC-aux object checkpoint.

    Shape conventions are deliberately fixed:

    - encoded trajectories and dynamics: ``[B, T, D]``
    - reachability candidate sets: ``[B, N, D]``
    - individual batched latents and planner goals: ``[B, D]``
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        profile: RCAuxProfile,
        planner_config: RCAuxPlannerConfig | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        action_scaler: Any | None = None,
        use_reachability_cost: bool = True,
        reachability_cost_weight: float = 0.85,
    ) -> None:
        self.profile = profile
        self.planner_config = planner_config or RCAuxPlannerConfig()
        self.device = torch.device(device or self._module_device(model))
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

        default_cache = os.getenv("STABLEWM_HOME", "~/.stable_worldmodel")
        self.cache_dir = Path(cache_dir or default_cache).expanduser()
        self._action_scaler = action_scaler
        self._next_init: torch.Tensor | None = None
        self._goal_signature: str | None = None

        self.latent_dim = int(self.model.predictor.pos_embedding.size(-1))
        self.history_size_model_steps = int(
            self.model.predictor.pos_embedding.size(1)
        )
        head = getattr(self.model, "reachability_head", None)
        if head is None:
            raise RuntimeError("The loaded model does not contain a reachability head")
        self.max_reachability_horizon_model_steps = int(head.max_horizon)

        cfg = self.planner_config
        if (
            use_reachability_cost
            and cfg.planning_horizon_model_steps
            > self.max_reachability_horizon_model_steps
        ):
            raise ValueError(
                "planning_horizon_model_steps exceeds the reachability head's "
                f"maximum of {self.max_reachability_horizon_model_steps}"
            )

        self.model.use_reachability_cost = bool(use_reachability_cost)
        self.model.reachability_cost_weight = float(reachability_cost_weight)

        self._image_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=self.profile.image_size),
            ]
        )
        self._solver = swm.solver.CEMSolver(
            model=_AdapterPlanningCost(self),
            batch_size=cfg.batch_size,
            num_samples=cfg.num_samples,
            var_scale=cfg.var_scale,
            n_steps=cfg.n_steps,
            topk=cfg.topk,
            device=self.device,
            seed=cfg.seed,
        )

    @classmethod
    def from_checkpoint(
        cls,
        policy: str = "tworoom_rcaux/rcaux_tworoom",
        *,
        profile: RCAuxProfile = TWOROOM_PROFILE,
        cache_dir: str | Path | None = None,
        device: str | torch.device = "cuda",
        **kwargs: Any,
    ) -> "RCAuxAdapter":
        """Load an official object checkpoint through stable-worldmodel."""

        default_cache = os.getenv("STABLEWM_HOME", "~/.stable_worldmodel")
        root = Path(cache_dir or default_cache).expanduser()
        model = swm.policy.AutoCostModel(policy, cache_dir=root)
        return cls(
            model,
            profile=profile,
            device=device,
            cache_dir=root,
            **kwargs,
        )

    @staticmethod
    def _module_device(model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def default_planning_horizon_env_steps(self) -> int:
        """Default planning horizon converted to environment steps."""

        return (
            self.planner_config.planning_horizon_model_steps
            * self.profile.model_step_env_steps
        )

    @property
    def default_execution_horizon_env_steps(self) -> int:
        """Default execution horizon converted to environment steps."""

        return (
            self.planner_config.execution_horizon_model_steps
            * self.profile.model_step_env_steps
        )

    def _canonicalize_images(self, images: Any) -> torch.Tensor:
        value = torch.as_tensor(images)
        if value.ndim not in (3, 4, 5):
            raise ValueError(
                "images must be CHW/HWC, BCHW/BHWC, or BTCHW/BTHWC; "
                f"got {tuple(value.shape)}"
            )

        if value.ndim == 3:
            if value.shape[-1] in (1, 3, 4):
                value = value.permute(2, 0, 1)
            value = value.unsqueeze(0).unsqueeze(0)
        elif value.ndim == 4:
            if value.shape[-1] in (1, 3, 4):
                value = value.permute(0, 3, 1, 2)
            value = value.unsqueeze(1)
        elif value.shape[-1] in (1, 3, 4):
            value = value.permute(0, 1, 4, 2, 3)

        if value.size(2) == 4:
            value = value[:, :, :3]
        if value.size(2) != 3:
            raise ValueError(f"expected three image channels, got {value.size(2)}")
        return value.contiguous()

    def _preprocess_images(self, images: Any, *, preprocessed: bool) -> torch.Tensor:
        value = self._canonicalize_images(images)
        if preprocessed:
            expected_hw = (self.profile.image_size, self.profile.image_size)
            if value.shape[-2:] != expected_hw:
                raise ValueError(
                    f"preprocessed images must already have shape {expected_hw}"
                )
            return value.to(device=self.device, dtype=torch.float32)

        if value.is_floating_point():
            min_value = float(value.min())
            max_value = float(value.max())
            if min_value < 0.0 or max_value > 1.0:
                raise ValueError(
                    "floating-point images must be in [0, 1]; pass uint8 for [0, 255]"
                )

        batch, trajectory_steps = value.shape[:2]
        flat = value.reshape(batch * trajectory_steps, *value.shape[2:])
        processed = torch.stack([self._image_transform(frame) for frame in flat])
        return processed.reshape(
            batch, trajectory_steps, *processed.shape[1:]
        ).to(self.device)

    @torch.inference_mode()
    def encode_observation(
        self,
        observation: Any,
        *,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        """Return checkpoint latents in canonical ``[B, T, D]`` layout."""

        pixels = self._preprocess_images(observation, preprocessed=preprocessed)
        latent = self.model.encode({"pixels": pixels})["emb"]
        return latent.to(dtype=torch.float32)

    def _get_action_scaler(self) -> Any:
        if self._action_scaler is None:
            dataset = swm.data.HDF5Dataset(
                self.profile.dataset_name,
                keys_to_cache=["action"],
                cache_dir=self.cache_dir,
            )
            action = dataset.get_col_data("action")
            action = action[~np.isnan(action).any(axis=1)]
            self._action_scaler = StandardScaler().fit(action)
        return self._action_scaler

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        shape = actions.shape
        flat = actions.detach().cpu().numpy().reshape(-1, self.profile.action_dim)
        normalized = self._get_action_scaler().transform(flat)
        return torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=self.device,
        ).reshape(shape)

    def _action_blocks(
        self,
        actions: Any,
        *,
        normalized: bool,
        expected_batch: int | None = None,
    ) -> torch.Tensor:
        value = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim not in (3, 4):
            raise ValueError(
                "actions must be [B,L,A], [B,T,E,A], or [B,T,E*A]"
            )
        if expected_batch is not None and value.size(0) != expected_batch:
            raise ValueError(
                f"action batch {value.size(0)} does not match batch {expected_batch}"
            )

        env_steps = self.profile.model_step_env_steps
        action_dim = self.profile.action_dim
        flat_dim = self.profile.action_block_dim
        if value.ndim == 4:
            if value.shape[-2:] != (env_steps, action_dim):
                raise ValueError(
                    f"action blocks must end in ({env_steps}, {action_dim})"
                )
            low_level = value
        elif value.size(-1) == flat_dim:
            low_level = value.reshape(*value.shape[:-1], env_steps, action_dim)
        elif value.size(-1) == action_dim:
            if value.size(1) % env_steps:
                raise ValueError(
                    "low-level env-step action count must be divisible by "
                    f"model_step_env_steps={env_steps}"
                )
            low_level = value.reshape(value.size(0), -1, env_steps, action_dim)
        else:
            raise ValueError(
                f"action last dimension must be {action_dim} or {flat_dim}"
            )

        if not normalized:
            low_level = self._normalize_actions(low_level)
        return low_level.reshape(low_level.size(0), low_level.size(1), flat_dim)

    @torch.inference_mode()
    def predict_latents(
        self,
        history_latents: torch.Tensor,
        history_actions: Any,
        future_actions: Any | None = None,
        *,
        actions_normalized: bool = False,
    ) -> torch.Tensor:
        """Run open-loop dynamics and return ``[B, T_pred, D]``.

        ``history_actions`` is measured in model steps and aligns one-to-one
        with ``history_latents``. Its last block produces the first prediction.
        Each model-step action block contains ``profile.model_step_env_steps``
        low-level actions. ``future_actions`` provides subsequent blocks.
        """

        latents = torch.as_tensor(
            history_latents,
            dtype=torch.float32,
            device=self.device,
        )
        if latents.ndim != 3 or latents.size(-1) != self.latent_dim:
            raise ValueError(
                f"history_latents must be [B,T,{self.latent_dim}], "
                f"got {tuple(latents.shape)}"
            )

        history_blocks = self._action_blocks(
            history_actions,
            normalized=actions_normalized,
            expected_batch=latents.size(0),
        )
        if history_blocks.size(1) != latents.size(1):
            raise ValueError(
                "history_actions model-step count must match history_latents T"
            )

        history_action_emb = self.model.action_encoder(history_blocks)
        if future_actions is None:
            future_action_emb = history_action_emb.new_empty(
                latents.size(0), 0, history_action_emb.size(-1)
            )
        else:
            future_blocks = self._action_blocks(
                future_actions,
                normalized=actions_normalized,
                expected_batch=latents.size(0),
            )
            future_action_emb = self.model.action_encoder(future_blocks)

        return self.model.rollout_open_loop(
            latents,
            history_action_emb,
            future_action_emb,
            horizon=1 + future_action_emb.size(1),
            history_size=self.history_size_model_steps,
        ).to(dtype=torch.float32)

    def _broadcast_reachability_pairs(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for name, value in (("source", source), ("target", target)):
            if value.ndim not in (2, 3) or value.size(-1) != self.latent_dim:
                raise ValueError(
                    f"{name} must be [B,D] or [B,N,D] with D={self.latent_dim}"
                )
        if source.size(0) != target.size(0):
            raise ValueError("source and target batch sizes must match")

        if source.ndim == 2 and target.ndim == 2:
            return source, target
        if source.ndim == 2:
            source = source.unsqueeze(1).expand(-1, target.size(1), -1)
        elif target.ndim == 2:
            target = target.unsqueeze(1).expand(-1, source.size(1), -1)
        elif source.size(1) != target.size(1):
            raise ValueError("source and target candidate counts N must match")
        return source, target

    def _reachability_horizons(
        self,
        horizon_model_steps: int | torch.Tensor,
        query_shape: torch.Size,
    ) -> torch.Tensor:
        budget = torch.as_tensor(horizon_model_steps, device=self.device)
        if budget.is_floating_point() and not torch.equal(budget, budget.round()):
            raise ValueError("horizon_model_steps values must be integers")
        budget = budget.to(torch.long)

        if budget.ndim == 0:
            budget = budget.expand(query_shape)
        elif len(query_shape) == 2 and budget.shape == query_shape[:1]:
            budget = budget.unsqueeze(1).expand(query_shape)
        else:
            try:
                budget = torch.broadcast_to(budget, query_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"horizon_model_steps is not broadcastable to {tuple(query_shape)}"
                ) from exc

        maximum = self.max_reachability_horizon_model_steps
        if torch.any(budget < 1) or torch.any(budget > maximum):
            raise ValueError(
                f"horizon_model_steps must be in [1, {maximum}]"
            )
        return budget

    @torch.inference_mode()
    def reachability(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        horizon_model_steps: int | torch.Tensor,
        *,
        return_logits: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | ReachabilityDiagnostics:
        """Query ``[B,D]`` pairs or batched ``[B,N,D]`` candidate sets."""

        source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        source_shape = tuple(source.shape)
        target_shape = tuple(target.shape)
        source, target = self._broadcast_reachability_pairs(source, target)
        query_shape = source.shape[:-1]
        budget = self._reachability_horizons(horizon_model_steps, query_shape)

        logits = self.model.score_reachability(source, target, budget)
        probabilities = torch.sigmoid(logits)
        if return_diagnostics:
            return ReachabilityDiagnostics(
                probabilities=probabilities,
                logits=logits,
                horizons_model_steps=budget,
                source_shape=source_shape,
                target_shape=target_shape,
                query_shape=tuple(query_shape),
                probability_min=float(probabilities.min()),
                probability_mean=float(probabilities.mean()),
                probability_max=float(probabilities.max()),
            )
        return logits if return_logits else probabilities

    def reset_warm_start(self) -> None:
        """Discard cached planner actions without changing the active goal."""

        self._next_init = None

    def reset_planner(self, *, seed: int | None = None) -> None:
        """Discard planner state and optionally reset the CEM generator."""

        self._next_init = None
        self._goal_signature = None
        if seed is not None:
            self._solver.torch_gen.manual_seed(int(seed))

    @staticmethod
    def _tensor_signature(mode: str, value: torch.Tensor) -> str:
        cpu = value.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
        return f"{mode}:{tuple(cpu.shape)}:{cpu.dtype}:{digest}"

    def _prepare_goal(
        self,
        observation_batch: int,
        *,
        goal_latent: Any | None,
        goal_image: Any | None,
        images_preprocessed: bool,
    ) -> tuple[Literal["latent", "image"], str, torch.Tensor]:
        if (goal_latent is None) == (goal_image is None):
            raise ValueError("exactly one of goal_latent or goal_image must be provided")

        if goal_latent is not None:
            goal = torch.as_tensor(
                goal_latent,
                dtype=torch.float32,
                device=self.device,
            )
            if goal.ndim != 2 or goal.size(-1) != self.latent_dim:
                raise ValueError(
                    f"goal_latent must be [B,{self.latent_dim}], got {tuple(goal.shape)}"
                )
            mode: Literal["latent", "image"] = "latent"
        else:
            goal = self._preprocess_images(
                goal_image,
                preprocessed=images_preprocessed,
            )
            if goal.size(1) != 1:
                raise ValueError("goal_image must contain exactly one image per batch item")
            mode = "image"

        if goal.size(0) == 1 and observation_batch > 1:
            goal = goal.expand(observation_batch, *goal.shape[1:])
        if goal.size(0) != observation_batch:
            raise ValueError("observation and goal batch sizes must match")
        return mode, self._tensor_signature(mode, goal), goal

    def _cost_from_latent_goal(
        self,
        info_dict: dict[str, torch.Tensor],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        info = dict(info_dict)
        goal_latent = info.pop("goal_latent").to(self.device)
        history_action_blocks = info.pop("history_action_blocks").to(self.device)
        if goal_latent.ndim == 3:
            goal_latent = goal_latent[:, :1]
        elif goal_latent.ndim == 2:
            goal_latent = goal_latent.unsqueeze(1)
        else:
            raise ValueError("expanded goal_latent must be [B,N,D] or [B,D]")

        batch, samples, horizon = action_candidates.shape[:3]
        initial = {
            key: value[:, 0].to(self.device)
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        encoded = self.model.encode(initial)
        history_latents = encoded["emb"]
        history_length = history_latents.size(1)
        history_latents = (
            history_latents.unsqueeze(1)
            .expand(batch, samples, -1, -1)
            .reshape(batch * samples, history_length, self.latent_dim)
        )

        candidate_blocks = action_candidates.reshape(
            batch * samples,
            horizon,
            self.profile.action_block_dim,
        )
        candidate_action_emb = self.model.action_encoder(candidate_blocks)
        if history_length > 1:
            past_blocks = history_action_blocks.reshape(
                batch * samples,
                history_length - 1,
                self.profile.action_block_dim,
            )
            past_action_emb = self.model.action_encoder(past_blocks)
        else:
            past_action_emb = candidate_action_emb.new_empty(
                batch * samples,
                0,
                candidate_action_emb.size(-1),
            )

        aligned_history_action_emb = torch.cat(
            [past_action_emb, candidate_action_emb[:, :1]],
            dim=1,
        )
        future_latents = self.model.rollout_open_loop(
            history_latents,
            aligned_history_action_emb,
            candidate_action_emb[:, 1:],
            horizon=horizon,
            history_size=self.history_size_model_steps,
        )
        source_and_future = torch.cat(
            [history_latents[:, -1:], future_latents],
            dim=1,
        ).reshape(batch, samples, horizon + 1, self.latent_dim)
        cost = self.model.criterion(
            {
                "predicted_emb": source_and_future,
                "goal_emb": goal_latent,
            }
        )

        action_l2_weight = float(
            getattr(self.model, "action_l2_cost_weight", 0.0)
        )
        if action_l2_weight > 0.0:
            action_l2 = action_candidates.square().mean(
                dim=tuple(range(2, action_candidates.ndim))
            )
            cost = cost + action_l2_weight * action_l2

        smooth_weight = float(
            getattr(self.model, "action_smooth_cost_weight", 0.0)
        )
        if smooth_weight > 0.0 and action_candidates.size(2) > 1:
            differences = action_candidates[:, :, 1:] - action_candidates[:, :, :-1]
            smoothness = differences.square().mean(
                dim=tuple(range(2, differences.ndim))
            )
            cost = cost + smooth_weight * smoothness
        return cost

    def _planner_action_space(self, batch: int) -> gym.spaces.Box:
        low = np.broadcast_to(
            np.asarray(self.profile.action_low, dtype=np.float32),
            (batch, self.profile.action_dim),
        )
        high = np.broadcast_to(
            np.asarray(self.profile.action_high, dtype=np.float32),
            (batch, self.profile.action_dim),
        )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _planning_history_action_blocks(
        self,
        pixels: torch.Tensor,
        history_action_blocks: Any | None,
        *,
        normalized: bool,
    ) -> torch.Tensor:
        history_length = pixels.size(1)
        if history_length > self.history_size_model_steps:
            raise ValueError(
                f"observation history L={history_length} exceeds the model's "
                f"maximum L={self.history_size_model_steps}"
            )
        required_blocks = history_length - 1
        if required_blocks == 0:
            if (
                history_action_blocks is not None
                and torch.as_tensor(history_action_blocks).numel() > 0
            ):
                raise ValueError(
                    "history_action_blocks must be empty when observation L=1"
                )
            return pixels.new_empty(
                pixels.size(0),
                0,
                self.profile.action_block_dim,
            )
        if history_action_blocks is None:
            raise ValueError(
                f"observation history L={history_length} requires exactly "
                f"L-1={required_blocks} history action blocks"
            )
        blocks = self._action_blocks(
            history_action_blocks,
            normalized=normalized,
            expected_batch=pixels.size(0),
        )
        if blocks.size(1) != required_blocks:
            raise ValueError(
                f"observation history L={history_length} requires exactly "
                f"L-1={required_blocks} history action blocks"
            )
        return blocks

    @staticmethod
    def _scalar_model_step_count(value: Any, *, name: str) -> int:
        count = torch.as_tensor(value)
        if count.numel() != 1:
            raise ValueError(f"{name} must be a scalar integer")
        if count.dtype == torch.bool or count.is_complex():
            raise ValueError(f"{name} must be an integer number of model steps")
        if count.is_floating_point() and not torch.equal(count, count.round()):
            raise ValueError(f"{name} must be an integer number of model steps")
        return int(count.item())

    def _resolve_plan_horizons(
        self,
        planning_horizon_model_steps: Any | None,
        execution_horizon_model_steps: Any | None,
    ) -> tuple[int, int]:
        cfg = self.planner_config
        planning = (
            cfg.planning_horizon_model_steps
            if planning_horizon_model_steps is None
            else self._scalar_model_step_count(
                planning_horizon_model_steps,
                name="planning_horizon_model_steps",
            )
        )
        execution = (
            cfg.execution_horizon_model_steps
            if execution_horizon_model_steps is None
            else self._scalar_model_step_count(
                execution_horizon_model_steps,
                name="execution_horizon_model_steps",
            )
        )
        if planning < 1:
            raise ValueError("planning_horizon_model_steps must be positive")
        if not 1 <= execution <= planning:
            raise ValueError(
                "execution_horizon_model_steps must be in "
                "[1, planning_horizon_model_steps]"
            )
        if (
            self.model.use_reachability_cost
            and planning > self.max_reachability_horizon_model_steps
        ):
            raise ValueError(
                "planning_horizon_model_steps exceeds the reachability head's "
                f"maximum of {self.max_reachability_horizon_model_steps}"
            )
        return planning, execution

    @torch.inference_mode()
    def plan(
        self,
        observation: Any,
        *,
        goal_latent: Any | None = None,
        goal_image: Any | None = None,
        initial_action_blocks: Any | None = None,
        history_action_blocks: Any | None = None,
        history_actions_normalized: bool = False,
        planning_horizon_model_steps: Any | None = None,
        execution_horizon_model_steps: Any | None = None,
        images_preprocessed: bool = False,
        force_reset_warm_start: bool = False,
    ) -> PlanResult:
        """Plan toward exactly one latent or image goal.

        Latent goals are the primary hierarchy-facing path. Image goals retain
        the official policy path for regression tests. High-level ``tau_star``
        initializes ``h_rem``; pass the current ``h_rem`` as
        ``planning_horizon_model_steps`` on every closed-loop replan.
        """

        planning_horizon, execution_horizon = self._resolve_plan_horizons(
            planning_horizon_model_steps,
            execution_horizon_model_steps,
        )

        pixels = self._preprocess_images(
            observation,
            preprocessed=images_preprocessed,
        )
        batch = pixels.size(0)
        goal_mode, goal_signature, goal = self._prepare_goal(
            batch,
            goal_latent=goal_latent,
            goal_image=goal_image,
            images_preprocessed=images_preprocessed,
        )

        previous_signature = self._goal_signature
        goal_changed = (
            previous_signature is not None and previous_signature != goal_signature
        )
        warm_start_reset = force_reset_warm_start or goal_changed
        reset_reason = None
        if force_reset_warm_start:
            reset_reason = "forced"
        elif goal_changed:
            reset_reason = "subgoal_changed"
        if warm_start_reset:
            self.reset_warm_start()
        self._goal_signature = goal_signature

        cfg = self.planner_config
        plan_cfg = swm.PlanConfig(
            horizon=planning_horizon,
            receding_horizon=execution_horizon,
            history_len=pixels.size(1),
            action_block=self.profile.model_step_env_steps,
            warm_start=cfg.warm_start,
        )
        self._solver.configure(
            action_space=self._planner_action_space(batch),
            n_envs=batch,
            config=plan_cfg,
        )

        if initial_action_blocks is not None:
            init = self._action_blocks(
                initial_action_blocks,
                normalized=False,
                expected_batch=batch,
            ).cpu()
            if init.size(1) > planning_horizon:
                raise ValueError(
                    "initial_action_blocks exceeds planning_horizon_model_steps"
                )
            warm_start_source: Literal["none", "previous_plan", "explicit"] = (
                "explicit"
            )
        elif cfg.warm_start and self._next_init is not None:
            init = self._next_init[:, :planning_horizon]
            warm_start_source = "previous_plan"
        else:
            init = None
            warm_start_source = "none"

        info = {"pixels": pixels}
        if goal_mode == "latent":
            info["goal_latent"] = goal
            prepared_history_action_blocks = (
                self._planning_history_action_blocks(
                    pixels,
                    history_action_blocks,
                    normalized=history_actions_normalized,
                )
            )
            info["history_action_blocks"] = prepared_history_action_blocks
            rollout_mode: Literal[
                "rcaux_open_loop", "official_image_goal"
            ] = "rcaux_open_loop"
            history_action_blocks_model_steps = (
                prepared_history_action_blocks.size(1)
            )
            predicted_future_latents_model_steps: int | None = (
                planning_horizon
            )
        else:
            info["goal"] = goal
            rollout_mode = "official_image_goal"
            history_action_blocks_model_steps = None
            predicted_future_latents_model_steps = None

        scaler = self._get_action_scaler()
        started = time.perf_counter()
        outputs = self._solver(info, init_action=init)
        planning_time = time.perf_counter() - started
        normalized_blocks = outputs["actions"]

        if cfg.warm_start:
            remaining = normalized_blocks[:, execution_horizon:]
            # CEMSolver pads a partial warm start on CPU before moving the
            # completed distribution to its configured solver device.
            self._next_init = remaining.cpu() if remaining.size(1) else None
        else:
            self._next_init = None

        planning_horizon_env_steps = (
            planning_horizon * self.profile.model_step_env_steps
        )
        execution_horizon_env_steps = (
            execution_horizon * self.profile.model_step_env_steps
        )
        planned_normalized = normalized_blocks.reshape(
            batch,
            planning_horizon_env_steps,
            self.profile.action_dim,
        )
        flat = planned_normalized.numpy().reshape(-1, self.profile.action_dim)
        planned_actions = scaler.inverse_transform(flat).reshape(
            planned_normalized.shape
        )
        planned_actions = planned_actions.astype(np.float32, copy=False)
        actions_to_execute = planned_actions[:, :execution_horizon_env_steps]

        costs = np.asarray(outputs["costs"], dtype=np.float32)
        scaler_mean = tuple(float(value) for value in scaler.mean_)
        scaler_scale = tuple(float(value) for value in scaler.scale_)
        diagnostics = PlannerDiagnostics(
            profile_name=self.profile.name,
            goal_mode=goal_mode,
            goal_signature=goal_signature,
            rollout_mode=rollout_mode,
            observation_history_model_steps=pixels.size(1),
            history_action_blocks_model_steps=(
                history_action_blocks_model_steps
            ),
            predicted_future_latents_model_steps=(
                predicted_future_latents_model_steps
            ),
            planning_horizon_model_steps=planning_horizon,
            execution_horizon_model_steps=execution_horizon,
            model_step_env_steps=self.profile.model_step_env_steps,
            planning_horizon_env_steps=planning_horizon_env_steps,
            execution_horizon_env_steps=execution_horizon_env_steps,
            warm_start_source=warm_start_source,
            warm_start_reset=warm_start_reset,
            warm_start_reset_reason=reset_reason,
            cem_num_samples=cfg.num_samples,
            cem_iterations=cfg.n_steps,
            cem_topk=cfg.topk,
            final_costs=costs,
            final_cost_min=float(costs.min()),
            final_cost_mean=float(costs.mean()),
            final_cost_max=float(costs.max()),
            planning_time_seconds=planning_time,
            reachability_cost_enabled=bool(self.model.use_reachability_cost),
            reachability_cost_weight=float(self.model.reachability_cost_weight),
            action_scaler_mean=scaler_mean,
            action_scaler_scale=scaler_scale,
        )
        return PlanResult(
            planned_actions_env_steps=planned_actions,
            actions_to_execute_env_steps=actions_to_execute,
            normalized_action_blocks=normalized_blocks,
            diagnostics=diagnostics,
        )

    def plan_to_latent(
        self,
        observation: Any,
        goal_latent: Any,
        **kwargs: Any,
    ) -> PlanResult:
        """Explicit latent-goal planner path used by hierarchical policies."""

        return self.plan(observation, goal_latent=goal_latent, **kwargs)

    def plan_to_image(
        self,
        observation: Any,
        goal_image: Any,
        **kwargs: Any,
    ) -> PlanResult:
        """Official image-goal planner path retained for regression tests."""

        return self.plan(observation, goal_image=goal_image, **kwargs)


__all__ = [
    "PlanResult",
    "PlannerDiagnostics",
    "RCAuxAdapter",
    "RCAuxPlannerConfig",
    "RCAuxProfile",
    "ReachabilityDiagnostics",
    "TWOROOM_PROFILE",
]
