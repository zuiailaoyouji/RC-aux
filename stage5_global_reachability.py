"""Stage 5A discounted witnessed hitting-time potential."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


LATENT_DIM = 192
HIDDEN_DIMS = (256, 128)
MAX_DISTANCE = 20
GAMMA = 0.9
RANK_WEIGHT = 0.1


class GlobalHittingTimePotential(nn.Module):
    """Predict a discounted witnessed hitting-time potential in (0, 1)."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("latent_dim and hidden_dims must be positive")
        dimensions = (3 * latent_dim, *hidden_dims, 1)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.network = nn.Sequential(*layers)

    def forward(self, source: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if source.shape != goal.shape:
            raise ValueError("source and goal must have identical shapes")
        if source.ndim != 2 or source.size(-1) != self.latent_dim:
            raise ValueError(
                f"source and goal must be [B,{self.latent_dim}]"
            )
        features = torch.cat([source, goal, source - goal], dim=-1)
        return torch.sigmoid(self.network(features).squeeze(-1))


def temporal_ranking_loss(
    same_source_near: torch.Tensor,
    same_source_far: torch.Tensor,
    same_goal_near: torch.Tensor,
    same_goal_far: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic loss requiring every near score to exceed far."""

    tensors = (
        same_source_near,
        same_source_far,
        same_goal_near,
        same_goal_far,
    )
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("ranking scores must be one-dimensional")
    if len({len(value) for value in tensors}) != 1:
        raise ValueError("ranking score batches must have the same length")
    same_source = F.softplus(same_source_far - same_source_near).mean()
    same_goal = F.softplus(same_goal_far - same_goal_near).mean()
    return 0.5 * (same_source + same_goal)


def load_global_hitting_potential(
    path,
    *,
    device: torch.device,
) -> tuple[GlobalHittingTimePotential, dict[str, Any]]:
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )
    model = GlobalHittingTimePotential(
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint
