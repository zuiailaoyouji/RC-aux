"""High-level generator utilities plus archived Stage 4 protocol helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


TAU_MODEL_STEPS = 3
MODEL_STEP_ENV_STEPS = 5
LATENT_DIM = 192
MAX_HISTORY = 3


class HighLevelSubgoalGenerator(nn.Module):
    """Two-layer Transformer predicting a normalized latent residual."""

    def __init__(
        self,
        *,
        latent_dim: int = LATENT_DIM,
        model_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_history: int = MAX_HISTORY,
        tau_model_steps: int = TAU_MODEL_STEPS,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("Stage 4 requires exactly two Transformer layers")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if max_history < 1 or tau_model_steps < 1:
            raise ValueError("history and duration must be positive")

        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.num_layers = int(num_layers)
        self.dropout_probability = float(dropout)
        self.max_history = int(max_history)
        self.tau_model_steps = int(tau_model_steps)

        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_history + 3, model_dim)
        )
        self.history_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.duration_embedding = nn.Embedding(tau_model_steps + 1, model_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.residual_head = nn.Linear(model_dim, latent_dim)
        self.register_buffer("residual_mean", torch.zeros(latent_dim))
        self.register_buffer("residual_std", torch.ones(latent_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.history_type, std=0.02)
        nn.init.normal_(self.goal_type, std=0.02)
        nn.init.normal_(self.query_token, std=0.02)

    def set_residual_normalizer(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (self.latent_dim,) or std.shape != (self.latent_dim,):
            raise ValueError("residual statistics must be [D]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("residual statistics must be finite")
        self.residual_mean.copy_(mean)
        self.residual_std.copy_(std.clamp_min(1.0e-6))

    def forward(
        self,
        history_latents: torch.Tensor,
        history_padding_mask: torch.Tensor,
        goal_latent: torch.Tensor,
        duration_model_steps: torch.Tensor | int = TAU_MODEL_STEPS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.as_tensor(
            history_latents,
            dtype=torch.float32,
            device=self.residual_mean.device,
        )
        mask = torch.as_tensor(
            history_padding_mask,
            dtype=torch.bool,
            device=history.device,
        )
        goal = torch.as_tensor(
            goal_latent,
            dtype=torch.float32,
            device=history.device,
        )
        if history.ndim != 3 or history.shape[1:] != (
            self.max_history,
            self.latent_dim,
        ):
            raise ValueError(
                f"history_latents must be [B,{self.max_history},{self.latent_dim}]"
            )
        if mask.shape != history.shape[:2]:
            raise ValueError("history_padding_mask must be [B,Lmax]")
        if goal.shape != (history.size(0), self.latent_dim):
            raise ValueError(f"goal_latent must be [B,{self.latent_dim}]")
        if torch.any(mask[:, -1]):
            raise ValueError("the current latent must be the final history token")

        duration = torch.as_tensor(duration_model_steps, device=history.device)
        if duration.ndim == 0:
            duration = duration.expand(history.size(0))
        if duration.shape != (history.size(0),):
            raise ValueError("duration_model_steps must be scalar or [B]")
        if duration.is_floating_point() and not torch.equal(
            duration, duration.round()
        ):
            raise ValueError("duration_model_steps must contain integers")
        duration = duration.to(torch.long)
        if torch.any(duration != self.tau_model_steps):
            raise ValueError(
                f"Stage 4 duration is fixed to {self.tau_model_steps} model steps"
            )

        history_tokens = self.latent_projection(history) + self.history_type
        goal_token = self.latent_projection(goal).unsqueeze(1) + self.goal_type
        duration_token = self.duration_embedding(duration).unsqueeze(1)
        query = self.query_token.expand(history.size(0), -1, -1)
        tokens = torch.cat(
            [history_tokens, goal_token, duration_token, query], dim=1
        )
        tokens = tokens + self.position_embedding
        special_mask = torch.zeros(
            history.size(0), 3, dtype=torch.bool, device=history.device
        )
        token_mask = torch.cat([mask, special_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=token_mask)
        normalized_residual = self.residual_head(encoded[:, -1])
        residual = (
            normalized_residual * self.residual_std + self.residual_mean
        )
        current = history[:, -1]
        subgoal = current + residual
        return subgoal, normalized_residual


def enable_generator_dropout_only(model: nn.Module) -> None:
    """Enable generator dropout without changing trainable model state."""

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, nn.TransformerEncoderLayer):
            # Disable the fused eval fast path so the enabled dropout modules
            # and attention-probability dropout below are actually evaluated.
            module.training = True
        elif isinstance(module, nn.MultiheadAttention):
            # Attention-probability dropout is controlled directly by the
            # MultiheadAttention training flag rather than an nn.Dropout child.
            module.training = True


@torch.inference_mode()
def sample_subgoal_candidates(
    model: HighLevelSubgoalGenerator,
    history_latents: torch.Tensor,
    history_padding_mask: torch.Tensor,
    goal_latent: torch.Tensor,
    *,
    num_candidates: int,
    stochastic: bool,
) -> torch.Tensor:
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    if stochastic:
        enable_generator_dropout_only(model)
    else:
        model.eval()
    batch = history_latents.size(0)
    repeated_history = history_latents.repeat_interleave(num_candidates, dim=0)
    repeated_mask = history_padding_mask.repeat_interleave(num_candidates, dim=0)
    repeated_goal = goal_latent.repeat_interleave(num_candidates, dim=0)
    subgoal, _ = model(repeated_history, repeated_mask, repeated_goal)
    model.eval()
    return subgoal.reshape(batch, num_candidates, model.latent_dim)


def load_latent_cache(path: Path) -> dict[str, Any]:
    cache_path = path.expanduser().resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    if metadata.get("model_step_env_steps") != MODEL_STEP_ENV_STEPS:
        raise ValueError("latent cache model-step size does not match Stage 4")
    if "train" not in cache or "test" not in cache:
        raise ValueError("latent cache must contain train and test trajectories")
    return cache


def split_cached_episodes(
    cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    first_half = cache["train"]
    second_half = cache["test"]
    train = [episode for episode in first_half if episode["episode_index"] < 4000]
    validation = [
        episode
        for episode in first_half
        if 4000 <= episode["episode_index"] < 5000
    ]
    test = [
        episode
        for episode in second_half
        if 5000 <= episode["episode_index"] < 10000
    ]
    ids = [
        {episode["episode_index"] for episode in split}
        for split in (train, validation, test)
    ]
    if not train or not validation or not test:
        raise ValueError("all Stage 4 episode splits must be nonempty")
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Stage 4 episode splits overlap")
    return train, validation, test


@dataclass(frozen=True)
class GeneratorSampleRef:
    episode_list_index: int
    source_index: int
    target_index: int


def build_generator_sample_refs(
    episodes: list[dict[str, Any]],
    *,
    tau_model_steps: int = TAU_MODEL_STEPS,
) -> list[GeneratorSampleRef]:
    future_env_steps = tau_model_steps * MODEL_STEP_ENV_STEPS
    refs = []
    for episode_list_index, episode in enumerate(episodes):
        rows = episode["rows"].tolist()
        row_to_index = {int(row): index for index, row in enumerate(rows)}
        terminal_index = len(rows) - 1
        for source_index in range(terminal_index):
            target_row = int(rows[source_index]) + future_env_steps
            target_index = row_to_index.get(target_row)
            if target_index is None or target_index >= terminal_index + 1:
                continue
            refs.append(
                GeneratorSampleRef(
                    episode_list_index=episode_list_index,
                    source_index=source_index,
                    target_index=target_index,
                )
            )
    return refs


class GeneratorTrajectoryDataset(Dataset):
    def __init__(
        self,
        episodes: list[dict[str, Any]],
        refs: list[GeneratorSampleRef],
        *,
        max_history: int = MAX_HISTORY,
    ) -> None:
        self.episodes = episodes
        self.refs = refs
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_list_index]
        latents = episode["latents"].to(torch.float32)
        history_start = max(0, ref.source_index - self.max_history + 1)
        history = latents[history_start : ref.source_index + 1]
        padded = torch.zeros(self.max_history, latents.size(-1))
        padding_mask = torch.ones(self.max_history, dtype=torch.bool)
        padded[-len(history) :] = history
        padding_mask[-len(history) :] = False
        current = latents[ref.source_index]
        target = latents[ref.target_index]
        return {
            "history_latents": padded,
            "history_padding_mask": padding_mask,
            "goal_latent": latents[-1],
            "current_latent": current,
            "target_latent": target,
            "target_residual": target - current,
            "episode_index": int(episode["episode_index"]),
            "source_row": int(episode["rows"][ref.source_index]),
            "target_row": int(episode["rows"][ref.target_index]),
        }


def compute_residual_statistics(
    dataset: GeneratorTrajectoryDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    total = torch.zeros(LATENT_DIM, dtype=torch.float64)
    total_square = torch.zeros(LATENT_DIM, dtype=torch.float64)
    for index in range(len(dataset)):
        residual = dataset[index]["target_residual"].to(torch.float64)
        total += residual
        total_square += residual.square()
        count += 1
    if count < 2:
        raise ValueError("at least two training residuals are required")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0.0)
    return mean.to(torch.float32), variance.sqrt().clamp_min(1.0e-6).to(
        torch.float32
    )


def generator_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    smooth_l1 = torch.nn.functional.smooth_l1_loss(predicted, target)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        predicted, target, dim=-1
    ).mean()
    return smooth_l1 + 0.1 * cosine, smooth_l1, cosine


def atomic_torch_save(value: Any, path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)


def generator_checkpoint(
    model: HighLevelSubgoalGenerator,
    *,
    seed: int,
    epoch: int,
    validation_smooth_l1: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_type": "HighLevelSubgoalGenerator",
        "config": {
            "latent_dim": model.latent_dim,
            "model_dim": model.model_dim,
            "num_heads": model.num_heads,
            "ffn_dim": model.ffn_dim,
            "num_layers": model.num_layers,
            "dropout": model.dropout_probability,
            "max_history": model.max_history,
            "tau_model_steps": model.tau_model_steps,
        },
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "seed": int(seed),
        "epoch": int(epoch),
        "validation_smooth_l1": float(validation_smooth_l1),
        "metadata": metadata,
    }


def load_generator_checkpoint(
    path: Path,
    *,
    device: str | torch.device,
) -> tuple[HighLevelSubgoalGenerator, dict[str, Any]]:
    checkpoint = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_type") != "HighLevelSubgoalGenerator":
        raise ValueError("not a Stage 4 generator checkpoint")
    model = HighLevelSubgoalGenerator(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def left_padded_history(
    latents: Iterable[torch.Tensor],
    *,
    latent_dim: int = LATENT_DIM,
    max_history: int = MAX_HISTORY,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [torch.as_tensor(value, dtype=torch.float32) for value in latents]
    values = values[-max_history:]
    if not values:
        raise ValueError("history cannot be empty")
    if any(value.shape != (latent_dim,) for value in values):
        raise ValueError(f"each history latent must be [{latent_dim}]")
    history = torch.zeros(1, max_history, latent_dim)
    mask = torch.ones(1, max_history, dtype=torch.bool)
    history[0, -len(values) :] = torch.stack(values)
    mask[0, -len(values) :] = False
    return history, mask


def distribution_summary(values: np.ndarray | torch.Tensor) -> dict[str, float]:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


__all__ = [
    "GeneratorSampleRef",
    "GeneratorTrajectoryDataset",
    "HighLevelSubgoalGenerator",
    "LATENT_DIM",
    "MAX_HISTORY",
    "MODEL_STEP_ENV_STEPS",
    "TAU_MODEL_STEPS",
    "atomic_torch_save",
    "build_generator_sample_refs",
    "compute_residual_statistics",
    "distribution_summary",
    "enable_generator_dropout_only",
    "generator_checkpoint",
    "generator_loss",
    "left_padded_history",
    "load_generator_checkpoint",
    "load_latent_cache",
    "sample_subgoal_candidates",
    "split_cached_episodes",
]
