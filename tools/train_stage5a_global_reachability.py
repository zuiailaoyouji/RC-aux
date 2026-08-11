#!/usr/bin/env python3
"""Train and validate the Stage 5A discounted witnessed hitting potential."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcaux_adapter import RCAuxAdapter, TWOROOM_PROFILE
from stage4_generator import load_latent_cache
from stage5_global_reachability import (
    GAMMA,
    HIDDEN_DIMS,
    LATENT_DIM,
    MAX_DISTANCE,
    RANK_WEIGHT,
    GlobalHittingTimePotential,
    temporal_ranking_loss,
)
from tools.evaluate_rc_filter_only import (
    CROSS_TRAJECTORY_STRICT,
    SAME_TRAJECTORY_STRICT,
    WITHIN_BUDGET,
    build_strict_candidate_pool,
    environment_speed,
    verify_within_budget_witness,
)


FORMAL_SEEDS = (3072, 3073, 3074)
CALIBRATION_RANGE = (4000, 4500)
HELDOUT_RANGE = (4500, 5000)
TRAIN_RANGE = (0, 4000)
LOCAL_CATEGORIES = (
    WITHIN_BUDGET,
    SAME_TRAJECTORY_STRICT,
    CROSS_TRAJECTORY_STRICT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5A discounted witnessed hitting-time potential."
    )
    parser.add_argument(
        "--latent-cache",
        type=Path,
        default=Path("outputs/stage3_progress_latents.pt"),
    )
    parser.add_argument("--policy", default="tworoom_rcaux/rcaux_tworoom")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/sxw/work/datasets/stable-wm"),
    )
    parser.add_argument("--dataset", default="tworoom.h5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="3072,3073,3074")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=65520)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=RANK_WEIGHT)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--max-distance", type=int, default=MAX_DISTANCE)
    parser.add_argument("--candidates-per-category", type=int, default=2)
    parser.add_argument("--t-env-step", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--success-radius", type=float, default=16.0)
    parser.add_argument("--witness-state-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--max-replay-pixel-diff", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--max-calibration-fpr", type=float, default=0.05)
    parser.add_argument("--min-calibration-precision", type=float, default=0.95)
    parser.add_argument(
        "--local-label-cache",
        type=Path,
        default=Path("outputs/stage5a_local_tau3_labels.pt"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/stage5a_global_reachability_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage5a_global_reachability.json"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate split and temporal-pair availability without training",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if seeds not in ((3072,), FORMAL_SEEDS):
        raise ValueError("seeds must be 3072 pilot or 3072,3073,3074")
    return seeds


def validate_args(args: argparse.Namespace) -> None:
    if not args.validate_only and args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    positive = (
        "epochs",
        "patience",
        "pairs_per_epoch",
        "batch_size",
        "eval_batch_size",
        "encode_batch_size",
        "candidates_per_category",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("epoch, patience, pair, batch, and candidate values must be positive")
    if args.pairs_per_epoch % MAX_DISTANCE:
        raise ValueError("pairs-per-epoch must be divisible by 20")
    if args.max_distance != MAX_DISTANCE:
        raise ValueError("Stage 5A fixes max-distance to 20")
    if not np.isclose(args.gamma, GAMMA, atol=0.0, rtol=0.0):
        raise ValueError("Stage 5A fixes gamma=0.9")
    if not np.isclose(args.lambda_rank, RANK_WEIGHT, atol=0.0, rtol=0.0):
        raise ValueError("Stage 5A fixes lambda-rank=0.1")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("optimizer values are invalid")
    if args.t_env_step != 0:
        raise ValueError("Stage 5A competence labels fix t-env-step=0")
    if not np.isclose(args.success_radius, 16.0):
        raise ValueError("TwoRoom local labels fix success-radius=16")
    for name in ("max_calibration_fpr", "min_calibration_precision"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")


def temporal_splits(
    cache: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Use only cache['train']; Stage 5A never inspects cache['test']."""

    first_half = cache["train"]
    train = [
        episode
        for episode in first_half
        if TRAIN_RANGE[0] <= int(episode["episode_index"]) < TRAIN_RANGE[1]
    ]
    calibration = [
        episode
        for episode in first_half
        if CALIBRATION_RANGE[0]
        <= int(episode["episode_index"])
        < CALIBRATION_RANGE[1]
    ]
    heldout = [
        episode
        for episode in first_half
        if HELDOUT_RANGE[0]
        <= int(episode["episode_index"])
        < HELDOUT_RANGE[1]
    ]
    if not train or not calibration or not heldout:
        raise ValueError("Stage 5A train/calibration/heldout splits must be nonempty")
    ids = [
        {int(episode["episode_index"]) for episode in split}
        for split in (train, calibration, heldout)
    ]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Stage 5A episode splits overlap")
    return train, calibration, heldout


def build_temporal_pair_buckets(
    episodes: list[dict[str, Any]],
    *,
    max_distance: int = MAX_DISTANCE,
) -> dict[int, np.ndarray]:
    buckets: dict[int, list[tuple[int, int, int]]] = {
        distance: [] for distance in range(1, max_distance + 1)
    }
    for episode_position, episode in enumerate(episodes):
        count = len(episode["latents"])
        if episode["latents"].ndim != 2 or episode["latents"].size(1) != LATENT_DIM:
            raise ValueError("trajectory latents must be raw [T,192]")
        for distance in range(1, min(max_distance, count - 1) + 1):
            buckets[distance].extend(
                (episode_position, source, source + distance)
                for source in range(count - distance)
            )
    arrays = {
        distance: np.asarray(refs, dtype=np.int64).reshape(-1, 3)
        for distance, refs in buckets.items()
    }
    if any(len(refs) == 0 for refs in arrays.values()):
        raise ValueError("every d=1..20 temporal bucket must be nonempty")
    return arrays


def gather_pair_refs(
    episodes: list[dict[str, Any]],
    refs: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources = [
        episodes[int(episode)]["latents"][int(source)]
        for episode, source, _ in refs
    ]
    goals = [
        episodes[int(episode)]["latents"][int(goal)]
        for episode, _, goal in refs
    ]
    return torch.stack(sources).to(device), torch.stack(goals).to(device)


def sample_balanced_pair_refs(
    buckets: dict[int, np.ndarray],
    total: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if total % len(buckets):
        raise ValueError("balanced sample total must be divisible by bucket count")
    per_bucket = total // len(buckets)
    refs = []
    distances = []
    for distance in sorted(buckets):
        bucket = buckets[distance]
        choices = rng.integers(0, len(bucket), size=per_bucket)
        refs.append(bucket[choices])
        distances.append(np.full(per_bucket, distance, dtype=np.int64))
    combined_refs = np.concatenate(refs)
    combined_distances = np.concatenate(distances)
    order = rng.permutation(total)
    return combined_refs[order], combined_distances[order]


def build_ranking_queries(
    episodes: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    same_source = []
    same_goal = []
    for episode_position, episode in enumerate(episodes):
        count = len(episode["latents"])
        for source in range(count - 2):
            maximum = min(MAX_DISTANCE, count - 1 - source)
            if maximum >= 2:
                same_source.append((episode_position, source, maximum))
        for goal in range(2, count):
            maximum = min(MAX_DISTANCE, goal)
            if maximum >= 2:
                same_goal.append((episode_position, goal, maximum))
    if not same_source or not same_goal:
        raise ValueError("ranking queries require trajectories with at least 3 states")
    return (
        np.asarray(same_source, dtype=np.int64),
        np.asarray(same_goal, dtype=np.int64),
    )


def sample_ranking_batch(
    episodes: list[dict[str, Any]],
    same_source_queries: np.ndarray,
    same_goal_queries: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    ss_source = []
    ss_near_goal = []
    ss_far_goal = []
    sg_near_source = []
    sg_far_source = []
    sg_goal = []
    for _ in range(batch_size):
        episode_index, source, maximum = same_source_queries[
            int(rng.integers(0, len(same_source_queries)))
        ]
        distances = np.sort(
            rng.choice(np.arange(1, maximum + 1), size=2, replace=False)
        )
        episode = episodes[int(episode_index)]["latents"]
        ss_source.append(episode[int(source)])
        ss_near_goal.append(episode[int(source + distances[0])])
        ss_far_goal.append(episode[int(source + distances[1])])

        episode_index, goal, maximum = same_goal_queries[
            int(rng.integers(0, len(same_goal_queries)))
        ]
        distances = np.sort(
            rng.choice(np.arange(1, maximum + 1), size=2, replace=False)
        )
        episode = episodes[int(episode_index)]["latents"]
        sg_goal.append(episode[int(goal)])
        sg_near_source.append(episode[int(goal - distances[0])])
        sg_far_source.append(episode[int(goal - distances[1])])
    return tuple(
        torch.stack(values).to(device)
        for values in (
            ss_source,
            ss_near_goal,
            ss_far_goal,
            sg_near_source,
            sg_far_source,
            sg_goal,
        )
    )


@torch.inference_mode()
def balanced_validation_regression(
    model: GlobalHittingTimePotential,
    episodes: list[dict[str, Any]],
    buckets: dict[int, np.ndarray],
    *,
    gamma: float,
    batch_size: int,
    device: torch.device,
) -> float:
    losses = []
    model.eval()
    for distance, refs in buckets.items():
        bucket_losses = []
        for start in range(0, len(refs), batch_size):
            source, goal = gather_pair_refs(
                episodes, refs[start : start + batch_size], device=device
            )
            prediction = model(source, goal)
            target = torch.full_like(prediction, gamma**distance)
            bucket_losses.append(
                F.smooth_l1_loss(prediction, target, reduction="sum").cpu()
            )
        losses.append(float(torch.stack(bucket_losses).sum()) / len(refs))
    return float(np.mean(losses))


def train_one_seed(
    seed: int,
    train_episodes: list[dict[str, Any]],
    calibration_episodes: list[dict[str, Any]],
    train_buckets: dict[int, np.ndarray],
    calibration_buckets: dict[int, np.ndarray],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[GlobalHittingTimePotential, list[dict[str, float]], int]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = GlobalHittingTimePotential().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    same_source, same_goal = build_ranking_queries(train_episodes)
    best_state = None
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        refs, distances = sample_balanced_pair_refs(
            train_buckets, args.pairs_per_epoch, rng=rng
        )
        totals = {"loss": 0.0, "regression": 0.0, "ranking": 0.0}
        examples = 0
        for start in range(0, len(refs), args.batch_size):
            stop = min(start + args.batch_size, len(refs))
            source, goal = gather_pair_refs(
                train_episodes, refs[start:stop], device=device
            )
            distance = torch.as_tensor(
                distances[start:stop], dtype=torch.float32, device=device
            )
            prediction = model(source, goal)
            target = torch.pow(
                torch.full_like(distance, args.gamma), distance
            )
            regression = F.smooth_l1_loss(prediction, target)
            (
                ss_source,
                ss_near_goal,
                ss_far_goal,
                sg_near_source,
                sg_far_source,
                sg_goal,
            ) = sample_ranking_batch(
                train_episodes,
                same_source,
                same_goal,
                stop - start,
                rng=rng,
                device=device,
            )
            ranking = temporal_ranking_loss(
                model(ss_source, ss_near_goal),
                model(ss_source, ss_far_goal),
                model(sg_near_source, sg_goal),
                model(sg_far_source, sg_goal),
            )
            loss = regression + args.lambda_rank * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            count = stop - start
            totals["loss"] += float(loss.detach()) * count
            totals["regression"] += float(regression.detach()) * count
            totals["ranking"] += float(ranking.detach()) * count
            examples += count
        validation = balanced_validation_regression(
            model,
            calibration_episodes,
            calibration_buckets,
            gamma=args.gamma,
            batch_size=args.eval_batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / examples,
            "regression_loss": totals["regression"] / examples,
            "ranking_loss": totals["ranking"] / examples,
            "calibration_balanced_smooth_l1": validation,
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch} loss={row['loss']:.6f} "
            f"reg={row['regression_loss']:.6f} rank={row['ranking_loss']:.6f} "
            f"cal={validation:.6f}",
            flush=True,
        )
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval().requires_grad_(False)
    return model, history, best_epoch


def distribution(values: np.ndarray | torch.Tensor | list[float]) -> dict[str, float]:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    array = array.reshape(-1)
    if not len(array):
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


@torch.inference_mode()
def score_temporal_pairs(
    model: GlobalHittingTimePotential,
    episodes: list[dict[str, Any]],
    buckets: dict[int, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected = {
        "refs": [],
        "distance": [],
        "r_g": [],
    }
    for distance, refs in buckets.items():
        rg_parts = []
        for start in range(0, len(refs), batch_size):
            source, goal = gather_pair_refs(
                episodes, refs[start : start + batch_size], device=device
            )
            rg_parts.append(model(source, goal).cpu())
        collected["refs"].append(refs)
        collected["distance"].append(
            np.full(len(refs), distance, dtype=np.int64)
        )
        collected["r_g"].append(torch.cat(rg_parts).numpy())
    return {
        name: np.concatenate(values)
        for name, values in collected.items()
    }


def pairwise_query_metrics(
    refs: np.ndarray,
    distances: np.ndarray,
    scores: np.ndarray,
    *,
    group_by: str,
    higher_for_near: bool,
) -> dict[str, Any]:
    if group_by not in ("source", "goal"):
        raise ValueError("group_by must be source or goal")
    grouped: dict[tuple[int, int], list[tuple[int, float]]] = {}
    state_column = 1 if group_by == "source" else 2
    for ref, distance, score in zip(refs, distances, scores):
        grouped.setdefault(
            (int(ref[0]), int(ref[state_column])), []
        ).append((int(distance), float(score)))
    credit = 0.0
    comparisons = 0
    correlations = []
    query_accuracies = []
    for values in grouped.values():
        if len(values) < 2:
            continue
        values.sort()
        distance = np.asarray([item[0] for item in values], dtype=np.float64)
        score = np.asarray([item[1] for item in values], dtype=np.float64)
        first, second = np.triu_indices(len(values), k=1)
        if higher_for_near:
            difference = score[first] - score[second]
            correlation = -float(spearmanr(score, distance).statistic)
        else:
            difference = score[second] - score[first]
            correlation = float(spearmanr(score, distance).statistic)
        pair_credit = float(
            np.sum(difference > 0.0) + 0.5 * np.sum(difference == 0.0)
        )
        count = len(first)
        credit += pair_credit
        comparisons += count
        query_accuracies.append(pair_credit / count)
        if np.isfinite(correlation):
            correlations.append(correlation)
    if not comparisons:
        raise ValueError("temporal ordering requires pairwise comparisons")
    return {
        "group_by": group_by,
        "query_count": len(query_accuracies),
        "pair_count": comparisons,
        "pairwise_temporal_order_accuracy": credit / comparisons,
        "per_query_pairwise_accuracy": distribution(query_accuracies),
        "oriented_per_query_spearman": distribution(correlations),
    }


def saturation_diagnostics(
    bucket_summaries: dict[str, dict[str, float]],
    *,
    higher_for_near: bool,
) -> dict[str, Any]:
    distances = np.asarray(sorted(int(key) for key in bucket_summaries))
    means = np.asarray(
        [bucket_summaries[str(int(distance))]["mean"] for distance in distances]
    )
    oriented = means if higher_for_near else -means
    adjacent = oriented[:-1] > oriented[1:]
    tail = distances >= 15
    tail_slope = float(np.polyfit(distances[tail], means[tail], deg=1)[0])
    full_range = float(means.max() - means.min())
    tail_range = float(means[tail].max() - means[tail].min())
    return {
        "adjacent_bucket_order_fraction": float(adjacent.mean()),
        "full_bucket_mean_range": full_range,
        "tail_d15_d20_mean_range": tail_range,
        "tail_to_full_range_ratio": (
            tail_range / full_range if full_range > 0.0 else None
        ),
        "tail_d15_d20_linear_slope": tail_slope,
        "saturation_boolean_not_declared_without_prespecified_threshold": True,
    }


def temporal_validation_report(
    scored: dict[str, np.ndarray],
    *,
    gamma: float,
) -> dict[str, Any]:
    refs = scored["refs"]
    distance = scored["distance"]
    output = {
        "pair_count": int(len(distance)),
        "distance_bucket_counts": {
            str(value): int(np.sum(distance == value))
            for value in range(1, MAX_DISTANCE + 1)
        },
        "target_gamma_power_d": {
            str(value): gamma**value for value in range(1, MAX_DISTANCE + 1)
        },
        "models": {},
    }
    for name, higher_for_near in (("r_g", True),):
        scores = scored[name]
        bucket = {
            str(value): distribution(scores[distance == value])
            for value in range(1, MAX_DISTANCE + 1)
        }
        raw_spearman = float(spearmanr(scores, distance).statistic)
        output["models"][name] = {
            "score_semantics": (
                "higher means earlier witnessed hitting"
                if higher_for_near
                else "lower cost means earlier witnessed hitting"
            ),
            "score_by_distance_bucket": bucket,
            "global_raw_spearman_score_vs_d": raw_spearman,
            "global_oriented_spearman": (
                -raw_spearman if higher_for_near else raw_spearman
            ),
            "same_source_ordering": pairwise_query_metrics(
                refs,
                distance,
                scores,
                group_by="source",
                higher_for_near=higher_for_near,
            ),
            "same_goal_ordering": pairwise_query_metrics(
                refs,
                distance,
                scores,
                group_by="goal",
                higher_for_near=higher_for_near,
            ),
            "saturation_diagnostics": saturation_diagnostics(
                bucket, higher_for_near=higher_for_near
            ),
        }
        if name == "r_g":
            targets = np.power(gamma, distance)
            output["models"][name]["smooth_l1_to_gamma_power_d"] = float(
                F.smooth_l1_loss(
                    torch.from_numpy(scores),
                    torch.from_numpy(targets).to(torch.float32),
                )
            )
    return output


def local_label_metadata(args: argparse.Namespace, dataset_path: Path) -> dict[str, Any]:
    stat = dataset_path.stat()
    return {
        "format_version": 1,
        "dataset_path": str(dataset_path),
        "dataset_size": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
        "policy": args.policy,
        "calibration_episode_range": list(CALIBRATION_RANGE),
        "heldout_episode_range": list(HELDOUT_RANGE),
        "tau_model_steps": 3,
        "model_step_env_steps": TWOROOM_PROFILE.model_step_env_steps,
        "t_env_step": args.t_env_step,
        "candidates_per_category": args.candidates_per_category,
        "success_radius": args.success_radius,
        "env_seed": args.env_seed,
        "witness_state_tolerance": args.witness_state_tolerance,
        "max_replay_pixel_diff": args.max_replay_pixel_diff,
        "categories": list(LOCAL_CATEGORIES),
    }


def collect_all_local_pools(
    handle: h5py.File,
    episode_range: tuple[int, int],
    *,
    args: argparse.Namespace,
    speed: float,
) -> list[dict[str, Any]]:
    pools = []
    range_value = range(*episode_range)
    budget_env_steps = 3 * TWOROOM_PROFILE.model_step_env_steps
    for episode_index in range_value:
        pool = build_strict_candidate_pool(
            handle,
            episode_index=episode_index,
            episode_range=range_value,
            t_env_step=args.t_env_step,
            tau_model_steps=3,
            candidates_per_category=args.candidates_per_category,
            speed=speed,
            success_radius=args.success_radius,
        )
        if pool is None:
            continue
        witnesses_valid = True
        candidates = []
        for candidate in pool["candidates"]:
            record = {
                "candidate_id": candidate["candidate_id"],
                "category": candidate["category"],
                "row": int(candidate["row"]),
                "candidate_episode_index": int(candidate["episode_index"]),
                "temporal_delta_model_steps": candidate[
                    "temporal_delta_model_steps"
                ],
                "lower_bound_env_steps": int(candidate["lower_bound_env_steps"]),
            }
            if candidate["category"] == WITHIN_BUDGET:
                witness = verify_within_budget_witness(
                    handle,
                    pool,
                    candidate,
                    env_seed=args.env_seed,
                    state_tolerance=args.witness_state_tolerance,
                    max_replay_pixel_diff=args.max_replay_pixel_diff,
                )
                witnesses_valid &= bool(witness["valid"])
                record["trajectory_witness"] = witness
            else:
                proof = candidate["lower_bound_env_steps"] > budget_env_steps
                if not proof:
                    raise RuntimeError("strict-over-budget proof is invalid")
                record["strict_over_budget_proof"] = True
            candidates.append(record)
        if not witnesses_valid:
            continue
        pools.append(
            {
                "episode_index": int(pool["episode_index"]),
                "source_row": int(pool["source_row"]),
                "candidates": candidates,
            }
        )
    if not pools:
        raise RuntimeError(f"no strict local pools in range {episode_range}")
    return pools


@torch.inference_mode()
def encode_local_pools(
    adapter: RCAuxAdapter,
    handle: h5py.File,
    pools: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    unique_rows = sorted(
        {
            *(pool["source_row"] for pool in pools),
            *(
                candidate["row"]
                for pool in pools
                for candidate in pool["candidates"]
            ),
        }
    )
    row_to_latent = {}
    for start in range(0, len(unique_rows), batch_size):
        rows = unique_rows[start : start + batch_size]
        images = np.asarray(handle["pixels"][rows])
        latents = adapter.encode_observation(images)[:, -1].detach().cpu()
        row_to_latent.update(
            {row: latent for row, latent in zip(rows, latents)}
        )
        print(
            f"encode Stage5A local labels {min(start + batch_size, len(unique_rows))}/"
            f"{len(unique_rows)}",
            flush=True,
        )
    encoded = []
    for pool in pools:
        source = row_to_latent[pool["source_row"]]
        for candidate in pool["candidates"]:
            encoded.append(
                {
                    **candidate,
                    "source_episode_index": pool["episode_index"],
                    "source_row": pool["source_row"],
                    "source_latent": source.clone(),
                    "target_latent": row_to_latent[candidate["row"]].clone(),
                    "label_within_tau3": candidate["category"] == WITHIN_BUDGET,
                }
            )
    return encoded


def atomic_torch_save(value: Any, path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, output)


def load_or_build_local_labels(
    adapter: RCAuxAdapter,
    dataset_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = local_label_metadata(args, dataset_path)
    cache_path = args.local_label_cache.expanduser().resolve()
    if cache_path.exists():
        artifact = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        if artifact.get("metadata") != metadata:
            raise ValueError("existing Stage 5A local-label cache is incompatible")
        return artifact
    speed = environment_speed(args.env_seed)
    with h5py.File(dataset_path, "r") as handle:
        calibration_pools = collect_all_local_pools(
            handle, CALIBRATION_RANGE, args=args, speed=speed
        )
        heldout_pools = collect_all_local_pools(
            handle, HELDOUT_RANGE, args=args, speed=speed
        )
        calibration = encode_local_pools(
            adapter,
            handle,
            calibration_pools,
            batch_size=args.encode_batch_size,
        )
        heldout = encode_local_pools(
            adapter,
            handle,
            heldout_pools,
            batch_size=args.encode_batch_size,
        )
    artifact = {
        "metadata": metadata,
        "calibration_pool_count": len(calibration_pools),
        "heldout_pool_count": len(heldout_pools),
        "calibration": calibration,
        "heldout": heldout,
    }
    atomic_torch_save(artifact, cache_path)
    return artifact


def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    predictions = scores >= threshold
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "threshold": float(threshold),
        "count": int(len(scores)),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": float((predictions == labels).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "predicted_positive_rate": float(predictions.mean()),
        "auroc": (
            float(roc_auc_score(labels, scores))
            if np.unique(labels).size == 2
            else None
        ),
    }


def select_local_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_fpr: float,
    min_precision: float,
) -> tuple[float, bool, list[dict[str, Any]]]:
    thresholds = sorted({0.0, 1.0, *np.asarray(scores).tolist()})
    rows = [
        classification_metrics(scores, labels, threshold)
        for threshold in thresholds
    ]
    feasible = [
        row
        for row in rows
        if row["precision"] is not None
        and row["precision"] >= min_precision
        and row["false_positive_rate"] is not None
        and row["false_positive_rate"] <= max_fpr
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["recall"],
                row["predicted_positive_rate"],
                -row["threshold"],
            ),
        )
        constraints_met = True
    else:
        nonempty = [row for row in rows if row["precision"] is not None]
        selected = max(
            nonempty,
            key=lambda row: (
                row["precision"],
                row["recall"],
                -row["false_positive_rate"],
            ),
        )
        constraints_met = False
    return float(selected["threshold"]), constraints_met, rows


@torch.inference_mode()
def score_local_records(
    model: GlobalHittingTimePotential,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scores = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        source = torch.stack([record["source_latent"] for record in chunk]).to(
            device
        )
        goal = torch.stack([record["target_latent"] for record in chunk]).to(
            device
        )
        scores.append(model(source, goal).cpu())
    return (
        torch.cat(scores).numpy(),
        np.asarray(
            [record["label_within_tau3"] for record in records], dtype=bool
        ),
        [record["category"] for record in records],
    )


def score_separation(
    scores: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
) -> dict[str, Any]:
    categories_array = np.asarray(categories)
    positive = scores[labels]
    negative = scores[~labels]
    pooled_std = np.sqrt(0.5 * (positive.var() + negative.var()))
    return {
        "witnessed_within_tau3": distribution(positive),
        "strict_over_budget_combined": distribution(negative),
        "same_trajectory_strict_over_budget": distribution(
            scores[categories_array == SAME_TRAJECTORY_STRICT]
        ),
        "cross_trajectory_strict_over_budget": distribution(
            scores[categories_array == CROSS_TRAJECTORY_STRICT]
        ),
        "positive_minus_strict_mean": float(positive.mean() - negative.mean()),
        "standardized_mean_separation": (
            float((positive.mean() - negative.mean()) / pooled_std)
            if pooled_std > 0.0
            else None
        ),
        "label_semantics": (
            "strict negatives prove only not reachable within tau=3; "
            "they are not global-unreachable labels"
        ),
    }


def local_validation_report(
    model: GlobalHittingTimePotential,
    local_labels: dict[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    calibration_scores, calibration_labels, calibration_categories = (
        score_local_records(
            model,
            local_labels["calibration"],
            batch_size=args.eval_batch_size,
            device=device,
        )
    )
    heldout_scores, heldout_labels, heldout_categories = score_local_records(
        model,
        local_labels["heldout"],
        batch_size=args.eval_batch_size,
        device=device,
    )
    eta_3, constraints_met, threshold_table = select_local_threshold(
        calibration_scores,
        calibration_labels,
        max_fpr=args.max_calibration_fpr,
        min_precision=args.min_calibration_precision,
    )
    return {
        "eta_3": eta_3,
        "threshold_source": "selected_only_on_4000_4500_calibration",
        "threshold_rule": {
            "same_as_stage2": True,
            "max_calibration_fpr": args.max_calibration_fpr,
            "min_calibration_precision": args.min_calibration_precision,
            "maximize": "recall, then coverage, then lower threshold",
            "constraints_met": constraints_met,
        },
        "calibration": {
            "episode_range": list(CALIBRATION_RANGE),
            "pool_count": local_labels["calibration_pool_count"],
            "classification": classification_metrics(
                calibration_scores, calibration_labels, eta_3
            ),
            "score_separation": score_separation(
                calibration_scores,
                calibration_labels,
                calibration_categories,
            ),
            "threshold_table": threshold_table,
        },
        "heldout": {
            "episode_range": list(HELDOUT_RANGE),
            "pool_count": local_labels["heldout_pool_count"],
            "classification": classification_metrics(
                heldout_scores, heldout_labels, eta_3
            ),
            "score_separation": score_separation(
                heldout_scores,
                heldout_labels,
                heldout_categories,
            ),
        },
    }


def mean_std(values: list[float | None]) -> dict[str, float | int | None]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {
            "mean": None,
            "std": None,
            "valid_seed_count": 0,
            "missing_seed_count": len(values),
        }
    array = np.asarray(valid, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "valid_seed_count": len(valid),
        "missing_seed_count": len(values) - len(valid),
    }


def aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    temporal_paths = {
        "r_g_global_oriented_spearman": (
            lambda item: item["long_range_validation"]["models"]["r_g"][
                "global_oriented_spearman"
            ]
        ),
        "r_g_same_source_pairwise": (
            lambda item: item["long_range_validation"]["models"]["r_g"][
                "same_source_ordering"
            ]["pairwise_temporal_order_accuracy"]
        ),
        "r_g_same_goal_pairwise": (
            lambda item: item["long_range_validation"]["models"]["r_g"][
                "same_goal_ordering"
            ]["pairwise_temporal_order_accuracy"]
        ),
        "r_g_tail_slope": (
            lambda item: item["long_range_validation"]["models"]["r_g"][
                "saturation_diagnostics"
            ]["tail_d15_d20_linear_slope"]
        ),
        "heldout_auroc": (
            lambda item: item["local_tau3_validation"]["heldout"][
                "classification"
            ]["auroc"]
        ),
        "heldout_accuracy": (
            lambda item: item["local_tau3_validation"]["heldout"][
                "classification"
            ]["accuracy"]
        ),
        "heldout_precision": (
            lambda item: item["local_tau3_validation"]["heldout"][
                "classification"
            ]["precision"]
        ),
        "heldout_recall": (
            lambda item: item["local_tau3_validation"]["heldout"][
                "classification"
            ]["recall"]
        ),
        "heldout_f1": (
            lambda item: item["local_tau3_validation"]["heldout"][
                "classification"
            ]["f1"]
        ),
        "heldout_score_separation": (
            lambda item: item["local_tau3_validation"]["heldout"][
                "score_separation"
            ]["positive_minus_strict_mean"]
        ),
        "eta_3": lambda item: item["local_tau3_validation"]["eta_3"],
    }
    aggregated = {
        name: mean_std([extract(item) for item in seed_results])
        for name, extract in temporal_paths.items()
    }
    aggregated["directional_checks"] = {
        "all_seeds_r_g_spearman_direction_correct": all(
            extract(seed_result) > 0.0
            for seed_result in seed_results
            for name, extract in temporal_paths.items()
            if name == "r_g_global_oriented_spearman"
        ),
        "all_seeds_same_source_ordering_above_chance": all(
            temporal_paths["r_g_same_source_pairwise"](item) > 0.5
            for item in seed_results
        ),
        "all_seeds_same_goal_ordering_above_chance": all(
            temporal_paths["r_g_same_goal_pairwise"](item) > 0.5
            for item in seed_results
        ),
        "all_seeds_local_auroc_above_chance": all(
            temporal_paths["heldout_auroc"](item) > 0.5
            for item in seed_results
        ),
        "not_a_strong_or_effective_acceptance_test": True,
    }
    return aggregated


def protocol(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "stage": "5A",
        "semantic_name": "discounted witnessed hitting-time potential",
        "not_absolute_global_reachability": True,
        "train_episode_range": list(TRAIN_RANGE),
        "long_range_validation_episode_range": [4000, 5000],
        "early_stopping_episode_range": list(CALIBRATION_RANGE),
        "local_threshold_calibration_episode_range": list(CALIBRATION_RANGE),
        "local_threshold_heldout_episode_range": list(HELDOUT_RANGE),
        "test_episode_range_5000_10000_used": False,
        "successful_trajectories_only_for_temporal_pairs": True,
        "latent_space": "raw unnormalized RC-aux encoder latent",
        "latent_dim": LATENT_DIM,
        "input": "[z,g,z-g]",
        "input_dim": 3 * LATENT_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "output": "sigmoid scalar",
        "distance_buckets": list(range(1, MAX_DISTANCE + 1)),
        "distance_unit": "one cached model step = 5 TwoRoom env steps",
        "distance_bucket_sampling": "exactly balanced per training epoch",
        "gamma": args.gamma,
        "regression_target": "gamma ** d",
        "regression_loss": "SmoothL1",
        "ranking_loss": (
            "mean pairwise logistic loss over same-source and same-goal ordering"
        ),
        "lambda_rank": args.lambda_rank,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "pairs_per_epoch": args.pairs_per_epoch,
        "grad_clip": args.grad_clip,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "checkpoint_selection": (
            "minimum balanced temporal SmoothL1 on [4000,4500)"
        ),
        "generator_seeds": list(seeds),
        "modules_frozen": [
            "RC-aux Encoder",
            "RC-aux world model",
            "R_local(z,g,h)",
            "RC-LeWM low-level planner",
        ],
        "strict_over_budget_used_for_training": False,
        "cross_trajectory_used_for_training": False,
        "local_labels": {
            "tau_model_steps": 3,
            "budget_env_steps": 15,
            "candidates_per_category": args.candidates_per_category,
            "positive": "witnessed-within-budget real trajectory target",
            "negative": "strict-over-budget geometric control target",
            "negative_not_global_unreachable": True,
            "threshold_rule_fixed_in_stage5a": True,
            "max_calibration_fpr": args.max_calibration_fpr,
            "min_calibration_precision": args.min_calibration_precision,
        },
        "closed_loop_run": False,
        "generator_modified": False,
        "gamma_tuned_on_closed_loop": False,
        "latent_cache": str(args.latent_cache.expanduser().resolve()),
        "local_label_cache": str(args.local_label_cache.expanduser().resolve()),
        "policy": args.policy,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "dataset": args.dataset,
    }


def checkpoint_path(checkpoint_dir: Path, seed: int) -> Path:
    return checkpoint_dir.expanduser().resolve() / (
        f"stage5a_global_reachability_seed{seed}.pt"
    )


def checkpoint_protocol_compatible(
    saved: dict[str, Any], current: dict[str, Any]
) -> bool:
    def normalized(value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        for key in (
            "generator_seeds",
            "progress_ranker",
            "progress_ranker_report",
            "stage2_report",
            "d_psi_used",
            "high_level_local_r_used",
        ):
            result.pop(key, None)
        result["modules_frozen"] = [
            name
            for name in result.get("modules_frozen", [])
            if name != "D_psi"
        ]
        local = dict(result.get("local_labels", {}))
        local.pop("threshold_rule_reuses_stage2_constraints", None)
        local.pop("threshold_rule_fixed_in_stage5a", None)
        local["positive"] = "witnessed-within-budget real trajectory target"
        local["negative"] = "strict-over-budget geometric control target"
        result["local_labels"] = local
        return result

    saved_value = normalized(saved)
    current_value = normalized(current)
    return saved_value == current_value


def main() -> int:
    args = parse_args()
    validate_args(args)
    seeds = parse_seeds(args.seeds)
    fixed_protocol = protocol(args, seeds)
    cache = load_latent_cache(args.latent_cache)
    train_episodes, calibration_episodes, heldout_episodes = temporal_splits(cache)
    train_buckets = build_temporal_pair_buckets(train_episodes)
    calibration_buckets = build_temporal_pair_buckets(calibration_episodes)
    validation_episodes = sorted(
        [*calibration_episodes, *heldout_episodes],
        key=lambda episode: int(episode["episode_index"]),
    )
    validation_buckets = build_temporal_pair_buckets(validation_episodes)
    pair_counts = {
        "train": {
            str(distance): int(len(refs))
            for distance, refs in train_buckets.items()
        },
        "calibration": {
            str(distance): int(len(refs))
            for distance, refs in calibration_buckets.items()
        },
        "validation_4000_5000": {
            str(distance): int(len(refs))
            for distance, refs in validation_buckets.items()
        },
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "train_successful_episodes": len(train_episodes),
                    "calibration_successful_episodes": len(calibration_episodes),
                    "heldout_successful_episodes": len(heldout_episodes),
                    "pair_counts": pair_counts,
                    "test_split_used": False,
                },
                indent=2,
            )
        )
        return 0

    device = torch.device(args.device)
    dataset_path = args.cache_dir.expanduser().resolve() / args.dataset
    adapter = RCAuxAdapter.from_checkpoint(
        args.policy,
        profile=TWOROOM_PROFILE,
        cache_dir=args.cache_dir.expanduser().resolve(),
        device=device,
    )
    adapter.model.interpolate_pos_encoding = True
    adapter.model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("Encoder, world model, and R_local must remain frozen")
    local_labels = load_or_build_local_labels(
        adapter, dataset_path, args=args
    )
    local_metadata = local_labels["metadata"]
    if local_metadata["calibration_episode_range"] != list(CALIBRATION_RANGE):
        raise ValueError("local calibration label range changed")
    if local_metadata["heldout_episode_range"] != list(HELDOUT_RANGE):
        raise ValueError("local heldout label range changed")

    output_path = args.output.expanduser().resolve()
    existing = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if not checkpoint_protocol_compatible(
            existing.get("protocol", {}), fixed_protocol
        ):
            raise ValueError("existing Stage 5A output uses a different protocol")
        if existing.get("stage5a_complete") is True:
            print(f"report already complete: {output_path}")
            return 0
    saved_results = {
        int(item["seed"]): item for item in existing.get("seed_results", [])
    }
    seed_results = []
    for seed in seeds:
        path = checkpoint_path(args.checkpoint_dir, seed)
        saved_result = saved_results.get(seed)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if (
                not checkpoint_protocol_compatible(
                    saved.get("protocol", {}), fixed_protocol
                )
                or saved.get("seed") != seed
            ):
                raise ValueError(f"incompatible Stage 5A checkpoint: {path}")
            model = GlobalHittingTimePotential(
                latent_dim=int(saved["latent_dim"]),
                hidden_dims=tuple(saved["hidden_dims"]),
            )
            model.load_state_dict(saved["state_dict"], strict=True)
            model.to(device).eval().requires_grad_(False)
            if saved_result is None:
                saved_result = saved["result"]
        else:
            model, history, best_epoch = train_one_seed(
                seed,
                train_episodes,
                calibration_episodes,
                train_buckets,
                calibration_buckets,
                args=args,
                device=device,
            )
            scored = score_temporal_pairs(
                model,
                validation_episodes,
                validation_buckets,
                batch_size=args.eval_batch_size,
                device=device,
            )
            long_range = temporal_validation_report(scored, gamma=args.gamma)
            local = local_validation_report(
                model, local_labels, args=args, device=device
            )
            saved_result = {
                "seed": seed,
                "checkpoint": str(path),
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                "training_history": history,
                "long_range_validation": long_range,
                "local_tau3_validation": local,
            }
            state_dict = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            atomic_torch_save(
                {
                    "format_version": 1,
                    "stage": "5A",
                    "semantic_name": "discounted witnessed hitting-time potential",
                    "not_absolute_global_reachability": True,
                    "seed": seed,
                    "latent_dim": LATENT_DIM,
                    "hidden_dims": HIDDEN_DIMS,
                    "gamma": args.gamma,
                    "eta_3": local["eta_3"],
                    "protocol": fixed_protocol,
                    "state_dict": state_dict,
                    "result": saved_result,
                },
                path,
            )
        if saved_result is None:
            raise RuntimeError(f"seed {seed} result is unavailable")
        seed_results.append(saved_result)
        partial = {
            "stage5a_complete": False,
            "pilot_only": seeds == (3072,),
            "protocol": fixed_protocol,
            "pair_counts": pair_counts,
            "local_label_summary": {
                "calibration_pool_count": local_labels[
                    "calibration_pool_count"
                ],
                "calibration_candidate_count": len(local_labels["calibration"]),
                "heldout_pool_count": local_labels["heldout_pool_count"],
                "heldout_candidate_count": len(local_labels["heldout"]),
                "strict_labels_used_for_training": False,
            },
            "seed_results": seed_results,
        }
        atomic_write_json(partial, args.output)

    formal = seeds == FORMAL_SEEDS
    report = {
        "stage5a_complete": formal,
        "pilot_only": seeds == (3072,),
        "protocol": fixed_protocol,
        "pair_counts": pair_counts,
        "local_label_summary": {
            "calibration_pool_count": local_labels["calibration_pool_count"],
            "calibration_candidate_count": len(local_labels["calibration"]),
            "heldout_pool_count": local_labels["heldout_pool_count"],
            "heldout_candidate_count": len(local_labels["heldout"]),
            "categories": list(LOCAL_CATEGORIES),
            "strict_labels_used_for_training": False,
            "strict_label_semantics": (
                "not reachable within tau=3 only; not globally unreachable"
            ),
        },
        "seed_results": seed_results,
        "across_seeds": (
            aggregate_seed_results(seed_results) if formal else None
        ),
        "stage5b_readiness": {
            "ready": None,
            "reason": (
                "strong long-range ordering and effective local discrimination "
                "were not assigned numeric acceptance thresholds in advance"
            ),
            "directional_checks_are_not_acceptance_criteria": True,
            "requires_review_of_both_validation_axes": True,
        },
        "interpretation_constraints": [
            "R_G is a discounted witnessed hitting-time potential.",
            "R_G is not established as absolute global reachability.",
            "Strict-over-budget and cross-trajectory labels are never training targets.",
            "Strict local negatives prove only failure within tau=3.",
            "No CEM, closed-loop rollout, generator modification, or gamma tuning occurs.",
            "The [5000,10000) test split is untouched.",
        ],
    }
    atomic_write_json(report, args.output)
    print(f"report_path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
