# Stage 5A: Discounted Witnessed Hitting-Time Potential

Stage 5A trains one new high-level scalar head while keeping the RC-aux
Encoder, world model, local reachability head, and low-level RC-LeWM frozen. It
does not load or compare against the retired Stage 3 progress ranker, run CEM,
or modify the generator.

The new head is deliberately not called absolute global reachability. Its
semantics are limited to a discounted witnessed hitting-time potential learned
from successful trajectories.

## Model And Training

The input is the raw, unnormalized RC-aux latent concatenation:

    [z, g, z-g] in R^576.

The MLP dimensions are 576, 256, 128, and 1, with ReLU hidden activations and a
final sigmoid. Only real same-trajectory latent pairs from successful episodes
in [0,4000) are training examples. For sampled model-step distance
d in {1,...,20}, the fixed target is:

    y = 0.9^d.

Every epoch samples exactly the same number of examples from each distance
bucket. Sparse long-distance buckets are sampled with replacement; no
over-budget or cross-trajectory record is ever a training target.

The loss is:

    SmoothL1(R_G(z_i,z_j), 0.9^d) + 0.1 * L_rank.

L_rank is the mean of two pairwise logistic losses:

- same source: a target reached earlier must receive a higher score;
- same goal: a later source must receive a higher score.

AdamW uses learning rate 3e-4, weight decay 1e-4, batch size 512, and gradient
clipping at 1. Training runs for at most 30 epochs with patience 5. Checkpoint
selection uses balanced temporal SmoothL1 on successful calibration episodes
in [4000,4500), leaving [4500,5000) untouched by model and threshold
selection. Formal training uses generator-aligned seeds 3072, 3073, and 3074.

## Long-Range Offline Validation

All real future pairs with d=1,...,20 from the 411 successful validation
episodes in [4000,5000) are scored. The report contains:

- per-distance score distributions and sample counts;
- raw and direction-normalized Spearman;
- same-source and same-goal pairwise temporal-order accuracy;
- balanced SmoothL1 against 0.9^d;
- adjacent-bucket ordering, full/tail range, and d=15,...,20 slope.

`R_G` receives no gradient updates from `[4000,5000)`. Saturation diagnostics
are reported without inventing a numeric pass threshold after seeing the
result. No separate progress ranker is loaded for this validation.

## Local Tau-3 Validation

The competence calibration uses fixed trajectory/geometric labels:

- positive: a real trajectory action witness reaches the target within at most
  three model steps, or 15 environment steps;
- strict negative: the collision-agnostic displacement lower bound alone
  exceeds 15 environment steps, for same-trajectory and cross-trajectory
  targets.

These strict negatives prove only that the target is not reachable within the
local tau=3 budget. They are never described as globally unreachable and never
enter R_G training.

All valid pools are rebuilt separately in [4000,4500) and [4500,5000).
Only calibration scores select eta_3. The Stage 5A rule is fixed in advance:
among thresholds with precision at least 0.95 and FPR at most 0.05,
maximize recall, then coverage, then prefer the lower threshold. The held-out
split reports AUROC, accuracy, precision, recall, F1, and score separation.

## Stage 5B Decision

The report must show both stable long-range temporal ordering and useful
held-out local discrimination before Stage 5B. No numeric definition of
"strong" or "effective" was specified in advance, so the runner does not
retroactively invent one. It reports minimal directional checks but leaves
ready_for_stage5b unset for explicit review.

The [5000,10000) test split is not inspected or used.

## Commands

Validate split and bucket availability:

    PYTHONPATH=. .venv/bin/python tools/train_stage5a_global_reachability.py \
      --validate-only

Formal three-seed training and offline validation:

    CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
    PYTHONPATH=. .venv/bin/python tools/train_stage5a_global_reachability.py \
      --device cuda \
      --seeds 3072,3073,3074 \
      --cache-dir /home/sxw/work/datasets/stable-wm \
      --output outputs/stage5a_global_reachability.json

The local-label artifact is cached at outputs/stage5a_local_tau3_labels.pt.
Checkpoints are written under
outputs/stage5a_global_reachability_checkpoints/.
