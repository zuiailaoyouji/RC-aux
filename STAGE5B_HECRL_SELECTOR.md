# Stage 5B: Frozen HECRL-Style Selector Control

> **Controlled validation, not deployment.** Exact 1-NN projection and the
> `D_psi` baseline below isolate mechanism effects. The deployed selector uses
> raw generated candidates, `R_G` filter+rank, and no `D_psi`; see
> `HRC_LEWM_HIGH_LEVEL.md`.

Stage 5B is an evaluation-only experiment. It does not train or modify the
RC-aux Encoder, world model, local reachability head, RC-LeWM planner, Stage 4
generator, `D_psi`, or any Stage 5A `R_G` head. The Stage 5A semantics remain
discounted witnessed hitting-time potential, not absolute global
reachability.

## Held-Out Long-Range Audit

Before candidate evaluation, each of the three Stage 5A checkpoints is audited
on successful episodes in `[4500,5000)`. These episodes were not used for
Stage 5A early stopping. The audit reports only:

- oriented Spearman between `R_G` and temporal distance;
- same-source and same-goal pairwise temporal ordering;
- score means for `d=1,...,20`;
- the linear score slope over `d=15,...,20`.

The runner stops before selector evaluation only for an obvious directional
collapse: a non-finite metric, non-positive oriented Spearman, pairwise
ordering at or below chance, or a non-negative tail slope. This is a fixed
directional sanity check, not a tunable performance threshold. No checkpoint,
`gamma`, or `eta_3` is changed after this audit.

## Fixed Candidate Control

The experiment uses all 411 successful `[4000,5000)` validation episodes and
paired generator/`R_G` seeds `(3072,3072)`, `(3073,3073)`, and `(3074,3074)`.
Each `R_G` checkpoint uses its own saved Stage 5A `eta_3`; thresholds are never
averaged or recalibrated.

Offline candidates are the exact Stage 4C artifacts. Closed-loop candidates
use the same Stage 4D dropout rule. Every raw candidate is projected to the
exact Euclidean 1-NN row in the raw, unnormalized 192D latent bank from
successful `[0,4000)` episodes. There is no nearest-neighbor averaging, and
closed-loop candidates are projected again after every high-level generation.

All methods first select `z_G` directly when
`R_G(z_t,z_G) >= eta_3(seed)`. Otherwise they use:

- `A_dpsi_baseline`: retain positive `D_psi` progress and select minimum
  `D_psi(g,z_G)`;
- `B_rg_rank_only`: select `argmax R_G(g,z_G)` without a local gate or a
  progress threshold;
- `C_rg_filter_rank`: retain `R_G(z_t,g) >= eta_3(seed)`, then select
  `argmax R_G(g,z_G)`.

When a selector has no candidate, it executes one flat `z_G` model step and
retries the high-level decision. Method B always has a generated candidate.

## Closed Loop And Reporting

The episode horizon is 100 environment steps. A selected subgoal remains fixed
for `H_plan/H_exec = 3/1 -> 2/1 -> 1/1`; each model step executes five
environment actions and updates the real observation history, whose maximum
length is three. Flat fallback uses planning horizon five. CEM is fixed at 300
samples, 30 iterations, and top-k 30. The low-level reachability-cost weight
is 0.85.

Offline reporting includes coverage, method-C pass rate and pass counts,
method-C rejection of method B's top-ranked candidate, both selected `R_G`
scores, `D_psi` progress, residual norm, and raw-space mean-5NN manifold
distance. Closed-loop reporting includes success, successful and
100-step-censored completion time, fallback, coverage, direct-goal rates,
realized `R_G` and `D_psi` progress, and timing.

For formal inference, the three paired seeds are averaged within each episode,
then 10,000 paired episode bootstraps produce 95% confidence intervals for:

- `Delta_filter = SR_C - SR_B`;
- `Delta_rank = SR_B - SR_A`;
- `Delta_full = SR_C - SR_A`.

Unified filtering and ranking is supported only if the lower confidence bounds
for both `Delta_filter` and `Delta_full` are strictly positive. A strictly
negative upper bound for `Delta_rank` retains the `D_psi` route. If filtering
is significantly harmful while B is not significantly worse than A, the
result retains `R_G` ranking and removes the hard gate. All other outcomes are
inconclusive, and any of the three key intervals crossing or touching zero
forces an inconclusive result. No result is used to tune the fixed protocol.

The `[5000,10000)` test split is never accessed.

## Commands

Validate all fixed artifacts without loading models:

    PYTHONPATH=. .venv/bin/python tools/evaluate_stage5b_hecrl.py \
      --validate-only

Run only the held-out long-range audit:

    PYTHONPATH=. .venv/bin/python tools/evaluate_stage5b_hecrl.py \
      --device cpu \
      --audit-only

Run the audit and offline candidate comparison, without CEM:

    CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
    PYTHONPATH=. .venv/bin/python tools/evaluate_stage5b_hecrl.py \
      --device cuda \
      --offline-only

Run the complete formal Stage 5B comparison:

    CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
    PYTHONPATH=. .venv/bin/python tools/evaluate_stage5b_hecrl.py \
      --device cuda \
      --output outputs/stage5b_hecrl_selector.json

The output is crash-resumable at rollout granularity. A one-seed pilot must use
both `--seeds 3072` and a separate output path so it cannot be confused with
the formal three-seed report.
