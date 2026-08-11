# Stage 4D: On-Manifold Matched RC Control

> **Archived failure analysis.** The finite-horizon RC gate, `D_psi` selector,
> and exact 1-NN projection below are experimental controls, not components of
> the current system. Proposal-aware alignment of this old high-level gate is
> retired.

This experiment decides whether the frozen Stage 2 hard RC gate remains useful
when generated-target distribution shift is removed locally by an explicit
control. It does not train or modify the generator, encoder, world model,
R_phi, D_psi, or low-level RC-LeWM, and it does not recalibrate the fixed
eta_R=0.5295726657.

## Fixed Projection

The sources are all 411 successful validation episodes in [4000,5000).
Generator seeds 3072, 3073, and 3074 remain separate. Offline analysis reuses
each seed's exact 32-candidate Stage 4C artifact and dropout realization.

Every generated candidate is used only by this experiment and projected to one
actual real latent in the successful-train bank [0,4000):

    g_i_proj = argmin over z in B_train of ||g_i - z||_2.

Projection uses Euclidean distance in raw, unnormalized 192D RC-aux latent
space. It is exact 1-NN and returns the selected bank row byte-for-byte; it
never averages neighbors. The existing raw-space mean distance to five nearest
neighbors is recomputed after projection only to confirm that projected
candidates return to the real-to-real manifold-distance scale.

The test split [5000,10000) is never inspected or used by this experiment.

## Matched Selectors

Both methods use the same projected 32-candidate set, D_psi, direct-goal
precheck, fallback, and all low-level settings:

    projected32_dpsi_direct_no_rc:
      positive D_psi progress -> maximum progress

    projected32_rc_dpsi_direct:
      R_phi(z_t,g,3) >= eta_R -> positive D_psi progress -> maximum progress

For both methods, direct z_T joins the pool only when its unchanged RC precheck
and positive-progress rule pass. Offline results report candidate RC pass and
coverage, rejection of the highest-progress projected candidate, and selected
D_psi progress, residual norm, and mean-5NN manifold distance.

## Closed Loop

All 411 validation episodes are evaluated for all three generator seeds and
both methods, for 2,466 rollouts. The environment horizon is the official 100
steps. A selected projected subgoal is fixed for the whole local segment:

    H_plan=3, H_exec=1
    H_plan=2, H_exec=1
    H_plan=1, H_exec=1

Fallback plans directly to z_T for one model step and retries the high level.
The two methods use the same episode-specific environment, CEM, and dropout
seed rules. Dropout is reset immediately before each high-level generation so
CEM random-number consumption cannot alter later generator masks. Once policy
trajectories diverge, candidate values naturally depend on their respective
real histories; their random seed rule remains matched.

The runner writes a partial JSON after every rollout and resumes automatically
when invoked again with exactly the same protocol.

## Decision

For every episode, metrics are first averaged over generator seeds
3072/3073/3074. Paired bootstrap then resamples the 411 validation episodes.
The primary quantity is:

    Delta SR = SR_RC - SR_noRC.

- Retaining RC requires Delta SR > 0 with lower 95% CI strictly above zero,
  plus an offline tradeoff that no longer shows substantial rejection of
  high-progress candidates for negligible improvement elsewhere.
- Delta SR < 0 with upper 95% CI strictly below zero means the current hard
  gate is harmful. This historical result closed the proposal-aware alignment
  direction for the old high-level local head.
- A confidence interval containing or touching zero is inconclusive.

The report does not invent a numerical cutoff for the secondary offline
tradeoff. It serializes the necessary paired measurements for explicit review.

## Commands

Validate fixed inputs without loading the models:

    PYTHONPATH=. .venv/bin/python tools/evaluate_stage4d_on_manifold_rc.py \
      --validate-only

Optional seed-3072 offline pilot:

    CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
    PYTHONPATH=. .venv/bin/python tools/evaluate_stage4d_on_manifold_rc.py \
      --device cuda \
      --generator-seeds 3072 \
      --offline-only \
      --cache-dir /home/sxw/work/datasets/stable-wm \
      --output outputs/stage4d_on_manifold_rc_pilot.json

Formal offline plus complete closed-loop control:

    CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
    PYTHONPATH=. .venv/bin/python tools/evaluate_stage4d_on_manifold_rc.py \
      --device cuda \
      --generator-seeds 3072,3073,3074 \
      --cache-dir /home/sxw/work/datasets/stable-wm \
      --output outputs/stage4d_on_manifold_rc.json
