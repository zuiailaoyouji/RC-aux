# Stage 4C: Generated-Target RC Behavior Diagnostic

> **Archived failure analysis.** This experiment studies the retired
> high-level `R_local(z,g,3)>=eta_R` gate. It does not describe the current
> selector and must not be used as a deployment entry point.

Stage 4C is diagnostic only. It freezes the RC-aux encoder and reachability
head, the Stage 3 progress ranker, and all three Stage 4 generators. It does not
run CEM, collect generated-target execution labels, recalibrate `eta_R`, add a
new reachability objective, or train any module.

## Fixed Protocol

- Sources are all 5,419 generator-eligible states from successful validation
  episodes `[4000,5000)`. Their order is fixed by the latent cache.
- Generator seeds `3072`, `3073`, and `3074` are evaluated separately. Dropout
  is seeded once per ordered validation loader with `20260809 + generator_seed`.
- Every source produces exactly 32 candidates once. All threshold scans reuse
  those saved candidates.
- The fixed Stage 2 threshold is `eta_R=0.5295726657`, with `tau=3` model steps.
- The real latent bank contains all 27,352 raw encoder latents from successful
  train episodes `[0,4000)`. Validation and bank episodes are disjoint.
- Manifold distance is the mean Euclidean distance to the five nearest bank
  latents in the raw, unnormalized 192D RC-aux latent space. There is no PCA,
  cosine distance, L2 normalization, or adaptive selection of `k`.
- All 6,959 real validation latents are compared with the same train bank to
  establish the real-to-real manifold-distance reference distribution.
- The Stage 1/2 witnessed-within-budget and strict over-budget/cross-trajectory
  targets are re-encoded and reported as labeled real-target reference groups.
  Their existing labels are not transferred to generated candidates.
- Test episodes and their latent bank `[5000,10000)` are not read by Stage 4C.

For every generated candidate, the saved seed artifact contains its exact
latent, RC score and pass flag, `D_psi` progress and rank, residual norm, and
five-neighbor manifold distance. It also contains source metadata, direct-goal
eligibility diagnostics, and the no-RC and RC-filtered selected indices.

## Selection Comparison

The diagnostic compares two selectors on the same 32 candidates:

```text
g_noRC = argmax Delta_i among Delta_i > 0
g_RC   = argmax Delta_i among R_i >= eta_R and Delta_i > 0
```

It reports how often RC rejects `g_noRC`, whether another RC-passing candidate
remains, and the resulting changes in selected progress, residual norm, RC
score, and manifold distance. RC relationships with residual norm, manifold
distance, and progress are reported both over all candidates as descriptive
correlations and as distributions of within-source correlations across the 32
candidates.

Threshold sensitivity uses the fixed grid
`{0,0.1,...,0.9} union {0.5295726657}`. It reports RC-pass rate, source-level
`N_RC=0/1/>=2`, RC-plus-positive-progress coverage, rejection of the highest
progress candidate, and selected-candidate progress, residual norm, and
manifold distance. This scan is not threshold selection.

## Optional Pilot

Seed 3072 may be used only as an implementation pilot:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/diagnose_stage4c_generated_rc.py \
  --device cuda \
  --generator-seeds 3072 \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --output outputs/stage4c_generated_target_rc_pilot.json
```

The pilot report is explicitly marked `pilot_only=true` and
`stage4c_complete=false`.

## Formal Three-Seed Diagnostic

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/diagnose_stage4c_generated_rc.py \
  --device cuda \
  --generator-seeds 3072,3073,3074 \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --output outputs/stage4c_generated_target_rc.json
```

The JSON report contains each seed separately and mean plus standard deviation
across the three seed-level results. Candidate arrays are never pooled across
seeds as independent samples. Exact per-candidate data are stored beside the
report as `stage4c_generated_target_rc_seed<seed>_candidates.pt`. Existing
compatible artifacts are reused after interruption.

## Interpretation Limits

The report may support a generated-target distribution-shift hypothesis if
generated manifold distances are abnormal relative to real-to-real and labeled
real-target references. A consistently negative RC-progress relationship may
support a local-reachability versus global-progress selection conflict. Smooth
threshold trends may indicate that a hard cutoff is too aggressive.

None of these observations establishes generated-target RC calibration because
generated candidates have no ground-truth reachability labels in Stage 4C. Do
not report generated-target AUROC or accuracy, select a new threshold, retrain
RC, or change the selector based on the held-out test set from this diagnostic.
