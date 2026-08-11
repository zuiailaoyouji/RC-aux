# Stage 4: High-Level Latent Subgoal Generator

> **Generator training remains reusable; selection experiments below are
> archived.** The old `R_local + D_psi` selector is not the current method.
> Deployment is defined in `HRC_LEWM_HIGH_LEVEL.md`.

Stage 4 trains only the high-level generator. The RC-aux Encoder, world model,
and low-level RC-LeWM remain frozen. The duration token is fixed to three
RC-aux model steps, or three action blocks and 15 TwoRoom environment steps.

## Supervision And Splits

The implementation reuses the frozen-Encoder latent cache produced by Stage 3.
Successful trajectories are sampled every five environment steps. A sample is
included only when its exact `t+15` row exists:

```text
(z_{t-L+1:t}, z_T, tau=3) -> g_GT = z_{t+3},  L <= 3.
```

The generator receives no action history, coordinates, map representation,
`pos_target`, or future latent. `g_GT` is used only by the loss. `z_T` is the
encoded terminal observation supplied by the HRC-LeWM task interface.

Episode splits are fixed and disjoint:

```text
train       [0, 4000)
validation  [4000, 5000)
test        [5000, 10000)
```

The model contains two Transformer encoder layers with model dimension 256,
four attention heads, FFN dimension 512, and dropout 0.1. Its tokens are three
left-padded history positions, `z_T`, a fixed duration token, and a subgoal query
token. The output is a standardized residual. Train-split statistics convert it
back to the original 192-dimensional latent space:

```text
g_hat = z_t + residual_std * residual_normalized + residual_mean.
```

The loss is exactly:

```text
SmoothL1(g_hat, z_{t+3}) + 0.1 * (1 - cosine(g_hat, z_{t+3})).
```

There is no reachability or progress-ranking training loss. AdamW uses learning rate `3e-4`, weight
decay `1e-4`, batch size 512, gradient clipping 1, and at most 30 epochs. The
minimum validation SmoothL1 checkpoint is retained with patience 5.

Generator training has no Stage 2 report or Stage 3 ranker dependency. Its
checkpoint records the duration token and generator data split only. The
current latent-cache filename still contains `stage3` for artifact
compatibility; the file supplies frozen Encoder latents, not a progress-ranker
model or score.

## Train Three Seeds

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/train_stage4_generator.py \
  --device cuda \
  --latent-cache outputs/stage3_progress_latents.pt \
  --seeds 3072,3073,3074 \
  --output outputs/stage4_generator_training.json
```

Checkpoints are written under `outputs/stage4_generator_checkpoints/` and are
selected independently for every seed.

## Historical Offline Candidate Validation

The remainder of this document reproduces the retired Stage 4 selector. It is
not the final high-level method.

Inference keeps `E_theta`, `F_theta`, `R_phi`, and `D_psi` in evaluation mode.
Only generator dropout is enabled for stochastic sampling. For each test source,
the evaluation reports deterministic and GT-oracle best-of-32 latent error and
cosine similarity, pairwise candidate diversity, and generated versus GT
residual norms.

Actual selection never uses GT. It applies:

```text
R_phi(z_t, g_i, 3) >= eta_R from the Stage 2 report
Delta_i = D_psi(z_t, z_T) - D_psi(g_i, z_T) > 0
g_star = argmin_i D_psi(g_i, z_T).
```

The complete selector also evaluates `z_T` as a direct-goal candidate. It joins
the same pool only when `R_phi(z_t,z_T,3) >= eta_R` and predicted progress is
positive; selection still minimizes `D_psi`. Deterministic and no-RC baselines
remain unchanged.

The report includes candidate RC-pass rate, per-source `N_RC=0/1/>=2`, joint
RC-and-progress coverage, no-candidate rate, selected RC and progress scores,
diversity, component timing, and the gap between generated-candidate and true
`z_{t+3}` RC scores.

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/evaluate_stage4_candidates.py \
  --device cuda \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --stage2-report outputs/rc_filter_only_tau3.json \
  --training-report outputs/stage4_generator_training.json \
  --output outputs/stage4_candidate_validation.json
```

`generated_target_rc_shift` means only the RC-score distribution difference
between generated candidates and the true `z_{t+3}`. It is not evidence that RC
is miscalibrated. Such a claim requires generated-candidate execution labels and
separate AUROC, ECE, Brier, FPR, and FNR measurements.

## Closed-Loop Comparison

Five methods run on identical test episodes with matched environment budgets and
CEM seed rules:

```text
reference trajectory waypoint
deterministic single generator candidate
32 candidates + D_psi without RC
32 candidates + RC + D_psi
flat RC-LeWM
```

Once a subgoal is selected, it stays fixed for the entire local segment. The
low-level sequence is exactly:

```text
h_rem=3: H_plan=3, H_exec=1, execute 5 environment actions
h_rem=2: H_plan=2, H_exec=1, execute 5 environment actions
h_rem=1: H_plan=1, H_exec=1, execute 5 environment actions
```

The real observation is encoded after every action block and updates the
generator history. The high level runs again only after the local budget is
exhausted. If RC plus progress leaves no candidate, the fallback plans directly
to `z_T`, executes one model step, and retries the high level. Environment
termination uses the trajectory's original TwoRoom task target and stops the
task immediately; no custom subgoal completion radius is used. The simulator's
`pos_target` is restored only to reproduce the official termination condition
and measure physical progress. It is never passed to the generator, RC, or
`D_psi`; their final goal remains `z_T = E(o_T)`.

`reference_trajectory_waypoint` uses the demonstration's future waypoint and is
not an Oracle: after the first action block, the real rollout can have left the
reference trajectory.

The closed loop uses the official TwoRoom environment horizon of 100 steps.
This comes directly from RC-aux `eval.py`, which sets `world.max_episode_steps`
to twice the TwoRoom `eval_budget` of 50. Demonstration length never controls
episode eligibility or an individual rollout horizon. It is reported only as a
description of test-task difficulty.

The formal subset is sampled once, without replacement, from the complete
held-out successful episode pool `[5000,10000)`. By default the fixed sample has
150 episodes and uses seed `20260811`. Every method and all three generator
seeds use exactly these episode indices, the same 100-step horizon, and the same
CEM seed rule.

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/evaluate_stage4_closed_loop.py \
  --device cuda \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --stage2-report outputs/rc_filter_only_tau3.json \
  --training-report outputs/stage4_generator_training.json \
  --num-episodes 150 \
  --episode-sample-seed 20260811 \
  --episode-horizon-env-steps 100 \
  --output outputs/stage4_closed_loop.json
```

The formal run contains `3 generator seeds * 5 methods * 150 episodes = 2250`
rollouts. Primary metrics are task success, completion steps, realized `D_psi`
progress, fallback rate, and candidate coverage. Euclidean target-distance
progress is diagnostic only. Reports also include selected RC score, diversity,
and generator/RC/`D_psi`/CEM time. Results are aggregated across all generator
seeds. Paired bootstrap 95% intervals use episode as the resampling unit,
average seeds within each episode, and compare the complete selector with every
baseline on identical episodes.

## RC And Direct-Goal Factorial Ablation

The original comparison changes two selector components at once: its no-RC
method has neither generated-candidate RC filtering nor a direct-goal precheck,
while the complete method has both. Two additional cells isolate their effects:

| Method | RC-filter generated candidates | RC-precheck direct `z_T` |
|---|---:|---:|
| `stochastic32_dpsi_no_rc` | no | no |
| `stochastic32_rc_dpsi_no_direct` | yes | no |
| `stochastic32_dpsi_direct_rc` | no | yes |
| `stochastic32_rc_dpsi` | yes | yes |

`stochastic32_rc_dpsi_no_direct` completely excludes `z_T` from the selection
pool. `stochastic32_dpsi_direct_rc` applies positive `D_psi` progress to all 32
generated candidates without RC-filtering them; `z_T` is admitted separately
only when its own RC score passes `eta_R` and its predicted progress is positive.
All feasible entries are still ranked by minimum `D_psi`.

The dedicated runner reads the exact 150 episode indices and three generator
checkpoints from the completed baseline report. It retains the official
100-environment-step horizon, matched CEM/dropout seed rules, fallback, and the
fixed-subgoal `H_plan=3,2,1`, `H_exec=1` execution protocol. It reuses the two
existing cells and runs only the two missing cells, for 900 new rollouts. A
partial report is written after every rollout and is resumed automatically by
rerunning the same command.

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/evaluate_stage4_ablations.py \
  --device cuda \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --baseline-report outputs/stage4_closed_loop.json \
  --output outputs/stage4_rc_direct_ablations.json
```

The report gives paired bootstrap 95% intervals for four contrasts: generated
RC with and without direct-goal precheck, and direct-goal precheck with and
without generated-candidate RC. For every episode, the three generator seeds
are averaged before episode-level bootstrap resampling.

## Decision Order

Interpret the results in this order:

1. If deterministic reconstruction is good but pairwise diversity collapses,
   replace the single query with multi-query or a GMM head.
2. If generated candidates have substantially lower RC coverage than true
   `z_{t+3}`, address generated-target RC distribution shift.
3. Compare complete selection against deterministic, no-RC, reference, and flat
   control in the real environment.
4. Consider diffusion only if the lightweight generator's missing multimodal
   coverage is established as the limiting factor for final success.
