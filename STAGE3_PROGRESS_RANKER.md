# Stage 3: Latent Progress Ranker

Stage 3 trains and validates only the long-range latent progress ranker
`D_psi`. The released RC-aux Encoder and reachability head `R_phi` remain
frozen. This stage has no environment execution and does not change the Stage 2
reachability threshold.

## Data Protocol

The script reads the episode split and `eta_R` directly from the validated
Stage 2 report. With the current TwoRoom report, successful episodes in
`[0, 5000)` are used for training and successful episodes in `[5000, 10000)`
are used only for testing. Episode IDs cannot overlap.

Each successful trajectory is sampled every five environment steps, matching
one RC-aux model step. The exact terminal observation is always appended and
encoded as `z_G`, including when the episode length is not divisible by five.
The Encoder produces all latents; Stage 3 reads no physical state or task-cost
labels.

For a training episode `(z_0, ..., z_T=z_G)`, pairs satisfy `i < j < T` and
supervise

```text
D_psi(z_j, z_G) < D_psi(z_i, z_G).
```

The ranker is a small MLP with input
`[g, z_G, g-z_G]`. Margin ranking loss is the primary objective. A weighted MSE
target equal to normalized remaining model steps is an auxiliary objective;
set `--regression-weight 0` to disable it.

Encoded train/test trajectories are cached in
`outputs/stage3_progress_latents.pt`. The cache records the dataset identity,
policy, model-step size, and exact episode IDs. A mismatched cache is rejected;
use `--rebuild-latent-cache` when intentionally changing those inputs.

## Test Protocol

For every eligible source state in a held-out successful episode, the candidate
pool contains later nonterminal model-step samples from that same episode. The
terminal latent is used only as `z_G`. Testing follows this fixed order:

1. Query the frozen local head at `tau=3` and retain candidates satisfying
   `R_phi(z_t, g_i, 3) >= eta_R`, where `eta_R` comes unchanged from Stage 2.
2. Evaluate `D_psi` on the complete RC-passed set. Pairwise accuracy and
   Spearman correlation are computed before predicted-progress filtering, so
   bad predictions cannot disappear from those metrics.
3. Compute `Delta_i = D_psi(z_t,z_G) - D_psi(g_i,z_G)`, remove candidates with
   `Delta_i <= 0`, and select the remaining candidate with minimum `D_psi`.
4. Compare the selected candidate with the candidate having the fewest real
   remaining trajectory model steps. If no positive-progress candidate remains,
   top-1 is counted as incorrect.

The only validation metrics are pairwise accuracy, per-query Spearman, and
top-1 accuracy. Their confidence intervals are bootstrapped by test episode.
Stage 3 passes when the pairwise interval is above `0.5`, the Spearman interval
is above `0`, the top-1 advantage interval is above its set-size-aware random
chance level, and at least 100 test queries contain two RC-passed candidates.

## Formal Run

Run the complete experiment on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/train_progress_ranker_stage3.py \
  --device cuda \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --stage2-report outputs/rc_filter_only_tau3.json \
  --latent-cache outputs/stage3_progress_latents.pt \
  --model-output outputs/stage3_progress_ranker.pt \
  --output outputs/stage3_progress_ranker.json \
  --require-validation
```

The command uses all successful episodes in both halves by default. It writes
the report and model checkpoint before returning status 1 when the validation
criteria do not pass.

Inspect the three test metrics:

```bash
jq '{
  stage3_progress_ranker_validated,
  validation_criteria,
  ranking_metrics: .test.ranking_metrics,
  evaluation_counts: .test.evaluation_counts
}' outputs/stage3_progress_ranker.json
```
