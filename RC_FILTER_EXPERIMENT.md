# RC Filter-Only Oracle Experiment

This experiment isolates local RC filtering from subgoal progress ranking and
from a learned subgoal generator. It does not compute `D_phi`, encode a final
task goal, or rank candidates by progress.

## Stage 1: Strict Oracle Pools

The local budget is fixed to three model steps, or 15 TwoRoom environment steps.
Each pool contains exactly two candidates from each category:

- `within_budget_witnessed`: a future state at offset 1, 2, or 3 model steps.
  Replaying the dataset actions from the source must reproduce the candidate
  state within tolerance in at most 15 environment steps.
- `same_trajectory_strict_over_budget`: a later state from the source trajectory
  whose collision-agnostic displacement lower bound is greater than 15 steps.
- `cross_trajectory_strict_over_budget`: a state from another trajectory in the
  same data split whose displacement lower bound is greater than 15 steps.

The lower bound uses the environment action limits, agent speed, action
dimension, and success radius. It ignores collisions, so walls can only increase
the real effort. The negative label is therefore strict for the local budget.

Calibration candidates use episodes `[0, 5000)`. Test candidates, including
cross-trajectory candidates, use episodes `[5000, 10000)`. No episode crosses
the split boundary.

Run only Stage 1 without loading the checkpoint or using a GPU:

```bash
PYTHONPATH=. .venv/bin/python tools/evaluate_rc_filter_only.py \
  --pool-only \
  --calibration-trials 10 \
  --test-trials 50 \
  --candidates-per-category 2 \
  --output outputs/rc_filter_stage1_tau3.json
```

The checked-in implementation has been run with these settings. It constructed
10 calibration pools and 50 test pools. All 120 within-budget candidates had
zero final-state witness error. All 240 negative candidates had lower bounds
strictly greater than 15; the observed range was 16 to 33 environment steps.

## Stage 2: RC Filtering Only

Every candidate is encoded by the frozen RC-aux encoder and scored only with

```text
r_local = R_phi(z_t, g_i, tau=3)
```

Every candidate is then executed with the same repeated CEM seeds. Each
execution uses planning horizons `3, 2, 1`, executes one model step, observes
the real environment, and replans. The strategy comparison is derived from the
shared execution table:

- uniform selection from the complete balanced pool;
- uniform selection from candidates with `r_local >= eta_R`.

These are exact expected completion rates over the evaluated candidates, not
single random draws. Conditional completion, unconditional completion with
abstention counted as zero, candidate coverage, trial coverage, and abstention
count are reported separately.

Unless `--eta-r` is supplied, `eta_R` is chosen only on the calibration split.
The selected threshold maximizes recall subject to calibration precision at
least 0.95 and false-positive rate at most 0.05. The test split is never used to
select the threshold. A threshold sweep on test is reported only as a
risk-coverage diagnostic.

Run the formal GPU experiment:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/evaluate_rc_filter_only.py \
  --device cuda \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --tau-model-steps 3 \
  --candidates-per-category 2 \
  --calibration-trials 10 \
  --test-trials 50 \
  --planner-repeats 3 \
  --output outputs/rc_filter_only_tau3.json
```

This executes 180 calibration candidate runs and 900 test candidate runs when
counting the three planner repeats. It is intentionally a formal experiment,
not a smoke test.

Inspect the decision fields:

```bash
jq '{
  stage1_strict_oracle_pool_validated,
  stage2_rc_filter_validated,
  ready_for_dphi_ranking_evaluation,
  validation_criteria,
  threshold: (.threshold | del(.calibration_threshold_table)),
  calibration: (.calibration | del(.by_category)),
  test: (.test | del(.by_category)),
  test_categories: .test.by_category
}' outputs/rc_filter_only_tau3.json
```

Stage 2 passes only when all of the following hold:

- calibration precision/FPR constraints are satisfied;
- test RC ROC AUC is greater than 0.5;
- test FPR remains within the calibrated maximum;
- at least half of test trials contain an RC-passing candidate;
- the paired bootstrap 95% confidence interval for conditional expected
  completion improvement over the complete pool is strictly above zero.

The process writes the full JSON report before returning status 1 when Stage 2
does not pass. A nonzero status can therefore be an experimental result rather
than a runtime error. `ready_to_train_high_level_generator` remains false because
the progress ranker has not been evaluated in this experiment.

## Stage 3: D-Phi Ranking Inside Feasible Sets

Stage 3 consumes the completed Stage 2 report. It does not rebuild the RC
threshold or mix RC classification performance into the primary ranking test.
The primary candidate set contains candidates whose Stage 2 closed-loop success
rate is at least `2/3`. The Stage 2 RC-passed set is evaluated separately as a
deployment diagnostic.

The final goal is the official TwoRoom goal image rendered with the agent at the
dataset `pos_target`. For every candidate, Stage 3 computes

```text
D_phi(g_i, z_G) = sum_{h=1..5} (1 - R_phi(g_i, z_G, h)).
```

The independent target is not coordinate distance. It is a continuous
obstacle-aware path cost that must pass through a valid door region and explicitly
accounts for both wall collision boundaries, door size, agent radius, success
radius, and environment speed. The remaining number of steps in the demonstrated
trajectory is also reported as a second diagnostic.

The following selectors are compared on exactly the same candidates:

- uniform random selection;
- minimum `D_phi`;
- minimum encoder latent squared L2;
- minimum obstacle-aware target cost, used only as an offline Oracle upper bound.

All Oracle-feasible candidates and any additional RC-passed candidates are
re-executed with the Stage 2 CEM seeds. This records the real final state and
allows obstacle-aware progress after execution to be compared without planner
sampling differences. The script also requires the rerun success rates to match
the Stage 2 report exactly.

Run Stage 3 on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
PYTHONPATH=. .venv/bin/python tools/evaluate_dphi_ranking.py \
  --device cuda \
  --stage2-report outputs/rc_filter_only_tau3.json \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --feasible-success-rate 0.6666666667 \
  --min-ranking-trials 30 \
  --output outputs/dphi_ranking_tau3.json
```

Inspect the result:

```bash
jq '{
  stage3_dphi_ranking_validated,
  ready_for_combined_stage4_evaluation,
  ready_to_train_high_level_generator,
  validation_criteria,
  rerun_success_rate_max_abs_difference,
  oracle_feasible_summary,
  rc_passed_summary
}' outputs/dphi_ranking_tau3.json
```

Stage 3 passes only if the Oracle-feasible set contains enough ranking trials,
rerun outcomes match Stage 2, and paired bootstrap confidence intervals establish
all of the following:

- `D_phi` pairwise accuracy is above chance;
- `D_phi` target cost is lower than uniform random selection;
- `D_phi` real executed progress is higher than uniform random selection;
- `D_phi` target cost is lower than latent L2 selection;
- `D_phi` real executed progress is higher than latent L2 selection.

Thus a positive result means that `D_phi` adds value beyond both random feasible
selection and the simpler latent-distance heuristic. A negative result leaves the
validated RC filter intact but rejects the current `D_phi` ranking rule. Generator
training remains disabled until a later combined Stage 4 evaluation passes.
