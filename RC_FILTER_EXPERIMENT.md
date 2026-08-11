# RC Filter-Only Oracle Experiment

> **Archived Stage 1/2 diagnostic.** This document validates the original
> finite-budget head as a local classifier. Its `eta_R` is not used by the
> current high-level selector. The same head remains active inside low-level
> RC-LeWM planning.

This experiment isolates local RC filtering from subgoal progress ranking and
from a learned subgoal generator. It does not encode a final task goal or rank
candidates by progress.

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
