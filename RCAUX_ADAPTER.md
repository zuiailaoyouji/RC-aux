# RC-aux Low-Level Adapter

Higher-level integrations can use `RCAuxAdapter` instead of depending on the
pickled checkpoint layout or `stable-worldmodel` policy internals:

```python
from rcaux_adapter import (
    RCAuxAdapter,
    RCAuxPlannerConfig,
    TWOROOM_PROFILE,
)

adapter = RCAuxAdapter.from_checkpoint(
    "tworoom_rcaux/rcaux_tworoom",
    profile=TWOROOM_PROFILE,
    cache_dir="/home/sxw/work/datasets/stable-wm",
    device="cuda",
    planner_config=RCAuxPlannerConfig(
        planning_horizon_model_steps=5,
        execution_horizon_model_steps=1,
    ),
)

latent = adapter.encode_observation(observation)  # [B, T, 192]
probability = adapter.reachability(
    source_latent,             # [B, 192] or [B, N, 192]
    candidate_target_latents,  # [B, 192] or [B, N, 192]
    horizon_model_steps=5,
)

# High level returns (g_star, tau_star); tau_star initializes h_rem.
h_rem_model_steps = tau_star_model_steps
plan = adapter.plan_to_latent(
    observation,
    g_star,  # [B, 192]
    planning_horizon_model_steps=h_rem_model_steps,
    execution_horizon_model_steps=1,
)
actions = plan.actions_to_execute_env_steps
h_rem_model_steps -= 1

# Official image-goal path retained for regression tests.
image_plan = adapter.plan_to_image(observation, goal_image)
```

`encode_observation` and `plan` accept raw uint8 CHW/HWC images, or float images
in `[0, 1]`, and apply the official ImageNet preprocessing internally.
`predict_latents` accepts aligned latent/action history plus optional future
action blocks and performs an open-loop rollout. `reachability` returns
probabilities by default; pass `return_logits=True` for raw logits.

The latent-goal planner also uses the checkpoint's RC-aux
`rollout_open_loop`. For an observation history `[B,L,C,H,W]`, where `1<=L<=3`,
provide the `L-1` previously executed action blocks through
`history_action_blocks`. The first candidate action block is aligned with the
latest history latent; the remaining `H_plan-1` candidates are future actions.
The rollout therefore always returns exactly `H_plan=h_rem` future latents,
independent of `L`. With `L=1`, no history action blocks are required.

The adapter uses explicit units. `planning_horizon_model_steps` controls how far
CEM plans, while `execution_horizon_model_steps` controls how much of that plan
is executed before the next high-level decision. The environment profile defines
`model_step_env_steps`; for TwoRoom one model step is five environment steps.
`PlanResult.planned_actions_env_steps` contains the full plan and
`PlanResult.actions_to_execute_env_steps` contains only the execution prefix.

The cross-level budget convention is:

\[
\begin{aligned}
\text{High level:} \quad & (g^\star,\tau^\star) \\
\text{Initialize:} \quad & h_{\mathrm{rem}}=\tau^\star \\
\text{Low-level MPC:} \quad & H_{\mathrm{plan}}=h_{\mathrm{rem}} \\
\text{Execute:} \quad & 1\ \text{model step} \\
\text{Update:} \quad & h_{\mathrm{rem}}\leftarrow h_{\mathrm{rem}}-1 \\
\text{Replan:} \quad & \text{same } g^\star,\ \text{smaller } H
\end{aligned}
\]

`tau_star_model_steps` is therefore only the initial budget. On each planner
call, pass the current `h_rem_model_steps` as
`planning_horizon_model_steps`; this makes the actual low-level
`PlanConfig.horizon` equal to `h_rem_model_steps`. The value can be a Python
integer or scalar integer tensor. `execution_horizon_model_steps` is fixed to one
for this loop and must never be folded into `h_rem_model_steps`.

`RCAuxPlannerConfig` defaults `execution_horizon_model_steps` to one for the
HRC-LeWM closed loop. Set it explicitly to five only when reproducing the
official RC-aux open-loop evaluation protocol.

Changing either the latent or image subgoal automatically invalidates CEM warm
start state. Planner diagnostics record both horizons in both units, CEM costs,
warm-start decisions, reachability settings, timing, and action-scaler statistics.
Use `return_diagnostics=True` on `reachability` to obtain logits, probabilities,
horizons, query shapes, and probability summaries together. Both diagnostics
objects provide `to_log_dict()` for JSON-compatible experiment records.

The official planner accepts a goal image. The policy preprocessing converts it
to a normalized image tensor, and the checkpoint's `get_cost` method encodes it
to a goal latent internally. It does not directly accept a pre-encoded goal
latent. `RCAuxAdapter` adds a numerically equivalent latent-goal cost path for
hierarchical planning while preserving the original image-goal path for
regression testing.

## Full Checkpoint Goal-Path Regression

Run the full official TwoRoom CEM configuration on GPU to compare the image-goal
and latent-goal planner paths end to end:

```bash
python tools/compare_adapter_goal_paths.py \
  --device cuda \
  --output outputs/adapter_goal_path_full.json \
  --fail-on-mismatch
```

The defaults match the official TwoRoom planner: 5 model-step planning and
execution horizons, 5 environment steps per model step, 300 candidates, 30 CEM
iterations, top-k 30, and seed 42. Both paths use fresh solvers with warm start
disabled. The JSON report compares normalized action blocks, the complete
environment-step plan, the execution prefix, and final CEM costs.

## Future-Latent Subgoal Oracle

Use a real future observation from the official dataset as a latent subgoal,
`g = E(o_{t+k})`, and run closed-loop control in the real TwoRoom environment:

```bash
python tools/run_latent_subgoal_oracle.py \
  --device cuda \
  --future-k-env-steps 25 \
  --tau-star-model-steps 5 \
  --require-replay-match \
  --require-success
```

The environment starts at the dataset state at `t` and uses the dataset state at
`t+k` only as the environment success target. The planner receives only the
encoded latent goal. It replans after one model step, which is five environment
steps for TwoRoom. The planning horizon follows the remaining cross-layer budget,
so the default horizon sequence is `5, 4, 3, 2, 1` and execution stops after 25
environment steps if the goal has not been reached. The output JSON records
state distances, executed actions,
reachability probabilities, warm-start decisions, and planner diagnostics. The
video places the live environment on the left and `o_{t+k}` on the right. Replay
validation permits a maximum uint8 pixel difference of one by default.

## Oracle RC Filtering Experiment

Evaluate the reachability filter before training a latent subgoal generator:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-rcaux \
.venv/bin/python tools/evaluate_rc_oracle_filter.py \
  --device cuda \
  --cache-dir /home/sxw/work/datasets/stable-wm \
  --tau-model-steps 3 \
  --eta-r 0.5 \
  --num-trials 10 \
  --output outputs/rc_oracle_filter_tau3.json
```

Each trial builds one Oracle candidate pool from real dataset observations. It
contains same-trajectory candidates at offsets within three model steps,
same-trajectory candidates at offsets beyond three model steps, and candidates
from other trajectories whose conservative displacement lower bound exceeds the
15-environment-step budget. The over-budget label describes temporal offset in
the demonstrated trajectory; it is not a shortest-path proof. Actual feasibility
is always measured by closed-loop execution in the environment.

The fixed threshold first removes candidates with
`R_phi(z_t, g_i, tau) < eta_r`. For each candidate and the current state, the
experiment queries the frozen reachability head toward the final goal at model
step horizons `H={1,2,3,4,5}` and computes
`D_phi(a,z_G)=sum_h(1-R_phi(a,z_G,h))`. Predicted progress is
`D_phi(z_t,z_G)-D_phi(g_i,z_G)`. The filtered strategy also rejects non-positive
progress and selects the remaining candidate with minimum `D_phi(g_i,z_G)`.
The progress-only baseline applies the same positive-progress and minimum-time
rule without local RC filtering. The random baseline samples from the complete
Oracle pool.

Every candidate is executed once with the same CEM seed within a trial. Results
for all three strategies are then derived from that shared execution table. Each
execution uses planning horizons `3, 2, 1`, executes one model step (five
environment steps), observes the real state, and replans. The JSON report includes
category pass/completion rates, threshold confusion statistics, ROC AUC, average
precision, strategy completion rates, physical and reachability-time progress
toward the final task goal, paired comparisons, and a threshold sweep.
`ready_to_train_high_level_generator` is true only when the pre-fixed RC strategy
strictly beats both baselines in completion rate and physical distance progress
toward the final task goal, and local RC score ROC AUC is greater than 0.5. The
physical metric keeps the validation independent of the RC head; reachability-time
progress remains an auxiliary diagnostic. A false verdict is written to the
report and returned as process exit status 1; it is an experimental result, not
necessarily a runtime failure.
