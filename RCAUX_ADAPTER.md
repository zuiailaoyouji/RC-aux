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

# Primary hierarchical-planning path: goal_latent is [B, 192].
plan = adapter.plan_to_latent(observation, goal_latent)
actions = plan.actions_to_execute_env_steps

# Official image-goal path retained for regression tests.
image_plan = adapter.plan_to_image(observation, goal_image)
```

`encode_observation` and `plan` accept raw uint8 CHW/HWC images, or float images
in `[0, 1]`, and apply the official ImageNet preprocessing internally.
`predict_latents` accepts aligned latent/action history plus optional future
action blocks and performs an open-loop rollout. `reachability` returns
probabilities by default; pass `return_logits=True` for raw logits.

The adapter uses explicit units. `planning_horizon_model_steps` controls how far
CEM plans, while `execution_horizon_model_steps` controls how much of that plan
is executed before the next high-level decision. The environment profile defines
`model_step_env_steps`; for TwoRoom one model step is five environment steps.
`PlanResult.planned_actions_env_steps` contains the full plan and
`PlanResult.actions_to_execute_env_steps` contains only the execution prefix.

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
