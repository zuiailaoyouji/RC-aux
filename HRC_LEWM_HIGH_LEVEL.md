# HRC-LeWM Current High-Level Method

This document is the authoritative high-level design. Stage 1-4 reports and
the Stage 5B 1-NN control remain reproducible historical experiments, but they
are not deployment specifications.

## Current Data Flow

```text
observation history -> frozen Encoder -> [B,T,192] raw latents
                                      |
                                      v
                               frozen Generator
                                      |
                              [B,N,192] raw candidates
                                      |
       R_G(z_t,g_i) >= eta_3 filter -> argmax R_G(g_i,z_G)
                                      |
                                      v
                         selected latent subgoal g_star
                                      |
                                      v
                  RC-LeWM low-level latent-goal planner
```

The direct-goal rule is exclusively:

```text
R_G(z_t,z_G) >= eta_3  =>  select z_G before generated candidates.
```

There is no `D_psi`, `D_phi`, positive-progress threshold, or high-level query
to the RC-aux finite-horizon reachability head. Generated candidates are passed
to `R_G` exactly as produced. Deployment has no latent retrieval bank, kNN
average, or 1-NN projection.

If neither the direct goal nor a generated candidate passes the competence
filter, the selector marks a flat-goal fallback. The controller plans directly
to `z_G` for one low-level model step and retries the high level from the new
real observation.

The implementation is `GlobalReachabilitySelector` in
`hrc_lewm_high_level.py`. It enforces `[B,T,D]`, `[B,N,D]`, and `[B,D]` shape
conventions and loads paired Generator/`R_G` checkpoint seeds. Each `R_G`
checkpoint supplies its own calibrated `eta_3`; thresholds are not averaged.

## Reachability Responsibilities

The two reachability modules are complementary:

| Module | Interface | Meaning | Owner |
|---|---|---|---|
| `R_G(z,g)` | horizon-free scalar | discounted witnessed hitting-time potential used for high-level competence filtering and long-range ranking | high level |
| `R_local(z,g,h)` | integer model-step horizon | finite-budget reachability used by RC-LeWM rollout cost and low-level planning reliability | low level |

`R_G` is not claimed to be absolute global reachability. Its competence region
is defined operationally by its Stage 5A calibration. The high level does not
assert that every selected candidate must be physically completed within three
model steps. The three-step duration remains the current generator/controller
cadence and low-level allocation, not the semantic contract of the retired
finite-budget high-level classifier.

The low-level `R_local` must not be removed from `RCAuxAdapter` or RC-LeWM. It
continues to receive explicit model-step horizons while MPC follows
`H_plan=h_rem` and `H_exec=1`.

## Projection Status

Exact raw-space 1-NN projection was used only in Stage 5B to control generated
candidate manifold shift. It established an on-manifold mechanism result, not
a deployable retrieval architecture. The next generator work must improve raw
candidate manifold quality so the final path remains:

```text
raw generated candidates -> R_G -> RC-LeWM.
```

## Active Ablations

Current high-level ablations are limited to:

- `R_G` rank-only versus `R_G` filter+rank;
- generator architecture, diversity, and raw-candidate manifold quality;
- candidate count or sampling strategy only in separately preregistered
  generator experiments.

`D_psi`, `D_phi`, the old high-level `R_local` gate, and proposal-aware tuning
of that old local head are retired from the active method.

## Minimal Usage

```python
from hrc_lewm_high_level import GlobalReachabilitySelector

selector = GlobalReachabilitySelector.from_checkpoints(
    "outputs/stage4_generator_checkpoints/stage4_generator_seed3072.pt",
    "outputs/stage5a_global_reachability_checkpoints/"
    "stage5a_global_reachability_seed3072.pt",
    device="cuda",
)

selection = selector.select(
    history_latents,  # [B,T,192], 1 <= T <= 3
    terminal_latent,  # z_G=E(o_T), [B,192]
    dropout_seed=seed,
)
subgoal = selection.selected_latent
```

When `selection.fallback_to_flat_goal[b]` is true, execute one flat-goal model
step and retry instead of treating the fallback goal as a three-step accepted
subgoal.
