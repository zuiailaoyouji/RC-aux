# Historical And Control Experiments

These files remain in the repository only to reproduce the evidence that led
to the current method. They are not runtime dependencies and must not be used
as deployment specifications.

| Experiment | Status | Historical question |
|---|---|---|
| `RC_FILTER_EXPERIMENT.md` | archived Stage 1/2 | Can the original finite-budget `R_local` separate witnessed local targets? |
| `STAGE3_PROGRESS_RANKER.md` | archived | Can a separate `D_psi` rank demonstrated temporal progress? |
| `STAGE4_HIGH_LEVEL_GENERATOR.md` evaluation sections | archived | How did `R_local + D_psi` behave with generated candidates? |
| `STAGE4C_GENERATED_TARGET_RC_DIAGNOSTIC.md` | archived failure analysis | Did generated targets shift the old local-RC score distribution? |
| `STAGE4D_ON_MANIFOLD_RC_CONTROL.md` | archived failure analysis | Did the old finite-horizon gate help after 1-NN control? |
| `STAGE5B_HECRL_SELECTOR.md` | retained controlled validation | Does the new `R_G` filter/rank mechanism work when 1-NN removes manifold shift? |

The corresponding scripts and tests are intentionally retained so published
numbers remain reproducible. In particular, their imports of `D_psi`, old
`eta_R`, positive-progress thresholds, or 1-NN projection do not imply that
those components belong to the current system.

The active design is documented only in `HRC_LEWM_HIGH_LEVEL.md`.
