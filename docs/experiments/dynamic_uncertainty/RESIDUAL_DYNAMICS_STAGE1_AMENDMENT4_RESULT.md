# Residual-Dynamics Stage 1 Amendment 4 Result

Status: **failed at the preregistered first residual block**  
Date: 2026-07-23  
RL status: **disabled**

Checkpoint block 1 collided under the structure-preserving residual interface,
so the probe stopped before the remaining three jobs.

- collision: yes;
- mean / maximum innovation authority: 0.551 / 1.000;
- final goal distance: 6.129 m;
- planner p95: 409.61 ms;
- stall fail-safe latched: yes.

The structure mask did not remove the closed-loop failure, so pose-channel
residuals are not the primary cause. The learned velocity/yaw-rate correction
itself changes MPPI's receding-horizon decisions enough to enter the obstacle's
return corridor.

This does not make the residual model useless. On the saved nominal seed
`730100003` trajectory, the structure-preserving models still improve H36
velocity/yaw-rate RMSE relative to nominal:

| block | nominal RMSE | masked residual RMSE | relative improvement |
|---:|---:|---:|---:|
| 0 | 0.06683 | 0.05625 | 15.83% |
| 1 | 0.06683 | 0.05572 | 16.62% |
| 2 | 0.06683 | 0.05601 | 16.18% |

The supported conclusion is narrower: these checkpoints are useful predictors
but are not safe as direct full-horizon MPPI dynamics corrections in this
crossing task. Direct residual integration is frozen pending a bounded or
task-aware retraining protocol. The nominal probabilistic MPPI remains the safe
control baseline for the next RL-prior stage.

Artifact:
`research_artifacts/dynamic_uncertainty_residual_stage1_amendment4_probe/`.

