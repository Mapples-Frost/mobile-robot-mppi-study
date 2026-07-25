# Residual-Dynamics Stage 1 Amendment 1 Result

Status: **FAIL — stopped by preregistered rule**  
Date: 2026-07-23  
Sealed seeds opened: **no**

The single-factor probe canonicalized global `x,y` to the common checkpoint
feature means before residual inference. The shared nominal seed-730100003
episode reproduced zero collision and final goal distance `0.3334 m`.

The first scheduled residual checkpoint was seed `20261202`. It:

- collided at approximately step 227;
- ended `6.1367 m` from the goal;
- reached minimum clearance `-0.0321 m`;
- had planner p95 `211.23 ms`, above the `150 ms` threshold.

Both the safety and compute conditions failed. The remaining two checkpoint
blocks were not completed. The early-stop artifact records two of four
scheduled jobs complete and two unobserved.

Canonicalization improves the posthoc aggregate horizon-36 velocity/yaw-rate
RMSE of the three checkpoints to approximately `0.0541`, versus `0.0662` for
nominal prediction, but it does not resolve the closed-loop collision. Spatial
feature OOD was therefore a genuine interface defect but not a sufficient
explanation for the control failure.

The retained interpretation is:

> Average multi-step dynamics accuracy is not sufficient to safely insert the
> current L57 residual into probability-constrained obstacle-avoidance MPPI.

No further canonical value, cost, safety, environment or checkpoint tuning is
authorized by this failed probe.

Evidence:

- `research_artifacts/dynamic_uncertainty_residual_stage1_amendment1_probe/early_stop.json`
- `research_artifacts/dynamic_uncertainty_residual_stage1_amendment1_probe/runs/nominal/seed_730100003/`
- `research_artifacts/dynamic_uncertainty_residual_stage1_amendment1_probe/runs/icode_residual/block_1/seed_730100003/`
