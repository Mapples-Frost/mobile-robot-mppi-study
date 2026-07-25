# Residual-Dynamics Stage 1 Amendment 1 Preregistration

Status: **frozen before Amendment 1 closed-loop outcomes**  
Date: 2026-07-23  
Scope: bounded integration-remediation probe

## Failure being addressed

Raw Stage 1 passed the prediction gate but failed safety, clearance and compute
gates. Posthoc support audit found global `x,y` inputs 5.97--6.23 standard
deviations from the L57 training center even though the MuJoCo floor and
actuator dynamics are translationally homogeneous.

## Single intervention

For residual inference only, replace state components `x,y` with the common,
checkpoint-embedded training feature means:

- `x = 3.245218276977539`
- `y = 0.3136115074157715`

The production `CanonicalizedStateResidualDynamics` adapter performs this
mapping. State components `theta,v,omega`, controls, model weights and all
environment/controller settings remain unchanged.

These values are checkpoint metadata fixed before this probe, not values fitted
to Amendment 17 outcomes.

## Screen

Run the failure-revealing obstacle seed `730100003` under:

- one shared nominal reference; and
- all three frozen residual checkpoint blocks.

No other development or sealed seed is opened by this screen.

## Gate and stopping rule

The amendment may advance to the complete three-seed development comparison
only if:

1. all three residual blocks remain finite and improve horizon-36
   velocity/yaw-rate rollout RMSE;
2. all three residual episodes have zero collision;
3. median completion delta is at least `-0.02`;
4. median clearance delta is at least `-0.02 m`;
5. maximum residual planner p95 is at most `150 ms`;
6. artifact and forecast contracts pass.

If any condition fails, stop. Do not change the canonical values, safety logic,
costs, environment, checkpoint set, or seed.
