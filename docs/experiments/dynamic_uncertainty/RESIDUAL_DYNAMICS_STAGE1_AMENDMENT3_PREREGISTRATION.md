# Residual-Dynamics Stage 1 Amendment 3 Preregistration

Status: **frozen before Amendment 3 closed-loop outcomes**  
Date: 2026-07-23  
Scope: early causal residual-reliability routing  
RL status: **disabled**

## Motivation

Amendment 2 proved that a stall-triggered residual shutdown occurs after the
dangerous lateral displacement has accumulated. Amendment 3 therefore changes
authority before that displacement: it activates the repository's existing
innovation-based reliability gate outside the stall guard.

After every completed physical transition, the controller compares normalized
one-step velocity and yaw-rate errors from the nominal and residual models.
Residual authority starts at zero and rises only when exponentially weighted
completed-transition evidence favors the residual. No current/future simulator
truth, obstacle future or collision outcome is available to the current action.

## Frozen intervention

The parameters are copied without tuning from the prior L54 causal reliability
gate calibration, with actuation-delay context disabled so this amendment tests
only measured innovation evidence:

- state channels: `v`, `omega`;
- scales: `0.25 m/s`, `0.60 rad/s`;
- forgetting factor: `0.95`;
- minimum completed transitions: 8;
- confidence multiplier: 0;
- off/on thresholds: `-0.05`, `0.05`;
- rise/fall rates: `0.25`, `0.50`.

The Amendment 2 stall latch remains as a downstream fail-safe. Predictor,
probability calculation, MPPI costs, safety controller, obstacle process,
checkpoints and nominal dynamics remain frozen.

An offline causal replay of the failed block-2 trace predicts zero residual
authority through the early divergence window and later nonzero authority. This
is mechanism evidence only, not a closed-loop safety result.

## Development probe and gate

- seed: `730100003`;
- one shared nominal run and three frozen residual checkpoint blocks;
- randomized schedule seed: `730199992`;
- collision stops the screen;
- compute failure is recorded but does not stop remaining safety blocks;
- no sealed seed is authorized.

Required:

- zero collision increase in every residual block;
- all three blocks reach reliability authority at least 0.5, preventing a
  nominal-only pass;
- median completion and clearance non-inferiority;
- raw residual H36 prediction improvement in all three blocks;
- forecast/artifact contracts pass;
- planner p95 at most 150 ms.

This reuses an observed development seed and therefore cannot support an
independent confirmation claim even if all gates pass.

