# Residual-Dynamics Stage 1 Preregistration

Status: **entered / protocol frozen before closed-loop outcomes**  
Date: 2026-07-23  
Platform: native Windows

## Question

With the Amendment 17 environment, predictor, risk calculation, MPPI sampling
budget, and safety chain held fixed, does a frozen structured iCODE residual
improve short-horizon robot-dynamics prediction and closed-loop efficiency over
nominal dynamics without increasing collisions?

This stage does not ask the residual model to solve localization drift, obstacle
forecasting, route topology, or crossing-intent selection.

## First comparison

- `nominal`: existing dynamic-unicycle prediction.
- `icode_residual`: nominal prediction plus one frozen L57 structured residual.

The three existing L57 training seeds form model blocks:

- `20261201`
- `20261202`
- `20261203`

No checkpoint will be selected using Amendment 17 closed-loop outcomes.

## Compatibility preflight

All three frozen checkpoints were loaded through the production
`PlatformResidualDynamics` runtime on CPU with TorchScript enabled.

- model type: `icode_residual`;
- state dimension: 5;
- control dimension: 2;
- output dimension: 5;
- zero-input probe: finite for all checkpoints;
- structured output: pose-derivative residual components remain zero while
  velocity and yaw-rate derivative residuals are active;
- current action bounds are contained within the L57 high-dynamic training
  bounds;
- control interval and MPPI horizon match the L57 `dt=0.1`, horizon-36 design.

## Experimental unit and blocking

One complete episode is the independent unit.

- Obstacle-process seed is a paired block.
- Residual checkpoint seed is a model block.
- Nominal and residual conditions use common episode seeds and identical
  obstacle trajectories.
- The nominal condition is run once per obstacle seed and reused as the common
  reference for all three checkpoint blocks; repeating the identical nominal
  controller is not treated as additional independent evidence.
- Time steps and prediction horizons are repeated measurements, not independent
  replicates.

Initial development episode seeds remain:

- `730100001`
- `730100003`
- `730100005`

No sealed seed is authorized in Stage 1.

## Frozen inputs

- Amendment 17 environment and recovery logic;
- V3 obstacle generator;
- Change-Aware IMM and probability overlay;
- collision-risk definition and hard threshold (`0.20`);
- localization and sensor model;
- start, goal, episode horizon, and safety chain;
- MPPI horizon (`36`) and candidate budget (`600`);
- RL and learned sampling prior disabled.

## Outcomes

### Prediction outcomes

- one-step velocity and yaw-rate derivative error;
- rollout error at horizons 5, 10, 20, and 36;
- position, heading, velocity, and yaw-rate components reported separately.

Prediction error is evaluated post hoc on the shared nominal trajectories using
simulator state only as an audit target. For the primary horizon-36 gate,
velocity/yaw-rate rollout RMSE is
`sqrt((RMSE_v^2 + RMSE_omega^2) / 2)`. This truth is never exposed to the
online controller.

### Closed-loop outcomes

- collision;
- goal completion and final goal distance;
- minimum clearance;
- active-escape and recovery counts;
- planner p50/p95/p99 compute time;
- residual inference authority/support diagnostics.

## Stage-1 gate

Proceed beyond the development integration gate only if:

1. every residual checkpoint loads and produces finite outputs;
2. at least two of three model blocks improve horizon-36 velocity/yaw-rate
   rollout error over nominal;
3. no model block increases the risk-enabled collision count;
4. median completion change is nonnegative or any loss is no worse than `0.02`;
5. median clearance loss is no worse than `0.02 m`;
6. enabled planner p95 remains below `150 ms`;
7. all relevant tests and artifact-integrity checks pass.

Failure is retained as evidence and does not authorize checkpoint selection,
environment retuning, or sealed evaluation.

## First executable task

Implement a paired runner that switches only `planner.prediction_mode` and the
frozen checkpoint path, writes the full resolved configuration and checkpoint
SHA-256, and evaluates nominal versus residual under common random numbers.
