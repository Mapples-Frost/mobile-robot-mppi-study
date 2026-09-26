# L50 Iterative Path-Data Confirmation: Pre-registered Protocol

Date frozen: 2026-07-16 (Asia/Shanghai)

## Purpose

L50 tests whether task-specific, on-policy residual retraining preserves the
independently confirmed smoothness benefit from L48 while resolving the weak and
model-seed-dependent path-efficiency result. It is not used to tune the three
L49 checkpoints.

The L50 closed-loop experiment is permitted to start only if all three L49
checkpoints first pass the separately specified offline H=36 prediction gate on
both the held-out test split and the unseen-domain split.

## Frozen design

- Comparison: `traditional_nominal` versus `traditional_icode`.
- Residual checkpoints: three independent L49 initialisation seeds.
- Paths: ellipse, sine, and figure-eight.
- Physics: both frozen L47/L48 command-delay domains.
- Paired evaluation seeds: `21460731` through `21460740`.
- Total episodes: 3 model blocks x 3 paths x 2 domains x 10 seeds x 2 methods
  = 360.
- Sealed seeds `21460741` through `21460750` remain unused.
- Planner uses the known command delay in both methods.
- The nominal reference, MPPI budget, path cost, safety logic, plant domains,
  and all L48 acceptance thresholds remain unchanged.

## Frozen acceptance gate

L50 deliberately inherits every L48 clause without relaxation:

1. complete artifact and seed-integrity audit;
2. positive mean path-length reduction in all 3 model blocks;
3. positive mean control-jerk reduction in all 3 model blocks;
4. overall mean path-length reduction at least 0.03 m;
5. hierarchical 95% bootstrap lower confidence bound above zero for path-length,
   issued-control-jerk, and applied-control-jerk reductions;
6. relative cross-track RMSE increase no greater than 5%;
7. no net success loss and no net collision increase;
8. mean completion-ratio difference at least -0.01;
9. candidate mean planner computation time no greater than 50 ms.

All clauses must pass. Failure is reported as failure; individual positive
endpoints may still be reported, but cannot be relabelled as a passed joint gate.

## Interpretation boundary

This is an independent closed-loop confirmation of a task-specific residual
training intervention. It does not test an RL sampling prior and it does not
justify claims about arbitrary obstacles, real-robot transfer, or unseen forms
of model mismatch.
