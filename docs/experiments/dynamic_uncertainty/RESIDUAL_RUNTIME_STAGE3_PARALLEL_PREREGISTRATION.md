# Residual Runtime Stage 3 Parallel-Shield Preregistration

Date: 2026-07-23  
Status: preregistered before implementation and parallel episodes.

## Trigger

The CUDA-Graph closed-loop matrix retained 12/12 goal reaches, zero collisions,
all prediction and noninferiority gates, and a 55.8% median residual-planner
P95 reduction relative to Stage 2. Its maximum P95 was still `136.79 ms`, so
the frozen `100 ms` real-time gate failed.

Residual MPPI solve time is now approximately 47--50 ms in observed episodes,
while nominal MPPI remains approximately 53 ms P95. The shield invokes the two
independent controllers sequentially before comparing their plans, making the
total close to their sum.

## Single authorized change

Add opt-in concurrent invocation of the nominal and residual planners using two
persistent worker threads. The controllers:

- have separate RNGs, previous sequences, dynamics and diagnostics;
- read the same immutable observation and reference for a decision;
- do not exchange state until both `PlanResult` objects are complete;
- retain the exact existing shield evaluation and selection order afterward.

Activation requires
`residual_safety_shield.parallel_planning_enabled: true`. Sequential execution
remains the default. Controller shutdown is added to the experiment cleanup
path so worker threads cannot leak across episodes.

## Frozen development matrix

All 12 cells over three obstacle seeds and three residual checkpoint blocks are
rerun. No episode is carried forward. Checkpoints, CUDA Graph rollout,
candidate budget, horizon, integrator, risk calculation and shield thresholds
remain unchanged. RL is disabled and sealed seeds remain unopened.

The same safety, completion, clearance, residual-participation and artifact
gates apply. Maximum residual-controller P95 must be at most `100 ms`.
Parallel scheduling is considered a runtime implementation detail only if the
episode-level outcomes remain safe and noninferior.
