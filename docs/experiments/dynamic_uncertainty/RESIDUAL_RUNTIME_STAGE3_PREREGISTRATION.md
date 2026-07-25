# Residual Runtime Stage 3 Preregistration

Date: 2026-07-23  
Status: preregistered before runtime screening.

## Question

Can the qualified task-aware residual rollout be accelerated without changing
its model, MPPI horizon, candidate count, RK4 integrator, shield thresholds or
control semantics?

The Stage 2 profiler identifies residual batch rollout as the dominant
component: approximately `91.6 ms` on average, versus approximately `9.8 ms`
for nominal rollout. The qualified configuration did not specify
`planner.device`, so residual inference used CPU with one Torch thread despite
the CUDA-capable training environment.

## Stage 3A: backend/thread screen

The screen compares:

- CPU with 1 thread (frozen baseline);
- CPU with 2, 4 and 8 Torch threads;
- CUDA with the same TorchScript model.

Every arm receives the same `[600, 36, 2]` control batch in every checkpoint
block. The blocks are the three qualified residual checkpoint seeds.
Arm/block execution order is randomized with seed `730199907`. Each cell has
three unmeasured warm-ups and twenty measured full-horizon RK4 rollouts.

Runtime repetitions characterize machine timing and are not treated as
independent scientific episodes. The checkpoint seed is the runtime block.

## Frozen screen gate

An arm is eligible only if:

1. every output is finite;
2. maximum state-trajectory deviation from CPU-one-thread is at most `1e-5`
   over all candidates, horizon steps and state channels in every block;
3. its worst-block P95 rollout time improves by at least 20% over baseline.

Among eligible arms, choose the smallest worst-block P95. If no simple backend
arm passes, proceed to an implementation optimization that removes repeated
host/device transfers or redundant rollouts, then rerun this same screen.

## Stage 3B: controller-level qualification

The selected optimization must then be tested with the complete residual
shield, not only the neural network:

- same development seeds `730100006`, `730100008`, `730100010`;
- nominal and the three Stage 2 checkpoint blocks;
- same 600 candidates, H36, RK4 and risk configuration;
- RL disabled;
- sealed seeds unopened.

The optimized system must retain all goal reaches, zero collision increase,
the frozen completion noninferiority limit, nondegenerate effective residual
participation, and valid risk diagnostics. The deployment target is P95 below
the `100 ms` control period. Passing a microbenchmark alone is not a deployment
claim.

## Stopping rule

Only measured bottlenecks may be changed. If a modification changes the
learned checkpoint, shield thresholds, candidate budget, integration method or
obstacle-risk semantics, it is a new algorithmic amendment rather than a
runtime optimization and is not authorized by this protocol.
