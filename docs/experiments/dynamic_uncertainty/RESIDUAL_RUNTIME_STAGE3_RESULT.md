# Residual Runtime Stage 3 Result

Date: 2026-07-23  
Status: **development runtime qualification passed; sealed and loaded-system
hard-real-time qualification remain open.**

## Starting bottleneck

The qualified Stage 2 residual shield had residual-controller P95 between
`221.84` and `422.08 ms` for a `100 ms` control period. Component profiling
showed that residual batch rollout dominated. The runtime configuration had no
explicit `planner.device`, so inference used CPU with one Torch thread.

## Backend screen

The frozen blocked screen compared CPU thread counts and CUDA using the same
`[600,36,2]` control batch in all three checkpoint blocks.

| Arm | Worst-block rollout P95 | Change from CPU-1 | Equivalent |
|---|---:|---:|---:|
| CPU, 1 thread | 302.03 ms | baseline | yes |
| CPU, 2 threads | 314.06 ms | -4.0% | yes |
| CPU, 4 threads | 325.48 ms | -7.8% | yes |
| CPU, 8 threads | 355.10 ms | -17.6% | yes |
| CUDA, legacy transfers | 163.58 ms | +45.8% | yes |

Additional CPU threads did not help this repeated small-network workload. CUDA
was eligible but still exceeded the control period before costs and shielding.

## Retained negative optimization

Amendment 1 kept the entire RK4 trajectory on CUDA but executed it eagerly.
It preserved trajectories to below `8e-16`, yet worsened concurrent
worst-block P95 from `154.41` to `183.65 ms`. Moving nominal arithmetic to the
GPU added many small kernel launches without removing launch-dispatch
overhead.

This approach was rejected and retained as a negative result.

## CUDA Graph optimization

Amendment 2 captured the fixed H36/K600/RK4 computation using static CUDA Graph
buffers. New state and control values are copied into the buffers, the graph is
replayed, and the completed trajectory is copied back once.

| Arm | Worst-block rollout P95 | Paired speedup | Maximum trajectory difference |
|---|---:|---:|---:|
| CUDA legacy-transfer | 93.86 ms | baseline | 0 |
| CUDA Graph | 15.19 ms | 83.8% | `7.8e-16` |

The microbenchmark gate passed on all three checkpoints.

## Sequential full-controller qualification

The first complete CUDA-Graph matrix retained:

- 12/12 goal reaches;
- zero collisions;
- zero collision-count increase in every block;
- all prediction, completion, clearance, forecast and participation gates.

Residual-controller P95 median fell to `118.51 ms`, a median paired reduction
of 55.8% from Stage 2. However, maximum P95 remained `136.79 ms`, so the only
failed gate was the frozen `100 ms` runtime target.

The remaining cost was structural: the shield waited for nominal MPPI and then
ran residual MPPI. Each planner was near 50 ms, so the shield approached their
sum.

## Parallel-shield qualification

The final opt-in implementation evaluates the independent nominal and residual
planners on two persistent worker threads. Both consume the same immutable
observation/reference and retain separate RNGs, previous sequences, dynamics
and diagnostics. The existing shield comparison remains sequential after both
plans complete.

All 12 parallel episodes completed:

| Block | Seed 006 P95 | Seed 008 P95 | Seed 010 P95 |
|---|---:|---:|---:|
| R01 | 90.03 ms | 86.59 ms | 76.16 ms |
| R02 | 91.04 ms | 97.19 ms | 93.75 ms |
| R03 | 62.37 ms | 72.54 ms | 96.55 ms |

The automatic gate passed with maximum P95 `97.19 ms`. Median paired P95
reduction was:

- 67.3% relative to the original Stage 2 CPU matrix;
- 26.0% relative to the sequential CUDA-Graph matrix.

## Behavior-equivalence audit

Parallel scheduling did not alter the sequential CUDA-Graph behavior in any of
the 12 episodes:

- step delta: `0` for every cell;
- maximum position difference: `0 m`;
- maximum executed-control difference: `0`;
- shield-acceptance delta: `0` for every residual cell;
- success and collision outcomes: exactly matched.

Relative to the original CPU Stage 2 matrix, CUDA numerical differences led to
bounded receding-horizon trajectory differences, but all task and safety
outcomes remained matched. The optimized matrix retained nondegenerate
residual contribution; block-level median effective-acceptance fractions were
40.7%, 44.7% and 34.7%.

## Scope and remaining limit

The result supports:

> On the current RTX 5060 Laptop GPU development machine, CUDA Graph residual
> rollout plus parallel shield planning reduces the qualified residual
> controller below the 100 ms P95 budget without changing the parallelized
> controller's closed-loop behavior.

It does not establish a hard-real-time guarantee. The worst observed P95 has
only `2.81 ms` margin, and the experiment did not add background GPU/CPU load,
thermal throttling or real-hardware I/O. Those belong to deployment stress
qualification.

RL remained disabled and sealed seeds were not opened. The residual subsystem
is now eligible for the next development stage: frozen RL Actor/HSS
integration, while retaining this runtime implementation as the baseline.

## Artifacts

- Initial screen:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_screen/`
- Device-resident negative result:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_amendment1/`
- CUDA Graph screen:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_amendment2/`
- Sequential full matrix:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_development/`
- Passing parallel matrix:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/`
- Parallel automatic gate:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/gate.json`
- Sequential-equivalence audit:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/sequential_equivalence_audit.json`
- Stage 2 comparison:
  `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/stage2_equivalence_audit.json`
