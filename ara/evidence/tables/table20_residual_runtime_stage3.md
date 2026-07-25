# Table 20: Residual Runtime Stage 3

## Frozen question

Can the Stage 2 task-aware residual controller be brought below the `100 ms`
P95 control-period budget without changing the qualified obstacle-avoidance
semantics, opening sealed seeds or enabling RL?

## Optimization screens

| Stage | Comparator | Worst-block rollout/controller P95 | Result |
|---|---|---:|---|
| Backend screen | CPU, 1 thread | 302.03 ms rollout | Baseline |
| Backend screen | CPU, 2/4/8 threads | 314.06 / 325.48 / 355.10 ms rollout | Rejected |
| Backend screen | Legacy-transfer CUDA | 163.58 ms rollout | Better, still over budget |
| Amendment 1 | Eager device-resident CUDA | 183.65 ms rollout versus 154.41 ms concurrent legacy | Rejected; trajectory difference below 8e-16 |
| Amendment 2 | Static-buffer CUDA Graph | 15.19 ms rollout versus 93.86 ms legacy | Passed micro gate; 83.8% faster |
| Full sequential | CUDA Graph, serial shield | 136.79 ms controller maximum | Safety passed; runtime failed |
| Full parallel | CUDA Graph, parallel shield | 97.19 ms controller maximum | Development runtime gate passed |

## Passing parallel matrix

| Residual block | Seed 730100006 | Seed 730100008 | Seed 730100010 |
|---|---:|---:|---:|
| R01 P95 | 90.03 ms | 86.59 ms | 76.16 ms |
| R02 P95 | 91.04 ms | 97.19 ms | 93.75 ms |
| R03 P95 | 62.37 ms | 72.54 ms | 96.55 ms |

All 12 complete episodes, including the three nominal references, reached the
goal with zero collisions. The nine residual cells had median P95 `90.03 ms`;
their median paired reduction was `67.3%` relative to Stage 2 and `26.0%`
relative to sequential CUDA-Graph execution.

## Behavior-equivalence and scope

Parallel versus sequential CUDA-Graph execution had zero step, position,
executed-control and shield-acceptance differences in every cell, with exactly
matching success and collision outcomes. The result is development-only on the
current RTX 5060 Laptop GPU. The worst P95 has only `2.81 ms` margin; background
load, thermal throttling, real-hardware I/O, hard-real-time guarantees, RL and
sealed seeds were outside scope.

## Evidence

- `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_screen/runtime_screen.json`
- `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_amendment1/runtime_screen.json`
- `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_amendment2/runtime_screen.json`
- `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_development/gate.json`
- `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/gate.json`
- `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/sequential_equivalence_audit.json`
- `docs/experiments/dynamic_uncertainty/RESIDUAL_RUNTIME_STAGE3_RESULT.md`
