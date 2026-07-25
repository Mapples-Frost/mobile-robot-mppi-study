# Residual Runtime Stage 3 Amendment 2

Date: 2026-07-23  
Status: preregistered before implementation and measurement.

## Trigger

Amendment 1 preserved the legacy trajectory to below `8e-16`, but eager
device-resident integration increased worst-block P95 from `154.41` to
`183.65 ms`. Moving small nominal and RK4 operations to CUDA created additional
kernel launches without removing launch scheduling overhead.

## Single authorized change

Retain the Amendment 1 device-resident arithmetic, but capture the fixed
`[600,36,2]`, H36, RK4 CUDA operation sequence once and replay it through
`torch.cuda.CUDAGraph`. Static input and output buffers are retained by the
adapter; each call copies new state/control values into the input buffers,
replays the graph, validates the output and performs one final host copy.

This mechanism is opt-in through
`planner.residual_cuda_graph_enabled: true`, requires CUDA and the
device-resident rollout path, and does not change checkpoints, candidate
controls, costs, safety thresholds or stochastic state.

## Frozen comparison and gate

CUDA legacy-transfer and CUDA-Graph arms receive the same control batch in all
three checkpoint blocks. Schedule seed is `730199910`; every cell receives five
warm-ups and thirty measurements.

The CUDA-Graph arm passes only if:

- all outputs are finite;
- maximum trajectory difference from legacy CUDA is at most `1e-5`;
- worst-block P95 improves by at least 30%;
- worst-block P95 is no more than `100 ms`.

A screen pass authorizes shield-level episode qualification, not sealed
evaluation.
