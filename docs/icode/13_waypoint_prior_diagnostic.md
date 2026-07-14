# Static-scene waypoint-prior causal diagnostic

## Status and scope

This document records a **single-seed smoke diagnostic**, not a paper result.
It asks one narrow question: when local MPPI cannot select a useful homotopy in
a cluttered static scene, does supplying a geometrically feasible sequence of
intermediate targets materially change the closed-loop outcome?

The offline A* path is used only to construct diagnostic waypoints. During
execution, obstacles still enter the planner through synthetic `LaserScan`,
`scan_guard`, and `local_obstacle_layer`. The planner never receives the global
obstacle list. Consequently, this test identifies a sampling/topology
bottleneck, but the offline route is **not** a deployable planner and is not part
of the proposed learned method.

## Pre-test scene audit

The legacy goal `(3.0, 3.0)` was collision-free but was not a sound terminal
benchmark point in the three cluttered scenes. Its robot-centre clearances were:

| Scene | Clearance at `(3.0, 3.0)` |
|---|---:|
| `lab_complex` | 0.297 m |
| `narrow_corridor` | 0.090 m |
| `u_trap_long_board` | 0.121 m |

The MPPI obstacle influence radius is 0.70 m. Thus a controller could be
penalized for entering the success region even though the configuration was not
in geometric collision. The diagnostic therefore used the variable goal
`(3.5, 3.5)`, whose clearances are 0.951 m, 0.797 m, and 0.812 m respectively.
This is a CLI-only diagnostic override and does not alter the stored legacy
scene configurations.

## Protocol

- MuJoCo actuated differential-drive plant;
- `K=400` MPPI samples, 36-step horizon, 0.1 s control period;
- at most 480 control steps;
- seed 0;
- memory off and RL off;
- nominal and frozen ICODE prediction models;
- direct point goal versus offline waypoints sampled every 0.45 m;
- 0.20 m offline route-clearance margin;
- 0.40 m intermediate waypoint switching tolerance;
- unchanged final success tolerance of 0.30 m.

## Observed smoke results

| Scene | Reference | Model | Success | Collision | Final distance | Safety interventions |
|---|---|---|---:|---:|---:|---:|
| `lab_complex` | direct | nominal | 0 | 0 | 4.330 m | 410 |
| `lab_complex` | direct | ICODE | 0 | 0 | 4.281 m | 404 |
| `lab_complex` | waypoints | nominal | 1 | 0 | 0.296 m | 40 |
| `lab_complex` | waypoints | ICODE | 1 | 0 | 0.287 m | 17 |
| `narrow_corridor` | direct | nominal | 0 | 1 | 3.609 m | 186 |
| `narrow_corridor` | direct | ICODE | 0 | 0 | 1.332 m | 11 |
| `narrow_corridor` | waypoints | nominal | 1 | 0 | 0.290 m | 2 |
| `narrow_corridor` | waypoints | ICODE | 0 | 0 | 0.309 m | 0 |
| `u_trap_long_board` | direct | nominal | 0 | 0 | 4.380 m | 351 |
| `u_trap_long_board` | direct | ICODE | 0 | 0 | 3.824 m | 379 |
| `u_trap_long_board` | waypoints | nominal | 0 | 0 | 0.629 m | 62 |
| `u_trap_long_board` | waypoints | ICODE | 0 | 0 | 0.464 m | 19 |

## Interpretation

1. A feasible route exists in every tested scene.
2. Direct-goal local MPPI cannot reliably discover the required route class.
3. Intermediate route information produces a large causal improvement for both
   prediction models. This motivates a future RL sampling prior that proposes
   useful candidate-sequence distributions rather than replacing MPPI.
4. ICODE improves prediction and often reduces intervention or terminal error,
   but it is not a global planner. Residual learning alone cannot solve a U-trap.
5. The U-trap endpoint loops show that route-class selection and stable terminal
   capture must be evaluated separately.
6. The narrow-corridor nominal direct run produced a real collision. It must be
   treated as a safety-regression case (especially lateral/box-corner coverage),
   not omitted from reporting.

No claim about mean performance, statistical significance, or superiority is
valid until the protocol is repeated across multiple seeds and frozen configs.

The narrow-corridor collision reported here was subsequently reproduced and
removed by the Python-3 safety-envelope change documented in
`14_safety_and_reference_gate.md`; this file intentionally retains the original
observation rather than rewriting historical smoke results.

## Reproduction

```bash
.venv/bin/python experiments/run_waypoint_prior_diagnostic.py \
  --configs configs/research/mujoco_lab_complex.yaml \
            configs/research/mujoco_narrow_corridor.yaml \
            configs/research/mujoco_u_trap_long_board.yaml \
  --output-dir /tmp/waypoint_diagnostic \
  --seeds 0 --num-samples 400 --max-steps 480 \
  --spacing 0.45 --route-margin 0.20 --goal 3.5 3.5 \
  --modes direct_goal offline_waypoints \
  --icode-checkpoint /path/to/icode/best.pt

.venv/bin/python experiments/plot_waypoint_prior_diagnostic.py \
  /tmp/waypoint_diagnostic \
  configs/research/mujoco_lab_complex.yaml \
  configs/research/mujoco_narrow_corridor.yaml \
  configs/research/mujoco_u_trap_long_board.yaml
```

Raw trajectory CSV, config snapshots, summaries, and figures from this smoke run
are archived under
`results/research_platform/waypoint_prior_diagnostic_smoke_20260712/` (ignored by
Git as experiment output).
