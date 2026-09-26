# Complex Static-Scene Gate

## Purpose

This gate asks whether the prediction improvements from residual learning are
sufficient for local MPPI to solve longer, topologically constrained static
navigation scenes. RL and dynamic obstacles remain disabled.

All scenes inherit the same strong-baseline task, robot, sensors, planner costs,
sampling settings and safety thresholds. Only obstacle geometry changes.

## Geometry quality gate

An offline inflated-grid audit now checks that start and goal are collision-free
and that a path exists for the 0.25 m robot plus a 0.03 m margin. This audit is
never used as planner input.

The original narrow-corridor configuration placed a vertical board through the
inflated goal region. It was corrected before controller comparison.

| Scene | Start clearance | Goal clearance | Grid path | Approx. shortest path |
| --- | ---: | ---: | --- | ---: |
| clean single obstacle | 1.591 m | 1.591 m | yes | 4.711 m |
| lab complex | 0.659 m | 0.297 m | yes | 5.121 m |
| narrow corridor | 0.565 m | 0.090 m | yes | 5.192 m |
| U-trap long board | 0.558 m | 0.121 m | yes | 5.697 m |

## Executed smoke tests (2026-07-12)

First pass: two matched seeds, 100 samples and 120 control steps. No method
reached the goal in any complex scene. There were no collisions, but all methods
showed heavy safety intervention and low progress.

To rule out a short run and inadequate sampling, a second diagnostic used one
matched seed, 400 samples and 360 steps for nominal and ICODE:

| Scene | Method | Success | Collision | Final distance | Safety interventions | Stuck steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lab complex | nominal | no | no | 3.566 m | 276 | 266 |
| lab complex | ICODE | no | no | 3.848 m | 306 | 289 |
| narrow corridor | nominal | no | no | 3.178 m | 307 | 289 |
| narrow corridor | ICODE | no | no | 3.234 m | 317 | 259 |
| U-trap | nominal | no | no | 3.760 m | 271 | 281 |
| U-trap | ICODE | no | no | 3.099 m | 262 | 281 |

The dominant arbitration reasons were `front_obstacle_slow` and `hard_stop`.
All scenes are geometrically feasible, so this gate fails at the local planning
and candidate-trajectory level rather than at dataset integrity or geometric
validity.

## Interpretation

ICODE corrects rollout dynamics; it does not provide global route topology. A
more accurate local rollout cannot by itself decide whether to pass left or
right of a long board, leave a U-shaped trap, or commit to a temporarily
goal-diverging maneuver.

This negative result supports the need to isolate a proposal/exploration layer:

```text
residual model -> improve physical rollout accuracy
sampling prior -> propose topologically useful candidate sequences
MPPI -> evaluate and refine candidates
scan_guard -> retain final safety authority
```

Before training RL, the next diagnostic should supply a frozen waypoint or
offline route prior. If local MPPI succeeds with that prior, the bottleneck is
candidate exploration and the later RL-prior hypothesis is justified. If it
still fails, perception/safety thresholds or local obstacle costs must be fixed
first.

## Evidence boundary

This gate is intentionally recorded as failed. No success claim should be made
from the complex scenes, and ICODE must not be blamed or credited for a planner
that lacks a useful route proposal. The archived raw trajectories and metrics
are stored under `results/research_platform/static_scene_smoke_20260712`.
