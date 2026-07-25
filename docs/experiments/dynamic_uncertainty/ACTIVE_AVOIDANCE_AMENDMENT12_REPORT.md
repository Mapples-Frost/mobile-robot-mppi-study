# Active avoidance Amendment 12 report

Date: 2026-07-23  
Branch: `codex/change-aware-probabilistic-mppi`

## Question

Can the robot avoid a recurrent moving obstacle without treating every risk
event as a command to stop?

The complete episode is the independent unit.  Obstacle-process seed is the
blocking factor.  The three registered development seeds are paired between
risk-disabled and risk-enabled arms.  The V3 obstacle generator, Change-Aware
IMM, probability calculation, hard threshold, localization, route, and seeds
remain unchanged.

## Intervention

Amendment 12 replaces unconditional stopping with a layered active-avoidance
contract:

1. MPPI candidates that violate the existing hard probability threshold are
   filtered when feasible candidates exist.
2. If the weighted trajectory is unsafe, the lowest-cost safe candidate is
   selected.
3. If all sampled candidates are unsafe, the minimum-risk active candidate is
   compared with the zero-velocity trajectory; stopping is selected only when
   it is no more risky.
4. A simulator-truth-free LaserScan flow estimate provides a TTC trigger before
   the obstacle enters the body-stop envelope.
5. The tracked obstacle bearing determines the escape direction.  The robot
   drives forward when the away heading is in its front half-plane and reverses
   when it is in its rear half-plane.
6. A near-body scan stop may be bypassed only when the scan return matches the
   tracked dynamic obstacle and the selected active trajectory is predicted to
   be safer than stopping.

Forward and reverse emergency speeds are symmetrically bounded at 0.35 m/s.

## Complete paired development result

Artifact:
`research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment12`

| Seed | Risk disabled | Risk enabled | Enabled minimum clearance |
|---:|---|---|---:|
| 730100001 | goal reached | no collision, 1.20 m remaining at 40 s | 0.257 m |
| 730100003 | collision | no collision, 1.34 m remaining at 40 s | 0.126 m |
| 730100005 | goal reached | goal reached | 0.349 m |

The risk-enabled arm produced zero collisions in all three complete episodes.
The risk-disabled arm collided in one episode.  Median paired clearance
improved by 0.244 m.  Maximum enabled planner p95 compute time was 58.7 ms,
below the 150 ms gate.

Eight of nine original smoke gates passed.  The sole failure was completion
regression: 0.1043 against a 0.10 limit.  This is a 0.0043 absolute miss and
must not be rounded into a pass.

## Behavior audit

| Seed | active escape steps | reverse escape steps | MPPI fallback steps |
|---:|---:|---:|---:|
| 730100001 | 16 | 16 | 24 |
| 730100003 | 34 | 21 | 49 |
| 730100005 | 4 | 4 | 8 |

The intervention is therefore behaviorally active rather than a renamed stop.
The two incomplete enabled episodes travelled 8.4--8.9 m and ended 1.2--1.34 m
from the goal; their remaining issue is post-evasion route recovery and
efficiency, not collision or localization drift.

## Decision

Active avoidance passes the primary safety objective on the registered
development seeds: no risk-enabled collisions and no collision regressions.
It is not yet a final completion controller because the original completion
gate narrowly fails.  The next isolated intervention should address
post-evasion recovery without changing the probability predictor, collision
risk definition, or escape trigger.
