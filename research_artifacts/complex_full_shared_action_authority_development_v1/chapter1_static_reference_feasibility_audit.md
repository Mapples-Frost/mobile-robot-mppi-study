# Chapter 1 static-reference feasibility audit

## Question

Is the frozen Chapter 1 manual reference physically feasible for the configured
robot footprint, and can the existing static-only A* recover a route from the
nominal start or the latest post-detour state?

## Method

- Map: `configs/research/mujoco_irregular_spiral_three_dynamic_v1.yaml`
- Frozen static obstacles: 42
- Robot footprint radius: `0.25 m`
- Every one of the 28 manual-reference segments was sampled at no more than
  `0.01 m` spacing.
- Clearance used the same segment/box/cylinder geometry implementation as
  `mobile_robot_mppi.planning.static_astar._point_clearance`.
- Static-only A* was tested at `0.10 m` resolution from:
  - nominal start `(-6.35, -3.20)`;
  - latest post-detour state `(-6.2786, -3.8150)`.
- Clearance margins tested: `0.08 m`, `0.04 m`, and `0.00 m`.

## Result

The frozen manual reference is not footprint-feasible. Its minimum clearance is
`-0.063685 m`, meaning the robot footprint intersects frozen static geometry.
Nine reference segments have negative sampled clearance. The worst segment is
from `(2.10, 0.35)` to `(1.45, 0.85)`, with the minimum near
`(1.6928, 0.6633)`.

Other negative-clearance examples include:

- segment 18: `-0.030900 m`;
- segment 6: `-0.030288 m`;
- segment 10: `-0.029351 m`;
- segments 4 and 5: `-0.028772 m`;
- segment 17: `-0.020703 m`;
- segment 3: `-0.014677 m`;
- segment 22: `-0.010000 m`;
- segment 13: `-0.003629 m`.

Static-only A* found no route from either tested start at any of the three
clearance margins, including zero margin.

## Conclusion

The Chapter 1 failure cannot be attributed solely to conservatism or local
dynamic-avoidance tuning. The experiment input itself violates its stated
static-feasibility contract: the reference intersects static geometry and the
goal is topologically unreachable under the current frozen obstacle layout for
the configured footprint.

The next change must repair and version the Chapter 1 scene/reference geometry,
then re-run this feasibility audit before any controller experiment. Changing
A* margins, MPPI weights, emergency thresholds, or speed would be
scientifically uninterpretable while this input defect remains.
