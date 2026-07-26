# Chapter 3 scattered-clutter + three-loop MuJoCo scene v1

## Role

This is the third complex map. It avoids the visible route structures used by
the multi-gate and spiral maps. The robot starts in the lower-left and receives
a deterministic sparse A* reference to the upper-right.

## Static design

An irregular ten-segment boundary contains 24 rotated boxes with varied
dimensions, heights and colors. The boxes are spread across the full
arena with deliberately larger gaps; they are scattered individually and in
small clusters; they do not form a corridor, ring or wall-guided route. The
direct start-to-goal line is blocked, but a footprint-inflated occupancy audit
confirms that at least one free route exists.

The ten-point reference is computed from the frozen static occupancy map, not
from dynamic-obstacle truth, and is hidden from the viewer. It provides
map-level progress while MPPI chooses the local motion and reacts to dynamic
forecasts. The preview hides route and conflict overlays; only start and goal
markers remain.

The frozen static-map geometry is scored exactly in every MPPI rollout in
addition to the causal scan obstacle layer. This prevents unseen-corner cuts
without constraining the robot to a narrow route tube, so dynamic avoidance
retains its full lateral freedom.

The task horizon is 240 s (2400 control steps at 0.1 s). This larger map has a
substantially longer collision-free route than the prior crossing maps; the
controller horizon, 600-rollout decision budget and vehicle dynamics remain
unchanged.

When a causally observed moving obstacle has a short TTC, standard MPPI reserves
six slots from its unchanged 600-candidate budget for forward/reverse and
left/right escape sequences. Their ordering includes directions derived from
the causal measured/predicted obstacle velocity, combining lateral separation
with a counterflow component. Every sequence is evaluated by the same frozen
probabilistic collision-risk model and exact static-map cost. The final safety
layer may execute only the selected vetted candidate; critical danger still
retains reactive reverse when no candidate is certified.

## Dynamic design

The three moving obstacles use a new closed-waypoint-loop motion model:

- a five-segment lower-left patrol;
- a six-segment unequal-speed central orbit;
- a seven-segment upper-right patrol.

Every carrier continually changes heading and visits multiple non-collinear
segments. The path closes into a loop instead of reversing along one line.
Segment durations differ, so speed changes across the same patrol. Episode
reset independently perturbs phase, overall duration scale and every waypoint.
The sampled plant program and obstacle identity remain hidden from the
controller, which receives causal LaserScan observations only.

The full worst-case waypoint-jitter envelope is separated from frozen static
geometry by at least 0.102 m, above the 0.04 m contract floor. This prevents a
true moving-body return from becoming geometrically indistinguishable from a
known static surface.

Known static-map returns are removed before causal motion tracking using a
0.025 m surface tolerance, below the frozen 0.04 m dynamic-to-static clearance
floor so a nearby moving body is not absorbed into the static mask. The tracker
then maintains at most three residual scan-cluster candidates and publishes a
probability forecast only after three consecutive world-frame observations
confirm at least 0.15 m/s motion. Static returns remain in the ordinary
LaserScan obstacle layer. During this short confirmation interval, MPPI uses
that static layer and the scan guard instead of inventing a dynamic forecast
for walls or boxes.

The controller preview renders every published moving track separately:
track-coloured weighted-mean paths, four 95% confidence-ellipse horizons,
confirmed-track speed labels, and start/goal markers. It deliberately omits a
start-to-goal reference segment so the display does not imply a privileged
route.

Each nominal loop encloses more than 2.0 square metres. All waypoint-jitter
corner combinations remain at least 0.04 m from static geometry after applying
the conservative moving-obstacle envelope.

## Development discipline

Map geometry and motion programs are frozen before controller tuning. Initial
runs are development smoke tests only. A controller failure cannot be repaired
by moving a blocking box or changing a loop after viewing that outcome; such a
change creates a new map revision and checkpoint. Formal server execution
requires separate user confirmation.
