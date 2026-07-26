# Chapter 3 scattered-clutter + three-loop MuJoCo scene v1

## Role

This is the third complex map. It avoids the visible route structures used by
the multi-gate and spiral maps. The robot starts in the lower-left and receives
only a point goal in the upper-right.

## Static design

An irregular ten-segment boundary contains 24 rotated boxes with varied
dimensions, heights and colors. The boxes are spread across the full
arena with deliberately larger gaps; they are scattered individually and in
small clusters; they do not form a corridor, ring or wall-guided route. The
direct start-to-goal line is blocked, but a footprint-inflated occupancy audit
confirms that at least one free route exists.

No polyline or waypoint reference is supplied to the controller. The preview
also hides route and conflict overlays. Only the start and goal markers remain.

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

The causal tracker maintains up to eight scan-cluster candidates but publishes
a probability forecast only after six consecutive world-frame observations
confirm at least 0.45 m/s motion. Unconfirmed static clusters remain in the
ordinary LaserScan obstacle layer. During this short confirmation interval,
MPPI uses that static layer and the scan guard instead of inventing a dynamic
forecast for walls or boxes.

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
