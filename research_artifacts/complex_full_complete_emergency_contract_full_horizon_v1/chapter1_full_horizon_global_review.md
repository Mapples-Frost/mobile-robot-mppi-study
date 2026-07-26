# Chapter 1 full-horizon global mechanism review

## Frozen run

- Scene: `chapter1`
- Seed: `790200023`
- Horizon: the map's full 1500 steps
- Result: no collision, max-steps termination
- Actual minimum clearance: 0.1677 m
- Trajectory length: 18.2179 m
- Final goal distance: 7.1487 m
- Maximum route progress: 0.4348 m of a 36.7435 m reference

The robot remained in the entrance region:

- x range: [-6.528, -6.001] m
- y range: [-4.273, -2.475] m
- maximum path-progress ratio: 1.18%

## Family stop

The emergency-activation family has now produced three non-passing outcomes:

1. zero thresholds: the escape lattice was unreachable and a moving obstacle
   hit the stopped robot;
2. thresholds only: the lattice was active but short fragments caused
   turn/stop/reverse paralysis;
3. complete preview contract: safety and motion improved, but the robot looped
   around the entrance for the full task horizon.

No further threshold, prefix, hold, or counterflow-weight tuning is allowed
without a cross-layer diagnosis.

## Cross-layer diagnosis

Perception/tracking itself was available: 1440/1500 steps had at least one
probabilistic forecast. The decisive inconsistency is between the raw temporal
scan trigger and the predictor's probabilistic collision risk:

- emergency candidates selected: 503 steps;
- 317/503 selections occurred with predicted maximum collision probability
  below 0.05;
- counterflow escape applied: 631 steps;
- 362/631 counterflow steps occurred below 0.05 predicted risk;
- mean predicted maximum risk over the episode: 0.0609.

The temporal scan estimator intentionally sees the raw scan for Safety. In a
static maze, robot motion toward walls also produces shrinking raw ranges. The
planner's dynamic emergency context uses that raw scan-flow signal and can
therefore repeatedly activate a dynamic escape even when the Change-Aware
forecast says the candidate conflict probability is negligible. This explains
both the earlier “static treated as dynamic” symptom and the entrance loops.

Static geometry was not the collision cause, and further A* or route changes
would not repair this signal disagreement. The existing soft reference cannot
make progress while a false dynamic emergency repeatedly takes authority.

## Next falsifiable mechanism

Keep raw temporal scan flow available to Safety, but require the planner's
non-critical dynamic emergency trigger to be corroborated by the existing
probabilistic forecast. Critical near-contact remains fail-closed.

This is a perception/prediction contract repair:

- it adds no navigation layer or map-specific logic;
- it does not change Actor, ICODE, Change-Aware, MPPI budget, static A*, or
  Safety;
- it directly tests whether low-probability raw-scan emergencies caused the
  loops.

The next fresh smoke run must sharply reduce emergency selections below 0.05
predicted risk while preserving critical-distance avoidance and increasing
route progress.
