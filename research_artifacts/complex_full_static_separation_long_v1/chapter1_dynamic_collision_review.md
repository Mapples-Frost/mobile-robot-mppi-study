# Chapter 1 dynamic-collision review

## Frozen run

- Scene: `chapter1`
- Seed: `790200017`
- Shared controller: `complex_mixed_full_method_common.yaml`
- Result: collision at step 304
- Final goal distance: 6.7733 m
- Minimum clearance: -0.0200 m
- Minimum dynamic-obstacle center distance: 0.4409 m
- Known-static weighted-update infeasible fraction: 0.0

The contact was with a dynamic obstacle, not a static wall. The robot remained
statically feasible throughout the run.

## Causal trace

The robot made useful forward progress before the collision. As the moving
obstacle closed, the forecasted maximum collision probability rose from near
zero to one. From step 286 onward the planner selected zero translation while
the obstacle continued closing from approximately 0.87 m to 0.44 m.

Stopping was itself predicted unsafe: the stop-trajectory maximum probability
rose to one. Nevertheless, every decision reported
`probabilistic_obstacle_emergency_candidate_count = 0`; no temporal, near-range,
or critical-range emergency trigger fired.

## Root cause

The emergency-candidate feature was enabled, but all of its activation
thresholds resolved to zero:

- `probabilistic_obstacle_emergency_candidate_trigger_ttc_s = 0`
- `probabilistic_obstacle_emergency_candidate_trigger_distance_m = 0`
- `probabilistic_obstacle_emergency_candidate_critical_distance_m = 0`

Therefore the existing predictor-aware escape lattice was unreachable. This is
a configuration-contract defect, not evidence that a new navigation algorithm
is needed.

## Falsifiable next change

Expose the already-used, map-independent complex-scene thresholds through the
single shared common override:

- TTC trigger: 2.20 s
- near-distance trigger: 0.85 m
- critical-distance trigger: 0.42 m

No Actor, ICODE, predictor, risk threshold, MPPI budget, Safety boundary, A*
algorithm, or map-specific controller logic will be changed. The next run must
use a fresh development seed and must show non-zero emergency candidate
coverage under a closing conflict without a static collision.
