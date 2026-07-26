# Chapter 1 partial emergency-contract review

## Frozen run

- Scene: `chapter1`
- Seed: `790200019`
- Horizon: 350 control steps
- Result: no collision, max-steps termination
- Actual minimum clearance: 0.1025 m
- Final goal distance: 7.1187 m
- Trajectory length: 1.5503 m
- Stuck steps: 233

## What improved

The previously unreachable escape lattice became active:

- emergency candidates selected: 203 steps
- temporal emergency triggered: 217 steps
- near-distance trigger: 213 steps
- minimum dynamic center distance: 0.6463 m

The fresh run did not repeat the dynamic collision.

## Why this is not a pass

The shared override restored only the three activation thresholds while leaving
the remainder of the already-existing complex-scene emergency contract at the
paper-v4 defaults. In particular:

- emergency prefix remained 3 rather than 12 steps;
- intent hold remained 0 rather than 12 steps;
- counterflow escape remained disabled;
- the active-avoidance-motion hard action and progress tiebreak were absent.

As a result, the planner repeatedly selected short turn/stop/reverse fragments.
The robot spent 203 steps under temporal escape, accumulated 119 spin steps,
and did not make net route progress. This is safe but excessively reactive.

## Next falsifiable change

Restore the complete, map-independent emergency contract that was already
identical in all three complex-scene preview configurations. This is one
configuration-plumbing repair of an existing MPPI mechanism, not a new
planner:

- 12-step candidate prefix and intent hold;
- 2.8 s rearm TTC with a 3-step clear streak;
- predictor-relative counterflow escape;
- `active_avoidance_motion` with the existing progress tiebreak.

The next fresh seed must retain collision-free behavior while reducing repeated
spin/stop fragments and producing positive route progress.
