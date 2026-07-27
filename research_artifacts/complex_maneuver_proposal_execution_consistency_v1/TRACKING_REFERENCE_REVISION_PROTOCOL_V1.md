# Tracking/reference revision synchronization protocol v1

## Scope

This is an evaluation-only repair. It does not modify the planner, Actor,
Risk, ICODE, MPPI costs, Safety, maps, or rollout budget.

The blocking failure occurred after static A* replaced the polyline route.
`PolylineReference` correctly restarted its own route progress, but
`TrackingEventMonitor` retained progress and route-relative event state from
the obsolete route. When the replacement was shorter, the stale progress was
outside the new route and the evaluation layer raised an exception.

## Single intervention

`PolylineReference` receives a monotonically increasing, read-only geometry
revision. A successful `replace_points` increments it once; ordinary `reset`
does not.

`TrackingEventMonitor` records the last revision it evaluated. Before each
update, a revision change resets route-relative state:

- progress;
- projected obstacle progress and next-obstacle index;
- recovery start time and recovery start progress.

World-coordinate, episode-level centre-crossing state remains intact.

Clamping old progress to the new route end is explicitly forbidden because it
would avoid the exception by creating a false measurement.

## Experimental control

The patch is applied identically to both sides of the equivalence audit:

- detached `af691f1` reference source plus the evaluation patch;
- current diagnostic source plus the identical evaluation patch.

This shared measurement repair is a nuisance-control intervention, not a
treatment factor. It cannot be interpreted as an Actor improvement.

## Gates

Implementation begins only after this protocol is committed. Unit tests must
cover shorter and longer replacement routes, failed replacements, unchanged
routes, obstacle reprojection, and preservation of world-coordinate episode
events.

Before Actor-on testing, Actor-off closed-loop behavior must remain exactly
equal on every non-timing common column under the concurrent-reference
contract, with only 0 or 600 rollouts.

Actor-on margin zero is then run as a paired concurrent comparison. Both arms
must complete, match exactly on all non-timing common columns, insert all three
Actor heads, add zero rollouts, and preserve the 0/600 budget.

Any failure stops the workflow before margin 0.02. Paid-server and formal
execution remain unauthorized.
