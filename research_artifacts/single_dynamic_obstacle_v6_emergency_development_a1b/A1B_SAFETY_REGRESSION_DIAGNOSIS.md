# Single-dynamic-obstacle v6 A1b diagnosis

## Decision

A1b failed its frozen engineering gate and is retained as a failed development
attempt. It is not evidence of a treatment effect and does not authorize formal
execution.

The paired eight-seed panel produced:

- safe success: 2/8 (V4) versus 2/8 (A1b);
- collisions: 4/8 (V4) versus 4/8 (A1b);
- one prevented collision (`id/750100038`);
- one safe non-completion converted to success (`ood/750200036`);
- one new collision and lost V4 success (`ood/750200020`).

This is the first safety regression in the
`causal_temporal_emergency_candidates` mechanism family. A second safety
regression closes the family under the frozen stopping rule.

## Global interpretation

The mechanism is not uniformly wrong: it exercised the intended temporal
emergency path, prevented one collision, and completed one previously stalled
episode. The gate failed because the safety arbiter accepted a same-cycle
planner command without checking whether its longitudinal direction conflicted
with the current scan-derived escape geometry.

This is an arbitration incompatibility, not another post-conflict release
parameter problem. Continuing to tune recovery thresholds would not address the
observed regression.

## Regression trace: `ood/750200020`

The frozen V4 control escaped by reversing and later reached the goal:

- success: true;
- collision: false;
- minimum clearance: 0.0013825455 m;
- episode steps: 392.

A1b collided at step 117:

- success: false;
- collision: true;
- minimum clearance: -0.0221554920 m;
- emergency candidate selected for 12 steps;
- temporal candidate vetted for 12 steps;
- Pareto forward commitment never activated.

At approximately 10.4 s, the online temporal scan reported closing speed near
0.926 m/s, time-to-collision near 1.435 s, and clearance near 1.09 m. The
scan-derived reactive geometry required a reverse escape. V4 selected reverse
motion, while A1b's `dynamic_escape_use_vetted_planner_control` branch accepted
the planner's forward command (about +0.23 m/s with clockwise steering). That
forward motion consumed the available margin. Hard risk became active around
10.8 s, after which stopping/rotation was too late to avoid collision.

## A2 falsifiable correction

A2 may change exactly one arbitration contract:

> When fresh online escape geometry requires reverse motion, a positive
> longitudinal planner command cannot replace the reactive reverse escape.

Zero or reverse planner commands remain eligible, so the change does not remove
the A1b paths that produced the two favorable paired conversions. All predictors,
risk thresholds, checkpoints, rollout budget, obstacle generation, and recovery
parameters remain frozen.

A2 must achieve all of the following on the same opened paired panel:

- safe-success gain at least +1;
- at least one prevented collision;
- collision count reduction at least 1;
- zero new paired collisions;
- zero lost V4 successes;
- mean minimum clearance not degraded;
- exactly 600 rollouts per decision.

If A2 introduces a new paired collision, the mechanism family is stopped
immediately for reaching two safety regressions.
