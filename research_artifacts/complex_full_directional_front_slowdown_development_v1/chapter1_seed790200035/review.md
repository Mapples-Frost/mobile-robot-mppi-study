# Directional front-slowdown smoke review

## Frozen question

Does preserving planner-vetted reverse motion during a non-emergency
front-sector slowdown remove the Safety-induced immobility without sacrificing
collision or boundary safety?

## Outcome

- Seed: `790200035`
- Horizon: 400 control steps
- Collision: no
- Boundary violations: 0
- Success: no (`max_steps`)
- Final goal distance: `7.430563 m`
- Trajectory length: `4.104069 m`
- Minimum clearance: `0.160186 m`
- Maximum path progress: `0.874808 m`
- Stuck steps: 82
- Safety interventions: 26
- Static-A* replans: 8

## Paired mechanism comparison with the preceding fresh smoke

The directional Safety correction improved every targeted mechanism metric
without a safety regression:

- Safety interventions: `176 -> 26`;
- stuck steps: `147 -> 82`;
- maximum path progress: `0.721239 -> 0.874808 m`;
- collision and boundary violations remained zero.

The episode still did not complete and spent much of the short horizon backing
out of the entrance-side conflict. In the final five steps, dynamic risk was
near zero, Safety no longer overrode the planner, and forward speed increased
from about `0.03` to `0.10 m/s` toward the current soft-reference target. This
means the 400-step smoke ends during recovery rather than demonstrating a new
terminal stall.

## Next falsifiable test

Do not add another controller mechanism yet. Run one fresh 800-step development
seed with the identical committed configuration to determine whether the
observed recovery matures into sustained route progress. This is still a
development probe, not a qualification result.
