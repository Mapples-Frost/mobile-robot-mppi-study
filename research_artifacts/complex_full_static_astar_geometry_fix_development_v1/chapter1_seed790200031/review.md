# Static-A* geometry-fix smoke review

## Frozen question

Does correcting the existing static-A* segment-width contract restore useful
soft-reference recovery on a fresh Chapter 1 seed without changing the map,
paper method, or shared controller parameters?

## Outcome

- Seed: `790200031`
- Horizon: 400 control steps
- Collision: no
- Boundary violations: 0
- Success: no (`max_steps`)
- Final goal distance: `7.332867 m`
- Trajectory length: `4.843498 m`
- Minimum clearance: `0.171522 m`
- Maximum path progress: `0.721239 m` (`2.2420%`)
- Stuck steps: 147
- Static-A* replans: 10, all successful
- Safety interventions: 176, all `front_obstacle_slow`
- Maximum planner compute time: `58.113 s`

## Mechanism result

The geometry repair worked as intended: the prior false "no route" condition is
gone, including from dynamically displaced poses. This smoke remained
collision-free and boundary-safe, and maximum route progress increased from
`0.370421 m` in the preceding shared-action-authority smoke to `0.721239 m`.

The episode still failed. Replanning was repeatedly triggered by stagnation
roughly every 31 steps, making the current CPU implementation far too slow.
More importantly, the final stall exposes a separate Safety handoff defect:
while a static obstacle was in the front sector and predicted dynamic risk was
very low, MPPI proposed `v=-0.35 m/s` to back away, but the generic
`should_slow_down` branch mapped every negative command to zero through
`max(0, v)`. The robot therefore could neither execute the planner-vetted
reverse escape nor create clearance for a new forward route.

## Next falsifiable repair

For an ordinary front slowdown (not an emergency stop or near-body hard stop),
scale only positive velocity and preserve a planner-vetted negative command.
This is a directional correction to the existing Safety contract, not a new
planner or state machine. It must be regression-tested before a fresh seed.
