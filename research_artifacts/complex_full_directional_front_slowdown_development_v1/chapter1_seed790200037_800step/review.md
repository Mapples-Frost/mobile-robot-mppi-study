# Directional front-slowdown 800-step review

## Frozen question

Was the preceding 400-step result merely truncated during a successful recovery,
or does the identical controller still fail to make sustained route progress
over a longer fresh episode?

## Outcome

- Seed: `790200037`
- Horizon: 800 control steps
- Collision: no
- Boundary violations: 0
- Success: no (`max_steps`)
- Final goal distance: `7.725632 m`
- Trajectory length: `13.376809 m`
- Minimum clearance: `0.165357 m`
- Maximum path progress: `0.692660 m`
- Stuck steps: 76
- Safety interventions: 2
- Static-A* replans: 19

## Interpretation

The longer horizon rejects the truncation hypothesis. The robot travelled more
than 13 m safely, but looped near the entrance and ended farther from the goal.
The directional Safety repair remains mechanistically successful: almost all
planner commands now pass through without intervention. Safety is no longer the
dominant blocker.

The remaining failure is in the existing soft-reference lifecycle. Of 19 A*
replans, 15 were triggered by stagnation. Several occurred with only
`0.004-0.12 m` cross-track error, where the robot was still on the current
static reference. Because the map is frozen, replanning from nearly the same
path after a short no-progress interval adds no new static information, resets
reference progress, costs roughly 50-58 s on this CPU, and repeatedly changes
the local target during dynamic waiting.

## Next falsifiable repair

For these frozen-static complex maps, disable stagnation-only A* replans while
retaining deviation-triggered replans. A value of zero for the existing
`static_astar_replan_stagnation_steps` setting will mean "disabled"; positive
values retain current behavior for other experiments. This changes neither A*
nor the paper method and applies identically to all three maps.
