# Deviation-only static-replan smoke review, replicate 3

## Outcome

- Seed: `790201003`
- Horizon: 400 control steps
- Collision: no
- Boundary violations: 0
- Success: no (`max_steps`)
- Final goal distance: `7.328247 m`
- Trajectory length: `5.462331 m`
- Minimum clearance: `0.249232 m`
- Maximum path progress: `0.319625 m`
- Stuck steps: 87
- Safety interventions: 92
- Static-A* replans: 1

## Stopping decision

This is the third consecutive fresh seed in which deviation-only replanning
removed reference churn safely but did not produce sustained navigation beyond
the entrance. The frozen local-mechanism stopping threshold is reached.

No further A* trigger, cooldown, margin, resolution, or reference-weight tuning
is permitted for this family. The change may be retained as an implementation
efficiency correction, but it is not a navigation solution and must not be
reported as one.

The next action is a cross-layer review of perception, Change-Aware forecast,
RL/Gaussian candidate feasibility, MPPI selection, and Safety authority across
all three replicates. No new episode should be launched until that review
identifies a different falsifiable mechanism.
