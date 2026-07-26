# Deviation-only static-replan smoke review, replicate 2

## Outcome

- Seed: `790201001`
- Horizon: 400 control steps
- Collision: no
- Boundary violations: 0
- Success: no (`max_steps`)
- Final goal distance: `7.477352 m`
- Trajectory length: `5.921101 m`
- Minimum clearance: `0.240913 m`
- Maximum path progress: `0.743707 m`
- Stuck steps: 70
- Safety interventions: 65
- Static-A* replans: 2

## Interpretation

The lifecycle mechanism reproduced: no stagnation replan occurred and only two
material-deviation replans were used. The run remained safe and was less stuck
than replicate 1, but it still failed to leave the entrance region and ended
farther from the goal.

This is the second consecutive non-improving navigation result for the
deviation-only family. Per the frozen stopping rule, run one final fresh
400-step replicate without parameter changes. If it also fails to create
sustained route progress, close this family and perform a cross-layer global
review rather than continuing to tune A* triggers.
