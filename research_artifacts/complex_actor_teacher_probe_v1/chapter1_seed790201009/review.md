# Complex Actor teacher upper-bound probe: Chapter 1

## Outcome

- Seed: `790201009`
- Training-only teacher: standard MPPI, 1,200 samples, 36-step frozen horizon
- Result: collision-free `max_steps` after 200 steps
- Final goal distance: `6.9971 m`
- Trajectory length: `1.7617 m`
- Net displacement: `0.3866 m`
- Maximum path progress: `0.4674 m` (below the frozen `0.80 m` gate)
- Stuck steps: 103
- Safety interventions: 17
- Proposed velocity: 44 forward, 55 reverse, 101 near-zero steps
- Counterflow escape active: 183/200 steps

## Decision

Increasing teacher sampling capacity while preserving the Change-Aware,
probability-risk, static-map, ICODE and Safety contracts did not create an
active-passage upper bound on Chapter 1. The teacher itself remained dominated
by counterflow/near-zero behavior, so its executed actions are not suitable
Actor targets.

This is the first non-improvement in the frozen teacher-upper-bound family.
The same unchanged teacher will be checked once on Chapter 2 and once on
Chapter 3. If both also fail to make the frozen `0.80 m` path-progress gate,
this family stops; no Actor retraining will be performed from a failed teacher.
