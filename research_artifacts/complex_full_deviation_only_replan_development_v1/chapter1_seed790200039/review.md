# Deviation-only static-replan smoke review

## Frozen question

Does removing redundant stagnation-only A* refreshes prevent entrance looping
while retaining deviation-triggered recovery?

## Outcome

- Seed: `790200039`
- Horizon: 400 control steps
- Collision: no
- Boundary violations: 0
- Success: no (`max_steps`)
- Final goal distance: `7.429528 m`
- Trajectory length: `5.716655 m`
- Minimum clearance: `0.180039 m`
- Maximum path progress: `0.403541 m`
- Stuck steps: 103
- Safety interventions: 45
- Static-A* replans: 2, both caused by deviation above `0.70 m`

## Interpretation

The targeted lifecycle effect is confirmed: replans fell from 8-10 per
400-step smoke to 2 and no stagnation reset occurred. Compute spikes and
reference churn were reduced, with no safety regression.

The navigation outcome did not improve on this fresh seed. The robot still
oscillated around the entrance conflict and made less route progress than the
preceding 400-step seed. This is the first non-improving behavioral replicate
for the deviation-only mechanism; one seed is insufficient to distinguish a
mechanism failure from the intentionally randomized dynamic-obstacle phase.

## Next step

Run a second fresh 400-step seed with the identical committed controller. Do
not tune parameters between replicates. If the same lack of route progress is
reproduced, count it as a second non-improvement and perform at most one final
replicate before closing this mechanism family.
