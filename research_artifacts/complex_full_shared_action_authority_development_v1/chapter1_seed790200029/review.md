# Shared action-authority smoke review

## Frozen question

Does applying the complex-scene action envelope (`v <= 0.70 m/s`) consistently
through the adapter remove the previously observed dynamic-collision failure
without changing the paper method or adding map-specific logic?

## Outcome

- Seed: `790200029`
- Horizon: 400 control steps
- Collision: no
- Success: no (`max_steps`)
- Final goal distance: `7.346748 m`
- Trajectory length: `5.456401 m`
- Minimum dynamic-obstacle centre distance: `0.586196 m`
- Minimum clearance: `0.192583 m`
- Boundary-safe success: no; 15 boundary-violation steps
- Stuck steps: 131
- Maximum path progress: `0.370421 m` (`1.0081%`)
- Emergency candidate selected: 153 steps
- Forecast-corroborated emergency: 109 steps
- Safety interventions: 177 steps

## Interpretation

The shared action-authority change removed the immediate collision mode seen in
the preceding smoke and allowed substantial physical motion, so the action
envelope mismatch was a real defect. It did not solve navigation. The robot
detoured and looped near the entrance, then stalled close to static geometry
without recovering meaningful global-route progress. This run therefore does
not pass the complex-scene gate.

The next falsifiable mechanism question is the existing static soft-reference
interface after a dynamic detour. No additional speed tuning is justified by
this result. Before changing controller behavior, the frozen map geometry and
manual reference must be checked for physical/topological feasibility under the
same robot footprint, and the existing static-only A* failure must be localized
to either scene geometry or implementation.
