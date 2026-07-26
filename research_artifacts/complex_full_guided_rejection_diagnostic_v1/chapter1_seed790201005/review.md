# Guided-candidate rejection decomposition

## Frozen episode

- Map: Chapter 1 irregular spiral
- Fresh seed: `790201005`
- Horizon: 400 control steps
- Result: safe `max_steps` (`collision=false`, 11 Safety interventions)
- Final goal distance: `7.1930 m`
- Stuck steps: 39

This run used commit `82b8f7c`. The added diagnostics do not change candidate
generation, feasibility, cost, MPPI selection, or final Safety behavior.

## Rejection decomposition

The Actor-guided batch was allocated only on the first control cycle:

- guided opportunities: 178 across two MPPI iterations;
- boundary-feasible fraction: `1.000`;
- known-static-feasible fraction: `1.000`;
- probabilistic-dynamic-risk-feasible fraction: `0.000`;
- joint-feasible fraction: `0.000`;
- mean first dynamic-risk violation step: `9.371`;
- guided elites: 0;
- Gaussian elites over the episode: 32,456.

The initial Actor action was `(v=0.06 m/s, omega=0.127 rad/s)`, while the
baseline proposal began at `(v=0.550 m/s, omega=-0.121 rad/s)`. A valid
Change-Aware forecast was present, and every guided trajectory crossed the
frozen hard probability threshold within the 3.6-second planning horizon.

After this rejected batch, residual-support confidence fell to zero and HSS
set the next guided fraction to zero. Mean HSS authority over the episode was
`0.0225`, so the planner used `97.75%` fallback authority.

## Mechanism conclusion

The previous ambiguity is resolved for this episode: Actor candidates were not
lost to the static map or A* corridor. They were rejected by the existing
probabilistic dynamic-risk filter before MPPI weighting. Relaxing HSS would
therefore expose unsafe guided proposals and is not an admissible repair.

The next shared mechanism must address Actor domain coverage for mixed
static/three-dynamic observations while retaining the same Actor architecture,
ICODE rollout, probability thresholds, MPPI budget, and Safety contract.
Threshold weakening, A* retuning, or forced Actor authority are ruled out.
