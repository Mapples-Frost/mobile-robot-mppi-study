# Chapter 3 map R2 vetted-lattice development review

## Frozen run

- Scene: `mujoco_scattered_clutter_three_loop_v1`
- Seed: `783100001`
- Controller budget: 600 MPPI rollouts per decision, horizon 36, dt 0.1 s
- Maximum episode length: 2400 steps
- Output: `chapter3_map_r2_vetted_lattice_seed_783100001`
- Scope: opened development evidence; not confirmatory evidence

## Result

- Status: **fail**
- Termination: collision at step 2116
- Trajectory length: 19.561 m
- Final goal distance: approximately 4.72 m
- Minimum physical clearance: 0.0552 m
- Minimum dynamic-obstacle centre distance: 0.4208 m
- Stuck steps: 1263

## Mechanism audit

The causal perception and probability pipeline was active at the terminal
conflict:

- three dynamic tracks were present before the collision;
- the final decisions contained two or three probabilistic forecasts;
- probabilistic hard violations and temporal emergency triggers were active;
- temporal emergency logic triggered on 389 decisions;
- six emergency candidates were injected when triggered;
- a forecast-vetted emergency candidate was selected on 29 decisions;
- the exact frozen static-map cost was enabled throughout.

The failure therefore is not evidence that the obstacle was simply unseen.
The bounded local controller repeatedly alternated between a forward
risk-equivalent candidate, reactive reverse, and near-body hard stop. It did
not reserve a collision-free space-time corridor long enough to pass the
moving obstacle. The final five decisions changed between forward and reverse
commands while the dynamic obstacle closed from about 0.66 m to 0.42 m
centre distance.

## Decision

Reject and close the local `fixed-arc / emergency-candidate / reactive-escape`
mechanism family. Do not continue threshold tuning on this family.

Any further attempt on this scene must be a mechanism-level change that plans
and commits to a short space-time detour around forecast occupancy, while
retaining:

- strict static/dynamic measurement separation;
- exact static-map collision checking;
- causal online dynamic forecasts;
- 600 total MPPI rollouts per decision;
- safety-layer hard-stop authority as the final fail-safe.

The run is permanently classified as development evidence and must never be
reused as a fresh held-out or confirmatory sample.
