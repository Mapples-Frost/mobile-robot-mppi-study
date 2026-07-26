# Complex Actor teacher upper-bound probe: Chapter 2

## Outcome

- Seed: `790202007`
- Training-only teacher: standard MPPI, 1,200 samples, 36-step frozen horizon
- Result: collision-free `max_steps` after 200 steps
- Maximum path progress: `4.6455 m` (passes the frozen `0.80 m` gate)
- Net displacement: `3.6670 m`
- Trajectory length: `4.2692 m`
- Proposed velocity: 141 forward, 0 reverse, 59 near-zero steps
- Counterflow escape active: 16/200 steps
- Safety interventions: 36

## Decision

Unlike Chapter 1, the unchanged higher-sample teacher produced decisive,
collision-free forward navigation on Chapter 2. This shows that a causal
risk-aware teacher is viable in at least one complex map, but it does not yet
establish a shared training source.

The frozen teacher must now be checked on Chapter 3. Actor dataset collection
remains blocked until the cross-map result is known; no checkpoint training is
authorized from only the successful map.
