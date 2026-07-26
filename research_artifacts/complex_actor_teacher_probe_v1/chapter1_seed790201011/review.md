# Final shared teacher-scaling probe: Chapter 1

## Outcome

- Seed: `790201011`
- Training-only teacher: standard MPPI, 2,400 samples, 36-step frozen horizon
- Result: collision at step 114
- Maximum path progress: `0.4043 m`
- Net displacement: `0.5223 m`
- Trajectory length: `0.9052 m`
- Proposed velocity: 37 forward, 17 reverse, 60 near-zero steps
- Counterflow escape active: 98/114 steps
- Safety interventions: 30

At the collision cycle, the known-static clearance was positive
(`0.0767 m`) while the nearest dynamic-obstacle centre distance was
`0.4368 m`. Both the selected and stop trajectories had maximum predicted
collision probability `1.0`. Safety executed a planner-vetted
`dynamic_active_escape` command `(v=0.508 m/s, omega=-0.453 rad/s)`, but the
dynamic collision still occurred.

## Stopping decision

Doubling teacher samples did not recover Chapter 1 and introduced a safety
regression. The frozen protocol permits no further sample escalation. This
teacher-scaling direction is closed and the episode is retained unchanged.

No Actor training is permitted from this family: Chapters 2 and 3 provide
usable motion demonstrations, but Chapter 1 has neither a safe progress-making
teacher nor a valid target distribution.
