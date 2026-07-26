# Complex Actor teacher upper-bound probe: Chapter 3

## Outcome

- Seed: `790203007`
- Training-only teacher: standard MPPI, 1,200 samples, 36-step frozen horizon
- Result: collision-free `max_steps` after 200 steps
- Net displacement: `4.9888 m`
- Trajectory length: `5.6535 m`
- Proposed velocity: 178 forward, 1 reverse, 21 near-zero steps
- Counterflow escape active: 37/200 steps
- Safety interventions: 37

Chapter 3 uses a point-goal task and therefore does not emit polyline path
progress. Net displacement is used for the preregistered motion gate and
comfortably exceeds `0.80 m`.

## Cross-map decision

The identical teacher passed the motion/safety gate on Chapters 2 and 3 but
failed on Chapter 1. Therefore a causal high-sample teacher is possible within
the existing algorithm family, but the present 1,200-sample budget is not a
shared three-map training source.

Actor training remains blocked. One shared compute-scale amendment may test
Chapter 1 at 2,400 samples while retaining the 36-step forecast horizon and all
risk, static-map, ICODE and Safety contracts. If Chapter 1 remains below the
motion gate, this teacher-scaling direction is closed rather than escalating
indefinitely.
