# Chapter 1 dynamic-collision action-contract review

## Frozen run

- Scene: `chapter1`
- Seed: `790200027`
- Result: dynamic collision at step 199
- Minimum dynamic center distance: 0.4602 m
- Actual minimum clearance: -0.0005 m
- Maximum route progress before contact: 1.6939 m
- Static weighted-update infeasible fraction: 0.0

The forecast-corroboration repair did improve mobility: the robot passed the
earlier entrance-loop region and made four times the previous full-horizon route
progress. The run nevertheless fails the safety contract.

## Causal trace

The Change-Aware forecast correctly raised stop risk to one. The preferred
escape direction pointed away from the obstacle, and the existing emergency
lattice selected forward motion because moving east was initially lower risk
than stopping or reversing. The robot used its maximum configured translation
of 0.35 m/s, but the crossing obstacle closed faster than the robot could clear
its path. Reverse motion began only after every candidate was already at
probability one, and contact followed.

## Experiment-interface defect

The frozen Chapter 1 feasibility contract explicitly assumes a maximum robot
speed of 0.70 m/s. Its map configuration declares the same 0.70 m/s action
limit. The complex Full adapter currently discards every map action envelope
and inherits the single-obstacle paper-v4 limit of 0.35 m/s.

Thus this run evaluated a controller with only half the translation authority
used to certify the conflict geometry. The failure cannot distinguish a
planning defect from an infeasible actuator contract.

## Next falsifiable mechanism

Define one shared complex-experiment action envelope of 0.70 m/s and 0.95 rad/s
in the common override, and apply it identically to all three maps and every
future comparison arm. Keep reverse limit, acceleration limits, plant,
prediction, costs, rollout budget, Safety, and all algorithm modules unchanged.

This is an experimental interface correction, not a new planner or a
map-specific rule. A fresh smoke run must clear the same type of crossing
without collision and without weakening the probability or static-safety
contracts.
