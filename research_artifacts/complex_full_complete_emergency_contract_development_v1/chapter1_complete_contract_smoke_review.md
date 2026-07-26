# Chapter 1 complete emergency-contract smoke review

## Frozen run

- Scene: `chapter1`
- Seed: `790200021`
- Deliberately truncated horizon: 350 of the map's configured 1500 steps
- Result: no collision, max-steps termination
- Actual minimum clearance: 0.2808 m
- Trajectory length: 5.0202 m
- Stuck steps: 60
- Spin steps: 4

## Targeted mechanism result

The complete shared contract fixed the short-fragment behavior seen in seed
`790200019`:

| Metric | Partial contract | Complete contract |
| --- | ---: | ---: |
| Trajectory length | 1.55 m | 5.02 m |
| Stuck steps | 233 | 60 |
| Spin steps | 119 | 4 |
| Actual minimum clearance | 0.1025 m | 0.2808 m |

The existing counterflow mechanism was active for 141 steps and emergency
candidates were selected for 106 steps. The last portion of the run had cleared
the immediate conflict and resumed approximately 0.30 m/s forward motion.

## Interpretation boundary

This was a mechanism smoke test, not an episode-level pass. Chapter 1 is a long
spiral map configured for 1500 steps; a 350-step run cannot reasonably establish
goal completion. The robot also made a large safe detour, so Euclidean goal
distance at the truncation point is not a valid completion verdict.

No additional parameter change is justified from this truncated run. The next
test must use a fresh seed and the map's full 1500-step horizon with the exact
same shared controller configuration.
