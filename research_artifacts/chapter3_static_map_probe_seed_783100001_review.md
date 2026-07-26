# Chapter 3 exact-static-map probe — safe but time-censored

- Seed: `783100001`
- Episodes: 1 development episode, 1200 control steps (120 s)
- Intervention: exact frozen static geometry in the MPPI rollout cost; no hard
  path corridor
- Result: no collision, but timeout before the goal
- Final pose: `(-2.034, 1.114)`
- Final goal distance: 8.536 m
- Trajectory length: 9.482 m
- Minimum measured clearance: 0.187 m
- Stuck steps: 765 / 1200
- Probabilistic `stop_is_safest_candidate` decisions: 79
- Planner P95: 90.53 ms

Interpretation: the exact static-map mechanism fixes the prior static-corner
collision and preserves substantially more mobility than the rejected hard
corridor. The episode is censored by the 120 s limit on a larger map: it is
moving forward at 0.64 m/s at the final step rather than trapped at a terminal
deadlock. The next frozen probe therefore retains the mechanism and extends
only the scene time budget to 240 s.

The raw result, trajectory, resolved configuration, provenance, stdout and
stderr are retained without rerunning or overwriting.
