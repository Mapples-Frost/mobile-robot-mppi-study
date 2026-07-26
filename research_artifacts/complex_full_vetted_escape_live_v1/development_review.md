# Vetted-escape mixed-scene development review

Status: mechanism smoke completed; outcome window intentionally truncated and
therefore not a map-success trial.

## Chapter 1 / seed 790200011

- Window: 180 steps (18.0 simulated seconds)
- Collision: false
- Success: false (`max_steps`)
- Actual minimum clearance: 0.32682884289187564 m
- Known-static weighted-update infeasible fraction: 0.0
- Trajectory length: 0.13609051219865573 m
- Stuck steps: 163
- Safety interventions: 131

The previous context-free Safety reverse and static collision disappeared.
The planner remained statically feasible throughout this window. The remaining
delay was dominated by 114 `front_obstacle_slow` decisions while the first
moving carrier occupied the entrance conflict, followed by renewed forward
motion near the end of the deliberately short window.

Decision: preserve this evidence without treating it as a pass. Continue with a
longer headless chapter-1 development run before modifying another mechanism.

