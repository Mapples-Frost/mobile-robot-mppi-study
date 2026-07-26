# Static/local-layer separation short-window review

Status: completed mechanism smoke; intentionally truncated and not a map-pass
trial.

## Chapter 1 / seed 790200015

- Window: 250 steps (25 simulated seconds)
- Collision: false
- Success: false (`max_steps`)
- Goal distance: 7.111 m to 6.889 m
- Trajectory length: 0.633 m
- Actual minimum clearance: 0.081 m
- Exact-static weighted-update infeasible fraction: 0

Compared with the pre-fix long-run prefix, the robot now made positive route
progress and the `front_soft_block` count fell to 1/250. During steps 150--199,
mean executed speed was 0.067 m/s and goal distance fell by 0.266 m. The final
50-step window coincided with the moving carrier reaching the entrance:
predicted risk saturated temporarily and the robot waited without collision.

This short run supports the static/local-layer separation mechanism, but it
does not prove eventual release or map completion. Preserve it and run a fresh,
longer seed with the same configuration before changing any other mechanism.

