# L157--L159: RL-Driven ICODE-MPPI Gate 1 results

Date: 2026-07-18
Status: completed development study; the SAC policy/value path is **not retained** as the primary controller.

## Question

This gate tested whether the repository's frozen SAC actor and twin critic add
measurable value when used in the roles described by RL-Driven MPPI:

1. the actor proposes an MPPI control-sequence prior;
2. the critic supplies an optional terminal value;
3. ICODE supplies the rollout dynamics.

The implementation is opt-in and leaves the standard MPPI, LaserScan obstacle
path, `scan_guard`, and final safety arbitration unchanged.

## Implemented research interfaces

- `RLDrivenMPPIOptimizer` supports multiple named proposal distributions,
  proposal-level elite accounting, and an optional terminal twin-Q term.
- `TorchSACPrior` can generate batched action proposals and evaluate terminal
  states without mutating online observation history.
- `HybridBaselinePrior` provides a no-RL control with the same optimizer and
  proposal mixture structure.
- All new behavior is disabled unless explicitly selected in configuration.

## Gate 1b: sparse actor and critic authority

The cleanest development comparison used nine paired episodes per condition.

| Condition | Success | Collision | Mean task cost | Mean planner time (ms) |
|---|---:|---:|---:|---:|
| Hybrid no-RL + ICODE | 9/9 | 0/9 | 0.466913 | 39.405 |
| Critic only + ICODE | 9/9 | 0/9 | 0.470443 | 59.330 |
| Sparse SAC policy + ICODE | 9/9 | 0/9 | 0.468480 | 39.884 |
| Sparse policy + critic + ICODE | 9/9 | 0/9 | 0.467720 | 59.890 |

The full sparse SAC condition was approximately 0.17% worse than the matched
no-RL hybrid control in mean task cost. The preregistered improvement threshold
was at least 1%, so the gate failed. The critic also added substantial compute
without improving task cost.

## Gate 1c: contextual covariance inside the hybrid optimizer

This diagnostic contained 24 completed episodes. Learned contextual covariance
versus the strongest fixed covariance produced:

- elapsed-time delta: +0.0083 s, 95% CI [-0.1500, +0.1583];
- cross-track RMSE delta: +1.301 mm, 95% CI [-0.289, +3.738] mm;
- mean planner compute delta: -0.110 ms, 95% CI [-1.184, +0.940] ms;
- identical success and collision outcomes.

The frozen +2 mm tracking noninferiority margin was not established because the
upper confidence limit was +3.738 mm. The primary gate therefore failed.

## Decision

The SAC proposal and terminal-value code remains available as a faithful,
reproducible ablation, but it is not presented as the successful RL component.
The retained RL mechanism is the interpretable contextual bandit that selects
an MPPI sampling covariance. It satisfies the broader principle of an offline
learned policy guiding online MPPI sampling, while avoiding a claim that the
repository has reproduced the complete SAC-based RL-Driven MPPI result.

Raw outputs are under:

- `results/research_platform/rl/gate1b_sparse_development_20260718/`
- `results/research_platform/rl/gate1c_hybrid_contextual_covariance_development_20260718/`
