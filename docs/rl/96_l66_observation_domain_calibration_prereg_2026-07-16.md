# L66 observation-domain calibration preregistration

Date: 2026-07-16

## Question

Before reintroducing RL, can the fixed L56 MuJoCo plant provide safe and
measurable state-observation perturbations that separate residual-dynamics
effects from localization, noise, and latency effects?

## Method-blind design

Only traditional nominal MPPI is run during calibration. The plant, paths,
planner, safety chain, K=600 and horizon=36 are frozen. Eight observation
domains vary only pose/twist source, Gaussian observation noise, or observation
latency. MLP, ICODE and RL outcomes are unavailable to the selector.

The primary calibration selects one noise level, one latency level, and one
combined noise/latency level. Candidates must retain at least 80% success, zero
collisions, 95% mean path completion and at least a 3% nominal cross-track RMSE
shift from clean ground truth.

Raw wheel odometry is recorded as an exploratory stress domain because earlier
development showed that uncorrected slip can dominate terminal tracking. It is
not allowed to determine whether the primary bounded-noise/latency calibration
passes.

## Replication and claim boundary

Three execution seeds are repeated across two path geometries. Episode seeds
are repeated measurements, not independent model replicates. Passing only
freezes observation domains for a later three-checkpoint residual comparison;
it is not evidence that MLP or ICODE is robust, and it is not an RL result.

## Version 2 amendment

Version 1 passed safety and completion for bounded noise/latency domains, but
pure observation noise changed nominal cross-track RMSE by only 1.5%, below the
declared 3% resolution floor. Latency and combined candidates are therefore
frozen unchanged. Version 2 tests two stronger pure-noise candidates on fresh
seeds, still without observing MLP, ICODE or RL outcomes. This is the final
planned calibration amendment; another failure removes the pure-noise factor
instead of triggering further tuning.
