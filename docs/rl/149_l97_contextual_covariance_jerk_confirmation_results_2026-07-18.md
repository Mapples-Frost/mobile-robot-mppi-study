# L97 contextual half-budget jerk confirmation: mechanism passes, final strict Gate fails

Date: 2026-07-18  
Conclusion level: preregistered, untouched-seed, single-process three-arm confirmation

## Integrity

- 144/144 complete MuJoCo episodes;
- 48/48 route x physics x seed blocks;
- 144 unique treatment keys and finite metrics;
- each arm occupied each run position exactly 16 times;
- 144/144 successes and zero collisions;
- no concurrent benchmark worker;
- frozen ICODE, L89 bandit, MPPI costs/horizon and safety chain.

## Contrast A: raw contextual K50 minus fixed K100

| Metric | Mean delta | Hierarchical bootstrap 95% CI | Gate |
|---|---:|---:|---|
| Cross-track RMSE | -1.034 mm | [-2.981, +0.625] mm | +2 mm noninferiority pass |
| Elapsed time | -1.683 s | [-2.573, -0.777] s | pass |
| Planner compute | -8.902 ms/step | [-10.509, -7.835] | pass |
| Issued jerk | +0.00994 | [+0.00175, +0.01793] | reported cost |
| Applied jerk | +0.00650 | [+0.00072, +0.01260] | reported cost |

This independently reproduces the L94/L95 half-budget precision, time, compute
and safety result, together with its jerk cost.

## Contrast B: smoothed contextual K50 minus raw contextual K50

| Metric | Mean delta | Hierarchical bootstrap 95% CI | Gate |
|---|---:|---:|---|
| Cross-track RMSE | +0.413 mm | [-0.237, +1.110] mm | pass |
| Elapsed time | -0.167 s | [-0.265, -0.069] s | pass |
| Issued jerk | -0.01078 | [-0.01271, -0.00887] | pass |
| Applied jerk | -0.00705 | [-0.00939, -0.00512] | pass |

The L96-selected mechanism independently reduces both commanded and physically
applied jerk without a safety, tracking or completion-time loss. The mechanism
Gate passes.

## Contrast C: smoothed contextual K50 minus fixed K100

| Metric | Mean delta | Hierarchical bootstrap 95% CI | Gate |
|---|---:|---:|---|
| Cross-track RMSE | -0.621 mm | [-2.607, +1.136] mm | pass |
| Elapsed time | -1.850 s | [-2.654, -1.025] s | pass |
| Planner compute | -8.896 ms/step | [-9.726, -8.166] | pass |
| Issued jerk | -0.00084 | [-0.00700, +0.00536] | fail strict zero-upper-bound clause |
| Applied jerk | -0.00055 | [-0.00494, +0.00402] | fail strict zero-upper-bound clause |

The final package has slightly lower mean jerk than fixed K100, but both
confidence intervals cross zero. Therefore the preregistered requirement that
the complete interval be nonpositive is not met. `primary_gate_passed=false`.
This may be described as removal of the *mean* jerk penalty or statistical
compatibility with parity, but not as confirmed jerk superiority or a passed
zero-margin noninferiority Gate.

## Scientific interpretation

The defensible result is now narrower and stronger:

1. route-context covariance selection repeatedly preserves precision and safety
   with half the MPPI sample budget while reducing time and CPU compute;
2. a fixed, hardware-interpretable yaw-slew constraint removes the contextual
   controller's excess jerk relative to its unsmoothed version;
3. the data do not prove that the complete package is smoother than the tuned
   fixed K100 baseline under a zero-margin confidence-interval rule.

No threshold, seed or claim was amended after opening L97. The failed clause is
retained as a scope boundary and future power/margin decisions must be
preregistered separately.

Artifacts:

- `results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/summary.json`
- `results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/episodes.csv`
- `results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/fig_l96_l97_jerk_remediation.pdf`
