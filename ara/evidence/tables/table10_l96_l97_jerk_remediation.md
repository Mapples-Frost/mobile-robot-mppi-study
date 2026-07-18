# Table 10: L96/L97 contextual half-budget jerk remediation

| Evidence unit | Result |
|---|---|
| L96 design | Randomized complete block 2x2 factorial; 24 route x physics x seed blocks; 96 episodes |
| L96 integrity | 96 unique keys, 24 complete blocks, every arm in every run position six times, 96 successes, zero collisions |
| Hard yaw slew vs current | Issued jerk -0.00935, 95% CI [-0.01127, -0.00749]; applied jerk -0.00603, 95% CI [-0.00820, -0.00421] |
| Stronger rate cost vs current | Issued jerk +0.00009, 95% CI [-0.00056, +0.00081]; applied jerk +0.00002, 95% CI [-0.00045, +0.00057]; jerk gates failed |
| L96 selected combined arm | Issued jerk -0.00941, 95% CI [-0.01069, -0.00812]; applied jerk -0.00608, 95% CI [-0.00818, -0.00435] |
| L97 design | Single process, cyclic three-arm order; 48 untouched route x physics x seed blocks; 144 episodes |
| L97 integrity | 144 unique keys, every arm in every run position 16 times, 144 successes, zero collisions |
| Raw contextual K50 vs fixed K100 | RMSE -1.034 mm, 95% CI [-2.981, +0.625]; time -1.683 s [-2.573, -0.777]; compute -8.902 ms/step [-10.509, -7.835] |
| Raw contextual jerk cost | Issued +0.00994, 95% CI [+0.00175, +0.01793]; applied +0.00650, 95% CI [+0.00072, +0.01260] |
| Smoothed vs raw contextual | Issued jerk -0.01078, 95% CI [-0.01271, -0.00887]; applied jerk -0.00705, 95% CI [-0.00939, -0.00512]; mechanism Gate passed |
| Smoothed contextual K50 vs fixed K100 | RMSE -0.621 mm; time -1.850 s; compute -8.896 ms/step; issued jerk -0.00084 CI [-0.00700, +0.00536]; applied jerk -0.00055 CI [-0.00494, +0.00402] |
| L97 overall decision | Failed: both final-package jerk intervals cross zero under the preregistered zero-upper-bound rule |

Sources: `results/research_platform/rl/l96_contextual_covariance_jerk_screening_20260718_v1/summary.json`,
`results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/summary.json`,
and `docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`.

Interpretation boundary: L97 supports the half-budget efficiency replication and
the smoothing mechanism. It does not support strict jerk superiority of the
complete smoothed package over fixed K100, arbitrary paths, dynamic obstacles or
real-robot transfer.
