# Table 09: L90--L95 covariance-adaptation scope and sample-efficiency evidence

| Evidence unit | Result |
|---|---|
| L90 local-context oracle | 240/240 branches; 11/12 contexts selected speed; progress +0.001385 m, 95% CI [0, +0.004884]; local heterogeneity gate failed |
| L91 dynamic-scene development | 160/160 episodes; success +0.30, 95% CI [+0.05, +0.55]; collision -0.15, 95% CI [-0.40, 0]; elapsed-time gate failed |
| L92 dynamic-safety confirmation | 300/300 episodes; success +0.0556, 95% CI [-0.1389, +0.25]; collision 0, 95% CI [-0.1389, +0.1667]; safety gate failed |
| L93 route-crossing oracle | 120/120 episodes; no success/collision benefit; elapsed -0.40 s, 95% CI [-2.236, +1.296]; no covariance headroom |
| L94 design | Held-out hairpin/reverse-S, four bounded physics domains, K=50/100/200/400, five untouched seeds; 320 unique episodes |
| L94 primary: contextual K=50 vs fixed K=100 | RMSE -0.004108 m, 95% CI [-0.007009, -0.000741]; elapsed -1.7375 s, 95% CI [-2.72, -0.76]; compute -18.2905 ms/step, 95% CI [-32.905, -2.693] |
| L94 task and safety | 40/40 paired successes in each arm; zero collisions; all primary gates passed |
| L94 smoothness tradeoff | Jerk +0.00944, 95% CI [+0.00330, +0.01573] |
| L95 controlled confirmation | Single process, alternating arm order, 40 untouched-seed pairs; 80/80 successes and zero collisions |
| L95 precision | RMSE -0.000273 m, 95% CI [-0.001816, +0.001520], within the preregistered +0.002 m noninferiority margin |
| L95 efficiency | Elapsed -1.725 s, 95% CI [-2.68, -0.77]; compute -10.359 ms/step, 95% CI [-15.085, -6.908] |
| L95 smoothness tradeoff | Jerk +0.01029, 95% CI [+0.00390, +0.01658] |

Sources: `results/research_platform/rl/l90_local_covariance_context_oracle_20260718_v1/summary.json`,
`results/research_platform/rl/l91_dynamic_covariance_context_oracle_20260718_v1/summary.json`,
`results/research_platform/rl/l92_dynamic_covariance_safety_confirmation_20260718_v1/summary.json`,
`results/research_platform/rl/l93_route_dynamic_covariance_headroom_20260718_v1/summary.json`,
`results/research_platform/rl/l94_contextual_covariance_sample_efficiency_20260718_v1/summary.json`,
and `results/research_platform/rl/l95_contextual_covariance_half_budget_confirmation_20260718_v1/summary.json`.

Interpretation boundary: the supported result is a simulation-only, route-level,
high-level covariance selection claim on two held-out route geometries and four
bounded physics domains. It does not establish local switching, dynamic-obstacle
safety, arbitrary OOD robustness, real-robot transfer, or universally lower jerk.
