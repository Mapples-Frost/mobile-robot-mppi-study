# Table 27: AR(1) Chapter 1 closed-loop screening result

Frozen protocol SHA256:
`f1dfd4098941bb9101c0efc525295ac31df3c49a6aa7bba79fa2d5c37cc68d0e`

| Metric | iid | AR(1), tau=2.0 s |
|---|---:|---:|
| Paired seeds | 12 | 12 |
| Success | 0 | 0 |
| Collision | 9 | 6 |
| Timeout | 3 | 6 |
| Final distance, episode median (m) | 5.3915 | 3.8397 |
| Minimum overall clearance, episode median (m) | -0.0066 | 0.0469 |
| Zero-feasible steps, total | 230 | 92 |
| Hard violations, total | 443 | 187 |
| Fallback activations, total | 1240 | 645 |
| Planner P95, episode mean (ms) | 601.20 | 561.95 |
| Post-limit Gaussian slew violation fraction | 0 | 0 |

Paired Gate quantities:

- net success pairs: 0;
- treatment-only collision pairs: 1 (safety failure requires at least 2);
- control-only collision pairs: 4;
- median paired final-distance reduction: 0.283041 m (confirmatory screen requires at least 0.25 m);
- verdict: `PASS_TO_CONFIRMATORY`.

Interpretation boundary: this table supports only a fresh-seed confirmatory
experiment. It does not show a success-rate improvement because neither arm
completed any episode, and the distance threshold margin was only about 0.033 m.

Primary proof:

- `research_artifacts/noise_basis_ab_v1/ab_screen_result.json`
- `research_artifacts/noise_basis_ab_v1/paired_results.csv`
- `research_artifacts/ar1_closed_loop_screen_ch1/AUDIT.md`
- `research_artifacts/ar1_closed_loop_screen_ch1/RATE_LIMIT_AUDIT.json`
