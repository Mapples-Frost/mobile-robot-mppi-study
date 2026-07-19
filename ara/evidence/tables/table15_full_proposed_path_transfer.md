# Table 15: Full Proposed sealed polyline path transfer

Source revision: `917b867f8bb6c4818f3b676a8ab8dc4b6b42e60d`

| Arm | Cross-track RMSE (m) | Completion | Success | Collision | Jerk | Planner (ms) |
|---|---:|---:|---:|---:|---:|---:|
| ordinary fixed | 1.8969 | 0.9516 | 0/45 | 0/45 | 0.0890 | 386.9 |
| value fixed | 1.9223 | 0.9711 | 0/45 | 0/45 | 0.0899 | 381.2 |
| ordinary adaptive | 1.7266 | 0.9870 | 0/45 | 0/45 | 0.0862 | 322.2 |
| full proposed | 1.7419 | 0.9673 | 0/45 | 0/45 | 0.0868 | 321.9 |

Full Proposed versus ordinary fixed:

| Outcome | Favorable effect | Seed-cluster 95% CI |
|---|---:|---:|
| cross-track RMSE | +0.1550 m | [0.1201, 0.1899] |
| cross-track maximum | +0.1135 m | [0.0950, 0.1332] |
| completion ratio | +0.0157 | [-0.0331, 0.0645] |
| jerk | +0.00226 | [0.00074, 0.00347] |
| planner time | +65.0 ms | [54.7, 75.3] |
| collision | 0 | [0, 0] |

Factorial cross-track main effects:

- adaptive HSS: favorable 0.1754 m, CI [0.1239, 0.2297];
- value-aligned ICODE: adverse 0.0204 m, CI [0.0010, 0.0434];
- interaction: favorable 0.0100 m, CI [-0.0256, 0.0467].

All methods failed terminal success. Projection completion must not be reported as
goal-reaching success.

## Forensic bindings

- `results/research_platform/rl/full_proposed_path_transfer_confirm_l183_analysis/analysis_summary.json`
- `results/research_platform/rl/full_proposed_path_transfer_confirm_l183_analysis/paired_comparisons.json`
- `results/research_platform/rl/full_proposed_path_transfer_confirm_l183_analysis/factorial_contrasts.json`
- `docs/rl/184_full_proposed_path_tracking_confirmation_results_2026-07-19.md`
