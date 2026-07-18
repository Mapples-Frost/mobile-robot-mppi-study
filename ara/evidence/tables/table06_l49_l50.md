# Table 06: L49/L50 task-specific ICODE evidence

| Evidence unit | Result |
|---|---|
| L49 test H=36 rollout reduction | 49.61% to 49.83% across 3 initialisation seeds |
| L49 unseen H=36 rollout reduction | 50.45% to 50.90% across 3 initialisation seeds |
| L49 position reduction | 64.86% to 70.31% |
| L49 heading reduction | 69.14% to 73.70% |
| L50 episodes | 360/360 complete; 180 paired comparisons |
| L50 applied jerk, all domains | +0.000732; 95% CI [0.000077, 0.001505] |
| L50 issued jerk, long delay | +0.001194; 95% CI [0.000065, 0.002344] |
| L50 applied jerk, long delay | +0.001282; 95% CI [0.000264, 0.002365] |
| L50 safety/task deltas | 0 collision increase; 0 net success loss |
| L50 global joint gate | Failed |

Sources: `results/research_platform/l49_icode_iterative_path_offline_gate_v1/summary.json`
and `results/research_platform/rl/l50_icode_iterative_path_confirmation_20260716_v1/efficiency_confirmation_summary.json`.

Interpretation boundary: long-delay intervals are a frozen stratum analysis, but the
domain interaction still requires new-seed confirmation before becoming a final-paper
generality claim.
