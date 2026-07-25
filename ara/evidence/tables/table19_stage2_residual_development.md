# Table 19: Stage 2 task-aware residual development matrix

## Complete-episode outcomes

| Obstacle seed | Nominal steps | R01 delta | R02 delta | R03 delta |
|---:|---:|---:|---:|---:|
| 730100006 | 327 | 0 | -3 | +5 |
| 730100008 | 331 | -14 | -2 | -10 |
| 730100010 | 330 | +7 | +5 | +9 |

All 12 episodes reached the goal with zero collisions and zero safety-layer
interventions. Across nine paired residual comparisons, step delta had median
`0`, mean `-0.33` and range `[-14, +9]`.

## Residual contribution audit

| Block | Median shield acceptance | Median paired-realized effective acceptance | Median executed difference > 0.01 | Maximum paired position difference (m) |
|---|---:|---:|---:|---:|
| R01 / 20261201 | 67.6% | 29.7% | 59.0% | 0.110 |
| R02 / 20261202 | 71.4% | 22.1% | 48.8% | 0.053 |
| R03 / 20261203 | 70.4% | 31.2% | 52.0% | 0.080 |

The identical global minimum clearance of approximately `0.39754 m` occurred
at the first recorded step in every run. Fifth-percentile clearance deltas
were therefore used as a distributional supplement; their median was
approximately zero with range `[-0.0357, +0.0158] m`.

Residual planner P95 ranged from `221.84` to `422.08 ms`, above the `100 ms`
control period despite passing the frozen `600 ms` development artifact limit.

Scope: three development obstacle seeds per checkpoint block; episode is the
independent unit. No sealed seed was opened. The result supports development
safety/noninferiority and nondegenerate residual participation, not a
population safety guarantee, speed superiority or checkpoint selection.

Sources:

- `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/gate.json`
- `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/episode_summary.csv`
- `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/closed_loop_pairs.csv`
- `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/contribution_audit.json`
- `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_DEVELOPMENT_RESULT.md`
