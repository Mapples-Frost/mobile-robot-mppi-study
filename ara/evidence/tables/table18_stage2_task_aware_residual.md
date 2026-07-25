# Table 18: Task-aware residual retraining and fresh-seed pilot

## Offline H36 position RMSE

| Checkpoint seed | Task L57 | Task Stage 2 | Task change | Original test change | Original unseen change | Block gate |
|---|---:|---:|---:|---:|---:|---:|
| 20261201 | 0.04990 | 0.04325 | -13.3% | -11.5% | +0.97% | pass |
| 20261202 | 0.04797 | 0.04733 | -1.3% | -0.92% | +0.02% | pass |
| 20261203 | 0.04862 | 0.04747 | -2.4% | -1.20% | -1.47% | pass |

All three task-test active residual RMSE values beat nominal. The frozen
offline gate required at least two passing blocks and no more than 2% original
test/unseen H36 position regression; 3/3 blocks passed.

## Fresh development seed 730100006

| Condition | Checkpoint seed | Collision | Goal | Steps | Final distance (m) | Minimum clearance (m) | Shield accept | Planner P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | — | 0 | reached | 327 | 0.29867 | 0.39754 | — | 54.77 |
| Stage 2 residual | 20261201 | 0 | reached | 327 | 0.29440 | 0.39754 | 67.6% | 422.08 |
| Stage 2 residual | 20261202 | 0 | reached | 324 | 0.28977 | 0.39754 | 72.5% | 267.25 |
| Stage 2 residual | 20261203 | 0 | reached | 332 | 0.29135 | 0.39754 | 73.2% | 419.09 |

Scope: development-only evidence on one fresh obstacle seed. Training,
validation and offline-test obstacle seeds were 730100001, 730100003 and
730100005 respectively. No sealed seed was opened.

Sources:

- `research_artifacts/dynamic_uncertainty_residual_stage2_offline/gate.json`
- `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_probe/gate.json`
- `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_probe/episode_summary.csv`
- `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_RESULT.md`
