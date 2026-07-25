# Residual Dynamics Stage 2 Task-Aware Retraining Result

Date: 2026-07-23  
Status: offline gate and fresh-seed pilot passed. The subsequent three-seed
development matrix also passed; see
`RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_DEVELOPMENT_RESULT.md`.

## Why Stage 2 was necessary

The frozen L57 checkpoints improved held-out dynamics prediction but caused
closed-loop collisions.  A full-action dual-model shield (Stage 1 Amendment 5)
and a speed-only shield (Amendment 6) both collided on development seed
`730100003`.  The stopping rule therefore ended wrapper and threshold tuning.

The L57 provenance showed that its 24-episode, 6053-transition training split
contained generic high-dynamic paths but no dynamic-obstacle interaction
episodes from the current task.

## Implemented correction

- Constructed an episode-disjoint task-aware dataset on Windows:
  - train: 6053 original + 400 obstacle transitions (`730100001`);
  - validation: 744 original + 400 obstacle transitions (`730100003`);
  - test: 2261 original + 378 obstacle transitions (`730100005`);
  - unseen: unchanged 3016 original transitions.
- Retained the L57 ICODE structure and output mask
  `[0, 0, 0, 1, 1]`; pose kinematics are still not learned.
- Added positive per-transition importance weights.  Task weights ranged from
  approximately 4.0 to 8.4 and increased near the moving obstacle.
- Fine-tuned each model from its matching L57 checkpoint at learning rate
  `5e-5`.
- Added a functional output-anchor loss on original
  `nominal_mppi_onpolicy` rows to reduce catastrophic forgetting.
- Kept RL disabled.
- Corrected shield diagnostics so downstream safety receives conservative risk
  values for the actual accepted sequence, including hybrid sequences.

## Offline qualification

Frozen gate artifact:
`research_artifacts/dynamic_uncertainty_residual_stage2_offline/gate.json`.

All three model blocks passed.

| Seed | Task H36 position RMSE, L57 | Task H36, Stage 2 | Change | Original test H36 change | Original unseen H36 change |
|---|---:|---:|---:|---:|---:|
| 20261201 | 0.04990 | 0.04325 | -13.3% | -11.5% | +0.97% |
| 20261202 | 0.04797 | 0.04733 | -1.3% | -0.92% | +0.02% |
| 20261203 | 0.04862 | 0.04747 | -2.4% | -1.20% | -1.47% |

Task-test active residual RMSE also beat the nominal model in all three
blocks.  The maximum original-data regression was 0.97%, below the frozen 2%
limit.  Thus the update did not obtain task improvement by discarding the old
model's general capability.

## Fresh-seed MuJoCo closed-loop pilot

Pilot seed `730100006` was registered before execution and was not used in
training, validation, or offline test.  Sealed seeds were not opened.

| Condition | Checkpoint seed | Collision | Goal reached | Steps | Final distance (m) | Minimum clearance (m) | Shield accept | Planner P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | — | 0 | yes | 327 | 0.29867 | 0.39754 | — | 54.77 |
| Stage 2 residual | 20261201 | 0 | yes | 327 | 0.29440 | 0.39754 | 67.6% | 422.08 |
| Stage 2 residual | 20261202 | 0 | yes | 324 | 0.28977 | 0.39754 | 72.5% | 267.25 |
| Stage 2 residual | 20261203 | 0 | yes | 332 | 0.29135 | 0.39754 | 73.2% | 419.09 |

The automatic pilot gate passed:

- 0 collisions in nominal and all 3 residual blocks;
- 4/4 episodes reached the goal;
- median completion delta `+0.000831`;
- median clearance delta `0.0 m`;
- all three blocks improved offline H36 prediction;
- the shield accepted 67.6–73.2% of residual plans, so success was not produced
  by permanent nominal fallback.

## Interpretation and remaining limit

This is the first evidence in the current environment that residual learning
can retain its predictive benefit without producing an observed closed-loop
safety penalty.  The result supports the diagnosis that training-distribution
and objective mismatch were the main causes of the prior negative effect.

This pilot was not a final safety claim because it covered one new
obstacle-process seed. The preregistered paired development matrix on seeds
`730100006`, `730100008`, and `730100010` has now been completed and passed.
The expanded result does not establish a consistent speed benefit or justify
selecting one checkpoint. Sealed evaluation remains unopened.

The remaining engineering problem is runtime.  Dual-planner shield P95 was
267–422 ms, above the 100 ms control interval.  Runtime optimization must be
performed after the multi-seed safety result and must not change the learned
model gate retroactively.
