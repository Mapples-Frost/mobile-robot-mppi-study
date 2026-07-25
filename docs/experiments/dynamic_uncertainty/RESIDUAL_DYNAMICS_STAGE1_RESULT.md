# Residual-Dynamics Stage 1 Development Result

Status: **FAIL — retained negative integration result**  
Date: 2026-07-23  
Sealed seeds opened: **no**

## Design

The frozen Amendment 17 risk-enabled environment was compared under:

- one shared nominal episode per obstacle seed; and
- three frozen L57 iCODE residual checkpoint blocks.

This produced 12 complete closed-loop episodes and nine paired residual-versus-
nominal comparisons. Only `planner.prediction_mode` and the registered
checkpoint path changed.

## Prediction result

All three residual blocks improved the primary horizon-36 velocity/yaw-rate
rollout RMSE on the shared nominal trajectories:

| checkpoint seed | nominal RMSE | residual RMSE | relative improvement |
|---:|---:|---:|---:|
| 20261201 | 0.06621 | 0.05501 | 16.91% |
| 20261202 | 0.06621 | 0.05579 | 15.74% |
| 20261203 | 0.06621 | 0.05672 | 14.33% |

Thus the prediction gate passed 3/3 model blocks.

## Closed-loop result

The overall gate failed:

- collision-count increases by model block: `+1`, `+1`, `+2`;
- median clearance delta: `-0.03018 m` against the `-0.02 m` floor;
- maximum residual planner p95: `206.01 ms` against `150 ms`;
- median completion delta passed at `-0.00667`.

Seed `730100003` collided under all three residual checkpoints at approximately
steps 226--227, while its shared nominal reference remained collision-free and
finished `0.3334 m` from the goal. Seed `730100005` reached the goal under all
three residual checkpoints. The failure is therefore trajectory dependent but
reproducible across model-training seeds.

For seed `730100003`, all three residual controllers first differed from the
nominal executed control at step 21, but position divergence did not exceed
`0.10 m` until steps 133--144. The eventual collision therefore followed a
long-horizon closed-loop mode divergence rather than a one-step inference
failure immediately before contact.

## Integration diagnosis

The L57 feature normalizer was fitted around absolute position
`x=3.2452, y=0.3136`, while the Amendment 17 route starts at
`x=-4.6, y=-1.8`. Across the three shared nominal trajectories:

- maximum spatial normalized deviation was `5.97--6.23`;
- the fraction of steps with either spatial feature beyond 5 standard
  deviations was `87.5%--100%`;
- maximum non-spatial deviation remained `3.93--4.06`.

This reveals a known coordinate-interface mismatch: on the spatially
homogeneous plant, the learned actuation residual should not depend on global
translation, but raw global `x,y` were passed far outside their training
support. The repository already contains a production
`CanonicalizedStateResidualDynamics` adapter used for expanded spatial maps.

This diagnosis does not erase the failed result. It authorizes only the
separately preregistered, single-factor canonicalization probe; it does not
authorize checkpoint selection, environment changes, or sealed evaluation.

## Evidence

- `research_artifacts/dynamic_uncertainty_residual_stage1_development/gate.json`
- `research_artifacts/dynamic_uncertainty_residual_stage1_development/episode_summary.csv`
- `research_artifacts/dynamic_uncertainty_residual_stage1_development/prediction_metrics.json`
- `research_artifacts/dynamic_uncertainty_residual_stage1_development/closed_loop_pairs.csv`
- `research_artifacts/dynamic_uncertainty_residual_stage1_development/failure_diagnostics.json`
