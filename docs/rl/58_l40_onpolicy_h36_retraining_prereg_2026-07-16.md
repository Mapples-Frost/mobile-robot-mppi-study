# L40 on-policy H=36 ICODE retraining — preregistration

## Motivation

L39 found that all three existing ICODE checkpoints were trained with `H=10`, worsened on-policy `H=36` endpoint position error by 6.6% on average and increased heading RMSE by approximately threefold. L40 changes the data distribution and prediction objective, not the MPPI controller or safety system.

## Frozen dataset

The canonical block-0 L38 nominal-MPPI trajectories are converted into residual transitions before contact. Complete episode seeds define disjoint splits:

- train: `20760731`, `20760732`;
- validation: `20760733`;
- test: `20760734`;
- held-out episode test: `20760735`.

Every split contains all three calibrated scenes and both 40 ms and 100 ms delay strata. Normalization is computed from train only. The commanded control remains the model input because that is what MPPI supplies at inference; delayed interval-average control is retained as an audited auxiliary field.

## Frozen objective

- control-affine ICODE structure unchanged;
- rollout horizon: 36 steps (3.6 s), matching MPPI;
- state weights: `[4, 4, 2, 1, 1]` for `(x,y,theta,v,omega)`;
- rollout weights increase linearly from 1 to 4;
- loss weights: derivative 0.25, one-step 1, multistep 4;
- best checkpoint selected only by validation 36-step weighted rollout RMSE;
- three independent initialization seeds will be trained if the first implementation smoke is finite.

## Eligibility sequence

1. Data quality and episode-disjointness must pass.
2. At least two of three checkpoints must improve held-out H=36 position RMSE over nominal without >20% heading regression.
3. Only eligible checkpoints enter traditional-MPPI closed-loop development evaluation.
4. Closed-loop confirmation uses new seeds; L38/L39 split seeds cannot serve as confirmation.

No RL component is active in L40.
