# L39 on-policy H=36 residual prediction diagnostic — preregistration

## Purpose

L38 refuted command-delay shift as a sufficient explanation of the L37 reversal. L39 therefore tests whether the current ICODE checkpoints retain their offline advantage on the state–command distribution actually visited by nominal MPPI and over the full 36-step planning horizon.

## Frozen data and unit of analysis

- Source: completed L38 `traditional_nominal` trajectories only.
- Block-0 nominal trajectories are the canonical environment trajectories; nominal trajectories in blocks 1 and 2 must match exactly and are not counted as additional replicates.
- Thirty canonical episodes cover three calibrated scenes, two delay strata and five episode seeds.
- Each of the three ICODE checkpoints is evaluated on the same canonical trajectories.
- Non-overlapping valid windows are formed separately for `H = 1, 5, 10, 20, 36`.
- Windows ending in collision are excluded because post-contact rigid-body dynamics are outside the residual-learning target.
- Windows are nested measurements. Metrics are first averaged within episode; model checkpoint and episode are retained as blocking levels.

## Metrics

For nominal and ICODE predictions:

- endpoint position RMSE;
- endpoint wrapped-heading RMSE;
- endpoint linear-velocity and yaw-rate RMSE;
- full-window position RMSE;
- maximum state-feature and control normalization z-score seen by ICODE.

The primary estimand is paired nominal-minus-ICODE endpoint position RMSE at `H=36`.

## Frozen gate

The existing ICODE family is eligible for direct MPPI cost-ranking analysis only if:

1. all canonical trajectories are identical across nominal model blocks;
2. finite predictions exist for every planned horizon;
3. at least two of three checkpoints improve episode-mean H=36 endpoint position RMSE;
4. pooled H=36 endpoint position RMSE improves by at least 5%;
5. no checkpoint worsens H=36 heading RMSE by more than 20%.

If the gate fails, the current checkpoints are not used for further controller tuning. The next stage must rebuild the dataset/training objective on MPPI on-policy data with the planning horizon represented explicitly.
