# L60 Parameter-Matched Residual Structure Ablation Preregistration

## Question

Does the public ICODE-MPPI control-affine residual structure provide a predictive
advantage over an ordinary direct MLP residual when data, output mask, optimizer,
rollout horizon, training seeds and parameter count are controlled?

## Frozen experimental unit and blocking

- Independent training replicate: one model-initialization/training seed.
- Blocks: seeds `20261201`, `20261202`, `20261203`.
- Repeated measurements within a block: evaluation windows and path episodes;
  these are not treated as independent model replicates.
- Dataset: immutable L56 episode-wise train/validation/test/unseen splits.
- Primary held-out horizon: `H=36`, matching the MPPI rollout horizon.
- Ordinary MLP: direct `MLP(encoded_state, control)` residual, two hidden layers
  of width 93, Softplus activation, dynamic-state output mask.
- ICODE: `f_theta(encoded_state) + G_theta(encoded_state) control`, two 64-wide
  branches, otherwise identical training contract.
- Capacity control: relative parameter-count difference at most 3%.

## Frozen decisions

1. Dataset hashes must equal the L57 hashes; training statistics come only from
   the train split.
2. Every valid evaluation window is used for `H=1,5,10,20,36`.
3. MLP closed-loop eligibility uses the same nominal-relative H=36 thresholds
   that qualified ICODE in L57.
4. Offline support for the structural hypothesis requires ICODE to have lower
   H=36 rollout RMSE than MLP in at least two of three blocks on both test and
   unseen splits, with positive mean advantage on both splits.
5. Closed-loop testing is allowed whenever both learned models pass their own
   nominal-relative eligibility gates. It is not conditioned on which learned
   structure wins offline.
6. No RL prior or memory augmentation is enabled in this ablation.

## Interpretation boundary

This experiment isolates network structure on one fixed MuJoCo plant. It cannot
establish cross-plant, obstacle, real-robot or RL-composed robustness. The term
ICODE refers only to the control-affine residual structure publicly described in the
ICODE-MPPI paper; no unimplemented stability or convergence guarantee is claimed.
