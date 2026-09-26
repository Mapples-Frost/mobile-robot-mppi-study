# L61 Residual-Structure Closed-Loop Ablation Preregistration

## Motivation

L60 found that the parameter-matched direct MLP was independently eligible for
closed-loop testing, while the frozen offline control-affine superiority hypothesis
was not supported. L61 tests whether the structures differ after MPPI optimization
and receding-horizon feedback rather than relabeling the offline result.

## Frozen design

- Conditions: nominal MPPI, parameter-matched MLP residual MPPI, ICODE residual MPPI.
- Blocks: three paired MLP/ICODE training seeds.
- Scenes: the four frozen L56 path families; reverse-S remains unseen in training.
- Development episode seeds: `21860781`--`21860785`.
- Sealed confirmation seeds: `21860791`--`21860800`; these remain unopened.
- Unit of model replication: training seed. Episode seeds and control steps are
  repeated measurements within model blocks.
- Common plant, action bounds, MPPI samples/horizon, safety chain and path costs.
- RL and memory remain disabled.

## Primary comparisons

1. MLP versus nominal cross-track RMSE.
2. ICODE versus nominal cross-track RMSE.
3. ICODE versus MLP cross-track RMSE; positive values favor ICODE.

Both learned models must independently retain success/collision/completion safety,
positive tracking effects in all model blocks, a hierarchical-bootstrap lower bound
above zero, and mean planning time below 50 ms. Structural support additionally
requires ICODE to beat MLP in at least two blocks overall and on the unseen path,
with both paired confidence-interval lower bounds above zero.

Failure of the structure clause will not invalidate residual learning in general;
it will reject the narrower claim that control-affine factorization is responsible
for the fixed-plant tracking benefit.

