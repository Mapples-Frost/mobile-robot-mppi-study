# MPPI Dynamics Integration

## Protected seam

Only the existing helper signature was extended:

```python
rollout_control_sequence(start_state, control_sequence, dt, dynamics_model=None)
```

When `dynamics_model is None`, the function retains the original loop, legacy
Euler `step`, per-step heading wrap, list return, tuple states, and arithmetic
order.  A golden regression locks those values.  `sample_control_sequences`
accepts the same optional model at the end of its signature for anisotropic
nominal geometry.

No MPPI cost, weighting, update, obstacle, Memory, or safety code was rewritten.

## Prediction modes

- `nominal`: nominal unicycle prediction;
- `oracle_residual`: nominal plus exact true-minus-nominal residual, explicitly
  warned and labeled as an upper-bound/interface ablation;
- `mlp_residual`: nominal plus a loaded MLP checkpoint;
- `icode_residual`: nominal plus a loaded control-affine checkpoint.

`MppiDynamicsAdapter` handles Euler/RK4 and clones only stateful/delayed models.
It never deep-copies a stateless PyTorch model per sample.  The planner imports
the adapter lazily only for non-default rollouts; the default planner import
does not load Torch.

## Planner versus plant

The planner rolls candidate controls through its prediction model.  The
environment executes only the first proposed/final control through a separate
true plant.  Learned modes are rejected if a true-plant object is passed into
their factory.  Only the oracle mode receives true dynamics.

## Perception and safety boundary

The formal MuJoCo/robot chain remains:

```text
LaserScan -> scan_guard -> local_obstacle_layer -> planner obstacles
-> MPPI proposal -> safety arbitration -> final control
```

The new clean benchmark has no obstacles by design and is named
`clean_dynamics`; it is not presented as a replacement for the perception
chain.  Existing `simple`, `lab_complex`, `narrow_corridor`, and
`u_trap_long_board` live scenarios remain untouched.  Memory is off in the
primary residual ablation.

## Sampling prior extension point

`SamplingPrior.mean_control_sequence(state,horizon)` currently has
`PreviousSequencePrior` and `GoalWarmStartPrior`.  A future `RLPolicyPrior` can
implement the same protocol without making current MPPI depend on PyTorch RL.
