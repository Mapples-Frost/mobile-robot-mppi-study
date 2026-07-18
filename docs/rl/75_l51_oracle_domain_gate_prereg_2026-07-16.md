# L51 Oracle-Domain Residual Gate: Development Upper-Bound Protocol

Date frozen: 2026-07-16 (Asia/Shanghai)

## Question

Before tuning the causal online reliability gate, L51 asks a narrower upper-bound
question: if a selector knew which of the two already frozen physics domains showed
reliable residual value, could selective ICODE activation preserve the strong-mismatch
smoothness benefit while exactly recovering nominal behavior in the weaker-mismatch
domain?

This is an oracle development ablation. It deliberately uses the configured domain
name and is therefore not deployable, not a learned gate, and not evidence that the
robot can identify mismatch online.

## Frozen selector

- `combined_long_delay`: use full L49 ICODE;
- `combined_matched_delay`: use nominal prediction;
- no future state, outcome or episode seed is used by the selector;
- no threshold is tuned inside L51.

## Frozen design

- Conditions: nominal, always-on ICODE, oracle-domain ICODE;
- 3 independently initialized L49 residual checkpoints;
- 3 frozen paths and 2 frozen physics domains;
- 10 new paired seeds `21560731` through `21560740`;
- 540 episodes total;
- seeds `21560741` through `21560750` remain sealed;
- memory and RL prior remain disabled;
- planner budget, cost, known-delay model, sensing and safety are shared.

## Integrity checks

The oracle condition must be stepwise identical to nominal in matched delay and
stepwise identical to always-on ICODE in long delay for state, issued control, applied
control, collision and safety fields. Any mismatch fails artifact integrity.

## Development gate

All clauses must pass:

1. complete episode/step coverage with no protected or sealed seed use;
2. exact oracle branch identity in both domains;
3. hierarchical 95% lower confidence bounds above zero for issued and applied jerk
   reductions versus nominal;
4. positive applied-jerk mean in all three model blocks;
5. no net success loss and no collision increase;
6. relative cross-track RMSE increase no greater than 5%;
7. completion-ratio difference at least -0.01;
8. mean oracle planner compute no greater than 50 ms.

Path length is reported as a secondary endpoint but is not a gate clause: L48/L50
showed that it is highly path-dependent and insufficiently stable to serve as the
mechanistic criterion for this selector upper bound.

Passing only permits online-gate calibration. It does not support a deployable gate
claim.

