# L62 Sealed Residual-Structure Confirmation Results

## Outcome

The complete inherited L62 gate passed. All 360 scheduled episodes completed using
the ten predeclared confirmation seeds, with no protected-seed overlap and exact
checkpoint hashes. Nominal, MLP and ICODE each achieved 120/120 successes and zero
collisions.

| Contrast | Relative cross-track reduction | Absolute improvement | 95% hierarchical-bootstrap interval |
|---|---:|---:|---:|
| Parameter-matched MLP vs nominal | 17.74% | 0.01136 m | [0.00922, 0.01308] m |
| ICODE vs nominal | 26.83% | 0.01593 m | [0.01368, 0.01790] m |
| ICODE vs parameter-matched MLP | 10.17% | 0.00457 m | [0.00355, 0.00564] m |

The direct ICODE-versus-MLP effect was positive in all three independently trained
model blocks. On the unseen reverse-S path, the absolute ICODE advantage was
0.00586 m with interval [0.00408, 0.00726] m, again positive in all three blocks.

Mean planning time was 4.99 ms for nominal, 30.81 ms for MLP and 35.57 ms for
ICODE. Both learned methods remained below the frozen 50 ms mean-compute limit.
ICODE also reduced applied-control jerk relative to MLP by 0.00116 in the recorded
normalized jerk metric while retaining indistinguishable completion and safety.

## Scientific interpretation

L60 and L62 jointly falsify a simplistic chain of reasoning in which the model with
the lowest aggregate offline rollout RMSE must yield the best MPPI controller. The
parameter-matched direct MLP had slightly lower H=36 offline RMSE, yet ICODE produced
significantly lower closed-loop tracking error on both development and sealed seeds.
This supports the empirical value of the control-affine residual factorization for
this MPPI optimization contract, even though it does not prove the causal mechanism.

The current defensible claim is limited to the published control-affine residual
structure on one fixed high-dynamic MuJoCo differential-drive plant with clean
ground-truth state, no obstacles, no RL prior and no memory augmentation. It does
not establish the original ICODE theory, stability, contraction or convergence;
nor does it establish cross-plant, odometry, perception, obstacle, real-robot or
RL-composed generalization.

## Next gate

The next experiment should vary physical parameters as an orthogonal blocked factor
without retraining: mass, friction, motor torque and command delay. Only after this
cross-plant gate should the project reintroduce obstacle perception and the bounded
RL sampling prior.

