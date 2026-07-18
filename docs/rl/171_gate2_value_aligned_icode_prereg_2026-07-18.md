# Gate 2 preregistration: value-aligned ICODE

Date frozen: 2026-07-18

## Question

Does a critic-informed loss improve the control-relevant prediction of a
validated ICODE model beyond the improvement obtained by ordinary
task-specific fine-tuning on exactly the same Actor-generated data?

The SAC Actor and target twin critics are frozen. Only the ICODE residual
parameters receive gradients. The learned model is not updated online.

## Development screen and separation

L177--L182 were implementation and finite-weight development runs from a dirty
working tree. They are not paper evidence. The fixed screen compared
`lambda_value` in `{0, 0.05, 0.25, 1, 5}` with all other settings unchanged.
The validation trend was monotonic through 5 while the rollout constraint
remained satisfied. Therefore `lambda_value=5` is frozen for L185.

L183 recollects all train/validation/test/unseen episodes from a committed
source revision and new episode seeds. L184 is the mandatory
`lambda_value=0` task-fine-tuning control. L185 changes only
`lambda_value=5`.

No further value-weight selection is permitted from L183 outcomes.

## Loss and gradient boundary

The proposed objective is

```text
base derivative + one-step + multi-step loss
+ lambda_value * confidence-weighted critic consistency
+ lambda_anchor * residual checkpoint anchor
```

For every true and predicted state, the same recorded goal, previous command,
safety flag and LaserScan sector context are used. Only the physical
state-dependent observation features are reconstructed. True-state critic
values are stop-gradient targets. Actor and critic parameters are frozen and
must have no gradients.

## Offline design

- Independent data split unit: complete episode.
- Seen and unseen physics roles are recorded in every transition.
- L184 and L185 use identical L183 data, initialization, optimizer, epoch
  schedule and random order.
- Model selection uses validation terminal-value RMSE subject to no more than
  3% validation rollout-RMSE degradation relative to the starting ICODE.
- Test and unseen splits are evaluated only after the weight is frozen.

Primary offline contrast:

```text
L185 value-aligned ICODE - L184 ordinary task fine-tuning
```

Primary outcomes:

- terminal critic-value RMSE;
- H=10 rollout RMSE;
- terminal position and heading RMSE;
- candidate terminal-value rank correlation.

The offline mechanism passes if L185 improves terminal-value RMSE over L184 on
both test and unseen splits, while remaining within the 3% rollout constraint.

## Closed-loop development confirmation

If the offline mechanism passes, L184 and L185 are separately inserted into
the frozen Gate 1 Simple Combination controller. Both use:

- the same L175 Actor/critic;
- two scenes;
- nominal-seen, long-delay-seen and combined-unseen physics;
- episode seeds 21, 22 and 23;
- 100 total prediction rollouts and two MPPI iterations;
- memory disabled;
- unchanged LaserScan, local-obstacle, scan_guard and safety chain.

Primary paired outcomes are final goal distance and control jerk. Success and
collision are required safety outcomes. A positive closed-loop development
result requires L185 to improve at least one primary continuous outcome over
L184 without worsening the other, success or collision.

## Scope

This Gate can establish a critic-informed fine-tuning mechanism. One Actor,
one ICODE initialization and three closed-loop simulation seeds cannot
establish a universal synergy or a final paper-level significance claim.
