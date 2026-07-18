# L67 residual-structure observation robustness preregistration

Date: 2026-07-16

## Primary question

With the physical MuJoCo plant fixed, do the frozen parameter-matched MLP and
ICODE predictors retain closed-loop improvements over nominal MPPI under 100 ms
state-observation latency and bounded noise-plus-latency? Does ICODE retain its
advantage over the direct MLP?

## Design

- Three independent matched MLP/ICODE training blocks.
- Two paths: training-family chicane and unseen reverse-S.
- Primary observation domains: clean ground truth, 100 ms delayed ground truth,
  and moderate noise plus 100 ms delay.
- Three fresh development execution seeds.
- Nominal, MLP and ICODE conditions; RL and memory disabled.
- 216 total development episodes.

The pure-noise factor is removed after two nominal-only calibration attempts
failed the 3% measurement-resolution floor. Raw wheel odometry remains a
clearly labeled non-gating stress domain because its nominal success was
0--16.7%, making it a state-estimation failure rather than a clean residual
comparison.

## Gate

Both learned models must improve nominal on every primary model block with a
hierarchical interval above zero and no success, collision or completion
regression. ICODE must beat MLP in all three primary domains, in both shifted
domains, in at least two model blocks, and on the unseen path with interval
lower bounds above zero. Raw wheel odometry is required only not to add
collisions; its tracking and success results are reported descriptively.

Passing unlocks the five untouched seeds 21860861--21860865 under an unchanged
sealed confirmation. Failure keeps those seeds sealed and moves the next work
to state-estimation correction rather than residual or RL tuning.
