# L278 Recovery-Retention Diagnosis

## Question

Did unanchored SAC in L277 erase the recovery-sequence behavior established by
L276, and if so did the loss appear after the first Actor update at step 3,000
or accumulate through step 6,000?

This is an evaluation-only causal diagnosis. It performs no training and does
not authorize changing reward, observation, Critic, planner, fusion, or safety.

## Frozen inputs

- all three L277 paired runs;
- exactly `initial`, `step_000003000`, and `step_000006000` checkpoints;
- the immutable L268 validation/test recovery chains only;
- the same deterministic Windows MuJoCo rollout semantics as L276;
- teacher-action RMSE, accepted-chain-horizon return, corridor reentry,
  collision, and boundary metrics.

L268 train chains, L258, sealed seeds, and final Hairpin/S-Chicane/Infinity maps
are forbidden. No checkpoint, chain, scene, threshold, or seed may be selected
after outcomes are viewed.

## Frozen interpretation

For each paired seed, compare step 3k and 6k to its own initial checkpoint.
Recovery retention is considered adequate at 6k only if all hold:

1. median test teacher-action RMSE is no more than 20% above initial;
2. median paired test return loss is no worse than `0.5`;
3. median test reentry loss is no worse than `0.10`;
4. at least four of six scenes have nonnegative paired test-return change;
5. collision/boundary failures do not increase and all outputs are finite.

If retention fails, the only authorized next intervention is to preregister an
L279 recovery-retention Actor anchor during the same 3k/6k SAC budget. If
retention passes, the next diagnosis must target validation distribution and
scene interference instead. L278 never authorizes final-map evaluation.

