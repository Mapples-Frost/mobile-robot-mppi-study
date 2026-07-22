# L276 Recovery-Sequence Imitation Initialization Gate

## Purpose and method boundary

L275 established that recovery benefit requires a sustained sequence rather
than one isolated action. L276 tests the smallest coordination intervention:
initialize a copy of the frozen direct Actor to reproduce complete, successful
recovery sequences while distilling its original behavior on its source replay.

L276 is an initialization Gate, not a paper result and not a replacement of
reinforcement learning. If it passes, a later separately preregistered SAC run
may start from the initialized Actor and continue optimizing the frozen RL
objective. The formal method remains RL with a leakage-safe demonstration warm
start. L276 alone does not authorize SAC or final-map evaluation.

## Frozen data split

Only the 108 L268 accepted recovery chains are used. Their pre-existing split
is immutable:

- 72 train chains provide recovery imitation transitions;
- 18 validation chains are evaluated but never fitted;
- 18 test chains are evaluated once after all three model seeds finish.

Sampling is scene-then-chain balanced, so long chains do not dominate. The
source checkpoint's 6,000 replay observations supply an equal-sized distillation
anchor. Anchor targets are generated once by the frozen source Actor. Oracle
returns, privileged path fields, L258, final Hairpin/S-Chicane/Infinity maps,
and sealed seeds are forbidden.

## Frozen training

Three paired model seeds train independent copies of the same source Actor for
3,000 actor-only updates. Every batch contains 128 recovery examples and 128
source-replay anchor examples. The target is the bounded normalized `(v, omega)`
mean. Critics, target critics, entropy temperature, normalizer, reward, 69D
observation, ICODE, MPPI, HSS, fusion, and safety chain remain byte-identical.
The 1,000-update checkpoint is engineering/resume-only; the frozen evaluation
checkpoint is 3,000 updates.

## Frozen evaluation and Gate

For each seed, evaluate the source and initialized Actors on all validation and
test chain starts for the accepted-chain horizon. Also evaluate teacher-action
RMSE on all held-out transitions and source-replay action drift.

The initialization passes only if all checks hold:

1. all three seeds improve held-out test teacher-action RMSE;
2. median relative test RMSE improvement is at least 0.20;
3. median validation and test discounted-return gain over the source Actor is
   positive;
4. test corridor-reentry fraction improves by at least 0.15;
5. at least four of six scenes improve test return;
6. source-replay mean absolute action drift is at most 0.10;
7. collision or boundary failure does not increase;
8. critics, target critics, alpha, and normalizer hashes remain unchanged and
   every output is finite.

No seed, checkpoint, threshold, chain, or scene may be selected after outcomes
are viewed. Failure stops before SAC and is retained as raw negative evidence.
Success authorizes only preregistration of a three-seed, 3k/6k small-budget SAC
probe initialized from the three corresponding L276 Actors.

