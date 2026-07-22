# L279 Recovery-Retention Anchor SAC Protocol

## Question and design

Does an ongoing, leakage-safe recovery-sequence anchor prevent the gradual
recovery-policy forgetting identified by L278 while preserving the favorable
direction of the L277 online SAC probe?

L279 is a paired single-variable intervention.  It reuses the three L277
training seeds, the corresponding frozen L276 Actor initializations, the same
six training scenes, three validation scenes, update schedule, fresh replay,
and 6,000-step budget.  The only treatment difference is a supervised Actor
anchor during the same SAC Actor updates beginning at step 3,000.  The completed
L277 runs are the concurrent control; no L277 artifact is overwritten.

## Frozen retention dataset

The anchor is materialized before training with its own fail-closed schema.
Only the 72 L268 **train-split** recovery chains and the frozen L262 source
replay may enter the training shard.  Each recovery chain contributes exactly
128 seeded samples with replacement.  An equal number of source-replay
observations, labelled by the frozen L262 Actor, provides the original-policy
distillation half.  Thus the train shard is exactly 9,216 recovery plus 9,216
source rows; uniform minibatches preserve the preregistered 50/50 mixture in
expectation without adaptive sampling.

L268 validation/test recovery chains are physically separate evaluation shards
and never sampled by the anchor.  Oracle returns, simulator truth, privileged
path fields, L258, sealed seeds, and final Hairpin/S-Chicane/Infinity artifacts
are forbidden.  The manifest locks source hashes, split membership, shard
hashes, teacher provenance, and the exact field allowlist.

## Fixed optimization

- paired seeds: 20263311, 20263312, 20263313;
- 6,000 environment steps; checkpoints/validation at 0, 3k, and 6k;
- Actor updates begin at step 3,000 exactly as in L277;
- anchor batch 256, mean-MSE weight 2.0;
- log-standard-deviation weight 0.001, target -2.0;
- reward, 69D observation, network, Critic targets, alpha, replay semantics,
  ICODE, MPPI, HSS, Actor/Traditional fusion, and safety remain frozen.

The dataset and all Gate thresholds are frozen before any L279 outcome is
viewed.  Seeds, checkpoints, chains, scenes, and thresholds cannot be selected
afterward.

## Frozen Gate

Engineering and provenance must be complete, finite, CUDA-native, and exactly
resumable.  Every seed must record the frozen anchor manifest and exactly 3,001
anchored Actor updates by step 6,000.

Recovery retention is required on the untouched L268 validation/test chains:

1. median test teacher-action RMSE at 6k is no more than 20% above initial;
2. relative to paired L277 controls, median test RMSE improves by at least 20%;
3. test return loss is at most 0.5 and reentry loss at most 0.10;
4. at least four of six test scenes have nonnegative return change;
5. collision/boundary failures do not increase.

The original L277 validation Gate is also repeated unchanged: no per-seed
collision or success regression, completion regression no worse than 0.02,
nonnegative median completion, median CTE improvement at least 0.05 or goal
distance improvement at least 0.10, at least two of three seed directions, and
at least two of three validation scenes improving in CTE.

If every check passes, L279 authorizes only preregistration of a larger
independent-seed validation.  Failure stops expansion and preserves raw data
plus a concise Gate status.  L279 never authorizes selecting or tuning on the
final three maps.

