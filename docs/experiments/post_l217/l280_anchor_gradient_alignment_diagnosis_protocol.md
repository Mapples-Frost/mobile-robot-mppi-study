# L280 Anchor Gradient-Alignment Diagnosis

## Question

Did L279 fail because the recovery/source retention gradients conflict with
the online SAC Actor gradient, particularly in the steering mean head, or
because the two anchor halves conflict with each other?

L280 is evaluation-only and performs no optimizer step.  It uses the frozen
L279 step-3k and step-6k checkpoints for all three paired seeds, their own
replay buffers, and the immutable L279 train anchor shard.  Validation/test
anchor rows, MuJoCo outcomes, final maps, L258, and sealed seeds are excluded.

## Fixed batches and gradients

For each seed, stage, and each of six replay scene groups:

- draw one fixed 256-row scene batch with a preregistered diagnostic seed;
- draw 256 recovery-anchor and 256 source-distillation rows independently;
- compute, without parameter updates, the SAC Actor gradient, recovery-anchor
  gradient, source-anchor gradient, and an equal 50/50 combined-anchor gradient;
- reuse a fixed Torch RNG seed for the stochastic SAC action sample.

Report cosine similarity and gradient-norm ratios for:

1. the shared two-layer trunk;
2. linear-speed mean output row;
3. angular-speed mean output row;
4. log-standard-deviation output rows;
5. all Actor parameters.

The quantile Critic aggregation, entropy term, anchor loss, normalizer, and
network are exactly those stored in the checkpoint.  No oracle return enters
the gradients.

## Frozen interpretation

- `online_vs_anchor_conflict`: at least four of six scenes show negative
  SAC-vs-combined-anchor cosine in the same mean-head component for at least
  two of three seeds, with median anchor/SAC norm ratio at least one.
- `internal_anchor_conflict`: recovery-vs-source cosine is negative in the
  same component under the same coverage rule.
- `anchor_dominance_without_directional_conflict`: cosine is not broadly
  negative, but the combined anchor median norm is at least four times the SAC
  norm in trunk or a mean head.
- otherwise: `gradient_conflict_not_supported`.

These thresholds, parameter partitions, batches, and stages are frozen before
reading L280 results.  A positive diagnosis authorizes only preregistration of
the corresponding minimal paired intervention (gradient projection for online
conflict, separated adapters for internal conflict, or normalized anchor scale
for pure dominance).  It does not authorize Actor training or final maps.

