# Independent initialization confirmation protocol

This is a prospective continuation of `icode_sac_observable_round.md`. The
development result selected one observable-context SAC K/H checkpoint on a
two-environment-seed development bank. This round tests whether the result
survives independent SAC initialization before any reserved final test.

## Frozen treatment

Train four ICODE arms: joint K/H, H-only, K-only, and joint K/H with dynamics
features masked. Each arm uses two independent SAC initialization seeds
`9091301` and `9091302`, 60,000 physical cycles, 15,000-cycle checkpoints,
the same scene-balanced replay, and the exact planner, 13 kg plant override,
observable history, reward price, hold length, and SAC settings from the
observable round. The environment seed bases are `9094000` and `9094100`;
their realized ranges overlap. These are independent SAC initializations with
partly shared training environments, not independent training datasets.
Training is serial and all source/spec snapshots are retained.

The final checkpoint by cycle count is the only checkpoint evaluated in this
round. No outcome-based checkpoint selection is performed. Fixed comparators
are fixed nominal and fixed ICODE at `K256/H24`; the progress-feedback
heuristic is retained. All methods share the same 24 family/speed contexts and
paired evaluation environment seeds `9194201` and `9194202`.

Pre-evaluation amendment, 2026-09-09 08:20 CST: the initially specified
`9094201`/`9094202` fall within the incrementing training ranges. No evaluation
episodes had run. The waiting evaluator was stopped and changed to the fresh
seeds above, with a guard checking every recorded training seed before loading
the evaluation schedule. No existing training data were changed. Retain both
fixed K64/H24 and K256/H24 for each model, as specified in the evaluator before
any evaluation outcomes; there are 13 methods and 624 paired episodes total.

The reserved final environment seeds `9090301`–`9090304` are untouched. They
may be used only after the independent round and its audit freeze the method,
checkpoint rule, and comparator set. This round is still development evidence;
it is not a confirmatory ICRA claim.

## Primary descriptive outputs

Report success, collisions, task Q, measured computation per episode, deadline
misses, and stage cost. Keep each initialization seed visible and use paired
family/speed summaries. Any final-seed test must be a new immutable artifact.
