# L199 path-policy-consistent value-ICODE preregistration

Date frozen: 2026-07-19
Status: frozen after L198 failed and before L199 data collection

## Motivation

L198 was safe and successful in all 30 Full Proposed episodes and improved
over the fixed value-aligned simple combination, but missed the pooled ordinary
baseline by 1.07%. Audit found a generation mismatch: the L192 value-aligned
ICODE ensemble was trained with the old 48-dimensional point-goal L175 critic,
while online guidance used the 54-dimensional, path-conditioned L185 policy.

L199 repairs this scientific mismatch rather than tuning the L198 Gate.

## Method change

1. collect a new transition dataset with the frozen L185 policy and its
   path-context observation;
2. recompute predicted cross-track, heading-error, and remaining-path critic
   features through a differentiable local tangent-frame approximation;
3. keep curvature and path-validity as recorded context;
4. fine-tune each of the three frozen ordinary ICODE members against the
   matching L185 target critic;
5. keep all Actor, MPPI, reliability, safety, and compute settings unchanged.

This makes the value-consistency objective correspond to the policy/value
actually used online. The local path approximation is valid only around each
recorded transition and is not claimed to be a global differentiable path
projection.

## Data and split

The dataset uses disjoint route families:

- train: acceleration straight and turn;
- validation: sweep;
- test: chicane;
- unseen: reverse-S and hairpin in the unseen physics role.

Splits are by complete scene--physics environment and episode. Normalization
comes from the frozen RL checkpoint and base ICODE checkpoints; no L198
holdout trajectory is used for training.

## Offline Gate

All three members must:

1. train without NaN/Inf and save complete checkpoints;
2. use the L185 checkpoint hash and 54-dimensional observation;
3. exercise nonzero path-feature gradients in unit tests;
4. satisfy the existing maximum 3% rollout-degradation constraint;
5. not worsen the configured value-consistency validation criterion.

Only after this Gate passes may a fresh closed-loop development factorial be
run. L198 remains a failed sealed confirmation and is never rerun or relabeled.
