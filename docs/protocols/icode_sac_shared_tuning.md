# Shared ICODE-MPPI backbone development, prospective amendment

2026-09-08. Prepared after the completed 288-episode ICODE/nominal model screen,
before any outcomes from this amendment. Parent protocol: icode_sac_compute.md.
Motivation: several moderate-speed obstacle contexts remain all-failure across
the original K/H bank. A budget learner cannot compensate arbitrarily for a
poor shared sampling/cost configuration. Preserve the current four SAC arms,
their development evaluation, and all old results.

Execution is serial, after the first four-arm SAC queue. No source used by the
ongoing frozen run is changed. No final test seeds are used in this amendment.

## A: Shared parameter factorial

All 12 families and both speeds, paired seed 9091101, ICODE prediction, fixed
K64/H24, eight configurations (192 episodes):

- MPPI temperature: 2.5 or 10.
- Exploration noise sigma: [0.12, 0.50] or [0.20, 0.90].
- Soft obstacle influence distance: 0.70 or 0.45 metres.

The incumbent is (2.5, narrow noise, 0.70). Collision penalty, robot radius,
MuJoCo geometry/plant, safety layer, latency physics, sensing, duration, and Q
remain the same. Lower soft influence is an objective change for all methods,
not a change in geometric clearance or the collision/safety definitions.
All components of original running cost are still reported, but running-cost
totals with different influence settings are not directly comparable as one
unchanged estimand. Primary comparison across parameter settings is Q plus
success, collision, duration and clearance.

Order is randomized within each family/speed pair, using seed 9091100. Raw
cycles, resolved configuration, sources, ICODE checkpoint hashes and measured
readiness are retained. No performance-dependent stopping within the factorial.

## B: Frozen candidate budget/model check

After A completes, retain the incumbent and up to two distinct tuned candidates.
To qualify, a tuned candidate cannot have more aggregate collisions than the
incumbent on A. Rank eligible candidates by success descending, collisions
ascending, Q ascending, then compute ascending. This is a development selection
rule, not evidence of collision noninferiority. Preserve all excluded outcomes.

For each retained candidate: both nominal and ICODE, three budgets
(16,20),(64,24),(128,32), all 24 contexts, fresh paired development seed9091102.
Up to 432 episodes. The same candidate parameters apply to both models and
every later fixed/learned comparator. Candidate selection from B, if any, must
be frozen before new SAC training and requires another named training round.

The purpose is to discover a stronger common backbone. No assumption that
joint adaptation, ICODE or lower influence must win. If a cheap fixed controller
dominates, report it and investigate other mechanisms in a subsequent named
round rather than hide the comparator.
