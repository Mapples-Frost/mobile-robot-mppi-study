# Full Proposed 2x2 factorial preregistration

Date frozen: 2026-07-19

## Scope and relation to the frozen plan

This experiment implements the frozen plan's `Full Proposed` Gate after the
candidate-level conservative terminal mechanism failed independent
confirmation. The paper direction is unchanged:

- critic-informed, competence-gated value-aligned ICODE;
- reliability-calibrated persistent Actor Hybrid Sampling;
- fixed SAC target-critic terminal value;
- unchanged MPPI, LaserScan perception and safety chain.

The failed terminal-authority gate remains default-off and is not present in
any arm below.

## Factorial design

The two factors are:

1. residual model: ordinary ICODE ensemble versus competence-gated
   value-aligned ICODE ensemble;
2. persistent Actor sampling: fixed 30% versus reliability-adaptive 0/30/60%.

| Cell | Residual dynamics | Actor-guided sampling |
|---|---|---|
| `ordinary_fixed` | ordinary ICODE ensemble mean | fixed 30% |
| `value_fixed` | value-aligned ICODE ensemble mean | fixed 30% |
| `ordinary_adaptive` | ordinary ICODE ensemble mean | adaptive 0/30/60% |
| `full_proposed` | value-aligned ICODE ensemble mean | adaptive 0/30/60% |

All cells use the same frozen SAC Actor and target critic, terminal-value
weight, \(K=100\) model rollouts, two MPPI refinement iterations, horizon,
cost, control limits, LaserScan path and safety arbitration.

The ordinary and value-aligned ensembles each contain three independently
initialized members. Reliability thresholds are calibrated separately for the
two ensembles using only the existing validation split. Each mapping must pass
an offline held-out ordering Gate before its adaptive arm is admissible.

## Primary experiment: completion-resolved clean task

The clean single-obstacle task is used to isolate model shift and policy
guidance from difficult global planning. Episode length is fixed at 300 control
steps before any arm is evaluated. This limit was chosen from the physical
speed/distance envelope and earlier arm-blind 180-step truncation evidence, not
from the outcomes of the four new cells.

Physics strata:

- nominal seen;
- long-delay seen;
- combined unseen.

Development seeds are 48--50. If the development Gate passes without changing
the method, sealed confirmation seeds are 51--55.

Each seed-by-physics block contains all four arms in seeded randomized order.
Seed is the independent bootstrap cluster; physics domains are repeated
strata. Controller timesteps are never counted as independent samples.

Primary outcomes, in order:

1. success;
2. final goal distance;
3. collision;
4. control jerk;
5. planner time.

The factorial interaction for a cost \(J\) is:

\[
\Delta_{\mathrm{int}}
=
J_{\mathrm{full}}
-J_{\mathrm{value}}
-J_{\mathrm{adaptive}}
+J_{\mathrm{ordinary}}.
\]

A negative interaction is favorable for lower-is-better outcomes.

## Development Gate

Development passes when:

1. every cell uses exactly \(K=100\);
2. every adaptive arm exercises low and non-low authority;
3. `full_proposed` has no more collisions than `ordinary_fixed`;
4. `full_proposed` improves success or final distance;
5. at least one primary progress endpoint has a favorable seed-cluster 95%
   bootstrap interval;
6. neither value alignment nor adaptive HSS has an adverse effect in every
   seed cluster;
7. the result is not driven by one physics domain alone.

The interaction is reported whether favorable, null or adverse. Full Proposed
may pass as an additive composition even if strict super-additive interaction
is not demonstrated; in that case the paper must not claim synergy.

## Independent confirmation Gate

Without changing checkpoints, thresholds, weights, task horizon or metrics,
confirmation passes only if:

1. the `full_proposed` versus `ordinary_fixed` success/final-distance direction
   is favorable;
2. at least one of success or final distance has a strictly favorable
   seed-cluster interval;
3. collisions do not increase;
4. jerk does not worsen by more than the prospectively fixed 5% engineering
   non-inferiority margin;
5. all three physics-domain point estimates are reported;
6. the equal-budget contract remains exact.

## Supplementary complex-scene experiment

Only after the clean task is frozen, the same four arms are evaluated in
`lab_complex` at the existing 180-step horizon. This is a robustness
supplement, not a success-rate primary endpoint. Its role is to determine
whether the full package remains neutral or degrades when navigation geometry,
perception and safety arbitration dominate dynamics learning.

## Claim boundaries

Passing supports the complete fixed-budget method in the tested MuJoCo scope.
It does not establish universal super-additive synergy, formal uncertainty,
closed-loop stability, general complex-navigation success or real-robot
performance.
