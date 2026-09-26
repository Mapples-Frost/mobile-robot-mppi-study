# L192 continuous reliability final development confirmation

Date frozen: 2026-07-19
Status: frozen before collecting any L192 transition
Parents: L190 and L191

## Why one final measurement amendment is necessary

L191 prospectively reproduced strong continuous authority--error ordering for
both ordinary and value-aligned ensembles:

- test Spearman correlations were approximately -0.94;
- unseen-development correlations were approximately -0.85;
- low-authority mean error exceeded high-authority error by 0.08--0.10;
- all error strata were directionally monotone.

The formal L191 Gate still failed because one or two test episodes occupied the
middle fixed runtime bin.  Middle-bin occupancy is an operational discretization
property, not the scientific claim that reliability ranks model error.  It can
fail when a useful continuous score is bimodal.

L191 remains a failed Gate.  Its test and unseen rows are development evidence
and are not reused here.

## Frozen confirmatory estimand

The independent unit is the episode.  For each split and ensemble, L192 reports:

1. episode-level Spearman correlation between authority and rollout error;
2. a 95% episode bootstrap interval for that correlation;
3. relative error separation between the lowest- and highest-authority
   quartiles.

The continuous Gate passes only if all three splits satisfy:

- at least 24 independent episodes;
- Spearman correlation no greater than -0.50;
- the 95% bootstrap upper bound is below zero;
- low-authority quartile error exceeds high-authority quartile error by at
  least 20% relative to the low-authority error.

Bootstrap resampling uses complete episodes, 5,000 draws, and a frozen seed.
Overlapping rollout windows are never resampled as independent observations.

Runtime low/medium/high fractions remain unchanged.  Their counts are reported
as diagnostics but are no longer treated as a statistical validity condition.

## Fresh data and stopping rule

L192 uses new seed bases 22845000--22847000 and three episodes per
scene/domain environment:

- validation: 24 episodes on straight/turn;
- test: 24 episodes on sweep/chicane;
- unseen-development: 30 episodes on reverse-S/hairpin.

Scene/domain splits remain disjoint.  L186 geometries and seeds 561--565 remain
sealed.

This is the last admissible offline reliability redesign in this experiment
family.  Both ordinary and value-aligned ensembles must pass.  If either fails,
adaptive reliability is removed from the retained path method and reported as
a negative ablation.  If both pass, only a closed-loop development factorial
is authorized; sealed confirmation still requires that subsequent Gate.
