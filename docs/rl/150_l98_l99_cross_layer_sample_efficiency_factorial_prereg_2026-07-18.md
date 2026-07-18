# L98/L99 ICODE--contextual-RL composition: preregistration

Date: 2026-07-18
Status: frozen before L98 execution; L99 remains sealed until L98 passes

## Purpose

This is not a route change.  It directly composes the two independently
validated modules already retained in the project:

1. ICODE residual dynamics improves MPPI prediction and closed-loop tracking;
2. the frozen route-context LinUCB policy selects MPPI sampling covariance and
   has repeatedly retained performance with half the candidate budget.

The experiment asks whether those contributions remain identifiable when both
modules are evaluated in one blocked factorial, rather than inferring the final
system from separate studies.

## Experimental unit, factors and controls

The independent unit is one complete closed-loop MuJoCo episode.  Controller
steps are repeated observations, not replicates.  Route x physics x seed is a
complete block and receives all six arms of a 2 x 3 factorial:

- prediction dynamics: nominal or frozen ICODE;
- sampler: strongest fixed covariance at K=50, contextual covariance at K=50,
  or strongest fixed covariance at K=100.

All arms share the same horizon, costs, safety chain, plant, command-delay
handling, L96-selected yaw slew bound and retained rate cost.  The contextual
policy sees only invariant reference-route geometry.  It receives no simulator
domain label, future state, plant parameter or obstacle ground truth.

Run order is a seeded randomized cyclic schedule.  L98 has 24 blocks and 144
episodes; every arm occurs four times in every run position.  L99 has 48 blocks
and 288 episodes; every arm occurs eight times in every run position.  Execution
is single-process so planner timing is not contaminated by concurrent workers.

## Frozen data domains

- held-out routes: hairpin and reverse-S, unused for LinUCB fitting;
- physics: training anchor, high friction, 100 ms delay and combined moderate;
- L98 development seeds: `20271001--20271003`;
- L99 untouched confirmation seeds: `20271101--20271106`.

No seed, route, plant, checkpoint, threshold or arm may change after the
corresponding phase is opened.

## Primary estimands and Gates

Intervals use a hierarchical bootstrap over route x physics context, then seed.
All safety clauses require nonnegative paired success change and nonpositive
paired collision change.

### A. ICODE contribution under contextual K50

`icode_contextual_k50 - nominal_contextual_k50` must satisfy:

- safety clauses;
- cross-track RMSE 95% CI upper bound strictly below zero;
- mean planner compute of `icode_contextual_k50` no greater than 50 ms.

### B. RL half-budget contribution inside ICODE

`icode_contextual_k50 - icode_fixed_k100` must satisfy:

- safety clauses;
- cross-track RMSE 95% CI upper bound at most +2 mm;
- elapsed-time 95% CI upper bound at most zero;
- planner-compute 95% CI upper bound strictly below zero.

Jerk remains a fully reported secondary outcome.  L97's failed strict jerk
superiority result is not redefined or erased by this study.

### C. Complete package versus conventional nominal K100

`icode_contextual_k50 - nominal_fixed_k100` must satisfy:

- safety clauses;
- cross-track RMSE 95% CI upper bound strictly below zero;
- absolute mean planner compute no greater than 50 ms.

This contrast tests accuracy superiority and real-time feasibility, not lower
total compute than nominal dynamics.  Learned residual inference has a measured
cost and candidate-count efficiency must not be mislabeled as universal CPU
superiority.

The primary phase Gate requires A, B and C.  Same-budget contextual-vs-fixed
effects, ICODE effects at the other sampler levels, final distance, clearance,
jerk, deadline misses and factorial difference-in-differences are secondary and
will be reported regardless of sign.

## Stopping rule and claim boundary

L99 opens only if L98 passes the frozen primary Gate and integrity audit.  If L98
fails, the failure is retained and L99 stays sealed.  If L99 passes, the supported
claim is modular composition on the tested clean path-tracking family: ICODE
improves prediction-dependent tracking while contextual RL preserves it with a
smaller MPPI candidate budget.  This does not prove super-additivity, arbitrary
OOD robustness, dynamic-obstacle prediction, real-robot transfer, stability,
contraction or convergence.
