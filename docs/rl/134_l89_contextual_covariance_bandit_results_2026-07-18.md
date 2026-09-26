# L89 contextual covariance bandit results

Date: 2026-07-18

## Decision

L89 passes the preregistered independent closed-loop gate.  This is the first
RL result in the current line of work that simultaneously demonstrates:

1. a deployable learned decision rather than a counterfactual oracle;
2. improvement over the strongest globally fixed covariance comparator;
3. held-out path-geometry transfer;
4. unchanged safety and non-inferior tracking precision;
5. no measurable planner-compute penalty after caching the frozen critic.

## Learned role

ICODE supplies the residual dynamics used by every rollout.  The LinUCB
contextual policy observes only invariant geometry of the route supplied to
MPPI and selects one of two exploration options:

- `narrow = [0.5, 0.5]`;
- `speed = [1.75, 0.75]`.

It selected `narrow` for the held-out hairpin and `speed` for the held-out
reverse-S.  It does not output wheel commands or bypass MPPI/scan_guard.

## Independent confirmation

The optimized confirmation contains 80 unique MuJoCo episodes:

- 40 learned-policy episodes and 40 paired fixed-comparator episodes;
- hairpin and reverse-S, neither used for policy fitting;
- four physics domains;
- five new seeds per geometry/domain/condition;
- 80/80 successes and 0/80 collisions.

Paired learned minus strongest-global-fixed results (hierarchical bootstrap,
geometry/physics context then seed) are:

| Metric | Mean delta | 95% CI | Gate |
|---|---:|---:|---|
| Time to goal | -1.435 s | [-2.488, -0.365] s | superiority pass |
| Cross-track RMSE | -0.652 mm | [-1.394, -0.022] mm | +2 mm non-inferiority pass |
| Control jerk | -0.00749 | [-0.01281, -0.00202] | improved |
| Planner compute mean | -0.175 ms | [-1.574, +1.081] ms | no detectable overhead |
| Success rate | 0 delta | -- | pass |
| Collision rate | 0 delta | -- | pass |

The complete primary gate is `true`.

## Engineering remediation

The first independent run already passed the control gate but exposed a
+30.21 ms planner overhead because the frozen linear critic recomputed two
matrix inverses at every control step.  The implementation now invalidates
the cache only when a bandit update occurs; deployment is frozen, so all later
steps reuse the inverse and coefficient vector.  Cached inference measured
approximately 14 microseconds per call.  A second complete 80-episode run
reproduced the control metrics exactly and reduced the paired compute delta to
-0.175 ms (CI includes zero).

## Claim boundary

The evidence supports context-dependent RL selection of MPPI exploration
covariance on clean static path-tracking tasks under several MuJoCo physics
domains.  It does not yet establish performance with dynamic obstacles,
online physical-system exploration, or a real robot.  LinUCB is a contextual
bandit (a one-step RL formulation), not a general long-horizon policy, and the
large confidence width on the hairpin must be retained as an OOD warning
rather than described as calibrated probability.

## Artifacts

- training checkpoint and summaries:
  `results/research_platform/rl/l89_contextual_covariance_bandit_20260718_v1`;
- first independent confirmation (pre-cache):
  `results/research_platform/rl/l89_contextual_covariance_independent_eval_20260718_v1`;
- optimized independent confirmation:
  `results/research_platform/rl/l89_contextual_covariance_independent_eval_cached_20260718_v2`.

