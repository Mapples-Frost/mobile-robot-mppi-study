# L160--L162: online anytime ICODE-MPPI results

Date: 2026-07-18
Status: completed development study; the online STOP/ADD budget policy is **not retained**.

## Question

The anytime controller first evaluates a nested common-random-number `K=50`
rollout set and then uses a frozen contextual bandit to choose either:

- `STOP`: use the first 50 samples; or
- `ADD`: evaluate 50 more samples and use `K=100`.

This tests whether sample budget can be allocated online without losing the
benefit of ICODE dynamics.

## Implementation finding

The initial smoke run found a real deployment defect: the linear contextual
bandit recomputed a 20 x 20 matrix inverse on every control step. A deployment
cache for the inverse and parameter vector was added and is invalidated after
every model update. Unit tests cover cache reuse and invalidation. This removes
an accidental CPU/BLAS overhead without changing the learned decision rule.

## L105: one-step decisions

The blocked study completed 72/72 episodes with 100% success and zero
collisions. The anytime controller used mean `K=76.06` and selected `ADD` on
52.1% of control steps.

Relative to fixed ICODE contextual `K=100`:

- cross-track RMSE: +0.550 mm, 95% CI [-0.542, +1.656] mm;
- elapsed time: -0.175 s, 95% CI [-0.371, +0.029] s;
- mean planner compute: -0.778 ms, 95% CI [-2.597, +0.809] ms;
- issued jerk: +0.00995;
- applied jerk: +0.00793.

Tracking noninferiority passed, but compute superiority did not because its
confidence interval crossed zero. The primary gate failed.

## L106: three-step latched decisions

The latched controller also completed 72/72 episodes with zero collisions. It
used mean `K=77.91`; the observed decision-refresh fraction was 0.335, matching
the three-step contract.

Relative to fixed ICODE contextual `K=100`:

- cross-track RMSE: +2.691 mm, 95% CI [+1.766, +3.621] mm;
- elapsed time: -0.292 s, 95% CI [-0.525, -0.050] s;
- mean planner compute: +0.240 ms, 95% CI [-0.908, +1.802] ms.

The latch reduced switching frequency but violated the tracking margin and did
not establish compute superiority. The primary gate failed.

## Decision

The anytime optimizer remains an opt-in research ablation and a useful future
efficiency interface. It is not part of the retained final controller. The
evidence says that conditional budget allocation is feasible, but the current
whole-controller timing and tracking effects are not robust enough to support a
paper claim.

Raw outputs are under:

- `results/research_platform/rl/l105_latched_anytime_development_20260718_v1/`
- `results/research_platform/rl/l106_latched_anytime_development_20260718_v1/`
