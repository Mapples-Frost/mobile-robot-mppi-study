# L94 contextual covariance sample-efficiency preregistration

Date: 2026-07-18

## Hypothesis

L89 independently showed that a frozen route-geometry contextual bandit beats
the strongest global fixed covariance at `K=100`.  L94 tests a stronger and
more practically relevant claim: better-directed exploration can preserve
closed-loop performance with fewer MPPI samples.

## Frozen components

- the L89 contextual-bandit checkpoint and its two actions;
- the L57 ICODE residual checkpoint;
- route reference, MPPI horizon/cost, actuator limits and safety chain;
- held-out hairpin and reverse-S geometries;
- four MuJoCo physics domains used in L89 independent confirmation.

Only `planner.num_samples` changes: `K in {50,100,200,400}`.  The learned
policy is not refit for any K.

## Design

- 2 held-out routes;
- 4 physics domains;
- 4 sample counts;
- learned contextual bandit and strongest global fixed comparator;
- 5 new seeds `20270601--20270605`;
- total `2 * 4 * 4 * 2 * 5 = 320` MuJoCo episodes.

All conditions are paired by route, physics domain and seed.  The schedule is
randomized by seed `2026071861`, resumable and shardable.

## Primary sample-efficiency comparison

Compare `learned K=50` minus `fixed K=100`.  The Gate requires:

1. nonnegative mean success difference and nonpositive mean collision
   difference;
2. cross-track RMSE 95% CI upper bound no greater than `+2 mm`;
3. elapsed-time 95% CI upper bound no greater than zero;
4. planner-compute 95% CI upper bound strictly below zero.

The hierarchical bootstrap resamples route × physics context first, then seed,
using seed `2026071862`.  Failures retain their elapsed episode budget.

## Secondary analyses

At each K, report learned minus fixed deltas for success, collision,
cross-track RMSE, elapsed time, jerk and planner compute.  Also report the full
K scaling curve.  Same-budget results are descriptive and cannot rescue a
failed primary half-budget Gate.

## Claim boundary

A pass would support “route-context RL reduces the MPPI sampling needed for
the tested tracking tasks”, not a universal sample-complexity theorem.  It
would not address dynamic-obstacle prediction or real-robot deployment.
