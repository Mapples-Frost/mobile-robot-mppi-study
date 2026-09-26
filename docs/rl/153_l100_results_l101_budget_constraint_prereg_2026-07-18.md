# L100 result and L101 compute-constrained preregistration

Date: 2026-07-18

## L100 retained result

L100 completed all 256 anchor branches from 32 independent MuJoCo episodes.
All nested-prefix hashes matched, all required features were finite and neither
K50 nor K100 counterfactual branches collided.

On the untouched L100 evaluation episodes:

- fixed K100 improved mean true branch cost over K50 by 3.68%;
- the hindsight Oracle used ADD50 in 52.34% of states (mean K=76.17) and
  improved mean true cost by 5.14%;
- the ridge policy improved mean true cost by 3.37%, with episode-level
  hierarchical 95% CI for raw cost delta `[-0.691, -0.176]`;
- it retained 65.5% of Oracle gain;
- but it used mean K=83.20, above the frozen K<=80 clause.

The L100 primary Gate therefore failed. The result is not reclassified as a
pass. It establishes selective, predictable headroom but shows that the
unconstrained decision threshold spends too much computation.

## L101 single method change

L101 retains the same state features, ridge penalty, routes, physics domains,
K50/K100 nesting, 1% Oracle label and all integrity rules. It changes only the
decision-threshold calibration:

```text
fit ridge benefit predictor on discovery episodes
        -> choose discovery prediction quantile yielding 50% ADD50
        -> freeze threshold
        -> evaluate on disjoint episode seeds
```

This is a compute constraint, not an outcome threshold fitted on evaluation
data. The expected discovery mean budget is K=75; evaluation mean K must still
be no greater than 80.

L101 uses new discovery seeds `20271301--20271303` and new evaluation seeds
`20271311--20271313`. L100 evaluation records and the sealed L99 seeds are not
used to fit or assess L101.

## Frozen L101 Gate

The existing L100 clauses remain unchanged:

1. complete finite collision-free data and exact nested prefixes;
2. Oracle ADD50 fraction in `[0.10, 0.75]`;
3. Oracle relative gain at least 1%;
4. ridge evaluation cost superiority over fixed K50 with 95% CI upper `< 0`;
5. ridge mean K no greater than 80;
6. ridge retention of at least 30% of Oracle relative gain.

Matched random, stump and fixed K100 remain fully reported secondary controls.
Even if L101 passes, it supports a constrained supervised budget predictor, not
yet an RL necessity claim. A sequential bandit/RL policy and closed-loop
evaluation remain separate subsequent Gates.
