# L95 half-budget controlled confirmation preregistration

Date: 2026-07-18

## Motivation

L94 passed the half-budget sample-efficiency Gate on 320 episodes, but those
episodes were executed in parallel.  Randomization protects the control
metrics, yet wall-clock planner timing can still be widened or distorted by
cross-process contention.  L95 is a new-seed, single-process confirmation of
only the primary comparison.

## Design

- learned L89 contextual bandit with `K=50`;
- strongest global fixed covariance with `K=100`;
- held-out hairpin and reverse-S;
- four MuJoCo physics domains;
- five new seeds `20270701--20270705`;
- 40 paired contexts, 80 total episodes;
- one process, no concurrent benchmark workers;
- pair order randomized with seed `2026071871`; arm order alternated within
  pair to avoid systematic warm-cache order.

ICODE, horizon, costs, reference, safety chain and checkpoints remain frozen.

## Primary Gate

Learned-50 minus fixed-100 must satisfy all four:

1. mean success difference >= 0 and collision difference <= 0;
2. cross-track RMSE 95% CI upper bound <= +2 mm;
3. elapsed-time 95% CI upper bound <= 0;
4. planner-compute 95% CI upper bound < 0.

Bootstrap resampling is route × physics context, then seed, with seed
`2026071872`.  Failure of any item prevents a controlled compute-efficiency
claim even though L94 remains a valid parallel closed-loop experiment.
