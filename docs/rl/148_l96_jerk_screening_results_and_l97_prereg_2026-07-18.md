# L96 jerk screen results and L97 independent confirmation preregistration

Date: 2026-07-18

## L96 integrity

- 96/96 episodes and 24/24 randomized complete blocks;
- 96 unique route x physics x seed x treatment keys;
- each treatment appeared six times in each of four run positions;
- 96/96 successes and zero collisions;
- all preregistered metrics finite;
- all three candidate contrasts, including failed gates, are retained.

## L96 frozen selection result

| Candidate minus current | Issued jerk delta, 95% CI | Applied jerk delta, 95% CI | RMSE delta, 95% CI | Time delta, 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| hard yaw slew | -0.00935 [-0.01127, -0.00749] | -0.00603 [-0.00820, -0.00421] | +0.396 mm [-0.503, +1.364] | -0.017 s [-0.067, +0.038] | pass |
| stronger rate cost | +0.00009 [-0.00056, +0.00081] | +0.00002 [-0.00045, +0.00057] | -0.090 mm [-1.185, +0.849] | -0.175 s [-0.329, -0.004] | fail jerk |
| combined | -0.00941 [-0.01069, -0.00812] | -0.00608 [-0.00818, -0.00435] | -0.069 mm [-1.323, +0.996] | -0.150 s [-0.279, -0.008] | pass, selected |

The result identifies the hard yaw-command slew bound as the operative factor.
Increasing the predictive control-rate cost alone did not reduce either jerk
metric. The preregistered lexicographic rule selected `combined`; this is a
selection result, not an independent claim.

## L97 frozen design

L97 uses six untouched seeds `20270901--20270906`. The independent unit remains
one complete episode. Route x physics x seed forms 48 blocks; every block
receives all three arms. Block order is seeded and randomized, and cyclic arm
order makes every arm occupy every run position exactly 16 times.

Arms:

1. `fixed_k100`: strongest global fixed covariance, K=100, current smoothing;
2. `contextual_k50_current`: frozen L89 bandit, K=50, current smoothing;
3. `contextual_k50_smoothed`: frozen L89 bandit, K=50, L96-selected combined smoothing.

All other ICODE, MPPI, reference, plant-domain and safety settings remain frozen.
Execution is single-process; no concurrent benchmark process is allowed.

## L97 preregistered contrasts

All intervals use a hierarchical route x physics then seed bootstrap.

### A. Raw contextual K50 minus fixed K100 replication

- success delta >= 0 and collision delta <= 0;
- RMSE upper 95% bound <= +2 mm;
- elapsed-time upper bound <= 0;
- planner-compute upper bound < 0.

### B. Smoothed contextual K50 minus raw contextual K50 mechanism

- safety unchanged or improved;
- RMSE upper bound <= +2 mm;
- elapsed-time upper bound <= +0.5 s;
- issued-jerk and applied-jerk upper bounds both < 0.

### C. Smoothed contextual K50 minus fixed K100 final package

- all four half-budget clauses from contrast A;
- issued-jerk and applied-jerk upper bounds both <= 0.

The primary L97 gate requires A, B and C all to pass. Failure of any clause is
reported as failure; no seed, margin or candidate will be amended after opening
the L97 results.
