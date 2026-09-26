# L96 contextual covariance jerk remediation: preregistered factorial screen

Date: 2026-07-18  
Status: frozen before executing the L96 selection seeds

## Question

Can the L95 contextual-bandit controller retain its half-budget behavior while
removing both issued-command and physically applied jerk increases?

## Experimental unit and blocking

The independent unit is one complete closed-loop MuJoCo episode. Controller
steps inside an episode are repeated measurements, not independent replicates.
Every route x physics-domain x seed block receives all four treatments. Block
order is seeded and randomized. Within blocks, treatment order follows a
randomized cyclic schedule, so each treatment occupies each run position
equally often over the 24 blocks.

- held-out routes: hairpin and reverse-S;
- physics blocks: anchor, high friction, long delay and combined moderate;
- selection seeds: `20270801--20270803`;
- 24 complete blocks, four arms, 96 total episodes;
- frozen L89 contextual-bandit checkpoint;
- frozen ICODE checkpoint, horizon, costs, safety chain and `K=50`.

## Full 2 x 2 factorial

| Arm | yaw command-rate limit | MPPI control-rate weight |
|---|---:|---:|
| current | 3.5 rad/s^2 | 0.08 |
| hard_yaw_slew | 2.5 rad/s^2 | 0.08 |
| stronger_rate_cost | 3.5 rad/s^2 | 0.16 |
| combined | 2.5 rad/s^2 | 0.16 |

The translational command-rate limit remains 1.10 m/s^2. This design separates
a hard actuator-facing bound from a predictive MPPI cost and can detect their
interaction.

## Frozen candidate gate

Each candidate is paired against `current`. It is eligible only if all clauses
hold:

1. mean success difference is nonnegative and collision difference nonpositive;
2. cross-track RMSE 95% CI upper bound is at most +2 mm;
3. elapsed-time 95% CI upper bound is at most +0.5 s;
4. issued control-jerk 95% CI upper bound is strictly below zero;
5. applied control-jerk 95% CI upper bound is strictly below zero.

Confidence intervals use a hierarchical bootstrap over route x physics blocks,
then seed, with seed `2026071882`. If multiple candidates pass, select the one
with the lowest mean applied jerk; ties use issued jerk, elapsed time and arm
name in that order. If none pass, L97 remains sealed and the jerk-remediation
claim fails.

L96 is selection, not independent confirmation. No L96 result alone will be
presented as a final paper claim.
