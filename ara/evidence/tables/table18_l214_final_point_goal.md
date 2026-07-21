# Table 18: L214 final seven-arm point-goal benchmark

## Design

- Formal seeds: 101--110 (10 independent seed clusters)
- Repeated strata: nominal seen, long-delay seen, combined unseen
- Complete blocks: 30
- Episodes: 210
- Qualification rows: 0
- Bootstrap: 10,000 seed-cluster resamples
- Progress SHA256: `d9c24e52286a438f6719d0839cf7152383315bd5f9c41a964d1c2d0e8546bd46`

## Overall descriptives

| Method | Success | Final distance | Jerk | Planner ms | Collisions |
|---|---:|---:|---:|---:|---:|
| Traditional MPPI | 0.6333 | 0.3883 | 0.06419 | 4.51 | 0 |
| ICODE-MPPI | 0.3333 | 0.4312 | 0.05716 | 85.96 | 0 |
| RL-driven MPPI | 0.4667 | 1.2528 | 0.09058 | 27.19 | 0 |
| Simple combination | 0.3000 | 1.9098 | 0.08986 | 166.84 | 0 |
| Value fixed | 0.3333 | 1.8314 | 0.09023 | 170.11 | 0 |
| Ordinary adaptive | 0.4333 | 1.4901 | 0.08397 | 160.42 | 0 |
| Full proposed | 0.5000 | 1.3815 | 0.08321 | 159.78 | 0 |

## Confirmatory mechanism contrasts

Positive values are favorable.

| Contrast | Success effect [95% CI] | Distance effect [95% CI] | Jerk effect [95% CI] | Compute effect [95% CI] |
|---|---:|---:|---:|---:|
| Value vs simple | 0.0333 [-0.1667, 0.2333] | 0.0784 [-0.3830, 0.5337] | -0.000365 [-0.001225, 0.000655] | -3.263 [-8.120, 1.708] |
| HSS vs simple | 0.1333 [-0.0667, 0.3333] | **0.4198 [0.0270, 0.7972]** | **0.005894 [0.003508, 0.008420]** | **6.428 [0.189, 12.135]** |
| Full vs simple | 0.2000 [-0.0333, 0.4333] | 0.5283 [-0.0335, 1.0968] | **0.006651 [0.004913, 0.008416]** | **7.065 [2.286, 11.157]** |

## Epistemic status

- Confirmed in L214: adaptive HSS improves distance, jerk and compute at fixed rollout budget.
- Not confirmed in L214: value-aligned ICODE main effect on closed-loop success/distance.
- Not supported: Full proposed globally outperforms Traditional MPPI.
- Descriptive signal requiring independent confirmation: Full proposed reaches 0.80 success in combined-unseen dynamics.
- Complex-obstacle navigation is not tested by L214; prior lab-complex supplement had zero successes in every arm.
