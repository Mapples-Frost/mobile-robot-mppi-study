# Table 13: Gate 3 reliability calibration and fixed-budget HSS

## Independent graded reliability stress

| Authority | Episodes | Mean H=10 normalized error |
|---|---:|---:|
| High | 3 | 0.0462 |
| Medium | 8 | 0.0595 |
| Low | 19 | 0.0776 |

Episode-level Spearman association: -0.878.

## Sealed confirmation, seeds 35--39

| Metric | Fixed 30% | Adaptive 0/30/60% | Favorable effect | Seed-cluster 95% CI |
|---|---:|---:|---:|---:|
| Final goal distance (m) | 2.1171 | 1.9726 | 0.1444 | [0.0639, 0.2578] |
| Control jerk | 0.1072 | 0.1051 | 0.00211 | [0.00022, 0.00355] |
| Stuck steps | 5.9667 | 5.5000 | 0.4667 | [-1.9000, 4.3333] |
| Planner time (ms) | 158.61 | 151.91 | 6.70 | [5.91, 7.49] |
| Collisions | 0 | 0 | 0 | [0, 0] |
| Success | 0 | 0 | 0 | [0, 0] |

All comparisons used exactly 100 rollouts and two MPPI iterations. Adaptive
low/non-low step fractions were 74.9%/25.1%.

## Boundary

The confirmed claim is progress and smoothness at fixed sampling budget. The
data do not support success-rate improvement or universal complex-navigation
improvement; the final-distance effect was strong in the clean task and
approximately neutral in `lab_complex`.

