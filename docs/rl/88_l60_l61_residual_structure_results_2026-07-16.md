# L60--L61 Residual-Structure Ablation Results

## L60 offline prediction

The parameter-matched MLP passed every nominal-relative offline eligibility check.
Its parameter count was 10,049 versus 10,191 for ICODE, a 1.39% difference. Across
three training seeds, mean H=36 rollout-RMSE reductions relative to nominal were:

| Split | MLP | ICODE |
|---|---:|---:|
| Test | 38.88% | 36.28% |
| Unseen reverse-S | 43.08% | 41.77% |

Thus the preregistered offline control-affine superiority hypothesis failed: ICODE
was 4.27% worse than MLP on test H=36 rollout RMSE and 2.24% worse on unseen H=36.

## L61 closed-loop development result

All 180 planned episodes completed with no protected or sealed seed use. Every
condition achieved 60/60 success and zero collisions.

| Contrast | Mean cross-track reduction | Absolute improvement | 95% hierarchical-bootstrap interval |
|---|---:|---:|---:|
| MLP vs nominal | 22.39% | 0.01276 m | [0.01121, 0.01420] m |
| ICODE vs nominal | 30.13% | 0.01732 m | [0.01545, 0.01906] m |
| ICODE vs MLP | 9.85% | 0.00456 m | [0.00330, 0.00583] m |

ICODE beat MLP in all three model blocks. On the unseen reverse-S path, its mean
absolute advantage was 0.00537 m with interval [0.00427, 0.00627] m. Mean planning
times were 5.04 ms nominal, 30.99 ms MLP and 36.68 ms ICODE; both learned models
passed the frozen 50 ms mean-compute limit.

## Interpretation

The result supports a narrow but important distinction: lower aggregate offline
state-rollout RMSE did not imply better receding-horizon control. A plausible
mechanism is that the control-affine factorization preserves the command-dependent
structure used when MPPI ranks perturbed control sequences. L61 does not prove this
mechanism, and no theoretical stability claim is made. L62 therefore opens the ten
predeclared seeds without changing models, costs, paths, plant or thresholds.

