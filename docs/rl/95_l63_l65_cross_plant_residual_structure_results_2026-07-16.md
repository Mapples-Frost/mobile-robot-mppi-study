# L63--L65 cross-plant residual-structure results

Date: 2026-07-16

## Outcome

The complete L65 sealed cross-plant gate passed. All 540 scheduled episodes
completed on the MuJoCo differential-drive backend. Each of Nominal, MLP, and
ICODE achieved 180/180 successes and zero collisions. No RL policy or memory
term was enabled.

The primary sealed ICODE-versus-parameter-matched-MLP contrast over the five
shifted plants was:

- mean cross-track RMSE improvement: 0.004956 m;
- mean relative reduction: approximately 9.23%;
- hierarchical 95% interval: [0.003993, 0.005885] m.

Across all six plants, including the training anchor, the paired estimate was
0.005057 m with hierarchical 95% interval [0.004040, 0.006064] m. Every plant
had a positive mean ICODE-versus-MLP effect:

| MuJoCo plant | Pairs | Improvement (m) | Relative reduction |
|---|---:|---:|---:|
| Training anchor | 30 | 0.005564 | 11.11% |
| Light mass | 30 | 0.004915 | 10.72% |
| High friction | 30 | 0.005292 | 9.30% |
| Weak actuator | 30 | 0.004705 | 8.33% |
| 100 ms delay | 30 | 0.004977 | 10.13% |
| Combined shift | 30 | 0.004889 | 7.64% |

On all plants and paths together, ICODE reduced cross-track RMSE by 26.88%
relative to nominal prediction, while the matched MLP reduced it by 19.17%.
Their absolute paired improvements over nominal were 0.017636 m (95% interval
[0.016506, 0.018774] m) and 0.012579 m (95% interval
[0.011712, 0.013459] m), respectively.

The unseen reverse-S ICODE-versus-MLP estimate was 0.005872 m with 95% interval
[0.004833, 0.006890] m. The combined-domain interval lower bound was
0.003399 m. Every shifted-plant, unseen-path, and combined-domain model block
was positive.

## Calibration provenance

Physical domains were calibrated without loading MLP or ICODE outcomes:

1. L63 v1 rejected torque, delay, and combined candidates whose nominal path
   effects were below the preregistered resolution threshold.
2. L63 v2 made all single factors measurable but rejected an extreme combined
   plant that achieved only 50% success.
3. L63 v3 froze the four accepted single factors, selected a moderate combined
   plant, and passed with 100% nominal success and zero collisions. Its nominal
   cross-track shifts ranged from 3.2% to 22.7%.

This sequence is retained as part of the experiment record. Learned-model
results were never used to choose a favorable physical domain.

## Development-to-confirmation replication

L64 used three development execution seeds (324 episodes) and passed before
the confirmation seeds were opened. Its shifted-plant ICODE-versus-MLP estimate
was 0.004754 m with 95% lower bound 0.004000 m. L65 then used five untouched
execution seeds (540 episodes) and reproduced an estimate of 0.004956 m with
95% lower bound 0.003993 m. No threshold was relaxed between the runs.

## Compute and safety

Mean planner computation time was 4.90 ms for Nominal, 29.45 ms for MLP, and
34.78 ms for ICODE, below the preregistered 50 ms mean-time limit. Maximum
per-episode step maxima were 175.89, 113.99, and 138.55 ms respectively; these
tail spikes remain an optimization target and should not be described as a
hard real-time guarantee.

## Claim supported

For the frozen MuJoCo mass, contact-friction, actuator, delay, and combined
parameter shifts evaluated here, a parameter-matched control-affine ICODE
residual produced lower closed-loop MPPI path-tracking error than both nominal
dynamics and an unstructured MLP residual. The effect replicated across three
independent training seeds, unseen path geometry, and untouched execution
seeds while preserving task success and collision outcomes.

## Claim not supported

This result does not prove why the structure helps, arbitrary OOD robustness,
real-robot transfer, stability, contraction, convergence, or any RL advantage.
The implementation represents the public control-affine residual form from the
ICODE-MPPI description; it does not claim all guarantees of the original ICODE
theory. RL remains disabled in L63--L65 and must be evaluated separately.

## Reproducibility artifacts

- Development config SHA-256:
  `2D03E5CD9340A1B89924D08A019E9C635CCE53E540922E66528239312912E860`
- Confirmation config SHA-256:
  `5C6A1E34958D45CAAEBCA65CF1574B3BD891139437F1765AE19AB7FF477F84E5`
- Run git SHA recorded by every block:
  `8124f48bae29902d23045dadceadcf8455647235`
- Development results:
  `results/research_platform/rl/l64_residual_structure_cross_plant_20260716_v1/`
- Sealed results:
  `results/research_platform/rl/l65_residual_structure_cross_plant_confirmation_20260716_v1/`
- Figure source:
  `experiments/rl/plot_residual_structure_cross_plant.py`
- Figure outputs:
  `figures/fig_l64_l65_cross_plant_residual_structure.{pdf,png}` under the L65
  result directory.
