# Gate 1 Positive Screen: Direct Actor + ICODE + MPPI

Date: 2026-07-18  
Status: exploratory positive screen, not a formal paper result

## Question

After implementing the paper-faithful low-level Actor interface, does the
Simple Combination behave as a functional and potentially useful integration,
or does coupling RL guidance with ICODE immediately degrade closed-loop
control?

## Training screen

The direct physical-control SAC Actor used a quantile distributional critic and
was trained for 15,000 environment steps on the clean single-obstacle MuJoCo
task. This was a learnability screen rather than the frozen plan's formal
multi-domain Stage A.

Validation mean final-goal distance evolved approximately as follows:

| Training step | Mean final-goal distance |
|---:|---:|
| 3,000 | 2.279 m |
| 6,000 | 2.047 m |
| 9,000 | 1.154 m |
| 12,000 | 2.071 m |
| 15,000 | 2.088 m |

The 9,000-step checkpoint was selected by held-out validation. The later
regression is evidence that last-checkpoint selection would be invalid.

## Equal-rollout closed-loop screen

Configuration:

- one MuJoCo scene;
- three paired seeds: 11, 12, and 13;
- 100 model rollouts per control decision for every method;
- standard MPPI used 100 samples once;
- paper RL-Driven cells used 50 candidates over two iterations;
- same horizon, plant, task, perception, scan guard, and safety chain;
- memory disabled;
- ICODE checkpoint fixed across the applicable cells;
- direct Actor/critic checkpoint fixed across the applicable cells.

| Method | Success | Collision | Final distance | Control jerk | Planner mean |
|---|---:|---:|---:|---:|---:|
| Traditional MPPI | 0/3 | 0/3 | 1.476 m | 0.1530 | 3.99 ms |
| ICODE-MPPI | 1/3 | 0/3 | 1.201 m | 0.1305 | 30.29 ms |
| RL-Driven MPPI | 2/3 | 0/3 | 0.660 m | 0.1072 | 143.70 ms |
| Simple Combination | 2/3 | 0/3 | 0.562 m | 0.1052 | 191.57 ms |

Relative to RL-Driven MPPI, the Simple Combination reduced mean final-goal
distance by approximately 14.8% and control jerk by approximately 1.9%. It did
not improve the 3-seed success count.

## Interpretation

This screen supports three limited claims:

1. The low-level RL-Driven MPPI implementation is not merely connected; it
   changes closed-loop outcomes substantially under an equal model-rollout
   budget.
2. ICODE and RL guidance can coexist without collision or obvious controller
   collapse.
3. Adding ICODE to RL-Driven MPPI improved continuous endpoint and smoothness
   metrics in this screen, although it did not yet increase success count.

It does **not** establish statistical significance, OOD robustness, a positive
ICODE-by-RL interaction effect, or superiority of the full proposed method.
Three seeds in one scene are insufficient for those claims.

## Failure and efficiency observations

- The direct Actor's validation performance peaked at 9k and then regressed.
  Formal Stage A therefore requires validation-based checkpointing and
  multi-domain training.
- Both RL-Driven cells exceeded the 100 ms control deadline on average. The
  current implementation prioritizes algorithm fidelity and auditable Python
  execution. Profiling and batching are required before any real-time claim.
- The screening run preceded the new randomized-block execution order. Future
  confirmatory runs randomize method order within each paired seed/domain
  block; this screen is retained as exploratory evidence only.

## Gate decision

The structural part of Gate 1 passes:

- physical low-level Actor semantics;
- Actor mean and covariance initialization;
- persistent guided sample set;
- iterative MPPI mean/covariance update;
- physical-action distributional terminal critic;
- identical declared rollout dynamics within each method;
- complete MPPI and safety execution chain;
- equal rollout-budget factorial entry point.

Gate 1 is not yet a formal experimental pass. Before freezing the baseline
result, Stage A must train the Actor/critic across declared seen physics
domains and the four-cell experiment must be repeated with randomized blocked
multi-seed evaluation. Gate 2 value-aligned ICODE work should start only after
that baseline is frozen.

## Reproducibility

Artifacts:

```text
results/research_platform/rl/
├── gate1_direct_control_learnability_screen_l171/
└── gate1_factorial_screen_l173/
```

Primary entry points:

```bash
python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/gate1_direct_control_learnability_screen_l171.yaml

python experiments/rl/run_gate1_simple_combination.py \
  --actor-checkpoint <direct-control-best.pt> \
  --icode-checkpoint <icode-best.pt> \
  --output-dir <output> \
  --seeds 11,12,13 \
  --total-rollouts 100 \
  --iterations 2
```
