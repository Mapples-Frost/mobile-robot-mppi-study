# Gate 1 Multi-Domain Factorial Result

Date: 2026-07-18
Status: development Gate passed; not a confirmatory paper result

## Question

Does the paper-faithful Simple Combination of a direct SAC Actor/critic,
ICODE rollout dynamics, and MPPI remain useful beyond the earlier
single-scene screen, under declared physics changes and a fixed equal rollout
budget?

This Gate deliberately does **not** include value-aligned ICODE training,
reliability gating, conservative terminal fallback, memory, or adaptive sample
count.

## Reproducibility correction

The first multi-domain training attempt, L174, was interrupted at 10k steps
after detecting that its newly added configuration had not yet been committed.
Its metadata referenced the preceding Git SHA and therefore could not uniquely
identify its source state. L174 is retained only as development provenance.

L175 restarted from zero after the implementation and configuration were
tested, committed, and pushed. Its checkpoint records Git SHA:

```text
5d49bdd184b285880f5cc0286519e31666c0d66b
```

The closed-loop L176 runner used the subsequent clean artifact-isolation fix:

```text
92974f4
```

## Online inference optimization

The paper-faithful implementation originally repeated two invariant operations
for every hypothetical state:

1. sectorizing the same latest real LaserScan;
2. separately rolling the deterministic Actor mean and stochastic Actor
   candidates through the same dynamics.

The optimized implementation:

- sectorizes the shared latest scan once per candidate batch;
- vectorizes kinematic observation construction and normalization;
- rolls the deterministic mean and persistent stochastic guided set in one
  autoregressive batch;
- preserves the fixed-seed unbatched result in a regression test;
- does not change candidate count, horizon, ICODE model, terminal critic,
  cost, or safety chain.

Under a 100-rollout short closed-loop timing check, Simple Combination changed
from approximately 191.6 ms in L173 to 71.8 ms mean and 79.2 ms P95. The
full L176 aggregate was 75.0 ms mean and 82.9 ms mean-of-episode-P95. The
narrow-corridor combined-unseen cell still reached 119.5 ms P95, so a universal
hard real-time claim is not supported.

## Multi-domain Actor training

The direct physical-control SAC Actor used:

- quantile twin critics with 25 quantiles;
- four training scenes;
- four seen physics domains;
- validation on clean-single-obstacle and narrow-corridor scenes;
- validation across four seen domains and one combined-unseen domain;
- three validation episodes per scene-domain cell;
- validation-selected checkpointing rather than final-checkpoint selection.

| Step | Validation success | Collision | Mean goal distance |
|---:|---:|---:|---:|
| 10k | 1/30 | 0/30 | 1.6585 m |
| 20k | 15/30 | 0/30 | 0.7071 m |
| 30k | 21/30 | 0/30 | 0.4773 m |
| **40k** | **29/30** | **0/30** | 0.4152 m |
| 50k | 27/30 | 0/30 | **0.3264 m** |
| 60k | 27/30 | 0/30 | 0.4184 m |

The 40k checkpoint was selected because it had the best configured validation
score and the highest success count. At 40k, combined-unseen achieved 6/6
success with zero collisions. The later regression confirms that more training
is not automatically better.

## Closed-loop factorial design

Treatments:

```text
Traditional MPPI
ICODE-MPPI
RL-Driven MPPI
Simple Combination (ICODE + RL-Driven MPPI)
```

Design:

- two scenes: clean single obstacle and narrow corridor;
- three physics domains: nominal seen, long-delay seen, combined unseen;
- three paired simulation seeds: 11, 12, 13;
- 18 scene-domain-seed factorial blocks;
- all four treatments occur in every block;
- treatment order is randomized within every block using schedule seed
  `20260718`;
- 100 prediction rollouts per decision for every treatment;
- standard cells use 100 candidates once;
- paper RL-Driven cells use 50 candidates over two iterations;
- memory is disabled;
- LaserScan, local obstacle layer, scan_guard, safety arbitration, and MuJoCo
  true-plant execution remain unchanged.

The same simulation seed is reused across scene/domain strata to provide common
random numbers. Therefore the inferential resampling unit is the **seed
cluster**, not a controller timestep and not each of the 18 blocks. L176 has
only three independent seed clusters. Its cluster bootstrap is useful for a
development Gate but insufficient for a final statistical claim.

## Aggregate results

| Method | Success | Collision | Final distance | Jerk | Planner mean | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Traditional MPPI | 1/18 (5.6%) | 0/18 | 2.414 m | 0.1586 | 4.49 ms | 6.30 ms |
| ICODE-MPPI | 2/18 (11.1%) | 0/18 | 2.597 m | 0.1567 | 32.18 ms | 37.76 ms |
| RL-Driven MPPI | 8/18 (44.4%) | 0/18 | 0.580 m | 0.0995 | 29.28 ms | 32.99 ms |
| **Simple Combination** | **9/18 (50.0%)** | **0/18** | **0.520 m** | **0.0971** | 75.02 ms | 82.92 ms |

Relative to RL-Driven MPPI, Simple Combination:

- gained one success episode: +5.6 percentage points;
- reduced mean final distance by 0.060 m (10.4%);
- reduced mean control jerk by 0.00246 (2.5%);
- preserved zero collisions;
- added 45.7 ms mean planning time.

The per-seed aggregate difference favored Simple Combination for both final
distance and jerk in all three seed clusters.

## Factorial effects

Effects use the convention that negative is favorable for distance and jerk,
while positive is favorable for success and clearance. The 95% intervals below
are seed-cluster percentile bootstrap intervals with only three independent
clusters; they must not be described as confirmatory significance intervals.

| Outcome | RL main effect | ICODE main effect | ICODE x RL interaction | Combination vs RL |
|---|---:|---:|---:|---:|
| Success | +0.3889 [0.1667, 0.5000] | +0.0556 [0, 0.1667] | 0.0000 [0, 0] | +0.0556 [0, 0.1667] |
| Final distance | -1.9557 [-2.2796, -1.4540] | +0.0613 [-0.0116, 0.1153] | **-0.2427 [-0.4834, -0.0052]** | **-0.0601 [-0.1616, -0.0044]** |
| Control jerk | -0.05938 [-0.06479, -0.05077] | -0.00215 [-0.00335, 0.00004] | -0.00062 [-0.00475, 0.00587] | **-0.00246 [-0.00550, -0.00042]** |
| Minimum clearance | +0.4256 [0.4134, 0.4427] | -0.0034 [-0.0136, 0.0119] | +0.0054 [-0.0065, 0.0192] | -0.0007 [-0.0168, 0.0136] |

The dominant result is the RL main effect. The endpoint-distance interaction is
directionally favorable in every seed cluster, but success interaction is zero
and clearance interaction is unresolved. This is preliminary evidence of
coupling, not proof of a general super-additive synergy.

## Scene and domain limitations

- In clean-single-obstacle nominal and combined-unseen cells, both RL methods
  achieved 3/3 success.
- In clean long-delay, Simple Combination achieved 3/3 versus RL-Driven 2/3.
- In every narrow-corridor cell, all four methods had 0/3 formal success.
  Nevertheless, the RL cells reduced final distance from approximately
  3.5 m to 0.67--0.82 m without collision.
- ICODE alone slightly improved aggregate success but worsened aggregate final
  distance. Prior ICODE prediction and dedicated dynamics experiments remain
  valid; this L176 result says that this fixed checkpoint is not uniformly
  beneficial across the present mixed closed-loop tasks.

## Gate decision

Gate 1 receives a **development pass**:

- the paper-faithful RL integration is functional;
- the Actor learned transferable low-level behavior across seen and unseen
  physics;
- all factorial cells ran through the unchanged perception and safety chain;
- no method collided;
- Simple Combination improved success, final distance, and jerk over
  RL-Driven MPPI in aggregate;
- online aggregate timing is within the 100 ms control period.

It is **not** a final paper claim because:

- only one Actor seed and one ICODE checkpoint were evaluated;
- there are only three independent simulation-seed clusters;
- narrow-corridor success remains zero;
- one hard cell exceeds the 100 ms P95 deadline;
- the ICODE-by-RL interaction is not resolved for success, jerk, or clearance.

These results justify entering Gate 2. Gate 2 must test whether value-aligned
ICODE improves the control-relevant interaction over this frozen Simple
Combination baseline, especially in long-delay and narrow-corridor conditions.

## Artifacts

```text
results/research_platform/rl/
├── gate1_direct_control_multidomain_screen_l174/  # interrupted, development only
├── gate1_direct_control_multidomain_screen_l175/
│   ├── checkpoints/best.pt                         # selected at 40k
│   ├── validation_episodes.csv
│   ├── training_summary.json
│   └── run_metadata.json
└── gate1_multidomain_factorial_l176/
    ├── episodes.csv
    ├── summary.csv
    ├── domain_summary.csv
    ├── scene_domain_summary.csv
    ├── factorial_contrasts.json                    # superseded block bootstrap
    ├── factorial_contrasts_seed_clustered.json     # valid development analysis
    ├── metrics.json
    └── runs/
```
