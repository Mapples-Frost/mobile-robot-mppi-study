# Gate 3 reliability-calibrated HSS results

Date: 2026-07-18

## Result status

Gate 3 is complete for the tested MuJoCo scope:

- Gate 3A, as originally frozen, failed because the four severe-unseen
  episodes all occupied the low-authority bin;
- the failure was retained rather than reclassified;
- Gate 3A2 repaired the evaluation design with a preregistered, independent
  30-episode graded stress set and passed;
- Gate 3B passed on development seeds 32--34;
- Gate 3C passed on sealed confirmation seeds 35--39.

These results support reliability-calibrated allocation of a fixed MPPI
sampling budget. They do not yet support a success-rate claim, because neither
arm reached the final goal within the frozen 180-step horizon.

## Frozen comparison

Both arms used:

- the same mean of three episode-bootstrap, value-aligned ICODE members;
- the same frozen L175 SAC Actor and target critic;
- the same 100 model rollouts and two MPPI iterations per decision;
- the same terminal-value weight;
- the same MuJoCo physics, LaserScan perception, obstacle layer and safety
  arbitration;
- randomized fixed/adaptive order inside every
  scene-by-physics-by-seed block.

The fixed arm assigned 30% of candidates to persistent Actor sampling. The
adaptive arm assigned 0%, 30% or 60% based only on signals available online:
ICODE ensemble disagreement, ICODE support distance, causal completed-step
innovation and Actor support distance. Physics-domain labels and simulator
truth were analysis metadata only.

## Offline reliability evidence

The L193 threshold selection used L183 validation only. On the original test
split, episode-level authority and ten-step rollout error had Spearman
association -0.905. The original severe-unseen split had association -1.000
but occupied only the low bin, so the original Gate 3A did not pass.

Gate 3A2 then evaluated the unchanged score on 30 newly collected,
mixed-severity episodes:

| Authority bin | Episodes | Mean normalized ten-step error |
|---|---:|---:|
| High | 3 | 0.0462 |
| Medium | 8 | 0.0595 |
| Low | 19 | 0.0776 |

The ordering was monotone, episode-level Spearman association was -0.878, and
low-authority mean error was 38.9% above non-low error. Combined-unseen had
lower authority and higher prediction error than nominal. All preregistered
Gate 3A2 conditions passed.

## Closed-loop development

Development included 18 paired cells: three seeds, two scenes and three
physics domains.

| Metric | Fixed 30% | Adaptive | Favorable change | Seed-cluster 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Final goal distance (m) | 2.1390 | 1.9213 | 0.2176 (10.18%) | [0.0932, 0.3131] |
| Stuck steps | 4.6667 | 3.0556 | 1.6111 (34.52%) | [0.3333, 2.5000] |
| Control jerk | 0.1061 | 0.1026 | 0.00354 (3.34%) | [-0.00001, 0.00650] |
| Planner time (ms) | 189.87 | 188.80 | 1.07 (0.56%) | [-12.69, 8.42] |
| Collisions | 0 | 0 | no regression | [0, 0] |
| Success | 0 | 0 | no change | [0, 0] |

Adaptive low/non-low step fractions were 67.8%/32.2%. The equal-budget,
authority-exercise, collision, goal-distance, jerk, compute and primary-outcome
criteria all passed.

## Independent confirmation

Confirmation included 30 paired cells from five sealed seeds, using the same
two scenes and three physics domains.

| Metric | Fixed 30% | Adaptive | Favorable change | Seed-cluster 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Final goal distance (m) | 2.1171 | 1.9726 | 0.1444 (6.82%) | [0.0639, 0.2578] |
| Stuck steps | 5.9667 | 5.5000 | 0.4667 (7.82%) | [-1.9000, 4.3333] |
| Control jerk | 0.1072 | 0.1051 | 0.00211 (1.97%) | [0.00022, 0.00355] |
| Planner time (ms) | 158.61 | 151.91 | 6.70 (4.23%) | [5.91, 7.49] |
| Collisions | 0 | 0 | no regression | [0, 0] |
| Success | 0 | 0 | no change | [0, 0] |

Adaptive low/non-low step fractions were 74.9%/25.1%. Final goal distance had
a strictly favorable seed-cluster bootstrap interval, jerk also improved, the
rollout budget remained exactly 100, and all Gate 3C conditions passed.

## Combined eight-seed audit

The combined audit contained 48 paired cells while retaining seed as the
independent bootstrap cluster.

| Metric | Fixed 30% | Adaptive | Favorable change | Seed-cluster 95% bootstrap CI | Paired \(d_z\) |
|---|---:|---:|---:|---:|---:|
| Final goal distance (m) | 2.1253 | 1.9534 | 0.1719 (8.09%) | [0.0976, 0.2523] | 1.44 |
| Control jerk | 0.1068 | 0.1041 | 0.00264 (2.48%) | [0.00093, 0.00426] | 1.04 |
| Stuck steps | 5.4792 | 4.5833 | 0.8958 (16.35%) | [-1.0208, 3.3750] | 0.26 |
| Planner time (ms) | 170.33 | 165.74 | 4.59 (2.70%) | [-0.55, 7.56] | 0.65 |

Combined authority step fractions were 72.2% low, 17.9% medium and 9.9% high.
This confirms that the method used all three regimes and did not collapse to a
fixed ratio.

## Scope and negative findings

The final-distance benefit was concentrated in the clean single-obstacle
family:

| Physics domain | Fixed distance | Adaptive distance | Favorable change |
|---|---:|---:|---:|
| Nominal | 1.9940 | 1.7564 | 0.2376 |
| Long delay | 1.9662 | 1.6916 | 0.2746 |
| Combined unseen | 1.8930 | 1.4107 | 0.4824 |

In `lab_complex`, the effect was close to neutral: +0.0062 m in nominal,
+0.0422 m in long delay and -0.0116 m in combined unseen. Therefore the
current defensible claim is:

> Reliability-calibrated HSS improves progress and smoothness at fixed rollout
> budget in the tested solvable/clean dynamics task, remains approximately
> neutral in the tested complex task, and avoids collision or compute
> regression.

It is not defensible to claim universal complex-navigation improvement,
improved success rate, or a formal uncertainty guarantee.

## Reproducible artifacts

- calibration:
  `results/research_platform/rl/gate3_reliability_calibration_l193/`;
- independent graded stress:
  `results/research_platform/rl/gate3_reliability_stress_evaluation_l194/`;
- development:
  `results/research_platform/rl/gate3_hss_development_l195/`;
- sealed confirmation:
  `results/research_platform/rl/gate3_hss_confirmation_l196/`;
- combined audit:
  `results/research_platform/rl/gate3_hss_combined_l197/`.

Large generated artifacts remain ignored by Git. Configs, runners,
preregistrations, hashes and this bounded interpretation are versioned.

