# Gate 2 results: competence-gated value-aligned ICODE

Date: 2026-07-18

## Executive result

Gate 2 produced a repeatable positive control result after retaining, rather
than hiding, a failed unconditional-value experiment.

The final mechanism uses the frozen SAC critic to shape ICODE only where three
conditions agree:

```text
value authority
  = observation-support confidence
  * twin-critic agreement
  * training-outcome-calibrated critic competence
```

On five new confirmation seeds and 30 paired MuJoCo cells, competence-gated
value alignment improved mean final goal distance by **19.86%**, increased
success from **40% to 50%**, reduced stuck steps by **26.70%**, and retained
zero collisions. Mean control jerk changed by only **+0.066%** (worse), with a
near-zero paired effect and a cluster interval spanning both directions.

The original zero-tolerance Gate is therefore formally not passed because the
jerk point estimate is infinitesimally adverse. The primary progress, success,
stuck and safety evidence is nevertheless independently positive. A practical
jerk-equivalence margin must be specified prospectively before a larger
paper-level confirmation; L191 is not retroactively relabeled.

## Evidence chain

### Clean data and controls

- L183: Actor-generated task data collected from committed source
  `526b53de7e663106a0c18f6ee1de5f6144f4fdbe`.
- L184: ordinary task-specific ICODE fine-tuning, `lambda_value=0`.
- L185: unconditional value-aligned fine-tuning, `lambda_value=5`.
- L184 and L185 share the same L183 data, initialization, optimizer family and
  20-epoch schedule.

### Retained failed experiment

L188 compared L184 with L185 on seeds 21--23, two scenes and three physics
domains. It completed all 18 paired cells and failed the frozen Gate:

| Metric | L184 | L185 | Relative/favorable change |
|---|---:|---:|---:|
| Final goal distance, m | 0.5766 | 0.6417 | 11.29% worse |
| Control jerk | 0.09947 | 0.09776 | 1.72% better |
| Success | 44.44% | 38.89% | 5.56 pp worse |
| Collision | 0% | 0% | equal |
| Minimum clearance, m | 0.5915 | 0.5979 | 1.09% better |

This falsified the claim that unconditional critic consistency is sufficient.
The narrow-corridor cells also had 0/9 success under both checkpoints at the
100-rollout budget. Their terminal critic mean was approximately -0.5, versus
approximately 11--13 in the clean scene. The failure motivated the frozen L189
remediation in
`docs/rl/172_gate2_competence_gated_remediation_prereg_2026-07-18.md`.

L186/L187 are not evidence. Duplicate background processes wrote to the same
development directories, so those artifacts were explicitly invalidated.

## L189 competence calibration

Only L183 **training episodes** calibrate competence:

| Calibration quantity | Value |
|---|---:|
| Successful episodes / transitions | 6 / 912 |
| Failed episodes / transitions | 10 / 1800 |
| Mean critic value, successful transitions | 12.6901 |
| Mean critic value, failed transitions | 1.5105 |
| Off threshold: failed-transition median | 2.1760 |
| On threshold: successful-transition median | 9.1302 |

No validation, test, unseen, L188 endpoint, scene label or future simulator
state enters the deployed checkpoint. The competence function only decides
which training samples may transmit critic gradients into ICODE.

## Offline L184 versus L189

The primary offline comparison is against ordinary task fine-tuning, not the
older base ICODE.

| Split | Metric | L184 | L189 | Favorable change |
|---|---|---:|---:|---:|
| Test | Terminal value RMSE | 3.15231 | 3.09172 | 1.92% |
| Test | H=10 rollout RMSE | 0.021650 | 0.022045 | -1.82% |
| Test | Terminal position RMSE | 0.011451 | 0.011436 | 0.13% |
| Test | Value rank correlation | 0.971813 | 0.971855 | +0.000041 |
| Unseen | Terminal value RMSE | 2.33695 | 2.26824 | 2.94% |
| Unseen | H=10 rollout RMSE | 0.029105 | 0.029199 | -0.32% |
| Unseen | Terminal position RMSE | 0.010380 | 0.010097 | 2.73% |
| Unseen | Value rank correlation | 0.982301 | 0.982378 | +0.000078 |

The offline mechanism Gate passes: terminal value error improves on test and
unseen, while rollout degradation remains below the frozen 3% limit. Heading
RMSE worsens by 4.39% on test and 0.24% on unseen and is retained as a
secondary negative result.

## L190 development replication

L190 used new seeds 24--26 and passed the original closed-loop Gate:

| Metric | L184 | L189 | Favorable change | Seed-cluster 95% interval |
|---|---:|---:|---:|---:|
| Final goal distance, m | 0.7062 | 0.5281 | 25.22% | [0.1694, 0.1832] m |
| Control jerk | 0.10019 | 0.09981 | 0.38% | [-0.00152, 0.00405] |
| Success | 44.44% | 50.00% | +5.56 pp | [0, 16.67] pp |
| Collision | 0% | 0% | equal | [0, 0] |
| Stuck steps | 7.22 | 5.44 | 24.62% | [-1.83, 4.83] |

The sample contains only three independent seed clusters; L190 is a
development replication, not the final evidence.

## L191 independent five-seed confirmation

L191 used seeds 27--31. The two compute shards had disjoint result directories,
identical source/checkpoint hashes and randomized checkpoint order within each
scene-by-domain-by-seed block. Their 30 pairs were merged only after both
shards completed.

| Metric | L184 | L189 | Favorable change | Paired seed-cluster 95% interval | Paired dz |
|---|---:|---:|---:|---:|---:|
| Final goal distance, m | 0.63884 | 0.51198 | 19.86% | [0.03246, 0.22010] m | 1.045 |
| Success | 40.00% | 50.00% | +10.00 pp | [3.33, 16.67] pp | 1.095 |
| Stuck steps | 5.87 | 4.30 | 26.70% | [0.067, 3.10] | 0.804 |
| Control jerk | 0.098115 | 0.098180 | -0.066% | [-0.00208, 0.00290] | -0.019 |
| Minimum clearance, m | 0.57174 | 0.56846 | -0.57% | [-0.00923, 0.00087] m | -0.509 |
| Collision | 0% | 0% | equal | [0, 0] | n/a |

Positive values in the intervals above are favorable. The final-distance,
success and stuck intervals exclude zero under the planned seed-cluster
bootstrap. With only five independent clusters, these intervals remain pilot
uncertainty estimates rather than a substitute for the final 10--20-seed
paper matrix.

### Scene and physics interpretation

| Stratum | L184 success | L189 success | L184 distance | L189 distance |
|---|---:|---:|---:|---:|
| Clean single obstacle, 15 pairs | 80% | 100% | 0.5500 | 0.2849 |
| Narrow corridor, 15 pairs | 0% | 0% | 0.7276 | 0.7391 |
| Combined unseen physics, 10 pairs | 30% | 50% | 0.7580 | 0.4466 |
| Long-delay seen physics, 10 pairs | 40% | 50% | 0.6496 | 0.5492 |
| Nominal seen physics, 10 pairs | 50% | 50% | 0.5089 | 0.5401 |

The confirmed benefit is a dynamics-and-progress result in the solvable clean
task, especially under combined-unseen and long-delay physics. It does **not**
solve the narrow-corridor planning floor. That harder planning failure remains
for later RL prior/guidance work and must not be attributed to residual
dynamics.

## What this supports

The results support the following bounded claim:

> A frozen RL critic can improve task-relevant ICODE fine-tuning, but critic
> agreement and OOD support alone are insufficient. Training-outcome-calibrated
> critic competence prevents the strongest negative transfer observed with
> unconditional value consistency and yields repeatable progress and success
> improvements in solvable MuJoCo dynamics-mismatch tasks.

This is genuine RL--ICODE coupling: RL value changes how ICODE learns, and the
critic's own demonstrated competence controls its training authority. It is
not merely `ICODE checkpoint + RL prior + terminal value`.

## What this does not support

- It does not establish a universal advantage in geometrically hard scenes.
- It does not establish a final paper-level significance claim with five seeds.
- It does not justify ignoring the L188 failure or the L191 jerk/clearance
  point estimates.
- It does not reproduce theoretical stability, contraction or convergence
  guarantees from any original ICODE work.
- It does not validate real-robot deployment.
- L191 ran two compute shards concurrently, so planner wall-time comparisons
  are excluded; efficiency requires a serial profiling experiment.

## Reproducibility

Key source revisions:

- `526b53d`: clean L183/L184/L185 data and checkpoints;
- `735814f`: auditable paired-checkpoint runner;
- `6556e96`: competence-gated value alignment and L189/L190/L191 source.

Key artifacts, intentionally Git-ignored:

```text
results/research_platform/datasets/gate2_value_alignment_l183/
results/research_platform/gate2_value_aligned_control_l184/
results/research_platform/gate2_value_aligned_icode_l185/
results/research_platform/gate2_competence_gated_icode_l189/
results/research_platform/rl/gate2_value_alignment_paired_l188/
results/research_platform/rl/gate2_competence_gated_paired_l190/
results/research_platform/rl/gate2_competence_confirmation_l191/
```

The combined L191 directory contains source CSV hashes, checkpoint hashes,
merged episode tables, the full offline/closed-loop analysis and the
seed-cluster bootstrap result.
