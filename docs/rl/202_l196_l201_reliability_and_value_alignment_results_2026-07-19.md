# L196--L201 reliability and path-value alignment results

Date: 2026-07-19
Status: completed development, sealed confirmation, and bounded remediation

## Executive result

Parity-calibrated Actor authority passed the L197 multi-physics development
Gate, but the frozen L198 path-and-physics confirmation missed its pooled
ordinary-ICODE comparator by 1.07%. The proposed stack remained safe and
successful, slightly improved over the fixed simple combination and the
combined-unseen ordinary comparator, and reduced jerk, but the seed-cluster
intervals include zero. It is therefore not a confirmed tracking-superiority
result.

The attempted path-policy-consistent value-ICODE repair was implemented and
tested correctly, but both L199 and route-balanced L201 failed their offline
generalization Gates. Those checkpoints are not eligible for closed-loop use.

## L196: identity-mapped source competence (failed)

Across 48 episodes, all arms succeeded and no collision occurred. Full
Proposed nevertheless had 7.99% higher pooled cross-track RMSE than ordinary
fixed and 13.32% higher RMSE in `combined_unseen`. The raw guided/Gaussian
elite-yield ratio was below parity but was mistakenly interpreted directly as
sampling authority.

## L197: parity-calibrated competence (passed development Gate)

The sole change mapped a relative elite yield of 0.75 or below to zero
competence and parity (1.0) to full competence. On fresh seeds 572--574:

| arm | episodes | success | collision | mean cross-track RMSE | mean jerk |
|---|---:|---:|---:|---:|---:|
| ordinary fixed | 12 | 12 | 0 | 0.041849 | 0.118074 |
| value fixed | 12 | 12 | 0 | 0.040351 | 0.116706 |
| ordinary adaptive | 12 | 12 | 0 | 0.040408 | 0.116805 |
| Full Proposed | 12 | 12 | 0 | **0.040066** | 0.118641 |

Full/ordinary pooled RMSE was 0.9574 (4.26% lower), combined-unseen was
0.8781 (12.19% lower), all four domains were within the frozen 5% stability
limit, and jerk ratio was 1.0048. Every preregistered L197 check passed.

## L198: sealed path-and-physics confirmation (failed narrowly)

The frozen implementation was evaluated on two unused L186 paths, three
physics domains, and sealed seeds 561--565: 120 episodes in 30 paired blocks.

| arm | episodes | success | collision | mean cross-track RMSE | mean jerk |
|---|---:|---:|---:|---:|---:|
| ordinary fixed | 30 | 30 | 0 | **0.085936** | 0.128394 |
| value fixed (simple combination) | 30 | 30 | 0 | 0.086987 | 0.127006 |
| ordinary adaptive | 30 | 30 | 0 | 0.086925 | 0.122032 |
| Full Proposed | 30 | 30 | 0 | 0.086852 | **0.123666** |

Deterministic comparisons:

- Full/ordinary pooled RMSE: 1.0107 (1.07% higher; primary check failed);
- Full/simple-combination pooled RMSE: 0.9984 (0.16% lower);
- combined-unseen Full/ordinary RMSE: 0.9962 (0.38% lower);
- Full/ordinary jerk: 0.9632 (3.68% lower);
- stable path--domain cells: 5/6;
- all 30 Full episodes succeeded with zero collisions.

The independent unit is seed. Full minus ordinary mean RMSE was +0.000916 m,
seed-cluster bootstrap 95% interval [-0.001359, +0.003333] m, paired
\(d_z=0.301\). Full minus simple combination was -0.000136 m, interval
[-0.002792, +0.002521] m, \(d_z=-0.038\). Both intervals include zero, so
neither superiority nor equivalence is established.

## L199 and L201: value-consistent ICODE remediation (rejected)

The prior L192 value-aligned ensemble used the old 48-dimensional point-goal
L175 critic while online sampling used the 54-dimensional path-conditioned
L185 policy. L199 added differentiable local path-feature re-encoding and
collected a matching 54-dimensional dataset.

In L199 all three members improved sweep validation rollout RMSE by 3.5--4.1%
and value RMSE by 5.7--8.0%, proving the gradient path was active. On chicane
test and unseen reverse-S/hairpin, value RMSE worsened by 6.8--18.7%; all
offline Gates failed.

L201 expanded training to straight, turn, sweep, and chicane while retaining
disjoint reverse-S validation and hairpin test. Checkpoint selection stopped at
epoch 0 for all three members: no fine-tuned checkpoint improved the frozen
validation criterion. L201 therefore ended this remediation without loss
weight search.

## Retained claim and next boundary

The evidence supports parity calibration as a useful development mechanism and
strongly supports keeping source-relative causal diagnostics. It does not yet
support a paper claim that the current Full Proposed controller universally
beats ordinary ICODE-MPPI on unseen paths.

The current path-value fine-tuning implementation is rejected. The next
research iteration should keep the proven ordinary ICODE model and couple the
RL value online (terminal evaluation, proposal selection, or compute
allocation) with an explicit reliability fallback. It must use fresh
development and confirmation data and must not reuse L198 as a tuning test.
