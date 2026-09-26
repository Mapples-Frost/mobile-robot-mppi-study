# L190--L194 path-aware reliability development results

Date: 2026-07-19
Evidence class: development only
Sealed geometry/seeds used: no

## Executive result

The path-aware reliability redesign reached a reproducible development
milestone, but it is not yet a paper-level superiority result.

1. L190 showed that the historical minimum fusion was not valid on held-out
   path geometry.
2. L191 isolated completed-transition innovation as the robust causal dynamics
   reliability signal.
3. L192 passed an independently generated continuous reliability Gate for both
   ordinary and value-aligned ICODE ensembles.
4. L193 failed in closed loop because high dynamics confidence was incorrectly
   treated as high Actor competence, raising the RL-guided fraction to 0.60 in
   an easy nominal domain.
5. L194 added the separately required \(c_\pi\) term using source-relative
   elite yield.  It passed all preregistered safety, noninferiority, mechanism,
   and equal-budget checks on three new development seeds.

The failed L190, L191 bin-count failure, and failed L193 integration Gate are
retained in the evidence package.  No threshold was changed after seeing an
experiment that was later called a pass.

## L190: historical fusion diagnosis

Dataset:

```text
results/research_platform/datasets/path_aware_reliability_l190
```

It contains disjoint validation, test, and unseen route environments; the raw
Actor observation has 54 dimensions and the explicit path context has six
dimensions.  All stored path-context validity flags passed.

The old conservative-min fusion did not generalize:

- ordinary ICODE: test rank correlation \(-0.859\), but the high-confidence
  bin had only one episode; unseen rank correlation \(-0.021\);
- value-aligned ICODE: test rank correlation \(-0.918\), but unseen rank
  correlation \(+0.024\).

Component analysis found that innovation confidence alone retained the desired
negative association with rollout error:

| Model | Validation | Test | Unseen |
|---|---:|---:|---:|
| ordinary ICODE | -0.891 | -0.865 | -0.910 |
| value-aligned ICODE | -0.885 | -0.847 | -0.899 |

The support-distance and Actor-input-distance terms reversed ordering on some
unseen path geometries.  This is why they were not allowed to continuously
attenuate authority in the next development variant.

## L191: causal innovation-anchor bin Gate

L191 used a fresh dataset and an opt-in `innovation_anchor` fusion:

- after the minimum completed-transition count, dynamics confidence follows
  causal innovation error;
- before it is ready, ensemble disagreement and residual support provide the
  conservative fallback;
- Actor support remains a hard out-of-support veto.

Both ordinary and value-aligned models were strongly monotone on test and
unseen splits.  The formal Gate nevertheless failed because one middle bin had
fewer than three episodes.  This was treated as a measurement-resolution
failure, not silently converted to a pass.

## L192: continuous held-out confirmation

L192 preregistered the final continuous measurement before collecting another
fresh dataset:

- 24 validation episodes;
- 24 test episodes;
- 30 unseen-disturbance episodes;
- episode-level Spearman rank correlation;
- 5,000 episode bootstrap replicates;
- required rank correlation \(\leq-0.50\);
- required bootstrap upper 95% bound \(<0\);
- required low-versus-high authority quartile error separation \(\geq20\%\).

### Ordinary ICODE ensemble

| Split | Rank correlation | 95% bootstrap interval | Tail separation |
|---|---:|---:|---:|
| validation | -0.949 | [-0.984, -0.834] | 54.3% |
| test | -0.924 | [-0.972, -0.781] | 46.0% |
| unseen | -0.890 | [-0.951, -0.736] | 49.3% |

### Value-aligned ICODE ensemble

| Split | Rank correlation | 95% bootstrap interval | Tail separation |
|---|---:|---:|---:|
| validation | -0.943 | [-0.984, -0.818] | 53.4% |
| test | -0.911 | [-0.971, -0.758] | 45.8% |
| unseen | -0.897 | [-0.955, -0.758] | 46.6% |

Both independent calibrations passed.  These results support the narrow claim
that historical innovation can rank near-term ICODE rollout reliability on
held-out route and disturbance configurations.  They do not prove calibrated
probabilities or closed-loop superiority.

## L193: closed-loop failure that separated \(c_{\mathrm{dyn}}\) and \(c_\pi\)

L193 ran a randomized four-arm block on development seeds 558--560 with
identical `K=100`, two MPPI iterations, and the same horizon.

All 12 episodes succeeded without collision.  However:

- Full Proposed cross-track RMSE was 0.04390 m;
- ordinary fixed was 0.04164 m;
- the ratio was 1.0544, just beyond the frozen 1.05 limit;
- all adaptive decisions remained in the high reliability level;
- mean guided fraction was approximately 0.598.

The Gate failed.  The diagnosis is structurally important: an accurate ICODE
rollout model does not imply that the Actor is the best proposal source.  This
is the same distinction made in the frozen plan between dynamics reliability
\(c_{\mathrm{dyn}}\) and policy competence \(c_\pi\).

## L194: source-relative Actor competence remediation

L194 added no network and no extra rollout.  It compares the elite opportunity
rate of persistent RL-guided candidates with that of current-Gaussian
candidates, smooths the ratio causally, and uses the result as \(c_\pi\).

The preregistered run used new development seeds 566--568.

| Arm | Success | Collision | Cross-track RMSE (m) | Control jerk |
|---|---:|---:|---:|---:|
| ordinary fixed | 3/3 | 0/3 | 0.04277 | 0.11668 |
| value fixed | 3/3 | 0/3 | 0.04477 | 0.11368 |
| ordinary adaptive | 3/3 | 0/3 | 0.04391 | 0.12109 |
| Full Proposed | 3/3 | 0/3 | 0.04351 | 0.11931 |

Frozen Gate comparisons:

- Full/ordinary cross-track ratio: 1.0171, pass;
- Full/value cross-track ratio: 0.9717, pass;
- Full/ordinary jerk ratio: 1.0226, pass;
- equal budget and iterations: pass;
- Full success and collision checks: pass.

Mechanism audit:

- Full Proposed \(c_\pi\) range: 0.266--0.855;
- mean Full Proposed raw guided fraction: 0.417;
- low/medium/high decision fractions: 0.068/0.543/0.389;
- every adaptive episode showed nonconstant competence and changed allocation
  away from the L193 fixed-high behavior.

The three-seed paired confidence intervals still cross zero.  Full Proposed is
therefore currently **noninferior with an exercised mechanism**, not
statistically superior to Simple Combination.  The next required experiment
is a preregistered multi-physics-domain development Gate in which
\(c_{\mathrm{dyn}}\) and \(c_\pi\) have reason to vary.

## Claim boundary and next Gate

Supported now:

- path context is represented explicitly and audited;
- causal innovation ranks rollout error on independent route/disturbance
  splits;
- dynamics confidence and Actor competence must be separate;
- source-relative \(c_\pi\) prevents the observed high-confidence
  over-allocation without increasing MPPI rollout count;
- the remediated controller retained success, safety, and tracking
  noninferiority in the nominal development scene.

Not supported yet:

- Full Proposed significantly outperforming Simple Combination;
- multi-domain closed-loop robustness;
- sealed-seed generalization;
- real-robot benefit;
- formal probability calibration, stability, or convergence guarantees.

The sealed L186 geometry and seeds 561--565 remain unopened.
