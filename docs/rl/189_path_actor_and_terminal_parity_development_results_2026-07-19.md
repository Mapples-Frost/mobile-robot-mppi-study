# L185--L188 path Actor and terminal-parity development results

Date: 2026-07-19  
Evidence class: development only  
Sealed status: L186 geometries and seeds 561--565 remain unopened

## Executive result

The route representation defect in the old point-goal Actor is solved
reproducibly.  Three independent SAC training runs each achieved 50/50
successes on the same 50-episode development matrix where the old Actor
achieved 0/50.  A separate paper-controller terminal-action defect was also
identified and repaired; after the repair all four MPPI factorial arms reached
the endpoint on both new development seeds.

The complete cross-layer method is **not yet sealed**.  Its L188 Gate missed
the pre-registered cross-track non-inferiority margin by 2.4 percentage points,
and an integrity audit found that the historical HSS reliability calibration
artifacts had `gate_passed: false`.  The factorial runner now rejects such
calibrations by default.

## 1. Path-conditioned Actor

### Design

- training paths: straight, turn, sweep, and chicane;
- development paths: reverse-S and hairpin;
- five physics domains per path;
- development seeds: 551--555;
- 50 episodes per checkpoint;
- three independent SAC training seeds;
- checkpoint selection: collision, success, cross-track RMSE, completion,
  return, then earliest checkpoint;
- no sealed path or seed was used for training or model selection.

### Direct-control development results

| Actor | Selected step | Success | Collision | Cross-track RMSE | Valid completion | Jerk |
|---|---:|---:|---:|---:|---:|---:|
| old L175 point-goal | historical best | 0/50 | 0/50 | 4.3847 m | 0.5276 | 3.1837 |
| L185 seed 20261901 | 50k | 50/50 | 0/50 | 0.1057 m | 0.9668 | 2.8971 |
| L185 seed 20261902 | 40k | 50/50 | 0/50 | 0.0995 m | 0.9645 | 2.8872 |
| L185 seed 20261903 | 20k | 50/50 | 0/50 | 0.1108 m | 0.9624 | 2.4858 |

Relative to the old Actor, cross-track RMSE fell by 97.5--97.7%.  A paired
bootstrap over the five independent development-seed clusters gave favorable
cross-track effects of 4.274--4.285 m; every 95% interval was strictly above
zero.  Success improved by exactly +1.0 in all three training replications.

This is a strong development result, not a paper confirmation result.  It
demonstrates that the old failure was primarily task representation: a
point-goal observation cannot distinguish paths sharing the same endpoint,
whereas signed cross-track error, tangent-heading error, curvature, and
remaining path length make the control task identifiable.

The reproducible analysis is written to:

```text
results/research_platform/rl/path_conditioned_l185_dev/development_analysis.json
```

## 2. Integrated MPPI terminal convergence

### L187 terminal-weight screen

The first integrated path screen tracked the route at approximately 5 cm
cross-track RMSE but often stopped just outside the fixed 0.25 m endpoint
tolerance.  The pre-registered change increased only
`goal_terminal_weight`, from 25 to 50, for every arm.

On diagnostic seed 553:

| Setting | Successful arms | Collision | Mean cross-track RMSE | Mean jerk |
|---|---:|---:|---:|---:|
| T0, terminal weight 25 | 0/4 | 0/4 | 0.04982 m | 0.20311 |
| T1, terminal weight 50 | 2/4 | 0/4 | 0.04483 m | 0.20413 |

T1 passed the frozen diagnostic rule: more successes, no collision, 10.0%
lower cross-track error, and only 0.5% higher jerk.

On confirmation seeds 554--555, Full Proposed succeeded on only one of two
episodes, so the Gate failed.  It nevertheless had 18.3% lower mean
cross-track RMSE and 0.3% lower jerk than ordinary fixed, with no collision.
The failure was exclusively endpoint success.

### L188 terminal action parity repair

The failed seed exposed a deterministic compatibility defect in
`PaperRLDrivenMppiController`.  The ordinary MPPI implementation already
contained a terminal bearing law that gates translation while preserving
rotation.  The paper controller constrained candidate samples but omitted the
same law after final slew-rate clipping.

At the failed seed-555 endpoint:

```text
true bearing error:         +0.3740 rad
executed angular velocity:  -0.9088 rad/s
alignment diagnostic:       inactive
```

The command turned away from the endpoint.  The repair shares one terminal
action helper between both RL-driven controllers and adds unit tests for
rotate-in-place behavior.

On previously unused development seeds 556--557 after the repair:

- all four factorial arms succeeded on both seeds: 8/8 total;
- there were zero collisions;
- Full Proposed succeeded 2/2;
- Full Proposed mean cross-track RMSE was 0.04122 m;
- ordinary fixed mean cross-track RMSE was 0.03837 m;
- Full Proposed was 7.43% worse on cross-track RMSE, outside the frozen 5%
  non-inferiority margin;
- Full Proposed jerk was 3.08% worse, inside the frozen 10% margin.

Thus the terminal-parity repair is validated, but the overall L188 integration
Gate remains failed by one criterion.

## 3. Reliability-calibration integrity finding

The historical value-aligned and ordinary reliability summaries used by the
development factorial both contain:

```text
gate_passed: false
```

They did contain selected candidates, and the factorial loader previously
checked only for candidate presence.  Runtime HSS therefore assigned zero
authority throughout L188 (`reliability_authority_mean = 0.0`) and suppressed
the newly competent Actor except for the terminal guidance floor.

This explains the direction of the L188 tradeoff:

- value-aligned fixed guidance: 0.03574 m mean cross-track RMSE;
- ordinary fixed guidance: 0.03837 m;
- adaptive Full candidate: 0.04122 m.

Value-aligned ICODE plus the path-conditioned Actor was favorable, while the
failed historical reliability gate removed useful Actor guidance and degraded
tracking.  The historical adaptive rows are retained for diagnosis but must
not be presented as final evidence for Full Proposed.

The factorial runner now:

1. rejects a reliability summary unless `gate_passed` is exactly true;
2. provides an explicitly named historical-reproduction override;
3. records override use in provenance;
4. never permits the override silently.

## 4. Scientific interpretation

The evidence currently supports:

1. residual-aware/path-aware RL guidance is learnable and reproducible;
2. value-aligned ICODE with fixed RL guidance is a promising integrated
   controller;
3. terminal convergence failures were an implementation defect, not evidence
   against ICODE or SAC;
4. the reliability-weighted guidance component is not yet validated.

The evidence does **not** yet support:

1. a final Full Proposed superiority claim;
2. a calibrated uncertainty authority claim;
3. cross-layer synergy on sealed geometries or physics domains;
4. an ICRA-ready statistical conclusion.

## 5. Frozen next Gate

The sealed set stays closed.  The next work item is a path-aware reliability
calibration whose training representation matches the L185 Actor observation.
It must pass validation, test, and unseen calibration Gates before it can be
loaded by the factorial runner.  Only then will the two frozen L186 geometries
and seeds 561--565 be opened for the Full Proposed interaction test.
