# L190 path-aware reliability calibration preregistration

Date frozen: 2026-07-19
Status: frozen before collecting any L190 transition
Scope: development calibration only

## Scientific question

Can the ICODE reliability mechanism rank multi-step prediction error when its
Actor-support term is computed in the same 54-dimensional observation space as
the frozen L185 path-conditioned SAC Actor?

The historical Gate 3 calibrations used a point-goal Actor and therefore cannot
be used as evidence for path-conditioned guidance.  This experiment replaces
only the calibration data and thresholds.  It does not change ICODE weights,
the Actor, MPPI cost, rollout budget, path geometry, success tolerance,
LaserScan, `scan_guard`, or safety arbitration.

## Frozen models

- Actor: L185 seed 20261901, checkpoint 50,000;
- ordinary ICODE: the three frozen L57 ensemble members;
- value-aligned ICODE: the three frozen Gate-3 ensemble members;
- integrator: RK4;
- reliability rollout horizon: 10 control steps.

The two ensembles are calibrated independently on exactly the same transitions.

## Experimental unit and split

An episode is an independent unit.  Overlapping ten-step windows are repeated
measurements inside an episode and must never be counted as independent
replicates.

Path geometry is blocked before collection:

| Split | Geometry | Physics role | Episodes |
|---|---|---|---:|
| validation | acceleration straight, acceleration turn | seen | 16 |
| test | sweep, chicane | seen | 16 |
| unseen-development | reverse-S, hairpin | seen and combined-unseen | 20 |

Validation selects the numerical thresholds.  Test and unseen files may be
opened only after one candidate is selected.  The collector must reject any
scene/domain overlap among these splits.

The L186 offset-serpentine and asymmetric-hairpin geometries and confirmation
seeds 561--565 remain sealed and are absent from this dataset.

## Representation integrity

Every transition must contain:

- a finite 54-dimensional Actor observation;
- the explicit six-dimensional path context;
- a valid-polyline flag equal to one;
- state, executed control, next state, scene, physics domain, seed and episode;
- dataset, checkpoint, config and Git provenance.

Calibration fails closed on an observation-dimension mismatch, missing path
context, non-polyline transition, NaN/Inf, missing episode, or split overlap.

## Frozen score and threshold search

The runtime authority remains the bounded product of:

1. ICODE ensemble-disagreement confidence;
2. causal prediction--execution innovation confidence;
3. residual training-support confidence;
4. full path-Actor observation-support confidence.

Only disagreement and innovation soft/hard thresholds are selected.  Candidate
values are the predeclared validation quantiles:

```text
soft: 0.20, 0.35, 0.50
hard: 0.65, 0.80, 0.90
```

No test or unseen result may change the candidate grid, Actor OOD limits,
confidence levels, or bin definitions.

## Offline Gate

Each of validation, test and unseen-development must independently satisfy:

1. at least two occupied authority bins;
2. at least three episodes in every occupied bin;
3. at least three low-authority episodes;
4. mean rollout error is monotone from high to medium to low authority;
5. episode-level authority/error rank correlation is non-positive.

Both ordinary and value-aligned calibrations must pass before the four-arm
factorial can load adaptive reliability without an override.  If either fails,
the closed-loop path Gate remains closed and no L186/seeds 561--565 result is
opened.

## Interpretation boundary

Passing supports a calibrated ranking claim within the tested differential-drive
path and physics families.  It does not establish a probability-calibrated
uncertainty estimate, universal OOD detection, formal stability, convergence,
dynamic-obstacle generalization, bicycle-model reproduction, or real-robot
validity.
