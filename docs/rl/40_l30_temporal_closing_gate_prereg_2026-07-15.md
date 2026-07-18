# L30 LaserScan temporal-closing gate remediation preregistration

Date frozen: 2026-07-15, after the L29 development failure and before any L30
closed-loop outcome is collected.

## Why this remediation exists

L29 remains a failed development gate and will not be relabelled.  Its dynamic
crossing audit showed that a single obstacle entering laterally can remain a
low-density, one-sided LaserScan feature.  The spatial complexity gate was
intentionally insensitive to a single wall, so its learned-prior weight rose
too late.  Always-on target-LCB RL was safer in that stratum, which identifies
a gate-detection failure rather than evidence that the learned prior should be
removed.

## Fixed method change

L30 adds exactly one signal to the existing complexity gate: the positive
closing rate of the front, left or right LaserScan sector minimum between two
consecutive observations.  The signal uses no MuJoCo body pose, obstacle label,
motion script or future trajectory.  It is therefore available on the current
real-robot LaserScan chain.

The fixed development constants are chosen from the prescribed obstacle speed
and control period, not from an outcome sweep:

- soft closing rate: `0.05 m/s`;
- full activation rate: `0.20 m/s`;
- maximum relevant sector clearance: `1.50 m`;
- activation hold: `0.60 s`.

The temporal activation is combined with spatial activation by `max`.  All
other MPPI, ICODE, RL checkpoint, target-LCB filter, perception and safety
settings remain unchanged.

## Development design

Only the two remediated cells are rerun:

1. temporal-risk gated RL with nominal prediction;
2. temporal-risk gated RL with ICODE prediction.

They use the same three model-training blocks, five L29 development episode
seeds, three scene roles and two physics domains.  This produces
`3 x 5 x 3 x 2 x 2 = 180` new episodes.  The reuse of development seeds is
explicitly for mechanism repair; these observations cannot be reported as a
fresh confirmation.  L29 traditional, always-on LCB and spatial-gate cells are
fixed comparators and are not rerun.

## Eligibility before opening a new confirmation set

All conditions below are required:

- clean-scene temporal gate is stepwise identical to residual-matched
  traditional MPPI;
- no collision increase relative to residual-matched always-on LCB in either
  dynamic physics stratum;
- dynamic-crossing success is at least the always-on LCB success in both
  physics strata;
- aggregate static-blocking success loses no more than four of the 43 L29
  spatial-gate successes across 60 matched cells;
- temporal-gated ICODE is best or tied by success in at least three of the four
  static/dynamic x seen/unseen blocking strata.

Failure keeps confirmation sealed.  Passing permits a separately frozen L30
confirmation protocol; it is not itself confirmatory evidence.

## Interpretation boundaries

This is a scan-reactive risk heuristic, not an obstacle tracker or velocity
predictor.  No theorem, safety guarantee, stability guarantee or convergence
claim is made.  ICODE remains the publicly described control-affine residual
structure and is not claimed to reproduce all guarantees of the original
ICODE theory.
