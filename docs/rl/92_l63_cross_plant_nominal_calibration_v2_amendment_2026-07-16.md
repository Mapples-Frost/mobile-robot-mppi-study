# L63 cross-plant nominal calibration v2 amendment

Date: 2026-07-16

## Why an amendment was necessary

The first learned-model-blind calibration passed every safety and completion
check, but the largest nominal cross-track shifts for torque, delay, and the
combined domain were only 2.57%, 1.37%, and 0.29%. Thus v1 failed its declared
3% resolution floor and no MLP/ICODE cross-plant comparison was started.

This is a calibration failure, not evidence for or against either residual
architecture.

## Changes made before observing learned-model outcomes

- Retain the already-informative `mass_light` and `friction_high` domains.
- Increase the separation between weak and strong actuator candidates.
- Make the combined candidate directionally coherent: heavy chassis, high
  friction, weak actuator, and 100 ms delay are applied together instead of
  mixing changes that cancelled in v1.
- Retain the physically supported delay range of 0--100 ms. Because the
  planner is explicitly given this delay and the MPPI interface limits it to
  one control interval, preregister a separate 1% measurement-resolution floor
  for the delay nuisance factor. All other groups retain the 3% floor.

The selector still reads only `traditional_nominal` outcomes. Neither MLP nor
ICODE is run until this amended calibration passes.

## Decision rule

If v2 fails, the cross-plant learned comparison remains blocked. If it passes,
the selected physical domains and their parameters are frozen before the
development comparison begins.

## Version 3 addendum

Version 2 made every single-factor domain measurable and eligible, but its
extreme combined domain achieved only 50% success and 92.7% mean completion.
It was therefore rejected. Version 3 freezes the four accepted single-factor
domains and calibrates two moderate combined candidates, again using only the
traditional nominal controller. This is the final planned calibration attempt;
failure blocks the combined-domain learned comparison rather than prompting
unbounded tuning.
