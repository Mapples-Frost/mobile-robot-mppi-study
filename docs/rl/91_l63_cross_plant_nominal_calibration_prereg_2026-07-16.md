# L63 Nominal-Only Cross-Plant Calibration Preregistration

## Purpose

Select physically distinct but task-feasible MuJoCo domains without inspecting MLP
or ICODE outcomes. This prevents selecting physics that happen to favor a learned
method.

## Frozen design

- Controller during calibration: traditional nominal MPPI only.
- Paths: high-dynamic chicane and the training-unseen reverse-S.
- Episode seeds: `21860801`--`21860803`.
- Anchor: the exact L56 training plant.
- Candidate groups: light/heavy mass, low/high friction, weak/strong drive torque,
  zero/long command delay, and one combined shift.
- The planner receives the measurable command-delay value, but no mass, friction or
  actuator-domain label.
- Eligibility: at least 80% success, zero collisions and mean path completion at
  least 0.95.
- Within each factor group, select the eligible candidate with the largest absolute
  nominal cross-track-RMSE shift from the anchor. The shift must be at least 3%.
- Run order is seeded and randomized. Control steps are not independent samples.

## Boundary

Calibration selects benchmark domains, not a learned-model result. MLP and ICODE
checkpoints are not evaluated until the selected domain names and overrides are
frozen in the next configuration.

