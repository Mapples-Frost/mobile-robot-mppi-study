# L193 innovation-anchor closed-loop development preregistration

Date frozen: 2026-07-19
Status: frozen before running seeds 558--560
Parent: L192 continuous reliability confirmation

## Question

After both path-aware ordinary and value-aligned reliability calibrations pass
their independent offline Gates, can the frozen `innovation_anchor` authority
be exercised inside paper-faithful ICODE--RL--MPPI without losing endpoint
success, safety, tracking, or control regularity?

This is a development integration Gate, not sealed confirmation.

## Frozen experiment

- scene: `high_dynamic_reverse_s_terminal_dev_l187`;
- physics domain: `nominal_seen`;
- new development seeds: 558, 559, 560;
- four randomized factorial arms: ordinary fixed, value fixed, ordinary
  adaptive, Full Proposed;
- L185 Actor seed 20261901 at 50,000 steps;
- ordinary and value-aligned three-member ICODE ensembles;
- corresponding passed L192 calibration summaries;
- `K=100`, two MPPI iterations, 36-step horizon;
- 360-step episode limit and 0.25 m success tolerance;
- terminal goal weight 50;
- terminal guidance radius 1.26 m and guided-fraction floor 0.30;
- memory disabled; LaserScan, local obstacle layer, `scan_guard`, and final
  safety arbitration unchanged.

The episode is the independent unit.  All four arms run inside every seed
block in seeded random order.

## Gate

The development integration Gate passes only if:

1. Full Proposed succeeds in all three episodes;
2. Full Proposed has zero collisions;
3. Full Proposed mean cross-track RMSE is no worse than ordinary fixed by more
   than 5%;
4. Full Proposed mean cross-track RMSE is no worse than value fixed by more
   than 5%;
5. Full Proposed mean control jerk is no worse than ordinary fixed by more
   than 10%;
6. adaptive arms exercise at least two reliability levels and report both
   nonzero authority and nonzero guided sampling;
7. all arms retain exactly the same rollout budget and paper iterations.

No result from seeds 558--560 may be described as independent confirmation.
Passing authorizes a separately preregistered multi-domain development block.
It does not open L186 or seeds 561--565.
