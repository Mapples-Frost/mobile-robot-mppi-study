# L198 sealed path-and-physics confirmation preregistration

Date frozen: 2026-07-19
Status: frozen after L197 passed and before seeds 561--565 were opened
Frozen implementation: commit `d496d83`

## Confirmatory question

Does the frozen value-aligned ICODE plus parity-calibrated, residual-aware RL
sampling stack improve path tracking under new route geometry and physics
shifts, at the same MPPI rollout budget, relative to both traditional
ordinary-ICODE MPPI and the fixed-guidance simple combination?

L198 is confirmatory. No parameter, checkpoint, scene, cost, threshold, or
Gate may be changed after any L198 result is read.

## Frozen design

- holdout paths:
  `path_offset_serpentine_holdout_l186` and
  `path_asymmetric_hairpin_holdout_l186`;
- domains: `nominal_seen`, `long_delay_seen`, `combined_unseen`;
- sealed seeds: 561, 562, 563, 564, 565;
- arms: ordinary fixed, value-aligned fixed, ordinary adaptive, Full Proposed;
- 2 paths x 3 domains x 5 seeds x 4 arms = 120 episodes;
- `K=100`, two paper iterations, horizon 36;
- the L197 source-competence mapping, L192 calibrations, L185 Actor, ordinary
  ICODE ensemble, and value-aligned ICODE ensemble are frozen;
- memory is disabled; MuJoCo plant execution, LaserScan, local obstacle layer,
  `scan_guard`, and safety arbitration remain active and unchanged.

Each path--domain--seed is a paired randomized block. Seed is the inferential
cluster; paths and domains are repeated strata, and timesteps are never
treated as independent observations. Parallel filesystem shards may reduce
wall-clock time but cannot share controller state.

## Primary and safety Gate

The Gate passes only if all conditions hold:

1. Full Proposed succeeds in all 30 confirmation episodes;
2. Full Proposed has zero collisions;
3. pooled Full Proposed cross-track RMSE is no greater than ordinary fixed;
4. pooled Full Proposed cross-track RMSE is no greater than value-aligned
   fixed, the simple-combination control;
5. Full Proposed remains within 5% of ordinary fixed in at least five of the
   six path--domain cells;
6. pooled `combined_unseen` Full Proposed cross-track RMSE is no greater than
   ordinary fixed;
7. Full Proposed control jerk is no more than 10% above ordinary fixed;
8. all 120 episodes use exactly `K=100` and two paper iterations.

The paired seed-level effect and cluster-bootstrap interval are reported
regardless of sign. A passed deterministic Gate with a wide interval supports
the tested bounded benchmark but is not evidence of universal superiority.

## Claim boundary

Passing supports a frozen-stack claim on the tested differential-drive,
holdout-path, shifted-physics MuJoCo benchmark. It does not establish a
bicycle-model reproduction, real-robot superiority, dynamic-obstacle
generalization, formal OOD detection, or stability/convergence guarantees.
