# Dynamic-Uncertainty Environment Stage Acceptance

Date: 2026-07-23  
Platform: native Windows  
Branch: `codex/change-aware-probabilistic-mppi`  
Decision authority: project research lead (user)

## Status

**CONDITIONAL PASS — environment stage accepted and frozen at Amendment 17.**

This is a stage-transition decision, not a rewrite of the preregistered
numerical gate. The original Amendment 17 gate remains recorded as 8/9 with
`overall_pass: false`.

## Evidence retained

- Risk-enabled development episodes: 3/3 without collision.
- Seed `730100003` final goal distance improved from `1.3383 m` in Amendment 12
  to `0.3334 m` in Amendment 17.
- Median enabled clearance delta: `+0.2437 m`.
- Maximum enabled planner p95 compute time: `52.87 ms`.
- Tracker availability, visible-measurement RMSE, forecast contract, route
  crossing, behavioral divergence, clearance, collision regression, and
  compute gates passed.
- Completion-regression gate failed narrowly:
  `0.104309 > 0.10`.
- Relevant regression suite: `162 passed`.

## Waiver

The project research lead explicitly instructed:

> 记作通过，进入下一阶段

Accordingly, the remaining completion-regression miss is accepted as a known
environment-baseline limitation. It is not relabeled as a numerical pass and
must remain visible in paper methods, limitations, and artifact provenance.

## Frozen environment baseline

- Configuration:
  `configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment17.yaml`
- Complete paired evidence:
  `research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment17/`
- Obstacle generator: frozen V3 recurrent semi-Markov process.
- Predictor: frozen Change-Aware IMM.
- Collision-risk definition and hard threshold: frozen (`0.20`).
- Localization, route, registered development seeds, safety chain, and episode
  horizon: frozen.
- Rejected Amendment 18 progress-weight probe remains a negative result and is
  not part of the frozen baseline.

## Authorized next stage

Proceed to residual-dynamics integration and evaluation. Environment-level
heuristic tuning is closed unless new evidence reveals a correctness or safety
defect rather than a performance preference.

