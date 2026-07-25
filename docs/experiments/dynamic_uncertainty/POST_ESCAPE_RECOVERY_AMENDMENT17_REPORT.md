# Post-Escape Recovery — Amendment 17

## Decision

Amendment 17 is the retained post-escape recovery implementation.

It preserves the Amendment 12 safety result (three of three risk-enabled
development episodes without collision) and removes the large heading-induced
stall in seed `730100003`. It does not yet solve the separate multi-encounter
time-loss failure in seed `730100001`; consequently the frozen 40 s gate remains
8/9 rather than 9/9.

## Experimental design

- Independent unit: one complete episode.
- Blocking factor: frozen V3 obstacle-process seed.
- Paired seeds: `730100001`, `730100003`, `730100005`.
- Paired arms: risk disabled and risk enabled.
- Frozen across amendments: obstacle generator, Change-Aware IMM predictor,
  collision-risk definition, hard probability threshold (`0.20`), localization,
  route, episode horizon, seed set, and gate.
- Simulator truth was used only for post-hoc audit.

## Diagnosis

Amendment 12 did not leave the robot permanently stuck. Both timeout episodes
made stable progress after the last escape. Seed `730100003`, however, contained
a distinct low-risk interval in which MPPI selected zero linear and angular
velocity while the robot was approximately perpendicular to the goal bearing.

The retained recovery trigger therefore requires all of the following:

1. A dynamic active escape occurred.
2. the scan guard is clear;
3. LiDAR TTC does not require another escape;
4. selected maximum collision probability is at most `0.10` for five
   consecutive control steps (`0.5 s`);
5. estimated goal-bearing error is at least `0.80 rad`.

Recovery commands zero translation and rotates toward the registered goal until
the estimated bearing error is within `0.20 rad`. Translation is never forced by
the recovery layer; authority returns to risk-evaluated MPPI after alignment.
The `0.10/0.15` values are entry/abort hysteresis for recovery, not replacements
for the frozen `0.20` hard collision threshold.

## Negative results retained

### Amendment 14

The first recovery design forced at least `0.30 m/s` forward speed after the
probability of the *stopped MPPI trajectory* became low. That probability did
not evaluate the newly forced forward command. Risk-enabled seeds `730100001`
and `730100003` collided. The design was rejected.

### Amendment 15

Translation forcing was removed, but the recovery controller read a bearing
field that is only populated when terminal heading gating is enabled. The field
was zero in this experiment, so behavior was identical to Amendment 12. The
design was rejected as a wiring failure.

### Amendment 16

The always-available estimated goal-bearing error was wired correctly.
Seed `730100003` improved, but seed `730100001` was unnecessarily reoriented
despite only `0.34 rad` of misalignment and subsequently collided. The design
was rejected.

Amendment 17 adds the predeclared `0.80 rad` stall-specific activation condition,
leaving the already-progressing seeds unmodified.

## Amendment 17 complete paired result

| Seed | Arm | Termination | Collision | Final goal distance (m) | Completion | Minimum clearance (m) |
|---:|---|---|---:|---:|---:|---:|
| 730100001 | disabled | goal reached | 0 | 0.2856 | 0.9675 | 0.0130 |
| 730100001 | enabled | max steps | 0 | 1.2035 | 0.8632 | 0.2567 |
| 730100003 | disabled | collision | 1 | 5.6616 | 0.3566 | -0.0039 |
| 730100003 | enabled | max steps | 0 | 0.3334 | 0.9621 | 0.1261 |
| 730100005 | disabled | goal reached | 0 | 0.2960 | 0.9664 | 0.0844 |
| 730100005 | enabled | goal reached | 0 | 0.2894 | 0.9671 | 0.3492 |

Relative to Amendment 12, seed `730100003` final goal distance fell from
`1.3383 m` to `0.3334 m`, while its collision outcome and minimum clearance
remained unchanged. It executed 18 rotation-only recovery steps and its active
escape count fell from 34 to 16.

## Gate

- Passed: forecast contract.
- Passed: route crossing qualification.
- Passed: tracker availability.
- Passed: visible measurement RMSE.
- Passed: zero enabled collision regressions.
- Passed: median clearance delta (`+0.2437 m`).
- Passed: behavioral divergence.
- Passed: enabled planner p95 compute time (`52.87 ms < 150 ms`).
- Failed: maximum enabled completion regression
  (`0.104309 > 0.10`).

The remaining failure is seed `730100001`, which is already aligned and resumes
forward motion after avoidance but loses time across repeated obstacle
encounters and scan-guard slowdowns. It is not the heading-stall failure addressed
by this amendment and should be studied as a separate intervention.

## Verification

The relevant regression suite completed with `162 passed` and 11 existing
Matplotlib/PyParsing deprecation warnings.

## Bounded follow-up and environment freeze

At the user's direction, one final single-factor probe was allowed before
freezing the environment work:

- Amendment 18 changed only `goal_terminal_weight`, from `25.0` to `35.0`.
- The registered screen used seed `730100001`.
- Acceptance required zero risk-enabled collision, no more than `0.02 m`
  clearance loss, and at least `0.10 m` final-distance improvement.
- The stopping rule prohibited further tuning if any condition failed.

The risk-enabled probe collided after 198 steps, with completion `0.506`.
The first acceptance condition therefore failed and the amendment was rejected
without a three-seed run.

This result shows that the remaining seed-001 completion loss cannot be repaired
safely by making the short-horizon objective more progress-aggressive. The
environment/controller baseline is frozen at Amendment 17. Subsequent work
should proceed to the planned algorithm stage, while reporting the remaining
8/9 gate honestly rather than continuing environment-level heuristic tuning.

## Project-lead stage decision

The automatic gate result remains **8/9** with `overall_pass: false`; the sole
miss is the narrow completion-regression threshold. On 2026-07-23, the project
lead explicitly accepted Amendment 17 as a **conditional pass** and authorized
the transition to the residual-dynamics stage. This is a documented stage
waiver, not a retroactive change to the measured gate result.

Amendment 17 is therefore frozen as the environment/controller baseline.
Further environment heuristics are out of scope unless a later residual-stage
result reveals a reproducible environment defect. The acceptance record is in
`ENVIRONMENT_STAGE_ACCEPTANCE.md`; the next-stage design is preregistered in
`RESIDUAL_DYNAMICS_STAGE1_PREREGISTRATION.md`.
