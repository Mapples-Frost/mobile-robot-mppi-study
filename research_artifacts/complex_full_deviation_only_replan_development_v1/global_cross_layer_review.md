# Complex Full cross-layer review after the static-replan family stop

## Decision

The deviation-only static-A* replan family is closed. Three consecutive fresh
Chapter-1 seeds (`790200039`, `790201001`, and `790201003`) were collision-free
but all terminated at `max_steps`, with final goal distances of `7.43`,
`7.48`, and `7.33 m`. Removing redundant stagnation replans reduced reference
churn, but did not produce sustained navigation.

No further A* trigger, cooldown, margin, grid-resolution, or reference-weight
tuning is permitted for this family. No new episode will be launched until the
candidate-generation interface is diagnosed offline.

## Evidence by layer

### 1. Static geometry and global reference are no longer the leading blocker

- The corrected static-A* geometry contract yields a valid route on the
  approved Chapter-1 map.
- Known-static candidate feasibility remained `0.84-0.90` across the three
  runs, so the exact static filter did not eliminate the whole sample set.
- Deviation-only replanning reduced the run to `1-2` replans, yet all three
  fresh seeds still failed. This falsifies the hypothesis that repeated
  same-map replanning was the primary cause.

### 2. Perception and Change-Aware forecast are operational at the interface

- Dynamic-track association was `100%`.
- Forecast-valid availability was `93.25-96.25%`.
- The observed dynamic-cluster count averaged `1.12-1.55`.
- Emergency selections were usually forecast-corroborated: `96/102`, `66/85`,
  and `93/75` corroborated steps. The final ratio can exceed one because the
  two counters describe corroborated trigger cycles and selected-candidate
  cycles, not a one-to-one event ledger.

This evidence does not prove that every probability is calibrated, but it
rules out a missing-forecast or completely broken tracker as the dominant
failure mode.

### 3. Safety is active but no longer the sole motion bottleneck

- The three runs had `45`, `65`, and `92` Safety interventions.
- All runs remained collision-free and inside the boundary after the
  directional front-slowdown correction.
- The proposed controls themselves still split almost evenly between forward
  and reverse motion, with `20-24%` zero-speed exposure. Therefore the looping
  behavior is already present before Safety applies the final command.

### 4. The learned proposal path is effectively absent

This is the strongest cross-run finding:

| Seed | Guided opportunities | Guided feasible fraction | Guided elites | Gaussian elites | Mean guided fraction | HSS fallback |
|---|---:|---:|---:|---:|---:|---:|
| 790200039 | 178 | 0.000 | 5 | 30,911 | 0.0210 | 0.9325 |
| 790201001 | 178 | 0.000 | 3 | 33,589 | 0.0143 | 0.9550 |
| 790201003 | 178 | 0.000 | 0 | 34,054 | 0.0120 | 0.9625 |

The `178` guided opportunities correspond to the initial guided batch only.
After that first control cycle, HSS drives the guided allocation to zero for
almost the entire episode. Mean residual-support confidence is only `0.072`,
`0.046`, and `0.039`, and its per-episode minimum is zero. The resulting mean
proposal authority is `0.068`, `0.045`, and `0.038`.

Consequently, the nominal Full arm is behaving predominantly as Gaussian MPPI
plus the existing emergency lattice, not as an RL-guided MPPI controller.
This directly matches the observed lack of decisive learned bypass behavior.

### 5. Emergency/counterflow behavior is compensating for missing guidance

- Emergency candidates were selected on `75-102` of `400` steps.
- Counterflow escape was active on `112-223` steps.
- Critical-distance triggers were zero in all three runs.
- Temporal emergency triggers were much more frequent than hard-risk cycles.

The emergency mechanism is therefore doing substantial navigation work even
without critical contact. It keeps the robot safe, but it is a reactive
candidate family and does not supply a stable long-horizon bypass preference.
The nearly balanced forward/reverse proposals and repeated loops are
consistent with that limitation.

## Evidence versus inference

**Directly measured:** the tracker and forecast are available; static feasible
candidates exist; guided candidates appear only in the initial batch, none are
jointly feasible, HSS then suppresses guided allocation; Gaussian and emergency
candidates dominate; the robot remains safe but does not make sustained
progress.

**Current inference:** the frozen Actor/ICODE-support interface is
out-of-distribution in the cluttered complex-map state distribution. The HSS
suppression may be correct, or the support calculation may be applying a
single-dynamic training envelope too broadly. The present metrics cannot yet
distinguish:

1. actor controls that immediately violate known-static geometry;
2. actor proposals rejected by dynamic-risk feasibility;
3. actor proposals suppressed because residual-support confidence collapses;
4. a combination of these mechanisms.

## Next falsifiable mechanism

The next step is an **offline guided-candidate rejection decomposition** on the
already-opened runs and frozen configuration. It must report, separately for
Actor-guided candidates:

- boundary feasibility;
- known-static feasibility;
- probabilistic dynamic-risk feasibility;
- joint feasibility;
- first failing horizon index and obstacle class;
- HSS allocation before and after the first reliability update.

Only after that decomposition may one shared mechanism change be proposed.
If the Actor proposals are geometrically invalid, the appropriate boundary is
Actor-domain coverage/retraining, not another A* or emergency threshold. If
they are feasible but are suppressed solely by an incorrectly coupled support
contract, the repair belongs in the HSS interface and must preserve fail-closed
behavior under genuinely unsupported dynamics.

No formal server run, baseline, ablation, or second-map tuning is authorized at
this stage.
