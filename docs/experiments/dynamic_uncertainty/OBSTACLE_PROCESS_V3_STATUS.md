# Obstacle Process V3 Qualification Status

Date: 2026-07-23  
Branch: `codex/change-aware-probabilistic-mppi`  
Platform: native Windows  
Decision: **V3 accepted and frozen as the single-obstacle data-generating process**

## What changed from V2

V2 selected a small set of events near four predefined times. V3 instead uses
a recurrent semi-Markov intent process:

- a decision is triggered by waypoint arrival or a renewal-timed intervention;
- the next goal, dwell duration, cruise speed, and intervention are seeded
  random variables;
- event count and event time are not fixed by an episode template;
- every physical transition remains continuous and bounded.

No prediction, IMM, collision-risk, MPPI, or RL module is enabled.

## Registered processes

1. **P1 Stochastic Cruise:** a simple constant-velocity control.
2. **P2 Stochastic Shuttle:** repeated A-B-A-B movement with independently
   sampled leg speeds and endpoint dwell times.
3. **P3 Branching Patrol:** recurrent motion over a six-waypoint graph with a
   new neighbor goal sampled after each arrival.
4. **P4 Hybrid Patrol:** graph patrol plus early retargeting, hesitation stops,
   speed replanning, and measurement dropouts.

Each formal episode lasts 45 seconds and runs exactly once without teleporting,
wrapping, or resetting.

## Development qualification

```text
4 processes x 2 noise levels x 20 new development seeds = 160 trajectories
```

Seeds 730100021--730100040 were used. They do not overlap the V1/V2 smoke seeds.
Held-out ID/OOD and sealed seeds were not opened.

All 160 trajectories passed; all 160 experimental keys were unique.

| Audit quantity | Observed maximum | Frozen limit |
|---|---:|---:|
| Speed | 0.927614 m/s | 1.10 m/s |
| Acceleration magnitude | 0.900000 m/s² | 0.90 m/s² |
| Moving yaw rate | 1.200000 rad/s | 1.20 rad/s |
| Per-step displacement | 0.046234 m | 0.055000 m |

Every trajectory was exactly reproducible from its seed. No position or velocity
teleport occurred.

## Recurrence and variability results

- Every P2 trajectory completed exactly three legs in 45 seconds, giving the
  visit pattern A-B-A-B and at least one full round trip.
- P2 leg speeds and endpoint dwell times varied within each episode, so the
  shuttle was recurrent but not exactly periodic.
- P3 completed 4--6 legs per episode.
- P4 completed 3--5 legs per episode while also undergoing spontaneous
  interventions.
- The 20 P3 seeds produced 20 unique waypoint-route signatures.
- The 20 P4 seeds produced 19 unique waypoint-route signatures.
- All six registered waypoints were visited in both P3 and P4 development sets.
- The first P4 intervention occurred at 20 distinct times across 20 seeds.

P4 intervention support across the 20 medium-noise development trajectories:

| Intervention | Count |
|---|---:|
| Early retarget | 42 |
| Hesitation stop | 56 |
| Speed replan | 76 |

These are support/coverage counts, not estimated real-world probabilities.

## Observation uncertainty

P4 sampled 2--5 dropout intervals per episode from an independent random
stream. During dropout the red physical obstacle continued moving; only the
position measurement was unavailable.

Changing observation-noise magnitude left physical truth, route choices, and
dropout masks unchanged.

## Motion-distribution and saturation audit

The frozen 160-trajectory qualification set was audited after visual acceptance.
This was a read-only characterization; no generator parameter was changed.

| Quantity | Mean | p90 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Acceleration magnitude (m/s^2) | 0.184280 | 0.650000 | 0.900000 | 0.900000 | 0.900000 |
| Moving yaw rate (rad/s) | 0.141794 | 0.283940 | 0.932326 | 1.200000 | 1.200000 |
| One-step displacement (m) | 0.026160 | 0.039444 | 0.042461 | 0.045278 | 0.046234 |

Using occupancy at or above 99.5% of the registered physical limit:

- acceleration saturation occupancy was 12.83% overall and 23.92% for P4;
- moving yaw-rate saturation occupancy was 4.05% overall and 7.88% for P4.

Thus V3 should be described as an aggressive but bounded synthetic process.
Saturation is concentrated in the deliberately difficult hybrid process and is
not negligible. This is recorded as a limitation, not removed after observing
predictor results, because changing V3 now would invalidate the frozen
predictor comparison.

The complete audit is stored at
`research_artifacts/dynamic_obstacle_generator_v3_qualification/analysis/motion_distribution_audit.json`.

## Verification

- V3 generator and MuJoCo compilation tests: 20 passed.
- Complete dynamic-uncertainty test directory after ordinary-IMM integration:
  63 passed.
- Full repository suite: 1040 passed, 5 failed. The five failures are the
  unchanged historical L34/L70/L72 missing-checkpoint failures; no V3 test
  failed.
- Qualification artifact integrity: passed.
- Complete artifact tree:
  `research_artifacts/dynamic_obstacle_generator_v3_qualification/`.

## Visual-review legend

- red cylinder: physical obstacle;
- dark red nose: current velocity direction;
- blue dots: path samples with available observations;
- magenta dots: observation-dropout portions;
- purple disks: registered waypoints;
- yellow disks: waypoint arrival/departure events;
- orange-red disks: spontaneous P4 interventions;
- green cylinder: stationary robot reference.

The path, waypoints, and event disks are human-review overlays only and are
forbidden inputs to future algorithms.

The Windows-native MuJoCo review was launched successfully, its automatic
transition from the shuttle/patrol sequence into the labeled P4 hybrid episode
was visually confirmed, and the user accepted V3 for predictor development.

## Remaining limitation

V3 currently models one moving obstacle at a time. It is suitable for validating
single-obstacle intent prediction and risk integration. Multi-obstacle
interaction and correlated pedestrian behavior are deliberately deferred until
the single-obstacle probability pipeline is scientifically validated.

## Gate decision

The V3 generator is substantially less fixed than V2 and satisfies the frozen
recurrence, route-diversity, stochastic-timing, physical-continuity, and
observation-separation gates. It is now frozen: later predictor strengths or
failures must not be used to tune the data-generating process.
