# Obstacle Process V3 Preregistration Amendment

Status: frozen before V3 trajectory generation  
Date: 2026-07-23  
Scope: obstacle data-generating process only

## Motivation

V2 passed its engineering and physical qualification, but visual review exposed
a scientific limitation: each process was generated from a small fixed template
with event groups near four anchor times. Some seeds reversed once, but the
obstacle did not exhibit sustained, non-periodic patrol behavior.

V3 keeps V1 and V2 unchanged and introduces a new generator, configuration,
artifact directory, and qualification report. No predictor, risk estimator, or
controller is enabled during V3 development.

## Research object

The physical state remains

```text
z = [px, py, vx, vy].
```

A hidden high-level intent selects a waypoint, dwell state, or spontaneous
behavioral intervention. The physical obstacle follows that intent with bounded
speed, acceleration, braking, and moving yaw rate. Hidden intent may change
abruptly; position and velocity may not teleport.

V3 is a semi-Markov process: the next intent depends on the current intent, and
the time spent in an intent is random. In plain language, the obstacle makes a
new decision after arriving somewhere or after a randomly timed intervention,
instead of executing four events at fixed clock times.

## Frozen physical contract

- Duration: 45.0 s;
- integration step: 0.05 s;
- maximum speed: 1.10 m/s;
- maximum normal acceleration: 0.65 m/s²;
- maximum braking acceleration: 0.90 m/s²;
- maximum moving yaw rate: 1.20 rad/s;
- obstacle radius: 0.25 m;
- no teleportation, loop reset, or endpoint wrap inside a formal episode.

## Four registered processes

### P1 Stochastic Cruise Control

Approximately constant-velocity control with bounded continuous process noise.
It has no discrete waypoint intent and no observation dropout.

### P2 Stochastic Shuttle

The obstacle repeatedly travels between two endpoints A and B. At each endpoint
it samples a new dwell time and cruise speed before returning. A formal episode
must contain at least three completed legs, visits to both endpoints, and at
least one A-B-A or B-A-B round trip.

The movement is recurrent but not exactly periodic because speed, dwell, process
noise, and measurement noise are resampled.

### P3 Branching Waypoint Patrol

The obstacle moves on a registered multi-waypoint graph. After each arrival it
samples one of the graph-neighbor goals, a dwell time, and a cruise speed.
A formal episode must complete at least four legs, visit at least three distinct
waypoints, and contain both leftward and rightward physical turns across the
development qualification set.

### P4 Hybrid Intent Patrol with Occlusion

The obstacle patrols a waypoint graph and is also subject to renewal-timed
spontaneous interventions:

- early retarget to another reachable waypoint;
- hesitation stop followed by resumption;
- speed replan while retaining the current target.

Intervention gaps are random and are not anchored to episode fractions.
Observation-dropout intervals are sampled from a separate random stream and do
not remove the physical obstacle. A trajectory must contain multiple completed
legs, at least two spontaneous interventions, and at least two dropout
intervals.

## Random-stream separation

The following seeded streams are independent:

1. high-level intent and route selection;
2. continuous physical process noise;
3. observation noise;
4. observation-dropout timing.

Changing only observation noise cannot change hidden truth, route choices, or
dropout availability. Low and medium process-noise conditions use paired
standard-normal draws scaled by the registered noise level.

## Experimental design

One complete 45-second trajectory seed is one independent unit. Time steps,
waypoint visits, and interventions inside it are repeated observations.

Development qualification:

```text
4 processes x 2 noise levels x 20 development seeds = 160 trajectories
```

The run order is randomized with a committed schedule seed. Only development
seeds 730100021--730100040 are allowed. Earlier V1/V2 development seeds,
held-out ID/OOD seeds, and the sealed registry are excluded.

## V3 generator gate

All conditions are required:

1. 160 unique run keys and complete artifacts;
2. exact seeded reproducibility;
3. finite truth and every available observation;
4. physical speed, acceleration, braking, yaw-rate, and per-step displacement
   within their registered bounds;
5. no state teleportation;
6. every P2 passes the shuttle recurrence contract;
7. every P3 passes its completed-leg and distinct-waypoint contract;
8. every P4 contains at least two spontaneous interventions and two observation
   dropouts;
9. development-set support includes every P4 intervention type and every
   registered waypoint;
10. changing observation noise leaves truth and availability unchanged;
11. no future truth, hidden intent, or event label is exposed as an observation;
12. no algorithm or sealed-seed module is enabled;
13. all V3, MuJoCo compilation, and regression tests pass.

Failure preserves artifacts and stops work before probability prediction.

## Viewer contract

The review viewer may show multiple development episodes sequentially. Each
formal episode runs once, pauses, then the viewer clearly labels and starts a
different seed. The viewer may draw future path, waypoint, event, and dropout
markers for human inspection only; these are forbidden inputs to future
algorithms.
