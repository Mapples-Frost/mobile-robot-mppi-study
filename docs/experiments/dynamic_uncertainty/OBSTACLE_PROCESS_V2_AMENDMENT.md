# Obstacle Process V2 Preregistration Amendment

Status: frozen before V2 trajectory generation  
Date: 2026-07-23  
Scope: obstacle data-generating process only; no predictor or controller

## Reason for the amendment

The V1 engineering smoke successfully validated seed reproducibility, separated
process and observation noise, and exercised one direction event plus one speed
event. Visual review showed that one eight-second realization was too simple to
represent the final research environment: a stop event left much of the replay
nearly stationary, and the viewer repeated the same realization.

The V1 code and all 40 V1 smoke artifacts remain unchanged. V2 uses a new
generator version, configuration, artifact directory, and qualification report.
No V1 outcome is discarded or relabeled.

## Research object

The obstacle remains a planar point-mass state

```text
z = [px, py, vx, vy]
```

with a rendered radius of 0.25 m. A hidden, seeded motion program changes its
desired speed and direction. Physical velocity follows that program through
bounded acceleration and turn rate, with continuous stochastic acceleration.

The hidden command may switch suddenly. The physical velocity must remain
continuous except when direction is changed at effectively zero speed during a
reverse transition.

## Frozen physical contract

- Duration: 16.0 s;
- integration step: 0.05 s;
- maximum speed: 1.10 m/s;
- maximum normal acceleration: 0.65 m/s^2;
- maximum braking acceleration: 0.90 m/s^2;
- maximum moving turn rate: 1.20 rad/s;
- obstacle radius: 0.25 m;
- minimum scheduled event-group separation: 2.2 s;
- turn command duration: 0.65--1.60 s;
- stop dwell before restart/reverse: 1.20--2.00 s;
- observation dropout duration: 0.40--0.80 s.

The exact values and distributions are committed in
`configs/research/dynamic_obstacle_process_v2.yaml`.

## Four processes

### P1 Noisy CV

No discrete event. The obstacle follows an approximately constant velocity
with bounded continuous process noise.

### P2 Speed Change

Two speed-event groups:

1. acceleration or deceleration;
2. brake/stop followed by either forward restart or reverse.

Direction is otherwise unchanged.

### P3 Direction Change

Two finite-duration turns. Angles are drawn from
{-90, -45, +45, +90} degrees and executed within the frozen yaw-rate bound.
Speed remains approximately constant.

### P4 Combined Change

Four separated event groups containing:

- at least two direction or speed changes;
- at least one turn;
- at least one acceleration/deceleration;
- one brake/stop;
- one forward restart or reverse;
- two short observation-dropout intervals.

The event order, time jitter, turn direction, angle, speed scale, stop dwell,
and restart type are seeded. Event-group separation prevents unrealistically
fast mode chatter.

## Observation contract

The physical obstacle never disappears during observation dropout. Dropout is
represented only by an unavailable position measurement. Process-noise,
event-program, and observation-noise random streams are separate.

Changing observation noise must not change truth. Changing the noise profile
uses the same standardized process draws scaled by the registered noise level.

## Experimental unit and design

One complete trajectory seed is the independent unit. Time steps and events
inside one trajectory are repeated observations.

V2 development qualification:

```text
4 processes x 2 noise levels x 20 development seeds = 160 trajectories
```

Noise levels are paired within process/seed. The schedule is randomized with a
committed schedule seed and written before trajectories are generated.

The qualification uses development seeds 730100001--730100020. Held-out ID,
OOD, and sealed seeds remain unopened.

## Generator Gate

All conditions are required:

1. 160/160 unique trajectories and complete artifacts;
2. exact seed reproducibility;
3. finite truth and every available observation;
4. maximum speed, acceleration, braking, and moving yaw rate within the frozen
   limits plus numerical tolerance;
5. P1 contains no discrete event or dropout;
6. every P2 contains a speed-scale event, stop, and restart/reverse;
7. every P3 contains two turns;
8. every P4 contains the required turn, speed, stop, restart/reverse, and two
   dropout intervals;
9. across the 20 P4 seeds, both left/right turns, forward restart/reverse,
   acceleration/deceleration, and all registered turn magnitudes occur;
10. no truth or event label is exposed as an observation;
11. no sealed seed is imported or used;
12. all generator and MuJoCo model tests pass.

Failure preserves all artifacts and stops work before IMM.

## Viewer versus formal episode

A formal episode runs one realization exactly once and never teleports.
The review viewer may automatically start a new episode after a visible pause,
but it must advance to a new development seed and clearly display the episode
number. Viewer replay is not an experimental artifact.
