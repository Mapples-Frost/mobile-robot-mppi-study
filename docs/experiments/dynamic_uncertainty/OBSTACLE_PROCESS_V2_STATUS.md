# Obstacle Process V2 Qualification Status

Date: 2026-07-23  
Branch: `codex/change-aware-probabilistic-mppi`  
Platform: native Windows  
Decision: **generator gate passed; await visual review before prediction work**

## Scope completed

This stage implements and qualifies only the dynamic-obstacle
data-generating process. IMM, learned prediction, collision probability, MPPI,
and RL-ICODE-HSS remain disabled by configuration.

V2 replaces the one-event/eight-second preview with a 16-second,
physically-constrained state machine:

- P1: noisy approximately constant velocity;
- P2: speed change plus stop and forward/reverse restart;
- P3: two finite-duration direction changes;
- P4: four separated event groups containing turn, speed change, stop,
  forward/reverse restart, and two observation dropouts.

The hidden command can change abruptly, but rendered velocity remains
continuous and respects hard speed, acceleration, braking, and moving-yaw-rate
limits.

## Qualification result

The frozen development design produced:

```text
4 processes x 2 noise levels x 20 development seeds = 160 trajectories
```

All 160 trajectories passed and all 160 experimental keys were unique.

| Audit quantity | Observed maximum | Frozen limit |
|---|---:|---:|
| Speed | 1.043647 m/s | 1.10 m/s |
| Acceleration magnitude | 0.900000 m/s² | 0.90 m/s² braking bound |
| Moving yaw rate | 1.200000 rad/s | 1.20 rad/s |

Every trajectory was exactly reproducible from its seed. All hidden states and
available observations were finite. Dropouts affected measurements only, not
the physical obstacle.

Across the 20 independent P4 seeds, the generator produced:

- 10 left turns and 20 right turns;
- all four registered turn angles: -90, -45, +45, +90 degrees;
- 5 acceleration and 25 deceleration events;
- 11 forward restarts and 9 reverse restarts;
- 20 stop events.

These counts are a development-time support check, not a claim that every
category is equally likely. Formal probability calibration will be handled
later, after the obstacle process is accepted.

## Verification

- New V2 unit and MuJoCo compilation tests: 19 passed.
- Complete dynamic-uncertainty test directory: 36 passed.
- Full repository suite: 1013 passed, 5 failed. The five failures are the
  unchanged historical missing-checkpoint failures in the L34/L70/L72 RL
  artifact tests; no V2 obstacle test failed.
- Qualification artifact integrity: passed.
- Held-out ID, held-out OOD, and sealed seed registries: not opened.

Artifacts are in
`research_artifacts/dynamic_obstacle_generator_v2_qualification/`.

## Visual-review viewer

The MuJoCo reviewer advances through four different P4 development seeds.
Each formal trajectory runs once, pauses at its endpoint, and then the viewer
starts a new episode. A world-space label shows the episode number and seed.

Scene legend:

- red cylinder: physical dynamic obstacle;
- dark red nose: current velocity direction;
- blue dots: available-position path samples;
- magenta dots: measurement-dropout portions;
- yellow disks: hidden change locations, shown for review only;
- green cylinder: stationary robot reference;
- bottom progress lights and text label: current episode.

The path and hidden change markers are visualization aids only. They are not
available to any future predictor or controller.

## Stop condition

Do not start IMM, probability prediction, collision risk, or MPPI work until
the user accepts the V2 obstacle behavior after the MuJoCo visual review.
