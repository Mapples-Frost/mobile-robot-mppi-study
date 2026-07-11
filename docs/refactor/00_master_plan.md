# Research Platform Refactor Master Plan

## Outcome

The repository now has a new Python 3 research package under
`src/mobile_robot_mppi`. Existing MPPI, Memory, MuJoCo live scripts, and ROS
Kinetic safety assets are retained as compatibility baselines.

The target is one experiment runtime with interchangeable plants, prediction
models, references, sampling priors, and evaluators.

## Architecture invariants

1. The plant is not the MPPI prediction model.
2. MuJoCo ground truth is not a planner observation.
3. Planner obstacles originate from LaserScan and the local obstacle layer.
4. Proposed and executed controls are both recorded.
5. Safety arbitration is downstream of MPPI, Memory, and RL.
6. Default imports do not require MuJoCo, ROS, or Torch.
7. The ROS Kinetic Python 2 process never imports Torch.

## Backends

`legacy_kinematic` preserves the historical low-order behavior.

`mujoco_diff_drive` uses an actuated free chassis, wheel hinges, contacts,
encoder/actuator/IMU sensors, odometry, and ray-cast LaserScan.

## Prediction modes

- `nominal`
- `oracle_residual`, only for explicit low-order oracle ablations
- `mlp_residual`
- `icode_residual`

An ordinary MuJoCo run intentionally rejects oracle access: the full simulator
has hidden wheel/contact state and must not be exposed to the planner.

## Research stages

1. Open-loop plant characterization.
2. Clean dynamics tracking.
3. Sensor/perception evaluation.
4. Full closed-loop navigation.
5. Physics/sensor domain randomization.
6. Sim-to-real replay and shadow mode.

## Compatibility

The previous 94 dynamics/learning/planner tests remain mandatory. New platform
tests are additive. Old CLIs remain available; the primary new entry is
`experiments/run_research_simulation.py`.
