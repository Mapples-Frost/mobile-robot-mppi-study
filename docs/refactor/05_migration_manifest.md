# Migration Manifest

## New primary modules

- `src/mobile_robot_mppi/core`: contracts, spaces, references, configuration.
- `planning`: generic MPPI and low-order prediction models.
- `simulation`: legacy and MuJoCo plants plus sensors.
- `perception`: protected scan pipeline facade.
- `safety`: final arbitration boundary.
- `learning`: dimension-agnostic MLP/ICODE training.
- `runtime`: composition and experiment loop.
- `evaluation`: metrics and artifacts.
- `policies` / `integration`: RL and Gym-compatible ports.

## Retained compatibility assets

- `src/planners/mppi_mujoco_receding_horizon_experiment.py`;
- `experiments/mujoco_memory_mppi_live_sim.py`;
- `experiments/mujoco_memory_mppi_ablation.py`;
- `src/dynamics` and `src/learning` three-state ICODE stack;
- all `mppi_hardware_bridge/scripts` safety and ROS code.

These files are not deleted because they define trusted regressions and the
current real-robot boundary. New experiments should use the new package.

Generated `.pyc`, `__pycache__`, local result files, and `.bak` files are not
research source and remain ignored.
