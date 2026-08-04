# Raspberry Pi 5 / SCOUT MINI Full Proposed deployment

This deployment is isolated under `/home/pi/rlmppi_pi5`. It does not install
ROS, modify system Python, alter NetworkManager, or replace the validated
Livox SDK2 and CAN test programs.

## Runtime contract

The algorithm is the current simulation `B11_full_proposed` stack:

- current Actor and ICODE checkpoints;
- three frozen HSS sidecars;
- AR(1), `tau=2.0 s` sampling;
- 300 samples x 2 paper iterations = 600 rollouts;
- three-slot change-aware IMM/CA-IMM tracking;
- probabilistic risk and traversal logic;
- the same scan guard and final safety arbiter as simulation.

Only execution-environment interfaces differ: the unused MuJoCo plant is
replaced by a dependency-light placeholder, CUDA becomes ARM CPU, simulator
obstacle truth is removed, and map-based static geometry/A* are disabled.

## No-map static/dynamic separation

Filtering is deliberately split into two paths:

1. The **raw** point-derived `LaserScan` always goes to scan guard, local
   geometric obstacles and final safety arbitration.
2. A tracker-only copy enters motion bootstrap and the causal mapless
   classifier. Only confirmed dynamic tracks may produce probabilistic
   forecasts. Static and unknown objects remain present in path 1.

This prevents walls from becoming dynamic IMM targets without hiding walls
from collision avoidance. Once the base moves, pose/beam ego-motion
compensation must be validated before this classification can be accepted.

## Milestones

1. **Silent stationary**: Livox live, Full Proposed computes and logs commands,
   CAN guard transmits zeros only. The lidar extrinsic may still be provisional.
2. **Calibrated stationary**: measure `lidar_to_base`, floor height and SCOUT
   body exclusion; verify static labels converge and a walking target becomes
   dynamic.
3. **Wheels up**: add and validate CAN/IMU odometry, still with conservative
   speed limits and immediate zero on stale input/error/exit.
4. **Wheels down**: operator-held remote stop, open area, low speed, one short
   run at a time.

No later milestone is authorized by a successful earlier data-path smoke.

## Silent run

The launcher refuses to start if another deployment Livox bridge is active.
It starts the bridge, runs the Python stack and terminates the bridge on every
exit path. `--zero-can` is hard-coded by the launcher; the zero-only CAN class
has no API accepting a non-zero planner command.

```bash
cd /home/pi/rlmppi_pi5/current
chmod +x deploy/raspberry_pi5_scout/run_silent.sh
deploy/raspberry_pi5_scout/run_silent.sh \
  /home/pi/rlmppi_pi5/runs/silent_$(date -u +%Y%m%dT%H%M%SZ) 15
```

Outputs are written to a new directory and include resolved config, per-cycle
JSONL, summary/timing statistics and bridge stdout/stderr.

## Required calibration before motion

`livox_adapter_pi5.yaml` currently contains an identity lidar transform and a
conservative provisional body box. Before wheels-down operation measure:

- sensor translation `(x, y, z)` from `base_link`;
- roll, pitch and yaw orientation;
- floor height in the transformed base frame;
- points returned by the robot body and cable/mount hardware;
- front/rear/side footprint dimensions.

Changing these is sensor calibration, not controller tuning. Preserve each
resolved YAML and its SHA-256 with the run.
