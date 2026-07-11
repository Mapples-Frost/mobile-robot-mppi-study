# Real-Robot Migration Plan

Learned residual control is not deployed to the robot in this phase.
`scan_guard` always retains the highest safety priority.

## Phase 1 - data only

Use a low-speed empty area with an operator and hardware emergency stop.  Run
straight lines, in-place rotations, fixed arcs, several `v/omega` levels,
different surfaces, and repeated trials.  Record synchronized timestamps,
odom pose, commanded `v/omega`, measured velocity/yaw rate, scan summary,
safety state, and arbitration state.  Also record software/config versions and
clock alignment.  Do not infer delay from unsynchronized logs.

## Phase 2 - offline prediction

Convert logs into episode-grouped datasets.  Compare nominal, MLP, and ICODE
one/multi-step predictions; keep entire runs/surfaces held out.  Report branch
cuts, dropped messages, stop events, and command-to-actuation delay.

## Phase 3 - shadow mode

Run a separate Python 3 inference process.  It receives copied state/control,
predicts only, and logs nominal/learned trajectories and timing.  It cannot
publish `/cmd_vel` and cannot influence arbitration.  Compare shadow
predictions with subsequent odometry and monitor missed deadlines.

## Phase 4 - guarded rollout

Only proceed after simulation, offline, and shadow acceptance.  Start at low
speed in an empty area, use conservative limits, watchdogs, checkpoint/hash
allowlists, fallback-to-nominal behavior, and independent emergency stop.
`scan_guard`, missing/stale sensor fail-safe, local obstacle layer, control
adapter, and final clamp remain authoritative.

## Deployment options

Review in this order: independent Python 3 inference node with IPC, TorchScript
or ONNX runtime, or audited NumPy parameter export.  Do not import PyTorch into
ROS Kinetic Python 2 scripts and do not make training/checkpoint dependencies
part of bridge startup.

## Acceptance evidence

Require bounded inference time under load, deterministic checkpoint loading,
prediction improvement on held-out logs, no increased safety-trigger rate in
shadow mode, and a documented rollback.  Prediction accuracy alone is not a
safety or stability guarantee.
