# V3 navigation localization-drift correction report

Date: 2026-07-23  
Branch: `codex/change-aware-probabilistic-mppi`

## Finding

The non-arrival observed in Amendment 5 was a localization failure, not a
failure of the residual-dynamics concept.  The online controller used raw
wheel odometry while `planner.prediction_mode` remained `nominal`.  In the
worst inspected run the controller believed it was about 0.09 m from the
goal while the MuJoCo chassis was still about 3.28 m away.

The existing learned residual is applied inside MPPI rollouts.  It does not
overwrite `RobotObservation.pose`, so it cannot recover an already
accumulated world-frame pose offset.

## Implemented correction

`SimulatedSensorSuite` now supports `wheel_imu_localized`:

- wheel speed supplies translational dead reckoning;
- a noisy IMU-like yaw-rate measurement supplies angular propagation;
- a noisy 2 Hz external pose update represents a SLAM, mocap, or GNSS-like
  absolute localization anchor;
- the exact MuJoCo pose and twist are not returned to the controller.

Amendment 6 records the localization change.  The online audit now reports
position RMSE, final position error, and heading RMSE against simulator truth.
The V3 generator, Change-Aware IMM, collision-risk probability calculation,
route, seeds, and hard probability threshold were not changed.

## Development evidence

The complete Amendment 7 paired smoke contains six episodes over the three
registered development seeds.  Localization performance was:

| Metric | Mean | Maximum |
|---|---:|---:|
| position RMSE | 0.0202 m | 0.0226 m |
| final position error | 0.0176 m | 0.0410 m |
| heading RMSE | 0.0105 rad | 0.0128 rad |

Four of six episodes reached the true goal.  In the enabled arm, seeds
730100003 and 730100005 reached the goal with final true distances below
0.30 m and without collision.

## Separate safety finding

The paired smoke did not pass its original risk gate.  Seed 730100001 reached
the goal with risk disabled but collided with risk enabled after the hard-risk
response stopped the robot in the moving obstacle's crossing lane.  This is
not localization drift: localization RMSE in that failed episode was about
0.019 m.  It demonstrates that a universal "hard risk means zero velocity"
response is unsafe when an obstacle is moving toward a stationary robot.

This failure is retained as development evidence.  Probability thresholds
must not be tuned to hide it.  The next safety change should compare feasible
evasive candidate selection against stopping, using the same paired seeds.

## Residual-learning scope

A residual-dynamics factorial remains necessary:

1. raw wheel odometry versus localized odometry;
2. nominal versus frozen iCODE residual dynamics.

That experiment will measure the residual's short-horizon dynamics benefit
without claiming that a rollout residual is an absolute localization system.
