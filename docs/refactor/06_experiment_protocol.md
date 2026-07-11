# Unified Experiment Protocol

## Control metrics

Success, final distance, collision, path length, clearance, omega magnitude,
jerk, stuck/spin events, safety interventions, and planner latency.

## Physics metrics

Command tracking, wheel-speed error, slip ratio, actuator saturation, effort,
energy, braking distance, odometry drift, and contact impulse.

## Learning metrics

Residual derivative RMSE, one-step error, H-step rollout RMSE, unseen-physics
error, inference latency, and sample efficiency.

Smoke runs validate wiring only. Formal claims require fixed configurations,
10–20 seeds, stored checkpoints, and aggregate uncertainty reporting.
