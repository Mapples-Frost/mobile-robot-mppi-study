# Recurrent Crossing Scenario Contract

Status: reserved for the later closed-loop control stage  
Date: 2026-07-23

## User scenario

A dynamic obstacle repeatedly and irregularly moves through the corridor
between the robot start and goal. The robot must predict the obstacle, choose a
safe time or path, and cross the corridor.

## Frozen design intent

- Robot start and goal lie on opposite sides of the conflict region.
- The obstacle repeatedly traverses the nominal robot route during one formal
  episode.
- Obstacle leg speed, dwell, restart time, and selected direction remain
  irregular and partially observed.
- The nominal robot route and obstacle occupancy tube must overlap in both
  space and time.
- The first encounter occurs after initialization and before the final approach.
- At least one dynamically feasible safe action exists: wait, slow, or detour.
- Immediate initial collision and unavoidable collision scenarios are rejected.
- No-conflict episodes are permitted only as an explicitly labeled negative
  control block.

## Required scene tests

```text
test_nominal_route_has_spatiotemporal_obstacle_conflict
test_obstacle_recurrently_crosses_robot_corridor
test_conflict_is_avoidable_not_forced
test_online_stack_cannot_access_future_encounter_labels
```

Offline scene compilation may use future truth to verify and freeze a valid
encounter. Online prediction and control may not access future obstacle truth,
future encounter time, or acceptance labels.

This contract does not modify the frozen V3 generator. It selects and freezes
robot start/goal geometry and scenario seeds downstream.
