# Architecture and Dependency Rules

```text
domain / spaces / tasks
        ↓
dynamics, perception, memory, safety, policies
        ↓
planning and simulation (independent peers)
        ↓
runtime
        ↓
evaluation, visualization, ROS/Gym integration
        ↓
thin experiment CLIs
```

Forbidden dependencies:

- planning → MuJoCo/ROS;
- simulation → planner implementation;
- core → Torch;
- package code → experiments;
- ROS Python 2 scripts → training or Torch;
- viewer → controller state mutation.

The runtime is the composition root. It is the only layer that combines a
plant, sensor suite, perception pipeline, controller, memory, safety arbiter,
recorder, and optional viewer.

`StateSpec`, `ActionSpec`, and Reference objects replace implicit vector
layouts and global goals. Periodic indices identify manifold-valued state
components such as heading.
