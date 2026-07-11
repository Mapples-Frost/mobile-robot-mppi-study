# MuJoCo Differential-Drive Plant

## Physical chain

```text
body velocity command
→ wheel reference conversion
→ ideal velocity servo or torque PI
→ wheel hinge torque
→ wheel/floor contact
→ free-body motion
```

The model explicitly records geometry, inertia, friction, actuator, solver,
and sensor parameters. The MJCF SHA-256 is stored with every episode.

`ideal_velocity` is a characterization profile with finite force limits.
`torque_pi` is the research profile and supports PI gains, integral clamp,
torque saturation, deadband, slew rate, and command delay.

Required characterization tests are zero-command drift, straight motion,
in-place rotation, fixed-radius arcs, step response, braking, low friction,
actuator saturation, collision contact, and timestep convergence.

Physical simulation is calibratable, not automatically real. Mass, inertia,
wheel geometry, motor response, and friction must later be identified from
real-robot logs.
