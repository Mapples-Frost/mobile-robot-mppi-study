# Nominal Model, True Plant, and Residual

## Definitions

The nominal model is the model available to the planner.  For the protected
repository baseline,

```text
state   x = [x, y, theta]
control u = [v, omega]
f_nom   = [v cos(theta), v sin(theta), omega]
```

The true plant is the system that executes the final control.  In synthetic
experiments it is `DisturbedUnicycle`; in the existing live chain it can be
MuJoCo, and in later phases it is the robot.  The planner prediction model and
the true plant are different objects even when their equations happen to be
equal.

The instantaneous oracle residual is

```text
f_res_oracle(x,u,t) = f_true(x,u,t) - f_nom(x,u,t)
```

and the combined prediction derivative is `f_nom + f_res`.  A learned
residual approximates the same correction from transitions without access to
the true plant parameters at planning time.

## Supported mismatch

`DisturbanceConfig` independently controls:

- velocity gain `v_real = alpha_v v`;
- yaw gain and bias `omega_real = alpha_omega omega + b_omega`;
- three world-frame sinusoidal derivative terms;
- state-dependent `[k_x x, k_y y, k_theta sin(theta)]` terms;
- an integer command-delay queue.

Neutral defaults are exactly nominal.  The delay queue advances once per
external plant step, not once per RK4 stage.  Delay is history-dependent and
is not Markov in `[x,y,theta,u]`; a model without command history cannot have
an exact delay residual.  Dataset metadata and the unseen-delay split expose
this limitation rather than hiding it.

## Angle handling

The raw state retains `theta`, but a residual network normally receives
`[x,y,sin(theta),cos(theta)]`.  The derivative output still contains
`dtheta`, because integration occurs in the original state coordinates.
Observed angular derivatives and state losses use
`atan2(sin(delta),cos(delta))`, avoiding a false error of almost `2*pi` at the
branch cut.

## Paper relation and difference

ICODE-MPPI Eq. (11) is the same additive derivative relation.  The paper uses
a five-state bicycle model `[x,y,theta,v,delta]` with controls
`[acceleration,steering_rate]`.  This repository intentionally keeps its
three-state unicycle so the trusted MPPI, MuJoCo, and bridge contracts do not
silently change.  The dynamics protocol allows a future bicycle model to be
added without another planner rewrite.

## Files

- `src/dynamics/interfaces.py`: dimensions, finite-value validation, protocol.
- `src/dynamics/nominal_unicycle.py`: protected nominal equations.
- `src/dynamics/disturbed_unicycle.py`: configurable true plant and delay.
- `src/dynamics/residual/oracle_residual.py`: exact instantaneous correction.
- `src/dynamics/combined_dynamics.py`: additive model composition.
- `src/dynamics/integrators.py`: Euler and RK4.
