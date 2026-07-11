"""Validated fixed-step integrators for continuous-time robot dynamics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .interfaces import (
    DynamicsModel,
    validate_derivative,
    validate_model_dimensions,
    validate_state_control,
    validate_time,
    wrap_state_angles,
)


def _validate_dt(dt: Any) -> float:
    if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt):
        raise TypeError("dt must be a positive finite scalar")
    try:
        numeric_dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("dt must be a positive finite scalar") from exc
    if not np.isfinite(numeric_dt):
        raise ValueError("dt must be finite")
    if numeric_dt <= 0.0:
        raise ValueError("dt must be greater than zero")
    return numeric_dt


def _evaluate(
    dynamics: DynamicsModel,
    state: np.ndarray,
    control: np.ndarray,
    time: float,
    state_dim: int,
) -> np.ndarray:
    if not np.all(np.isfinite(state)):
        raise ValueError("integration produced a non-finite intermediate state")
    derivative = dynamics.derivative(state, control, time=time)
    return validate_derivative(derivative, state_dim)


def integrate_step(
    dynamics: DynamicsModel,
    state: Any,
    control: Any,
    dt: float,
    method: str = "rk4",
    time: Optional[float] = None,
    wrap_angles: bool = True,
) -> np.ndarray:
    """Integrate one zero-order-held control interval.

    Supported methods are explicit Euler and classical fourth-order
    Runge--Kutta (RK4).  When ``time`` is omitted, the interval starts at
    ``t=0``.  The control is held constant across RK4 stages.  Consequently a
    stateful effect such as a discrete command-delay queue must be advanced by
    a plant-level ``step`` method, never from ``derivative``.
    """

    state_dim, _, angle_indices = validate_model_dimensions(dynamics)
    state_array, control_array = validate_state_control(dynamics, state, control)
    numeric_dt = _validate_dt(dt)
    numeric_time = validate_time(time)
    t0 = 0.0 if numeric_time is None else numeric_time
    if not isinstance(method, str):
        raise TypeError("integration method must be a string")
    normalized_method = method.strip().lower()
    if normalized_method not in ("euler", "rk4"):
        raise ValueError(
            "unknown integration method {!r}; expected 'euler' or 'rk4'".format(
                method
            )
        )
    if not isinstance(wrap_angles, (bool, np.bool_)):
        raise TypeError("wrap_angles must be boolean")

    if normalized_method == "euler":
        k1 = _evaluate(dynamics, state_array, control_array, t0, state_dim)
        next_state = state_array + numeric_dt * k1
    else:
        half_dt = 0.5 * numeric_dt
        k1 = _evaluate(dynamics, state_array, control_array, t0, state_dim)
        k2 = _evaluate(
            dynamics,
            state_array + half_dt * k1,
            control_array,
            t0 + half_dt,
            state_dim,
        )
        k3 = _evaluate(
            dynamics,
            state_array + half_dt * k2,
            control_array,
            t0 + half_dt,
            state_dim,
        )
        k4 = _evaluate(
            dynamics,
            state_array + numeric_dt * k3,
            control_array,
            t0 + numeric_dt,
            state_dim,
        )
        next_state = state_array + (numeric_dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    if not np.all(np.isfinite(next_state)):
        raise ValueError("integration produced NaN or Inf")
    if bool(wrap_angles):
        return wrap_state_angles(next_state, angle_indices, state_dim=state_dim)
    return np.asarray(next_state, dtype=np.float64)


def euler_step(
    dynamics: DynamicsModel,
    state: Any,
    control: Any,
    dt: float,
    time: Optional[float] = None,
    wrap_angles: bool = True,
) -> np.ndarray:
    """Convenience wrapper for :func:`integrate_step` using Euler."""

    return integrate_step(
        dynamics,
        state,
        control,
        dt,
        method="euler",
        time=time,
        wrap_angles=wrap_angles,
    )


def rk4_step(
    dynamics: DynamicsModel,
    state: Any,
    control: Any,
    dt: float,
    time: Optional[float] = None,
    wrap_angles: bool = True,
) -> np.ndarray:
    """Convenience wrapper for :func:`integrate_step` using RK4."""

    return integrate_step(
        dynamics,
        state,
        control,
        dt,
        method="rk4",
        time=time,
        wrap_angles=wrap_angles,
    )


__all__ = ["integrate_step", "euler_step", "rk4_step"]

