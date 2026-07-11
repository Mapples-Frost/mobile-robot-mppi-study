"""Shared contracts and validation for continuous-time dynamics models."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class DynamicsModel(Protocol):
    """Structural interface consumed by integrators and MPPI adapters.

    ``derivative`` operates on one state/control pair.  Batching belongs in a
    planner adapter or learned-model wrapper, keeping the numerical integrator
    contract small and unambiguous.
    """

    state_dim: int
    control_dim: int
    angle_indices: Tuple[int, ...]

    def derivative(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Return the continuous-time state derivative at ``(state, control)``."""


def _validate_positive_dimension(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be a positive integer, got {!r}".format(name, value))
    value = int(value)
    if value <= 0:
        raise ValueError("{} must be positive, got {}".format(name, value))
    return value


def validate_angle_indices(
    angle_indices: Sequence[int], state_dim: int
) -> Tuple[int, ...]:
    """Validate and canonicalize angular state indices."""

    state_dim = _validate_positive_dimension(state_dim, "state_dim")
    if angle_indices is None:
        return ()
    try:
        raw_indices = tuple(angle_indices)
    except TypeError as exc:
        raise TypeError("angle_indices must be an iterable of integers") from exc

    canonical = []
    for index in raw_indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("angle index must be an integer, got {!r}".format(index))
        index = int(index)
        if index < 0 or index >= state_dim:
            raise ValueError(
                "angle index {} is outside state dimension {}".format(index, state_dim)
            )
        if index in canonical:
            raise ValueError("duplicate angle index {}".format(index))
        canonical.append(index)
    return tuple(canonical)


def validate_model_dimensions(model: DynamicsModel) -> Tuple[int, int, Tuple[int, ...]]:
    """Return validated ``(state_dim, control_dim, angle_indices)`` metadata."""

    if not hasattr(model, "state_dim"):
        raise TypeError("dynamics model is missing state_dim")
    if not hasattr(model, "control_dim"):
        raise TypeError("dynamics model is missing control_dim")
    if not hasattr(model, "angle_indices"):
        raise TypeError("dynamics model is missing angle_indices")
    state_dim = _validate_positive_dimension(model.state_dim, "state_dim")
    control_dim = _validate_positive_dimension(model.control_dim, "control_dim")
    angle_indices = validate_angle_indices(model.angle_indices, state_dim)
    if not callable(getattr(model, "derivative", None)):
        raise TypeError("dynamics model must provide a callable derivative method")
    return state_dim, control_dim, angle_indices


def validate_vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    """Convert a one-dimensional numeric vector and reject invalid values."""

    expected_dim = _validate_positive_dimension(expected_dim, "expected_dim")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(name)) from exc
    if array.shape != (expected_dim,):
        raise ValueError(
            "{} must have shape ({},), got {}".format(name, expected_dim, array.shape)
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or Inf".format(name))
    return array


def validate_state_control(
    model: DynamicsModel, state: Any, control: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate dimensions, state, and control against a model contract."""

    state_dim, control_dim, _ = validate_model_dimensions(model)
    return (
        validate_vector(state, state_dim, "state"),
        validate_vector(control, control_dim, "control"),
    )


def validate_derivative(value: Any, state_dim: int) -> np.ndarray:
    """Validate a derivative returned by a dynamics implementation."""

    return validate_vector(value, state_dim, "dynamics derivative")


def validate_time(time: Optional[float]) -> Optional[float]:
    """Validate optional scalar time without silently accepting NaN or arrays."""

    if time is None:
        return None
    if isinstance(time, (bool, np.bool_)) or not np.isscalar(time):
        raise TypeError("time must be a finite scalar or None")
    try:
        numeric_time = float(time)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("time must be a finite scalar or None") from exc
    if not np.isfinite(numeric_time):
        raise ValueError("time must be finite")
    return numeric_time


def wrap_angle(angle: Any) -> Any:
    """Wrap angle(s) to ``[-pi, pi]`` using a numerically stable identity."""

    array = np.asarray(angle)
    wrapped = np.arctan2(np.sin(array), np.cos(array))
    if array.ndim == 0:
        return float(wrapped)
    return wrapped


def wrap_state_angles(
    state: Any, angle_indices: Sequence[int], state_dim: Optional[int] = None
) -> np.ndarray:
    """Return a copy of one state with configured angular elements wrapped."""

    array = np.asarray(state, dtype=np.float64)
    if state_dim is None:
        if array.ndim != 1:
            raise ValueError("state must be one-dimensional")
        state_dim = int(array.shape[0])
    array = validate_vector(array, state_dim, "state").copy()
    indices = validate_angle_indices(angle_indices, state_dim)
    if indices:
        index_array = np.asarray(indices, dtype=np.int64)
        array[index_array] = wrap_angle(array[index_array])
    return array

