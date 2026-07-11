"""Base interfaces and neutral residual implementations."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ..interfaces import validate_angle_indices, validate_state_control, validate_time


@runtime_checkable
class ResidualModel(Protocol):
    """Structural residual interface matching a continuous dynamics model."""

    state_dim: int
    control_dim: int
    angle_indices: Tuple[int, ...]

    def derivative(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Return an additive correction in raw state-derivative coordinates."""


class ZeroResidual:
    """Dimension-aware additive identity for regression tests and ablations."""

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 2,
        angle_indices: Sequence[int] = (2,),
    ) -> None:
        if isinstance(state_dim, (bool, np.bool_)) or not isinstance(
            state_dim, (int, np.integer)
        ):
            raise TypeError("state_dim must be a positive integer")
        if isinstance(control_dim, (bool, np.bool_)) or not isinstance(
            control_dim, (int, np.integer)
        ):
            raise TypeError("control_dim must be a positive integer")
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        if self.state_dim <= 0 or self.control_dim <= 0:
            raise ValueError("state_dim and control_dim must be positive")
        self.angle_indices = validate_angle_indices(angle_indices, self.state_dim)

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        validate_state_control(self, state, control)
        validate_time(time)
        return np.zeros(self.state_dim, dtype=np.float64)


__all__ = ["ResidualModel", "ZeroResidual"]

