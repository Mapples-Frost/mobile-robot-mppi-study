"""Repository-compatible nominal dynamics for a planar unicycle robot."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .interfaces import validate_derivative, validate_state_control, validate_time


class NominalUnicycle:
    """Three-state kinematic unicycle model.

    State is ``[x, y, theta]`` and control is ``[v, omega]``:

    ``x_dot = v cos(theta)``, ``y_dot = v sin(theta)``,
    ``theta_dot = omega``.

    This intentionally preserves the existing MPPI baseline instead of
    silently substituting the five-state bicycle model used by ICODE-MPPI.
    """

    state_dim: int = 3
    control_dim: int = 2
    angle_indices: Tuple[int, ...] = (2,)

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        state_array, control_array = validate_state_control(self, state, control)
        validate_time(time)
        theta = state_array[2]
        velocity, yaw_rate = control_array
        return validate_derivative(
            np.asarray(
                [
                    velocity * np.cos(theta),
                    velocity * np.sin(theta),
                    yaw_rate,
                ],
                dtype=np.float64,
            ),
            self.state_dim,
        )

    def step(
        self,
        state: Any,
        control: Any,
        dt: float,
        method: str = "rk4",
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Integrate one interval without introducing mutable plant state."""

        from .integrators import integrate_step

        return integrate_step(self, state, control, dt, method=method, time=time)


# Compatibility-friendly explicit aliases for downstream experiment code.
NominalUnicycleDynamics = NominalUnicycle
UnicycleDynamics = NominalUnicycle


__all__ = ["NominalUnicycle", "NominalUnicycleDynamics", "UnicycleDynamics"]
