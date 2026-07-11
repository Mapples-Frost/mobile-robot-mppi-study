"""Analytic oracle residual used as an experimental upper-bound ablation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..interfaces import (
    DynamicsModel,
    validate_derivative,
    validate_model_dimensions,
    validate_state_control,
    validate_time,
)


class OracleResidual:
    """Exact instantaneous residual ``f_true(x,u,t) - f_nominal(x,u,t)``.

    The oracle is permitted only for controlled ablations; a learned planner
    must not inspect the true plant.  For a plant with discrete command delay,
    ``control`` here must be the *applied* delayed control.  Delay itself is
    history-dependent and cannot be represented by a residual over the
    unaugmented state ``(x, u)``.
    """

    def __init__(
        self,
        true_dynamics: DynamicsModel,
        nominal_dynamics: DynamicsModel,
    ) -> None:
        true_state_dim, true_control_dim, true_angles = validate_model_dimensions(
            true_dynamics
        )
        nominal_state_dim, nominal_control_dim, nominal_angles = (
            validate_model_dimensions(nominal_dynamics)
        )
        if (true_state_dim, true_control_dim) != (
            nominal_state_dim,
            nominal_control_dim,
        ):
            raise ValueError("true and nominal dynamics dimensions must match")
        if true_angles != nominal_angles:
            raise ValueError("true and nominal dynamics angle_indices must match")
        self.true_dynamics = true_dynamics
        self.nominal_dynamics = nominal_dynamics
        self.state_dim = nominal_state_dim
        self.control_dim = nominal_control_dim
        self.angle_indices: Tuple[int, ...] = nominal_angles

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        state_array, control_array = validate_state_control(self, state, control)
        numeric_time = validate_time(time)
        true_derivative = validate_derivative(
            self.true_dynamics.derivative(
                state_array, control_array, time=numeric_time
            ),
            self.state_dim,
        )
        nominal_derivative = validate_derivative(
            self.nominal_dynamics.derivative(
                state_array, control_array, time=numeric_time
            ),
            self.state_dim,
        )
        residual = true_derivative - nominal_derivative
        return validate_derivative(residual, self.state_dim)


OracleResidualDynamics = OracleResidual


__all__ = ["OracleResidual", "OracleResidualDynamics"]

