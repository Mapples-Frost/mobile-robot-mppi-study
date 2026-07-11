"""Composition of a protected nominal model and an additive residual model."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .interfaces import (
    DynamicsModel,
    validate_derivative,
    validate_model_dimensions,
    validate_state_control,
    validate_time,
)
from .residual.base import ResidualModel


class CombinedDynamics:
    """Expose ``f_nominal + f_residual`` through one dynamics interface."""

    def __init__(
        self,
        nominal: DynamicsModel,
        residual: ResidualModel,
    ) -> None:
        nominal_state_dim, nominal_control_dim, nominal_angles = (
            validate_model_dimensions(nominal)
        )
        residual_state_dim, residual_control_dim, residual_angles = (
            validate_model_dimensions(residual)
        )
        if (nominal_state_dim, nominal_control_dim) != (
            residual_state_dim,
            residual_control_dim,
        ):
            raise ValueError("nominal and residual model dimensions must match")
        if nominal_angles != residual_angles:
            raise ValueError("nominal and residual angle_indices must match")
        self.nominal = nominal
        self.residual = residual
        self.state_dim = nominal_state_dim
        self.control_dim = nominal_control_dim
        self.angle_indices: Tuple[int, ...] = nominal_angles

    @property
    def nominal_dynamics(self) -> DynamicsModel:
        """Explicit alias used by checkpoint/evaluation code."""

        return self.nominal

    @property
    def residual_dynamics(self) -> ResidualModel:
        return self.residual

    def derivative_components(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return validated ``(nominal, residual)`` contributions."""

        state_array, control_array = validate_state_control(self, state, control)
        numeric_time = validate_time(time)
        nominal = validate_derivative(
            self.nominal.derivative(state_array, control_array, time=numeric_time),
            self.state_dim,
        )
        residual = validate_derivative(
            self.residual.derivative(state_array, control_array, time=numeric_time),
            self.state_dim,
        )
        return nominal, residual

    def derivative(
        self,
        state: Any,
        control: Any,
        time: Optional[float] = None,
    ) -> np.ndarray:
        nominal, residual = self.derivative_components(state, control, time=time)
        return validate_derivative(nominal + residual, self.state_dim)

    def step(
        self,
        state: Any,
        control: Any,
        dt: float,
        method: str = "rk4",
        time: Optional[float] = None,
    ) -> np.ndarray:
        from .integrators import integrate_step

        return integrate_step(self, state, control, dt, method=method, time=time)


CombinedDynamicsModel = CombinedDynamics


__all__ = ["CombinedDynamics", "CombinedDynamicsModel"]
