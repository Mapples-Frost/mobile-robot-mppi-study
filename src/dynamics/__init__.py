"""Composable dynamics models and numerical integration utilities.

The first implementation targets the repository's existing three-state
unicycle model.  The interfaces deliberately expose dimensions and angular
state indices so that higher-dimensional vehicle models can be added without
changing planners or learning code.
"""

from .combined_dynamics import CombinedDynamics, CombinedDynamicsModel
from .disturbed_unicycle import DisturbanceConfig, DisturbedUnicycle
from .integrators import euler_step, integrate_step, rk4_step
from .interfaces import (
    DynamicsModel,
    validate_angle_indices,
    validate_derivative,
    validate_model_dimensions,
    validate_state_control,
    validate_time,
    validate_vector,
    wrap_angle,
    wrap_state_angles,
)
from .nominal_unicycle import (
    NominalUnicycle,
    NominalUnicycleDynamics,
    UnicycleDynamics,
)
from .state_encoding import StateEncoder

__all__ = [
    "CombinedDynamics",
    "CombinedDynamicsModel",
    "DisturbanceConfig",
    "DisturbedUnicycle",
    "DynamicsModel",
    "NominalUnicycle",
    "NominalUnicycleDynamics",
    "StateEncoder",
    "UnicycleDynamics",
    "euler_step",
    "integrate_step",
    "rk4_step",
    "validate_angle_indices",
    "validate_derivative",
    "validate_model_dimensions",
    "validate_state_control",
    "validate_time",
    "validate_vector",
    "wrap_angle",
    "wrap_state_angles",
]

