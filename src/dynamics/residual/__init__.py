"""Residual dynamics contracts and analytic ablation models."""

from .base import ResidualModel, ZeroResidual
from .oracle_residual import OracleResidual, OracleResidualDynamics

__all__ = [
    "OracleResidual",
    "OracleResidualDynamics",
    "ResidualModel",
    "ZeroResidual",
]

