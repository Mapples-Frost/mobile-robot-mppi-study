"""Control-affine PyTorch residual dynamics model."""

from __future__ import annotations

from typing import Dict, Sequence, Union

import torch
from torch import Tensor, nn

from .mlp_residual import (
    _activation_name,
    _build_mlp,
    _final_layer_scale,
    _hidden_dimensions,
    _positive_dimension,
    _require_finite,
    _validate_inputs,
    _validate_state_features,
)


class ICODEResidual(nn.Module):
    """Control-affine residual ``drift(z) + gain(z) @ control``.

    The model consumes already encoded/normalized state features ``z`` and
    normalized controls. For a batch, its component shapes are:

    - drift: ``(B, nx)``
    - gain: ``(B, nx, nu)``
    - control contribution: ``(B, nx)``
    - residual: ``(B, nx)``

    Unbatched inputs omit the leading batch dimension in every returned
    component. Pass ``return_components=True`` to receive a dictionary with
    the four named tensors; the default return value is only the residual.
    """

    MODEL_TYPE = "icode_residual"

    def __init__(
        self,
        state_feature_dim: int,
        state_dim: int,
        control_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "softplus",
        final_layer_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.state_feature_dim = _positive_dimension(
            state_feature_dim, "state_feature_dim"
        )
        self.state_dim = _positive_dimension(state_dim, "state_dim")
        self.control_dim = _positive_dimension(control_dim, "control_dim")
        self.hidden_sizes = _hidden_dimensions(hidden_sizes)
        self.activation = _activation_name(activation)
        self.final_layer_scale = _final_layer_scale(final_layer_scale)

        self.drift_network = _build_mlp(
            self.state_feature_dim,
            self.state_dim,
            self.hidden_sizes,
            self.activation,
            self.final_layer_scale,
        )
        self.gain_network = _build_mlp(
            self.state_feature_dim,
            self.state_dim * self.control_dim,
            self.hidden_sizes,
            self.activation,
            self.final_layer_scale,
        )

    @property
    def drift_net(self) -> nn.Sequential:
        """Short, non-registering alias for the drift network."""

        return self.drift_network

    @property
    def gain_net(self) -> nn.Sequential:
        """Short, non-registering alias for the gain network."""

        return self.gain_network

    def drift(self, state_features: Tensor) -> Tensor:
        """Evaluate only the drift, preserving batched/unbatched rank."""

        unbatched = _validate_state_features(
            state_features, self.state_feature_dim, self
        )
        network_input = state_features.unsqueeze(0) if unbatched else state_features
        result = self.drift_network(network_input)
        _require_finite(result, "drift output")
        return result.squeeze(0) if unbatched else result

    def gain(self, state_features: Tensor) -> Tensor:
        """Evaluate the gain with shape ``(nx, nu)`` or ``(B, nx, nu)``."""

        unbatched = _validate_state_features(
            state_features, self.state_feature_dim, self
        )
        network_input = state_features.unsqueeze(0) if unbatched else state_features
        result = self.gain_network(network_input).reshape(
            network_input.shape[0], self.state_dim, self.control_dim
        )
        _require_finite(result, "gain output")
        return result.squeeze(0) if unbatched else result

    def forward(
        self,
        state_features: Tensor,
        control: Tensor,
        return_components: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if not isinstance(return_components, bool):
            raise TypeError("return_components must be a bool")
        unbatched = _validate_inputs(
            state_features,
            control,
            self.state_feature_dim,
            self.control_dim,
            self,
        )
        if unbatched:
            state_features = state_features.unsqueeze(0)
            control = control.unsqueeze(0)

        drift = self.drift_network(state_features)
        gain = self.gain_network(state_features).reshape(
            state_features.shape[0], self.state_dim, self.control_dim
        )
        control_contribution = torch.bmm(gain, control.unsqueeze(-1)).squeeze(-1)
        residual = drift + control_contribution
        _require_finite(drift, "drift output")
        _require_finite(gain, "gain output")
        _require_finite(control_contribution, "control contribution")
        _require_finite(residual, "residual output")

        if unbatched:
            residual = residual.squeeze(0)
            drift = drift.squeeze(0)
            gain = gain.squeeze(0)
            control_contribution = control_contribution.squeeze(0)
        if return_components:
            return {
                "residual": residual,
                "drift": drift,
                "gain": gain,
                "control_contribution": control_contribution,
            }
        return residual

    def parameter_count(self) -> int:
        """Return the total number of scalar model parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> Dict[str, object]:
        """Return JSON-safe architecture metadata for a checkpoint."""

        return {
            "model_type": self.MODEL_TYPE,
            "class_name": type(self).__name__,
            "state_feature_dim": self.state_feature_dim,
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "final_layer_scale": self.final_layer_scale,
        }


# Both names appear in experimental descriptions; keeping aliases avoids
# coupling downstream checkpoint factories to capitalization conventions.
ICodeResidual = ICODEResidual
ControlAffineResidual = ICODEResidual


__all__ = ["ControlAffineResidual", "ICODEResidual", "ICodeResidual"]
