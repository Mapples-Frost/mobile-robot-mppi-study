"""PyTorch multilayer-perceptron residual dynamics model.

This module intentionally is not imported by :mod:`src.dynamics.residual` so
that the package's NumPy dynamics interfaces remain usable without PyTorch.
Inputs are expected to have been encoded and normalized upstream.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Dict, Sequence, Tuple, Type

import torch
from torch import Tensor, nn


_ACTIVATION_TYPES: Dict[str, Type[nn.Module]] = {
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _positive_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be a positive integer".format(name))
    result = int(value)
    if result <= 0:
        raise ValueError("{} must be positive".format(name))
    return result


def _hidden_dimensions(hidden_sizes: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(hidden_sizes, (str, bytes)) or not isinstance(
        hidden_sizes, Sequence
    ):
        raise TypeError("hidden_sizes must be a sequence of positive integers")
    result = []
    for index, size in enumerate(hidden_sizes):
        result.append(_positive_dimension(size, "hidden_sizes[{}]".format(index)))
    return tuple(result)


def _activation_name(activation: str) -> str:
    if not isinstance(activation, str):
        raise TypeError("activation must be a string")
    name = activation.strip().lower()
    if name not in _ACTIVATION_TYPES:
        raise ValueError(
            "unsupported activation {!r}; expected one of {}".format(
                activation, tuple(_ACTIVATION_TYPES)
            )
        )
    return name


def _final_layer_scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("final_layer_scale must be a finite positive scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("final_layer_scale must be finite and positive")
    return result


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Tuple[int, ...],
    activation: str,
    final_layer_scale: float,
) -> nn.Sequential:
    layers = []
    previous_dim = input_dim
    activation_type = _ACTIVATION_TYPES[activation]
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(activation_type())
        previous_dim = hidden_dim

    output_layer = nn.Linear(previous_dim, output_dim)
    # A near-zero initial residual protects the nominal model at the start of
    # training while still allowing gradients through every output weight.
    nn.init.uniform_(
        output_layer.weight,
        a=-final_layer_scale,
        b=final_layer_scale,
    )
    nn.init.zeros_(output_layer.bias)
    layers.append(output_layer)
    return nn.Sequential(*layers)


def _validate_tensor(tensor: Tensor, expected_dim: int, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if tensor.layout != torch.strided:
        raise TypeError("{} must use strided tensor layout".format(name))
    if tensor.ndim not in (1, 2):
        raise ValueError(
            "{} must have shape ({},) or (batch, {}), got {}".format(
                name, expected_dim, expected_dim, tuple(tensor.shape)
            )
        )
    if tensor.shape[-1] != expected_dim:
        raise ValueError(
            "{} last dimension must be {}, got {}".format(
                name, expected_dim, tensor.shape[-1]
            )
        )
    if tensor.ndim == 2 and tensor.shape[0] <= 0:
        raise ValueError("{} batch dimension must be nonempty".format(name))
    if not tensor.is_floating_point():
        raise TypeError("{} must have a floating-point dtype".format(name))
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("{} contains NaN or Inf".format(name))


def _validate_state_features(
    state_features: Tensor,
    state_feature_dim: int,
    module: nn.Module,
) -> bool:
    _validate_tensor(state_features, state_feature_dim, "state_features")
    parameter = next(module.parameters())
    if state_features.device != parameter.device:
        raise ValueError(
            "state_features device {} does not match model device {}".format(
                state_features.device, parameter.device
            )
        )
    return state_features.ndim == 1


def _validate_inputs(
    state_features: Tensor,
    control: Tensor,
    state_feature_dim: int,
    control_dim: int,
    module: nn.Module,
) -> bool:
    unbatched = _validate_state_features(state_features, state_feature_dim, module)
    _validate_tensor(control, control_dim, "control")
    if state_features.ndim != control.ndim:
        raise ValueError(
            "state_features and control must both be batched or both be unbatched"
        )
    if not unbatched and state_features.shape[0] != control.shape[0]:
        raise ValueError(
            "state_features and control batch sizes must match, got {} and {}".format(
                state_features.shape[0], control.shape[0]
            )
        )
    if state_features.device != control.device:
        raise ValueError("state_features and control must be on the same device")
    if state_features.dtype != control.dtype:
        raise TypeError("state_features and control must have the same dtype")
    return unbatched


def _require_finite(tensor: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("{} contains NaN or Inf".format(name))
    return tensor


class MLPResidual(nn.Module):
    """Unstructured residual ``MLP([state_features, control])``.

    ``state_features`` and ``control`` must respectively have shape ``(nz,)``
    and ``(nu,)``, or matching batched shapes ``(B, nz)`` and ``(B, nu)``.
    The returned normalized residual has shape ``(nx,)`` or ``(B, nx)``.
    Encoding, normalization, and output denormalization are deliberately kept
    outside this model so training and inference use the same stored metadata.
    """

    MODEL_TYPE = "mlp_residual"

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
        self.network = _build_mlp(
            self.state_feature_dim + self.control_dim,
            self.state_dim,
            self.hidden_sizes,
            self.activation,
            self.final_layer_scale,
        )

    @property
    def net(self) -> nn.Sequential:
        """Short, non-registering alias for the feed-forward network."""

        return self.network

    def forward(self, state_features: Tensor, control: Tensor) -> Tensor:
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
        residual = self.network(torch.cat((state_features, control), dim=-1))
        _require_finite(residual, "residual output")
        return residual.squeeze(0) if unbatched else residual

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


MLPResidualModel = MLPResidual


__all__ = ["MLPResidual", "MLPResidualModel"]
