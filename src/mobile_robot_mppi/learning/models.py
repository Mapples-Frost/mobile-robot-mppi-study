import math
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _activation(name):
    choices = {
        "softplus": nn.Softplus,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    key = str(name).lower()
    if key not in choices:
        raise ValueError("unsupported activation: %s" % name)
    return choices[key]


def encode_state(state, angle_indices):
    angle_set = set(int(index) for index in angle_indices)
    values = []
    for index in range(state.shape[-1]):
        component = state[..., index:index + 1]
        if index in angle_set:
            values.extend((torch.sin(component), torch.cos(component)))
        else:
            values.append(component)
    return torch.cat(values, dim=-1)


def _mlp(input_dim, output_dim, hidden_sizes, activation, final_scale):
    layers = []
    previous = int(input_dim)
    activation_class = _activation(activation)
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, int(width)), activation_class()))
        previous = int(width)
    final = nn.Linear(previous, int(output_dim))
    nn.init.uniform_(final.weight, -float(final_scale), float(final_scale))
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ResidualNetwork(nn.Module):
    """MLP or control-affine ICODE residual with embedded normalization."""

    def __init__(self, state_dim, control_dim, config, statistics):
        super().__init__()
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.model_type = str(config.get("type", "icode_residual"))
        self.angle_indices = tuple(int(v) for v in config.get("angle_indices", (2,)))
        self.feature_dim = self.state_dim + len(self.angle_indices)
        hidden = tuple(int(v) for v in config.get("hidden_sizes", (128, 128)))
        activation = str(config.get("activation", "softplus"))
        scale = float(config.get("final_layer_scale", 1e-3))
        if self.model_type == "icode_residual":
            self.drift = _mlp(self.feature_dim, self.state_dim, hidden, activation, scale)
            self.gain = _mlp(
                self.feature_dim, self.state_dim * self.control_dim, hidden, activation, scale
            )
            self.direct = None
        elif self.model_type == "mlp_residual":
            self.direct = _mlp(
                self.feature_dim + self.control_dim, self.state_dim, hidden, activation, scale
            )
            self.drift = None
            self.gain = None
        else:
            raise ValueError("unknown residual model type: %s" % self.model_type)
        for name in ("feature_mean", "feature_scale", "control_mean", "control_scale", "residual_mean", "residual_scale"):
            value = torch.as_tensor(statistics[name], dtype=torch.float32)
            self.register_buffer(name, value)
        self.model_config = dict(config)

    def components(self, state, control):
        feature = encode_state(state, self.angle_indices)
        feature_normalized = (feature - self.feature_mean) / self.feature_scale
        control_normalized = (control - self.control_mean) / self.control_scale
        if self.model_type == "icode_residual":
            drift_normalized = self.drift(feature_normalized)
            gain_normalized = self.gain(feature_normalized).reshape(
                state.shape[:-1] + (self.state_dim, self.control_dim)
            )
            control_contribution = torch.matmul(
                gain_normalized, control_normalized.unsqueeze(-1)
            ).squeeze(-1)
            normalized = drift_normalized + control_contribution
        else:
            normalized = self.direct(torch.cat((feature_normalized, control_normalized), dim=-1))
            drift_normalized = normalized
            gain_normalized = torch.zeros(
                state.shape[:-1] + (self.state_dim, self.control_dim),
                dtype=state.dtype, device=state.device,
            )
            control_contribution = torch.zeros_like(normalized)
        residual = self.residual_mean + self.residual_scale * normalized
        if not torch.isfinite(residual).all():
            raise FloatingPointError("residual network produced NaN or Inf")
        return residual, {
            "drift_normalized": drift_normalized,
            "gain_normalized": gain_normalized,
            "control_contribution_normalized": control_contribution,
        }

    def forward(self, state, control):
        return self.components(state, control)[0]

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint_config(self):
        return {
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "model": dict(self.model_config),
            "statistics": {
                name: getattr(self, name).detach().cpu().tolist()
                for name in ("feature_mean", "feature_scale", "control_mean", "control_scale", "residual_mean", "residual_scale")
            },
        }


def load_platform_checkpoint(path, device="cpu"):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Torch < 2.0 compatibility; checkpoints remain project-owned.
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "mobile_robot_mppi_residual_v1":
        raise ValueError("not a refactored-platform residual checkpoint")
    config = payload["model_config"]
    model = ResidualNetwork(
        config["state_dim"], config["control_dim"], config["model"], config["statistics"]
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


class PlatformResidualDynamics:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.state_dim = int(model.state_dim)
        self.control_dim = int(model.control_dim)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        model, _ = load_platform_checkpoint(path, device)
        return cls(model, device)

    def derivative(self, state, control, time=None):
        del time
        state_value = np.asarray(state, dtype=np.float32)
        control_value = np.asarray(control, dtype=np.float32)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_value, device=self.device)
            control_tensor = torch.as_tensor(control_value, device=self.device)
            output = self.model(state_tensor, control_tensor)
        return output.detach().cpu().numpy().astype(np.float64, copy=False)
