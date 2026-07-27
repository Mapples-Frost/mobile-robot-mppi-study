"""Multimodal supervised control-sequence proposals for MPPI."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ManeuverProposalActor(nn.Module):
    """Map one causal observation to several bounded control sequences.

    The network has no execution authority.  Its outputs are proposal
    sequences that must still pass ICODE rollout, Collision Risk, MPPI
    selection, and Safety.
    """

    def __init__(
        self,
        observation_dim,
        *,
        heads=3,
        horizon=36,
        action_dim=2,
        hidden_sizes=(128, 128),
    ):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.heads = int(heads)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        if min(
            self.observation_dim, self.heads, self.horizon, self.action_dim
        ) <= 0:
            raise ValueError("maneuver Actor dimensions must be positive")
        layers = []
        width = self.observation_dim
        for hidden in hidden_sizes:
            hidden = int(hidden)
            if hidden <= 0:
                raise ValueError("hidden sizes must be positive")
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(
            width, self.heads * self.horizon * self.action_dim
        ))
        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        values = self.network(observations)
        values = values.reshape(
            -1, self.heads, self.horizon, self.action_dim
        )
        return torch.tanh(values)


def best_of_m_loss(
    predictions,
    targets,
    *,
    time_weights=None,
    diversity_margin=0.08,
    diversity_weight=0.02,
):
    """Time-weighted best-head imitation with a mild anti-collapse penalty."""

    if predictions.ndim != 4 or targets.ndim != 3:
        raise ValueError("maneuver predictions/targets have invalid ranks")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("maneuver batch sizes differ")
    if predictions.shape[2:] != targets.shape[1:]:
        raise ValueError("maneuver target geometry differs")
    horizon = predictions.shape[2]
    if time_weights is None:
        weights = torch.linspace(
            1.0, 0.45, horizon,
            dtype=predictions.dtype,
            device=predictions.device,
        )
    else:
        weights = torch.as_tensor(
            time_weights,
            dtype=predictions.dtype,
            device=predictions.device,
        )
        if tuple(weights.shape) != (horizon,):
            raise ValueError("time weights must match maneuver horizon")
    weights = weights / torch.mean(weights)
    squared = (predictions - targets[:, None, :, :]).square()
    per_head = torch.mean(
        squared * weights[None, None, :, None], dim=(2, 3)
    )
    best, assignments = torch.min(per_head, dim=1)

    pair_distances = []
    for left in range(predictions.shape[1]):
        for right in range(left + 1, predictions.shape[1]):
            pair_distances.append(torch.mean(
                (predictions[:, left] - predictions[:, right]).square(),
                dim=(1, 2),
            ))
    if pair_distances:
        distances = torch.stack(pair_distances, dim=1)
        diversity = torch.relu(
            float(diversity_margin) - distances
        ).mean()
    else:
        diversity = predictions.new_zeros(())
    loss = best.mean() + float(diversity_weight) * diversity
    return loss, {
        "best_mse": best.mean(),
        "diversity_penalty": diversity,
        "assignments": assignments,
        "per_head_mse": per_head,
    }


def normalized_to_physical(sequences, lower, upper):
    values = np.asarray(sequences, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return np.clip(
        0.5 * (lower + upper)
        + 0.5 * (upper - lower) * values,
        lower,
        upper,
    )


class FrozenManeuverProposalPolicy:
    """Inference-only adapter for a frozen proposal-only Actor checkpoint."""

    def __init__(self, model, observation_mean, observation_std, *, device):
        self.model = model.eval()
        self.device = torch.device(device)
        self.observation_mean = np.asarray(
            observation_mean, dtype=np.float32
        ).reshape(-1)
        self.observation_std = np.asarray(
            observation_std, dtype=np.float32
        ).reshape(-1)
        if (
            self.observation_mean.shape != (self.model.observation_dim,)
            or self.observation_std.shape != self.observation_mean.shape
            or not np.isfinite(self.observation_mean).all()
            or not np.isfinite(self.observation_std).all()
            or np.any(self.observation_std <= 0.0)
        ):
            raise ValueError("maneuver Actor normalizer is invalid")

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu"):
        payload = torch.load(
            path, map_location=torch.device(device), weights_only=False
        )
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported maneuver Actor checkpoint schema")
        if payload.get("execution_authority") != "proposal_only":
            raise ValueError("maneuver Actor checkpoint is not proposal-only")
        if not bool(payload.get("student_input_causal_only", False)):
            raise ValueError("maneuver Actor checkpoint input is not causal")
        config = dict(payload["model_config"])
        model = ManeuverProposalActor(
            int(payload["observation_dim"]),
            heads=int(config["heads"]),
            horizon=int(config["horizon"]),
            action_dim=int(config["action_dim"]),
            hidden_sizes=tuple(config["hidden_sizes"]),
        ).to(torch.device(device))
        model.load_state_dict(payload["model"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(
            model,
            payload["observation_mean"],
            payload["observation_std"],
            device=device,
        )

    @property
    def heads(self):
        return int(self.model.heads)

    @property
    def horizon(self):
        return int(self.model.horizon)

    @property
    def action_dim(self):
        return int(self.model.action_dim)

    def propose(self, raw_observation, action_spec):
        raw = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
        if raw.shape != self.observation_mean.shape or not np.isfinite(raw).all():
            raise ValueError("maneuver Actor observation is invalid")
        normalized = (raw - self.observation_mean) / self.observation_std
        tensor = torch.as_tensor(
            normalized[None, :], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            sequences = self.model(tensor)[0].cpu().numpy()
        physical = normalized_to_physical(
            sequences, action_spec.lower, action_spec.upper
        )
        if physical.shape != (
            self.heads, self.horizon, self.action_dim
        ):
            raise ValueError("maneuver Actor proposal geometry is invalid")
        return physical


__all__ = [
    "FrozenManeuverProposalPolicy",
    "ManeuverProposalActor",
    "best_of_m_loss",
    "normalized_to_physical",
]
