"""Interpretable contextual-bandit selection of MPPI sampling covariance.

The bandit acts above MPPI: it chooses one audited covariance option from
features of the task reference, while MPPI still optimizes the control
sequence and the ordinary safety chain remains downstream.  No simulator
scene label, obstacle ground truth, or future plant outcome is an input.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mobile_robot_mppi.policies.priors import PriorOutput, ProposalDistribution


REFERENCE_GEOMETRY_FEATURE_NAMES = (
    "bias",
    "absolute_turn_fraction",
    "route_indirectness",
    "peak_curvature_fraction",
    "absolute_turn_fraction_squared",
    "route_indirectness_squared",
    "peak_curvature_fraction_squared",
    "turn_x_indirectness",
    "turn_x_peak_curvature",
    "indirectness_x_peak_curvature",
)


def _polyline_arrays(points):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("polyline points must have shape [N,2+] with N >= 2")
    values = values[:, :2]
    if not np.isfinite(values).all():
        raise ValueError("polyline points must be finite")
    segments = np.diff(values, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("polyline cannot contain duplicate consecutive points")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return values, segments, lengths, cumulative


def polyline_pose_at_progress(points, progress):
    """Interpolate route position and tangent heading at metric progress."""

    values, segments, lengths, cumulative = _polyline_arrays(points)
    distance = float(np.clip(float(progress), 0.0, cumulative[-1]))
    index = min(
        max(int(np.searchsorted(cumulative, distance, side="right") - 1), 0),
        len(lengths) - 1,
    )
    fraction = (distance - cumulative[index]) / lengths[index]
    point = values[index] + fraction * segments[index]
    theta = float(np.arctan2(segments[index, 1], segments[index, 0]))
    return point, theta


def polyline_window(points, start_progress, window_distance):
    """Extract an interpolated forward route window without duplicate points."""

    values, _segments, _lengths, cumulative = _polyline_arrays(points)
    window_distance = float(window_distance)
    if not np.isfinite(window_distance) or window_distance <= 0.0:
        raise ValueError("polyline window distance must be positive")
    total = float(cumulative[-1])
    start = float(np.clip(float(start_progress), 0.0, total))
    end = min(total, start + window_distance)
    if end - start <= 1e-8:
        start = max(0.0, total - min(window_distance, total))
        end = total
    start_point, _ = polyline_pose_at_progress(values, start)
    end_point, _ = polyline_pose_at_progress(values, end)
    interior = [
        values[index]
        for index in range(1, len(values) - 1)
        if start + 1e-9 < cumulative[index] < end - 1e-9
    ]
    result = np.asarray([start_point, *interior, end_point], dtype=np.float64)
    keep = np.concatenate((
        np.asarray((True,)),
        np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-9,
    ))
    result = result[keep]
    if result.shape[0] < 2:
        raise ValueError("polyline window is degenerate")
    return result


def project_polyline_progress(points, position, minimum=0.0, maximum=None):
    """Project an XY point onto a bounded progress interval of a polyline."""

    values, segments, lengths, cumulative = _polyline_arrays(points)
    position = np.asarray(position, dtype=np.float64).reshape(-1)
    if position.size < 2 or not np.isfinite(position[:2]).all():
        raise ValueError("projection position must contain finite x/y")
    fractions = np.sum((position[:2][None, :] - values[:-1]) * segments, axis=1)
    fractions = np.clip(fractions / np.square(lengths), 0.0, 1.0)
    projections = values[:-1] + fractions[:, None] * segments
    progress = cumulative[:-1] + fractions * lengths
    lower = float(minimum)
    upper = float(cumulative[-1] if maximum is None else maximum)
    admissible = (progress >= lower - 1e-9) & (progress <= upper + 1e-9)
    if not np.any(admissible):
        return float(np.clip(lower, 0.0, cumulative[-1]))
    distances = np.linalg.norm(projections - position[:2][None, :], axis=1)
    distances = np.where(admissible, distances, np.inf)
    return float(progress[int(np.argmin(distances))])


def polyline_geometry_features(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return bounded, rigid-transform-invariant polyline features.

    These quantities are available from the same route supplied to MPPI.  In
    particular, the encoder does not use a scene identifier or MuJoCo state.
    """

    values, segments, lengths, _cumulative = _polyline_arrays(points)
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    turns = np.diff(headings)
    total_turn = float(np.sum(np.abs(turns)))
    route_length = float(np.sum(lengths))
    direct_distance = float(np.linalg.norm(values[-1] - values[0]))
    indirectness = 1.0 - direct_distance / route_length
    if turns.size:
        local_lengths = 0.5 * (lengths[:-1] + lengths[1:])
        peak_curvature = float(np.max(np.abs(turns) / local_lengths))
    else:
        peak_curvature = 0.0
    turn = float(np.clip(total_turn / (2.0 * np.pi), 0.0, 2.0))
    indirect = float(np.clip(indirectness, 0.0, 1.0))
    curvature = float(np.clip(peak_curvature / 3.0, 0.0, 2.0))
    features = np.asarray(
        (
            1.0,
            turn,
            indirect,
            curvature,
            turn * turn,
            indirect * indirect,
            curvature * curvature,
            turn * indirect,
            turn * curvature,
            indirect * curvature,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise FloatingPointError("polyline feature encoder produced NaN or Inf")
    return features


@dataclass(frozen=True)
class BanditDecision:
    action: str
    predicted_rewards: Mapping[str, float]
    confidence_widths: Mapping[str, float]


class LinUCBCovarianceBandit:
    """Ridge-linear reward critic with an optional LinUCB exploration bonus."""

    def __init__(self, actions, feature_dim, ridge=1.0, exploration_alpha=0.0):
        self.actions = tuple(str(value) for value in actions)
        self.feature_dim = int(feature_dim)
        self.ridge = float(ridge)
        self.exploration_alpha = float(exploration_alpha)
        if len(self.actions) < 2 or len(set(self.actions)) != len(self.actions):
            raise ValueError("bandit requires at least two unique actions")
        if self.feature_dim <= 0 or not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("feature_dim and ridge must be positive")
        if not np.isfinite(self.exploration_alpha) or self.exploration_alpha < 0.0:
            raise ValueError("exploration_alpha must be non-negative")
        identity = self.ridge * np.eye(self.feature_dim, dtype=np.float64)
        self._a = {action: identity.copy() for action in self.actions}
        self._b = {
            action: np.zeros(self.feature_dim, dtype=np.float64)
            for action in self.actions
        }
        self._inverse_cache = {action: None for action in self.actions}
        self._theta_cache = {action: None for action in self.actions}
        self.counts = {action: 0 for action in self.actions}

    def _feature(self, features):
        value = np.asarray(features, dtype=np.float64).reshape(-1)
        if value.shape != (self.feature_dim,) or not np.isfinite(value).all():
            raise ValueError("bandit feature vector has an invalid shape or value")
        return value

    def update(self, features, action, reward):
        value = self._feature(features)
        action = str(action)
        reward = float(reward)
        if action not in self._a:
            raise ValueError("unknown bandit action: %s" % action)
        if not np.isfinite(reward):
            raise ValueError("bandit reward must be finite")
        self._a[action] += np.outer(value, value)
        self._b[action] += reward * value
        self._inverse_cache[action] = None
        self._theta_cache[action] = None
        self.counts[action] += 1

    def predict(self, features):
        value = self._feature(features)
        means = {}
        widths = {}
        for action in self.actions:
            inverse = self._inverse_cache[action]
            theta = self._theta_cache[action]
            if inverse is None or theta is None:
                inverse = np.linalg.inv(self._a[action])
                theta = inverse @ self._b[action]
                self._inverse_cache[action] = inverse
                self._theta_cache[action] = theta
            means[action] = float(value @ theta)
            widths[action] = float(np.sqrt(max(0.0, value @ inverse @ value)))
        return means, widths

    def decide(self, features, explore=False):
        means, widths = self.predict(features)
        alpha = self.exploration_alpha if explore else 0.0
        scores = {
            action: means[action] + alpha * widths[action]
            for action in self.actions
        }
        # Explicit tuple tie-break makes the deployed result independent of a
        # dict implementation while preserving the configured action order.
        index = {action: -position for position, action in enumerate(self.actions)}
        selected = max(self.actions, key=lambda action: (scores[action], index[action]))
        return BanditDecision(selected, means, widths)

    def state_dict(self):
        return {
            "schema_version": 1,
            "model_class": type(self).__name__,
            "actions": list(self.actions),
            "feature_dim": self.feature_dim,
            "ridge": self.ridge,
            "exploration_alpha": self.exploration_alpha,
            "a": {key: value.tolist() for key, value in self._a.items()},
            "b": {key: value.tolist() for key, value in self._b.items()},
            "counts": dict(self.counts),
        }

    @classmethod
    def from_state_dict(cls, state):
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("unsupported contextual-bandit schema")
        result = cls(
            state["actions"],
            state["feature_dim"],
            state.get("ridge", 1.0),
            state.get("exploration_alpha", 0.0),
        )
        for action in result.actions:
            result._a[action] = np.asarray(state["a"][action], dtype=np.float64)
            result._b[action] = np.asarray(state["b"][action], dtype=np.float64)
            result.counts[action] = int(state.get("counts", {}).get(action, 0))
            if result._a[action].shape != (result.feature_dim, result.feature_dim):
                raise ValueError("checkpoint bandit A matrix has an invalid shape")
            if result._b[action].shape != (result.feature_dim,):
                raise ValueError("checkpoint bandit b vector has an invalid shape")
        return result


class ContextualBanditCovariancePrior:
    """Deploy a frozen contextual-bandit decision through the prior interface."""

    def __init__(self, baseline_prior, noise_sigma, bandit, action_scales):
        self.baseline_prior = baseline_prior
        self.noise_sigma = np.asarray(noise_sigma, dtype=np.float64).reshape(-1)
        self.bandit = bandit
        self.action_scales = {
            str(key): np.asarray(value, dtype=np.float64).reshape(-1)
            for key, value in dict(action_scales).items()
        }
        if set(self.action_scales) != set(self.bandit.actions):
            raise ValueError("bandit actions and covariance scales differ")
        for value in self.action_scales.values():
            if value.shape != self.noise_sigma.shape or np.any(value <= 0.0):
                raise ValueError("covariance scales must be positive and match noise_sigma")

    def reset(self):
        reset = getattr(self.baseline_prior, "reset", None)
        if callable(reset):
            reset()

    def propose(self, observation, reference, horizon, action_spec):
        baseline = self.baseline_prior.propose(
            observation, reference, horizon, action_spec
        )
        if not hasattr(reference, "points"):
            raise ValueError("contextual covariance currently requires a polyline reference")
        features = polyline_geometry_features(reference.points)
        decision = self.bandit.decide(features, explore=False)
        scale = self.action_scales[decision.action]
        standard_deviation = self.noise_sigma * scale
        metadata = dict(baseline.metadata)
        metadata.update({
            "type": "contextual_bandit_covariance",
            "bandit_action": decision.action,
            "covariance_scale": scale.tolist(),
            "covariance_standard_deviation": standard_deviation.tolist(),
            "context_features": features.tolist(),
            "predicted_rewards": dict(decision.predicted_rewards),
            "confidence_widths": dict(decision.confidence_widths),
        })
        return PriorOutput(
            baseline.mean,
            np.diag(np.square(standard_deviation)),
            metadata,
            proposals=(
                ProposalDistribution(
                    "rl",
                    baseline.mean,
                    np.diag(np.square(standard_deviation)),
                ),
                ProposalDistribution(
                    "base", baseline.mean, baseline.covariance
                ),
            ),
        )

    @classmethod
    def from_checkpoint(cls, path, baseline_prior, noise_sigma):
        with Path(path).open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if tuple(state.get("feature_names", ())) != REFERENCE_GEOMETRY_FEATURE_NAMES:
            raise ValueError("checkpoint reference feature schema differs")
        bandit = LinUCBCovarianceBandit.from_state_dict(state["bandit"])
        return cls(
            baseline_prior,
            noise_sigma,
            bandit,
            state["action_scales"],
        )


__all__ = [
    "BanditDecision",
    "ContextualBanditCovariancePrior",
    "LinUCBCovarianceBandit",
    "REFERENCE_GEOMETRY_FEATURE_NAMES",
    "polyline_geometry_features",
    "polyline_pose_at_progress",
    "polyline_window",
    "project_polyline_progress",
]
