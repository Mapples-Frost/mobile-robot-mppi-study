"""Conservative Gaussian-mixture collision risk for batched robot paths.

The implementation deliberately uses only predicted obstacle distributions.
Ground-truth states, motion modes, and change labels are absent from the
public contract.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _readonly_finite_array(value, name):
    result = np.asarray(value, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError("%s must contain only finite values" % name)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GaussianMixtureObstacleForecast:
    """Planner-aligned position forecast for one circular obstacle."""

    timestamp: float
    dt: float
    component_means: np.ndarray
    component_covariances: np.ndarray
    component_weights: np.ndarray
    radius_m: float
    source: str = "unknown"

    def __post_init__(self):
        timestamp = float(self.timestamp)
        dt = float(self.dt)
        radius = float(self.radius_m)
        if not np.isfinite(timestamp):
            raise ValueError("forecast timestamp must be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("forecast dt must be finite and positive")
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("obstacle radius must be finite and non-negative")

        means = _readonly_finite_array(
            self.component_means, "component means"
        )
        covariances = _readonly_finite_array(
            self.component_covariances, "component covariances"
        )
        weights = _readonly_finite_array(
            self.component_weights, "component weights"
        )
        if means.ndim != 3 or means.shape[2] != 2:
            raise ValueError("component means must have shape [H,M,2]")
        horizon, modes, _ = means.shape
        if horizon <= 0 or modes <= 0:
            raise ValueError("forecast must contain at least one step and mode")
        if covariances.shape != (horizon, modes, 2, 2):
            raise ValueError(
                "component covariances must have shape [H,M,2,2]"
            )
        if weights.shape != (horizon, modes):
            raise ValueError("component weights must have shape [H,M]")
        if np.any(weights < -1.0e-12):
            raise ValueError("component weights must be non-negative")
        if not np.allclose(
            weights.sum(axis=1), 1.0, atol=1.0e-10, rtol=0.0
        ):
            raise ValueError("component weights must sum to one per step")
        if not np.allclose(
            covariances,
            np.swapaxes(covariances, -1, -2),
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("component covariances must be symmetric")
        minimum_eigenvalue = float(
            np.linalg.eigvalsh(covariances).min()
        )
        if minimum_eigenvalue < -1.0e-9:
            raise ValueError("component covariances must be PSD")

        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "component_means", means)
        object.__setattr__(self, "component_covariances", covariances)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(self, "source", str(self.source))

    @property
    def horizon(self):
        return int(self.component_means.shape[0])

    @property
    def mode_count(self):
        return int(self.component_means.shape[1])

    @classmethod
    def from_prediction(
        cls,
        prediction,
        timestamp,
        dt,
        radius_m,
        source="unknown",
    ):
        """Convert an IMM mixture or one-component Gaussian prediction."""

        if hasattr(prediction, "component_means"):
            means = np.asarray(prediction.component_means)[..., :2]
            covariances = np.asarray(
                prediction.component_covariances
            )[..., :2, :2]
            weights = np.asarray(prediction.mode_probabilities)
        else:
            means = np.asarray(prediction.means)[:, None, :2]
            if prediction.covariances is None:
                covariances = np.zeros(
                    (means.shape[0], 1, 2, 2), dtype=np.float64
                )
            else:
                covariances = np.asarray(
                    prediction.covariances
                )[:, None, :2, :2]
            weights = np.ones((means.shape[0], 1), dtype=np.float64)
        return cls(
            timestamp=timestamp,
            dt=dt,
            component_means=means,
            component_covariances=covariances,
            component_weights=weights,
            radius_m=radius_m,
            source=source,
        )


@dataclass(frozen=True)
class CollisionRiskConfig:
    robot_radius_m: float = 0.25
    safety_margin_m: float = 0.10
    minimum_position_std_m: float = 0.01
    hard_probability_threshold: float = 0.20

    def __post_init__(self):
        values = (
            float(self.robot_radius_m),
            float(self.safety_margin_m),
            float(self.minimum_position_std_m),
            float(self.hard_probability_threshold),
        )
        if not np.isfinite(values).all():
            raise ValueError("collision risk configuration must be finite")
        if values[0] < 0.0 or values[1] < 0.0 or values[2] <= 0.0:
            raise ValueError(
                "radii/margin must be non-negative and minimum std positive"
            )
        if not 0.0 < values[3] <= 1.0:
            raise ValueError(
                "hard collision probability threshold must be in (0,1]"
            )
        object.__setattr__(self, "robot_radius_m", values[0])
        object.__setattr__(self, "safety_margin_m", values[1])
        object.__setattr__(self, "minimum_position_std_m", values[2])
        object.__setattr__(self, "hard_probability_threshold", values[3])

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]):
        return cls(
            robot_radius_m=float(values.get("robot_radius_m", 0.25)),
            safety_margin_m=float(values.get("safety_margin_m", 0.10)),
            minimum_position_std_m=float(
                values.get("minimum_position_std_m", 0.01)
            ),
            hard_probability_threshold=float(
                values.get("hard_probability_threshold", 0.20)
            ),
        )


@dataclass(frozen=True)
class CollisionRiskEvaluation:
    step_probability_upper_bound: np.ndarray
    accumulated_probability_mass: np.ndarray
    horizon_union_bound: np.ndarray
    maximum_step_probability: np.ndarray
    hard_violation: np.ndarray

    def __post_init__(self):
        step = _readonly_finite_array(
            self.step_probability_upper_bound,
            "step probability upper bound",
        )
        accumulated = _readonly_finite_array(
            self.accumulated_probability_mass,
            "accumulated probability mass",
        )
        union = _readonly_finite_array(
            self.horizon_union_bound, "horizon union bound"
        )
        maximum = _readonly_finite_array(
            self.maximum_step_probability, "maximum step probability"
        )
        hard = np.asarray(self.hard_violation, dtype=bool).copy()
        if step.ndim != 2:
            raise ValueError("step probabilities must have shape [K,H]")
        batch = step.shape[0]
        if (
            accumulated.shape != (batch,)
            or union.shape != (batch,)
            or maximum.shape != (batch,)
            or hard.shape != (batch,)
        ):
            raise ValueError("collision risk summaries must have shape [K]")
        if (
            np.any(step < -1.0e-12)
            or np.any(step > 1.0 + 1.0e-12)
            or np.any(union < -1.0e-12)
            or np.any(union > 1.0 + 1.0e-12)
            or np.any(maximum < -1.0e-12)
            or np.any(maximum > 1.0 + 1.0e-12)
            or np.any(accumulated < -1.0e-12)
        ):
            raise FloatingPointError("collision risk output is out of range")
        hard.setflags(write=False)
        object.__setattr__(self, "step_probability_upper_bound", step)
        object.__setattr__(self, "accumulated_probability_mass", accumulated)
        object.__setattr__(self, "horizon_union_bound", union)
        object.__setattr__(self, "maximum_step_probability", maximum)
        object.__setattr__(self, "hard_violation", hard)


def standard_normal_cdf(values):
    """Vectorized normal CDF with maximum absolute error below 8e-8."""

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("normal CDF inputs must be finite")
    absolute = np.abs(values)
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = t * (
        0.319381530
        + t
        * (
            -0.356563782
            + t
            * (
                1.781477937
                + t * (-1.821255978 + t * 1.330274429)
            )
        )
    )
    upper = 1.0 - (
        np.exp(-0.5 * absolute ** 2)
        / np.sqrt(2.0 * np.pi)
    ) * polynomial
    return np.where(values >= 0.0, upper, 1.0 - upper)


def component_collision_probability_upper_bound(
    robot_positions,
    forecast: GaussianMixtureObstacleForecast,
    config: CollisionRiskConfig,
):
    """Return conservative per-component bounds with shape ``[K,H,M]``."""

    positions = np.asarray(robot_positions, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[2] != 2
        or positions.shape[0] <= 0
        or not np.isfinite(positions).all()
    ):
        raise ValueError("robot positions must be finite with shape [K,H,2]")
    horizon = int(positions.shape[1])
    if horizon > forecast.horizon:
        raise ValueError("robot horizon cannot exceed obstacle forecast horizon")

    delta = (
        forecast.component_means[None, :horizon, :, :]
        - positions[:, :, None, :]
    )
    distance = np.linalg.norm(delta, axis=-1)
    safe_distance = np.maximum(distance, 1.0e-15)
    direction = delta / safe_distance[..., None]
    radial_variance = np.einsum(
        "khmi,hmij,khmj->khm",
        direction,
        forecast.component_covariances[:horizon],
        direction,
    )
    if float(radial_variance.min()) < -1.0e-8:
        raise FloatingPointError("radial variance became negative")
    radial_std = np.sqrt(
        np.maximum(radial_variance, 0.0)
        + config.minimum_position_std_m ** 2
    )
    combined_radius = (
        config.robot_radius_m
        + forecast.radius_m
        + config.safety_margin_m
    )
    z_score = (combined_radius - distance) / radial_std
    probability = standard_normal_cdf(z_score)
    probability = np.where(
        distance <= combined_radius, 1.0, probability
    )
    return np.clip(probability, 0.0, 1.0)


def mixture_collision_probability_upper_bound(
    robot_positions,
    forecast: GaussianMixtureObstacleForecast,
    config: CollisionRiskConfig,
):
    component = component_collision_probability_upper_bound(
        robot_positions, forecast, config
    )
    result = np.sum(
        component
        * forecast.component_weights[None, : component.shape[1], :],
        axis=-1,
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("mixture collision risk is not finite")
    return np.clip(result, 0.0, 1.0)


def evaluate_collision_risk(
    robot_positions,
    forecasts: Sequence[GaussianMixtureObstacleForecast],
    config: CollisionRiskConfig,
):
    """Evaluate one or more obstacle forecasts without temporal independence."""

    positions = np.asarray(robot_positions, dtype=np.float64)
    forecasts = tuple(forecasts)
    if not forecasts:
        raise ValueError("collision risk requires at least one forecast")
    if (
        positions.ndim != 3
        or positions.shape[2] != 2
        or positions.shape[0] <= 0
        or positions.shape[1] <= 0
        or not np.isfinite(positions).all()
    ):
        raise ValueError("robot positions must be finite with shape [K,H,2]")
    per_obstacle = [
        mixture_collision_probability_upper_bound(
            positions, forecast, config
        )
        for forecast in forecasts
    ]
    probability_mass_by_step = np.sum(per_obstacle, axis=0)
    step_upper_bound = np.clip(probability_mass_by_step, 0.0, 1.0)
    accumulated = np.sum(probability_mass_by_step, axis=1)
    union = np.clip(accumulated, 0.0, 1.0)
    maximum = np.max(step_upper_bound, axis=1)
    hard = maximum >= config.hard_probability_threshold
    return CollisionRiskEvaluation(
        step_probability_upper_bound=step_upper_bound,
        accumulated_probability_mass=accumulated,
        horizon_union_bound=union,
        maximum_step_probability=maximum,
        hard_violation=hard,
    )


__all__ = [
    "CollisionRiskConfig",
    "CollisionRiskEvaluation",
    "GaussianMixtureObstacleForecast",
    "component_collision_probability_upper_bound",
    "evaluate_collision_risk",
    "mixture_collision_probability_upper_bound",
    "standard_normal_cdf",
]
