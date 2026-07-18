"""Online, simulator-agnostic competence signals for RL activation."""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ProgressCompetenceConfig:
    """Thresholds for detecting inadequate closed-loop goal progress.

    The detector uses only timestamped goal distance, which is available from
    the ordinary localization/reference chain.  It never reads a scene label,
    simulator geometry, future outcome, or oracle dynamics.
    """

    observation_window_s: float = 2.0
    minimum_window_coverage_s: float = 1.8
    full_activation_progress_m: float = 0.04
    zero_activation_progress_m: float = 0.20
    activation_hold_s: float = 1.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            observation_window_s=float(
                values.get("observation_window_s", 2.0)
            ),
            minimum_window_coverage_s=float(
                values.get("minimum_window_coverage_s", 1.8)
            ),
            full_activation_progress_m=float(
                values.get("full_activation_progress_m", 0.04)
            ),
            zero_activation_progress_m=float(
                values.get("zero_activation_progress_m", 0.20)
            ),
            activation_hold_s=float(
                values.get("activation_hold_s", 1.0)
            ),
        )

    def validate(self):
        numeric = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("progress competence thresholds must be finite")
        if self.observation_window_s <= 0.0:
            raise ValueError("progress observation window must be positive")
        if (
            self.minimum_window_coverage_s <= 0.0
            or self.minimum_window_coverage_s > self.observation_window_s
        ):
            raise ValueError(
                "progress minimum coverage must be in (0, observation window]"
            )
        if (
            self.full_activation_progress_m < 0.0
            or self.zero_activation_progress_m
            <= self.full_activation_progress_m
        ):
            raise ValueError(
                "zero-activation progress must exceed non-negative "
                "full-activation progress"
            )
        if self.activation_hold_s < 0.0:
            raise ValueError("progress activation hold must be non-negative")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ProgressCompetence:
    progress_m: float
    window_coverage_s: float
    ready: bool
    stagnation_activation: float
    held: bool

    def to_dict(self):
        return asdict(self)


def _stagnation_activation(progress_m, config):
    if progress_m <= config.full_activation_progress_m:
        return 1.0
    if progress_m >= config.zero_activation_progress_m:
        return 0.0
    span = (
        config.zero_activation_progress_m
        - config.full_activation_progress_m
    )
    return float(
        (config.zero_activation_progress_m - progress_m) / span
    )


class ProgressCompetenceGate:
    """Stateful progress detector with a short anti-chatter activation hold."""

    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, ProgressCompetenceConfig)
            else ProgressCompetenceConfig.from_mapping(config)
        )
        self.config.validate()
        self.history = deque()
        self.last_timestamp = None
        self.hold_until = None
        self.hold_activation = 0.0

    def reset(self):
        self.history.clear()
        self.last_timestamp = None
        self.hold_until = None
        self.hold_activation = 0.0

    def update(self, timestamp, goal_distance):
        now = float(timestamp)
        distance = float(goal_distance)
        if not np.isfinite((now, distance)).all() or distance < 0.0:
            raise ValueError(
                "progress competence requires finite time and goal distance"
            )
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError(
                "progress competence timestamps must be strictly increasing"
            )
        self.last_timestamp = now
        self.history.append((now, distance))
        cutoff = now - self.config.observation_window_s
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()

        oldest_time, oldest_distance = self.history[0]
        coverage = max(0.0, now - float(oldest_time))
        progress = float(oldest_distance - distance)
        ready = bool(
            coverage + 1e-12
            >= self.config.minimum_window_coverage_s
        )
        raw_activation = (
            _stagnation_activation(progress, self.config)
            if ready else 0.0
        )

        hold_active = bool(
            self.hold_until is not None
            and now <= self.hold_until + 1e-12
        )
        if raw_activation > 0.0:
            self.hold_activation = max(
                self.hold_activation if hold_active else 0.0,
                raw_activation,
            )
            self.hold_until = now + self.config.activation_hold_s
            hold_active = self.config.activation_hold_s > 0.0
        elif not hold_active:
            self.hold_activation = 0.0
            self.hold_until = None
        activation = max(
            raw_activation,
            self.hold_activation if hold_active else 0.0,
        )
        return ProgressCompetence(
            progress_m=progress,
            window_coverage_s=coverage,
            ready=ready,
            stagnation_activation=float(activation),
            held=bool(hold_active and activation > raw_activation),
        )
