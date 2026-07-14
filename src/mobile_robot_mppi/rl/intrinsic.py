"""Intrinsic exploration bonuses for RL-guided MPPI training.

The first implementation deliberately stays small and auditable: it rewards
visiting a new odometry pose bin within the current episode.  It never reads
simulator obstacle geometry or collision-free routes, and it is disabled by
default.  The bonus is a training signal only; validation uses extrinsic
reward and task metrics.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class EpisodicPoseCountConfig:
    """Configuration for an episodic count-based pose novelty bonus."""

    enabled: bool = False
    weight: float = 0.05
    position_bin_size: float = 0.25
    heading_bins: int = 8
    count_exponent: float = 0.5
    maximum_bonus: float = 0.10

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            weight=float(values.get("weight", 0.05)),
            position_bin_size=float(values.get("position_bin_size", 0.25)),
            heading_bins=int(values.get("heading_bins", 8)),
            count_exponent=float(values.get("count_exponent", 0.5)),
            maximum_bonus=float(values.get("maximum_bonus", 0.10)),
        )

    def validate(self):
        numeric = (
            self.weight,
            self.position_bin_size,
            self.count_exponent,
            self.maximum_bonus,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("intrinsic exploration parameters must be finite")
        if self.weight < 0.0 or self.maximum_bonus < 0.0:
            raise ValueError("intrinsic exploration bonuses must be non-negative")
        if self.position_bin_size <= 0.0:
            raise ValueError("intrinsic exploration position_bin_size must be positive")
        if self.heading_bins <= 0:
            raise ValueError("intrinsic exploration heading_bins must be positive")
        if self.count_exponent <= 0.0:
            raise ValueError("intrinsic exploration count_exponent must be positive")

    def to_dict(self):
        return asdict(self)


class EpisodicPoseCountBonus:
    """Reward odometry pose bins that have been visited fewer times.

    A bin count is reset at every episode.  The first visit receives
    ``min(maximum_bonus, weight)`` and repeated visits decay as
    ``weight / count**count_exponent``.  The initial reset pose is registered
    before the first action, so standing still cannot repeatedly earn the
    first-visit bonus.
    """

    def __init__(self, config=None):
        if isinstance(config, EpisodicPoseCountConfig):
            self.config = config
        else:
            self.config = EpisodicPoseCountConfig.from_mapping(config)
        self.config.validate()
        self.counts = {}
        self.last_key = None

    @staticmethod
    def _pose_values(pose):
        if all(hasattr(pose, name) for name in ("x", "y", "theta")):
            values = (pose.x, pose.y, pose.theta)
        else:
            array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if array.size < 3:
                raise ValueError("intrinsic exploration pose requires x, y, theta")
            values = array[:3]
        values = tuple(float(value) for value in values)
        if not np.isfinite(values).all():
            raise ValueError("intrinsic exploration pose must be finite")
        return values

    def _key(self, pose):
        x, y, theta = self._pose_values(pose)
        size = self.config.position_bin_size
        # Use the half-open interval [-pi, pi) so +pi and -pi map to the
        # same periodic heading bin.
        wrapped = (theta + math.pi) % (2.0 * math.pi) - math.pi
        heading_fraction = (wrapped + math.pi) / (2.0 * math.pi)
        heading_index = int(math.floor(heading_fraction * self.config.heading_bins))
        heading_index = min(max(heading_index, 0), self.config.heading_bins - 1)
        return (
            int(math.floor(x / size)),
            int(math.floor(y / size)),
            heading_index,
        )

    def reset(self, pose):
        self.counts = {}
        self.last_key = None
        if not self.config.enabled:
            return self.diagnostics(0.0, None, 0)
        key = self._key(pose)
        self.counts[key] = 1
        self.last_key = key
        return self.diagnostics(0.0, key, 1)

    def observe(self, pose):
        if not self.config.enabled:
            return 0.0, self.diagnostics(0.0, None, 0)
        key = self._key(pose)
        count = int(self.counts.get(key, 0)) + 1
        self.counts[key] = count
        self.last_key = key
        raw = self.config.weight / float(count ** self.config.count_exponent)
        bonus = float(min(self.config.maximum_bonus, raw))
        if not math.isfinite(bonus):
            raise FloatingPointError("intrinsic exploration bonus produced NaN or Inf")
        return bonus, self.diagnostics(bonus, key, count)

    def diagnostics(self, bonus, key, count):
        return {
            "enabled": bool(self.config.enabled),
            "bonus": float(bonus),
            "bin": None if key is None else list(key),
            "bin_count": int(count),
            "unique_bins": int(len(self.counts)),
        }
