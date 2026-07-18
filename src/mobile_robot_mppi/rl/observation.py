"""Observable, normalized inputs for an RL sampling-prior policy.

The encoder deliberately uses LaserScan rather than simulator obstacle truth.
This keeps the policy input compatible with the existing robot sensing chain.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ObservationEncoderConfig:
    lidar_sectors: int = 36
    lidar_max_range: float = 4.0
    goal_distance_scale: float = 5.0
    velocity_scale: float = 0.5
    yaw_rate_scale: float = 1.2
    include_absolute_pose: bool = False
    include_previous_action: bool = True
    include_safety_state: bool = True
    history_frames: int = 1

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            lidar_sectors=int(values.get("lidar_sectors", 36)),
            lidar_max_range=float(values.get("lidar_max_range", 4.0)),
            goal_distance_scale=float(values.get("goal_distance_scale", 5.0)),
            velocity_scale=float(values.get("velocity_scale", 0.5)),
            yaw_rate_scale=float(values.get("yaw_rate_scale", 1.2)),
            include_absolute_pose=bool(values.get("include_absolute_pose", False)),
            include_previous_action=bool(values.get("include_previous_action", True)),
            include_safety_state=bool(values.get("include_safety_state", True)),
            history_frames=int(values.get("history_frames", 1)),
        )

    def validate(self):
        positive = (
            self.lidar_sectors,
            self.lidar_max_range,
            self.goal_distance_scale,
            self.velocity_scale,
            self.yaw_rate_scale,
        )
        numeric = np.asarray(positive, dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
            raise ValueError("RL observation scales and lidar_sectors must be positive")
        if self.history_frames <= 0:
            raise ValueError("RL observation history_frames must be positive")

    def to_dict(self):
        return asdict(self)


class ObservationEncoder:
    """Encode robot-relative task state and sectorized LaserScan.

    The vector contains body-frame goal displacement, goal distance/bearing,
    measured body velocity, heading sin/cos, optional previous command and
    safety state, followed by normalized sector minima and a scan-valid flag.
    Absolute pose is optional and disabled by default to improve transfer.
    """

    def __init__(self, config, action_spec):
        self.config = (
            config
            if isinstance(config, ObservationEncoderConfig)
            else ObservationEncoderConfig.from_mapping(config)
        )
        self.config.validate()
        self.action_spec = action_spec
        self._history = []

    @property
    def frame_dimension(self):
        # body dx/dy, distance, bearing, v, omega, sin/cos(theta)
        size = 8 + self.config.lidar_sectors + 1
        if self.config.include_absolute_pose:
            size += 2
        if self.config.include_previous_action:
            size += self.action_spec.dimension
        if self.config.include_safety_state:
            size += 1
        return size

    @property
    def dimension(self):
        return self.frame_dimension * self.config.history_frames

    def reset(self):
        self._history = []

    def _scan_features(self, scan):
        sectors = self.config.lidar_sectors
        if scan is None:
            return np.ones(sectors, dtype=np.float32), 0.0
        ranges = np.asarray(scan.ranges, dtype=np.float64).reshape(-1)
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(scan.range_min))
            & (ranges <= float(scan.range_max))
        )
        sanitized = np.where(valid, ranges, self.config.lidar_max_range)
        sanitized = np.clip(sanitized, 0.0, self.config.lidar_max_range)
        chunks = np.array_split(sanitized, sectors)
        minima = np.asarray(
            [float(np.min(chunk)) if chunk.size else self.config.lidar_max_range for chunk in chunks],
            dtype=np.float64,
        )
        return (minima / self.config.lidar_max_range).astype(np.float32), 1.0

    def _normalized_action(self, action):
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape != (self.action_spec.dimension,):
            raise ValueError("previous action does not match configured action space")
        center = 0.5 * (self.action_spec.upper + self.action_spec.lower)
        half_range = 0.5 * (self.action_spec.upper - self.action_spec.lower)
        return np.clip((value - center) / half_range, -1.0, 1.0)

    def encode(
        self,
        observation,
        reference,
        previous_action=None,
        safety_override=False,
    ):
        pose = observation.pose
        target = reference.target_at(observation.timestamp, pose.as_array())
        dx = float(target.pose.x - pose.x)
        dy = float(target.pose.y - pose.y)
        cosine = float(np.cos(pose.theta))
        sine = float(np.sin(pose.theta))
        body_dx = cosine * dx + sine * dy
        body_dy = -sine * dx + cosine * dy
        distance = float(np.hypot(dx, dy))
        bearing = float(np.arctan2(body_dy, body_dx))
        scale = self.config.goal_distance_scale
        features = [
            np.clip(body_dx / scale, -1.0, 1.0),
            np.clip(body_dy / scale, -1.0, 1.0),
            np.clip(distance / scale, 0.0, 1.0),
            bearing / np.pi,
            np.clip(observation.twist.v / self.config.velocity_scale, -2.0, 2.0),
            np.clip(observation.twist.omega / self.config.yaw_rate_scale, -2.0, 2.0),
            sine,
            cosine,
        ]
        if self.config.include_absolute_pose:
            features.extend((pose.x / scale, pose.y / scale))
        if self.config.include_previous_action:
            if previous_action is None:
                previous_action = np.zeros(self.action_spec.dimension, dtype=np.float64)
            features.extend(self._normalized_action(previous_action).tolist())
        if self.config.include_safety_state:
            features.append(float(bool(safety_override)))
        scan_features, scan_valid = self._scan_features(observation.scan)
        features.extend(scan_features.tolist())
        features.append(scan_valid)
        frame = np.asarray(features, dtype=np.float32)
        if frame.shape != (self.frame_dimension,) or not np.isfinite(frame).all():
            raise FloatingPointError("RL observation encoder produced invalid features")
        if not self._history:
            self._history = [
                frame.copy() for _ in range(self.config.history_frames)
            ]
        else:
            self._history.append(frame.copy())
            self._history = self._history[-self.config.history_frames :]
        encoded = np.concatenate(self._history).astype(np.float32, copy=False)
        if encoded.shape != (self.dimension,) or not np.isfinite(encoded).all():
            raise FloatingPointError("RL observation history produced invalid features")
        return encoded


class RunningNormalizer:
    """Welford observation statistics shared by training, OOD scoring and inference."""

    def __init__(self, dimension, min_std=0.05, clip=10.0):
        self.dimension = int(dimension)
        self.min_std = float(min_std)
        self.clip = float(clip)
        if self.dimension <= 0 or self.min_std <= 0.0 or self.clip <= 0.0:
            raise ValueError("normalizer dimension, min_std and clip must be positive")
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)

    @property
    def std(self):
        if self.count < 2:
            return np.ones(self.dimension, dtype=np.float64)
        return np.maximum(np.sqrt(self.m2 / float(self.count - 1)), self.min_std)

    def update(self, values):
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.ndim != 2 or batch.shape[1] != self.dimension or not np.isfinite(batch).all():
            raise ValueError("normalizer input must be finite [batch, observation_dim]")
        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / float(self.count)
            self.m2 += delta * (row - self.mean)

    def normalize(self, values):
        data = np.asarray(values, dtype=np.float32)
        if data.shape[-1] != self.dimension or not np.isfinite(data).all():
            raise ValueError("normalizer input has an invalid final dimension")
        normalized = (data - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def ood_score(self, values):
        """Maximum absolute standardized feature distance.

        This is an auditable heuristic distance, not a calibrated probability
        and not a formal epistemic-uncertainty guarantee.
        """
        normalized = np.abs(self.normalize(values))
        return float(np.max(normalized))

    def state_dict(self):
        return {
            "dimension": self.dimension,
            "min_std": self.min_std,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
        }

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state["dimension"], state.get("min_std", 0.05), state.get("clip", 10.0))
        result.count = int(state["count"])
        result.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        result.m2 = np.asarray(state["m2"], dtype=np.float64).copy()
        if result.mean.shape != (result.dimension,) or result.m2.shape != (result.dimension,):
            raise ValueError("normalizer checkpoint dimensions are inconsistent")
        if result.count < 0:
            raise ValueError("normalizer checkpoint count cannot be negative")
        if not np.isfinite(result.mean).all() or not np.isfinite(result.m2).all():
            raise ValueError("normalizer checkpoint statistics must be finite")
        # Welford's second central moment is non-negative.  Tolerate only
        # floating-point roundoff, never a materially corrupt variance state.
        if np.any(result.m2 < -1e-12):
            raise ValueError("normalizer checkpoint variance cannot be negative")
        result.m2 = np.maximum(result.m2, 0.0)
        return result
