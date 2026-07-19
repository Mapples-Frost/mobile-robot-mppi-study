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
    include_path_context: bool = False
    include_residual_context: bool = False
    residual_context_dimension: int = 7
    path_cross_track_scale: float = 1.0
    path_curvature_scale: float = 2.0
    path_remaining_scale: float = 6.0
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
            include_path_context=bool(values.get("include_path_context", False)),
            include_residual_context=bool(
                values.get("include_residual_context", False)
            ),
            residual_context_dimension=int(
                values.get("residual_context_dimension", 7)
            ),
            path_cross_track_scale=float(values.get("path_cross_track_scale", 1.0)),
            path_curvature_scale=float(values.get("path_curvature_scale", 2.0)),
            path_remaining_scale=float(values.get("path_remaining_scale", 6.0)),
            history_frames=int(values.get("history_frames", 1)),
        )

    def validate(self):
        positive = (
            self.lidar_sectors,
            self.lidar_max_range,
            self.goal_distance_scale,
            self.velocity_scale,
            self.yaw_rate_scale,
            self.path_cross_track_scale,
            self.path_curvature_scale,
            self.path_remaining_scale,
        )
        numeric = np.asarray(positive, dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
            raise ValueError("RL observation scales and lidar_sectors must be positive")
        if self.history_frames <= 0:
            raise ValueError("RL observation history_frames must be positive")
        if self.residual_context_dimension <= 0:
            raise ValueError("residual_context_dimension must be positive")

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
        if self.config.include_path_context:
            # signed cross-track, sin/cos heading error, curvature, remaining,
            # and an explicit path-valid flag.
            size += 6
        if self.config.include_residual_context:
            size += self.config.residual_context_dimension
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

    def _frame_for_target(
        self,
        observation,
        target,
        previous_action=None,
        safety_override=False,
        scan_encoding=None,
        path_context=None,
        residual_context=None,
    ):
        pose = observation.pose
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
        if self.config.include_path_context:
            if path_context is None:
                # Explicitly encode a non-polyline target.  This fallback
                # keeps point-goal tasks valid in mixed-task training while
                # preventing the Actor from confusing a fabricated straight
                # path with a configured route.
                path_context = np.asarray(
                    (0.0, 0.0, 1.0, 0.0, np.clip(
                        distance / self.config.path_remaining_scale, 0.0, 1.0
                    ), 0.0),
                    dtype=np.float64,
                )
            path_context = np.asarray(path_context, dtype=np.float64).reshape(-1)
            if path_context.shape != (6,) or not np.isfinite(path_context).all():
                raise ValueError("path context must contain six finite features")
            features.extend(path_context.tolist())
        if scan_encoding is None:
            scan_features, scan_valid = self._scan_features(observation.scan)
        else:
            scan_features, scan_valid = scan_encoding
            scan_features = np.asarray(scan_features, dtype=np.float32)
            if scan_features.shape != (self.config.lidar_sectors,):
                raise ValueError("cached scan features have an invalid shape")
            if (
                not np.isfinite(scan_features).all()
                or not np.isfinite(float(scan_valid))
            ):
                raise ValueError("cached scan features must be finite")
        features.extend(scan_features.tolist())
        features.append(scan_valid)
        if self.config.include_residual_context:
            if residual_context is None:
                residual_context = np.zeros(
                    self.config.residual_context_dimension, dtype=np.float64
                )
            residual_context = np.asarray(
                residual_context, dtype=np.float64
            ).reshape(-1)
            if (
                residual_context.shape
                != (self.config.residual_context_dimension,)
                or not np.isfinite(residual_context).all()
            ):
                raise ValueError(
                    "residual context must be a finite configured-size vector"
                )
            # Append after every legacy feature so an expanded Actor can copy
            # the old input layer verbatim and initialize only new columns.
            features.extend(residual_context.tolist())
        frame = np.asarray(features, dtype=np.float32)
        if frame.shape != (self.frame_dimension,) or not np.isfinite(frame).all():
            raise FloatingPointError("RL observation encoder produced invalid features")
        return frame

    def encode_to_target(
        self,
        observation,
        target,
        previous_action=None,
        safety_override=False,
        update_history=True,
        scan_encoding=None,
        path_context=None,
        residual_context=None,
    ):
        """Encode against an explicit target, optionally without state mutation.

        Hypothetical MPPI terminal states must not advance the online history
        buffer.  When ``update_history`` is false, the candidate frame is
        appended to a temporary copy of the current history only.
        """

        frame = self._frame_for_target(
            observation,
            target,
            previous_action=previous_action,
            safety_override=safety_override,
            scan_encoding=scan_encoding,
            path_context=path_context,
            residual_context=residual_context,
        )
        history_frames = self.config.history_frames
        if update_history:
            if not self._history:
                self._history = [frame.copy() for _ in range(history_frames)]
            else:
                self._history.append(frame.copy())
                self._history = self._history[-history_frames:]
            history = self._history
        elif not self._history:
            history = [frame.copy() for _ in range(history_frames)]
        elif history_frames == 1:
            history = [frame]
        else:
            history = self._history[-(history_frames - 1):] + [frame]
        encoded = np.concatenate(history).astype(np.float32, copy=False)
        if encoded.shape != (self.dimension,) or not np.isfinite(encoded).all():
            raise FloatingPointError("RL observation history produced invalid features")
        return encoded

    def encode_kinematic_batch(
        self,
        poses,
        twists,
        target_positions,
        previous_actions,
        scan_encoding,
        safety_override=False,
        path_context_features=None,
        residual_context_features=None,
    ):
        """Vectorize history-free hypothetical policy observations.

        MPPI evaluates many candidate states against one latest LaserScan.
        Reconstructing a ``RobotObservation`` and sectorizing that same scan
        for every candidate is mathematically redundant.  This method emits
        the exact single-frame feature layout used by :meth:`encode_to_target`
        while making the shared-sensor assumption explicit and auditable.
        """

        if self.config.history_frames != 1:
            raise ValueError(
                "batched kinematic encoding requires history_frames=1"
            )
        poses = np.asarray(poses, dtype=np.float64)
        twists = np.asarray(twists, dtype=np.float64)
        targets = np.asarray(target_positions, dtype=np.float64)
        previous = np.asarray(previous_actions, dtype=np.float64)
        batch = poses.shape[0] if poses.ndim == 2 else 0
        if (
            batch <= 0
            or poses.shape != (batch, 3)
            or twists.shape != (batch, 2)
            or targets.shape != (batch, 2)
            or previous.shape != (batch, self.action_spec.dimension)
        ):
            raise ValueError("batched kinematic observation shapes are invalid")
        if not all(
            np.isfinite(values).all()
            for values in (poses, twists, targets, previous)
        ):
            raise ValueError("batched kinematic observations must be finite")

        scan_features, scan_valid = scan_encoding
        scan_features = np.asarray(scan_features, dtype=np.float64)
        if (
            scan_features.shape != (self.config.lidar_sectors,)
            or not np.isfinite(scan_features).all()
            or not np.isfinite(float(scan_valid))
        ):
            raise ValueError("cached scan features are invalid")

        x, y, theta = poses.T
        dx = targets[:, 0] - x
        dy = targets[:, 1] - y
        cosine = np.cos(theta)
        sine = np.sin(theta)
        body_dx = cosine * dx + sine * dy
        body_dy = -sine * dx + cosine * dy
        scale = float(self.config.goal_distance_scale)
        columns = [
            np.clip(body_dx / scale, -1.0, 1.0),
            np.clip(body_dy / scale, -1.0, 1.0),
            np.clip(np.hypot(dx, dy) / scale, 0.0, 1.0),
            np.arctan2(body_dy, body_dx) / np.pi,
            np.clip(
                twists[:, 0] / self.config.velocity_scale, -2.0, 2.0
            ),
            np.clip(
                twists[:, 1] / self.config.yaw_rate_scale, -2.0, 2.0
            ),
            sine,
            cosine,
        ]
        blocks = [np.column_stack(columns)]
        if self.config.include_absolute_pose:
            blocks.append(np.column_stack((x / scale, y / scale)))
        if self.config.include_previous_action:
            center = 0.5 * (
                self.action_spec.upper + self.action_spec.lower
            )
            half_range = 0.5 * (
                self.action_spec.upper - self.action_spec.lower
            )
            blocks.append(np.clip(
                (previous - center) / half_range, -1.0, 1.0
            ))
        if self.config.include_safety_state:
            flag = np.asarray(safety_override, dtype=np.float64)
            if flag.ndim == 0:
                flag = np.full(batch, float(bool(flag)))
            flag = flag.reshape(-1)
            if flag.shape != (batch,) or not np.isfinite(flag).all():
                raise ValueError("batched safety state is invalid")
            blocks.append(flag[:, None])
        if self.config.include_path_context:
            if path_context_features is None:
                remaining = np.clip(
                    np.hypot(dx, dy) / self.config.path_remaining_scale,
                    0.0,
                    1.0,
                )
                path_context_features = np.column_stack((
                    np.zeros(batch),
                    np.zeros(batch),
                    np.ones(batch),
                    np.zeros(batch),
                    remaining,
                    np.zeros(batch),
                ))
            path_context_features = np.asarray(
                path_context_features, dtype=np.float64
            )
            if (
                path_context_features.shape != (batch, 6)
                or not np.isfinite(path_context_features).all()
            ):
                raise ValueError(
                    "batched path context must be finite with shape [B,6]"
                )
            blocks.append(path_context_features)
        blocks.extend((
            np.broadcast_to(scan_features[None, :], (batch, scan_features.size)),
            np.full((batch, 1), float(scan_valid), dtype=np.float64),
        ))
        if self.config.include_residual_context:
            if residual_context_features is None:
                residual_context_features = np.zeros(
                    (batch, self.config.residual_context_dimension),
                    dtype=np.float64,
                )
            residual_context_features = np.asarray(
                residual_context_features, dtype=np.float64
            )
            if (
                residual_context_features.shape
                != (batch, self.config.residual_context_dimension)
                or not np.isfinite(residual_context_features).all()
            ):
                raise ValueError(
                    "batched residual context has an invalid shape or value"
                )
            blocks.append(residual_context_features)
        encoded = np.concatenate(blocks, axis=1).astype(
            np.float32, copy=False
        )
        if (
            encoded.shape != (batch, self.dimension)
            or not np.isfinite(encoded).all()
        ):
            raise FloatingPointError(
                "batched RL observation encoder produced invalid features"
            )
        return encoded

    def encode(
        self,
        observation,
        reference,
        previous_action=None,
        safety_override=False,
        residual_context=None,
    ):
        target = reference.target_at(
            observation.timestamp, observation.pose.as_array()
        )
        path_context = self.path_context(
            reference,
            observation.pose.as_array(),
            target=target,
        )
        return self.encode_to_target(
            observation,
            target,
            previous_action=previous_action,
            safety_override=safety_override,
            update_history=True,
            path_context=path_context,
            residual_context=residual_context,
        )

    def path_context(
        self,
        reference,
        pose,
        target=None,
        progress_floor=None,
    ):
        """Encode path geometry for one real or hypothetical pose.

        The returned values are dimensionless and bounded.  This method is
        side-effect free: it calls the polyline's public read-only projection
        API and never advances the live route.
        """

        if not self.config.include_path_context:
            return None
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("path context pose must be finite [x,y,theta]")
        project = getattr(reference, "project", None)
        if not callable(project):
            if target is None:
                target = reference.target_at(0.0, pose)
            distance = float(np.hypot(
                target.pose.x - pose[0], target.pose.y - pose[1]
            ))
            return np.asarray((
                0.0,
                0.0,
                1.0,
                0.0,
                np.clip(
                    distance / self.config.path_remaining_scale, 0.0, 1.0
                ),
                0.0,
            ), dtype=np.float32)
        floor = (
            getattr(reference, "progress", None)
            if progress_floor is None
            else progress_floor
        )
        projection = project(pose[:2], minimum_progress=floor)
        heading_error = float(np.arctan2(
            np.sin(pose[2] - projection.tangent_heading),
            np.cos(pose[2] - projection.tangent_heading),
        ))
        result = np.asarray((
            np.clip(
                projection.signed_cross_track_error
                / self.config.path_cross_track_scale,
                -1.0,
                1.0,
            ),
            np.sin(heading_error),
            np.cos(heading_error),
            np.clip(
                projection.curvature / self.config.path_curvature_scale,
                -1.0,
                1.0,
            ),
            np.clip(
                projection.remaining / self.config.path_remaining_scale,
                0.0,
                1.0,
            ),
            1.0,
        ), dtype=np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("path context encoder produced NaN or Inf")
        return result


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
