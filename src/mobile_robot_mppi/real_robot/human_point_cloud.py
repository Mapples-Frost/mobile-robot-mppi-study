"""Causal 3-D human-shape evidence for physical lidar tracks.

The Mid-360 scan used by the planner is deliberately two-dimensional: the
nearest return in each angular beam must remain available to geometric hard
safety.  That projection discards the vertical structure which distinguishes
a fragmented leg/body return from an arbitrary compact scan cluster.  This
module consumes the separately retained, deskewed base-frame point cloud and
attaches shape evidence to existing causal tracker slots.

It is not a detector which can create a trajectory or a chassis command.  A
slot still needs measured motion and a valid CA-IMM forecast before the
mapless filter may publish it.  The 3-D evidence only replaces the old
"number of 2-D beams looks human-sized" proxy.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class HumanPointCloudConfig:
    enabled: bool = False
    auxiliary_key: str = "human_point_cloud_base"
    maximum_range_m: float = 5.0
    association_radius_m: float = 0.62
    minimum_points: int = 18
    minimum_points_per_height_band: int = 4
    minimum_occupied_height_bands: int = 2
    minimum_vertical_span_m: float = 0.45
    maximum_horizontal_extent_m: float = 0.95
    minimum_height_m: float = 0.08
    lower_band_maximum_m: float = 0.55
    middle_band_maximum_m: float = 1.20
    maximum_height_m: float = 1.90

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] = None):
        source = dict(values or {})
        unknown = sorted(set(source) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown human point-cloud fields: %s" % ", ".join(unknown)
            )
        return cls(**source)

    def __post_init__(self) -> None:
        if not str(self.auxiliary_key):
            raise ValueError("human point-cloud auxiliary key must be non-empty")
        if self.maximum_range_m <= 0.0 or self.association_radius_m <= 0.0:
            raise ValueError("human point-cloud ranges must be positive")
        if self.minimum_points < 1:
            raise ValueError("human point-cloud minimum points must be positive")
        if self.minimum_points_per_height_band < 1:
            raise ValueError("human height-band support must be positive")
        if not 1 <= self.minimum_occupied_height_bands <= 3:
            raise ValueError("human occupied height bands must be in [1, 3]")
        if self.minimum_vertical_span_m <= 0.0:
            raise ValueError("human vertical span must be positive")
        if self.maximum_horizontal_extent_m <= 0.0:
            raise ValueError("human horizontal extent must be positive")
        if not (
            self.minimum_height_m
            < self.lower_band_maximum_m
            < self.middle_band_maximum_m
            < self.maximum_height_m
        ):
            raise ValueError("human point-cloud height bands must increase")


class HumanPointCloudEvidenceExtractor:
    """Attach full-height point-cloud support to low-level tracker slots."""

    def __init__(self, config: Mapping[str, Any] = None):
        self.config = HumanPointCloudConfig.from_mapping(config)

    @staticmethod
    def _world_to_base(position, pose) -> np.ndarray:
        dx = float(position[0]) - float(pose.x)
        dy = float(position[1]) - float(pose.y)
        cosine = float(np.cos(float(pose.theta)))
        sine = float(np.sin(float(pose.theta)))
        return np.asarray(
            (cosine * dx + sine * dy, -sine * dx + cosine * dy),
            dtype=np.float64,
        )

    @staticmethod
    def _base_to_world(position, pose) -> np.ndarray:
        cosine = float(np.cos(float(pose.theta)))
        sine = float(np.sin(float(pose.theta)))
        return np.asarray(
            (
                float(pose.x) + cosine * position[0] - sine * position[1],
                float(pose.y) + sine * position[0] + cosine * position[1],
            ),
            dtype=np.float64,
        )

    def _prepare_points(self, observation):
        raw = getattr(observation, "auxiliary", {}).get(
            self.config.auxiliary_key
        )
        if raw is None:
            return np.empty((0, 3), dtype=np.float64), "missing_point_cloud"
        points = np.asarray(raw)
        if points.ndim != 2 or points.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float64), "invalid_point_cloud_shape"
        if not np.issubdtype(points.dtype, np.number):
            return np.empty((0, 3), dtype=np.float64), "invalid_point_cloud_dtype"
        # Livox points arrive as float32.  Preserve that representation here;
        # promoting a 200k-point frame to float64 added a multi-megabyte copy
        # to every control cycle without improving the centimetre-scale gate.
        points = points.astype(np.float32, copy=False)
        finite = np.isfinite(points).all(axis=1)
        horizontal_range = np.hypot(points[:, 0], points[:, 1])
        keep = (
            finite
            & (horizontal_range <= float(self.config.maximum_range_m))
            & (points[:, 2] >= float(self.config.minimum_height_m))
            & (points[:, 2] <= float(self.config.maximum_height_m))
        )
        return points[keep], "available"

    def _describe(self, points: np.ndarray, center_base: np.ndarray) -> Dict[str, Any]:
        center = np.asarray(center_base, dtype=np.float32)
        dx = points[:, 0] - center[0]
        dy = points[:, 1] - center[1]
        radius_squared = np.float32(self.config.association_radius_m ** 2)
        selected = points[dx * dx + dy * dy <= radius_squared]
        point_count = int(selected.shape[0])
        if point_count:
            low = np.quantile(selected, 0.05, axis=0)
            high = np.quantile(selected, 0.95, axis=0)
            span = high - low
            centroid_base = np.median(selected[:, :2], axis=0)
            band_counts = (
                int(np.sum(selected[:, 2] < self.config.lower_band_maximum_m)),
                int(np.sum(
                    (selected[:, 2] >= self.config.lower_band_maximum_m)
                    & (selected[:, 2] < self.config.middle_band_maximum_m)
                )),
                int(np.sum(selected[:, 2] >= self.config.middle_band_maximum_m)),
            )
            vertical_span = float(span[2])
            horizontal_extent = float(max(span[0], span[1]))
        else:
            centroid_base = center_base.copy()
            band_counts = (0, 0, 0)
            vertical_span = 0.0
            horizontal_extent = 0.0
        occupied_bands = int(sum(
            count >= int(self.config.minimum_points_per_height_band)
            for count in band_counts
        ))
        candidate = bool(
            point_count >= int(self.config.minimum_points)
            and occupied_bands >= int(self.config.minimum_occupied_height_bands)
            and vertical_span >= float(self.config.minimum_vertical_span_m)
            and horizontal_extent
            <= float(self.config.maximum_horizontal_extent_m)
        )
        if candidate and occupied_bands == 3:
            shape = "full_body"
        elif candidate and band_counts[1] and band_counts[2]:
            shape = "upper_body"
        elif candidate and band_counts[0] and band_counts[1]:
            shape = "lower_body"
        elif candidate:
            shape = "vertical_partial"
        else:
            shape = "not_human_shaped"
        return {
            "candidate": candidate,
            "shape": shape,
            "point_count": point_count,
            "height_band_counts": tuple(int(value) for value in band_counts),
            "occupied_height_bands": occupied_bands,
            "vertical_span_m": vertical_span,
            "horizontal_extent_m": horizontal_extent,
            "centroid_base": centroid_base,
        }

    def annotate(self, tracks: Sequence[Mapping[str, Any]], observation):
        values = [dict(track or {}) for track in tracks or ()]
        points, reason = self._prepare_points(observation)
        diagnostics = {
            "enabled": bool(self.config.enabled),
            "reason": "disabled" if not self.config.enabled else reason,
            "input_point_count": int(points.shape[0]),
            "evaluated_track_count": 0,
            "candidate_track_indices": (),
        }
        if not self.config.enabled or reason != "available" or not points.size:
            return values, diagnostics

        candidate_indices = []
        for ordinal, track in enumerate(values):
            x_value = track.get("measurement_x")
            y_value = track.get("measurement_y")
            if (
                not bool(track.get("associated", False))
                or x_value is None
                or y_value is None
            ):
                continue
            center_base = self._world_to_base(
                (float(x_value), float(y_value)), observation.pose
            )
            evidence = self._describe(points, center_base)
            centroid_world = self._base_to_world(
                evidence.pop("centroid_base"), observation.pose
            )
            track_index = int(track.get("track_index", ordinal))
            track.update({
                "point_cloud_human_candidate": bool(evidence["candidate"]),
                "point_cloud_human_shape": str(evidence["shape"]),
                "point_cloud_human_point_count": int(evidence["point_count"]),
                "point_cloud_human_height_band_counts": tuple(
                    evidence["height_band_counts"]
                ),
                "point_cloud_human_occupied_height_bands": int(
                    evidence["occupied_height_bands"]
                ),
                "point_cloud_human_vertical_span_m": float(
                    evidence["vertical_span_m"]
                ),
                "point_cloud_human_horizontal_extent_m": float(
                    evidence["horizontal_extent_m"]
                ),
                "point_cloud_human_centroid_x": float(centroid_world[0]),
                "point_cloud_human_centroid_y": float(centroid_world[1]),
            })
            diagnostics["evaluated_track_count"] += 1
            if evidence["candidate"]:
                candidate_indices.append(track_index)
        diagnostics["candidate_track_indices"] = tuple(sorted(candidate_indices))
        diagnostics["candidate_track_count"] = len(candidate_indices)
        return values, diagnostics


__all__ = ["HumanPointCloudConfig", "HumanPointCloudEvidenceExtractor"]
