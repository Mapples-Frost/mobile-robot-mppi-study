"""Auditable LaserScan-only local geometry complexity for RL prior gating."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class SceneComplexityConfig:
    """Physical thresholds used by the local geometry score.

    The score deliberately excludes simulator geometry, actor features and
    future rollout outcomes.  It can therefore be reproduced from the same
    LaserScan available to the real-robot perception chain.
    """

    front_half_angle_deg: float = 40.0
    side_outer_angle_deg: float = 125.0
    near_distance_m: float = 0.45
    far_distance_m: float = 1.30
    density_distance_m: float = 1.25
    density_full_fraction: float = 0.30
    soft_threshold: float = 0.35
    hard_threshold: float = 0.70

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            front_half_angle_deg=float(
                values.get("front_half_angle_deg", 40.0)
            ),
            side_outer_angle_deg=float(
                values.get("side_outer_angle_deg", 125.0)
            ),
            near_distance_m=float(values.get("near_distance_m", 0.45)),
            far_distance_m=float(values.get("far_distance_m", 1.30)),
            density_distance_m=float(
                values.get("density_distance_m", 1.25)
            ),
            density_full_fraction=float(
                values.get("density_full_fraction", 0.30)
            ),
            soft_threshold=float(values.get("soft_threshold", 0.35)),
            hard_threshold=float(values.get("hard_threshold", 0.70)),
        )

    def validate(self):
        numeric = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("scene complexity thresholds must be finite")
        if not 0.0 < self.front_half_angle_deg < self.side_outer_angle_deg <= 180.0:
            raise ValueError(
                "scene complexity angles must satisfy 0 < front < side <= 180"
            )
        if not 0.0 < self.near_distance_m < self.far_distance_m:
            raise ValueError(
                "scene complexity distances must satisfy 0 < near < far"
            )
        if self.density_distance_m <= 0.0:
            raise ValueError("scene complexity density distance must be positive")
        if not 0.0 < self.density_full_fraction <= 1.0:
            raise ValueError(
                "scene complexity density full fraction must be in (0, 1]"
            )
        if not 0.0 <= self.soft_threshold < self.hard_threshold <= 1.0:
            raise ValueError(
                "scene complexity gate thresholds must satisfy 0 <= soft < hard <= 1"
            )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SceneComplexity:
    score: float
    front_proximity: float
    constriction: float
    density: float
    front_clearance_m: float
    left_clearance_m: float
    right_clearance_m: float
    near_obstacle_fraction: float
    scan_valid: bool

    def to_dict(self):
        return asdict(self)


def _proximity(distance, near, far):
    return float(np.clip((far - distance) / (far - near), 0.0, 1.0))


def _sector_minimum(distances, mask, empty_value):
    selected = distances[mask]
    selected = selected[np.isfinite(selected)]
    return float(np.min(selected)) if selected.size else float(empty_value)


def score_scene_complexity(scan, config=None):
    """Return a finite, monotone local geometry score in ``[0, 1]``.

    Missing scan data fails closed to zero RL activation.  A present scan with
    no obstacle returns valid clearances at ``range_max`` and score zero.
    ``obstacle_ranges`` is preferred when supplied by the sensor backend; it
    remains a LaserScan-derived field and is not simulator obstacle truth.
    """

    cfg = (
        config
        if isinstance(config, SceneComplexityConfig)
        else SceneComplexityConfig.from_mapping(config)
    )
    cfg.validate()
    if scan is None:
        return SceneComplexity(
            score=0.0,
            front_proximity=0.0,
            constriction=0.0,
            density=0.0,
            front_clearance_m=cfg.far_distance_m,
            left_clearance_m=cfg.far_distance_m,
            right_clearance_m=cfg.far_distance_m,
            near_obstacle_fraction=0.0,
            scan_valid=False,
        )

    raw = (
        scan.obstacle_ranges
        if scan.obstacle_ranges is not None
        else scan.ranges
    )
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("scene complexity requires at least two scan beams")
    valid = (
        np.isfinite(values)
        & (values >= float(scan.range_min))
        & (values <= float(scan.range_max))
    )
    distances = np.where(valid, values, np.inf)
    angles = float(scan.angle_min) + np.arange(values.size) * float(
        scan.angle_increment
    )
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    front_angle = np.deg2rad(cfg.front_half_angle_deg)
    outer_angle = np.deg2rad(cfg.side_outer_angle_deg)
    front_mask = np.abs(angles) <= front_angle
    left_mask = (angles > front_angle) & (angles <= outer_angle)
    right_mask = (angles < -front_angle) & (angles >= -outer_angle)
    empty = max(float(scan.range_max), cfg.far_distance_m)
    front_clearance = _sector_minimum(distances, front_mask, empty)
    left_clearance = _sector_minimum(distances, left_mask, empty)
    right_clearance = _sector_minimum(distances, right_mask, empty)

    front = _proximity(
        front_clearance, cfg.near_distance_m, cfg.far_distance_m
    )
    left = _proximity(
        left_clearance, cfg.near_distance_m, cfg.far_distance_m
    )
    right = _proximity(
        right_clearance, cfg.near_distance_m, cfg.far_distance_m
    )
    constriction = min(left, right)
    near_fraction = float(
        np.mean(valid & (values <= cfg.density_distance_m))
    )
    density = float(
        np.clip(near_fraction / cfg.density_full_fraction, 0.0, 1.0)
    )
    score = max(front, constriction, density)
    result = SceneComplexity(
        score=float(score),
        front_proximity=float(front),
        constriction=float(constriction),
        density=float(density),
        front_clearance_m=float(front_clearance),
        left_clearance_m=float(left_clearance),
        right_clearance_m=float(right_clearance),
        near_obstacle_fraction=float(near_fraction),
        scan_valid=True,
    )
    output = np.asarray(tuple(result.to_dict().values())[:-1], dtype=np.float64)
    if not np.isfinite(output).all() or not 0.0 <= result.score <= 1.0:
        raise FloatingPointError("scene complexity produced an invalid score")
    return result

