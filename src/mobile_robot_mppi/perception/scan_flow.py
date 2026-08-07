"""Robust, simulator-truth-free temporal features from consecutive LaserScan frames.

The estimator intentionally reports *relative closing risk*, not an object
velocity.  It compares corresponding beams, rejects physically implausible
jumps, and requires a contiguous beam consensus.  This makes it suitable for
both the Python-3 safety boundary and auditable RL gating without exposing
MuJoCo obstacle state.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


def _linear_activation(value: float, soft: float, hard: float) -> float:
    if value <= soft:
        return 0.0
    if value >= hard:
        return 1.0
    return float((value - soft) / (hard - soft))


def _inverse_activation(value: float, hard: float, soft: float) -> float:
    if value <= hard:
        return 1.0
    if value >= soft:
        return 0.0
    return float((soft - value) / (soft - hard))


@dataclass(frozen=True)
class ScanFlowConfig:
    enabled: bool = False
    safety_enabled: bool = False
    ego_motion_compensation_enabled: bool = False
    safety_continuous_slowdown_enabled: bool = False
    field_of_view_deg: float = 270.0
    window_beams: int = 7
    minimum_support_beams: int = 4
    minimum_positive_rate_mps: float = 0.02
    maximum_absolute_rate_mps: float = 1.50
    maximum_dt_s: float = 0.40
    maximum_clearance_m: float = 1.50
    rate_soft_mps: float = 0.05
    rate_hard_mps: float = 0.25
    ttc_hard_s: float = 1.00
    ttc_soft_s: float = 3.00
    safety_hard_stop_ttc_s: float = 0.80
    safety_slow_ttc_s: float = 2.00
    safety_slow_scale: float = 0.25
    hold_s: float = 0.40
    # A temporal-flow hard stop is allowed only when the closing consensus is
    # sufficiently dense and not dominated by rejected beam jumps.  Defaults
    # preserve the historical safety contract; physical deployment opts in.
    safety_min_support_beams: int = 0
    safety_max_rejected_jump_fraction: float = 1.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        defaults = cls()
        kwargs = {
            name: values.get(name, getattr(defaults, name))
            for name in asdict(defaults)
        }
        for name in (
            "enabled",
            "safety_enabled",
            "ego_motion_compensation_enabled",
            "safety_continuous_slowdown_enabled",
        ):
            kwargs[name] = bool(kwargs[name])
        for name in (
            "window_beams",
            "minimum_support_beams",
            "safety_min_support_beams",
        ):
            kwargs[name] = int(kwargs[name])
        for name in kwargs:
            if name not in (
                "enabled",
                "safety_enabled",
                "ego_motion_compensation_enabled",
                "safety_continuous_slowdown_enabled",
                "window_beams",
                "minimum_support_beams",
                "safety_min_support_beams",
            ):
                kwargs[name] = float(kwargs[name])
        return cls(**kwargs)

    def validate(self) -> None:
        if not 0.0 < self.field_of_view_deg <= 360.0:
            raise ValueError("temporal scan field_of_view_deg must be in (0, 360]")
        if self.window_beams < 3 or self.window_beams % 2 == 0:
            raise ValueError("temporal scan window_beams must be an odd integer >= 3")
        if not 1 <= self.minimum_support_beams <= self.window_beams:
            raise ValueError("temporal scan support must be within the beam window")
        positive = (
            self.minimum_positive_rate_mps,
            self.maximum_absolute_rate_mps,
            self.maximum_dt_s,
            self.maximum_clearance_m,
            self.rate_soft_mps,
            self.rate_hard_mps,
            self.ttc_hard_s,
            self.ttc_soft_s,
            self.safety_hard_stop_ttc_s,
            self.safety_slow_ttc_s,
            self.safety_slow_scale,
        )
        if not np.isfinite(positive).all() or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("temporal scan thresholds must be finite and positive")
        if self.minimum_positive_rate_mps >= self.maximum_absolute_rate_mps:
            raise ValueError("temporal scan maximum rate must exceed minimum rate")
        if self.rate_hard_mps <= self.rate_soft_mps:
            raise ValueError("temporal scan hard rate must exceed soft rate")
        if self.ttc_soft_s <= self.ttc_hard_s:
            raise ValueError("temporal scan soft TTC must exceed hard TTC")
        if self.safety_slow_ttc_s <= self.safety_hard_stop_ttc_s:
            raise ValueError("temporal safety slow TTC must exceed hard-stop TTC")
        if self.safety_slow_scale > 1.0:
            raise ValueError("temporal safety slow scale must not exceed one")
        if not np.isfinite(self.hold_s) or self.hold_s < 0.0:
            raise ValueError("temporal scan hold_s must be finite and non-negative")
        if self.safety_min_support_beams < 0:
            raise ValueError("temporal safety support floor must be non-negative")
        if not 0.0 <= self.safety_max_rejected_jump_fraction <= 1.0:
            raise ValueError(
                "temporal safety rejected-jump ceiling must be in [0, 1]"
            )

    def safety_slowdown_scale(self, ttc_s: float) -> float:
        """Return a continuous soft-braking scale without weakening hard stop."""

        if not self.safety_continuous_slowdown_enabled:
            return float(self.safety_slow_scale)
        span = self.safety_slow_ttc_s - self.safety_hard_stop_ttc_s
        fraction = (
            float(ttc_s) - self.safety_hard_stop_ttc_s
        ) / span
        return float(np.clip(
            max(self.safety_slow_scale, fraction),
            self.safety_slow_scale,
            1.0,
        ))


@dataclass(frozen=True)
class ScanFlowEstimate:
    valid: bool = False
    reason: str = "disabled"
    closing_rate_mps: float = 0.0
    clearance_m: float = float("inf")
    ttc_s: float = float("inf")
    risk_alpha: float = 0.0
    support_beams: int = 0
    support_fraction: float = 0.0
    rejected_jump_fraction: float = 0.0
    center_angle_rad: float = 0.0
    held: bool = False

    def to_dict(self):
        return asdict(self)


class TemporalSafetyHysteresis:
    """Immediate risk entry with confirmed, monotone safety release."""

    _SEVERITY = {"clear": 0, "slow": 1, "stop": 2}

    def __init__(self, enabled=False, release_clear_frames=2):
        self.enabled = bool(enabled)
        self.release_clear_frames = int(release_clear_frames)
        if self.release_clear_frames < 1:
            raise ValueError(
                "temporal safety release_clear_frames must be positive"
            )
        self.reset()

    def reset(self):
        self.state = "clear"
        self.clear_streak = 0

    def update(self, raw_state):
        raw_state = str(raw_state)
        if raw_state not in self._SEVERITY:
            raise ValueError("unknown temporal safety state: %s" % raw_state)
        previous = self.state
        entered_immediately = False
        release_confirmed = False
        if not self.enabled:
            self.state = raw_state
            self.clear_streak = 0
        else:
            raw_severity = self._SEVERITY[raw_state]
            active_severity = self._SEVERITY[self.state]
            if raw_severity > active_severity:
                self.state = raw_state
                self.clear_streak = 0
                entered_immediately = True
            elif raw_severity == active_severity:
                self.clear_streak = 0
            else:
                self.clear_streak += 1
                if self.clear_streak >= self.release_clear_frames:
                    self.state = raw_state
                    self.clear_streak = 0
                    release_confirmed = True
        return self.state, {
            "temporal_scan_safety_raw_state": raw_state,
            "temporal_scan_safety_state": self.state,
            "temporal_scan_safety_previous_state": previous,
            "temporal_scan_safety_entered_immediately": bool(
                entered_immediately
            ),
            "temporal_scan_safety_release_confirmed": bool(
                release_confirmed
            ),
            "temporal_scan_safety_release_pending": bool(
                self.enabled
                and self._SEVERITY[raw_state]
                < self._SEVERITY[self.state]
            ),
            "temporal_scan_safety_clear_streak": int(self.clear_streak),
            "temporal_scan_safety_release_clear_frames": int(
                self.release_clear_frames
            ),
        }


class RobustScanFlowEstimator:
    """Estimate contiguous relative range contraction between scan frames."""

    def __init__(self, config=None):
        self.config = (
            config
            if isinstance(config, ScanFlowConfig)
            else ScanFlowConfig.from_mapping(config)
        )
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self._previous_ranges = None
        self._previous_timestamp = None
        self._previous_geometry = None
        self._previous_pose = None
        self._held_estimate = None
        self._hold_until = None

    @staticmethod
    def _timestamp(scan, timestamp):
        value = scan.timestamp if timestamp is None else timestamp
        return float(value)

    @staticmethod
    def _pose_values(pose):
        if pose is None:
            return None
        if hasattr(pose, "as_array"):
            values = np.asarray(pose.as_array(), dtype=np.float64).reshape(-1)
        elif all(hasattr(pose, name) for name in ("x", "y", "yaw")):
            values = np.asarray(
                (pose.x, pose.y, pose.yaw), dtype=np.float64
            )
        else:
            values = np.asarray(pose, dtype=np.float64).reshape(-1)
        if values.size < 3 or not np.isfinite(values[:3]).all():
            raise ValueError("temporal scan pose must contain finite x/y/yaw")
        return values[:3].copy()

    def _remember(self, ranges, scan, timestamp, pose):
        self._previous_ranges = ranges.copy()
        self._previous_timestamp = float(timestamp)
        self._previous_geometry = (
            int(ranges.size),
            float(scan.angle_min),
            float(scan.angle_increment),
        )
        self._previous_pose = self._pose_values(pose)

    @staticmethod
    def _contiguous_runs(indices):
        if indices.size == 0:
            return ()
        boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
        return tuple(np.split(indices, boundaries))

    def _warp_previous_ranges(self, previous, scan, previous_pose, pose):
        """Reproject previous endpoints into the current robot frame.

        Interpolation is restricted to contiguous valid beam runs, so the
        warp cannot bridge an unobserved angular gap or invent an obstacle.
        """

        current_pose = self._pose_values(pose)
        if previous_pose is None or current_pose is None:
            return previous
        beam_angles = float(scan.angle_min) + np.arange(previous.size) * float(
            scan.angle_increment
        )
        warped = np.full(previous.size, np.nan, dtype=np.float64)
        valid_indices = np.flatnonzero(np.isfinite(previous))
        previous_x, previous_y, previous_yaw = previous_pose
        current_x, current_y, current_yaw = current_pose
        cp = np.cos(previous_yaw)
        sp = np.sin(previous_yaw)
        cc = np.cos(current_yaw)
        sc = np.sin(current_yaw)

        for run in self._contiguous_runs(valid_indices):
            local_x = previous[run] * np.cos(beam_angles[run])
            local_y = previous[run] * np.sin(beam_angles[run])
            world_x = previous_x + cp * local_x - sp * local_y
            world_y = previous_y + sp * local_x + cp * local_y
            dx = world_x - current_x
            dy = world_y - current_y
            transformed_x = cc * dx + sc * dy
            transformed_y = -sc * dx + cc * dy
            transformed_ranges = np.hypot(transformed_x, transformed_y)
            transformed_angles = np.unwrap(
                np.arctan2(transformed_y, transformed_x)
            )
            order = np.argsort(transformed_angles)
            transformed_angles = transformed_angles[order]
            transformed_ranges = transformed_ranges[order]
            if transformed_angles.size == 1:
                candidates = beam_angles + 2.0 * np.pi * np.round(
                    (transformed_angles[0] - beam_angles) / (2.0 * np.pi)
                )
                index = int(np.argmin(np.abs(candidates - transformed_angles[0])))
                warped[index] = transformed_ranges[0]
                continue
            for shift in (-2.0 * np.pi, 0.0, 2.0 * np.pi):
                targets = beam_angles + shift
                inside = (
                    targets >= transformed_angles[0]
                ) & (targets <= transformed_angles[-1])
                if not np.any(inside):
                    continue
                interpolated = np.interp(
                    targets[inside], transformed_angles, transformed_ranges
                )
                indices = np.flatnonzero(inside)
                empty = ~np.isfinite(warped[indices])
                warped[indices[empty]] = interpolated[empty]
                warped[indices[~empty]] = np.minimum(
                    warped[indices[~empty]], interpolated[~empty]
                )
        return warped

    def _hold_or(self, estimate, timestamp):
        if estimate.valid and estimate.risk_alpha > 0.0:
            self._held_estimate = estimate
            self._hold_until = float(timestamp) + self.config.hold_s
            return estimate
        if (
            self._held_estimate is not None
            and self._hold_until is not None
            and float(timestamp) <= self._hold_until + 1e-12
        ):
            values = self._held_estimate.to_dict()
            values.update({"reason": "held", "held": True})
            return ScanFlowEstimate(**values)
        self._held_estimate = None
        self._hold_until = None
        return estimate

    def update(self, scan, timestamp=None, pose=None) -> ScanFlowEstimate:
        if not self.config.enabled:
            return ScanFlowEstimate(reason="disabled")
        if scan is None:
            return ScanFlowEstimate(reason="no_scan")
        now = self._timestamp(scan, timestamp)
        raw = np.asarray(scan.ranges, dtype=np.float64).reshape(-1)
        current_valid = (
            np.isfinite(raw)
            & (raw >= float(scan.range_min))
            & (raw <= float(scan.range_max))
        )
        current = np.where(current_valid, raw, np.nan)
        geometry = (raw.size, float(scan.angle_min), float(scan.angle_increment))
        if self._previous_ranges is None:
            self._remember(current, scan, now, pose)
            return ScanFlowEstimate(reason="warmup")
        if geometry != self._previous_geometry:
            self._remember(current, scan, now, pose)
            return ScanFlowEstimate(reason="geometry_changed")
        dt = now - float(self._previous_timestamp)
        previous = self._previous_ranges
        previous_pose = self._previous_pose
        self._remember(current, scan, now, pose)
        if not np.isfinite(dt) or dt <= 1e-9 or dt > self.config.maximum_dt_s:
            return self._hold_or(ScanFlowEstimate(reason="invalid_dt"), now)
        if self.config.ego_motion_compensation_enabled:
            previous = self._warp_previous_ranges(
                previous, scan, previous_pose, pose
            )

        angles = float(scan.angle_min) + np.arange(raw.size) * float(
            scan.angle_increment
        )
        wrapped = np.arctan2(np.sin(angles), np.cos(angles))
        in_fov = np.abs(wrapped) <= np.deg2rad(self.config.field_of_view_deg) * 0.5
        comparable = np.isfinite(previous) & np.isfinite(current) & in_fov
        rates = np.full(raw.size, np.nan, dtype=np.float64)
        rates[comparable] = (previous[comparable] - current[comparable]) / dt
        implausible = comparable & (
            np.abs(rates) > self.config.maximum_absolute_rate_mps
        )
        candidate = (
            comparable
            & ~implausible
            & (rates >= self.config.minimum_positive_rate_mps)
            & (current <= self.config.maximum_clearance_m)
        )
        denominator = max(1, int(np.sum(comparable)))
        rejected_fraction = float(np.sum(implausible) / denominator)

        radius = self.config.window_beams // 2
        best = None
        for center in np.flatnonzero(candidate):
            start = max(0, int(center) - radius)
            stop = min(raw.size, int(center) + radius + 1)
            members = np.flatnonzero(candidate[start:stop]) + start
            if members.size < self.config.minimum_support_beams:
                continue
            robust_rate = float(np.median(rates[members]))
            clearance = float(np.median(current[members]))
            score = (robust_rate, -clearance, int(members.size))
            if best is None or score > best[0]:
                best = (score, center, members, robust_rate, clearance)
        if best is None:
            return self._hold_or(
                ScanFlowEstimate(
                    reason="insufficient_consensus",
                    rejected_jump_fraction=rejected_fraction,
                ),
                now,
            )

        _, center, members, rate, clearance = best
        ttc = clearance / rate if rate > 1e-12 else float("inf")
        rate_alpha = _linear_activation(
            rate, self.config.rate_soft_mps, self.config.rate_hard_mps
        )
        ttc_alpha = _inverse_activation(
            ttc, self.config.ttc_hard_s, self.config.ttc_soft_s
        )
        estimate = ScanFlowEstimate(
            valid=True,
            reason="closing_consensus",
            closing_rate_mps=rate,
            clearance_m=clearance,
            ttc_s=ttc,
            risk_alpha=max(rate_alpha, ttc_alpha),
            support_beams=int(members.size),
            support_fraction=float(members.size / self.config.window_beams),
            rejected_jump_fraction=rejected_fraction,
            center_angle_rad=float(wrapped[int(center)]),
        )
        return self._hold_or(estimate, now)
