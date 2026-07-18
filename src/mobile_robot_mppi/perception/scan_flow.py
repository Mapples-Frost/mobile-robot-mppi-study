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

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        defaults = cls()
        kwargs = {
            name: values.get(name, getattr(defaults, name))
            for name in asdict(defaults)
        }
        for name in ("enabled", "safety_enabled"):
            kwargs[name] = bool(kwargs[name])
        for name in ("window_beams", "minimum_support_beams"):
            kwargs[name] = int(kwargs[name])
        for name in kwargs:
            if name not in (
                "enabled",
                "safety_enabled",
                "window_beams",
                "minimum_support_beams",
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
        self._held_estimate = None
        self._hold_until = None

    @staticmethod
    def _timestamp(scan, timestamp):
        value = scan.timestamp if timestamp is None else timestamp
        return float(value)

    def _remember(self, ranges, scan, timestamp):
        self._previous_ranges = ranges.copy()
        self._previous_timestamp = float(timestamp)
        self._previous_geometry = (
            int(ranges.size),
            float(scan.angle_min),
            float(scan.angle_increment),
        )

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

    def update(self, scan, timestamp=None) -> ScanFlowEstimate:
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
            self._remember(current, scan, now)
            return ScanFlowEstimate(reason="warmup")
        if geometry != self._previous_geometry:
            self._remember(current, scan, now)
            return ScanFlowEstimate(reason="geometry_changed")
        dt = now - float(self._previous_timestamp)
        previous = self._previous_ranges
        self._remember(current, scan, now)
        if not np.isfinite(dt) or dt <= 1e-9 or dt > self.config.maximum_dt_s:
            return self._hold_or(ScanFlowEstimate(reason="invalid_dt"), now)

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
