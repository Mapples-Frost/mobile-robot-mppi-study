"""Small, dependency-light data contracts shared by every backend."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float

    def as_array(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.theta), dtype=np.float64)


@dataclass(frozen=True)
class Twist2D:
    v: float
    omega: float

    def as_array(self) -> np.ndarray:
        return np.asarray((self.v, self.omega), dtype=np.float64)


@dataclass(frozen=True)
class ControlCommand:
    values: np.ndarray
    timestamp: float = 0.0
    source: str = "controller"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("control values must be a finite, non-empty vector")
        object.__setattr__(self, "values", values)

    @property
    def v(self) -> float:
        return float(self.values[0])

    @property
    def omega(self) -> float:
        return float(self.values[1])


@dataclass(frozen=True)
class LaserScan:
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    timestamp: float
    frame_id: str = "laser"
    obstacle_ranges: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        ranges = np.asarray(self.ranges, dtype=np.float64).reshape(-1)
        if ranges.size < 2:
            raise ValueError("LaserScan requires at least two beams")
        object.__setattr__(self, "ranges", ranges)
        if self.obstacle_ranges is not None:
            values = np.asarray(self.obstacle_ranges, dtype=np.float64).reshape(-1)
            if values.shape != ranges.shape:
                raise ValueError("obstacle_ranges must match ranges")
            object.__setattr__(self, "obstacle_ranges", values)


@dataclass(frozen=True)
class RobotObservation:
    timestamp: float
    pose: Pose2D
    twist: Twist2D
    scan: Optional[LaserScan] = None
    local_obstacles: Tuple[Tuple[float, float, float], ...] = ()
    auxiliary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth:
    timestamp: float
    pose: Pose2D
    twist: Twist2D
    wheel_speeds: Tuple[float, float] = (0.0, 0.0)
    actuator_effort: Tuple[float, float] = (0.0, 0.0)
    collision: bool = False
    minimum_clearance: float = float("inf")
    slip_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlantStep:
    ground_truth: GroundTruth
    executed_control: ControlCommand
    dt: float


@dataclass(frozen=True)
class PlanResult:
    proposed_control: ControlCommand
    control_sequence: np.ndarray
    predicted_trajectory: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyDecision:
    proposed_control: ControlCommand
    executed_control: ControlCommand
    overridden: bool
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
