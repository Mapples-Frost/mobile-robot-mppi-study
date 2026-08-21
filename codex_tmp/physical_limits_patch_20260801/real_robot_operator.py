"""Run the complete B11 RL-MPPI stack against live robot telemetry.

The default mode is shadow and cannot publish.  Live mode requires an explicit
confirmation token and publishes only to /mppi_safe_cmd; the robot-side v2
gateway remains the sole owner of /cmd_vel.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY_ADAPTER = (
    ROOT / "codex_tmp/real_robot_single_obstacle_20260730"
)
VENDOR = ROOT / "codex_tmp/real_robot_deployment_20260730/python_vendor"
for path in (ROOT, LEGACY_ADAPTER, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import roslibpy  # noqa: E402

from build_real_config import build_real_config  # noqa: E402
from mobile_robot_mppi.runtime.factories import make_components  # noqa: E402
from mobile_robot_mppi.core.references import PoseGoal  # noqa: E402
from real_robot_shadow import (  # noqa: E402
    LiveTelemetry,
    _json_value,
    _observation,
    _runtime_manifest,
    _sha256,
    _yaw,
)
from human_leg_tracker import HumanLegMotionBootstrapTracker  # noqa: E402
from complex_scene_tracker import (  # noqa: E402
    MotionBootstrapMultiObstacleTracker,
)
from background_static_filter import (  # noqa: E402
    BackgroundFilteredMotionBootstrapTracker,
    CalibratedStaticBackgroundFilter,
)
from reverse_safety_contract import (  # noqa: E402
    apply_front_clearance_arc_governor,
    apply_near_body_speed_governor,
    apply_reactive_arc_hold,
    apply_turn_direction_lock,
    apply_provisional_collision_intervention,
    bound_signed_speed,
    couple_differential_drive_command,
    limit_angular_command_step,
    protected_reverse_escape,
    stabilize_goal_heading,
    verified_near_body_turn_escape,
)
from ego_motion_compensation import synchronized_snapshot  # noqa: E402
from mapless_static_dynamic_filter import (  # noqa: E402
    MaplessStaticDynamicFilter,
)


LIVE_CONFIRM_TOKEN = "I_HAVE_PHYSICAL_ESTOP"
ALGORITHM = "B11FullProposed"
OBSTACLE_PROFILES = ("Generic", "HumanLeg", "ComplexScene")
HARD_MAX_LINEAR_MPS = 0.70
HARD_MAX_REVERSE_MPS = 0.70
HARD_MAX_ANGULAR_RADPS = 3.00
BOUNDARY_MARGIN_M = 0.25
DEPLOYMENT_CONTRACT = "real_robot_reverse_full_stack_v1"
REALTIME_MPPI_SAMPLES = 50
REALTIME_COMMAND_DEADLINE_S = 0.50
PHYSICAL_WHEEL_TRACK_M = 0.330
PHYSICAL_WHEEL_SPEED_LIMIT_MPS = 0.60
OUTBOUND_GOAL_TOLERANCE_M = 0.30
HOME_POSITION_TOLERANCE_M = 0.20
HOME_HEADING_TOLERANCE_RAD = 0.10
HOME_ALIGNMENT_MAX_ANGULAR_RADPS = 0.20
RETURN_BOUNDARY_GUARD_DISTANCE_M = 0.35
RETURN_BOUNDARY_LINEAR_CAP_MPS = 0.12


def _fresh_rear_gateway(status, maximum_age_s=0.40):
    if not isinstance(status, dict):
        return False
    ages = (
        status.get("rear_right_age_s"),
        status.get("rear_left_age_s"),
    )
    return all(
        age is not None
        and 0.0 <= float(age) <= float(maximum_age_s)
        for age in ages
    )


def _scan_turn_escape_evidence(scan):
    if scan is None:
        return None
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    angles = float(scan.angle_min) + np.arange(ranges.size) * float(
        scan.angle_increment
    )
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan.range_min))
        & (ranges <= float(scan.range_max))
    )
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    nearest_index = int(valid_indices[np.argmin(ranges[valid])])
    left = valid & (angles >= math.radians(25.0)) & (
        angles <= math.radians(120.0)
    )
    right = valid & (angles <= math.radians(-25.0)) & (
        angles >= math.radians(-120.0)
    )
    return {
        "nearest_obstacle_angle_rad": float(angles[nearest_index]),
        "left_clearance_m": (
            float(np.min(ranges[left])) if np.any(left) else None
        ),
        "right_clearance_m": (
            float(np.min(ranges[right])) if np.any(right) else None
        ),
    }


def _scan_front_obstacle_angle(scan, half_angle_rad=math.radians(30.0)):
    if scan is None:
        return None
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    angles = float(scan.angle_min) + np.arange(ranges.size) * float(
        scan.angle_increment
    )
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan.range_min))
        & (ranges <= float(scan.range_max))
        & (np.abs(angles) <= float(half_angle_rad))
    )
    if not np.any(valid):
        return None
    indices = np.flatnonzero(valid)
    index = int(indices[np.argmin(ranges[valid])])
    return float(angles[index])


def _turn_limited_robot_positions(
    observation,
    goal_x,
    goal_y,
    times,
    maximum_linear_mps,
    maximum_angular_radps,
):
    """Causal unicycle projection that respects the physical yaw-rate cap."""
    values = np.asarray(times, dtype=np.float64)
    position = np.asarray(
        (observation.pose.x, observation.pose.y), dtype=np.float64
    )
    theta = float(observation.pose.theta)
    previous_time = 0.0
    result = []
    for timestamp in values:
        dt = max(float(timestamp) - previous_time, 0.0)
        previous_time = float(timestamp)
        to_goal = np.asarray((goal_x, goal_y), dtype=np.float64) - position
        goal_distance = float(np.linalg.norm(to_goal))
        if goal_distance <= 0.05 or dt <= 0.0:
            result.append(position.copy())
            continue
        desired_heading = math.atan2(float(to_goal[1]), float(to_goal[0]))
        heading_error = math.atan2(
            math.sin(desired_heading - theta),
            math.cos(desired_heading - theta),
        )
        omega = min(
            max(
                1.5 * heading_error,
                -abs(float(maximum_angular_radps)),
            ),
            abs(float(maximum_angular_radps)),
        )
        speed = min(abs(float(maximum_linear_mps)), goal_distance / dt)
        if abs(omega) <= 1.0e-9:
            position = position + speed * dt * np.asarray(
                (math.cos(theta), math.sin(theta)), dtype=np.float64
            )
        else:
            next_theta = theta + omega * dt
            position = position + (speed / omega) * np.asarray(
                (
                    math.sin(next_theta) - math.sin(theta),
                    -math.cos(next_theta) + math.cos(theta),
                ),
                dtype=np.float64,
            )
            theta = next_theta
        theta = math.atan2(math.sin(theta), math.cos(theta))
        result.append(position.copy())
    return np.asarray(result, dtype=np.float64)


def _path_relevant_forecasts(
    observation,
    goal_x,
    goal_y,
    maximum_linear_mps=0.50,
    maximum_angular_radps=0.50,
):
    """Select forecasts using synchronized robot/obstacle closest approach.

    Spatial intersection with the entire goal path is insufficient: an
    obstacle can cross a point long before the robot arrives there.  This gate
    advances a conservative turn-rate-limited unicycle at the same forecast
    timestamps and measures the two actors at equal time indices.
    """
    key = "probabilistic_obstacle_forecasts"
    forecasts = tuple(observation.auxiliary.get(key, ()))
    diagnostics = {
        "input_count": len(forecasts),
        "retained_count": 0,
        "dropped_count": 0,
        "minimum_path_distance_m": None,
        "minimum_synchronized_clearance_m": None,
        "collision_candidates": (),
    }
    if not forecasts:
        return observation, diagnostics
    start = np.asarray(
        (observation.pose.x, observation.pose.y), dtype=np.float64
    )
    retained = []
    collision_candidates = []
    tracker_diagnostics = observation.auxiliary.get(
        "dynamic_obstacle_tracker", {}
    )
    mapless_dynamic_confirmed = bool(
        isinstance(tracker_diagnostics, dict)
        and tracker_diagnostics.get(
            "mapless_static_dynamic_filter_enabled", False
        )
        and tracker_diagnostics.get("valid_forecast_count", 0)
    )
    tracker_tracks = tuple(
        tracker_diagnostics.get("tracks", ())
        if isinstance(tracker_diagnostics, dict)
        else ()
    )
    forecast_track_indices = tuple(
        tracker_diagnostics.get("forecast_track_indices", ())
        if isinstance(tracker_diagnostics, dict)
        else ()
    )
    minimum_distance = float("inf")
    minimum_clearance = float("inf")
    for forecast_offset, forecast in enumerate(forecasts):
        track_index = (
            int(forecast_track_indices[forecast_offset])
            if forecast_offset < len(forecast_track_indices)
            else None
        )
        vehicle_dynamic = bool(
            track_index is not None
            and 0 <= track_index < len(tracker_tracks)
            and tracker_tracks[track_index].get("mapless_dynamic_mode")
            == "wide_vehicle"
        )
        means = np.asarray(forecast.component_means, dtype=np.float64)
        weights = np.asarray(
            forecast.component_weights, dtype=np.float64
        )
        mixture_mean = np.sum(means * weights[..., None], axis=1)
        times = (
            np.arange(1, mixture_mean.shape[0] + 1, dtype=np.float64)
            * float(forecast.dt)
        )
        robot_positions = _turn_limited_robot_positions(
            observation,
            goal_x,
            goal_y,
            times,
            maximum_linear_mps,
            maximum_angular_radps,
        )
        synchronized_relative = mixture_mean - robot_positions
        synchronized_distances = np.linalg.norm(
            synchronized_relative, axis=1
        )
        robot_distances = np.linalg.norm(
            mixture_mean - start[None, :], axis=1
        )
        minimum_distance = min(
            minimum_distance, float(np.min(synchronized_distances))
        )
        covariances = np.asarray(
            forecast.component_covariances, dtype=np.float64
        )
        eigenvalues = np.linalg.eigvalsh(covariances)
        step_std = np.sqrt(
            np.maximum(np.max(eigenvalues, axis=(1, 2)), 0.0)
        )
        uncertainty_margin = np.minimum(3.0 * step_std, 0.45)
        collision_extent = (
            0.25
            + float(forecast.radius_m)
            + 0.10
            + (0.15 if vehicle_dynamic else 0.0)
            + uncertainty_margin
        )
        synchronized_clearance = (
            synchronized_distances - collision_extent
        )
        minimum_clearance = min(
            minimum_clearance, float(np.min(synchronized_clearance))
        )
        conflict_indices = np.flatnonzero(
            synchronized_clearance <= 0.35
        )
        immediate_conflict = bool(
            robot_distances[0] <= 0.85 + uncertainty_margin[0]
        )
        if conflict_indices.size or immediate_conflict:
            retained.append(forecast)
            closest_index = int(
                np.argmin(synchronized_clearance)
                if conflict_indices.size
                else np.argmin(robot_distances)
            )
            left_axis = np.asarray(
                (
                    -math.sin(float(observation.pose.theta)),
                    math.cos(float(observation.pose.theta)),
                ),
                dtype=np.float64,
            )
            collision_candidates.append(
                {
                    "closest_approach_time_s": float(
                        (closest_index + 1) * float(forecast.dt)
                    ),
                    "closest_approach_distance_m": float(
                        synchronized_distances[closest_index]
                    ),
                    "distance_m": float(robot_distances[0]),
                    "collision_clearance_m": float(
                        synchronized_clearance[closest_index]
                    ),
                    "lateral_m": float(
                        np.dot(
                            synchronized_relative[closest_index],
                            left_axis,
                        )
                    ),
                    "source": str(getattr(forecast, "source", "forecast")),
                    "mapless_dynamic_confirmed": (
                        mapless_dynamic_confirmed
                    ),
                    "dynamic_shape": (
                        "wide_vehicle" if vehicle_dynamic else "compact"
                    ),
                }
            )
    diagnostics.update(
        {
            "retained_count": len(retained),
            "dropped_count": len(forecasts) - len(retained),
            "minimum_path_distance_m": (
                None if not np.isfinite(minimum_distance) else minimum_distance
            ),
            "minimum_synchronized_clearance_m": (
                None
                if not np.isfinite(minimum_clearance)
                else minimum_clearance
            ),
            "collision_candidates": tuple(collision_candidates),
        }
    )
    if len(retained) == len(forecasts):
        return observation, diagnostics
    auxiliary = dict(observation.auxiliary)
    auxiliary[key] = tuple(retained)
    auxiliary["forecast_path_relevance"] = diagnostics
    return replace(observation, auxiliary=auxiliary), diagnostics


class OperatorTelemetry(LiveTelemetry):
    def __init__(self, sensor_forward_offset_m):
        super().__init__()
        self.sensor_forward_offset_m = float(sensor_forward_offset_m)
        self.odom_history = deque(maxlen=256)
        self.gateway_status = None
        self.gateway_received = 0.0

    def on_odom(self, message):
        with self.lock:
            self.odom = message
            self.odom_received = time.monotonic()
            self.odom_history.append(message)

    def synchronized_snapshot(self):
        with self.lock:
            snapshot = (
                self.odom,
                self.scan,
                self.emergency,
                self.odom_received,
                self.scan_received,
                self.emergency_received,
            )
            history = tuple(self.odom_history)
        return synchronized_snapshot(
            snapshot,
            history,
            self.sensor_forward_offset_m,
        )

    def on_gateway(self, message):
        raw = message.get("data", "")
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = {"parse_error": True, "raw": str(raw)}
        with self.lock:
            self.gateway_status = parsed
            self.gateway_received = time.monotonic()

    def gateway_snapshot(self):
        with self.lock:
            return self.gateway_status, self.gateway_received


def _validate(args):
    if args.algorithm != ALGORITHM:
        raise ValueError(
            "only the audited algorithm %s is deployable" % ALGORITHM
        )
    if args.obstacle_profile not in OBSTACLE_PROFILES:
        raise ValueError(
            "unsupported obstacle profile %s" % args.obstacle_profile
        )
    if args.mode == "Live" and args.confirm_motion != LIVE_CONFIRM_TOKEN:
        raise ValueError(
            "Live mode requires --confirm-motion %s"
            % LIVE_CONFIRM_TOKEN
        )
    if args.max_linear_mps <= 0.0:
        raise ValueError("max-linear-mps must be positive")
    if args.max_linear_mps > HARD_MAX_LINEAR_MPS:
        raise ValueError(
            "max-linear-mps %.3f exceeds hard cap %.3f"
            % (args.max_linear_mps, HARD_MAX_LINEAR_MPS)
        )
    if args.max_reverse_mps < 0.0:
        raise ValueError("max-reverse-mps cannot be negative")
    if args.max_reverse_mps > HARD_MAX_REVERSE_MPS:
        raise ValueError(
            "max-reverse-mps %.3f exceeds hard cap %.3f"
            % (args.max_reverse_mps, HARD_MAX_REVERSE_MPS)
        )
    if args.max_angular_radps <= 0.0:
        raise ValueError("max-angular-radps must be positive")
    if args.max_angular_radps > HARD_MAX_ANGULAR_RADPS:
        raise ValueError(
            "max-angular-radps %.3f exceeds hard cap %.3f"
            % (args.max_angular_radps, HARD_MAX_ANGULAR_RADPS)
        )
    if args.boundary_min_x >= args.boundary_max_x:
        raise ValueError("invalid x boundary")
    if args.boundary_min_y >= args.boundary_max_y:
        raise ValueError("invalid y boundary")
    if not (
        args.boundary_min_x + BOUNDARY_MARGIN_M < 0.0
        < args.boundary_max_x - BOUNDARY_MARGIN_M
        and args.boundary_min_y + BOUNDARY_MARGIN_M < 0.0
        < args.boundary_max_y - BOUNDARY_MARGIN_M
    ):
        raise ValueError(
            "boundary must contain startup origin with %.2f m margin"
            % BOUNDARY_MARGIN_M
        )
    if not (
        args.boundary_min_x + BOUNDARY_MARGIN_M
        <= args.goal_x
        <= args.boundary_max_x - BOUNDARY_MARGIN_M
        and args.boundary_min_y + BOUNDARY_MARGIN_M
        <= args.goal_y
        <= args.boundary_max_y - BOUNDARY_MARGIN_M
    ):
        raise ValueError("goal is outside the usable boundary")
    if math.hypot(args.goal_x, args.goal_y) < 0.30:
        raise ValueError("goal is already inside the completion radius")
    if args.period_s < 0.30:
        raise ValueError(
            "period-s below 0.30 is unsupported by measured planner timing"
        )
    if args.duration_s <= 0.0 or args.duration_s > 300.0:
        raise ValueError("duration-s must be in (0, 300]")
    if args.warmup_cycles < 3:
        raise ValueError("at least three no-publish warmup cycles are required")


def _mission_transition(
    mission_phase,
    goal_distance_m,
    home_heading_error_rad,
    auto_return_home,
):
    """Return the next round-trip phase and a transition reason, if any."""
    phase = str(mission_phase)
    if phase == "outbound":
        if float(goal_distance_m) > OUTBOUND_GOAL_TOLERANCE_M:
            return phase, None
        if bool(auto_return_home):
            return "return_home", "outbound_goal_reached"
        return "complete", "goal_reached"
    if phase == "return_home":
        if (
            float(goal_distance_m) <= HOME_POSITION_TOLERANCE_M
            and abs(float(home_heading_error_rad))
            <= HOME_HEADING_TOLERANCE_RAD
        ):
            return "complete", "home_pose_reached"
        return phase, None
    raise ValueError("unknown mission phase: %s" % phase)


def _home_heading_alignment_command(
    heading_error_rad,
    maximum_angular_radps,
):
    """Low-rate in-place final alignment command for the recorded home pose."""
    error = float(heading_error_rad)
    if abs(error) <= HOME_HEADING_TOLERANCE_RAD:
        return 0.0
    limit = min(
        abs(float(maximum_angular_radps)),
        HOME_ALIGNMENT_MAX_ANGULAR_RADPS,
    )
    magnitude = min(limit, max(0.06, 0.80 * abs(error)))
    return math.copysign(magnitude, error)


def _refresh_reactive_turn_memory(
    turn_sign,
    last_cycle,
    cycle,
    intervention,
):
    """Keep one avoidance direction alive for every consecutive threat."""
    if intervention is None:
        return turn_sign, last_cycle
    if turn_sign is None:
        turn_sign = float(intervention["turn_sign"])
    return float(turn_sign), int(cycle)


def _apply_return_boundary_guard(
    x,
    y,
    theta,
    linear_mps,
    angular_radps,
    goal_x,
    goal_y,
    boundary_min_x,
    boundary_max_x,
    boundary_min_y,
    boundary_max_y,
    maximum_angular_radps,
):
    """Turn toward home before the round-trip path consumes its margin."""
    usable = (
        float(boundary_min_x) + BOUNDARY_MARGIN_M,
        float(boundary_max_x) - BOUNDARY_MARGIN_M,
        float(boundary_min_y) + BOUNDARY_MARGIN_M,
        float(boundary_max_y) - BOUNDARY_MARGIN_M,
    )
    clearances = (
        float(x) - usable[0],
        usable[1] - float(x),
        float(y) - usable[2],
        usable[3] - float(y),
    )
    nearest_clearance = min(clearances)
    if nearest_clearance >= RETURN_BOUNDARY_GUARD_DISTANCE_M:
        return float(linear_mps), float(angular_radps), None
    desired_heading = math.atan2(
        float(goal_y) - float(y),
        float(goal_x) - float(x),
    )
    heading_error = math.atan2(
        math.sin(desired_heading - float(theta)),
        math.cos(desired_heading - float(theta)),
    )
    maximum_angular = abs(float(maximum_angular_radps))
    guarded_linear = min(
        max(0.0, float(linear_mps)),
        RETURN_BOUNDARY_LINEAR_CAP_MPS,
    )
    guarded_angular = float(angular_radps)
    if abs(heading_error) > 0.25:
        if (
            nearest_clearance < 0.20
            or math.cos(heading_error) < 0.50
        ):
            guarded_linear = 0.0
        turn_magnitude = min(0.35, maximum_angular)
        guarded_angular = math.copysign(
            max(abs(guarded_angular), turn_magnitude),
            heading_error,
        )
    return guarded_linear, guarded_angular, {
        "nearest_clearance_m": nearest_clearance,
        "home_heading_error_rad": heading_error,
        "linear_cap_mps": RETURN_BOUNDARY_LINEAR_CAP_MPS,
    }


def _inside_boundary(args, x, y):
    return (
        args.boundary_min_x + BOUNDARY_MARGIN_M
        <= float(x)
        <= args.boundary_max_x - BOUNDARY_MARGIN_M
        and args.boundary_min_y + BOUNDARY_MARGIN_M
        <= float(y)
        <= args.boundary_max_y - BOUNDARY_MARGIN_M
    )


def _twist_message(v, omega):
    return roslibpy.Message(
        {
            "linear": {"x": float(v), "y": 0.0, "z": 0.0},
            "angular": {
                "x": 0.0,
                "y": 0.0,
                "z": float(omega),
            },
        }
    )


def _fresh_observation(
    live,
    origin,
    timeout_s=1.0,
    after_timestamp=None,
):
    """Wait for a scan with bracketing odometry after a long CUDA solve.

    No command is published while waiting.  In Live mode the robot-side
    command watchdog therefore remains authoritative and emits zero if the
    planner or network is late.  A cached LaserScan is never processed twice:
    the multi-object tracker intentionally requires strictly increasing scan
    timestamps.  The tracker time base is the scan midpoint, never the latest
    asynchronously received odometry timestamp.
    """

    deadline = time.monotonic() + float(timeout_s)
    last_error = None
    duplicate_timestamp = None
    while time.monotonic() <= deadline:
        try:
            result = _synchronized_observation(live, origin)
            observation = result[0]
            scan_timestamp = float(observation.scan.timestamp)
            if (
                after_timestamp is not None
                and scan_timestamp <= float(after_timestamp)
            ):
                duplicate_timestamp = scan_timestamp
                time.sleep(0.02)
                continue
            return result
        except RuntimeError as exc:
            retryable = (
                "live telemetry is stale",
                "insufficient_odom_history",
                "odom_not_yet_at_scan",
            )
            if not any(value in str(exc) for value in retryable):
                raise
            last_error = exc
            time.sleep(0.02)
    if duplicate_timestamp is not None:
        raise RuntimeError(
            "scan_not_advanced: latest timestamp %.9f is not newer "
            "than %.9f"
            % (duplicate_timestamp, float(after_timestamp))
        )
    if last_error is not None:
        raise last_error
    raise RuntimeError("fresh telemetry wait expired")


def _synchronized_observation(live, origin):
    if not hasattr(live, "synchronized_snapshot"):
        return _observation(live.snapshot(), origin)
    snapshot, tracker_scan, diagnostics = live.synchronized_snapshot()
    observation, ages, emergency = _observation(snapshot, origin)
    raw_scan = replace(
        observation.scan,
        timestamp=float(diagnostics["reference_timestamp"]),
    )
    auxiliary = dict(observation.auxiliary)
    auxiliary["ego_motion_compensation"] = diagnostics
    auxiliary["motion_compensated_tracker_scan"] = tracker_scan
    observation = replace(
        observation,
        timestamp=float(diagnostics["reference_timestamp"]),
        scan=raw_scan,
        auxiliary=auxiliary,
    )
    return observation, ages, emergency


def _wait_for_synchronized_snapshot(live, timeout_s=2.0):
    deadline = time.monotonic() + float(timeout_s)
    last_error = None
    while time.monotonic() <= deadline:
        try:
            return live.synchronized_snapshot()
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.02)
    if last_error is not None:
        raise last_error
    raise RuntimeError("synchronized telemetry wait expired")


def _publish_zero(topic, count=12):
    if topic is None:
        return
    for _ in range(int(count)):
        try:
            topic.publish(_twist_message(0.0, 0.0))
        except Exception:
            pass
        time.sleep(0.05)


def _wait_for_fresh_gateway_status(
    live,
    maximum_age_s=0.75,
    timeout_s=0.90,
):
    """Allow a rosbridge callback burst to recover while motion stays zero."""
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() <= deadline:
        status, received = live.gateway_snapshot()
        if (
            isinstance(status, dict)
            and received is not None
            and time.monotonic() - float(received)
            <= float(maximum_age_s)
        ):
            return status, received
        time.sleep(0.02)
    return None, None


def _wait_for_telemetry(live, timeout_s=10.0):
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        snapshot = live.snapshot()
        if all(value is not None for value in snapshot[:3]):
            return snapshot
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for odom/scan/emergency")


def _wait_for_armed_gateway(live, timeout_s=8.0):
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        status, received = live.gateway_snapshot()
        if (
            status is not None
            and time.monotonic() - received < 0.75
            and status.get("contract") == "rl_mppi_gateway_v2"
            and bool(status.get("armed"))
        ):
            return status
        time.sleep(0.05)
    raise RuntimeError("fresh armed rl_mppi_gateway_v2 status not found")


def _origin(snapshot):
    odom = snapshot[0]
    pose = odom["pose"]["pose"]
    return (
        float(pose["position"]["x"]),
        float(pose["position"]["y"]),
        _yaw(pose["orientation"]),
    )


def _make_row(
    *,
    phase,
    cycle,
    observation,
    ages,
    emergency,
    perceived,
    plan,
    decision,
    command,
    timings,
    gateway_status,
    published,
):
    tracker = perceived.diagnostics.get("dynamic_obstacle_tracker", {})
    return {
        "phase": phase,
        "cycle": int(cycle),
        "timestamp": observation.timestamp,
        "pose": [
            observation.pose.x,
            observation.pose.y,
            observation.pose.theta,
        ],
        "twist": [observation.twist.v, observation.twist.omega],
        "ego_motion_compensation": {
            key: _json_value(value)
            for key, value in observation.auxiliary.get(
                "ego_motion_compensation", {}
            ).items()
        },
        "emergency": int(emergency),
        "telemetry_age_s": ages,
        "guard_reason": perceived.guard.get("reason"),
        "guard_emergency_stop": bool(
            perceived.guard.get("emergency_stop", False)
        ),
        "local_obstacle_count": len(
            perceived.observation.local_obstacles
        ),
        "forecast_count": len(
            perceived.observation.auxiliary.get(
                "probabilistic_obstacle_forecasts", ()
            )
        ),
        "tracker": {
            key: _json_value(value) for key, value in tracker.items()
        },
        "proposed": plan.proposed_control.values.tolist(),
        "arbitrated": decision.executed_control.values.tolist(),
        "bounded_command": list(command),
        "reverse_requested": bool(
            float(decision.executed_control.values[0]) < -1.0e-12
        ),
        "reverse_executed": bool(float(command[0]) < -1.0e-12),
        "linear_bound_applied": bool(
            abs(
                float(decision.executed_control.values[0])
                - float(command[0])
            )
            > 1.0e-12
        ),
        "safety_overridden": bool(decision.overridden),
        "safety_reason": str(decision.reason),
        "planner_optimizer": plan.diagnostics.get("optimizer"),
        "planner_diagnostics": {
            key: _json_value(plan.diagnostics.get(key))
            for key in (
                "optimizer",
                "rho_t",
                "guided_fraction",
                "actor_elite_fraction",
                "paper_guided_unique_sequences",
                "supervised_proposal_count",
                "reliability_guided_fraction_applied",
                "reliability_guided_fraction_next",
                "reliability_guided_fraction_raw_applied",
                "completion_handover_authority",
                "residual_safety_shield_accepted",
                "residual_safety_shield_selected_source",
                "probabilistic_obstacle_maximum_step_probability",
                "compute_ms",
                "candidate_feasible_fraction",
                "residual_safety_shield_nominal_compute_ms",
                "residual_safety_shield_residual_compute_ms",
                "residual_safety_shield_parallel_planning_enabled",
                "profile_mppi_sampling_ms",
                "profile_mppi_batch_rollout_ms",
                "profile_mppi_cost_ms",
                "profile_mppi_weighting_update_ms",
                "profile_mppi_final_rollout_ms",
                "profile_mppi_solve_total_ms",
                "profile_planner_state_reference_ms",
                "profile_planner_prior_ms",
            )
            if key in plan.diagnostics
        },
        "timing_ms": timings,
        "gateway_status": gateway_status,
        "published": bool(published),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.31.200")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--mode", choices=("Shadow", "Live"), default="Shadow"
    )
    parser.add_argument("--algorithm", default=ALGORITHM)
    parser.add_argument(
        "--obstacle-profile",
        choices=OBSTACLE_PROFILES,
        default="Generic",
    )
    parser.add_argument("--confirm-motion", default="")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--boundary-min-x", type=float, default=-0.75)
    parser.add_argument("--boundary-max-x", type=float, default=3.50)
    parser.add_argument("--boundary-min-y", type=float, default=-1.50)
    parser.add_argument("--boundary-max-y", type=float, default=1.50)
    parser.add_argument("--max-linear-mps", type=float, default=0.06)
    parser.add_argument("--max-reverse-mps", type=float, default=0.10)
    parser.add_argument(
        "--max-angular-radps", type=float, default=0.25
    )
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--period-s", type=float, default=0.35)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--auto-return-home", action="store_true")
    parser.add_argument("--static-background-map", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _validate(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the complete method")
    args.output.mkdir(parents=True, exist_ok=False)

    config = build_real_config(
        seed=args.seed,
        goal_forward_m=args.goal_x,
        goal_left_m=args.goal_y,
    )
    # The lab GPU showed 0.86--0.95 s tail latency at the paper evaluation
    # budget (300 samples x 2 iterations), exceeding both the 0.35 s control
    # period and the robot gateway watchdog.  Keep the same complete planner,
    # learned checkpoints, 36-step horizon and two-stage optimizer, but use a
    # bounded real-time rollout budget (50 x 2) for physical deployment.
    config["planner"]["num_samples"] = REALTIME_MPPI_SAMPLES
    # The long physical run on 2026-08-02 produced a non-finite residual
    # inside the captured CUDA graph after 280 cycles.  Keep the GPU residual
    # rollout, but execute it eagerly so bad graph state cannot accumulate
    # across the complete outbound/return session.
    config["planner"]["residual_cuda_graph_enabled"] = True
    if args.obstacle_profile == "ComplexScene":
        tracker_config = config["perception"][
            "dynamic_obstacle_tracker"
        ]
        tracker_config.update(
            {
                "maximum_tracks": 3,
                "association_gate_m": 0.80,
                "maximum_unobserved_duration_s": 0.80,
                "track_retirement_duration_s": 1.00,
                "motion_confirmation_enabled": True,
                "motion_confirmation_required_observations": 3,
                "motion_confirmation_minimum_speed_mps": 0.15,
                "known_static_filter_enabled": False,
                "known_static_track_rejection_enabled": False,
            }
        )
        config["perception"]["scan_guard"][
            "dynamic_escape_use_vetted_planner_control"
        ] = True
        # Do not freeze the robot merely because no dynamic forecast is
        # currently valid.  Standard MPPI continues from the raw scan/local
        # obstacle layer; a valid forecast restores the complete Paper path.
        config["planner"][
            "probabilistic_obstacle_missing_forecast_action"
        ] = "scan_only"
        # Keep both shield controllers concurrent on the laptop GPU.  Dynamic
        # forecasts are consumed by the bounded TTC/path-intersection arbiter
        # below; the non-real-time batch probability path is not invoked.
        config["planner"]["residual_safety_shield"][
            "parallel_planning_enabled"
        ] = True
    config["real_robot_deployment"].update(
        {
            "contract": DEPLOYMENT_CONTRACT,
            "mode": args.mode,
            "algorithm": args.algorithm,
            "obstacle_profile": args.obstacle_profile,
            "publish_enabled": args.mode == "Live",
            "boundary": {
                "min_x": args.boundary_min_x,
                "max_x": args.boundary_max_x,
                "min_y": args.boundary_min_y,
                "max_y": args.boundary_max_y,
                "margin_m": BOUNDARY_MARGIN_M,
            },
            "speed_limits": {
                "max_linear_mps": args.max_linear_mps,
                "max_reverse_mps": args.max_reverse_mps,
                "max_angular_radps": args.max_angular_radps,
            },
            "static_background": {
                "enabled": args.static_background_map is not None,
                "map_path": (
                    str(args.static_background_map.resolve())
                    if args.static_background_map is not None
                    else None
                ),
                "scope": "dynamic_tracker_only",
            },
            "ego_motion_compensation": {
                "enabled": True,
                "scope": "dynamic_tracker_only",
                "time_base": "laser_scan_midpoint",
                "odom_mode": "bracketed_planar_interpolation",
                "maximum_odom_gap_s": 0.20,
                "beam_deskew": True,
                "raw_scan_safety_path_unchanged": True,
            },
            "realtime_control": {
                "mppi_samples_per_iteration": REALTIME_MPPI_SAMPLES,
                "command_deadline_s": REALTIME_COMMAND_DEADLINE_S,
                "wheel_track_m": PHYSICAL_WHEEL_TRACK_M,
                "wheel_speed_limit_mps": (
                    PHYSICAL_WHEEL_SPEED_LIMIT_MPS
                ),
                "late_result_action": "publish_zero",
                "forecast_risk_mode": "per_scan_ttc_path_intersection",
                "full_probabilistic_batch_risk_enabled": False,
                "residual_cuda_graph_enabled": True,
            },
            "auto_return_home": {
                "enabled": bool(args.auto_return_home),
                "home_pose": [0.0, 0.0, 0.0],
                "position_tolerance_m": HOME_POSITION_TOLERANCE_M,
                "heading_tolerance_rad": HOME_HEADING_TOLERANCE_RAD,
                "alignment_max_angular_radps": (
                    HOME_ALIGNMENT_MAX_ANGULAR_RADPS
                ),
            },
        }
    )
    background_filter = None
    if args.static_background_map is not None:
        if args.obstacle_profile != "ComplexScene":
            raise ValueError(
                "--static-background-map is supported only for ComplexScene"
            )
        background_filter = CalibratedStaticBackgroundFilter(
            args.static_background_map
        )
        # Opt-in with the calibrated map only.  The no-map ComplexScene path
        # retains its historical configuration exactly.
        config["perception"]["dynamic_obstacle_tracker"][
            "minimum_cluster_beams"
        ] = 3
        config["real_robot_deployment"]["static_background"].update(
            {
                "map_sha256": background_filter.map_sha256,
                "metadata_sha256": background_filter.metadata_sha256,
                "contract": background_filter.metadata["contract"],
                "endpoint_gate_m": background_filter.endpoint_gate_m,
                "cluster_center_gate_m": (
                    background_filter.cluster_center_gate_m
                ),
                "minimum_alignment_fraction": (
                    background_filter.minimum_alignment_fraction
                ),
                "alignment_gate_m": background_filter.alignment_gate_m,
                "minimum_voxel_hits": background_filter.minimum_voxel_hits,
            }
        )
    config_path = args.output / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

    components = make_components(config, ROOT)
    controller = components["controller"]
    perception = components["perception"]
    safety = components["safety"]
    reference = components["reference"]
    outbound_reference = reference
    home_reference = PoseGoal(
        0.0,
        0.0,
        0.0,
        position_tolerance=HOME_POSITION_TOLERANCE_M,
        heading_tolerance=HOME_HEADING_TOLERANCE_RAD,
        reference_id="recorded_start_pose",
    )
    if args.obstacle_profile == "HumanLeg":
        perception.dynamic_obstacle_tracker = (
            HumanLegMotionBootstrapTracker.from_existing(
                perception.dynamic_obstacle_tracker
            )
        )
    elif args.obstacle_profile == "ComplexScene":
        perception.dynamic_obstacle_tracker = (
            MotionBootstrapMultiObstacleTracker.from_existing(
                perception.dynamic_obstacle_tracker
            )
        )
        if background_filter is not None:
            perception.dynamic_obstacle_tracker = (
                BackgroundFilteredMotionBootstrapTracker.from_existing(
                    perception.dynamic_obstacle_tracker,
                    background_filter,
                )
            )
        perception.dynamic_obstacle_tracker = (
            MaplessStaticDynamicFilter.from_existing(
                perception.dynamic_obstacle_tracker
            )
        )
    controller_chain = [controller.__class__.__name__]
    inner_controller = getattr(controller, "residual_controller", None)
    if inner_controller is not None:
        controller_chain.append(inner_controller.__class__.__name__)
    if "PaperRLDrivenMppiController" not in controller_chain:
        raise RuntimeError(
            "PaperRLDrivenMppiController missing from controller chain: %r"
            % controller_chain
        )
    controller.reset(args.seed)
    perception.reset()
    safety.reset()

    runtime_manifest = _runtime_manifest(config)
    runtime_manifest.update(
        {
            "contract": DEPLOYMENT_CONTRACT,
            "controller_class": controller.__class__.__name__,
            "controller_chain": controller_chain,
            "cuda_device": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "mode": args.mode,
            "algorithm": args.algorithm,
            "obstacle_profile": args.obstacle_profile,
        }
    )
    manifest_path = args.output / "runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(runtime_manifest, indent=2), encoding="utf-8"
    )

    client = roslibpy.Ros(host=args.host, port=args.port)
    sensor_forward_offset_m = float(
        config["perception"]["dynamic_obstacle_tracker"].get(
            "sensor_forward_offset_m", 0.0
        )
    )
    live = OperatorTelemetry(sensor_forward_offset_m)
    telemetry_topics = [
        roslibpy.Topic(client, "/odom", "nav_msgs/Odometry"),
        roslibpy.Topic(client, "/scan", "sensor_msgs/LaserScan"),
        roslibpy.Topic(
            client, "/emergencybt_status", "std_msgs/Int16"
        ),
    ]
    callbacks = [
        live.on_odom,
        live.on_scan,
        live.on_emergency,
    ]
    if args.mode == "Live":
        telemetry_topics.append(
            roslibpy.Topic(
                client, "/mppi_gateway/status", "std_msgs/String"
            )
        )
        callbacks.append(live.on_gateway)
    command_topic = None
    rows = []
    published_count = 0
    reached_goal = False
    outbound_goal_reached = False
    home_pose_reached = False
    mission_phase = "outbound"
    active_goal_x = float(args.goal_x)
    active_goal_y = float(args.goal_y)
    active_reference = outbound_reference
    final_home_position_error_m = None
    final_home_heading_error_rad = None
    planner_numeric_recovery_count = 0
    consecutive_planner_numeric_failures = 0
    stop_reason = "duration_complete"
    origin = None
    last_observation_timestamp = None
    reactive_turn_sign = None
    reactive_last_cycle = -1000
    previous_command_w = 0.0
    failure = None
    log_path = args.output / "operator_cycles.jsonl"
    try:
        client.run(timeout=8)
        if not client.is_connected:
            raise RuntimeError("rosbridge connection did not become ready")
        for topic, callback in zip(telemetry_topics, callbacks):
            topic.subscribe(callback)
        _wait_for_telemetry(live)
        first_snapshot, _, _ = _wait_for_synchronized_snapshot(live)
        origin = _origin(first_snapshot)

        if background_filter is not None:
            first_observation = _synchronized_observation(
                live, origin
            )[0]
            relocalization = background_filter.relocalize(
                first_observation, origin
            )
            alignment = background_filter.validate_alignment(
                first_observation
            )
            print(
                "static_map_relocalization matched=%.3f median=%.3fm "
                "heading_correction=%.3frad"
                % (
                    relocalization["matched_fraction"],
                    relocalization["median_distance_m"],
                    relocalization["heading_correction_rad"],
                ),
                flush=True,
            )

        if args.mode == "Live":
            gateway_status = _wait_for_armed_gateway(live)
            if int(first_snapshot[2]) != 0:
                raise RuntimeError("physical emergency is active")
            limits = gateway_status.get("limits", {})
            if (
                not bool(limits.get("allow_reverse", False))
                or abs(
                    float(limits.get("max_linear_mps", -1.0))
                    - args.max_linear_mps
                )
                > 1.0e-9
                or abs(
                    float(limits.get("max_reverse_mps", -1.0))
                    - args.max_reverse_mps
                )
                > 1.0e-9
                or abs(
                    float(limits.get("max_angular_radps", -1.0))
                    - args.max_angular_radps
                )
                > 1.0e-9
            ):
                raise RuntimeError("laptop and gateway limits differ")
            command_topic = roslibpy.Topic(
                client, "/mppi_safe_cmd", "geometry_msgs/Twist"
            )
            command_topic.advertise()

        with log_path.open("w", encoding="utf-8") as stream:
            # Warm CUDA graphs and recurrent controller state without publishing.
            for warmup_index in range(int(args.warmup_cycles)):
                cycle_started = time.perf_counter()
                observation, ages, emergency = _fresh_observation(
                    live,
                    origin,
                    after_timestamp=last_observation_timestamp,
                )
                last_observation_timestamp = float(
                    observation.timestamp
                )
                perceived = perception.process(observation)
                planner_observation, forecast_relevance = (
                    _path_relevant_forecasts(
                        perceived.observation,
                        args.goal_x,
                        args.goal_y,
                        args.max_linear_mps,
                        args.max_angular_radps,
                    )
                )
                forecast_relevance = dict(forecast_relevance)
                forecast_relevance["full_risk_evaluated"] = False
                if forecast_relevance["retained_count"]:
                    planner_auxiliary = dict(
                        planner_observation.auxiliary
                    )
                    planner_auxiliary[
                        "probabilistic_obstacle_forecasts"
                    ] = ()
                    planner_observation = replace(
                        planner_observation,
                        auxiliary=planner_auxiliary,
                    )
                plan = controller.plan(
                    planner_observation, reference
                )
                decision = safety.arbitrate(
                    plan.proposed_control,
                    perceived.guard,
                    plan.diagnostics,
                )
                controller.observe_safety_decision(decision)
                bounded = (0.0, 0.0)
                total_ms = 1000.0 * (
                    time.perf_counter() - cycle_started
                )
                row = _make_row(
                    phase="warmup",
                    cycle=warmup_index,
                    observation=observation,
                    ages=ages,
                    emergency=emergency,
                    perceived=perceived,
                    plan=plan,
                    decision=decision,
                    command=bounded,
                    timings={"total_wall_ms": total_ms},
                    gateway_status=live.gateway_snapshot()[0],
                    published=False,
                )
                rows.append(row)
                row["forecast_path_relevance"] = forecast_relevance
                stream.write(
                    json.dumps(row, default=_json_value) + "\n"
                )
                stream.flush()
                optimizer_identity = plan.diagnostics.get("optimizer")
                if optimizer_identity not in (None, "paper_rl_driven"):
                    raise RuntimeError("optimizer identity Gate failed")
                print(
                    "warmup=%d total_ms=%.1f optimizer=%s"
                    % (
                        warmup_index,
                        total_ms,
                        (
                            optimizer_identity
                            if optimizer_identity is not None
                            else "shield_early_exit"
                        ),
                    ),
                    flush=True,
                )
                warmup_delay = float(args.period_s) - (
                    time.perf_counter() - cycle_started
                )
                if warmup_delay > 0.0:
                    time.sleep(warmup_delay)

            session_started = time.monotonic()
            cycle = 0
            while (
                time.monotonic() - session_started
                < float(args.duration_s)
            ):
                cycle_started = time.perf_counter()
                observation, ages, emergency = _fresh_observation(
                    live,
                    origin,
                    after_timestamp=last_observation_timestamp,
                )
                last_observation_timestamp = float(
                    observation.timestamp
                )
                if emergency != 0:
                    stop_reason = "physical_emergency_active"
                    raise RuntimeError(stop_reason)
                if not _inside_boundary(
                    args, observation.pose.x, observation.pose.y
                ):
                    stop_reason = "workspace_boundary"
                    raise RuntimeError(stop_reason)
                goal_distance = math.hypot(
                    active_goal_x - observation.pose.x,
                    active_goal_y - observation.pose.y,
                )
                goal_heading = math.atan2(
                    active_goal_y - observation.pose.y,
                    active_goal_x - observation.pose.x,
                )
                goal_heading_error = math.atan2(
                    math.sin(goal_heading - observation.pose.theta),
                    math.cos(goal_heading - observation.pose.theta),
                )
                home_heading_error = math.atan2(
                    math.sin(-observation.pose.theta),
                    math.cos(-observation.pose.theta),
                )
                control_heading_error = (
                    home_heading_error
                    if mission_phase == "return_home"
                    and goal_distance <= HOME_POSITION_TOLERANCE_M
                    else goal_heading_error
                )
                next_phase, transition_reason = _mission_transition(
                    mission_phase,
                    goal_distance,
                    home_heading_error,
                    args.auto_return_home,
                )
                if transition_reason == "outbound_goal_reached":
                    reached_goal = True
                    outbound_goal_reached = True
                    mission_phase = next_phase
                    active_goal_x = 0.0
                    active_goal_y = 0.0
                    active_reference = home_reference
                    controller.reset(args.seed)
                    safety.reset()
                    reactive_turn_sign = None
                    reactive_last_cycle = -1000
                    previous_command_w = 0.0
                    _publish_zero(command_topic)
                    print(
                        "mission_transition=return_home "
                        "recorded_home=(0.000,0.000,0.000)",
                        flush=True,
                    )
                    continue
                if transition_reason == "goal_reached":
                    reached_goal = True
                    outbound_goal_reached = True
                    mission_phase = next_phase
                    stop_reason = transition_reason
                    break
                if transition_reason == "home_pose_reached":
                    home_pose_reached = True
                    mission_phase = next_phase
                    final_home_position_error_m = goal_distance
                    final_home_heading_error_rad = home_heading_error
                    stop_reason = "round_trip_complete"
                    break

                perception_started = time.perf_counter()
                perceived = perception.process(observation)
                perception_ms = 1000.0 * (
                    time.perf_counter() - perception_started
                )
                planner_observation, forecast_relevance = (
                    _path_relevant_forecasts(
                        perceived.observation,
                        active_goal_x,
                        active_goal_y,
                        args.max_linear_mps,
                        args.max_angular_radps,
                    )
                )
                forecast_relevance = dict(forecast_relevance)
                full_forecast_risk_evaluated = False
                forecast_relevance["full_risk_evaluated"] = (
                    full_forecast_risk_evaluated
                )
                if forecast_relevance["retained_count"]:
                    planner_auxiliary = dict(
                        planner_observation.auxiliary
                    )
                    planner_auxiliary[
                        "probabilistic_obstacle_forecasts"
                    ] = ()
                    planner_observation = replace(
                        planner_observation,
                        auxiliary=planner_auxiliary,
                    )
                planner_started = time.perf_counter()
                try:
                    plan = controller.plan(
                        planner_observation, active_reference
                    )
                except FloatingPointError:
                    planner_numeric_recovery_count += 1
                    consecutive_planner_numeric_failures += 1
                    _publish_zero(command_topic)
                    previous_command_w = 0.0
                    controller.reset(args.seed)
                    safety.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    print(
                        "planner_numeric_recovery=%d cycle=%d"
                        % (planner_numeric_recovery_count, cycle),
                        flush=True,
                    )
                    if consecutive_planner_numeric_failures >= 3:
                        raise
                    cycle += 1
                    continue
                consecutive_planner_numeric_failures = 0
                planner_ms = 1000.0 * (
                    time.perf_counter() - planner_started
                )
                decision = safety.arbitrate(
                    plan.proposed_control,
                    perceived.guard,
                    plan.diagnostics,
                )
                controller.observe_safety_decision(decision)
                raw_v = float(decision.executed_control.values[0])
                raw_w = float(decision.executed_control.values[1])
                gateway_status, gateway_rx = live.gateway_snapshot()
                tracker_diagnostics = perceived.diagnostics.get(
                    "dynamic_obstacle_tracker", {}
                )
                provisional_intervention = None
                reactive_arc_hold_applied = False
                front_clearance_arc = None
                near_body_speed_governor = None
                near_body_turn_escape = None
                goal_heading_stabilizer_applied = False
                protected_reverse_applied = False
                wheel_envelope_applied = False
                realtime_deadline_missed = False
                angular_slew_limited = False
                home_heading_alignment_applied = False
                return_boundary_guard = None
                gateway_status_recovered = False
                bounded_v = bound_signed_speed(
                    raw_v,
                    args.max_linear_mps,
                    0.0,
                    allow_reverse=False,
                )
                bounded_w = min(
                    max(
                        raw_w, -float(args.max_angular_radps)
                    ),
                    float(args.max_angular_radps),
                )
                if bool(
                    perceived.guard.get("emergency_stop", False)
                ):
                    dynamic_threat_confirmed = bool(
                        tracker_diagnostics.get("motion_confirmed", False)
                        or int(
                            tracker_diagnostics.get(
                                "valid_forecast_count", 0
                            )
                            or 0
                        )
                        > 0
                    )
                    if perceived.guard.get("reason") == "near_body_hard_stop":
                        bounded_v, bounded_w, protected_reverse_applied = (
                            protected_reverse_escape(
                                raw_v,
                                bounded_w,
                                args.max_reverse_mps,
                                (
                                    gateway_status.get("rear_clearance_m")
                                    if isinstance(gateway_status, dict)
                                    else None
                                ),
                                0.80,
                                _fresh_rear_gateway(gateway_status),
                                dynamic_threat_confirmed,
                            )
                        )
                        if not protected_reverse_applied:
                            escape_evidence = _scan_turn_escape_evidence(
                                observation.scan
                            )
                            if escape_evidence is not None:
                                (
                                    escape_w,
                                    near_body_turn_escape,
                                ) = verified_near_body_turn_escape(
                                    raw_w,
                                    escape_evidence[
                                        "nearest_obstacle_angle_rad"
                                    ],
                                    escape_evidence["left_clearance_m"],
                                    escape_evidence["right_clearance_m"],
                                    args.max_angular_radps,
                                )
                                if near_body_turn_escape is not None:
                                    bounded_v = 0.0
                                    bounded_w = escape_w
                                    reactive_turn_sign = float(
                                        near_body_turn_escape["turn_sign"]
                                    )
                                    reactive_last_cycle = cycle
                    else:
                        bounded_v = 0.0
                        bounded_w = 0.0
                else:
                    if cycle - reactive_last_cycle > 3:
                        reactive_turn_sign = None
                    reactive_candidates = list(
                        tracker_diagnostics.get(
                            "provisional_collision_candidates", ()
                        )
                    )
                    if not full_forecast_risk_evaluated:
                        reactive_candidates.extend(
                            forecast_relevance.get(
                                "collision_candidates", ()
                            )
                        )
                    goal_turn_sign = (
                        None
                        if abs(control_heading_error) < 0.10
                        else (
                            -1.0 if control_heading_error < 0.0 else 1.0
                        )
                    )
                    (
                        bounded_v,
                        bounded_w,
                        provisional_intervention,
                    ) = apply_provisional_collision_intervention(
                        bounded_v,
                        bounded_w,
                        reactive_candidates,
                        args.max_angular_radps,
                        preferred_turn_sign=reactive_turn_sign,
                        goal_turn_sign=goal_turn_sign,
                    )
                    (
                        reactive_turn_sign,
                        reactive_last_cycle,
                    ) = _refresh_reactive_turn_memory(
                        reactive_turn_sign,
                        reactive_last_cycle,
                        cycle,
                        provisional_intervention,
                    )
                    gateway_front = (
                        gateway_status.get("front_clearance_m")
                        if isinstance(gateway_status, dict)
                        else None
                    )
                    gateway_near = (
                        gateway_status.get("near_body_clearance_m")
                        if isinstance(gateway_status, dict)
                        else None
                    )
                    direction_lock_hazard = any(
                        value is not None and float(value) < 1.20
                        for value in (gateway_front, gateway_near)
                    )
                    if (
                        provisional_intervention is None
                        and reactive_turn_sign is not None
                        and 0 < cycle - reactive_last_cycle <= 3
                        and direction_lock_hazard
                    ):
                        (
                            bounded_v,
                            bounded_w,
                            reactive_arc_hold_applied,
                        ) = apply_turn_direction_lock(
                            bounded_v,
                            bounded_w,
                            reactive_turn_sign,
                            args.max_angular_radps,
                        )
                    (
                        bounded_v,
                        bounded_w,
                        front_clearance_arc,
                    ) = apply_front_clearance_arc_governor(
                        bounded_v,
                        bounded_w,
                        (
                            gateway_status.get("front_clearance_m")
                            if isinstance(gateway_status, dict)
                            else None
                        ),
                        args.max_angular_radps,
                        front_obstacle_angle_rad=(
                            None
                            if provisional_intervention is not None
                            else _scan_front_obstacle_angle(
                                observation.scan
                            )
                        ),
                        minimum_arc_angular_radps=(
                            0.18
                            if provisional_intervention is not None
                            else 0.30
                        ),
                    )
                    (
                        bounded_v,
                        near_body_speed_governor,
                    ) = apply_near_body_speed_governor(
                        bounded_v,
                        (
                            gateway_status.get("near_body_clearance_m")
                            if isinstance(gateway_status, dict)
                            else None
                        ),
                    )
                    # MPPI already optimizes point-goal heading jointly with
                    # obstacle clearance.  A post-planner heading override
                    # caused two measured 0.6 rad/s turn reversals, so it is
                    # intentionally disabled in the live command path.
                    if (
                        mission_phase == "return_home"
                        and goal_distance <= HOME_POSITION_TOLERANCE_M
                    ):
                        bounded_v = 0.0
                        bounded_w = _home_heading_alignment_command(
                            home_heading_error,
                            args.max_angular_radps,
                        )
                        home_heading_alignment_applied = True

                if (
                    mission_phase == "return_home"
                    and not bool(
                        perceived.guard.get("emergency_stop", False)
                    )
                ):
                    (
                        bounded_v,
                        bounded_w,
                        return_boundary_guard,
                    ) = _apply_return_boundary_guard(
                        observation.pose.x,
                        observation.pose.y,
                        observation.pose.theta,
                        bounded_v,
                        bounded_w,
                        active_goal_x,
                        active_goal_y,
                        args.boundary_min_x,
                        args.boundary_max_x,
                        args.boundary_min_y,
                        args.boundary_max_y,
                        args.max_angular_radps,
                    )

                (
                    bounded_v,
                    bounded_w,
                    wheel_envelope_applied,
                ) = couple_differential_drive_command(
                    bounded_v,
                    bounded_w,
                    PHYSICAL_WHEEL_TRACK_M,
                    PHYSICAL_WHEEL_SPEED_LIMIT_MPS,
                )

                projected_x = (
                    observation.pose.x
                    + bounded_v
                    * math.cos(observation.pose.theta)
                    * 0.75
                )
                projected_y = (
                    observation.pose.y
                    + bounded_v
                    * math.sin(observation.pose.theta)
                    * 0.75
                )
                if not _inside_boundary(
                    args, projected_x, projected_y
                ):
                    bounded_v = 0.0
                    bounded_w = 0.0
                    stop_reason = "workspace_boundary_prediction"

                # Never send a stale optimization result after the robot-side
                # watchdog budget.  The gateway is already fail-closed; this
                # explicit zero also makes the laptop log unambiguous.
                if (
                    time.perf_counter() - cycle_started
                    > REALTIME_COMMAND_DEADLINE_S
                ):
                    bounded_v = 0.0
                    bounded_w = 0.0
                    realtime_deadline_missed = True

                immediate_zero_required = bool(
                    bounded_w == 0.0
                    and (
                        realtime_deadline_missed
                        or stop_reason == "workspace_boundary_prediction"
                        or (
                            perceived.guard.get("emergency_stop", False)
                            and near_body_turn_escape is None
                            and not protected_reverse_applied
                        )
                    )
                )
                if not immediate_zero_required:
                    bounded_w, angular_slew_limited = (
                        limit_angular_command_step(
                            bounded_w, previous_command_w
                        )
                    )
                previous_command_w = bounded_w

                published = False
                if command_topic is not None:
                    if (
                        not isinstance(gateway_status, dict)
                        or gateway_rx is None
                        or time.monotonic() - gateway_rx > 0.75
                    ):
                        # A long CUDA/Python section can briefly starve the
                        # rosbridge callback thread even while /scan and the
                        # robot-side gateway remain healthy.  Hold zero while
                        # waiting for a fresh status; never send the stale
                        # optimization result after recovery.
                        _publish_zero(command_topic, count=3)
                        gateway_status, gateway_rx = (
                            _wait_for_fresh_gateway_status(live)
                        )
                        if gateway_status is None:
                            stop_reason = "gateway_status_stale"
                            raise RuntimeError(stop_reason)
                        bounded_v = 0.0
                        bounded_w = 0.0
                        previous_command_w = 0.0
                        gateway_status_recovered = True
                    if not bool(gateway_status.get("armed")):
                        stop_reason = "gateway_disarmed"
                        raise RuntimeError(stop_reason)
                    command_topic.publish(
                        _twist_message(bounded_v, bounded_w)
                    )
                    published_count += 1
                    published = True

                total_ms = 1000.0 * (
                    time.perf_counter() - cycle_started
                )
                row = _make_row(
                    phase="control",
                    cycle=cycle,
                    observation=observation,
                    ages=ages,
                    emergency=emergency,
                    perceived=perceived,
                    plan=plan,
                    decision=decision,
                    command=(bounded_v, bounded_w),
                    timings={
                        "perception_ms": perception_ms,
                        "planner_wall_ms": planner_ms,
                        "total_wall_ms": total_ms,
                    },
                    gateway_status=live.gateway_snapshot()[0],
                    published=published,
                )
                row["goal_distance_m"] = goal_distance
                row["mission_phase"] = mission_phase
                row["active_goal"] = [active_goal_x, active_goal_y]
                row["home_position_error_m"] = math.hypot(
                    observation.pose.x,
                    observation.pose.y,
                )
                row["home_heading_error_rad"] = home_heading_error
                row["forecast_path_relevance"] = forecast_relevance
                row["command_arbitration"] = {
                    "provisional_collision_intervention": (
                        provisional_intervention
                    ),
                    "reactive_arc_hold": bool(
                        reactive_arc_hold_applied
                    ),
                    "front_clearance_arc_governor": front_clearance_arc,
                    "near_body_speed_governor": near_body_speed_governor,
                    "near_body_turn_escape": near_body_turn_escape,
                    "angular_slew_limited": bool(angular_slew_limited),
                    "goal_heading_error_rad": control_heading_error,
                    "goal_heading_stabilizer": bool(
                        goal_heading_stabilizer_applied
                    ),
                    "home_heading_alignment": bool(
                        home_heading_alignment_applied
                    ),
                    "return_boundary_guard": return_boundary_guard,
                    "gateway_status_recovered": bool(
                        gateway_status_recovered
                    ),
                    "protected_reverse_escape": bool(
                        protected_reverse_applied
                    ),
                    "wheel_envelope_applied": bool(
                        wheel_envelope_applied
                    ),
                    "realtime_deadline_missed": bool(
                        realtime_deadline_missed
                    ),
                    "deadline_s": REALTIME_COMMAND_DEADLINE_S,
                }
                rows.append(row)
                stream.write(
                    json.dumps(row, default=_json_value) + "\n"
                )
                stream.flush()
                print(
                    "cycle=%d phase=%s pose=(%.3f,%.3f,%.3f) "
                    "goal_d=%.3f cmd=(%.3f,%.3f) "
                    "safety=%s total_ms=%.1f publish=%s"
                    % (
                        cycle,
                        mission_phase,
                        observation.pose.x,
                        observation.pose.y,
                        observation.pose.theta,
                        goal_distance,
                        bounded_v,
                        bounded_w,
                        decision.reason,
                        total_ms,
                        published,
                    ),
                    flush=True,
                )
                cycle += 1
                next_time = cycle_started + float(args.period_s)
                delay = next_time - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if stop_reason == "duration_complete":
            stop_reason = "operator_exception:%s" % failure["type"]
        (args.output / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        (args.output / "stderr.log").write_text(
            failure["traceback"], encoding="utf-8"
        )
        raise
    finally:
        _publish_zero(command_topic)
        if command_topic is not None:
            try:
                command_topic.unadvertise()
            except Exception:
                pass
        for topic in telemetry_topics:
            try:
                topic.unsubscribe()
            except Exception:
                pass
        client.terminate()

        control_rows = [
            row for row in rows if row["phase"] == "control"
        ]
        timings = np.asarray(
            [
                row["timing_ms"]["total_wall_ms"]
                for row in control_rows
            ],
            dtype=np.float64,
        )
        summary = {
            "contract": DEPLOYMENT_CONTRACT,
            "mode": args.mode,
            "algorithm": args.algorithm,
            "obstacle_profile": args.obstacle_profile,
            "controller_class": controller.__class__.__name__,
            "controller_chain": controller_chain,
            "cuda_device": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "config_sha256": config_sha,
            "runtime_manifest_sha256": _sha256(manifest_path),
            "origin_odom": origin,
            "goal": [args.goal_x, args.goal_y],
            "auto_return_home": bool(args.auto_return_home),
            "mission_phase": mission_phase,
            "outbound_goal_reached": outbound_goal_reached,
            "home_pose_reached": home_pose_reached,
            "home_pose_target": [0.0, 0.0, 0.0],
            "home_position_tolerance_m": HOME_POSITION_TOLERANCE_M,
            "home_heading_tolerance_rad": HOME_HEADING_TOLERANCE_RAD,
            "final_home_position_error_m": (
                final_home_position_error_m
                if final_home_position_error_m is not None
                else (
                    control_rows[-1].get("home_position_error_m")
                    if control_rows
                    else None
                )
            ),
            "final_home_heading_error_rad": (
                final_home_heading_error_rad
                if final_home_heading_error_rad is not None
                else (
                    control_rows[-1].get("home_heading_error_rad")
                    if control_rows
                    else None
                )
            ),
            "boundary": {
                "min_x": args.boundary_min_x,
                "max_x": args.boundary_max_x,
                "min_y": args.boundary_min_y,
                "max_y": args.boundary_max_y,
                "margin_m": BOUNDARY_MARGIN_M,
            },
            "speed_limits": {
                "max_linear_mps": args.max_linear_mps,
                "max_reverse_mps": args.max_reverse_mps,
                "max_angular_radps": args.max_angular_radps,
            },
            "warmup_cycles": args.warmup_cycles,
            "control_cycles": len(control_rows),
            "publish_count": published_count,
            "gateway_status_recovery_cycles": len([
                row
                for row in control_rows
                if row.get("command_arbitration", {}).get(
                    "gateway_status_recovered", False
                )
            ]),
            "planner_numeric_recovery_count": (
                planner_numeric_recovery_count
            ),
            "reverse": {
                "requested_cycles": len([
                    row
                    for row in control_rows
                    if row.get("reverse_requested", False)
                ]),
                "executed_cycles": len([
                    row
                    for row in control_rows
                    if row.get("reverse_executed", False)
                ]),
                "requested_but_not_executed_cycles": len([
                    row
                    for row in control_rows
                    if row.get("reverse_requested", False)
                    and not row.get("reverse_executed", False)
                ]),
                "executed_duration_s": float(args.period_s) * len([
                    row
                    for row in control_rows
                    if row.get("reverse_executed", False)
                ]),
                "minimum_gateway_rear_clearance_m": (
                    min(
                        float(row["gateway_status"][
                            "rear_clearance_m"
                        ])
                        for row in control_rows
                        if row.get("gateway_status") is not None
                        and row["gateway_status"].get(
                            "rear_clearance_m"
                        ) is not None
                    )
                    if any(
                        row.get("gateway_status") is not None
                        and row["gateway_status"].get(
                            "rear_clearance_m"
                        ) is not None
                        for row in control_rows
                    )
                    else None
                ),
            },
            "reached_goal": reached_goal,
            "stop_reason": stop_reason,
            "failure": failure,
            "timing_ms": (
                None
                if timings.size == 0
                else {
                    "mean": float(np.mean(timings)),
                    "p50": float(np.percentile(timings, 50)),
                    "p95": float(np.percentile(timings, 95)),
                    "max": float(np.max(timings)),
                }
            ),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
