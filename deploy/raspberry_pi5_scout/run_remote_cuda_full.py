#!/usr/bin/env python3
"""Run latest Full Proposed on a CUDA PC and command a guarded Pi gateway."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch
import yaml

from deploy.raspberry_pi5_scout.build_pi5_full_config import (
    apply_pi5_algorithm_features,
    build_pi5_full_config,
    resolve_pi5_algorithm_features,
)
from deploy.raspberry_pi5_scout.run_silent_full import (
    _install_mapless_tracker, _json_value, _load_adapter_config, _timing,
)
from mobile_robot_mppi.core.types import (
    ControlCommand,
    Pose2D,
    RobotObservation,
    SafetyDecision,
    Twist2D,
)
from mobile_robot_mppi.real_robot import (
    EncounterControlAuthority,
    EncounterControlConfig,
    EncounterModeConfig,
    EncounterModeManager,
    EncounterReferenceAuthority,
    ForwardPassageConfig,
    ForwardPassageController,
    LivoxPointCloudFrame,
    LivoxScanAdapter,
)
from mobile_robot_mppi.real_robot.remote_transport import RemoteDeploymentClient
from mobile_robot_mppi.planning.mppi import MppiController
from mobile_robot_mppi.runtime.factories import make_components


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AsyncJsonlWriter:
    """Serialize and flush diagnostic rows outside the control loop."""

    _STOP = object()

    def __init__(self, path, max_queue_size=64):
        self.path = Path(path)
        self._stream = self.path.open("w", encoding="utf-8")
        self._queue = queue.Queue(maxsize=int(max_queue_size))
        self._error = None
        self._written_rows = 0
        self._max_queue_depth = 0
        self._thread = threading.Thread(
            target=self._run,
            name="real-robot-diagnostic-writer",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        try:
            while True:
                value = self._queue.get()
                try:
                    if value is self._STOP:
                        return
                    self._stream.write(
                        json.dumps(_json_value(value), sort_keys=True) + "\n"
                    )
                    self._stream.flush()
                    self._written_rows += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # propagate through write/close
            self._error = exc

    def write(self, row):
        while True:
            if self._error is not None:
                raise RuntimeError("diagnostic writer failed") from self._error
            try:
                self._queue.put(row, timeout=0.10)
                self._max_queue_depth = max(
                    self._max_queue_depth,
                    self._queue.qsize(),
                )
                return
            except queue.Full:
                # Give the worker a chance to drain without doing JSON work in
                # the real-time control thread.
                continue

    def close(self):
        if self._thread.is_alive():
            self._queue.put(self._STOP)
            self._thread.join()
        try:
            self._stream.flush()
        finally:
            self._stream.close()
        if self._error is not None:
            raise RuntimeError("diagnostic writer failed") from self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class _CommandHeartbeat:
    """Refresh the last safe command while the PC computes the next one.

    A full forecast/control cycle can occasionally exceed the PC period.  If
    no command crosses the TCP link during that gap, the Pi watchdog zeros the
    chassis and the next plan restarts it, which is the observed staircase
    motion.  This helper repeats the latest command at a fixed rate for a
    short lease; after the lease expires it sends an immediate translation
    stop, so a genuinely stalled planner still fails closed.
    """

    def __init__(self, remote, *, period_s=0.05, lease_s=0.28):
        self.remote = remote
        self.period_s = float(period_s)
        self.lease_s = float(lease_s)
        if not 0.01 <= self.period_s <= 0.20:
            raise ValueError("command heartbeat period must be in [0.01, 0.20]")
        if not self.period_s < self.lease_s < 0.35:
            raise ValueError(
                "command heartbeat lease must be above its period and below "
                "the Pi watchdog timeout"
            )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._sequence = 0
        self._v = 0.0
        self._omega = 0.0
        self._arm = False
        self._translation_stop = False
        self._all_stop = False
        self._last_publish = time.monotonic()
        self.refresh_count = 0
        self.lease_expiry_count = 0

    def start(self):
        if self._thread is not None:
            raise RuntimeError("command heartbeat already started")
        self._thread = threading.Thread(
            target=self._run, name="remote-command-heartbeat", daemon=True
        )
        self._thread.start()

    def publish(
        self,
        v_mps,
        omega_radps,
        *,
        arm,
        immediate_translation_stop=False,
        immediate_all_stop=False,
    ):
        with self._lock:
            self._v = float(v_mps)
            self._omega = float(omega_radps)
            self._arm = bool(arm)
            self._translation_stop = bool(immediate_translation_stop)
            self._all_stop = bool(immediate_all_stop)
            self._last_publish = time.monotonic()
            self._sequence += 1
            sequence = self._sequence
            values = (
                self._v,
                self._omega,
                self._arm,
                self._translation_stop,
                self._all_stop,
            )
            # Keep sequence allocation and wire transmission atomic with
            # respect to the heartbeat thread; otherwise a background tick
            # could allocate N+1 and arrive before this N packet.
            self.remote.send_command(
                sequence,
                values[0],
                values[1],
                arm=values[2],
                immediate_translation_stop=values[3],
                immediate_all_stop=values[4],
            )
        return sequence

    def _run(self):
        next_tick = time.monotonic() + self.period_s
        while not self._stop.wait(max(0.0, next_tick - time.monotonic())):
            now = time.monotonic()
            with self._lock:
                age = now - self._last_publish
                self._sequence += 1
                sequence = self._sequence
                if age <= self.lease_s:
                    v = self._v
                    omega = self._omega
                    arm = self._arm
                    translation_stop = self._translation_stop
                    all_stop = self._all_stop
                    self.refresh_count += 1
                else:
                    # Keep the session armed but never keep stale translation
                    # beyond the bounded lease.
                    v = 0.0
                    omega = self._omega
                    arm = self._arm
                    translation_stop = True
                    all_stop = False
                    self.lease_expiry_count += 1
                try:
                    self.remote.send_command(
                        sequence,
                        v,
                        omega,
                        arm=arm,
                        immediate_translation_stop=translation_stop,
                        immediate_all_stop=all_stop,
                    )
                except (OSError, ConnectionError):
                    return
            next_tick += self.period_s

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


_PLANNER_DIAGNOSTIC_PREFIXES = (
    "probabilistic_obstacle_",
    "dynamic_obstacle_tracker_",
    "known_static_map_",
    "paper_guided_",
    "paper_gaussian_",
    "supervised_",
    "reliability_",
    "residual_",
    "optimizer_",
    "boundary_",
    "terminal_",
    "real_robot_forward_passage_",
    "physical_tracker_",
    "physical_goal_",
    "encounter_control_",
    # build_pi5_full_config leaves profile_components enabled, so the planner
    # already computes a per-stage cost breakdown on every solve.  Without
    # this prefix every profile_* field was dropped before reaching the log,
    # which left armed-cycle latency unattributable after the fact.
    "profile_",
)

# Retained diagnostics that carry no shared prefix.
_RETAINED_PLANNER_DIAGNOSTICS = ("compute_ms",)

_SEPARATELY_LOGGED_PLANNER_DIAGNOSTICS = {
    "dynamic_obstacle_tracker_trace",
    "probabilistic_obstacle_forecast_trace",
}


# These are translation-authorising decisions made by the safety arbiter
# itself.  A fixed start-to-goal line must not subsequently veto the lateral
# detour that the dynamic-obstacle stack has just certified.
_DYNAMIC_PATH_AUTHORITY_REASONS = frozenset({
    "dynamic_active_escape",
    "dynamic_corridor_escape",
    "dynamic_hard_stop_escape",
    "dynamic_hard_stop_side_rear_release",
    "dynamic_recovery_advance",
    "dynamic_recovery_align_creep",
    "rear_pass_through",
    # A slowdown is already the final scan/forecast-vetted positive command.
    # Reapplying the fixed start-to-goal heading stop turned isolated slowdown
    # cycles into visible zero-speed pulses between two valid escape commands.
    "temporal_slowdown",
})

_UNCONDITIONAL_TRANSLATION_STOP_REASONS = frozenset({
    "near_body_hard_stop",
    "temporal_collision_risk",
})

_GOAL_REJOIN_HEADING_RAD = 0.70
_GOAL_REJOIN_LARGE_HEADING_RAD = 1.10
_GOAL_REJOIN_SPEED_MPS = 0.30
_GOAL_REJOIN_LARGE_HEADING_SPEED_MPS = 0.20
_GOAL_REJOIN_LARGE_CROSS_TRACK_M = 0.90
_GOAL_REJOIN_LARGE_CROSS_TRACK_SPEED_MPS = 0.15
_GOAL_REJOIN_TURN_GAIN = 1.50
_GOAL_REJOIN_RISK_CEILING = 0.08
_GOAL_REJOIN_RISK_MASS_CEILING = 1.50
_GOAL_REJOIN_RELEASE_HEADING_RAD = 0.25
_GOAL_REJOIN_RELEASE_CROSS_TRACK_M = 0.65
_GOAL_REJOIN_RELEASE_STEPS = 3
_GOAL_REJOIN_REAR_HEMISPHERE_RAD = 0.5 * math.pi
_GOAL_REJOIN_REVERSE_SPEED_MPS = 0.20

# While a just-avoided dynamic hazard is still live, return toward the goal
# as a full-speed arc.  Bounding only the opposite steering avoids both the
# old +0.6/-0.6 reversal and the later zero-yaw commit that drove parallel to
# the goal line after the vehicle had already turned nearly 90 degrees.
_DYNAMIC_PASSAGE_REJOIN_OMEGA_RADPS = 0.30

# A fresh arbiter escape is always authoritative.  Brief forecast/scan
# fragmentation must not immediately reverse the selected passage side.
_DYNAMIC_PASSAGE_HAZARD_HOLD_STEPS = 6

# A fragmented safety reason may preserve a planner-vetted reverse exit for a
# few cycles, but it must not turn into the 33-frame retreat recorded in
# 20260805_003752.  The arbiter uses the same four-cycle allowance before its
# one retry, so this downstream guard closes the identical temporal gap.
_DYNAMIC_PASSAGE_MAX_HAZARD_REVERSE_STEPS = 4

# Mid-360 fragmentation can drop a rear-only near-body cluster for one or two
# cycles.  Preserve only forward authority across that brief ``front_clear``
# gap; any explicit front/temporal stop still preempts immediately.
_REAR_PASS_THROUGH_HOLD_STEPS = 3


def _dynamic_hazard_sector(hazard_active, safety_diagnostics):
    """Split a live dynamic hazard into forward and explicit rear-only state."""

    hazard_active = bool(hazard_active)
    if not hazard_active:
        return False, False
    diagnostics = dict(safety_diagnostics or {})
    bearings = []
    obstacle_bearing = diagnostics.get("dynamic_obstacle_bearing_rad")
    try:
        obstacle_bearing = float(obstacle_bearing)
    except (TypeError, ValueError):
        obstacle_bearing = float("nan")
    if math.isfinite(obstacle_bearing):
        bearings.append(obstacle_bearing)

    temporal_bearing = diagnostics.get("temporal_scan_center_angle_rad")
    temporal_ttc = diagnostics.get("temporal_scan_ttc_s", float("inf"))
    try:
        temporal_bearing = float(temporal_bearing)
        temporal_ttc = float(temporal_ttc)
    except (TypeError, ValueError):
        temporal_bearing = float("nan")
        temporal_ttc = float("inf")
    if (
        diagnostics.get("temporal_scan_valid", False)
        and math.isfinite(temporal_bearing)
        and temporal_ttc <= 3.0
    ):
        bearings.append(temporal_bearing)

    if not bearings:
        return True, False
    half_angle = math.radians(100.0)
    forward_hazard = any(
        abs(math.atan2(math.sin(angle), math.cos(angle))) <= half_angle
        for angle in bearings
    )
    return bool(forward_hazard), bool(not forward_hazard)


def _physical_tracker_motion_context(
    plan_diagnostics, tracker_diagnostics, pose_yaw, planner_config
):
    """Fill a missing passage direction from the associated physical track.

    Scan TTC can activate avoidance one solve before MPPI exports its motion
    direction.  The ego-compensated physical tracker already has that causal
    velocity; dropping it made the arbiter choose whichever side looked wider
    even during a clear crossing.  A strong opposite physical lateral motion
    also requests the arbiter's normal direction-refresh gate; weak conflicts
    keep the existing MPPI direction.
    """

    context = dict(plan_diagnostics or {})
    context["physical_tracker_motion_fallback_applied"] = False
    preferred = context.get(
        "probabilistic_obstacle_preferred_escape_heading_error_rad"
    )
    try:
        preferred = float(preferred)
    except (TypeError, ValueError):
        preferred = float("nan")
    planner_direction_available = bool(
        bool(context.get(
            "probabilistic_obstacle_forward_lateral_countermotion_applied",
            False,
        ))
        and math.isfinite(preferred)
        and abs(preferred) >= 0.08
    )
    try:
        planner_lateral = float(context.get(
            "probabilistic_obstacle_motion_lateral_body_mps", 0.0
        ))
        yaw = float(pose_yaw)
    except (TypeError, ValueError):
        planner_lateral = 0.0
        yaw = float("nan")
    stale_planner_side = bool(
        planner_direction_available
        and math.isfinite(yaw)
        and abs(planner_lateral) >= 0.35
        # Same signs mean the retained robot heading follows rather than
        # opposes the measured human lateral motion.
        and preferred * planner_lateral > 0.0
    )
    if stale_planner_side:
        lateral_weight = float(planner_config[
            "probabilistic_obstacle_forward_lateral_countermotion_weight"
        ])
        heading_error = -math.copysign(
            math.atan(lateral_weight), planner_lateral
        )
        heading = yaw + heading_error
        context.update({
            "probabilistic_obstacle_preferred_escape_direction_x": (
                math.cos(heading)
            ),
            "probabilistic_obstacle_preferred_escape_direction_y": (
                math.sin(heading)
            ),
            "probabilistic_obstacle_preferred_escape_heading_error_rad": (
                heading_error
            ),
            "probabilistic_obstacle_escape_direction_refreshed": True,
            "probabilistic_obstacle_escape_direction_source": (
                "physical_strong_lateral_stale_direction_refresh"
            ),
            "physical_tracker_motion_fallback_applied": True,
            "physical_tracker_motion_refresh_requested": True,
        })
        return context

    tracker = dict(tracker_diagnostics or {})
    dynamic_indices = {
        int(value)
        for value in tracker.get("mapless_dynamic_track_indices", ())
    }
    nearest_index = tracker.get("nearest_track_index")
    try:
        nearest_index = int(nearest_index)
    except (TypeError, ValueError):
        nearest_index = None
    if (
        not bool(tracker.get("associated", False))
        or not bool(tracker.get("forecast_valid", False))
        or not bool(tracker.get("motion_confirmed", False))
        or int(tracker.get("selected_support_beams", 0) or 0) < 3
        or nearest_index is None
        or nearest_index not in dynamic_indices
    ):
        return context

    try:
        velocity = np.asarray((
            float(tracker["measurement_velocity_x_mps"]),
            float(tracker["measurement_velocity_y_mps"]),
        ), dtype=np.float64)
        yaw = float(pose_yaw)
    except (KeyError, TypeError, ValueError):
        return context
    if not np.isfinite(velocity).all() or not math.isfinite(yaw):
        return context

    direction, longitudinal, lateral, fraction = (
        MppiController._forward_lateral_countermotion_direction(
            velocity,
            yaw,
            float(planner_config[
                "probabilistic_obstacle_forward_lateral_countermotion_weight"
            ]),
            float(planner_config[
                "probabilistic_obstacle_forward_lateral_minimum_speed_mps"
            ]),
            float(planner_config[
                "probabilistic_obstacle_forward_lateral_minimum_fraction"
            ]),
        )
    )
    context.update({
        "probabilistic_obstacle_motion_longitudinal_body_mps": float(
            longitudinal
        ),
        "probabilistic_obstacle_motion_lateral_body_mps": float(lateral),
        "probabilistic_obstacle_motion_lateral_fraction": float(fraction),
    })
    if direction is None:
        return context

    preferred_heading = float(np.arctan2(direction[1], direction[0]))
    heading_error = float(np.arctan2(
        np.sin(preferred_heading - yaw),
        np.cos(preferred_heading - yaw),
    ))
    if planner_direction_available:
        same_side = bool(preferred * heading_error > 0.0)
        strong_reversal = bool(abs(float(lateral)) >= 0.35)
        if same_side or not strong_reversal:
            return context
    context.update({
        "probabilistic_obstacle_forward_lateral_countermotion_applied": True,
        "probabilistic_obstacle_preferred_escape_direction_x": float(
            direction[0]
        ),
        "probabilistic_obstacle_preferred_escape_direction_y": float(
            direction[1]
        ),
        "probabilistic_obstacle_preferred_escape_heading_error_rad": (
            heading_error
        ),
        "probabilistic_obstacle_escape_direction_source": (
            "physical_tracker_measurement_forward_lateral_countermotion"
        ),
        "physical_tracker_motion_fallback_applied": True,
        "physical_tracker_motion_refresh_requested": bool(
            planner_direction_available
        ),
        "physical_tracker_motion_fallback_track_index": nearest_index,
    })
    if planner_direction_available:
        context[
            "probabilistic_obstacle_escape_direction_refreshed"
        ] = True
    return context


def _physical_goal_context(context, pose_x, pose_y, pose_yaw, goal_x, goal_y):
    """Attach an odometry-derived goal bearing for final escape arbitration."""

    enriched = dict(context or {})
    dx = float(goal_x) - float(pose_x)
    dy = float(goal_y) - float(pose_y)
    distance = float(math.hypot(dx, dy))
    if distance <= 1.0e-9:
        bearing_error = 0.0
    else:
        bearing_error = math.atan2(
            math.sin(math.atan2(dy, dx) - float(pose_yaw)),
            math.cos(math.atan2(dy, dx) - float(pose_yaw)),
        )
    enriched.update({
        "physical_goal_bearing_error_rad": float(bearing_error),
        "physical_goal_distance_m": distance,
    })
    return enriched


def _planner_diagnostic_trace(diagnostics):
    """Keep decision-relevant Full Proposed diagnostics without changing it.

    The controller diagnostics contain large internal payloads in addition to
    the causal fields needed to audit real-robot behaviour.  The tracker and
    forecast payloads are logged separately, while this view retains every
    probability-risk, traversal, proposal, residual, and optimizer field.
    """
    values = dict(diagnostics or {})
    return {
        str(key): value
        for key, value in values.items()
        if key not in _SEPARATELY_LOGGED_PLANNER_DIAGNOSTICS
        and (
            str(key).startswith(_PLANNER_DIAGNOSTIC_PREFIXES)
            or str(key) in _RETAINED_PLANNER_DIAGNOSTICS
        )
    }


def _real_robot_diagnostic_payload(tracker, plan_diagnostics, safety_diagnostics):
    """Build the read-only diagnostic payload persisted for every cycle."""
    tracker_values = dict(tracker or {})
    planner_values = dict(plan_diagnostics or {})
    return {
        "schema_version": "pc_pi_full_proposed_diagnostics_v3",
        # Preserve the producer's complete per-track records.  These include
        # position, fitted velocity, support/extent, association quality,
        # static/dynamic/unknown classification, and the raw motion evidence.
        "tracker": tracker_values,
        # The controller already binds forecast entries to their source track
        # and records mixture means, covariance, weights, radius, and horizon.
        "forecasts": planner_values.get(
            "probabilistic_obstacle_forecast_trace", ()
        ),
        "planner": _planner_diagnostic_trace(planner_values),
        # This is the final post-planner safety boundary, including override,
        # hard-stop, recovery, and applied-command diagnostics.
        "safety": _safety_diagnostic_trace(safety_diagnostics),
    }


def _control_cause_chain(
    *,
    plan,
    planning_context,
    decision,
    path_guard,
    commanded_v,
    commanded_omega,
    goal_stop_triggered,
    status,
    heartbeat,
):
    """Persist the command ownership chain without logging large payloads.

    This is deliberately a causal summary, not a second controller.  It lets
    a replay answer which layer changed the command and whether the change was
    an emergency candidate, a hard safety stop, a path guard, the goal stop,
    or merely the nominal MPPI output.
    """
    planner = dict(getattr(plan, "diagnostics", {}) or {})
    safety = dict(getattr(decision, "diagnostics", {}) or {})
    guard = dict(path_guard or {})
    fallback_kind = str(
        planning_context.get("probabilistic_obstacle_active_fallback_kind", "none")
    )
    emergency_selected = bool(
        planning_context.get(
            "probabilistic_obstacle_emergency_candidate_selected", False
        )
    )
    if goal_stop_triggered:
        dominant = "goal_stop"
    elif str(decision.reason) in _UNCONDITIONAL_TRANSLATION_STOP_REASONS:
        dominant = str(decision.reason)
    elif guard.get("active", False):
        dominant = str(guard.get("reason", "path_guard"))
    elif emergency_selected:
        dominant = "probabilistic_emergency_candidate"
    elif bool(decision.overridden):
        dominant = str(decision.reason)
    else:
        dominant = "nominal_mppi"
    if hasattr(status, "target_v_mps"):
        heartbeat_status = {
            "gateway_target": [
                float(status.target_v_mps), float(status.target_omega_radps)
            ],
            "gateway_applied": [
                float(status.applied_v_mps), float(status.applied_omega_radps)
            ],
        }
    else:
        heartbeat_status = dict(status or {})
    return {
        "dominant_cause": dominant,
        "planner_proposed": [
            float(plan.proposed_control.v),
            float(plan.proposed_control.omega),
        ],
        "planner_fallback_kind": fallback_kind,
        "planner_emergency_candidate_selected": emergency_selected,
        "planner_emergency_candidate_index": int(
            planning_context.get(
                "probabilistic_obstacle_active_fallback_index", -1
            )
            or -1
        ),
        "planner_optimizer_best_first": [
            float(planning_context.get("optimizer_best_first_v", 0.0) or 0.0),
            float(planning_context.get("optimizer_best_first_omega", 0.0) or 0.0),
        ],
        "planner_optimizer_selected_first": [
            float(planning_context.get("optimizer_selected_first_v", 0.0) or 0.0),
            float(planning_context.get("optimizer_selected_first_omega", 0.0) or 0.0),
        ],
        "planner_optimizer_cost_gap": float(
            planning_context.get("optimizer_selected_cost_gap", 0.0) or 0.0
        ),
        "safety_reason": str(decision.reason),
        "safety_overridden": bool(decision.overridden),
        "safety_executed": [
            float(decision.executed_control.v),
            float(decision.executed_control.omega),
        ],
        "path_guard_active": bool(guard.get("active", False)),
        "path_guard_reason": str(guard.get("reason", "none")),
        "path_guard_output": [float(commanded_v), float(commanded_omega)],
        "goal_stop_triggered": bool(goal_stop_triggered),
        "heartbeat_target": [
            float(heartbeat_status.get("gateway_target", [0.0, 0.0])[0]),
            float(heartbeat_status.get("gateway_target", [0.0, 0.0])[1]),
        ],
        "pi_applied": [
            float(heartbeat_status.get("gateway_applied", [0.0, 0.0])[0]),
            float(heartbeat_status.get("gateway_applied", [0.0, 0.0])[1]),
        ],
        "heartbeat_refresh_count": int(getattr(heartbeat, "refresh_count", 0)),
        "heartbeat_lease_expiry_count": int(
            getattr(heartbeat, "lease_expiry_count", 0)
        ),
    }


def _safety_diagnostic_trace(diagnostics):
    """Retain safety decisions without duplicating hundreds of scan points."""
    values = dict(diagnostics or {})
    point_keys = tuple(
        key
        for key, value in values.items()
        if (
            key == "raw_points_base" or str(key).endswith("_points")
        ) and isinstance(value, (tuple, list))
    )
    for key in point_keys:
        points = tuple(values.pop(key, ()) or ())
        stem = str(key[:-1]) if str(key).endswith("s") else str(key)
        values[f"{stem}_count"] = len(points)
        ranges = [
            float(item["range"])
            for item in points
            if isinstance(item, dict) and item.get("range") is not None
        ]
        values[f"{stem}_minimum_range_m"] = min(ranges) if ranges else None
    return values


def _local_obstacle_diagnostics(obstacles, pose_x, pose_y, robot_radius):
    """Summarise the exact local scan geometry supplied to the planner."""
    values = tuple(obstacles or ())
    if not values:
        return {
            "count": 0,
            "minimum_center_range_m": None,
            "minimum_footprint_clearance_m": None,
        }
    center_ranges = []
    footprint_clearances = []
    for obstacle in values:
        ox, oy = float(obstacle[0]), float(obstacle[1])
        radius = float(obstacle[2]) if len(obstacle) >= 3 else 0.08
        center_range = float(np.hypot(
            ox - float(pose_x), oy - float(pose_y)
        ))
        center_ranges.append(center_range)
        footprint_clearances.append(
            center_range - radius - float(robot_radius)
        )
    return {
        "count": len(values),
        "minimum_center_range_m": min(center_ranges),
        "minimum_footprint_clearance_m": min(footprint_clearances),
    }


def _scout_fault_labels(fault):
    """Decode SCOUT MINI chassis-status byte 5 per manual table 3.2."""
    labels = (
        "battery_undervoltage_fault",
        "battery_undervoltage_warning",
        "remote_controller_link_lost",
        "motor_1_communication_fault",
        "motor_2_communication_fault",
        "motor_3_communication_fault",
        "motor_4_communication_fault",
        "reserved_fault_bit_7",
    )
    value = int(fault or 0) & 0xFF
    return tuple(label for bit, label in enumerate(labels) if value & (1 << bit))


def _synthetic_livox_frame(adapter, base_points, timestamp_ns):
    """Return a Livox frame whose base-frame projection is ``base_points``.

    ``LivoxScanAdapter`` maps lidar points to base as ``points @ R.T + t``, so
    the exact inverse is applied here.  Warm-start geometry is specified in the
    frame the planner reasons about while still travelling through the same
    calibrated extrinsic, filtering and beam projection as the live run.
    """
    transform = np.asarray(
        adapter.config.lidar_to_base, dtype=np.float64
    ).reshape(4, 4)
    base = np.asarray(base_points, dtype=np.float64).reshape(-1, 3)
    lidar = (base - transform[:3, 3]) @ transform[:3, :3]
    count = int(lidar.shape[0])
    return LivoxPointCloudFrame(
        timestamp_ns=int(timestamp_ns),
        points=lidar.astype(np.float32),
        reflectivity=np.full(count, 120, dtype=np.uint8),
        tags=np.zeros(count, dtype=np.uint8),
    )


def _deskew_livox_frame(frame, adapter, v_mps, omega_radps,
                        maximum_age_s=0.25):
    """Express packet-timestamped points in the latest base pose.

    Raw geometry remains untouched for the fail-safe scan guard.  This copy is
    used only by the dynamic tracker, whose inter-frame motion estimate would
    otherwise confuse ego motion during each accumulated Livox frame with
    obstacle motion.
    """
    timestamps = frame.point_timestamps_ns
    diagnostics = {
        "applied": False,
        "point_count": int(frame.points.shape[0]),
        "maximum_age_s": 0.0,
        "v_mps": float(v_mps),
        "omega_radps": float(omega_radps),
    }
    if timestamps is None or frame.points.shape[0] == 0:
        diagnostics["reason"] = "missing_point_timestamps"
        return frame, diagnostics

    raw_ages = np.maximum(
        0.0,
        (int(frame.timestamp_ns) - np.asarray(timestamps, dtype=np.int64))
        * 1.0e-9,
    )
    ages = np.minimum(raw_ages, float(maximum_age_s))
    diagnostics["raw_maximum_age_s"] = float(np.max(raw_ages, initial=0.0))
    diagnostics["clipped_point_count"] = int(
        np.sum(raw_ages > float(maximum_age_s))
    )
    diagnostics["maximum_age_s"] = float(np.max(ages, initial=0.0))
    transform = np.asarray(
        adapter.config.lidar_to_base, dtype=np.float64
    ).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    base = np.asarray(frame.points, dtype=np.float64) @ rotation.T + translation

    theta = float(omega_radps) * ages
    if abs(float(omega_radps)) < 1.0e-8:
        chassis_dx = float(v_mps) * ages
        chassis_dy = np.zeros_like(ages)
    else:
        radius = float(v_mps) / float(omega_radps)
        chassis_dx = radius * np.sin(theta)
        chassis_dy = radius * (1.0 - np.cos(theta))
    shifted_x = base[:, 0] - chassis_dx
    shifted_y = base[:, 1] - chassis_dy
    cosine = np.cos(theta)
    sine = np.sin(theta)
    latest_x = cosine * shifted_x + sine * shifted_y
    latest_y = -sine * shifted_x + cosine * shifted_y
    latest_base = base.copy()
    latest_base[:, 0] = latest_x
    latest_base[:, 1] = latest_y
    latest_lidar = (latest_base - translation) @ rotation
    diagnostics["applied"] = bool(np.any(ages > 0.0))
    diagnostics["reason"] = (
        "constant_twist_deskew" if diagnostics["applied"] else "single_timestamp"
    )
    return LivoxPointCloudFrame(
        timestamp_ns=frame.timestamp_ns,
        points=latest_lidar.astype(np.float32),
        reflectivity=frame.reflectivity,
        tags=frame.tags,
        point_timestamps_ns=np.full(
            frame.points.shape[0], frame.timestamp_ns, dtype=np.int64
        ),
    ), diagnostics


def _warm_start_base_points(obstacle_center_y=None):
    """Corridor walls plus an optional translating cluster, in the base frame.

    The walls sit outside the task corridor half-width and outside the self
    box, so they populate enough beams for the geometric guard, local obstacle
    extraction and clustering to execute their real paths without holding the
    robot inside a near-body stop for the whole warm start.
    """
    columns = []
    wall_x = np.arange(0.60, 6.00, 0.05)
    for wall_y in (-1.50, 1.50):
        for wall_z in (0.25, 0.60, 1.05):
            columns.append(np.stack([
                wall_x,
                np.full_like(wall_x, wall_y),
                np.full_like(wall_x, wall_z),
            ], axis=1))
    end_y = np.arange(-1.50, 1.50, 0.05)
    for wall_z in (0.25, 0.60, 1.05):
        columns.append(np.stack([
            np.full_like(end_y, 7.50),
            end_y,
            np.full_like(end_y, wall_z),
        ], axis=1))
    if obstacle_center_y is not None:
        # A 0.40 m diameter, coherently translating cluster: large enough to
        # satisfy the support-beam and compactness gates of both the default
        # and the human-leg tracker profile.
        angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        for obstacle_z in (0.25, 0.55, 0.85):
            columns.append(np.stack([
                2.40 + 0.20 * np.cos(angle),
                float(obstacle_center_y) + 0.20 * np.sin(angle),
                np.full_like(angle, obstacle_z),
            ], axis=1))
    return np.concatenate(columns, axis=0)


def _warm_start(adapter, controller, perception, safety, reference,
                forward_passage, seed, static_cycles, forecast_cycles,
                dt_s, require_forecast):
    """Pay every lazy first-use cost before the timed, armed loop begins.

    Two separate one-off costs were measured on every physical run: CUDA graph
    capture on the first solve, and a second capture plus allocation on the
    first solve that carries an obstacle forecast.  The residual rollout graph
    cache is keyed partly on control batch shape, so a forecast-present solve
    cannot reuse the no-forecast graph.  The second cost therefore landed
    mid-drive while armed and exceeded the Pi command watchdog.  Both shapes
    are exercised here, disarmed, on synthetic geometry, and before the gateway
    connection is opened so that no chassis command is reachable and the Pi
    session window is not consumed.
    """
    report = {
        "requested_static_cycles": int(static_cycles),
        "requested_forecast_cycles": int(forecast_cycles),
        "dt_s": float(dt_s),
        "synthetic_scan": True,
        "plan_ms": [],
        "forecast_counts": [],
        "residual_shield_pair_timings": [],
        "residual_shield_pair_profiles": [],
    }
    planner_profiles = []
    step_ns = int(round(1.0e9 * float(dt_s)))
    timestamp_ns = step_ns
    obstacle_start_y = -0.90
    obstacle_speed_mps = 0.55
    for index in range(int(static_cycles) + int(forecast_cycles)):
        if index < int(static_cycles):
            center_y = None
        else:
            center_y = (
                obstacle_start_y
                + obstacle_speed_mps * float(dt_s) * (index - int(static_cycles))
            )
        scan, _ = adapter.convert(_synthetic_livox_frame(
            adapter, _warm_start_base_points(center_y), timestamp_ns
        ))
        timestamp_ns += step_ns
        observation = RobotObservation(
            timestamp=scan.timestamp,
            pose=Pose2D(0.0, 0.0, 0.0),
            twist=Twist2D(0.0, 0.0),
            scan=scan,
            auxiliary={
                "real_robot": True,
                "simulator_truth_used": False,
                "warm_start_synthetic_scan": True,
            },
        )
        if center_y is not None:
            prime_corroboration = getattr(
                perception.dynamic_obstacle_tracker,
                "prime_dynamic_classification_temporal_corroboration",
                None,
            )
            if callable(prime_corroboration):
                # The synthetic lateral cluster exists only to allocate and
                # capture the forecast-present CUDA/Actor path before the Pi
                # connection opens.  Prime the real gate's bounded state for
                # this disarmed scan; perception.reset() below removes it
                # before any physical observation is accepted.
                prime_corroboration()
        perceived = perception.process(observation)
        stage = time.perf_counter()
        plan = controller.plan(perceived.observation, reference)
        torch.cuda.synchronize()
        report["plan_ms"].append(1000.0 * (time.perf_counter() - stage))
        planner_profiles.append({
            str(key): float(value)
            for key, value in dict(plan.diagnostics).items()
            if str(key).startswith("profile_")
            and isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(float(value))
        })
        report["residual_shield_pair_timings"].append(dict(
            getattr(controller, "_last_pair_timing", {})
        ))
        report["residual_shield_pair_profiles"].append(dict(
            getattr(controller, "_last_pair_profiles", {})
        ))
        plan = forward_passage.apply(plan)
        safety.arbitrate(
            plan.proposed_control, perceived.guard, plan.diagnostics
        )
        tracker = dict(perceived.diagnostics.get(
            "dynamic_obstacle_tracker", {}
        ))
        report["forecast_counts"].append(
            int(tracker.get("valid_forecast_count", 0) or 0)
        )
    report["forecast_cycles_observed"] = sum(
        1 for count in report["forecast_counts"] if count > 0
    )
    report["maximum_plan_ms"] = (
        max(report["plan_ms"]) if report["plan_ms"] else None
    )
    report["final_plan_ms"] = (
        report["plan_ms"][-1] if report["plan_ms"] else None
    )
    profile_keys = sorted({
        key
        for profile in planner_profiles
        for key in profile
    })
    forecast_indices = [
        index
        for index, count in enumerate(report["forecast_counts"])
        if count > 0
    ]
    report["forecast_profile_median_ms"] = {
        key: float(np.median([
            planner_profiles[index].get(key, 0.0)
            for index in forecast_indices
        ]))
        for key in profile_keys
    } if forecast_indices else {}
    clear_indices = [
        index
        for index, count in enumerate(report["forecast_counts"])
        if count <= 0 and index > 0
    ]
    if forecast_indices:
        forecast_times = np.asarray([
            report["plan_ms"][index] for index in forecast_indices
        ])
        report["forecast_plan_p50_ms"] = float(
            np.quantile(forecast_times, 0.50)
        )
        report["forecast_plan_p95_ms"] = float(
            np.quantile(forecast_times, 0.95)
        )
    if clear_indices:
        clear_times = np.asarray([
            report["plan_ms"][index] for index in clear_indices
        ])
        report["clear_plan_p50_ms"] = float(
            np.quantile(clear_times, 0.50)
        )
    if require_forecast and not report["forecast_cycles_observed"]:
        raise RuntimeError(
            "warm start never produced an obstacle forecast, so the "
            "probabilistic risk path would still pay its first-use cost while "
            "armed and moving; re-tune the warm-start cluster or pass "
            "--allow-cold-forecast-path to accept that risk explicitly"
        )
    # The synthetic scene must not leak into the physical episode.
    controller.reset(int(seed))
    perception.reset()
    safety.reset()
    forward_passage.reset()
    return report


def _goal_stop_requested(pose_x, pose_y, goal_x, goal_y, radius_m, enabled):
    distance_m = math.hypot(float(pose_x) - float(goal_x),
                            float(pose_y) - float(goal_y))
    return bool(enabled and distance_m <= float(radius_m)), distance_m


def _immediate_translation_stop_requested(
    safety_reason, safety_diagnostics, commanded_v
):
    """Keep emergency braking except for a rear-certified escape reverse."""

    diagnostics = dict(safety_diagnostics or {})
    hard_stop_reverse_authorized = bool(
        diagnostics.get(
            "dynamic_escape_hard_stop_reverse_authorized", False
        )
        and float(commanded_v) < 0.0
        and str(safety_reason) == "dynamic_hard_stop_escape"
    )
    return bool(
        (
            diagnostics.get("emergency_stop", False)
            and not hard_stop_reverse_authorized
        )
        or str(safety_reason) in _UNCONDITIONAL_TRANSLATION_STOP_REASONS
    )


def _wait_for_complete_chassis_status(remote, timeout_s=3.0):
    """Wait for the first CAN-backed status, not merely the TCP heartbeat."""

    deadline = time.monotonic() + float(timeout_s)
    latest = remote.wait_for_status(timeout_s=float(timeout_s))
    while time.monotonic() < deadline:
        latest = remote.status() or latest
        if (
            latest.battery_v is not None
            and latest.control_mode is not None
            and latest.fault is not None
        ):
            return latest
        time.sleep(0.01)
    raise RuntimeError(
        "Pi gateway did not receive a complete CAN chassis status before "
        f"preflight timeout: {latest!r}"
    )


def _wait_for_newer_chassis_status(remote, previous_received_monotonic,
                                   timeout_s=2.0):
    """Wait for a status newer than the sample that went stale.

    The gateway can remain healthy while a Windows scheduling/network hiccup
    delays delivery of a few 20 Hz frames.  Callers must stop translation
    before entering this wait; a genuinely dead receiver still fails closed
    after the bounded grace period.
    """

    deadline = time.monotonic() + float(timeout_s)
    previous = float(previous_received_monotonic)
    while time.monotonic() < deadline:
        if remote._error is not None:
            raise RuntimeError(
                "remote deployment receiver failed while recovering chassis "
                "status"
            ) from remote._error
        latest = remote.status()
        if (
            latest is not None
            and latest.received_monotonic > previous
        ):
            return latest
        time.sleep(0.01)
    raise TimeoutError(
        "Pi gateway status did not recover after a stale-status hold "
        f"({float(timeout_s):.2f}s, received={remote.status_packets_received}, "
        f"tcp_thread_alive={remote._thread.is_alive()})"
    )


def _path_deviation_guard(
    pose_x,
    pose_y,
    pose_yaw,
    goal_x,
    goal_y,
    proposed_v,
    corridor_half_width_m=0.90,
    lookahead_m=0.80,
):
    """Only reduce forward authority when the physical path diverges.

    This post-arbitration governor never changes steering or increases speed.
    It is therefore subordinate to the planner and safety arbiter while
    preventing a fast departure from the two-point physical reference.
    """
    goal = np.asarray((float(goal_x), float(goal_y)), dtype=np.float64)
    position = np.asarray((float(pose_x), float(pose_y)), dtype=np.float64)
    length = float(np.linalg.norm(goal))
    if length <= 1.0e-9:
        return 0.0, {
            "active": True,
            "reason": "degenerate_goal",
            "cross_track_error_m": float(np.linalg.norm(position)),
            "heading_error_rad": 0.0,
            "forward_speed_cap_mps": 0.0,
        }
    tangent = goal / length
    along = float(np.clip(np.dot(position, tangent), 0.0, length))
    closest = tangent * along
    cross_track = float(np.linalg.norm(position - closest))
    target_along = min(length, along + float(lookahead_m))
    target = tangent * target_along
    delta = target - position
    target_heading = (
        float(np.arctan2(delta[1], delta[0]))
        if float(np.linalg.norm(delta)) > 1.0e-9
        else float(np.arctan2(goal[1] - position[1], goal[0] - position[0]))
    )
    heading_error = float(np.arctan2(
        np.sin(target_heading - float(pose_yaw)),
        np.cos(target_heading - float(pose_yaw)),
    ))
    speed_cap = float("inf")
    reasons = []
    if abs(heading_error) > 0.70:
        speed_cap = min(speed_cap, 0.10)
        reasons.append("heading")
    if abs(heading_error) > 1.10:
        speed_cap = 0.0
        reasons.append("heading_stop")
    if cross_track > float(corridor_half_width_m):
        speed_cap = min(speed_cap, 0.05)
        reasons.append("corridor")
    if cross_track > float(corridor_half_width_m) + 0.30:
        speed_cap = 0.0
        reasons.append("corridor_stop")
    output_v = float(proposed_v)
    if output_v > 0.0 and np.isfinite(speed_cap):
        output_v = min(output_v, speed_cap)
    return output_v, {
        "active": bool(reasons),
        "reason": "+".join(reasons) if reasons else "clear",
        "cross_track_error_m": cross_track,
        "heading_error_rad": heading_error,
        "forward_speed_cap_mps": (
            None if not np.isfinite(speed_cap) else float(speed_cap)
        ),
        "input_v_mps": float(proposed_v),
        "output_v_mps": output_v,
    }


def _arbiter_steering_authoritative(safety_reason, safety_diagnostics):
    """Whether ScanGuard intentionally owns yaw in this exact cycle.

    ``dynamic_active_escape`` is also used when ScanGuard merely accepts a
    risk-vetted MPPI sample after its finite geometric turn has ended.  Treating
    the reason string itself as yaw authority allowed those later samples to
    revive the completed turn and drive away from the goal.  This contract
    exposes actual state-machine ownership instead of inferring it from a broad
    label.
    """

    reason = str(safety_reason)
    diagnostics = dict(safety_diagnostics or {})
    if reason == "rear_pass_through":
        # The arbiter deliberately requests straight forward separation.
        return True
    if reason in {
        "dynamic_corridor_escape",
        "dynamic_hard_stop_escape",
        "dynamic_hard_stop_side_rear_release",
        "dynamic_recovery_advance",
        "dynamic_recovery_align",
        "dynamic_recovery_align_creep",
    }:
        return True
    if reason == "temporal_slowdown":
        return bool(diagnostics.get(
            "dynamic_escape_temporal_preturn_applied", False
        ))
    if reason != "dynamic_active_escape":
        return False
    try:
        commit_remaining = int(diagnostics.get(
            "dynamic_escape_direction_commit_remaining", 0
        ) or 0)
        coast_remaining = int(diagnostics.get(
            "dynamic_escape_coast_remaining", 0
        ) or 0)
    except (TypeError, ValueError):
        commit_remaining = 0
        coast_remaining = 0
    reactive_owned = bool(
        diagnostics.get("dynamic_escape_reactive", False)
        and not diagnostics.get(
            "dynamic_escape_vetted_planner_control", False
        )
    )
    return bool(
        commit_remaining > 0
        or coast_remaining != 0
        or diagnostics.get("dynamic_escape_geometric_forward", False)
        or diagnostics.get("dynamic_escape_geometric_forward_coast", False)
        or diagnostics.get("dynamic_escape_geometric_temporal_override", False)
        or diagnostics.get(
            "dynamic_escape_hard_stop_transaction_active", False
        )
        or diagnostics.get("dynamic_escape_held", False)
        or diagnostics.get(
            "dynamic_escape_post_retry_reverse_hold_applied", False
        )
        or diagnostics.get(
            "dynamic_escape_post_retry_side_forward_applied", False
        )
        or reactive_owned
    )


class _DynamicPathGuardSupervisor:
    """Keep dynamic avoidance authoritative and make path recovery controllable.

    The two-point physical reference is useful when the scene is clear, but
    its heading/corridor stops conflict with a deliberate lateral pass.  The
    old governor set translation to zero while leaving turn-away steering
    untouched, making the heading error an absorbing state.  This supervisor
    instead uses a bounded goal-directed arc whenever safety allows forward
    motion and the heading has diverged.  No old command or timer is retained.
    """

    def __init__(self, *, single_dynamic_authority=False):
        # The legacy state below remains available only for recorded
        # counterfactuals.  Physical deployment enables single authority: the
        # MPPI planner owns clear-scene control and ScanGuard owns a dynamic
        # transaction.  This supervisor then observes but never synthesizes a
        # second speed/yaw command.
        self._single_dynamic_authority = bool(single_dynamic_authority)
        self._goal_rejoin_latched = False
        self._goal_behind_turn_sign = 0.0
        self._rear_only_goal_turn_sign = 0.0
        self._last_dynamic_turn_sign = 0.0
        self._hazard_reverse_steps = 0
        self._hazard_reverse_limit_latched = False
        self._hazard_hold_remaining = 0
        self._goal_rejoin_release_count = 0
        self._rear_pass_through_hold_remaining = 0
        self._dynamic_goal_rejoin_requested = False

    def reset(self):
        self._goal_rejoin_latched = False
        self._goal_behind_turn_sign = 0.0
        self._rear_only_goal_turn_sign = 0.0
        self._last_dynamic_turn_sign = 0.0
        self._hazard_reverse_steps = 0
        self._hazard_reverse_limit_latched = False
        self._hazard_hold_remaining = 0
        self._goal_rejoin_release_count = 0
        self._rear_pass_through_hold_remaining = 0
        self._dynamic_goal_rejoin_requested = False

    def apply(
        self,
        pose_x,
        pose_y,
        pose_yaw,
        goal_x,
        goal_y,
        proposed_v,
        safety_reason,
        emergency_stop=False,
        proposed_omega=0.0,
        maximum_omega_radps=0.6,
        selected_probability=0.0,
        selected_probability_mass=0.0,
        hazard_active=False,
        rear_only_hazard=False,
        reverse_escape_exhausted=False,
        arbiter_steering_authoritative=True,
        arbiter_rejoin_requested=False,
        dynamic_safety_arbitration_enabled=True,
    ):
        reason = str(safety_reason)
        if self._single_dynamic_authority:
            _, diagnostics = _path_deviation_guard(
                pose_x,
                pose_y,
                pose_yaw,
                goal_x,
                goal_y,
                proposed_v,
            )
            diagnostics = dict(diagnostics)
            unconditional_stop = bool(
                reason in _UNCONDITIONAL_TRANSLATION_STOP_REASONS
            )
            dynamic_owner = bool(
                dynamic_safety_arbitration_enabled
                and (
                    reason in _DYNAMIC_PATH_AUTHORITY_REASONS
                    or hazard_active
                )
            )
            output_v = 0.0 if unconditional_stop else float(proposed_v)
            diagnostics.update({
                "active": unconditional_stop,
                "reason": (
                    reason
                    if unconditional_stop
                    else "single_dynamic_authority_passthrough"
                ),
                "bypassed": bool(
                    not unconditional_stop
                    and diagnostics.get("active", False)
                ),
                "bypass_reason": (
                    None
                    if unconditional_stop
                    else "single_control_owner"
                ),
                "would_be_active": bool(diagnostics.get("active", False)),
                "would_be_reason": str(
                    diagnostics.get("reason", "clear")
                ),
                "dynamic_authority": dynamic_owner,
                "single_dynamic_authority": True,
                "single_control_owner": (
                    "scan_guard" if dynamic_owner else "mppi"
                ),
                "goal_rejoin_latched": False,
                "commanded_omega_override_radps": None,
                "input_v_mps": float(proposed_v),
                "output_v_mps": output_v,
            })
            return output_v, diagnostics
        unconditional_stop = reason in _UNCONDITIONAL_TRANSLATION_STOP_REASONS
        fresh_rear_pass_through = reason == "rear_pass_through"
        if fresh_rear_pass_through:
            self._rear_pass_through_hold_remaining = (
                _REAR_PASS_THROUGH_HOLD_STEPS
            )
        elif self._rear_pass_through_hold_remaining > 0:
            self._rear_pass_through_hold_remaining -= 1
        rear_only_hazard = bool(rear_only_hazard)
        if rear_only_hazard and not fresh_rear_pass_through:
            # Explicit rear-sector evidence ends the scan-gap bridge as soon
            # as the arbiter no longer requests rear pass-through itself.
            # Keeping this hold alive for two extra front-clear cycles replayed
            # stale avoidance yaw after the pedestrian had already crossed
            # behind the chassis (20260804_225246 cycles 64--65).
            self._rear_pass_through_hold_remaining = 0
        rear_pass_through_hold_active = bool(
            self._rear_pass_through_hold_remaining > 0
        )
        dynamic_authority = bool(
            reason in _DYNAMIC_PATH_AUTHORITY_REASONS
            or (
                rear_pass_through_hold_active
                and reason == "front_clear"
                and float(proposed_v) > 0.0
            )
        )
        fresh_hazard_active = bool(hazard_active and not rear_only_hazard)
        if rear_only_hazard:
            # Once every causal live bearing is behind the protected forward
            # sector, the old front-passage smoothing must not keep limiting
            # the goalward turn.  Rear-pass authority above still bridges scan
            # gaps and remains independently bounded.
            self._hazard_hold_remaining = 0
        elif fresh_hazard_active:
            self._hazard_hold_remaining = (
                _DYNAMIC_PASSAGE_HAZARD_HOLD_STEPS
            )
        elif self._hazard_hold_remaining > 0:
            self._hazard_hold_remaining -= 1
        hazard_active = bool(self._hazard_hold_remaining > 0)
        dynamic_authority = bool(
            dynamic_authority
            or (
                hazard_active
                and reason == "front_obstacle_slow"
                and float(proposed_v) > 0.0
            )
        )
        if not hazard_active:
            self._hazard_reverse_steps = 0
            self._hazard_reverse_limit_latched = False
        elif reverse_escape_exhausted:
            self._hazard_reverse_limit_latched = True
            self._hazard_reverse_steps = max(
                self._hazard_reverse_steps,
                _DYNAMIC_PASSAGE_MAX_HAZARD_REVERSE_STEPS,
            )
        if dynamic_authority:
            # Once an avoidance event has occurred, keep a deterministic route
            # return available for the rest of the episode. Releasing this at
            # a moderate heading error let the stochastic warm start turn away
            # again on the immediately following cycle (20260804_022501).
            self._goal_rejoin_latched = True
            self._goal_behind_turn_sign = 0.0
            if abs(float(proposed_omega)) > 1.0e-6:
                self._last_dynamic_turn_sign = float(
                    np.sign(float(proposed_omega))
                )
            if (
                reason == "dynamic_hard_stop_escape"
                or (
                    float(proposed_v) > 0.0
                    and not reverse_escape_exhausted
                )
            ):
                self._hazard_reverse_steps = 0
                self._hazard_reverse_limit_latched = False
            self._goal_rejoin_release_count = 0
            if arbiter_rejoin_requested:
                self._dynamic_goal_rejoin_requested = True
            elif (
                arbiter_steering_authoritative
                and reason not in {"rear_pass_through", "temporal_slowdown"}
            ):
                self._dynamic_goal_rejoin_requested = False
        guarded_v, diagnostics = _path_deviation_guard(
            pose_x,
            pose_y,
            pose_yaw,
            goal_x,
            goal_y,
            proposed_v,
        )
        diagnostics = dict(diagnostics)
        diagnostics.update({
            "fresh_hazard_active": fresh_hazard_active,
            "hazard_active": hazard_active,
            "rear_only_hazard": rear_only_hazard,
            "fresh_rear_pass_through": fresh_rear_pass_through,
            "rear_pass_through_hold_active": (
                rear_pass_through_hold_active
            ),
            "rear_pass_through_hold_remaining": int(
                self._rear_pass_through_hold_remaining
            ),
            "dynamic_passage_hazard_hold_remaining": int(
                self._hazard_hold_remaining
            ),
        })

        if unconditional_stop:
            diagnostics.update({
                "active": True,
                "reason": reason,
                "bypassed": False,
                "bypass_reason": None,
                "dynamic_authority": False,
                "goal_rejoin_latched": bool(self._goal_rejoin_latched),
                "commanded_omega_override_radps": None,
                "input_v_mps": float(proposed_v),
                "output_v_mps": 0.0,
            })
            return 0.0, diagnostics

        if dynamic_authority:
            would_be_active = bool(diagnostics.get("active", False))
            would_be_reason = str(diagnostics.get("reason", "clear"))
            # The safety arbiter has already applied the physical limits,
            # near-body guard and rear-direction guard.  Preserve its complete
            # risk-vetted command here.  Replacing every reverse with
            # (v=0.35, omega=0) erased both the planner's escape direction and
            # the bounded close-range reverse transaction.
            output_v = float(proposed_v)
            omega_override = None
            heading_error = float(diagnostics["heading_error_rad"])
            dynamic_goal_rejoin_active = bool(
                self._dynamic_goal_rejoin_requested
                and not arbiter_steering_authoritative
                and output_v > 0.0
                and abs(heading_error) > _GOAL_REJOIN_RELEASE_HEADING_RAD
            )
            if dynamic_goal_rejoin_active:
                # ScanGuard has explicitly completed/released its finite turn.
                # Preserve its translation authority, but do not let a later
                # risk-vetted MPPI sample resurrect yaw away from the goal.
                # The bounded counter-turn avoids a +0.6 -> -0.6 snap.
                omega_limit = min(
                    abs(float(maximum_omega_radps)),
                    _DYNAMIC_PASSAGE_REJOIN_OMEGA_RADPS,
                )
                omega_override = float(np.clip(
                    _GOAL_REJOIN_TURN_GAIN * heading_error,
                    -omega_limit,
                    omega_limit,
                ))
                if abs(heading_error) > _GOAL_REJOIN_HEADING_RAD:
                    output_v = min(output_v, _GOAL_REJOIN_SPEED_MPS)
            rear_only_goal_steer_active = False
            diagnostics.update({
                "active": bool(dynamic_goal_rejoin_active),
                "reason": (
                    "dynamic_goal_rejoin_authority"
                    if dynamic_goal_rejoin_active
                    else "dynamic_authority"
                ),
                "bypassed": would_be_active,
                "bypass_reason": reason,
                "would_be_active": would_be_active,
                "would_be_reason": would_be_reason,
                "dynamic_authority": True,
                "goal_rejoin_latched": True,
                "arbiter_steering_authoritative": bool(
                    arbiter_steering_authoritative
                ),
                "arbiter_rejoin_requested": bool(arbiter_rejoin_requested),
                "dynamic_goal_rejoin_requested": bool(
                    self._dynamic_goal_rejoin_requested
                ),
                "dynamic_goal_rejoin_active": dynamic_goal_rejoin_active,
                "rear_only_goal_steer_active": rear_only_goal_steer_active,
                "rear_only_goal_turn_sign": float(
                    self._rear_only_goal_turn_sign
                ),
                "commanded_omega_override_radps": omega_override,
                "input_v_mps": float(proposed_v),
                "output_v_mps": output_v,
            })
            return output_v, diagnostics

        selected_probability = float(selected_probability or 0.0)
        selected_probability_mass = float(selected_probability_mass or 0.0)
        risk_safe = bool(
            selected_probability <= _GOAL_REJOIN_RISK_CEILING
            and selected_probability_mass <= _GOAL_REJOIN_RISK_MASS_CEILING
        )
        rejoin_eligible = bool(
            not emergency_stop
            and float(proposed_v) > 0.0
            and risk_safe
        )
        heading_error = float(diagnostics["heading_error_rad"])
        cross_track = float(diagnostics["cross_track_error_m"])
        goal_behind = bool(
            abs(heading_error) > _GOAL_REJOIN_REAR_HEMISPHERE_RAD
        )
        if not goal_behind:
            self._goal_behind_turn_sign = 0.0
        release_ready = bool(
            self._goal_rejoin_latched
            and not hazard_active
            and rejoin_eligible
            and abs(heading_error) <= _GOAL_REJOIN_RELEASE_HEADING_RAD
            and cross_track <= _GOAL_REJOIN_RELEASE_CROSS_TRACK_M
        )
        if release_ready:
            self._goal_rejoin_release_count += 1
        else:
            self._goal_rejoin_release_count = 0
        released_to_planner = bool(
            self._goal_rejoin_latched
            and self._goal_rejoin_release_count >= _GOAL_REJOIN_RELEASE_STEPS
        )
        if released_to_planner:
            self._goal_rejoin_latched = False
            self._goal_behind_turn_sign = 0.0
            self._last_dynamic_turn_sign = 0.0
            self._hazard_reverse_steps = 0
            self._goal_rejoin_release_count = 0
            self._dynamic_goal_rejoin_requested = False
        # A geometric deviation by itself is not evidence that deployment
        # safety owns the manoeuvre.  The Full Proposed planner is expected to
        # turn away from the start-to-goal chord in both static and dynamic
        # scenes.  Latching here used to let this thin deployment supervisor
        # overwrite MPPI with a saturated goal-bearing turn even when no
        # dynamic event had ever occurred (025611 cycle 56 and 025700 for 40
        # cycles).  Only an explicit dynamic-authority decision above may arm
        # route rejoin.
        if not self._goal_rejoin_latched:
            would_be_active = bool(diagnostics.get("active", False))
            would_be_reason = str(diagnostics.get("reason", "clear"))
            diagnostics.update({
                "active": False,
                "reason": "planner_authority",
                "bypassed": would_be_active,
                "bypass_reason": "no_dynamic_event",
                "would_be_active": would_be_active,
                "would_be_reason": would_be_reason,
                "dynamic_authority": False,
                "goal_rejoin_latched": False,
                "goal_rejoin_released_to_planner": released_to_planner,
                "goal_rejoin_release_count": int(
                    self._goal_rejoin_release_count
                ),
                "commanded_omega_override_radps": None,
                "input_v_mps": float(proposed_v),
                "output_v_mps": float(proposed_v),
            })
            return float(proposed_v), diagnostics

        # A live forecast may legitimately select a reverse sample after the
        # explicit dynamic-arbiter reason has fragmented for one cycle.  Keep
        # that bounded command and its steering while the hazard is still
        # observed; once it clears, the branch below rejects stale reverse and
        # performs deterministic goal rejoin.
        post_escape_reverse = bool(
            self._goal_rejoin_latched
            and not emergency_stop
            and reason not in _DYNAMIC_PATH_AUTHORITY_REASONS
            and reason not in _UNCONDITIONAL_TRANSLATION_STOP_REASONS
            and float(proposed_v) < 0.0
        )
        if (
            post_escape_reverse
            and hazard_active
            and self._hazard_reverse_limit_latched
        ):
            if not goal_behind and risk_safe:
                # The finite reverse has brought the path target back into the
                # forward hemisphere and the instantaneous scan/arbiter did not
                # request a stop.  Rejoin deliberately instead of preserving a
                # now goal-negative reverse merely because a forecast remains.
                output_v = _GOAL_REJOIN_SPEED_MPS
                if abs(heading_error) > _GOAL_REJOIN_LARGE_HEADING_RAD:
                    output_v = min(
                        output_v, _GOAL_REJOIN_LARGE_HEADING_SPEED_MPS
                    )
                if cross_track > _GOAL_REJOIN_LARGE_CROSS_TRACK_M:
                    output_v = min(
                        output_v, _GOAL_REJOIN_LARGE_CROSS_TRACK_SPEED_MPS
                    )
                omega_override = float(np.clip(
                    _GOAL_REJOIN_TURN_GAIN * heading_error,
                    -abs(float(maximum_omega_radps)),
                    abs(float(maximum_omega_radps)),
                ))
                diagnostics.update({
                    "active": True,
                    "reason": "dynamic_hazard_goal_rejoin",
                    "bypassed": False,
                    "bypass_reason": None,
                    "would_be_active": bool(
                        diagnostics.get("active", False)
                    ),
                    "would_be_reason": str(
                        diagnostics.get("reason", "clear")
                    ),
                    "dynamic_authority": False,
                    "goal_rejoin_latched": True,
                    "hazard_active": True,
                    "commanded_omega_override_radps": omega_override,
                    "input_v_mps": float(proposed_v),
                    "output_v_mps": output_v,
                })
                return output_v, diagnostics
            # The live hazard is no longer safely behind the chassis, or the
            # bounded reverse allowance has been consumed.  Do not translate
            # either direction until a fresh arbiter decision or clear geometry
            # permits deterministic goal rejoin.
            diagnostics.update({
                "active": True,
                "reason": "dynamic_hazard_reverse_exhausted",
                "bypassed": False,
                "bypass_reason": None,
                "would_be_active": bool(diagnostics.get("active", False)),
                "would_be_reason": str(diagnostics.get("reason", "clear")),
                "dynamic_authority": False,
                "goal_rejoin_latched": True,
                "hazard_active": True,
                "dynamic_clearance_reverse_step": int(
                    self._hazard_reverse_steps
                ),
                "commanded_omega_override_radps": 0.0,
                "input_v_mps": float(proposed_v),
                "output_v_mps": 0.0,
            })
            return 0.0, diagnostics

        if post_escape_reverse and hazard_active:
            # Ordinary forecast fragmentation retains the historical vetted
            # reverse behaviour.  The strict four-step cap is latched only
            # after the arbiter has explicitly exhausted its one retry.
            self._hazard_reverse_steps = 0
            diagnostics.update({
                "active": False,
                "reason": "dynamic_hazard_reverse",
                "bypassed": False,
                "bypass_reason": None,
                "would_be_active": bool(diagnostics.get("active", False)),
                "would_be_reason": str(diagnostics.get("reason", "clear")),
                "dynamic_authority": False,
                "goal_rejoin_latched": True,
                "hazard_active": True,
                "dynamic_clearance_reverse_step": 0,
                "commanded_omega_override_radps": None,
                "input_v_mps": float(proposed_v),
                "output_v_mps": float(proposed_v),
            })
            return float(proposed_v), diagnostics

        if post_escape_reverse and risk_safe:
            if goal_behind:
                # The planner-originated reverse has already passed ScanGuard,
                # including the rear-direction and near-body checks.  Keep its
                # sign while the path target is behind the chassis instead of
                # replacing it with forward motion that has negative goal
                # progress.  Limit only the magnitude for a controlled rejoin.
                self._goal_behind_turn_sign = 0.0
                output_v = max(
                    float(proposed_v), -_GOAL_REJOIN_REVERSE_SPEED_MPS
                )
                goal_progress = output_v * math.cos(heading_error)
                diagnostics.update({
                    "active": bool(
                        output_v > float(proposed_v) + 1.0e-12
                    ),
                    "reason": "goal_behind_reverse_rejoin",
                    "bypassed": False,
                    "bypass_reason": None,
                    "would_be_active": bool(
                        diagnostics.get("active", False)
                    ),
                    "would_be_reason": str(
                        diagnostics.get("reason", "clear")
                    ),
                    "dynamic_authority": False,
                    "goal_rejoin_latched": True,
                    "goal_behind": True,
                    "hazard_active": False,
                    "goal_rejoin_selected_probability": selected_probability,
                    "goal_rejoin_selected_probability_mass": (
                        selected_probability_mass
                    ),
                    "goal_rejoin_reverse_speed_cap_mps": (
                        _GOAL_REJOIN_REVERSE_SPEED_MPS
                    ),
                    "goal_progress_projection_mps": goal_progress,
                    "commanded_omega_override_radps": None,
                    "input_v_mps": float(proposed_v),
                    "output_v_mps": output_v,
                })
                return output_v, diagnostics
            output_v = _GOAL_REJOIN_SPEED_MPS
            if abs(heading_error) > _GOAL_REJOIN_LARGE_HEADING_RAD:
                output_v = min(
                    output_v, _GOAL_REJOIN_LARGE_HEADING_SPEED_MPS
                )
            if cross_track > _GOAL_REJOIN_LARGE_CROSS_TRACK_M:
                output_v = min(
                    output_v, _GOAL_REJOIN_LARGE_CROSS_TRACK_SPEED_MPS
                )
            omega_override = float(np.clip(
                _GOAL_REJOIN_TURN_GAIN * heading_error,
                -abs(float(maximum_omega_radps)),
                abs(float(maximum_omega_radps)),
            ))
            diagnostics.update({
                "active": True,
                "reason": "stale_reverse_goal_rejoin",
                "bypassed": False,
                "bypass_reason": None,
                "would_be_active": bool(diagnostics.get("active", False)),
                "would_be_reason": str(diagnostics.get("reason", "clear")),
                "dynamic_authority": False,
                "goal_rejoin_latched": True,
                "hazard_active": False,
                "goal_rejoin_selected_probability": selected_probability,
                "goal_rejoin_selected_probability_mass": (
                    selected_probability_mass
                ),
                "commanded_omega_override_radps": omega_override,
                "input_v_mps": float(proposed_v),
                "output_v_mps": output_v,
            })
            return output_v, diagnostics

        if float(proposed_v) >= 0.0:
            self._hazard_reverse_steps = 0
        needs_rejoin = bool(
            rejoin_eligible
            and self._goal_rejoin_latched
        )
        if needs_rejoin:
            large_heading = bool(
                abs(heading_error) > _GOAL_REJOIN_LARGE_HEADING_RAD
            )
            speed_cap = float(proposed_v)
            if large_heading:
                speed_cap = min(
                    speed_cap, _GOAL_REJOIN_LARGE_HEADING_SPEED_MPS
                )
            elif abs(heading_error) > _GOAL_REJOIN_HEADING_RAD:
                speed_cap = min(speed_cap, _GOAL_REJOIN_SPEED_MPS)
            if cross_track > _GOAL_REJOIN_LARGE_CROSS_TRACK_M:
                speed_cap = min(
                    speed_cap, _GOAL_REJOIN_LARGE_CROSS_TRACK_SPEED_MPS
                )
            output_v = min(float(proposed_v), speed_cap)
            omega_override = float(np.clip(
                _GOAL_REJOIN_TURN_GAIN * heading_error,
                -abs(float(maximum_omega_radps)),
                abs(float(maximum_omega_radps)),
            ))
            goal_behind_turn = bool(goal_behind and not hazard_active)
            if goal_behind_turn:
                # A positive command cannot make progress when the path target
                # is in the rear hemisphere.  Do not synthesize reverse without
                # rear-clearance evidence; rotate at the configured yaw limit
                # until the target re-enters the forward hemisphere.  Lock the
                # initial side so the +/-pi wrap cannot flip steering each
                # cycle and recreate the observed in-place indecision.
                if self._goal_behind_turn_sign == 0.0:
                    self._goal_behind_turn_sign = float(
                        np.sign(heading_error)
                    )
                    if self._goal_behind_turn_sign == 0.0:
                        self._goal_behind_turn_sign = 1.0
                omega_override = float(
                    self._goal_behind_turn_sign
                    * abs(float(maximum_omega_radps))
                )
                output_v = 0.0
                diagnostics.update({
                    "active": True,
                    "reason": "goal_behind_turn_rejoin",
                    "bypassed": False,
                    "bypass_reason": None,
                    "would_be_active": bool(
                        diagnostics.get("active", False)
                    ),
                    "would_be_reason": str(
                        diagnostics.get("reason", "clear")
                    ),
                    "dynamic_authority": False,
                    "goal_rejoin_latched": True,
                    "goal_behind": True,
                    "goal_behind_turn_sign": self._goal_behind_turn_sign,
                    "goal_rejoin_selected_probability": (
                        selected_probability
                    ),
                    "goal_rejoin_selected_probability_mass": (
                        selected_probability_mass
                    ),
                    "goal_rejoin_speed_cap_mps": 0.0,
                    "goal_progress_projection_mps": 0.0,
                    "hazard_active": False,
                    "commanded_omega_override_radps": omega_override,
                    "input_v_mps": float(proposed_v),
                    "output_v_mps": output_v,
                })
                return output_v, diagnostics
            passage_stabilized = bool(
                hazard_active
                and self._last_dynamic_turn_sign != 0.0
                and omega_override * self._last_dynamic_turn_sign < 0.0
            )
            if passage_stabilized:
                # Keep full forward authority but bend back toward the goal.
                # A zero-yaw commit preserved a nearly perpendicular escape
                # heading in run 20260804_042943; an immediate full opposite
                # turn, on the other hand, caused side-to-side oscillation.
                omega_override = float(np.clip(
                    omega_override,
                    -_DYNAMIC_PASSAGE_REJOIN_OMEGA_RADPS,
                    _DYNAMIC_PASSAGE_REJOIN_OMEGA_RADPS,
                ))
                output_v = float(proposed_v)
            diagnostics.update({
                "active": bool(
                    output_v < float(proposed_v) - 1.0e-12
                    or abs(omega_override - float(proposed_omega)) > 1.0e-12
                ),
                "reason": (
                    "dynamic_passage_commit"
                    if passage_stabilized
                    else "goal_heading_rejoin"
                ),
                "bypassed": False,
                "bypass_reason": None,
                "would_be_active": bool(diagnostics.get("active", False)),
                "would_be_reason": str(diagnostics.get("reason", "clear")),
                "dynamic_authority": False,
                "goal_rejoin_latched": True,
                "goal_rejoin_selected_probability": selected_probability,
                "goal_rejoin_selected_probability_mass": (
                    selected_probability_mass
                ),
                "goal_rejoin_speed_cap_mps": speed_cap,
                "hazard_active": bool(hazard_active),
                "commanded_omega_override_radps": omega_override,
                "input_v_mps": float(proposed_v),
                "output_v_mps": output_v,
            })
            return output_v, diagnostics

        diagnostics.update({
            "bypassed": False,
            "bypass_reason": None,
            "would_be_active": bool(diagnostics.get("active", False)),
            "would_be_reason": str(diagnostics.get("reason", "clear")),
            "dynamic_authority": False,
            "goal_rejoin_latched": bool(self._goal_rejoin_latched),
            "commanded_omega_override_radps": None,
        })
        return float(guarded_v), diagnostics


def _direction_reversal_guard(
    proposed_v,
    proposed_omega,
    previous_commanded_v,
    zero_tolerance=1.0e-6,
):
    """Require one zero-translation command before changing drive sign.

    Steering authority is retained during the transition.  The next cycle
    must still present a fresh planner/safety decision, so an obsolete escape
    command is never held across a changing human trajectory.
    """
    proposed_v = float(proposed_v)
    proposed_omega = float(proposed_omega)
    previous_commanded_v = float(previous_commanded_v)
    sign_reversal = bool(
        abs(proposed_v) > float(zero_tolerance)
        and abs(previous_commanded_v) > float(zero_tolerance)
        and proposed_v * previous_commanded_v < 0.0
    )
    output_v = 0.0 if sign_reversal else proposed_v
    return output_v, proposed_omega, {
        "active": sign_reversal,
        "reason": "direction_reversal_zero_transition" if sign_reversal else "clear",
        "previous_commanded_v_mps": previous_commanded_v,
        "input_v_mps": proposed_v,
        "output_v_mps": output_v,
        "omega_preserved_radps": proposed_omega,
    }


def _physical_command_slew_guard(
    proposed_v,
    proposed_omega,
    previous_commanded_v,
    previous_commanded_omega,
    *,
    dt_s,
    maximum_v_rate_mps2,
    maximum_omega_rate_radps2,
    hazard_active=False,
    immediate_translation_stop=False,
    immediate_all_stop=False,
):
    """Restore the actuator-rate contract after all real-robot overrides.

    MPPI's regular samples obey ``action_space.rate_limits``, but predictive
    emergency candidates and post-planner supervisors intentionally bypass
    that internal clip.  Applying the same rate contract once, at the final
    physical boundary, prevents the observed full-scale command staircase.
    A collision/near-body stop retains immediate zero-translation authority.
    """

    values = np.asarray((proposed_v, proposed_omega), dtype=np.float64)
    previous = np.asarray(
        (previous_commanded_v, previous_commanded_omega), dtype=np.float64
    )
    if not np.isfinite(values).all() or not np.isfinite(previous).all():
        raise ValueError("physical slew commands must be finite")
    dt_s = float(dt_s)
    v_delta = float(maximum_v_rate_mps2) * dt_s
    omega_delta = float(maximum_omega_rate_radps2) * dt_s
    if dt_s <= 0.0 or v_delta <= 0.0 or omega_delta <= 0.0:
        raise ValueError("physical slew rates and timestep must be positive")

    if bool(immediate_all_stop):
        output = np.zeros(2, dtype=np.float64)
        reason = "immediate_all_stop"
    else:
        output = np.clip(
            values,
            previous - np.asarray((v_delta, omega_delta)),
            previous + np.asarray((v_delta, omega_delta)),
        )
        reason = "rate_limited"
        sign_reversal = bool(
            abs(previous[0]) > 1.0e-6
            and abs(values[0]) > 1.0e-6
            and previous[0] * values[0] < 0.0
        )
        if bool(immediate_translation_stop) or (
            bool(hazard_active) and sign_reversal
        ):
            # Brake before reversing away from a live hazard.  The following
            # cycle ramps smoothly from zero instead of jumping to -0.3 m/s.
            output[0] = 0.0
            reason = (
                "immediate_translation_stop"
                if immediate_translation_stop
                else "hazard_reversal_brake"
            )
    active = bool(not np.allclose(output, values, rtol=0.0, atol=1.0e-12))
    if not active:
        reason = "clear"
    return float(output[0]), float(output[1]), {
        "active": active,
        "reason": reason,
        "input_v_mps": float(values[0]),
        "input_omega_radps": float(values[1]),
        "previous_v_mps": float(previous[0]),
        "previous_omega_radps": float(previous[1]),
        "output_v_mps": float(output[0]),
        "output_omega_radps": float(output[1]),
        "maximum_delta_v_mps": v_delta,
        "maximum_delta_omega_radps": omega_delta,
        "hazard_active": bool(hazard_active),
        "immediate_translation_stop": bool(immediate_translation_stop),
        "immediate_all_stop": bool(immediate_all_stop),
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _select_runtime_controller(controller, disable_residual_learning=False):
    """Select the explicit nominal-only deployment fallback when requested."""

    if not disable_residual_learning:
        return controller
    nominal = getattr(controller, "nominal_controller", None)
    if nominal is None:
        raise TypeError(
            "residual-learning fallback requires a matched nominal controller"
        )
    return nominal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-host", default="10.141.194.219")
    parser.add_argument("--port", type=int, default=57720)
    parser.add_argument("--token", required=True)
    parser.add_argument("--weight-root", type=Path, required=True)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--goal-x", type=float, default=3.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--max-v-mps", type=float, required=True)
    parser.add_argument("--max-reverse-v-mps", type=float, required=True)
    parser.add_argument("--max-omega-radps", type=float, required=True)
    parser.add_argument("--period-s", type=float, default=0.10)
    parser.add_argument("--accumulation-s", type=float, default=0.06)
    # In-loop warm-up cycles only withhold arming and relax the status
    # staleness bound; they do not pay any first-use compute cost.
    parser.add_argument("--warmup-cycles", type=int, default=3)
    # Pre-loop warm start.  These cycles run on synthetic geometry before the
    # gateway connection exists, so no chassis command is reachable from them.
    parser.add_argument("--warm-start-static-cycles", type=int, default=3)
    parser.add_argument("--warm-start-forecast-cycles", type=int, default=14)
    parser.add_argument("--warm-start-dt-s", type=float, default=0.08)
    parser.add_argument("--allow-cold-forecast-path", action="store_true")
    parser.add_argument("--human-leg-mode", action="store_true")
    parser.add_argument("--full-proposed", action="store_true")
    parser.add_argument("--traditional-mppi", action="store_true")
    parser.add_argument("--enable-actor-guidance", action="store_true")
    parser.add_argument("--enable-hss-reliability", action="store_true")
    parser.add_argument(
        "--cpu-policy-inference",
        action="store_true",
        help=(
            "diagnostic rollback only: keep the Actor/HSS inference on CPU "
            "while the MPPI rollout remains on CUDA"
        ),
    )
    parser.add_argument("--enable-residual-learning", action="store_true")
    parser.add_argument(
        "--enable-change-aware-prediction", action="store_true"
    )
    parser.add_argument("--enable-probabilistic-risk", action="store_true")
    parser.add_argument("--enable-ar1-sampling", action="store_true")
    parser.add_argument(
        "--enable-forward-passage",
        "--real-robot-forward-passage",
        dest="enable_forward_passage",
        action="store_true",
    )
    parser.add_argument("--forward-passage-risk-ceiling", type=float, default=0.08)
    parser.add_argument("--forward-passage-mass-ceiling", type=float, default=1.50)
    parser.add_argument("--goal-stop-radius-m", type=float, default=0.0)
    parser.add_argument("--until-goal", action="store_true")
    parser.add_argument("--disable-residual-learning", action="store_true")
    parser.add_argument(
        "--disable-dynamic-safety-arbitration",
        action="store_true",
        help=(
            "bypass dynamic/rear-pass/slowdown arbitration; retain only "
            "action clipping and explicit emergency translation stops"
        ),
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.until_goal and not args.publish:
        parser.error("--until-goal requires --publish")
    if args.until_goal and args.goal_stop_radius_m <= 0.0:
        parser.error("--until-goal requires a positive goal-stop radius")
    try:
        algorithm_features = resolve_pi5_algorithm_features(
            full_proposed=args.full_proposed,
            traditional_mppi=args.traditional_mppi,
            enable_actor_guidance=args.enable_actor_guidance,
            enable_hss_reliability=args.enable_hss_reliability,
            enable_residual_learning=args.enable_residual_learning,
            enable_change_aware_prediction=(
                args.enable_change_aware_prediction
            ),
            enable_probabilistic_risk=args.enable_probabilistic_risk,
            enable_ar1_sampling=args.enable_ar1_sampling,
            enable_forward_passage=args.enable_forward_passage,
            disable_residual_learning=args.disable_residual_learning,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote MPPI deployment")

    config = build_pi5_full_config(
        args.weight_root,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        max_v_mps=args.max_v_mps,
        max_reverse_v_mps=args.max_reverse_v_mps,
        max_omega_radps=args.max_omega_radps,
    )
    config["perception"]["scan_guard"][
        "dynamic_safety_arbitration_enabled"
    ] = not bool(args.disable_dynamic_safety_arbitration)
    apply_pi5_algorithm_features(config, algorithm_features)
    # Keep the physical deployment identical to the simulation controller.
    # The later Encounter/YOLO-style semantic modes (frontal approach,
    # crossing, rear-pass and rejoin) are intentionally unavailable here;
    # their authority used to rewrite the simulation MPPI command after the
    # optimizer had already selected it.  The implementation remains in the
    # package for offline ablations and rollback, but this runner cannot arm
    # it.
    encounter_mode_config = EncounterModeConfig(
        enabled=False,
        shadow_only=True,
    )
    encounter_control_config = EncounterControlConfig(
        enabled=False,
        maximum_forward_speed_mps=float(args.max_v_mps),
        maximum_reverse_speed_mps=float(args.max_reverse_v_mps),
        maximum_omega_radps=float(args.max_omega_radps),
    )
    config["planner"].update({
        "device": "cuda",
        "probabilistic_obstacle_risk_cuda_enabled": bool(
            algorithm_features["probabilistic_risk"]
        ),
        "probabilistic_obstacle_risk_cuda_device": "cuda",
        # Real-robot emergency admission: ordinary probabilistic risk remains
        # in the MPPI cost, while direct takeover requires a short, reliable
        # causal scan certificate.  Critical near-body hard stops remain
        # authoritative below this planner boundary.
        "probabilistic_obstacle_emergency_candidate_trigger_ttc_s": 1.6,
        "probabilistic_obstacle_emergency_candidate_trigger_distance_m": 0.85,
        "probabilistic_obstacle_emergency_candidate_prefix_steps": 5,
        "probabilistic_obstacle_emergency_candidate_intent_hold_steps": 5,
        "probabilistic_obstacle_emergency_candidate_rearm_ttc_s": 2.0,
        "probabilistic_obstacle_emergency_candidate_rearm_clear_steps": 3,
        "probabilistic_obstacle_emergency_require_scan_quality": True,
        "probabilistic_obstacle_emergency_min_support_beams": 4,
        "probabilistic_obstacle_emergency_max_rejected_jump_fraction": 0.60,
        "probabilistic_obstacle_emergency_candidate_slew_enabled": True,
        # Keep the full candidate/feasibility evidence in the asynchronous
        # audit stream; no candidate arrays are serialized.
        "optimizer_diagnostics_enabled": True,
        "residual_device_rollout_enabled": bool(
            algorithm_features["residual_learning"]
        ),
        "residual_cuda_graph_enabled": bool(
            algorithm_features["residual_learning"]
        ),
        # Preserve the simulation-aligned FP64 integration contract for the
        # large CUDA MPPI rollout. The latency fix below targets the measured
        # sequential host/device transfers rather than changing arithmetic.
        "residual_rollout_dtype": "float64",
    })
    # The previous physical profile intentionally pinned the learned Actor and
    # HSS sidecar to CPU.  That left the dominant forecast-path inference
    # outside the CUDA planner and forced a mixed-device execution graph.
    # CUDA is a hard deployment requirement above, so make CUDA the default
    # for every learned inference component.  Keep an explicit diagnostic
    # rollback switch for CPU-vs-CUDA parity investigations; it is never the
    # normal armed profile.
    inference_device = "cpu" if args.cpu_policy_inference else "cuda"
    config["rl"]["device"] = inference_device
    sidecar_config = config["planner"].get("paper_rl_driven", {}).get(
        "reliability_sidecar", {}
    )
    if sidecar_config:
        sidecar_config["device"] = inference_device
    config["real_robot_deployment"]["learned_inference_device"] = (
        inference_device
    )
    config["real_robot_deployment"]["learned_inference_cuda_enabled"] = bool(
        inference_device == "cuda"
    )
    # The two planners contain substantial Python-side autoregressive Actor
    # work. Threads contend on the GIL and were 7-15% slower than executing the
    # exact same matched pair sequentially after the CUDA RK4 step fast path.
    config["planner"]["residual_safety_shield"][
        "parallel_planning_enabled"
    ] = False
    config["real_robot_deployment"].update({
        "contract": "pc_cuda_pi_gateway_mppi_ablation_v1",
        "computer": "windows_cuda_pc",
        "publish_enabled": bool(args.publish),
        "pi_gateway": args.pi_host,
        "human_leg_dynamic_filter": bool(
            args.human_leg_mode
            and algorithm_features["change_aware_prediction"]
        ),
        "dynamic_safety_arbitration_enabled": bool(
            not args.disable_dynamic_safety_arbitration
        ),
        "dynamic_safety_arbitration_profile": (
            "full"
            if not args.disable_dynamic_safety_arbitration
            else "hard_safety_only"
        ),
        "goal_stop_radius_m": float(args.goal_stop_radius_m),
        "residual_learning_enabled": bool(
            algorithm_features["residual_learning"]
        ),
        "runtime_algorithm_mode": (
            config["real_robot_deployment"]["algorithm_profile"]
        ),
        # The physical continuation transaction is deployment-only and is
        # intentionally excluded from the simulation-parity control path.
        "forward_passage_enabled": False,
        "forward_passage_policy": "disabled_simulation_parity",
        "forward_passage_risk_ceiling": float(
            args.forward_passage_risk_ceiling
        ),
        "forward_passage_mass_ceiling": float(
            args.forward_passage_mass_ceiling
        ),
        "dynamic_path_authority_reasons": sorted(
            _DYNAMIC_PATH_AUTHORITY_REASONS
        ),
        "goal_rejoin_heading_rad": (
            _GOAL_REJOIN_HEADING_RAD
        ),
        "goal_rejoin_speed_mps": _GOAL_REJOIN_SPEED_MPS,
        "goal_rejoin_large_heading_speed_mps": (
            _GOAL_REJOIN_LARGE_HEADING_SPEED_MPS
        ),
        "goal_rejoin_risk_ceiling": _GOAL_REJOIN_RISK_CEILING,
        "goal_rejoin_release_heading_rad": (
            _GOAL_REJOIN_RELEASE_HEADING_RAD
        ),
        "goal_rejoin_release_cross_track_m": (
            _GOAL_REJOIN_RELEASE_CROSS_TRACK_M
        ),
        "goal_rejoin_release_steps": _GOAL_REJOIN_RELEASE_STEPS,
        "direction_reversal_zero_transition_steps": 1,
        # Encounter recognition and authority are both disabled in the
        # simulation-parity deployment.  Keep explicit fields so old log
        # readers can verify that no semantic mode layer was armed.
        "encounter_mode_shadow_enabled": bool(encounter_mode_config.enabled),
        "encounter_mode_control_enabled": bool(
            encounter_control_config.enabled
        ),
        "encounter_policy": "disabled_simulation_parity",
        "encounter_mode_config": asdict(encounter_mode_config),
        "encounter_control_config": asdict(encounter_control_config),
    })
    # Parity with run_silent_full: the controller timestep and the tracker
    # forecast timestep must agree, otherwise every live forecast becomes
    # invalid the moment a dynamic track is first confirmed.  The publishing
    # runner previously omitted this invariant.
    controller_dt_s = float(config["experiment"]["control_dt"])
    action_rate_limits = tuple(
        float(value) for value in config["action_space"]["rate_limits"]
    )
    if len(action_rate_limits) != 2:
        raise RuntimeError("physical deployment requires v/omega rate limits")
    forecast_dt_s = float(
        config["perception"]["dynamic_obstacle_tracker"]["forecast_dt_s"]
    )
    forecast_auxiliary_key = str(
        config["perception"]["dynamic_obstacle_tracker"].get(
            "forecast_auxiliary_key", "probabilistic_obstacle_forecasts"
        )
    )
    if abs(controller_dt_s - forecast_dt_s) > 1.0e-12:
        raise RuntimeError(
            "controller and dynamic forecast timesteps disagree: %.9g vs %.9g"
            % (controller_dt_s, forecast_dt_s)
        )
    config_path = args.output / "config_resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "resolved_algorithm_profile": config["real_robot_deployment"][
            "algorithm_profile"
        ],
        "resolved_algorithm_features": algorithm_features,
    }, indent=2, sort_keys=True), flush=True)

    adapter = LivoxScanAdapter(_load_adapter_config(args.lidar_config))
    components = make_components(config, PROJECT_ROOT)
    # Residual ablation is applied to the resolved configuration before
    # construction, so disabled models and the duplicate shield planner are
    # not loaded at all.
    controller = components["controller"]
    perception = components["perception"]
    safety = components["safety"]
    reference = components["reference"]
    if algorithm_features["change_aware_prediction"]:
        # Human-leg handling and static filtering are required physical
        # perception gates; they do not grant any semantic encounter-control
        # authority to the final command.
        _install_mapless_tracker(
            perception, human_leg_mode=args.human_leg_mode
        )
    forward_passage_config = ForwardPassageConfig(
        enabled=False,
        maximum_probability=float(args.forward_passage_risk_ceiling),
        maximum_probability_mass=float(args.forward_passage_mass_ceiling),
    )
    forward_passage = ForwardPassageController(forward_passage_config)
    encounter_modes = EncounterModeManager(encounter_mode_config)
    encounter_reference = EncounterReferenceAuthority(
        encounter_control_config
    )
    encounter_control = EncounterControlAuthority(encounter_control_config)
    controller.reset(20260803)
    perception.reset()
    safety.reset()

    # Warm start before the gateway socket is opened.  This both removes the
    # first-use latency from the measured, armed loop and keeps the cost out of
    # the Pi's session window, which only starts once the PC is attached.
    warm_start = _warm_start(
        adapter, controller, perception, safety, reference, forward_passage,
        seed=20260803,
        static_cycles=args.warm_start_static_cycles,
        forecast_cycles=args.warm_start_forecast_cycles,
        dt_s=args.warm_start_dt_s,
        require_forecast=bool(
            algorithm_features["change_aware_prediction"]
            and not args.allow_cold_forecast_path
        ),
    )
    print(json.dumps({"warm_start": warm_start}, indent=2, sort_keys=True),
          flush=True)

    timings = {name: [] for name in (
        "receive", "scan", "perception", "encounter", "plan", "authority",
        "total"
    )}
    rows = []
    sequence = 0
    pose_x = pose_y = pose_yaw = 0.0
    last_pose_update = time.monotonic()
    started = time.monotonic()
    log_path = args.output / "cycles.jsonl"
    goal_stop_triggered = False
    last_goal_distance_m = math.hypot(args.goal_x, args.goal_y)
    livox_timeout_count = 0
    livox_timeout_streak = 0
    # A single empty accumulation window is a recoverable UDP/RPi hiccup.  It
    # must not tear down an otherwise healthy UntilGoal session; the previous
    # runs stopped after ~10 s because this exception escaped the control loop.
    maximum_livox_timeout_streak = 20
    chassis_status_recovery_count = 0
    chassis_status_max_age_s = 0.0
    chassis_status_recovery_timeout_s = 2.0
    path_guard_supervisor = _DynamicPathGuardSupervisor(
        single_dynamic_authority=True
    )
    initial_chassis_fault = None
    initial_chassis_fault_labels = ()
    diagnostic_writer_stats = {
        "written_rows": 0,
        "max_queue_depth": 0,
    }
    heartbeat_stats = {
        "period_s": 0.05,
        "lease_s": 0.28,
        "refresh_count": 0,
        "lease_expiry_count": 0,
    }
    try:
        with RemoteDeploymentClient(args.pi_host, args.port, args.token) as remote, \
                _AsyncJsonlWriter(log_path) as stream, \
                _CommandHeartbeat(remote) as heartbeat:
            status = _wait_for_complete_chassis_status(remote)
            initial_chassis_fault = status.fault
            initial_chassis_fault_labels = _scout_fault_labels(status.fault)
            if status.battery_v is None or status.battery_v < 22.0:
                raise RuntimeError("Pi gateway chassis preflight failed: %r" % (status,))
            if args.publish and status.fault != 0:
                raise RuntimeError(
                    "Pi gateway chassis preflight fault: "
                    f"value={status.fault}, labels={initial_chassis_fault_labels}"
                )
            if args.publish and status.control_mode == 3:
                raise RuntimeError(
                    "Pi gateway preflight found remote-control mode; "
                    "select standby before an armed session"
                )
            can_mode_confirmed = bool(status.control_mode == 1)
            next_cycle = time.monotonic()
            while time.monotonic() - started < args.duration_s:
                cycle_start = time.perf_counter()
                stage = time.perf_counter()
                motion_status = status
                try:
                    frame = remote.receive_livox_frame(args.accumulation_s)
                except TimeoutError as exc:
                    livox_timeout_count += 1
                    livox_timeout_streak += 1
                    # Do not continue translating without a fresh scan.  Send
                    # an immediate translation stop, keep the gateway armed,
                    # and retry the sensor stream on the next cycle.
                    sequence = heartbeat.publish(
                        0.0,
                        0.0,
                        arm=bool(args.publish and len(rows) >= args.warmup_cycles),
                        immediate_translation_stop=True,
                        immediate_all_stop=False,
                    )
                    if livox_timeout_streak >= maximum_livox_timeout_streak:
                        raise RuntimeError(
                            "Livox stream remained unavailable for "
                            f"{livox_timeout_streak} consecutive cycles "
                            f"({float(livox_timeout_streak) * args.period_s:.1f}s)"
                        ) from exc
                    time.sleep(min(float(args.period_s), 0.10))
                    continue
                livox_timeout_streak = 0
                timings["receive"].append(1000.0 * (time.perf_counter() - stage))
                stage = time.perf_counter()
                scan, scan_diagnostics = adapter.convert(frame)
                tracker_frame, deskew_diagnostics = _deskew_livox_frame(
                    frame,
                    adapter,
                    motion_status.v_mps,
                    motion_status.omega_radps,
                )
                tracker_scan, _ = adapter.convert(tracker_frame)
                timings["scan"].append(1000.0 * (time.perf_counter() - stage))
                status = remote.status()
                # CUDA Graph capture makes the first Full Proposed solve an
                # intentional one-off long cycle.  It occurs while disarmed,
                # so allow the receiver thread to catch up during warm-up only;
                # the armed/steady-state contract remains the strict 300 ms.
                status_limit_s = (
                    2.0
                    if len(rows) < args.warmup_cycles
                    else (0.50 if args.publish else 1.00)
                )
                status_age_s = (math.inf if status is None else
                                time.monotonic() - status.received_monotonic)
                chassis_status_max_age_s = max(
                    chassis_status_max_age_s,
                    float(status_age_s),
                )
                if status_age_s > status_limit_s:
                    stale_status_received_monotonic = (
                        -math.inf
                        if status is None
                        else status.received_monotonic
                    )
                    # Fail closed during the hold: the Pi watchdog also
                    # zeros the chassis if this command is not refreshed.
                    sequence = heartbeat.publish(
                        0.0,
                        0.0,
                        arm=bool(
                            args.publish and len(rows) >= args.warmup_cycles
                        ),
                        immediate_translation_stop=True,
                        immediate_all_stop=False,
                    )
                    status = _wait_for_newer_chassis_status(
                        remote,
                        stale_status_received_monotonic,
                        timeout_s=chassis_status_recovery_timeout_s,
                    )
                    chassis_status_recovery_count += 1
                    status_age_s = max(
                        0.0,
                        time.monotonic() - status.received_monotonic,
                    )
                    chassis_status_max_age_s = max(
                        chassis_status_max_age_s,
                        float(status_age_s),
                    )
                if args.publish and status.fault != 0:
                    raise RuntimeError(
                        "chassis fault changed: "
                        f"value={status.fault}, labels={_scout_fault_labels(status.fault)}"
                    )
                if args.publish:
                    if status.control_mode == 3:
                        raise RuntimeError(
                            "remote-control takeover detected; armed session revoked"
                        )
                    if can_mode_confirmed and status.control_mode != 1:
                        raise RuntimeError(
                            "CAN control mode was lost; armed session revoked"
                        )
                    if status.control_mode == 1:
                        can_mode_confirmed = True
                pose_now = time.monotonic()
                pose_dt = max(0.0, min(0.5, pose_now - last_pose_update))
                pose_x += status.v_mps * math.cos(pose_yaw) * pose_dt
                pose_y += status.v_mps * math.sin(pose_yaw) * pose_dt
                pose_yaw += status.omega_radps * pose_dt
                pose_yaw = math.atan2(math.sin(pose_yaw), math.cos(pose_yaw))
                last_pose_update = pose_now
                observation = RobotObservation(
                    timestamp=scan.timestamp,
                    pose=Pose2D(pose_x, pose_y, pose_yaw),
                    twist=Twist2D(status.v_mps, status.omega_radps),
                    scan=scan,
                    auxiliary={
                        "real_robot": True,
                        "simulator_truth_used": False,
                        "motion_compensated_tracker_scan": tracker_scan,
                    },
                )
                stage = time.perf_counter()
                perceived = perception.process(observation)
                timings["perception"].append(1000.0 * (time.perf_counter() - stage))
                tracker = dict(
                    perceived.diagnostics.get(
                        "dynamic_obstacle_tracker", {}
                    )
                )
                stage = time.perf_counter()
                encounter_diagnostics = encounter_modes.update(
                    timestamp_s=observation.timestamp,
                    pose=(pose_x, pose_y, pose_yaw),
                    goal=(args.goal_x, args.goal_y),
                    robot_speed_mps=status.v_mps,
                    tracker_diagnostics=tracker,
                    forecasts=perceived.observation.auxiliary.get(
                        forecast_auxiliary_key, ()
                    ),
                    local_obstacles=perceived.observation.local_obstacles,
                )
                timings["encounter"].append(
                    1000.0 * (time.perf_counter() - stage)
                )
                # Admission continuity is based on measured chassis motion,
                # never on a merely proposed/disarmed command.
                forward_passage.observe_feedback(
                    ControlCommand([status.v_mps, status.omega_radps])
                )
                stage = time.perf_counter()
                planning_reference = encounter_reference.select(
                    reference,
                    encounter_diagnostics,
                    (pose_x, pose_y, pose_yaw),
                    (args.goal_x, args.goal_y),
                )
                plan = controller.plan(
                    perceived.observation, planning_reference
                )
                torch.cuda.synchronize()
                timings["plan"].append(1000.0 * (time.perf_counter() - stage))
                stage = time.perf_counter()
                plan = encounter_control.apply(
                    plan,
                    encounter_diagnostics,
                    (pose_x, pose_y, pose_yaw),
                    perceived.guard,
                )
                encounter_context = encounter_control.planning_context(
                    encounter_diagnostics
                )
                forward_inhibit_reason = (
                    "encounter_mode:%s" % str(
                        encounter_diagnostics.get("encounter_phase", "idle")
                    )
                    if encounter_context[
                        "encounter_control_inhibit_forward_passage"
                    ]
                    else None
                )
                plan = forward_passage.apply(
                    plan, inhibit_reason=forward_inhibit_reason
                )
                planning_context = _physical_tracker_motion_context(
                    plan.diagnostics,
                    tracker,
                    pose_yaw,
                    config["planner"],
                )
                planning_context = _physical_goal_context(
                    planning_context,
                    pose_x,
                    pose_y,
                    pose_yaw,
                    args.goal_x,
                    args.goal_y,
                )
                # Semantic ownership is injected after physical tracker
                # fallbacks so no lower layer can silently replace its side.
                planning_context.update(encounter_context)
                timings["authority"].append(
                    1000.0 * (time.perf_counter() - stage)
                )
                time.sleep(0)
                decision = safety.arbitrate(
                    plan.proposed_control, perceived.guard, planning_context
                )
                armed_command = bool(args.publish and len(rows) >= args.warmup_cycles)
                goal_stop_triggered, last_goal_distance_m = _goal_stop_requested(
                    pose_x, pose_y, args.goal_x, args.goal_y,
                    args.goal_stop_radius_m,
                    enabled=(
                        args.goal_stop_radius_m > 0.0
                        and len(rows) >= args.warmup_cycles
                    ),
                )
                commanded_v = (
                    0.0 if goal_stop_triggered else decision.executed_control.v
                )
                commanded_omega = (
                    0.0 if goal_stop_triggered else decision.executed_control.omega
                )
                if goal_stop_triggered:
                    armed_command = False
                hazard_active = bool(
                    int(tracker.get("valid_forecast_count", 0) or 0) > 0
                    or planning_context.get(
                        "probabilistic_obstacle_temporal_emergency_triggered",
                        False,
                    )
                    or bool(tracker.get(
                        "temporal_flow_threat_track_indices", ()
                    ))
                    or (
                        bool(decision.diagnostics.get(
                            "temporal_scan_valid", False
                        ))
                        and float(decision.diagnostics.get(
                            "temporal_scan_ttc_s", float("inf")
                        )) <= 3.0
                    )
                )
                hazard_active, rear_only_hazard = _dynamic_hazard_sector(
                    hazard_active, decision.diagnostics
                )
                commanded_v, path_guard = path_guard_supervisor.apply(
                    pose_x,
                    pose_y,
                    pose_yaw,
                    args.goal_x,
                    args.goal_y,
                    commanded_v,
                    decision.reason,
                    emergency_stop=bool(
                        decision.diagnostics.get("emergency_stop", False)
                    ),
                    proposed_omega=commanded_omega,
                    maximum_omega_radps=args.max_omega_radps,
                    selected_probability=float(
                        max(
                            float(decision.diagnostics.get(
                                "dynamic_escape_selected_probability", 0.0
                            ) or 0.0),
                            float(planning_context.get(
                                "probabilistic_obstacle_maximum_step_probability",
                                0.0,
                            ) or 0.0),
                        )
                    ),
                    selected_probability_mass=float(
                        planning_context.get(
                            "probabilistic_obstacle_probability_mass", 0.0
                        ) or 0.0
                    ),
                    hazard_active=hazard_active,
                    rear_only_hazard=rear_only_hazard,
                    reverse_escape_exhausted=bool(
                        decision.diagnostics.get(
                            "dynamic_escape_post_retry_reverse_exhausted",
                            False,
                        )
                    ),
                    arbiter_steering_authoritative=(
                        _arbiter_steering_authoritative(
                            decision.reason, decision.diagnostics
                        )
                    ),
                    arbiter_rejoin_requested=bool(
                        decision.diagnostics.get(
                            "dynamic_escape_geometric_goal_release_applied",
                            False,
                        )
                        or decision.diagnostics.get(
                            "dynamic_escape_geometric_passage_completion_applied",
                            False,
                        )
                    ),
                    dynamic_safety_arbitration_enabled=bool(
                        decision.diagnostics.get(
                            "dynamic_safety_arbitration_enabled", True
                        )
                    ),
                )
                omega_override = path_guard.get(
                    "commanded_omega_override_radps"
                )
                if omega_override is not None:
                    commanded_omega = float(omega_override)
                immediate_translation_stop = (
                    _immediate_translation_stop_requested(
                        decision.reason,
                        decision.diagnostics,
                        commanded_v,
                    )
                )
                # The PC loop is forecast-dependent and measured only
                # 8--10 Hz.  A fixed per-cycle slew limit here both created
                # visible command stairs and delayed safety braking.  Send the
                # final target unchanged; the Pi interpolates normal motion at
                # its independent 20 Hz CAN rate and bypasses interpolation
                # for the explicit stop flags below.
                physical_slew_guard = {
                    "active": False,
                    "reason": "delegated_to_pi_20hz",
                    "input_v_mps": float(commanded_v),
                    "input_omega_radps": float(commanded_omega),
                    "output_v_mps": float(commanded_v),
                    "output_omega_radps": float(commanded_omega),
                    "immediate_translation_stop": bool(
                        immediate_translation_stop or goal_stop_triggered
                    ),
                    "immediate_all_stop": bool(goal_stop_triggered),
                }
                direction_guard = {
                    "active": False,
                    "reason": "delegated_to_pi_20hz",
                    "input_v_mps": float(commanded_v),
                    "output_v_mps": float(commanded_v),
                    "omega_preserved_radps": float(commanded_omega),
                }
                # Feed back the final post-arbitration command, not the planner
                # proposal.  A zero/reverse command or a safety takeover owns
                # the direction and immediately releases any passage prefix.
                forward_passage.observe_commanded(
                    ControlCommand([commanded_v, commanded_omega]),
                    decision.reason,
                )
                sequence = heartbeat.publish(
                    commanded_v,
                    commanded_omega,
                    arm=armed_command,
                    immediate_translation_stop=(
                        immediate_translation_stop or goal_stop_triggered
                    ),
                    immediate_all_stop=goal_stop_triggered,
                )
                if armed_command:
                    executed_feedback = ControlCommand(
                        [commanded_v, commanded_omega],
                        observation.timestamp,
                        "physical_command_boundary",
                    )
                    feedback_overridden = bool(
                        decision.overridden
                        or not np.allclose(
                            executed_feedback.values,
                            plan.proposed_control.values,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    )
                    controller.observe_safety_decision(SafetyDecision(
                        proposed_control=plan.proposed_control,
                        executed_control=executed_feedback,
                        overridden=feedback_overridden,
                        reason=(
                            str(path_guard.get("reason", decision.reason))
                            if path_guard.get("active", False)
                            else decision.reason
                        ),
                        diagnostics={
                            **dict(decision.diagnostics),
                            "physical_command_slew_guard": dict(
                                physical_slew_guard
                            ),
                            "path_guard": dict(path_guard),
                        },
                    ))
                timings["total"].append(1000.0 * (time.perf_counter() - cycle_start))
                row = {
                    "cycle": len(rows), "sequence": sequence,
                    "timestamp": observation.timestamp,
                    "pose": [pose_x, pose_y, pose_yaw],
                    "feedback": [status.v_mps, status.omega_radps],
                    "proposed": list(plan.proposed_control.values),
                    "arbitrated": list(decision.executed_control.values),
                    "commanded": [commanded_v, commanded_omega],
                    "arm_requested": armed_command,
                    "safety_reason": decision.reason,
                    "goal_distance_m": last_goal_distance_m,
                    "goal_stop_triggered": goal_stop_triggered,
                    "path_guard": path_guard,
                    "physical_command_slew_guard": physical_slew_guard,
                    "direction_reversal_guard": direction_guard,
                    "control_cause_chain": _control_cause_chain(
                        plan=plan,
                        planning_context=planning_context,
                        decision=decision,
                        path_guard=path_guard,
                        commanded_v=commanded_v,
                        commanded_omega=commanded_omega,
                        goal_stop_triggered=goal_stop_triggered,
                        status=status,
                        heartbeat=heartbeat,
                    ),
                    # The stable field name preserves shadow-run tooling.  Its
                    # own control_enabled bit states whether this run merely
                    # observed or exercised semantic authority.
                    "encounter_mode_shadow": encounter_diagnostics,
                    "tracker_dynamic_indices": tracker.get("mapless_dynamic_track_indices", ()),
                    "tracker_static_indices": tracker.get("mapless_static_track_indices", ()),
                    "tracker_unknown_indices": tracker.get("mapless_unknown_track_indices", ()),
                    "forecast_count": tracker.get("valid_forecast_count", 0),
                    "remote_status": {
                        "age_s_before_cycle_plan": status_age_s,
                        "battery_v": status.battery_v,
                        "control_mode": status.control_mode,
                        "fault": status.fault,
                        "gateway_armed": status.armed,
                        "gateway_last_sequence": status.last_sequence,
                        "gateway_target": [
                            status.target_v_mps,
                            status.target_omega_radps,
                        ],
                        "gateway_applied": [
                            status.applied_v_mps,
                            status.applied_omega_radps,
                        ],
                        "status_packets_received": remote.status_packets_received,
                        "scan_packets_received": remote.received_packets,
                        "scan_sequence_gaps": remote.sequence_gaps,
                        "scan_decode_errors": remote.decode_errors,
                        "scan_backlog_packets_dropped": (
                            remote.backlog_packets_dropped
                        ),
                        "scan_backlog_packets_preserved": (
                            remote.backlog_packets_preserved
                        ),
                        "scan_stale_points_dropped": (
                            remote.stale_points_dropped
                        ),
                    },
                    "local_obstacles": _local_obstacle_diagnostics(
                        perceived.observation.local_obstacles,
                        pose_x,
                        pose_y,
                        config["planner"]["robot_radius"],
                    ),
                    "diagnostics": _real_robot_diagnostic_payload(
                        tracker, planning_context, decision.diagnostics
                    ),
                    "scan": scan_diagnostics.__dict__,
                    "livox_motion_compensation": deskew_diagnostics,
                    "timing_ms": {name: values[-1] for name, values in timings.items()},
                }
                rows.append(row)
                stream.write(row)
                if goal_stop_triggered:
                    break
                next_cycle += args.period_s
                delay = next_cycle - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_cycle = time.monotonic()
            diagnostic_writer_stats = {
                "written_rows": int(stream._written_rows),
                "max_queue_depth": int(stream._max_queue_depth),
            }
            heartbeat_stats = {
                "period_s": float(heartbeat.period_s),
                "lease_s": float(heartbeat.lease_s),
                "refresh_count": int(heartbeat.refresh_count),
                "lease_expiry_count": int(heartbeat.lease_expiry_count),
            }
    finally:
        summary = {
            "contract": "pc_cuda_pi_gateway_mppi_ablation_v1",
            "publish_enabled": bool(args.publish),
            "initial_chassis_fault": initial_chassis_fault,
            "initial_chassis_fault_labels": initial_chassis_fault_labels,
            "completed_cycles": len(rows),
            "livox_timeout_count": int(livox_timeout_count),
            "livox_timeout_streak_at_exit": int(livox_timeout_streak),
            "chassis_status_recovery_count": int(
                chassis_status_recovery_count
            ),
            "chassis_status_max_age_s": float(chassis_status_max_age_s),
            "diagnostic_writer": diagnostic_writer_stats,
            "command_heartbeat": heartbeat_stats,
            "timing": {name: _timing(values) for name, values in timings.items()},
            "config_sha256": _sha256(config_path),
            "lidar_config_sha256": _sha256(args.lidar_config),
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024.0 * 1024.0),
            "planner_device": config["planner"]["device"],
            "actor_device": config["rl"]["device"],
            "residual_learning_enabled": bool(
                algorithm_features["residual_learning"]
            ),
            "algorithm_profile": config["real_robot_deployment"][
                "algorithm_profile"
            ],
            "algorithm_features": dict(algorithm_features),
            "runtime_algorithm_mode": config["real_robot_deployment"][
                "runtime_algorithm_mode"
            ],
            "hss_device": config["planner"]["paper_rl_driven"][
                "reliability_sidecar"
            ].get("device"),
            "diagnostic_schema_version": "pc_pi_full_proposed_diagnostics_v3",
            "encounter_mode_shadow_enabled": bool(
                encounter_mode_config.enabled
            ),
            "encounter_mode_control_enabled": bool(
                encounter_control_config.enabled
            ),
            "encounter_policy": "disabled_simulation_parity",
            "encounter_mode_config": asdict(encounter_mode_config),
            "encounter_control_config": asdict(encounter_control_config),
            "encounter_control_applied_cycles": sum(
                bool(
                    row.get("diagnostics", {})
                    .get("planner", {})
                    .get("encounter_control_applied", False)
                )
                for row in rows
            ),
            "encounter_phase_counts": {
                phase: sum(
                    row.get("encounter_mode_shadow", {}).get(
                        "encounter_phase"
                    ) == phase
                    for row in rows
                )
                for phase in (
                    "idle", "straight_crossing", "oblique_crossing",
                    "frontal_approach", "rejoin",
                )
            },
            "diagnostic_log": str(log_path),
            "controller_dt_s": controller_dt_s,
            "forecast_dt_s": forecast_dt_s,
            "warm_start": warm_start,
            "human_leg_mode": bool(
                args.human_leg_mode
                and algorithm_features["change_aware_prediction"]
            ),
            "real_robot_forward_passage": False,
            "forward_passage_policy": "disabled_simulation_parity",
            "forward_passage_risk_ceiling": float(
                args.forward_passage_risk_ceiling
            ),
            "forward_passage_mass_ceiling": float(
                args.forward_passage_mass_ceiling
            ),
            "forward_passage_commit_steps": int(
                forward_passage_config.commit_steps
            ),
            "forward_passage_feedback_steps": int(
                forward_passage_config.minimum_forward_feedback_steps
            ),
            "forward_passage_reentry_cooldown_steps": int(
                forward_passage_config.reentry_cooldown_steps
            ),
            "forward_passage_feedback_mismatch_steps": int(
                forward_passage_config.maximum_feedback_mismatch_steps
            ),
            "forward_passage_applied_cycles": sum(
                bool(
                    row.get("diagnostics", {})
                    .get("planner", {})
                    .get("real_robot_forward_passage_applied", False)
                )
                for row in rows
            ),
            "physical_action_limits": {
                "max_v_mps": float(args.max_v_mps),
                "max_reverse_v_mps": float(args.max_reverse_v_mps),
                "max_omega_radps": float(args.max_omega_radps),
            },
            "path_guard_applied_cycles": sum(
                bool(row.get("path_guard", {}).get("active", False))
                for row in rows
            ),
            "path_guard_bypassed_cycles": sum(
                bool(row.get("path_guard", {}).get("bypassed", False))
                for row in rows
            ),
            "physical_command_slew_applied_cycles": sum(
                bool(row.get("physical_command_slew_guard", {}).get(
                    "active", False
                ))
                for row in rows
            ),
            "direction_reversal_transition_cycles": sum(
                bool(row.get("direction_reversal_guard", {}).get(
                    "active", False
                ))
                for row in rows
            ),
            "livox_motion_compensated_cycles": sum(
                bool(
                    row.get("livox_motion_compensation", {}).get(
                        "applied", False
                    )
                )
                for row in rows
            ),
            "scan_backlog_packets_dropped": (
                rows[-1]["remote_status"]["scan_backlog_packets_dropped"]
                if rows else 0
            ),
            "scan_backlog_packets_preserved": (
                rows[-1]["remote_status"]["scan_backlog_packets_preserved"]
                if rows else 0
            ),
            "scan_stale_points_dropped": (
                rows[-1]["remote_status"]["scan_stale_points_dropped"]
                if rows else 0
            ),
            "goal_stop_radius_m": float(args.goal_stop_radius_m),
            "termination_mode": (
                "goal_with_watchdog" if args.until_goal else "duration_or_goal"
            ),
            "watchdog_timeout_s": float(args.duration_s),
            "goal_stop_triggered": bool(goal_stop_triggered),
            "final_goal_distance_m": float(last_goal_distance_m),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.until_goal and not goal_stop_triggered:
        raise RuntimeError(
            "goal was not reached before the armed-run watchdog expired"
        )


if __name__ == "__main__":
    main()
