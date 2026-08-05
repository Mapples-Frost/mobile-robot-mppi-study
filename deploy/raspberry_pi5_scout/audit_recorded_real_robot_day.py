#!/usr/bin/env python3
"""Audit every recorded PC-CUDA/Pi-gateway run from one physical-test day.

The compact armed logs do not contain the full Mid-360 point cloud, so this is
not a dynamics simulation.  It reuses every recorded proposal, pose, tracker,
forecast and compact scan-guard result, then runs the *current* final safety
arbiter and deployment path supervisor in their original temporal order.  The
result catches cross-layer command discontinuities and state-machine regressions
without pretending to reproduce unrecorded geometry.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from deploy.raspberry_pi5_scout.build_pi5_full_config import (
    build_pi5_full_config,
)
from deploy.raspberry_pi5_scout.replay_recorded_dynamic_escape import (
    _restore_rear_pass_guard,
)
from deploy.raspberry_pi5_scout.run_remote_cuda_full import (
    _DYNAMIC_PATH_AUTHORITY_REASONS,
    _GOAL_REJOIN_REAR_HEMISPHERE_RAD,
    _DynamicPathGuardSupervisor,
    _dynamic_hazard_sector,
    _physical_goal_context,
    _physical_tracker_motion_context,
)
from mobile_robot_mppi.core.spaces import action_spec_from_config
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CYCLE_KEYS = frozenset({
    "cycle",
    "pose",
    "proposed",
    "commanded",
    "diagnostics",
    "safety_reason",
})
HARD_FRONT_REASONS = frozenset({"hard_stop", "near_body_hard_stop"})
DYNAMIC_REASONS = frozenset({
    "dynamic_active_escape",
    "dynamic_corridor_escape",
    "dynamic_hard_stop_escape",
    "dynamic_recovery_advance",
    "dynamic_recovery_align",
    "dynamic_recovery_align_creep",
    "rear_pass_through",
    "temporal_collision_risk",
    "temporal_slowdown",
})
GOAL_REJOIN_REASONS = frozenset({
    "goal_heading_rejoin",
    "stale_reverse_goal_rejoin",
    "goal_behind_reverse_rejoin",
    "goal_behind_turn_rejoin",
})


def _finite(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_rows(path):
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({
                    "line": line_number,
                    "error": str(exc),
                })
    return rows, errors


def _motion_context(row, planner_config):
    """Return current relative-motion context and body-frame motion values."""

    tracker = dict(row.get("diagnostics", {}).get("tracker", {}))
    pose = row.get("pose", (0.0, 0.0, 0.0))
    yaw = float(pose[2]) if len(pose) >= 3 else 0.0
    context = _physical_tracker_motion_context(
        row.get("diagnostics", {}).get("planner", {}),
        tracker,
        yaw,
        planner_config,
    )
    longitudinal = _finite(context.get(
        "probabilistic_obstacle_motion_longitudinal_body_mps"
    ), 0.0)
    lateral = _finite(context.get(
        "probabilistic_obstacle_motion_lateral_body_mps"
    ), 0.0)
    fraction = _finite(context.get(
        "probabilistic_obstacle_motion_lateral_fraction"
    ), 0.0)
    return context, float(longitudinal), float(lateral), float(fraction)


def _scenario(row, longitudinal, lateral, lateral_fraction):
    diagnostics = row.get("diagnostics", {})
    guard = diagnostics.get("safety", {})
    tracker = diagnostics.get("tracker", {})
    planner = diagnostics.get("planner", {})
    reason = str(row.get("safety_reason", "front_clear"))
    associated = bool(tracker.get("associated", False))
    forecast_count = int(
        planner.get("probabilistic_obstacle_forecast_count", 0) or 0
    )
    surface_range = _finite(guard.get("dynamic_obstacle_surface_range_m"))
    active = bool(
        reason in DYNAMIC_REASONS
        or (
            associated
            and forecast_count > 0
            and surface_range is not None
            and surface_range <= 3.0
        )
    )
    if not active:
        return None

    bearing = _finite(guard.get("dynamic_obstacle_bearing_rad"))
    if bearing is None:
        bearing = _finite(guard.get("temporal_scan_center_angle_rad"))
    speed = math.hypot(longitudinal, lateral)
    if (
        speed >= 0.20
        and lateral_fraction >= 0.55
        and bearing is not None
        and abs(bearing) <= math.radians(120.0)
    ):
        return "lateral_crossing"
    if bearing is None:
        return "dynamic_unlocalized"
    absolute_deg = abs(math.degrees(bearing))
    temporal_closing = str(guard.get("temporal_scan_reason", "")) == (
        "closing_consensus"
    )
    if absolute_deg <= 60.0 and (
        longitudinal <= -0.10 or temporal_closing
    ):
        return "frontal_approach"
    if absolute_deg <= 105.0:
        return "side_or_oblique_approach"
    return "rear_approach"


def _episode_count(cycles, maximum_gap=3):
    if not cycles:
        return 0
    count = 1
    previous = cycles[0]
    for cycle in cycles[1:]:
        if cycle - previous > maximum_gap:
            count += 1
        previous = cycle
    return count


def _goal_for_rows(rows):
    first_distance = _finite(rows[0].get("goal_distance_m"), 5.0)
    return (5.0, 2.0) if first_distance > 5.20 else (5.0, 0.0)


def _replay_run(rows, summary, action_spec, guard_config, planner_config):
    arbiter = ScanGuardArbiter(action_spec, guard_config)
    path_supervisor = _DynamicPathGuardSupervisor()
    goal_x, goal_y = _goal_for_rows(rows)
    violation_examples = defaultdict(list)
    scenario_cycles = defaultdict(list)
    safety_reasons = Counter()
    replay_reasons = Counter()
    prediction = Counter()
    replay_zero_cycles = 0
    recorded_zero_cycles = 0
    max_dynamic_spin = 0
    dynamic_spin = 0
    max_goal_behind_turn = 0
    goal_behind_turn = 0
    last_goal_behind_turn_sign = 0.0
    max_negative_goal_progress_streak = 0
    negative_goal_progress_streak = 0
    last_rear_sign = 0.0
    last_rear_cycle = -1000
    replay_series = []
    recorded_series = []
    performance_events = Counter()

    lower = action_spec.lower
    upper = action_spec.upper
    v_index = action_spec.index("v_cmd")
    omega_index = action_spec.index("omega_cmd")

    for row_index, row in enumerate(rows):
        cycle = int(row.get("cycle", row_index))
        diagnostics = row.get("diagnostics", {})
        recorded_guard = dict(diagnostics.get("safety", {}))
        if bool(recorded_guard.get("rear_pass_through_active", False)):
            guard = _restore_rear_pass_guard(row)
        else:
            guard = recorded_guard
        context, longitudinal, lateral, fraction = _motion_context(
            row, planner_config
        )
        context = _physical_goal_context(
            context,
            pose_x=row["pose"][0],
            pose_y=row["pose"][1],
            pose_yaw=row["pose"][2],
            goal_x=goal_x,
            goal_y=goal_y,
        )
        scenario = _scenario(row, longitudinal, lateral, fraction)
        if scenario is not None:
            scenario_cycles[scenario].append(cycle)

        proposed = ControlCommand(row["proposed"])
        decision = arbiter.arbitrate(proposed, guard, context)
        pose = row["pose"]
        selected_probability = _finite(context.get(
            "probabilistic_obstacle_maximum_step_probability"
        ), 0.0)
        selected_mass = _finite(context.get(
            "probabilistic_obstacle_probability_mass"
        ), 0.0)
        recorded_path = row.get("path_guard", {})
        hazard_active = bool(
            recorded_path.get("fresh_hazard_active", False)
            or recorded_path.get("rear_only_hazard", False)
        )
        hazard_active, rear_only_hazard = _dynamic_hazard_sector(
            hazard_active, decision.diagnostics
        )
        output_v, path = path_supervisor.apply(
            *pose,
            goal_x,
            goal_y,
            decision.executed_control.v,
            decision.reason,
            emergency_stop=bool(
                decision.diagnostics.get("emergency_stop", False)
            ),
            proposed_omega=decision.executed_control.omega,
            maximum_omega_radps=float(upper[omega_index]),
            selected_probability=selected_probability,
            selected_probability_mass=selected_mass,
            hazard_active=hazard_active,
            rear_only_hazard=rear_only_hazard,
            reverse_escape_exhausted=bool(
                decision.diagnostics.get(
                    "dynamic_escape_post_retry_reverse_exhausted", False
                )
            ),
        )
        output_omega = float(decision.executed_control.omega)
        omega_override = path.get("commanded_omega_override_radps")
        if omega_override is not None:
            output_omega = float(omega_override)

        safety_reasons[str(row.get("safety_reason"))] += 1
        replay_reasons[str(decision.reason)] += 1
        prediction["threat_cycles"] += int(scenario is not None)
        prediction["forecast_valid_cycles"] += int(bool(
            diagnostics.get("tracker", {}).get("forecast_valid", False)
        ))
        prediction["active_avoidance_cycles"] += int(bool(context.get(
            "probabilistic_obstacle_active_avoidance_enabled", False
        )))
        prediction["temporal_emergency_cycles"] += int(bool(context.get(
            "probabilistic_obstacle_temporal_emergency_triggered", False
        )))
        recorded_zero_cycles += int(abs(float(row["commanded"][0])) < 0.02)
        replay_zero_cycles += int(abs(output_v) < 0.02)
        replay_series.append({
            "cycle": cycle,
            "v": float(output_v),
            "omega": float(output_omega),
            "safety_reason": str(decision.reason),
            "path_reason": str(path.get("reason", "")),
            "post_retry_hold": bool(decision.diagnostics.get(
                "dynamic_escape_post_retry_reverse_hold_applied", False
            )),
            "post_retry_side_forward": bool(decision.diagnostics.get(
                "dynamic_escape_post_retry_side_forward_applied", False
            )),
            "timestamp": _finite(row.get("timestamp")),
            "hard_stop": decision.reason in {
                "near_body_hard_stop",
                "temporal_collision_risk",
                "dynamic_hard_stop_escape",
            } or bool(decision.diagnostics.get(
                "dynamic_escape_post_retry_reverse_hold_applied", False
            )),
        })
        recorded_series.append({
            "cycle": cycle,
            "v": float(row["commanded"][0]),
            "omega": float(row["commanded"][1]),
            "timestamp": _finite(row.get("timestamp")),
            "applied_v": _finite(
                row.get("remote_status", {}).get(
                    "gateway_applied", (None, None)
                )[0]
                if len(row.get("remote_status", {}).get(
                    "gateway_applied", ()
                )) >= 2
                else None,
                float(row["commanded"][0]),
            ),
            "applied_omega": _finite(
                row.get("remote_status", {}).get(
                    "gateway_applied", (None, None)
                )[1]
                if len(row.get("remote_status", {}).get(
                    "gateway_applied", ()
                )) >= 2
                else None,
                float(row["commanded"][1]),
            ),
            "hard_stop": str(row.get("safety_reason")) in {
                "near_body_hard_stop",
                "temporal_collision_risk",
                "dynamic_hard_stop_escape",
            },
        })
        performance_events["temporal_slowdown_path_authority_cycles"] += int(
            decision.reason == "temporal_slowdown"
            and str(path.get("reason")) == "dynamic_authority"
        )
        path_reason = str(path.get("reason", ""))
        performance_events["goal_behind_reverse_rejoin_cycles"] += int(
            path_reason == "goal_behind_reverse_rejoin"
        )
        performance_events["goal_behind_turn_rejoin_cycles"] += int(
            path_reason == "goal_behind_turn_rejoin"
        )
        performance_events["geometric_goal_release_cycles"] += int(bool(
            decision.diagnostics.get(
                "dynamic_escape_geometric_goal_release_applied", False
            )
        ))
        performance_events["retry_goal_rejected_cycles"] += int(bool(
            decision.diagnostics.get(
                "dynamic_escape_persistent_front_retry_goal_rejected", False
            )
        ))
        performance_events["post_retry_reverse_hold_cycles"] += int(bool(
            decision.diagnostics.get(
                "dynamic_escape_post_retry_reverse_hold_applied", False
            )
        ))
        performance_events["post_retry_side_forward_cycles"] += int(bool(
            decision.diagnostics.get(
                "dynamic_escape_post_retry_side_forward_applied", False
            )
        ))
        performance_events["reverse_goal_realign_cycles"] += int(bool(
            decision.diagnostics.get(
                "dynamic_escape_reverse_goal_realign_applied", False
            )
        ))
        performance_events["path_hazard_reverse_exhausted_cycles"] += int(
            path_reason == "dynamic_hazard_reverse_exhausted"
        )
        performance_events["path_hazard_goal_rejoin_cycles"] += int(
            path_reason == "dynamic_hazard_goal_rejoin"
        )

        # A cleared-threat route-rejoin command must not keep translating in
        # the negative direction of its own path target.  This is the exact
        # cross-layer failure observed after successful avoidance: the planner
        # proposed reverse with the target behind, then the deployment layer
        # replaced it with positive speed for several cycles.  Require three
        # consecutive negative-projection samples before flagging so a single
        # pose wrap/noisy frame cannot fail an otherwise continuous run.
        rejoin_heading_error = _finite(path.get("heading_error_rad"))
        rejoin_progress = None
        if rejoin_heading_error is not None:
            rejoin_progress = float(
                output_v * math.cos(rejoin_heading_error)
            )
        negative_rejoin_progress = bool(
            path_reason in GOAL_REJOIN_REASONS
            and not bool(path.get("hazard_active", False))
            and rejoin_progress is not None
            and rejoin_progress < -0.01
        )
        if negative_rejoin_progress:
            negative_goal_progress_streak += 1
            max_negative_goal_progress_streak = max(
                max_negative_goal_progress_streak,
                negative_goal_progress_streak,
            )
            if negative_goal_progress_streak == 3:
                violation_examples[
                    "sustained_negative_goal_rejoin_progress"
                ].append(cycle)
        else:
            negative_goal_progress_streak = 0

        replay_values = np.asarray((output_v, output_omega))
        if np.any(replay_values < lower - 1.0e-12) or np.any(
            replay_values > upper + 1.0e-12
        ):
            violation_examples["action_bounds"].append(cycle)
        if (
            decision.diagnostics.get(
                "dynamic_escape_post_retry_reverse_exhausted", False
            )
            and output_v < -0.02
        ):
            violation_examples[
                "post_retry_reverse_after_exhaustion"
            ].append(cycle)
        if (
            decision.diagnostics.get(
                "dynamic_escape_post_retry_side_forward_applied", False
            )
            and (
                output_v <= 0.0
                or output_v > 0.20 + 1.0e-12
                or abs(output_omega) > 1.0e-12
            )
        ):
            violation_examples["invalid_post_retry_side_forward"].append(
                cycle
            )

        original_reason = str(guard.get("reason", "front_clear"))
        original_bearing = None
        near_body_points = guard.get("near_body_points", ())
        if (
            original_reason == "near_body_hard_stop"
            and isinstance(near_body_points, (list, tuple))
            and near_body_points
        ):
            point = near_body_points[0]
            original_bearing = _finite(point.get("base_angle"))
            if original_bearing is None:
                point_x = _finite(point.get("x"))
                point_y = _finite(point.get("y"))
                if point_x is not None and point_y is not None:
                    original_bearing = math.atan2(point_y, point_x)
        if original_bearing is None:
            original_bearing = _finite(
                guard.get("dynamic_obstacle_bearing_rad")
            )
        if original_bearing is None:
            original_bearing = _finite(
                guard.get("temporal_scan_center_angle_rad")
            )
        if (
            bool(guard.get("emergency_stop", False))
            and original_reason in HARD_FRONT_REASONS
            and original_bearing is not None
            and abs(original_bearing) <= math.radians(100.0)
            and output_v > 1.0e-12
        ):
            violation_examples["protected_front_advance"].append(cycle)

        force_rear = bool(
            decision.reason == "rear_pass_through"
            and decision.diagnostics.get(
                "rear_pass_through_force_forward_ready", False
            )
        )
        if force_rear and decision.executed_control.v < 0.35 - 1.0e-12:
            violation_examples["rear_force_forward"].append(cycle)
        if force_rear and output_v < 0.35 - 1.0e-12:
            violation_examples["rear_path_continuity"].append(cycle)
        rear_goal_steer = bool(path.get(
            "rear_only_goal_steer_active", False
        ))
        rear_goal_heading_error = _finite(path.get("heading_error_rad"))
        if (
            force_rear
            and rear_goal_steer
            and rear_goal_heading_error is not None
            and abs(rear_goal_heading_error)
            <= _GOAL_REJOIN_REAR_HEMISPHERE_RAD
            and abs(output_omega) >= 0.10
            and output_omega * rear_goal_heading_error < 0.0
        ):
            violation_examples["rear_goal_steer_wrong_side"].append(cycle)
        if (
            force_rear
            and not rear_goal_steer
            and abs(output_omega) >= 0.10
        ):
            rear_sign = float(np.sign(output_omega))
            if (
                cycle - last_rear_cycle <= 3
                and last_rear_sign != 0.0
                and rear_sign != last_rear_sign
            ):
                violation_examples["rear_turn_sign_flip"].append(cycle)
            last_rear_sign = rear_sign
            last_rear_cycle = cycle

        if (
            decision.reason in _DYNAMIC_PATH_AUTHORITY_REASONS
            and decision.executed_control.v > 0.02
            and output_v <= 0.02
        ):
            violation_examples["dynamic_path_zero_override"].append(cycle)

        turn_source = str(decision.diagnostics.get(
            "dynamic_escape_geometric_turn_source", ""
        ))
        prediction_owned_turn = bool(
            turn_source == "predicted_relative_motion"
            or decision.diagnostics.get(
                "dynamic_escape_prediction_direction_late_acquisition_applied",
                False,
            )
        )
        if (
            prediction_owned_turn
            and abs(lateral) >= 0.20
            and fraction >= 0.35
            and abs(output_omega) >= 0.10
            and output_v > 0.02
            and lateral * output_omega >= 0.0
        ):
            violation_examples["crossing_wrong_turn_side"].append(cycle)

        dynamic_spin_active = bool(
            abs(output_v) < 0.02
            and abs(output_omega) >= 0.10
            and path_reason != "goal_behind_turn_rejoin"
            and (
                decision.reason in DYNAMIC_REASONS
                or scenario is not None
                or decision.diagnostics.get(
                    "dynamic_escape_hard_stop_transaction_active", False
                )
            )
        )
        if dynamic_spin_active:
            dynamic_spin += 1
            max_dynamic_spin = max(max_dynamic_spin, dynamic_spin)
        else:
            dynamic_spin = 0

        if path_reason == "goal_behind_turn_rejoin":
            goal_behind_turn += 1
            max_goal_behind_turn = max(
                max_goal_behind_turn, goal_behind_turn
            )
            current_turn_sign = float(np.sign(output_omega))
            if (
                last_goal_behind_turn_sign != 0.0
                and current_turn_sign != 0.0
                and current_turn_sign != last_goal_behind_turn_sign
            ):
                violation_examples[
                    "goal_behind_turn_sign_flip"
                ].append(cycle)
            if current_turn_sign != 0.0:
                last_goal_behind_turn_sign = current_turn_sign
        else:
            goal_behind_turn = 0
            last_goal_behind_turn_sign = 0.0

    if max_dynamic_spin > 9:
        violation_examples["unbounded_dynamic_spin"].append(max_dynamic_spin)

    def single_cycle_pulses(series):
        return sum(
            1
            for index in range(1, len(series) - 1)
            if series[index - 1]["v"] > 0.10
            and abs(series[index]["v"]) < 0.02
            and series[index + 1]["v"] > 0.10
            and not series[index]["hard_stop"]
        )

    def single_cycle_pulse_cycles(series):
        return [
            int(series[index]["cycle"])
            for index in range(1, len(series) - 1)
            if series[index - 1]["v"] > 0.10
            and abs(series[index]["v"]) < 0.02
            and series[index + 1]["v"] > 0.10
            and not series[index]["hard_stop"]
        ]

    def single_cycle_pulse_windows(series):
        windows = []
        for index in range(1, len(series) - 1):
            if (
                series[index - 1]["v"] > 0.10
                and abs(series[index]["v"]) < 0.02
                and series[index + 1]["v"] > 0.10
                and not series[index]["hard_stop"]
            ):
                windows.append([
                    {
                        key: item[key]
                        for key in (
                            "cycle",
                            "v",
                            "omega",
                            "safety_reason",
                            "path_reason",
                            "post_retry_hold",
                            "post_retry_side_forward",
                        )
                    }
                    for item in series[index - 1:index + 2]
                ])
        return windows

    def moving_turn_sign_flips(series):
        return sum(
            1
            for index in range(1, len(series))
            if abs(series[index - 1]["omega"]) >= 0.20
            and abs(series[index]["omega"]) >= 0.20
            and series[index - 1]["omega"] * series[index]["omega"] < 0.0
            and (
                series[index - 1]["v"] > 0.02
                or series[index]["v"] > 0.02
            )
        )

    def maximum_reverse_streak(series):
        maximum = 0
        current = 0
        for item in series:
            if item["v"] < -0.02:
                current += 1
                maximum = max(maximum, current)
            else:
                current = 0
        return maximum

    recorded_applied_series = [
        {
            "v": item["applied_v"],
            "omega": item["applied_omega"],
            "hard_stop": item["hard_stop"],
        }
        for item in recorded_series
    ]
    replay_applied_series = []
    applied_v = float(recorded_applied_series[0]["v"])
    applied_omega = float(recorded_applied_series[0]["omega"])
    previous_timestamp = replay_series[0]["timestamp"]
    for item in replay_series:
        timestamp = item["timestamp"]
        if timestamp is None or previous_timestamp is None:
            dt_s = 0.10
        else:
            dt_s = float(np.clip(
                timestamp - previous_timestamp, 0.05, 0.25
            ))
        previous_timestamp = timestamp
        target_v = float(item["v"])
        target_omega = float(item["omega"])
        if item["hard_stop"]:
            applied_v = 0.0
        else:
            accelerating = bool(
                target_v * applied_v >= 0.0
                and abs(target_v) > abs(applied_v)
            )
            maximum_v_delta = (1.0 if accelerating else 2.0) * dt_s
            applied_v = float(np.clip(
                target_v,
                applied_v - maximum_v_delta,
                applied_v + maximum_v_delta,
            ))
        maximum_omega_delta = 4.0 * dt_s
        applied_omega = float(np.clip(
            target_omega,
            applied_omega - maximum_omega_delta,
            applied_omega + maximum_omega_delta,
        ))
        replay_applied_series.append({
            "v": applied_v,
            "omega": applied_omega,
            "hard_stop": item["hard_stop"],
        })

    performance = {
        "recorded_single_cycle_stop_pulses": single_cycle_pulses(
            recorded_series
        ),
        "replay_single_cycle_stop_pulses": single_cycle_pulses(replay_series),
        "replay_single_cycle_stop_pulse_cycles": single_cycle_pulse_cycles(
            replay_series
        ),
        "replay_single_cycle_stop_pulse_windows": single_cycle_pulse_windows(
            replay_series
        ),
        "recorded_moving_turn_sign_flips": moving_turn_sign_flips(
            recorded_series
        ),
        "replay_moving_turn_sign_flips": moving_turn_sign_flips(replay_series),
        "recorded_applied_single_cycle_stop_pulses": single_cycle_pulses(
            recorded_applied_series
        ),
        "replay_applied_single_cycle_stop_pulses": single_cycle_pulses(
            replay_applied_series
        ),
        "recorded_applied_moving_turn_sign_flips": moving_turn_sign_flips(
            recorded_applied_series
        ),
        "replay_applied_moving_turn_sign_flips": moving_turn_sign_flips(
            replay_applied_series
        ),
        "recorded_reverse_cycles": sum(
            item["v"] < -0.02 for item in recorded_series
        ),
        "replay_reverse_cycles": sum(
            item["v"] < -0.02 for item in replay_series
        ),
        "recorded_maximum_reverse_streak": maximum_reverse_streak(
            recorded_series
        ),
        "replay_maximum_reverse_streak": maximum_reverse_streak(
            replay_series
        ),
        **dict(performance_events),
    }

    scenarios = {
        label: {
            "cycles": len(cycles),
            "episodes": _episode_count(cycles),
        }
        for label, cycles in sorted(scenario_cycles.items())
    }
    return {
        "completed_cycles": len(rows),
        "goal_stop_triggered": bool(summary.get("goal_stop_triggered", False)),
        "final_goal_distance_m": _finite(summary.get(
            "final_goal_distance_m"
        )),
        "algorithm_profile": summary.get("algorithm_profile") or "legacy_full",
        "scenarios": scenarios,
        "prediction": dict(prediction),
        "recorded_zero_translation_cycles": recorded_zero_cycles,
        "replay_zero_translation_cycles": replay_zero_cycles,
        "maximum_dynamic_spin_cycles": max_dynamic_spin,
        "maximum_goal_behind_turn_cycles": max_goal_behind_turn,
        "maximum_negative_goal_rejoin_progress_cycles": (
            max_negative_goal_progress_streak
        ),
        "performance": performance,
        "recorded_safety_reasons": dict(safety_reasons),
        "replay_safety_reasons": dict(replay_reasons),
        "violations": {
            key: values[:20]
            for key, values in sorted(violation_examples.items())
            if values
        },
    }


def audit(runs_root, weight_root, date_prefix):
    config = build_pi5_full_config(
        weight_root,
        goal_x=5.0,
        goal_y=2.0,
        max_v_mps=0.5,
        max_reverse_v_mps=0.3,
        max_omega_radps=0.6,
    )
    action_spec = action_spec_from_config(config["action_space"])
    guard_config = config["perception"]["scan_guard"]
    planner_config = config["planner"]

    inventory = []
    run_reports = {}
    data_quality_issues = []
    all_violation_counts = Counter()
    scenario_cycles = Counter()
    scenario_episodes = Counter()
    armed_cycles = 0
    armed_runs = 0
    reached_runs = 0
    total_bytes = 0
    performance_totals = Counter()

    for run_directory in sorted(Path(runs_root).glob(f"{date_prefix}_*")):
        if not run_directory.is_dir():
            continue
        summary_path = run_directory / "summary.json"
        cycles_path = run_directory / "cycles.jsonl"
        config_path = run_directory / "config_resolved.yaml"
        for path in (summary_path, cycles_path, config_path):
            if path.exists():
                total_bytes += path.stat().st_size
        item = {
            "run": run_directory.name,
            "summary_present": summary_path.exists(),
            "cycles_present": cycles_path.exists(),
            "config_present": config_path.exists(),
        }
        if not summary_path.exists():
            item.update({"armed": False, "cycles": 0})
            data_quality_issues.append({
                "run": run_directory.name,
                "issue": "missing_summary",
            })
            inventory.append(item)
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            item.update({"armed": False, "cycles": 0})
            data_quality_issues.append({
                "run": run_directory.name,
                "issue": "invalid_summary_json",
                "detail": str(exc),
            })
            inventory.append(item)
            continue

        rows = []
        json_errors = []
        if cycles_path.exists() and cycles_path.stat().st_size > 0:
            rows, json_errors = _load_rows(cycles_path)
        armed = bool(summary.get("publish_enabled", False) and rows)
        item.update({
            "armed": armed,
            "cycles": len(rows),
            "publish_enabled": bool(summary.get("publish_enabled", False)),
            "goal_stop_triggered": bool(summary.get(
                "goal_stop_triggered", False
            )),
        })
        inventory.append(item)
        for error in json_errors:
            data_quality_issues.append({
                "run": run_directory.name,
                "issue": "invalid_cycle_json",
                **error,
            })
        for row_index, row in enumerate(rows):
            missing = sorted(REQUIRED_CYCLE_KEYS - set(row))
            if missing:
                data_quality_issues.append({
                    "run": run_directory.name,
                    "issue": "missing_cycle_keys",
                    "row_index": row_index,
                    "keys": missing,
                })
        if not armed:
            continue

        armed_runs += 1
        armed_cycles += len(rows)
        reached_runs += int(bool(summary.get("goal_stop_triggered", False)))
        run_report = _replay_run(
            rows,
            summary,
            action_spec,
            guard_config,
            planner_config,
        )
        run_reports[run_directory.name] = run_report
        for label, values in run_report["scenarios"].items():
            scenario_cycles[label] += int(values["cycles"])
            scenario_episodes[label] += int(values["episodes"])
        for label, examples in run_report["violations"].items():
            all_violation_counts[label] += len(examples)
        performance_totals.update(run_report["performance"])

    behavior_checks = {
        "action_bounds": all_violation_counts["action_bounds"] == 0,
        "protected_front_never_advances": (
            all_violation_counts["protected_front_advance"] == 0
        ),
        "crossings_turn_opposite_recorded_lateral_motion": (
            all_violation_counts["crossing_wrong_turn_side"] == 0
        ),
        "rear_only_force_forward": (
            all_violation_counts["rear_force_forward"] == 0
        ),
        "rear_only_path_preserves_forward": (
            all_violation_counts["rear_path_continuity"] == 0
        ),
        "rear_only_turn_side_is_stable": (
            all_violation_counts["rear_turn_sign_flip"] == 0
            and all_violation_counts["rear_goal_steer_wrong_side"] == 0
        ),
        "dynamic_path_never_rezeros_authorized_forward_motion": (
            all_violation_counts["dynamic_path_zero_override"] == 0
        ),
        "dynamic_spin_is_bounded": (
            all_violation_counts["unbounded_dynamic_spin"] == 0
        ),
        "cleared_rejoin_never_sustains_negative_goal_progress": (
            all_violation_counts[
                "sustained_negative_goal_rejoin_progress"
            ] == 0
        ),
        "goal_behind_turn_side_is_stable": (
            all_violation_counts["goal_behind_turn_sign_flip"] == 0
        ),
        "goal_behind_turn_is_bounded": all(
            int(values["maximum_goal_behind_turn_cycles"]) <= 30
            for values in run_reports.values()
        ),
        "post_retry_reverse_is_bounded": (
            all_violation_counts[
                "post_retry_reverse_after_exhaustion"
            ] == 0
            and all_violation_counts[
                "invalid_post_retry_side_forward"
            ] == 0
        ),
    }
    # Compact logs do not retain the exact Pi 20 Hz command-phase alignment.
    # Permit at most three additional replay flips per thousand source cycles;
    # this is a temporal-reconstruction tolerance, not a safety exemption.
    moving_turn_flip_tolerance = max(1, int(math.ceil(0.003 * armed_cycles)))
    performance_totals["moving_turn_sign_flip_tolerance"] = (
        moving_turn_flip_tolerance
    )
    report = {
        "schema": "recorded_real_robot_day_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "date_prefix": date_prefix,
        "runs_root": str(Path(runs_root).resolve()),
        "format": {
            "summary": "JSON hierarchical metadata",
            "cycles": "JSON Lines; one temporally ordered controller cycle per line",
            "config": "YAML resolved runtime configuration",
        },
        "inventory_summary": {
            "run_directories": len(inventory),
            "armed_runs": armed_runs,
            "armed_cycles": armed_cycles,
            "goal_reached_runs": reached_runs,
            "non_armed_or_empty_runs": len(inventory) - armed_runs,
            "total_input_bytes": total_bytes,
        },
        "data_quality": {
            "issue_count": len(data_quality_issues),
            "issues": data_quality_issues,
        },
        "scenario_coverage": {
            label: {
                "cycles": scenario_cycles[label],
                "episodes": scenario_episodes[label],
            }
            for label in sorted(scenario_cycles)
        },
        "behavior_checks": behavior_checks,
        "performance_summary": dict(performance_totals),
        "performance_checks": {
            "single_cycle_stop_pulses_reduced": bool(
                performance_totals["replay_single_cycle_stop_pulses"]
                <= performance_totals["recorded_single_cycle_stop_pulses"]
            ),
            "moving_turn_sign_flips_within_replay_tolerance": bool(
                performance_totals[
                    "replay_moving_turn_sign_flips"
                ]
                <= performance_totals[
                    "recorded_moving_turn_sign_flips"
                ] + moving_turn_flip_tolerance
            ),
            "applied_single_cycle_stop_pulses_not_increased": bool(
                performance_totals[
                    "replay_applied_single_cycle_stop_pulses"
                ]
                <= performance_totals[
                    "recorded_applied_single_cycle_stop_pulses"
                ]
            ),
        },
        "violation_counts": dict(all_violation_counts),
        "data_quality_complete": not data_quality_issues,
        "passed": bool(all(behavior_checks.values())),
        "inventory": inventory,
        "armed_run_reports": run_reports,
    }
    return report


def _markdown(report):
    summary = report["inventory_summary"]
    scenarios = report["scenario_coverage"]
    lines = [
        f"# {report['date_prefix']} real-robot full-day audit",
        "",
        "## Metadata",
        "",
        f"- Generated (UTC): `{report['generated_utc']}`",
        f"- Input root: `{report['runs_root']}`",
        f"- Input size: `{summary['total_input_bytes'] / (1024 ** 2):.2f} MiB`",
        "- Format: JSON summaries, JSONL ordered control cycles, YAML configs",
        "",
        "## Basic information and data quality",
        "",
        f"- Run directories: **{summary['run_directories']}**",
        f"- Armed physical runs: **{summary['armed_runs']}**",
        f"- Armed control cycles: **{summary['armed_cycles']}**",
        f"- Goal-reached runs: **{summary['goal_reached_runs']}**",
        f"- Warm-up/startup/empty runs: **{summary['non_armed_or_empty_runs']}**",
        f"- Structural data issues: **{report['data_quality']['issue_count']}**",
        "",
        "## Scenario coverage",
        "",
        "| Scenario | Episodes | Cycles |",
        "|---|---:|---:|",
    ]
    for label, values in scenarios.items():
        lines.append(
            f"| {label} | {values['episodes']} | {values['cycles']} |"
        )
    lines.extend([
        "",
        "## Current-code sequential replay checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ])
    for label, passed in report["behavior_checks"].items():
        lines.append(f"| {label} | {'PASS' if passed else 'FAIL'} |")
    for label, passed in report["performance_checks"].items():
        lines.append(f"| {label} | {'PASS' if passed else 'FAIL'} |")
    performance = report["performance_summary"]
    lines.extend([
        "",
        "### Continuity counters",
        "",
        f"- Single-cycle stop pulses: {performance.get('recorded_single_cycle_stop_pulses', 0)} recorded -> {performance.get('replay_single_cycle_stop_pulses', 0)} replayed",
        f"- Moving turn-sign flips: {performance.get('recorded_moving_turn_sign_flips', 0)} recorded -> {performance.get('replay_moving_turn_sign_flips', 0)} replayed",
        f"- Temporal-slowdown path authority: {performance.get('temporal_slowdown_path_authority_cycles', 0)} cycles",
        f"- Goal-behind safe reverse preserved: {performance.get('goal_behind_reverse_rejoin_cycles', 0)} cycles",
        f"- Goal-behind unsupported forward suppressed: {performance.get('goal_behind_turn_rejoin_cycles', 0)} cycles",
        f"- Pi-applied single-cycle stop pulses: {performance.get('recorded_applied_single_cycle_stop_pulses', 0)} recorded -> {performance.get('replay_applied_single_cycle_stop_pulses', 0)} replayed",
        f"- Pi-applied moving turn-sign flips: {performance.get('recorded_applied_moving_turn_sign_flips', 0)} recorded -> {performance.get('replay_applied_moving_turn_sign_flips', 0)} replayed",
    ])
    lines.extend([
        "",
        "## Armed run summary",
        "",
        "| Run | Cycles | Goal | Final m | Forecast valid | Scenarios | Violations |",
        "|---|---:|---|---:|---:|---|---|",
    ])
    for run, values in report["armed_run_reports"].items():
        scenario_text = ", ".join(
            f"{key}:{item['episodes']}"
            for key, item in values["scenarios"].items()
        ) or "none"
        violations = ", ".join(values["violations"]) or "none"
        distance = values["final_goal_distance_m"]
        distance_text = "n/a" if distance is None else f"{distance:.2f}"
        lines.append(
            f"| {run} | {values['completed_cycles']} | "
            f"{'yes' if values['goal_stop_triggered'] else 'no'} | "
            f"{distance_text} | "
            f"{values['prediction'].get('forecast_valid_cycles', 0)} | "
            f"{scenario_text} | {violations} |"
        )
    lines.extend([
        "",
        "## Key findings",
        "",
        "- The audit replays the current safety arbiter and deployment path supervisor in temporal order; it does not synthesize unrecorded LiDAR geometry.",
        "- Warm-up, connection-failure and zero-cycle directories are retained in the inventory but excluded from behavior scoring.",
        "- Rear-only forward escape, crossing-side selection, protected-front stopping, downstream path authority and bounded turning are checked independently.",
        "",
        "## Recommendations",
        "",
        "- Keep this full-day replay as a pre-deployment regression alongside the focused counterfactual close-range replay.",
        "- Tomorrow repeat frontal, oblique-rear and crowd-release trials because raw point payloads and physical contact outcomes are intentionally absent from compact logs.",
        "- Treat an individual scenario label as a motion/sector classification, not an operator-supplied ground-truth annotation.",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "codex_tmp" / "real_robot_cuda_runs",
    )
    parser.add_argument(
        "--weight-root",
        type=Path,
        default=PROJECT_ROOT / "codex_tmp" / "pi5_release_staging_v2",
    )
    parser.add_argument("--date-prefix", default="20260804")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    report = audit(args.runs_root, args.weight_root, args.date_prefix)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _markdown(report), encoding="utf-8"
        )
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
