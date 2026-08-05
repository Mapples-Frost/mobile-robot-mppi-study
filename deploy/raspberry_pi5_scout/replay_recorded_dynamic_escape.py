#!/usr/bin/env python3
"""Counterfactual replay of the two 2026-08-04 physical failure records.

The armed logs intentionally omit the raw 720-point cloud, so close-range
reverse is replayed under both explicit rear-clear and rear-blocked evidence.
Everything else (pose, tracker velocity, front guard, planner context and old
commands) comes from the recorded cycle.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from deploy.raspberry_pi5_scout.build_pi5_full_config import (
    build_pi5_full_config,
)
from deploy.raspberry_pi5_scout.run_remote_cuda_full import (
    _DynamicPathGuardSupervisor,
    _immediate_translation_stop_requested,
)
from mobile_robot_mppi.core.spaces import action_spec_from_config
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.planning.mppi import MppiController
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rows(run_directory):
    with (Path(run_directory) / "cycles.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _direction_context(row, planner_config):
    tracker = dict(row["diagnostics"]["tracker"])
    old_planner = dict(row["diagnostics"]["planner"])
    theta = float(row["pose"][2])
    motion = np.asarray((
        tracker.get("measurement_velocity_x_mps", 0.0),
        tracker.get("measurement_velocity_y_mps", 0.0),
    ), dtype=np.float64)
    direction, longitudinal, lateral, fraction = (
        MppiController._forward_lateral_countermotion_direction(
            motion,
            theta,
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
    context = dict(old_planner)
    context.update({
        "probabilistic_obstacle_motion_longitudinal_body_mps": longitudinal,
        "probabilistic_obstacle_motion_lateral_body_mps": lateral,
        "probabilistic_obstacle_motion_lateral_fraction": fraction,
    })
    if direction is None:
        # Direct approaches deliberately delegate passage side to measured
        # left/right clearance in the physical arbiter.
        heading_error = 0.0
        context.update({
            "probabilistic_obstacle_forward_lateral_countermotion_applied": False,
            "probabilistic_obstacle_preferred_escape_heading_error_rad": 0.0,
        })
    else:
        preferred_heading = float(np.arctan2(direction[1], direction[0]))
        heading_error = float(np.arctan2(
            np.sin(preferred_heading - theta),
            np.cos(preferred_heading - theta),
        ))
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
                "recorded_measurement_forward_lateral_countermotion"
            ),
        })
    return context, {
        "human_motion_world_mps": motion.tolist(),
        "human_lateral_body_mps": float(lateral),
        "human_longitudinal_body_mps": float(longitudinal),
        "human_lateral_fraction": float(fraction),
        "preferred_heading_error_rad": float(heading_error),
    }


def _front_and_rear_points(guard, rear_range):
    bearing = float(guard.get("dynamic_obstacle_bearing_rad", 0.0) or 0.0)
    front_range = float(guard.get("min_front_range", 0.45) or 0.45)
    return (
        {
            "base_angle": bearing,
            "range": front_range,
            "x": front_range * math.cos(bearing),
            "y": front_range * math.sin(bearing),
        },
        {
            "base_angle": math.pi,
            "range": float(rear_range),
            "x": -float(rear_range),
            "y": 0.0,
        },
    )


def _restore_rear_pass_guard(row):
    """Reconstruct the causal rear-only evidence omitted from compact logs."""

    guard = dict(row["diagnostics"]["safety"])
    if not bool(guard.get("rear_pass_through_active", False)):
        return guard
    original_reason = str(guard.get(
        "directional_guard_original_reason", "near_body_hard_stop"
    ))
    bearing = float(guard.get(
        "rear_pass_through_bearing_rad",
        guard.get("dynamic_obstacle_bearing_rad", math.pi),
    ))
    guard["reason"] = original_reason
    guard["emergency_stop"] = original_reason in {
        "near_body_hard_stop", "temporal_collision_risk"
    }
    guard["should_slow_down"] = original_reason == "temporal_slowdown"
    if original_reason == "near_body_hard_stop":
        distance = float(guard.get("min_near_body_range", 0.30) or 0.30)
        guard["near_body_points"] = ({
            "base_angle": bearing,
            "range": distance,
            "x": distance * math.cos(bearing),
            "y": distance * math.sin(bearing),
        },)
    return guard


def replay(runs_root, weight_root):
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

    crossing_row = _rows(Path(runs_root) / "20260804_071115")[70]
    crossing_guard = dict(crossing_row["diagnostics"]["safety"])
    crossing_context, crossing_motion = _direction_context(
        crossing_row, planner_config
    )
    crossing_decision = ScanGuardArbiter(
        action_spec, guard_config
    ).arbitrate(
        ControlCommand(crossing_row["proposed"]),
        crossing_guard,
        crossing_context,
    )
    crossing_supervisor = _DynamicPathGuardSupervisor()
    crossing_v, crossing_path = crossing_supervisor.apply(
        *crossing_row["pose"],
        5.0,
        2.0,
        crossing_decision.executed_control.v,
        crossing_decision.reason,
        emergency_stop=bool(
            crossing_decision.diagnostics.get("emergency_stop", False)
        ),
        proposed_omega=crossing_decision.executed_control.omega,
        maximum_omega_radps=0.6,
        hazard_active=True,
    )
    crossing_omega = crossing_decision.executed_control.omega
    crossing_override = crossing_path.get("commanded_omega_override_radps")
    if crossing_override is not None:
        crossing_omega = float(crossing_override)
    crossing_opposite = bool(
        crossing_motion["human_lateral_body_mps"] * crossing_omega < 0.0
    )

    frontal_rows = _rows(Path(runs_root) / "20260804_071201")
    frontal_row = frontal_rows[85]
    frontal_context, frontal_motion = _direction_context(
        frontal_row, planner_config
    )
    frontal_results = {}
    for label, rear_range in (("rear_clear", 2.0), ("rear_blocked", 0.45)):
        arbiter = ScanGuardArbiter(action_spec, guard_config)
        guard = dict(frontal_row["diagnostics"]["safety"])
        guard["raw_points_base"] = _front_and_rear_points(
            guard, rear_range
        )
        decisions = [
            arbiter.arbitrate(
                ControlCommand(frontal_row["proposed"]),
                guard,
                frontal_context,
            )
            for _ in range(5)
        ]
        fifth = decisions[-1]
        supervisor = _DynamicPathGuardSupervisor()
        output_v, path = supervisor.apply(
            *frontal_row["pose"],
            5.0,
            2.0,
            fifth.executed_control.v,
            fifth.reason,
            emergency_stop=bool(
                fifth.diagnostics.get("emergency_stop", False)
            ),
            proposed_omega=fifth.executed_control.omega,
            maximum_omega_radps=0.6,
            hazard_active=True,
        )
        output_omega = fifth.executed_control.omega
        override = path.get("commanded_omega_override_radps")
        if override is not None:
            output_omega = float(override)
        frontal_results[label] = {
            "first_four_commands": [
                decision.executed_control.values.tolist()
                for decision in decisions[:4]
            ],
            "fifth_arbitrated": fifth.executed_control.values.tolist(),
            "fifth_commanded": [float(output_v), float(output_omega)],
            "phase": fifth.diagnostics[
                "dynamic_escape_hard_stop_phase"
            ],
            "reverse_authorized": bool(fifth.diagnostics[
                "dynamic_escape_hard_stop_reverse_authorized"
            ]),
            "immediate_translation_stop": bool(
                _immediate_translation_stop_requested(
                    fifth.reason,
                    fifth.diagnostics,
                    output_v,
                )
            ),
        }

    # 071201 cycle 72 was changed from a fully steered reverse to a straight
    # +0.35 m/s deployment override.  Replay the supervisor after a live event.
    override_row = frontal_rows[72]
    override_supervisor = _DynamicPathGuardSupervisor()
    override_supervisor.apply(
        *frontal_rows[70]["pose"],
        5.0,
        2.0,
        0.35,
        "dynamic_active_escape",
        proposed_omega=-0.6,
        hazard_active=True,
    )
    new_v, new_path = override_supervisor.apply(
        *override_row["pose"],
        5.0,
        2.0,
        float(override_row["arbitrated"][0]),
        "front_clear",
        proposed_omega=float(override_row["arbitrated"][1]),
        hazard_active=True,
    )
    new_omega = float(override_row["arbitrated"][1])
    if new_path.get("commanded_omega_override_radps") is not None:
        new_omega = float(new_path["commanded_omega_override_radps"])

    # The 075007 run exposed two additional one-frame discontinuities.  The
    # close-range transaction paused after two turns and allowed +0.35 m/s on
    # cycle 69, then the deployment supervisor flipped an arbitrated reverse
    # into forward motion on cycle 80 when the forecast disappeared once.
    latest_rows = _rows(Path(runs_root) / "20260804_075007")
    latest_front_results = {}
    for label, rear_range in (("rear_clear", 2.0), ("rear_blocked", 0.45)):
        arbiter = ScanGuardArbiter(action_spec, guard_config)
        decisions = []
        for cycle in (67, 68, 69, 70, 71):
            row = latest_rows[cycle]
            guard = dict(row["diagnostics"]["safety"])
            guard["raw_points_base"] = _front_and_rear_points(
                guard, rear_range
            )
            decisions.append(arbiter.arbitrate(
                ControlCommand(row["proposed"]),
                guard,
                dict(row["diagnostics"]["planner"]),
            ))
        latest_front_results[label] = {
            "old_commands": [
                latest_rows[cycle]["commanded"]
                for cycle in (67, 68, 69, 70, 71)
            ],
            "new_commands": [
                decision.executed_control.values.tolist()
                for decision in decisions
            ],
            "phases": [
                decision.diagnostics[
                    "dynamic_escape_hard_stop_phase"
                ]
                for decision in decisions
            ],
            "held": [
                bool(decision.diagnostics.get(
                    "dynamic_escape_hard_stop_transaction_held", False
                ))
                for decision in decisions
            ],
            "fifth_reverse_authorized": bool(
                decisions[-1].diagnostics[
                    "dynamic_escape_hard_stop_reverse_authorized"
                ]
            ),
        }

    latest_supervisor = _DynamicPathGuardSupervisor()
    latest_deployment = []
    for cycle in range(67, 81):
        row = latest_rows[cycle]
        output_v, path = latest_supervisor.apply(
            *row["pose"],
            5.0,
            2.0,
            float(row["arbitrated"][0]),
            row["safety_reason"],
            emergency_stop=bool(
                row["diagnostics"]["safety"].get(
                    "emergency_stop", False
                )
            ),
            proposed_omega=float(row["arbitrated"][1]),
            maximum_omega_radps=0.6,
            selected_probability=float(row["diagnostics"]["planner"].get(
                "probabilistic_obstacle_maximum_step_probability", 0.0
            ) or 0.0),
            selected_probability_mass=float(row["diagnostics"]["planner"].get(
                "probabilistic_obstacle_probability_mass", 0.0
            ) or 0.0),
            hazard_active=bool(row["path_guard"].get(
                "fresh_hazard_active", False
            )),
        )
        output_omega = float(row["arbitrated"][1])
        if path.get("commanded_omega_override_radps") is not None:
            output_omega = float(path["commanded_omega_override_radps"])
        latest_deployment.append({
            "cycle": cycle,
            "old_arbitrated": row["arbitrated"],
            "old_commanded": row["commanded"],
            "new_commanded": [float(output_v), output_omega],
            "new_reason": path["reason"],
        })

    # The 085438 frontal approach selected the clearer left passage at cycle
    # 43, then two low-lateral-speed direction refreshes flipped the arc at
    # cycles 46 and 53.  Replay the entire pre-contact prefix sequentially so
    # stateful side commitment and one-frame TTC gaps are both exercised.
    latest_frontal_rows = _rows(Path(runs_root) / "20260804_085438")
    latest_frontal_arbiter = ScanGuardArbiter(action_spec, guard_config)
    latest_frontal_prefix = []
    for cycle in range(43, 59):
        row = latest_frontal_rows[cycle]
        decision = latest_frontal_arbiter.arbitrate(
            ControlCommand(row["proposed"]),
            dict(row["diagnostics"]["safety"]),
            dict(row["diagnostics"]["planner"]),
        )
        latest_frontal_prefix.append({
            "cycle": cycle,
            "old_commanded": row["commanded"],
            "new_arbitrated": decision.executed_control.values.tolist(),
            "refresh_requested": bool(decision.diagnostics.get(
                "dynamic_escape_prediction_direction_refresh_requested", False
            )),
            "refresh_rejected": bool(decision.diagnostics.get(
                "dynamic_escape_prediction_direction_refresh_rejected", False
            )),
            "coast_direction_locked": bool(decision.diagnostics.get(
                "dynamic_escape_coast_direction_locked", False
            )),
        })

    # 091739 is the many-person enclosure.  Freeze the recorded planner,
    # bearing and side-clearance sequence, then counterfactually keep the rear
    # blocked through the long stall and open it at cycle 416.  The transaction
    # must turn only once, wait without spinning, and take the newly open rear
    # exit without replaying the turn phase.
    crowd_rows = _rows(Path(runs_root) / "20260804_091739")
    crowd_arbiter = ScanGuardArbiter(action_spec, guard_config)
    crowd_replay = []
    for cycle in range(382, 421):
        row = crowd_rows[cycle]
        guard = dict(row["diagnostics"]["safety"])
        guard["raw_points_base"] = _front_and_rear_points(
            guard, 0.45 if cycle < 416 else 2.0
        )
        decision = crowd_arbiter.arbitrate(
            ControlCommand(row["proposed"]),
            guard,
            dict(row["diagnostics"]["planner"]),
        )
        crowd_replay.append({
            "cycle": cycle,
            "old_commanded": row["commanded"],
            "new_arbitrated": decision.executed_control.values.tolist(),
            "phase": decision.diagnostics[
                "dynamic_escape_hard_stop_phase"
            ],
            "direction_refresh_requested": bool(
                decision.diagnostics.get(
                    "dynamic_escape_prediction_direction_refresh_requested",
                    False,
                )
            ),
            "hard_stop_direction_refresh_applied": bool(
                decision.diagnostics.get(
                    "dynamic_escape_hard_stop_direction_refresh_applied",
                    False,
                )
            ),
            "rear_retry_started": bool(decision.diagnostics.get(
                "dynamic_escape_hard_stop_rear_clear_retry_started", False
            )),
            "reverse_remaining": int(decision.diagnostics.get(
                "dynamic_escape_hard_stop_reverse_remaining", 0
            )),
            "rear_wait_remaining": int(decision.diagnostics.get(
                "dynamic_escape_hard_stop_rear_blocked_wait_remaining", 0
            )),
        })

    # 092133 is the oblique/rear approach.  Compact logs omit point payloads,
    # so reconstruct only the already-recorded causal rear evidence.  Replay
    # both the arbiter and deployment path supervisor: neither layer may turn
    # a rear-only escape into a zero-speed path-deviation stop.
    rear_rows = _rows(Path(runs_root) / "20260804_092133")
    rear_arbiter = ScanGuardArbiter(action_spec, guard_config)
    rear_supervisor = _DynamicPathGuardSupervisor()
    rear_replay = []
    for cycle in range(224, 253):
        row = rear_rows[cycle]
        guard = _restore_rear_pass_guard(row)
        decision = rear_arbiter.arbitrate(
            ControlCommand(row["proposed"]),
            guard,
            dict(row["diagnostics"]["planner"]),
        )
        output_v, path = rear_supervisor.apply(
            *row["pose"],
            5.0,
            0.0,
            decision.executed_control.v,
            decision.reason,
            emergency_stop=bool(
                decision.diagnostics.get("emergency_stop", False)
            ),
            proposed_omega=decision.executed_control.omega,
            maximum_omega_radps=0.6,
            selected_probability=float(row["diagnostics"]["planner"].get(
                "probabilistic_obstacle_maximum_step_probability", 0.0
            ) or 0.0),
            selected_probability_mass=float(
                row["diagnostics"]["planner"].get(
                    "probabilistic_obstacle_probability_mass", 0.0
                ) or 0.0
            ),
            hazard_active=bool(row["path_guard"].get(
                "fresh_hazard_active", False
            )),
        )
        rear_replay.append({
            "cycle": cycle,
            "old_proposed": row["proposed"],
            "old_commanded": row["commanded"],
            "old_rear_pass": bool(
                row["diagnostics"]["safety"].get(
                    "rear_pass_through_active", False
                )
            ),
            "new_reason": decision.reason,
            "new_arbitrated": decision.executed_control.values.tolist(),
            "new_path_v": float(output_v),
            "new_path_reason": str(path["reason"]),
            "force_forward_ready": bool(decision.diagnostics.get(
                "rear_pass_through_force_forward_ready", False
            )),
        })

    report = {
        "schema": "recorded_dynamic_escape_replay_v5",
        "crossing_071115_cycle_70": {
            **crossing_motion,
            "old_commanded": crossing_row["commanded"],
            "new_arbitrated": crossing_decision.executed_control.values.tolist(),
            "new_commanded": [float(crossing_v), float(crossing_omega)],
            "turn_source": crossing_decision.diagnostics[
                "dynamic_escape_geometric_turn_source"
            ],
            "opposite_lateral_motion": crossing_opposite,
        },
        "frontal_071201_cycle_85": {
            **frontal_motion,
            "old_commanded": frontal_row["commanded"],
            "counterfactuals": frontal_results,
        },
        "deployment_override_071201_cycle_72": {
            "old_arbitrated": override_row["arbitrated"],
            "old_commanded": override_row["commanded"],
            "new_commanded": [float(new_v), float(new_omega)],
            "new_reason": new_path["reason"],
        },
        "latest_075007_front_transaction": latest_front_results,
        "latest_075007_deployment_cycle_80": latest_deployment[-1],
        "latest_085438_frontal_prefix": latest_frontal_prefix,
        "crowd_091739_cycles_382_420": crowd_replay,
        "rear_092133_cycles_224_252": rear_replay,
    }
    checks = {
        "crossing_turns_opposite_human": crossing_opposite,
        "crossing_not_overridden": bool(
            np.allclose(
                report["crossing_071115_cycle_70"]["new_arbitrated"],
                report["crossing_071115_cycle_70"]["new_commanded"],
            )
        ),
        "front_clear_reverses_after_four_turns": bool(
            frontal_results["rear_clear"]["fifth_commanded"][0] < 0.0
            and not frontal_results["rear_clear"][
                "immediate_translation_stop"
            ]
        ),
        "front_blocked_never_reverses": bool(
            frontal_results["rear_blocked"]["fifth_commanded"][0] == 0.0
            and frontal_results["rear_blocked"]["immediate_translation_stop"]
        ),
        "deployment_preserves_reverse_steering": bool(
            new_v < 0.0 and abs(new_omega) > 0.1
        ),
        "latest_front_runs_four_continuous_turns": bool(
            all(
                command[0] == 0.0
                for command in latest_front_results[
                    "rear_clear"
                ]["new_commands"][:4]
            )
        ),
        "latest_front_clear_then_reverses": bool(
            latest_front_results["rear_clear"]["new_commands"][4][0] < 0.0
            and latest_front_results[
                "rear_clear"
            ]["fifth_reverse_authorized"]
        ),
        "latest_front_blocked_never_reverses": bool(
            latest_front_results[
                "rear_blocked"
            ]["new_commands"][4][0] == 0.0
            and not latest_front_results[
                "rear_blocked"
            ]["fifth_reverse_authorized"]
        ),
        "latest_deployment_never_flips_reverse_to_forward": bool(
            latest_deployment[-1]["old_arbitrated"][0] < 0.0
            and latest_deployment[-1]["new_commanded"][0] < 0.0
            and np.sign(latest_deployment[-1]["old_arbitrated"][1])
            == np.sign(latest_deployment[-1]["new_commanded"][1])
        ),
        "latest_frontal_false_reversals_rejected": bool(
            all(
                latest_frontal_prefix[cycle - 43]["refresh_requested"]
                and latest_frontal_prefix[cycle - 43]["refresh_rejected"]
                for cycle in (46, 53)
            )
        ),
        "latest_frontal_prefix_keeps_one_passage_side": bool(
            all(
                item["new_arbitrated"][1] >= 0.0
                for item in latest_frontal_prefix
            )
            and any(
                item["new_arbitrated"][1] > 0.0
                for item in latest_frontal_prefix
            )
        ),
        "crowd_turn_transaction_runs_once": bool(
            [
                item["cycle"]
                for item in crowd_replay
                if item["phase"] == "turn_in_place"
            ] == [382, 383, 384, 385]
        ),
        "crowd_track_switch_never_restarts_hard_stop_turn": bool(
            any(
                item["direction_refresh_requested"]
                for item in crowd_replay
            )
            and not any(
                item["hard_stop_direction_refresh_applied"]
                for item in crowd_replay
            )
        ),
        "crowd_blocked_wait_does_not_spend_reverse_budget": bool(
            all(
                item["reverse_remaining"] == 12
                for item in crowd_replay
                if 386 <= item["cycle"] < 391
            )
            and crowd_replay[391 - 382]["phase"]
            == "rear_blocked_wait_exhausted"
        ),
        "crowd_completed_transaction_hold_is_finite": bool(
            all(
                np.allclose(item["new_arbitrated"], (0.0, 0.0))
                for item in crowd_replay
                if 391 <= item["cycle"] < 416
            )
            and [
                item["cycle"]
                for item in crowd_replay
                if item["phase"] == "bounded_transaction_complete"
            ] == [392, 393]
            and all(
                item["phase"] == "completed_fail_closed"
                for item in crowd_replay
                if 394 <= item["cycle"] < 416
            )
        ),
        "crowd_open_rear_retries_reverse_without_new_turn": bool(
            crowd_replay[416 - 382]["rear_retry_started"]
            and crowd_replay[416 - 382]["new_arbitrated"][0] < 0.0
            and crowd_replay[416 - 382]["phase"]
            == "rear_clear_retry_reverse"
        ),
        "rear_only_cycles_force_continuous_forward_escape": bool(
            all(
                item["new_reason"] == "rear_pass_through"
                and item["new_arbitrated"][0] >= 0.35 - 1.0e-12
                and item["new_path_v"] >= 0.35 - 1.0e-12
                and item["new_path_reason"] == "dynamic_authority"
                for item in rear_replay
                if item["old_rear_pass"]
            )
        ),
        "rear_scan_fragment_gaps_do_not_reactivate_path_stop": bool(
            all(
                rear_replay[cycle - 224]["new_path_v"] > 0.0
                for cycle in (236, 252)
            )
        ),
    }
    report["checks"] = checks
    report["passed"] = bool(all(checks.values()))
    if not report["passed"]:
        raise RuntimeError("recorded dynamic escape replay failed: %r" % checks)
    return report


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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = replay(args.runs_root, args.weight_root)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
