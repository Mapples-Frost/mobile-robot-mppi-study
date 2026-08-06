"""Mode-owned MPPI reference and admissible control authority.

This module is intentionally separate from perception and from hard safety.
The mode manager supplies a frozen semantic intent; MPPI optimizes against a
temporary bent reference; this adapter then rejects only commands that violate
the locked passage side.  The downstream safety arbiter retains the final
hard-stop and physical-limit veto.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.core.types import ControlCommand, PlanResult


_ACTIVE_PHASES = frozenset({
    "straight_crossing", "oblique_crossing", "frontal_approach", "rejoin"
})


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _wrap(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


@dataclass(frozen=True)
class EncounterControlConfig:
    enabled: bool = False
    maximum_forward_speed_mps: float = 0.50
    maximum_reverse_speed_mps: float = 0.30
    maximum_omega_radps: float = 0.60
    straight_crossing_speed_mps: float = 0.40
    oblique_crossing_speed_mps: float = 0.36
    frontal_approach_speed_mps: float = 0.32
    front_pass_speed_mps: float = 0.48
    rejoin_speed_mps: float = 0.36
    rejoin_alignment_speed_mps: float = 0.15
    heading_gain: float = 1.35
    heading_blend: float = 0.80
    minimum_turn_omega_radps: float = 0.16
    turn_deadband_rad: float = 0.06
    rejoin_alignment_heading_rad: float = 0.45
    # While rejoin is still laterally displaced, use the moving rejoin
    # waypoint bearing to bring the chassis back to the frozen goal line.  The
    # frozen goal-line heading is only sufficient after the lateral error is
    # already small; using it too early makes the robot drive parallel to the
    # goal indefinitely.
    rejoin_lateral_convergence_tolerance_m: float = 0.25
    # A frontal bypass waypoint can lie far ahead of the chassis.  If the
    # current yaw already points at that waypoint, pure waypoint pursuit
    # produces almost zero omega and the robot drives toward the person before
    # building lateral clearance.  Hold a bounded side-facing heading until a
    # modest lateral offset is achieved; after that, return to normal waypoint
    # pursuit so the detour cannot over-rotate.
    frontal_bypass_min_heading_rad: float = 0.50
    frontal_bypass_lateral_hold_m: float = 0.55
    rejoin_lateral_heading_gain: float = 1.80
    safe_probability_ceiling: float = 0.12
    safe_probability_mass_ceiling: float = 2.0
    frontal_reverse_trigger_m: float = 0.85
    frontal_reverse_speed_mps: float = 0.18
    frontal_reverse_minimum_rear_range_m: float = 0.90
    # A temporal TTC alarm is not by itself a geometric collision.  During an
    # admitted encounter, keep the mode-owned passage command when the live
    # front corridor is still open.  Geometric stops may only release along a
    # separately measured clear corridor, or retain zero translation.
    temporal_risk_defer_min_front_range_m: float = 0.75
    side_hard_stop_defer_min_clearance_m: float = 0.60
    side_hard_stop_turn_min_clearance_m: float = 0.75
    # When the front hard-stop beam is close but the encounter-owned side is
    # clearly open, a tiny forward creep lets the chassis complete the locked
    # bypass instead of pivoting in place.  This is deliberately stricter than
    # the turn-only release and is disabled by near-body evidence.
    side_hard_stop_turn_creep_min_front_range_m: float = 0.45
    side_hard_stop_turn_creep_min_clearance_m: float = 1.00
    side_hard_stop_turn_creep_speed_mps: float = 0.12
    control_prefix_steps: int = 3
    reference_lookahead_m: float = 1.0
    reference_corridor_half_width_m: float = 0.95

    def __post_init__(self) -> None:
        if self.maximum_forward_speed_mps <= 0.0:
            raise ValueError("encounter maximum forward speed must be positive")
        if self.maximum_reverse_speed_mps <= 0.0:
            raise ValueError("encounter maximum reverse speed must be positive")
        if self.maximum_omega_radps <= 0.0:
            raise ValueError("encounter maximum omega must be positive")
        if not 0.0 <= self.heading_blend <= 1.0:
            raise ValueError("encounter heading blend must be in [0,1]")
        if self.control_prefix_steps < 1:
            raise ValueError("encounter control prefix must be positive")


class EncounterReferenceAuthority:
    """Own a frozen temporary polyline while an encounter is active."""

    def __init__(self, config: EncounterControlConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._signature = None
        self._reference = None  # type: Optional[PolylineReference]

    @staticmethod
    def _point(value: Any) -> Optional[np.ndarray]:
        try:
            point = np.asarray(value, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            return None
        return point if np.isfinite(point).all() else None

    def select(
        self,
        base_reference: Any,
        diagnostics: Mapping[str, Any],
        pose: Sequence[float],
        goal: Sequence[float],
    ) -> Any:
        """Return the base reference or the current encounter polyline."""

        phase = str(diagnostics.get("encounter_phase", "idle"))
        if not self.config.enabled or phase not in _ACTIVE_PHASES:
            self.reset()
            return base_reference
        waypoint = self._point(diagnostics.get("encounter_temporary_waypoint"))
        start = self._point(diagnostics.get("encounter_entry_goal_origin"))
        if start is None:
            start = self._point(pose[:2])
        goal_point = self._point(goal)
        if waypoint is None or start is None or goal_point is None:
            self.reset()
            return base_reference
        # A late encounter can place the nominal clearance point beyond the
        # terminal goal.  Retain its lateral offset but clamp longitudinal
        # progress before the goal so the semantic route remains well posed.
        goal_vector = goal_point - start
        goal_distance = float(np.linalg.norm(goal_vector))
        if goal_distance > 1.0e-6:
            goal_direction = goal_vector / goal_distance
            lateral_direction = np.asarray(
                (-goal_direction[1], goal_direction[0]), dtype=np.float64
            )
            relative = waypoint - start
            longitudinal = min(
                float(np.dot(relative, goal_direction)),
                max(0.35, goal_distance - 0.25),
            )
            lateral = float(np.dot(relative, lateral_direction))
            waypoint = (
                start + max(0.35, longitudinal) * goal_direction
                + lateral * lateral_direction
            )
        if (
            float(np.linalg.norm(waypoint - start)) <= 0.10
            or float(np.linalg.norm(goal_point - waypoint)) <= 0.10
        ):
            self.reset()
            return base_reference
        signature = (
            phase,
            diagnostics.get("encounter_track_index"),
            diagnostics.get("encounter_strategy"),
            tuple(np.round(start, 4)),
            tuple(np.round(waypoint, 4)),
            tuple(np.round(goal_point, 4)),
        )
        if signature != self._signature:
            tolerance = _finite(
                getattr(base_reference, "tolerance", None),
                _finite(getattr(base_reference, "position_tolerance", None), 0.25),
            )
            footprint = getattr(base_reference, "footprint_radius", None)
            footprint = (
                0.25 if footprint is None else _finite(footprint, 0.25)
            )
            corridor = max(
                self.config.reference_corridor_half_width_m,
                _finite(
                    getattr(base_reference, "corridor_half_width", None),
                    self.config.reference_corridor_half_width_m,
                ),
            )
            if corridor <= footprint:
                corridor = footprint + 0.20
            self._reference = PolylineReference(
                (start, waypoint, goal_point),
                tolerance=tolerance,
                lookahead_distance=self.config.reference_lookahead_m,
                terminal_approach_distance=max(
                    self.config.reference_lookahead_m, 0.9
                ),
                projection_backtrack_distance=1.5,
                projection_forward_distance=3.0,
                corridor_half_width=corridor,
                footprint_radius=footprint,
            )
            self._signature = signature
        return self._reference


class EncounterControlAuthority:
    """Constrain MPPI output to the mode-owned side before hard safety."""

    def __init__(self, config: EncounterControlConfig):
        self.config = config

    def planning_context(
        self, diagnostics: Mapping[str, Any]
    ) -> Dict[str, Any]:
        phase = str(diagnostics.get("encounter_phase", "idle"))
        active = bool(self.config.enabled and phase in _ACTIVE_PHASES)
        return {
            "encounter_control_authoritative": active,
            "encounter_control_phase": phase,
            "encounter_control_track_index": diagnostics.get(
                "encounter_track_index"
            ),
            "encounter_control_strategy": diagnostics.get(
                "encounter_strategy", "none"
            ),
            "encounter_control_locked_steering_side": int(
                diagnostics.get("encounter_locked_steering_side", 0) or 0
            ),
            "encounter_control_inhibit_rear_pass": active,
            "encounter_control_inhibit_forward_passage": active,
            "encounter_control_inhibit_dynamic_escape": active,
            # While a semantic transaction owns direction, hard safety has a
            # veto but no alternative motion generator.  In particular it may
            # not replay the legacy turn/reverse escape transaction.
            "encounter_control_stop_only_hard_safety": active,
            "encounter_control_hard_safety_retained": True,
            "encounter_control_hard_stop_escape_motion_allowed": False,
        }

    def _target_speed(
        self, phase: str, strategy: str, heading_error: float
    ) -> float:
        if phase == "straight_crossing":
            speed = self.config.straight_crossing_speed_mps
        elif phase == "oblique_crossing":
            speed = self.config.oblique_crossing_speed_mps
        elif phase == "frontal_approach":
            speed = self.config.frontal_approach_speed_mps
        else:
            speed = (
                self.config.rejoin_alignment_speed_mps
                if abs(heading_error)
                >= self.config.rejoin_alignment_heading_rad
                else self.config.rejoin_speed_mps
            )
        if strategy == "front_pass" and phase != "rejoin":
            speed = max(speed, self.config.front_pass_speed_mps)
        return min(speed, self.config.maximum_forward_speed_mps)

    def apply(
        self,
        plan: PlanResult,
        diagnostics: Mapping[str, Any],
        pose: Sequence[float],
        guard_result: Mapping[str, Any],
    ) -> PlanResult:
        """Apply semantic admissibility while preserving hard-safety vetoes."""

        phase = str(diagnostics.get("encounter_phase", "idle"))
        if not self.config.enabled or phase not in _ACTIVE_PHASES:
            return plan
        source = dict(plan.diagnostics or {})
        output_diagnostics = dict(source)
        waypoint = EncounterReferenceAuthority._point(
            diagnostics.get("encounter_temporary_waypoint")
        )
        pose_values = np.asarray(pose, dtype=np.float64).reshape(3)
        emergency = bool(guard_result.get("emergency_stop", False))
        guard_reason = str(guard_result.get("reason", ""))
        front_range = _finite(
            guard_result.get("min_front_range"), float("nan")
        )
        near_body_count = int(guard_result.get("valid_near_body_count", 0) or 0)
        try:
            locked_side = int(
                diagnostics.get("encounter_locked_steering_side", 0) or 0
            )
        except (TypeError, ValueError):
            locked_side = 0
        side_key = (
            "min_left_side_range" if locked_side > 0
            else "min_right_side_range" if locked_side < 0
            else None
        )
        selected_side_clearance = _finite(
            guard_result.get(side_key), float("nan")
        ) if side_key is not None else float("nan")
        side_hard_stop_deferred = bool(
            emergency
            and guard_reason == "near_body_hard_stop"
            and math.isfinite(front_range)
            and front_range >= self.config.temporal_risk_defer_min_front_range_m
            and locked_side != 0
            and math.isfinite(selected_side_clearance)
            and selected_side_clearance
            >= self.config.side_hard_stop_defer_min_clearance_m
        )
        side_hard_stop_turn_only = bool(
            emergency
            and guard_reason in {"near_body_hard_stop", "hard_stop"}
            and locked_side != 0
            and math.isfinite(selected_side_clearance)
            and selected_side_clearance
            >= self.config.side_hard_stop_turn_min_clearance_m
            and not side_hard_stop_deferred
        )
        side_hard_stop_turn_creep = bool(
            side_hard_stop_turn_only
            and near_body_count <= 0
            and math.isfinite(front_range)
            and front_range >= self.config.side_hard_stop_turn_creep_min_front_range_m
            and math.isfinite(selected_side_clearance)
            and selected_side_clearance
            >= self.config.side_hard_stop_turn_creep_min_clearance_m
        )
        temporal_risk_deferred = bool(
            emergency
            and guard_reason == "temporal_collision_risk"
            and math.isfinite(front_range)
            and front_range >= self.config.temporal_risk_defer_min_front_range_m
            and near_body_count <= 0
            and not bool(guard_result.get("dynamic_obstacle_near_body_match", False))
        )
        if (
            temporal_risk_deferred
            or side_hard_stop_deferred
            or side_hard_stop_turn_only
        ):
            # The alarm is retained as a diagnostic, but it cannot replace an
            # admitted encounter's side/forward command when the live forward
            # and selected-side corridors both certify separation.
            emergency = False
        if waypoint is None or not np.isfinite(pose_values).all():
            output_diagnostics.update({
                "encounter_control_enabled": True,
                "encounter_control_applied": False,
                "encounter_control_reason": "missing_temporary_waypoint",
            })
            return PlanResult(
                plan.proposed_control,
                plan.control_sequence,
                plan.predicted_trajectory,
                output_diagnostics,
            )
        desired_heading = math.atan2(
            waypoint[1] - pose_values[1], waypoint[0] - pose_values[0]
        )
        waypoint_heading_error = _wrap(desired_heading - pose_values[2])
        heading_error = waypoint_heading_error
        frontal_lateral_hold_active = False
        frontal_lateral_heading_error = float("nan")
        rejoin_waypoint_convergence_active = False
        if (
            phase == "frontal_approach"
            and str(diagnostics.get("encounter_strategy", ""))
            in {"left_bypass", "right_bypass"}
        ):
            try:
                locked_side = int(
                    diagnostics.get("encounter_locked_steering_side", 0) or 0
                )
            except (TypeError, ValueError):
                locked_side = 0
            try:
                origin = np.asarray(
                    diagnostics.get("encounter_entry_goal_origin", ()),
                    dtype=np.float64,
                ).reshape(-1)
            except (TypeError, ValueError):
                origin = np.empty(0, dtype=np.float64)
            entry_heading = _finite(
                diagnostics.get("encounter_entry_goal_heading_rad"),
                float("nan"),
            )
            if (
                locked_side != 0
                and origin.size >= 2
                and np.isfinite(origin[:2]).all()
                and math.isfinite(entry_heading)
            ):
                lateral_direction = np.asarray(
                    (-math.sin(entry_heading), math.cos(entry_heading)),
                    dtype=np.float64,
                )
                lateral_offset = float(np.dot(
                    pose_values[:2] - origin[:2], lateral_direction
                ))
                if abs(lateral_offset) < self.config.frontal_bypass_lateral_hold_m:
                    lateral_target_heading = _wrap(
                        entry_heading
                        + float(locked_side)
                        * self.config.frontal_bypass_min_heading_rad
                    )
                    frontal_lateral_heading_error = _wrap(
                        lateral_target_heading - pose_values[2]
                    )
                    # Only strengthen a weak waypoint turn.  If the waypoint
                    # already requests a larger correction, preserve it.
                    if abs(frontal_lateral_heading_error) > abs(heading_error):
                        heading_error = frontal_lateral_heading_error
                        desired_heading = lateral_target_heading
                        frontal_lateral_hold_active = True
        if phase == "rejoin":
            # The moving lookahead is useful for lateral convergence, but its
            # bearing can become nearly tangent to the goal line while the
            # chassis is still yawed away from it.  In that state the old
            # controller believed it was aligned and kept driving parallel
            # to the goal.  Prefer the manager's frozen goal-line heading
            # error for yaw control; retain the waypoint for the reference
            # geometry and fall back to it for older diagnostics.
            rejoin_heading_error = _finite(
                diagnostics.get("encounter_rejoin_heading_error_rad"),
                float("nan"),
            )
            if math.isfinite(rejoin_heading_error):
                rejoin_cross_track = _finite(
                    diagnostics.get("encounter_rejoin_cross_track_m"),
                    float("nan"),
                )
                if (
                    math.isfinite(rejoin_cross_track)
                    and
                    abs(rejoin_cross_track)
                    > self.config.rejoin_lateral_convergence_tolerance_m
                ):
                    # Keep the waypoint bearing while the chassis is outside
                    # the goal-line corridor.  This is the lateral correction
                    # that the frozen goal-line heading cannot provide.
                    heading_error = waypoint_heading_error
                    rejoin_waypoint_convergence_active = True
                else:
                    heading_error = _wrap(rejoin_heading_error)
                    desired_heading = _wrap(pose_values[2] + heading_error)
        heading_gain = (
            self.config.rejoin_lateral_heading_gain
            if rejoin_waypoint_convergence_active
            else self.config.heading_gain
        )
        target_omega = float(np.clip(
            heading_gain * heading_error,
            -self.config.maximum_omega_radps,
            self.config.maximum_omega_radps,
        ))
        if abs(heading_error) >= self.config.turn_deadband_rad:
            target_omega = math.copysign(
                max(abs(target_omega), self.config.minimum_turn_omega_radps),
                heading_error,
            )
        else:
            target_omega = 0.0

        values = np.asarray(plan.proposed_control.values, dtype=np.float64).copy()
        original = values.copy()
        probability = _finite(source.get(
            "probabilistic_obstacle_maximum_step_probability"
        ), float("inf"))
        probability_mass = _finite(source.get(
            "probabilistic_obstacle_probability_mass"
        ), float("inf"))
        risk_safe = bool(
            not source.get("probabilistic_obstacle_hard_violation", True)
            and probability <= self.config.safe_probability_ceiling
            and probability_mass <= self.config.safe_probability_mass_ceiling
            and not emergency
            and not side_hard_stop_deferred
        )
        strategy = str(diagnostics.get("encounter_strategy", "none"))
        target_speed = self._target_speed(phase, strategy, heading_error)
        reverse_applied = False
        distance = _finite(diagnostics.get("encounter_distance_m"), float("inf"))
        rear_range = _finite(
            guard_result.get("min_rear_range"), float("nan")
        )
        reverse_available = bool(
            phase == "frontal_approach"
            and distance <= self.config.frontal_reverse_trigger_m
            and math.isfinite(rear_range)
            and rear_range >= self.config.frontal_reverse_minimum_rear_range_m
            and not emergency
            and not side_hard_stop_deferred
        )
        planner_reverse_rejected = False
        if emergency:
            # Preserve the independent scan veto, but hand it an unambiguous
            # stop.  Passing the planner's emergency-template reverse through
            # here allowed the downstream hard-stop escape state machine to
            # turn and back up even though the semantic mode owned the side.
            values[:2] = 0.0
            reason = "hard_safety_stop_only"
        else:
            if reverse_available:
                values[0] = max(
                    -self.config.frontal_reverse_speed_mps,
                    -self.config.maximum_reverse_speed_mps,
                )
                reverse_applied = True
            else:
                planner_reverse_rejected = bool(float(values[0]) < 0.0)
                # Risk remains observable and the final scan guard can still
                # slow or stop.  It cannot choose longitudinal direction once
                # the mode has admitted and locked an avoidance transaction.
                values[0] = target_speed
            proposed_omega = float(values[1])
            same_direction = bool(
                target_omega == 0.0
                or proposed_omega == 0.0
                or target_omega * proposed_omega > 0.0
            )
            if same_direction:
                values[1] = (
                    self.config.heading_blend * target_omega
                    + (1.0 - self.config.heading_blend) * proposed_omega
                )
            else:
                values[1] = target_omega
            values[0] = float(np.clip(
                values[0],
                -self.config.maximum_reverse_speed_mps,
                self.config.maximum_forward_speed_mps,
            ))
            values[1] = float(np.clip(
                values[1],
                -self.config.maximum_omega_radps,
                self.config.maximum_omega_radps,
            ))
            reason = (
                "frontal_clearance_reverse"
                if reverse_applied
                else "mode_owned_temporary_reference"
            )
        if side_hard_stop_turn_only:
            values[0] = (
                min(
                    target_speed,
                    self.config.side_hard_stop_turn_creep_speed_mps,
                )
                if side_hard_stop_turn_creep
                else 0.0
            )
            values[1] = (
                (1.0 if locked_side > 0 else -1.0)
                * max(abs(values[1]), self.config.minimum_turn_omega_radps)
            )
            values[1] = float(np.clip(
                values[1],
                -self.config.maximum_omega_radps,
                self.config.maximum_omega_radps,
            ))
            reverse_applied = False
            reason = (
                "side_hard_stop_turn_creep"
                if side_hard_stop_turn_creep
                else "side_hard_stop_turn_only"
            )

        applied = bool(not np.allclose(
            values, original, rtol=0.0, atol=1.0e-12
        ))
        sequence = np.asarray(plan.control_sequence, dtype=np.float64).copy()
        if applied and sequence.ndim == 2 and sequence.shape[1] >= 2:
            prefix = min(self.config.control_prefix_steps, len(sequence))
            sequence[:prefix, :2] = values[:2]
        output_diagnostics.update({
            "encounter_control_enabled": True,
            "encounter_control_authoritative": True,
            "encounter_control_applied": applied,
            "encounter_control_reason": reason,
            "encounter_control_phase": phase,
            "encounter_control_strategy": strategy,
            "encounter_control_desired_heading_rad": float(desired_heading),
            "encounter_control_heading_error_rad": float(heading_error),
            "encounter_control_target_v_mps": float(target_speed),
            "encounter_control_target_omega_radps": float(target_omega),
            "encounter_control_frontal_lateral_hold_active": (
                frontal_lateral_hold_active
            ),
            "encounter_control_frontal_lateral_heading_error_rad": (
                frontal_lateral_heading_error
            ),
            "encounter_control_rejoin_waypoint_convergence_active": (
                rejoin_waypoint_convergence_active
            ),
            "encounter_control_original_v_mps": float(original[0]),
            "encounter_control_original_omega_radps": float(original[1]),
            "encounter_control_selected_v_mps": float(values[0]),
            "encounter_control_selected_omega_radps": float(values[1]),
            "encounter_control_risk_safe": risk_safe,
            "encounter_control_temporal_risk_deferred": temporal_risk_deferred,
            "encounter_control_side_hard_stop_deferred": side_hard_stop_deferred,
            "encounter_control_side_hard_stop_turn_only": side_hard_stop_turn_only,
            "encounter_control_side_hard_stop_turn_creep": side_hard_stop_turn_creep,
            "encounter_control_original_emergency_stop": bool(
                guard_result.get("emergency_stop", False)
            ),
            "encounter_control_reverse_available": reverse_available,
            "encounter_control_reverse_applied": reverse_applied,
            "encounter_control_planner_reverse_rejected": bool(
                planner_reverse_rejected
            ),
            "encounter_control_motion_owner": (
                "hard_stop"
                if emergency
                else "explicit_frontal_reverse"
                if reverse_applied
                else "encounter"
            ),
            "encounter_control_hard_safety_pending": emergency,
            "encounter_control_predicted_trajectory_recomputed": False,
        })
        command = ControlCommand(
            values,
            timestamp=plan.proposed_control.timestamp,
            source="encounter_mode_authority_v1",
        )
        return PlanResult(
            command, sequence, plan.predicted_trajectory, output_diagnostics
        )


__all__ = [
    "EncounterControlAuthority",
    "EncounterControlConfig",
    "EncounterReferenceAuthority",
]
