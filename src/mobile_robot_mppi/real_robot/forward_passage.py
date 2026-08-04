"""Stateful real-robot admission of planner-vetted forward passage.

The core Full Proposed controller remains unchanged.  This deployment adapter
addresses one physical-only discontinuity: the emergency lattice may switch a
currently advancing robot to a minimum-risk reverse candidate even when a
low-risk continuation exists.  A continuation is admitted only after stable
measured forward motion and only when its complete committed prefix is forward
and steering-continuous.  That immutable prefix is consumed exactly once while
fresh risk diagnostics remain below the frozen ceilings.  The unchanged safety
arbiter still owns the final command.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from mobile_robot_mppi.core.types import ControlCommand, PlanResult


_EMERGENCY_KIND = "temporal_scan_vetted_emergency_candidate"
_TRANSACTION_BLOCKS = (
    "probabilistic_obstacle_traversal_commit_active",
    "probabilistic_obstacle_traversal_retreat_requested",
    "probabilistic_obstacle_traversal_rearm_pending",
    "probabilistic_obstacle_traversal_exit_deadline_retreat_escape_transaction_active",
    "probabilistic_obstacle_traversal_post_center_temporal_escape_latched",
    "probabilistic_obstacle_traversal_post_center_forward_exit_commit_requested",
    "probabilistic_obstacle_traversal_uncommitted_temporal_staging_hold_requested",
)
_SAFETY_RELEASE_REASONS = {
    "near_body_hard_stop",
    "dynamic_recovery_align",
    "dynamic_recovery_advance",
}


@dataclass(frozen=True)
class ForwardPassageConfig:
    """Frozen real-robot continuation and transaction limits."""

    enabled: bool = False
    maximum_probability: float = 0.08
    maximum_probability_mass: float = 1.50
    minimum_forward_speed_mps: float = 1.0e-6
    minimum_continuation_speed_mps: float = 0.05
    maximum_steering_change_radps: float = 0.35
    minimum_forward_feedback_steps: int = 3
    commit_steps: int = 12
    reentry_cooldown_steps: int = 4
    maximum_feedback_mismatch_steps: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_probability <= 1.0:
            raise ValueError("maximum_probability must be in [0, 1]")
        if self.maximum_probability_mass < 0.0:
            raise ValueError("maximum_probability_mass must be non-negative")
        if self.minimum_forward_speed_mps < 0.0:
            raise ValueError("minimum_forward_speed_mps must be non-negative")
        if self.minimum_continuation_speed_mps <= 0.0:
            raise ValueError("minimum_continuation_speed_mps must be positive")
        if self.maximum_steering_change_radps < 0.0:
            raise ValueError("maximum_steering_change_radps must be non-negative")
        if self.minimum_forward_feedback_steps <= 0:
            raise ValueError("minimum_forward_feedback_steps must be positive")
        if self.commit_steps <= 0:
            raise ValueError("commit_steps must be positive")
        if self.reentry_cooldown_steps < 0:
            raise ValueError("reentry_cooldown_steps must be non-negative")
        if self.maximum_feedback_mismatch_steps <= 0:
            raise ValueError("maximum_feedback_mismatch_steps must be positive")


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _eligible_forward_candidates(
    trace: Iterable[Mapping[str, Any]], config: ForwardPassageConfig
) -> Tuple[Mapping[str, Any], ...]:
    admitted = []
    for candidate in trace:
        if not isinstance(candidate, Mapping):
            continue
        probability = _finite(candidate.get("maximum_probability"), np.inf)
        mass = _finite(candidate.get("probability_mass"), np.inf)
        speed = _finite(candidate.get("first_v"), -np.inf)
        sequence = _candidate_sequence(candidate)
        prefix = sequence[: config.commit_steps]
        prefix_is_forward = bool(
            len(prefix) >= config.commit_steps
            and np.all(np.isfinite(prefix))
            and np.all(prefix[:, 0] >= config.minimum_forward_speed_mps)
        )
        if (
            bool(candidate.get("eligible", False))
            and not bool(candidate.get("hard_violation", True))
            and speed >= config.minimum_forward_speed_mps
            and probability <= config.maximum_probability
            and mass <= config.maximum_probability_mass
            and prefix_is_forward
        ):
            admitted.append(candidate)
    return tuple(admitted)


def _candidate_sequence(candidate: Mapping[str, Any]) -> np.ndarray:
    sequence = np.asarray(candidate.get("sequence", ()), dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] < 2:
        sequence = np.asarray(
            [[candidate.get("first_v"), candidate.get("first_omega")]],
            dtype=np.float64,
        )
    return np.asarray(sequence[:, :2], dtype=np.float64)


def _candidate_order(candidate: Mapping[str, Any]) -> Tuple[float, float, float, int]:
    return (
        _finite(candidate.get("cost"), np.inf),
        _finite(candidate.get("maximum_probability"), np.inf),
        abs(_finite(candidate.get("first_omega"), np.inf)),
        int(candidate.get("index", 2**31 - 1)),
    )


class ForwardPassageController:
    """Own a short, freshly risk-gated forward-continuation transaction."""

    def __init__(self, config: ForwardPassageConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_executed_v = 0.0
        self._last_executed_omega = 0.0
        self._last_commanded_v = 0.0
        self._last_commanded_omega = 0.0
        self._forward_feedback_streak = 0
        self._feedback_mismatch_streak = 0
        self._cooldown_remaining = 0
        self._pending_release_reason: Optional[str] = None
        self._sequence: Optional[np.ndarray] = None
        self._sequence_index = 0
        self._remaining = 0

    @property
    def active(self) -> bool:
        return bool(self._sequence is not None and self._remaining > 0)

    def observe_feedback(self, command: ControlCommand) -> None:
        """Feed measured chassis motion into continuation admission."""
        self._last_executed_v = float(command.v)
        self._last_executed_omega = float(command.omega)
        if command.v >= self.config.minimum_continuation_speed_mps:
            self._forward_feedback_streak += 1
        else:
            self._forward_feedback_streak = 0
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if command.v < -1.0e-6 and self._last_commanded_v > 0.0:
            self._feedback_mismatch_streak += 1
        else:
            self._feedback_mismatch_streak = 0
        if (
            self.active
            and command.v < -1.0e-6
            and (
                self._last_commanded_v <= 0.0
                or self._feedback_mismatch_streak
                >= self.config.maximum_feedback_mismatch_steps
            )
        ):
            self._clear_transaction(
                cooldown=True, reason="feedback_direction_mismatch"
            )

    def observe_commanded(self, command: ControlCommand, reason: str) -> None:
        """Observe the final command that was actually sent to the Pi."""
        self._last_commanded_v = float(command.v)
        self._last_commanded_omega = float(command.omega)
        if reason in _SAFETY_RELEASE_REASONS:
            self._clear_transaction(cooldown=True, reason=f"safety_{reason}")
        elif command.v <= 0.0:
            self._clear_transaction(
                cooldown=True, reason="final_nonforward_command"
            )

    def observe_safety(self, reason: str) -> None:
        """Release a transaction when the unchanged arbiter takes ownership."""
        if reason in _SAFETY_RELEASE_REASONS:
            self._clear_transaction(cooldown=True, reason=f"safety_{reason}")

    def observe_executed(self, command: ControlCommand, reason: str) -> None:
        """Compatibility helper for deterministic unit/offline tests."""
        self.observe_feedback(command)
        self.observe_commanded(command, reason)

    def _clear_transaction(
        self, cooldown: bool = False, reason: Optional[str] = None
    ) -> None:
        had_transaction = self._sequence is not None
        self._sequence = None
        self._sequence_index = 0
        self._remaining = 0
        if cooldown and had_transaction:
            self._cooldown_remaining = max(
                self._cooldown_remaining, self.config.reentry_cooldown_steps
            )
        if had_transaction and reason:
            self._pending_release_reason = reason

    def _fresh_risk_safe(self, diagnostics: Mapping[str, Any]) -> bool:
        required = (
            "probabilistic_obstacle_hard_violation",
            "probabilistic_obstacle_maximum_step_probability",
            "probabilistic_obstacle_probability_mass",
        )
        if any(key not in diagnostics for key in required):
            return False
        return bool(
            not diagnostics["probabilistic_obstacle_hard_violation"]
            and _finite(
                diagnostics["probabilistic_obstacle_maximum_step_probability"],
                np.inf,
            )
            <= self.config.maximum_probability
            and _finite(
                diagnostics["probabilistic_obstacle_probability_mass"],
                np.inf,
            )
            <= self.config.maximum_probability_mass
        )

    def _compatible(
        self, candidates: Iterable[Mapping[str, Any]], steering_reference: float
    ) -> Tuple[Mapping[str, Any], ...]:
        limit = self.config.maximum_steering_change_radps
        compatible = []
        for candidate in candidates:
            prefix = _candidate_sequence(candidate)[: self.config.commit_steps]
            if len(prefix) < self.config.commit_steps:
                continue
            steering = prefix[:, 1]
            deltas = np.diff(np.concatenate(
                [[float(steering_reference)], steering]
            ))
            if np.all(np.isfinite(steering)) and np.all(np.abs(deltas) <= limit):
                compatible.append(candidate)
        return tuple(compatible)

    def _latch(self, candidate: Mapping[str, Any]) -> None:
        sequence = _candidate_sequence(candidate)
        self._sequence = np.array(sequence[: self.config.commit_steps], copy=True)
        self._sequence_index = 0
        self._remaining = len(self._sequence)

    def _aligned_sequence(self, plan: PlanResult, index: int) -> np.ndarray:
        """Return a horizon whose first row is the command used this cycle."""
        remaining = np.asarray(self._sequence[index:, :2], dtype=np.float64)
        horizon = max(1, len(np.asarray(plan.control_sequence)))
        if len(remaining) < horizon:
            remaining = np.concatenate([
                remaining,
                np.repeat(remaining[-1:, :], horizon - len(remaining), axis=0),
            ], axis=0)
        return np.array(remaining[:horizon], copy=True)

    def _diagnostic_plan(
        self, plan: PlanResult, diagnostics: Dict[str, Any]
    ) -> PlanResult:
        return PlanResult(
            proposed_control=plan.proposed_control,
            control_sequence=plan.control_sequence,
            predicted_trajectory=plan.predicted_trajectory,
            diagnostics=diagnostics,
        )

    def apply(self, plan: PlanResult) -> PlanResult:
        """Apply or maintain a safe forward continuation before arbitration."""
        if not self.config.enabled:
            return plan

        source = dict(plan.diagnostics or {})
        diagnostics: Dict[str, Any] = dict(source)
        pending_release_reason = self._pending_release_reason
        self._pending_release_reason = None
        blocked_by = tuple(
            key for key in _TRANSACTION_BLOCKS if bool(source.get(key, False))
        )
        trace = source.get("probabilistic_obstacle_emergency_candidate_trace", ())
        forward = _eligible_forward_candidates(trace or (), self.config)
        diagnostics.update({
            "real_robot_forward_passage_enabled": True,
            "real_robot_forward_passage_applied": False,
            "real_robot_forward_passage_started": False,
            "real_robot_forward_passage_refreshed": False,
            "real_robot_forward_passage_active": bool(self.active),
            "real_robot_forward_passage_eligible_candidate_count": len(forward),
            "real_robot_forward_passage_probability_ceiling": float(
                self.config.maximum_probability
            ),
            "real_robot_forward_passage_mass_ceiling": float(
                self.config.maximum_probability_mass
            ),
            "real_robot_forward_passage_original_v": float(plan.proposed_control.v),
            "real_robot_forward_passage_original_omega": float(
                plan.proposed_control.omega
            ),
            "real_robot_forward_passage_last_executed_v": float(
                self._last_executed_v
            ),
            "real_robot_forward_passage_last_executed_omega": float(
                self._last_executed_omega
            ),
            "real_robot_forward_passage_last_commanded_v": float(
                self._last_commanded_v
            ),
            "real_robot_forward_passage_last_commanded_omega": float(
                self._last_commanded_omega
            ),
            "real_robot_forward_passage_forward_feedback_streak": int(
                self._forward_feedback_streak
            ),
            "real_robot_forward_passage_feedback_mismatch_streak": int(
                self._feedback_mismatch_streak
            ),
            "real_robot_forward_passage_cooldown_remaining": int(
                self._cooldown_remaining
            ),
            "real_robot_forward_passage_blocked_by": blocked_by,
        })

        if pending_release_reason is not None:
            diagnostics["real_robot_forward_passage_release"] = (
                pending_release_reason
            )
            diagnostics["real_robot_forward_passage_active"] = False
            return self._diagnostic_plan(plan, diagnostics)

        if blocked_by or not self._fresh_risk_safe(source):
            self._clear_transaction(cooldown=True)
            diagnostics["real_robot_forward_passage_release"] = (
                "existing_transaction" if blocked_by else "fresh_risk"
            )
            diagnostics["real_robot_forward_passage_active"] = False
            return self._diagnostic_plan(plan, diagnostics)

        if not self.active:
            fallback_kind = source.get(
                "probabilistic_obstacle_active_fallback_kind", ""
            )
            continuation_allowed = bool(
                self._forward_feedback_streak
                >= self.config.minimum_forward_feedback_steps
                and self._cooldown_remaining == 0
            )
            compatible = self._compatible(forward, self._last_executed_omega)
            if not (
                plan.proposed_control.v < 0.0
                and fallback_kind == _EMERGENCY_KIND
                and continuation_allowed
                and compatible
            ):
                diagnostics["real_robot_forward_passage_continuation_allowed"] = (
                    continuation_allowed
                )
                return self._diagnostic_plan(plan, diagnostics)
            selected = min(compatible, key=_candidate_order)
            self._latch(selected)
            diagnostics["real_robot_forward_passage_started"] = True

        if not self.active:
            return self._diagnostic_plan(plan, diagnostics)
        index = min(self._sequence_index, len(self._sequence) - 1)
        values = np.asarray(self._sequence[index, :2], dtype=np.float64)
        if values[0] < self.config.minimum_forward_speed_mps:
            self._clear_transaction(cooldown=True)
            diagnostics["real_robot_forward_passage_release"] = "sequence_nonforward"
            diagnostics["real_robot_forward_passage_active"] = False
            return self._diagnostic_plan(plan, diagnostics)

        self._sequence_index += 1
        self._remaining -= 1
        remaining = self._remaining
        diagnostics.update({
            "real_robot_forward_passage_applied": True,
            "real_robot_forward_passage_active": True,
            "real_robot_forward_passage_remaining_steps": int(remaining),
            "real_robot_forward_passage_selected_v": float(values[0]),
            "real_robot_forward_passage_selected_omega": float(values[1]),
            "real_robot_forward_passage_predicted_trajectory_recomputed": False,
        })
        command = ControlCommand(
            values,
            timestamp=plan.proposed_control.timestamp,
            source="real_robot_forward_passage_v3",
        )
        result = PlanResult(
            proposed_control=command,
            control_sequence=self._aligned_sequence(plan, index),
            predicted_trajectory=plan.predicted_trajectory,
            diagnostics=diagnostics,
        )
        if remaining <= 0:
            diagnostics["real_robot_forward_passage_completed"] = True
            self._clear_transaction(cooldown=True)
            diagnostics["real_robot_forward_passage_active"] = False
            diagnostics["real_robot_forward_passage_cooldown_remaining"] = int(
                self._cooldown_remaining
            )
        return result


__all__ = ["ForwardPassageConfig", "ForwardPassageController"]
