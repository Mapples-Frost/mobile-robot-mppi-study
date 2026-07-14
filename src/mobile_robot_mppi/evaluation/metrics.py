import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    goal_x: float
    goal_y: float
    goal_tolerance: float
    control_dt: float = 0.0
    records: List[Dict[str, float]] = field(default_factory=list)

    def update(self, truth, safety_decision, planner_diagnostics, applied_control=None):
        distance = math.hypot(truth.pose.x - self.goal_x, truth.pose.y - self.goal_y)
        applied = applied_control or safety_decision.executed_control
        prior = dict(planner_diagnostics.get("prior", {}))
        self.records.append({
            "time": truth.timestamp,
            "x": truth.pose.x,
            "y": truth.pose.y,
            "theta": truth.pose.theta,
            "v": truth.twist.v,
            "omega": truth.twist.omega,
            "goal_distance": distance,
            "collision": float(truth.collision),
            "clearance": truth.minimum_clearance,
            "slip_ratio": truth.slip_ratio,
            "proposed_v": safety_decision.proposed_control.v,
            "proposed_omega": safety_decision.proposed_control.omega,
            "executed_v": safety_decision.executed_control.v,
            "executed_omega": safety_decision.executed_control.omega,
            "applied_v": applied.v,
            "applied_omega": applied.omega,
            "safety_override": float(safety_decision.overridden),
            "safety_reason": str(safety_decision.reason),
            "reference_id": str(planner_diagnostics.get("reference_id", "unknown")),
            "planner_compute_ms": float(planner_diagnostics.get("compute_ms", 0.0)),
            "planner_cost_min": float(planner_diagnostics.get("cost_min", 0.0)),
            "planner_cost_mean": float(planner_diagnostics.get("cost_mean", 0.0)),
            "effective_sample_size": float(
                planner_diagnostics.get("effective_sample_size", 0.0)
            ),
            "sample_saturation_fraction": float(
                planner_diagnostics.get("sample_saturation_fraction", 0.0)
            ),
            "prior_type": str(prior.get("type", "unknown")),
            "rl_gate_mode": str(prior.get("gate_mode", "disabled")),
            "rl_gate_alpha": float(prior.get("gate_alpha", 0.0)),
            "rl_ood_score": float(prior.get("ood_score", 0.0)),
            "rl_critic_disagreement": float(
                prior.get("critic_disagreement", 0.0)
            ),
            "rl_exploration_activation": float(
                prior.get("exploration_activation", 0.0)
            ),
            "rl_exploration_latch_alpha": float(
                prior.get("exploration_latch_alpha", 0.0)
            ),
            "rl_subgoal_distance": float(
                prior.get("subgoal_distance", 0.0)
            ),
            "rl_subgoal_bearing": float(
                prior.get("subgoal_bearing", 0.0)
            ),
            "rl_subgoal_x_body": float(
                prior.get("subgoal_x_body", 0.0)
            ),
            "rl_subgoal_y_body": float(
                prior.get("subgoal_y_body", 0.0)
            ),
        })

    def summary(self, termination_reason=None):
        if not self.records:
            return {
                "steps": 0,
                "success": False,
                "termination_reason": termination_reason or "no_steps",
            }
        values = self.records
        path_length = sum(
            math.hypot(values[index]["x"] - values[index - 1]["x"], values[index]["y"] - values[index - 1]["y"])
            for index in range(1, len(values))
        )
        controls = np.asarray([(row["executed_v"], row["executed_omega"]) for row in values])
        jerk = np.diff(controls, axis=0) if len(values) > 1 else np.zeros((0, 2))
        applied_controls = np.asarray(
            [(row["applied_v"], row["applied_omega"]) for row in values]
        )
        applied_jerk = (
            np.diff(applied_controls, axis=0)
            if len(values) > 1
            else np.zeros((0, 2))
        )
        compute = np.asarray([row["planner_compute_ms"] for row in values])
        effective_samples = np.asarray([row["effective_sample_size"] for row in values])
        finite_clearance = [row["clearance"] for row in values if math.isfinite(row["clearance"])]
        collision = bool(any(row["collision"] for row in values))
        success = bool(values[-1]["goal_distance"] <= self.goal_tolerance and not collision)
        deadline_ms = 1000.0 * self.control_dt if self.control_dt > 0.0 else None
        deadline_misses = int(np.sum(compute > deadline_ms)) if deadline_ms is not None else 0
        stuck_steps = sum(
            abs(row["v"]) < 0.02 and row["goal_distance"] > self.goal_tolerance
            for row in values
        )
        spin_steps = sum(abs(row["v"]) < 0.03 and abs(row["omega"]) > 0.5 for row in values)
        safety_reason_counts = {}
        for row in values:
            if row["safety_override"]:
                reason = row["safety_reason"]
                safety_reason_counts[reason] = safety_reason_counts.get(reason, 0) + 1
        return {
            "steps": len(values),
            "success": success,
            "termination_reason": termination_reason or ("goal_reached" if success else "unknown"),
            "time_to_goal_s": float(values[-1]["time"]) if success else None,
            "collision": collision,
            "final_goal_distance": values[-1]["goal_distance"],
            "trajectory_length": path_length,
            "minimum_clearance": min(finite_clearance) if finite_clearance else None,
            "mean_abs_omega": float(np.mean(np.abs(controls[:, 1]))),
            "control_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0,
            "applied_control_jerk": (
                float(np.mean(np.linalg.norm(applied_jerk, axis=1)))
                if applied_jerk.size
                else 0.0
            ),
            "stuck_steps": int(stuck_steps),
            "spin_steps": int(spin_steps),
            "mean_slip_ratio": float(np.mean([row["slip_ratio"] for row in values])),
            "safety_interventions": int(sum(row["safety_override"] for row in values)),
            "safety_reason_counts": dict(sorted(safety_reason_counts.items())),
            "planner_compute_ms_mean": float(compute.mean()),
            "planner_compute_ms_p50": float(np.percentile(compute, 50)),
            "planner_compute_ms_p95": float(np.percentile(compute, 95)),
            "planner_compute_ms_p99": float(np.percentile(compute, 99)),
            "planner_compute_ms_max": float(compute.max()),
            "planner_deadline_ms": deadline_ms,
            "planner_deadline_misses": deadline_misses,
            "planner_deadline_miss_rate": (
                float(deadline_misses / len(values)) if deadline_ms is not None else None
            ),
            "effective_sample_size_mean": float(effective_samples.mean()),
            "effective_sample_size_min": float(effective_samples.min()),
            "sample_saturation_fraction_mean": float(
                np.mean([row["sample_saturation_fraction"] for row in values])
            ),
            "rl_gate_alpha_mean": float(
                np.mean([row.get("rl_gate_alpha", 0.0) for row in values])
            ),
            "rl_gate_fallback_steps": int(
                sum(
                    row.get("prior_type") == "rl_sac"
                    and row.get("rl_gate_alpha", 0.0) <= 1e-12
                    for row in values
                )
            ),
            "rl_ood_score_mean": float(
                np.mean([row.get("rl_ood_score", 0.0) for row in values])
            ),
            "rl_ood_score_max": float(
                np.max([row.get("rl_ood_score", 0.0) for row in values])
            ),
            "rl_critic_disagreement_mean": float(np.mean([
                row.get("rl_critic_disagreement", 0.0) for row in values
            ])),
            "rl_critic_disagreement_max": float(np.max([
                row.get("rl_critic_disagreement", 0.0) for row in values
            ])),
            "rl_exploration_activation_mean": float(np.mean([
                row.get("rl_exploration_activation", 0.0) for row in values
            ])),
            "rl_exploration_latch_alpha_mean": float(np.mean([
                row.get("rl_exploration_latch_alpha", 0.0) for row in values
            ])),
            "rl_subgoal_distance_mean": float(
                np.mean([row.get("rl_subgoal_distance", 0.0) for row in values])
            ),
            "rl_subgoal_abs_bearing_mean": float(
                np.mean([
                    abs(row.get("rl_subgoal_bearing", 0.0)) for row in values
                ])
            ),
        }
