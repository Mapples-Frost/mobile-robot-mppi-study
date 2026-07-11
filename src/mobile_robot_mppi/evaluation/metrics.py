import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    goal_x: float
    goal_y: float
    goal_tolerance: float
    records: List[Dict[str, float]] = field(default_factory=list)

    def update(self, truth, safety_decision, planner_diagnostics):
        distance = math.hypot(truth.pose.x - self.goal_x, truth.pose.y - self.goal_y)
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
            "safety_override": float(safety_decision.overridden),
            "planner_compute_ms": float(planner_diagnostics.get("compute_ms", 0.0)),
        })

    def summary(self):
        if not self.records:
            return {"steps": 0, "success": False}
        values = self.records
        path_length = sum(
            math.hypot(values[index]["x"] - values[index - 1]["x"], values[index]["y"] - values[index - 1]["y"])
            for index in range(1, len(values))
        )
        controls = np.asarray([(row["executed_v"], row["executed_omega"]) for row in values])
        jerk = np.diff(controls, axis=0) if len(values) > 1 else np.zeros((0, 2))
        compute = np.asarray([row["planner_compute_ms"] for row in values])
        finite_clearance = [row["clearance"] for row in values if math.isfinite(row["clearance"])]
        return {
            "steps": len(values),
            "success": bool(values[-1]["goal_distance"] <= self.goal_tolerance and not any(row["collision"] for row in values)),
            "collision": bool(any(row["collision"] for row in values)),
            "final_goal_distance": values[-1]["goal_distance"],
            "trajectory_length": path_length,
            "minimum_clearance": min(finite_clearance) if finite_clearance else None,
            "mean_abs_omega": float(np.mean(np.abs(controls[:, 1]))),
            "control_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0,
            "mean_slip_ratio": float(np.mean([row["slip_ratio"] for row in values])),
            "safety_interventions": int(sum(row["safety_override"] for row in values)),
            "planner_compute_ms_mean": float(compute.mean()),
            "planner_compute_ms_max": float(compute.max()),
        }
