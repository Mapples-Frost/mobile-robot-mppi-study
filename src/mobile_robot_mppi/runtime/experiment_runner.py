import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from mobile_robot_mppi.evaluation.artifacts import ArtifactWriter
from mobile_robot_mppi.evaluation.metrics import EpisodeMetrics
from mobile_robot_mppi.visualization.mujoco_viewer import MujocoViewer
from .factories import make_components


@dataclass(frozen=True)
class ExperimentResult:
    summary: Dict[str, Any]
    output_dir: str


class ExperimentRunner:
    def __init__(self, config: Mapping[str, Any], project_root, output_dir=None, headless=True, rl_policy=None):
        self.config = dict(config)
        self.project_root = Path(project_root).resolve()
        experiment = self.config["experiment"]
        default_output = self.project_root / "results" / "research_platform" / str(experiment.get("name", "run"))
        self.output_dir = Path(output_dir or experiment.get("output_dir", default_output)).resolve()
        self.headless = bool(headless)
        self.components = make_components(self.config, self.project_root, rl_policy=rl_policy)

    def run(self):
        experiment = self.config["experiment"]
        seed = int(experiment.get("seed", 0))
        dt = float(experiment["control_dt"])
        max_steps = int(experiment["max_steps"])
        initial = np.asarray(experiment.get("initial_state", (0.0, 0.0, 0.0)), dtype=np.float64)
        plant = self.components["plant"]
        reference = self.components["reference"]
        controller = self.components["controller"]
        memory = self.components["memory"]
        truth = plant.reset(seed, initial)
        observation = self.components["sensors"].reset(truth, seed)
        reference_reset = getattr(reference, "reset", None)
        if callable(reference_reset):
            reference_reset()
        controller.reset()
        if memory is not None:
            memory.reset()
        if hasattr(reference, "waypoints"):
            final_values = reference.waypoints[-1]
            final_theta = final_values[2] if len(final_values) >= 3 else float(initial[2])
            target = type(truth.pose)(final_values[0], final_values[1], final_theta)
        elif hasattr(reference, "points"):
            final_values = reference.points[-1]
            target = type(truth.pose)(final_values[0], final_values[1], float(initial[2]))
        elif hasattr(reference, "poses"):
            final_values = reference.poses[-1]
            target = type(truth.pose)(final_values[0], final_values[1], final_values[2])
        else:
            target = reference.target_at(0.0, initial).pose
        metrics = EpisodeMetrics(
            target.x,
            target.y,
            float(self.config["task"].get("position_tolerance", 0.2)),
            control_dt=dt,
        )
        writer = ArtifactWriter(self.output_dir, self.config, self.project_root)
        viewer = MujocoViewer(plant, enabled=not self.headless)
        termination_reason = "max_steps"
        try:
            for _ in range(max_steps):
                wall_step_start = time.perf_counter()
                perceived = self.components["perception"].process(observation)
                plan = controller.plan(perceived.observation, reference)
                decision = self.components["safety"].arbitrate(plan.proposed_control, perceived.guard)
                controller.observe_safety_decision(decision)
                step = plant.step(decision.executed_control, dt)
                truth = step.ground_truth
                observation = self.components["sensors"].observe(truth)
                if memory is not None:
                    memory.update(observation, reference, decision.executed_control, perceived.guard)
                metrics.update(
                    truth,
                    decision,
                    plan.diagnostics,
                    applied_control=type(step.executed_control)(
                        np.asarray(
                            step.metadata.get(
                                "average_applied_control",
                                step.executed_control.values,
                            ),
                            dtype=np.float64,
                        ),
                        step.executed_control.timestamp,
                        "plant_interval_average",
                    ),
                )
                viewer.sync()
                if not self.headless:
                    # MuJoCo can simulate much faster than wall time.  Pace only
                    # interactive viewing so the window remains observable;
                    # headless benchmarks retain their maximum throughput.
                    remaining = dt - (time.perf_counter() - wall_step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
                current_target = reference.target_at(truth.timestamp, truth.pose.as_array())
                goal_distance = math.hypot(
                    current_target.pose.x - truth.pose.x,
                    current_target.pose.y - truth.pose.y,
                )
                if (
                    current_target.is_terminal
                    and goal_distance <= current_target.position_tolerance
                ):
                    termination_reason = "goal_reached"
                    break
                if truth.collision and bool(experiment.get("terminate_on_collision", True)):
                    termination_reason = "collision"
                    break
        finally:
            viewer.close()
            plant.close()
        summary = metrics.summary(termination_reason=termination_reason)
        metadata = dict(truth.metadata)
        metadata.update({
            "seed": seed,
            "plant_backend": self.config["plant"].get("backend"),
            "prediction_mode": self.components["prediction_mode"],
            "planner_importance_sampling_correction": bool(
                controller.config.importance_sampling_correction
            ),
            "sampling_prior": str(
                self.config.get("planner", {}).get(
                    "sampling_prior", "goal_warm_start"
                )
            ),
            "rl_enabled": bool(self.config.get("rl", {}).get("enabled", False)),
            "rl_policy_id": self.config.get("rl", {}).get("policy_id"),
            "rl_checkpoint": self.config.get("rl", {}).get("checkpoint"),
            "rl_gate": dict(self.config.get("rl", {}).get("gate", {})),
        })
        writer.write(metrics.records, summary, metadata)
        return ExperimentResult(summary, str(self.output_dir))
