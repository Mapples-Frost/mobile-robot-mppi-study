import math
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
        controller.reset()
        if memory is not None:
            memory.reset()
        target = reference.target_at(0.0, initial).pose
        metrics = EpisodeMetrics(
            target.x, target.y, float(self.config["task"].get("position_tolerance", 0.2))
        )
        writer = ArtifactWriter(self.output_dir, self.config, self.project_root)
        viewer = MujocoViewer(plant, enabled=not self.headless)
        try:
            for _ in range(max_steps):
                perceived = self.components["perception"].process(observation)
                plan = controller.plan(perceived.observation, reference)
                decision = self.components["safety"].arbitrate(plan.proposed_control, perceived.guard)
                step = plant.step(decision.executed_control, dt)
                truth = step.ground_truth
                observation = self.components["sensors"].observe(truth)
                if memory is not None:
                    memory.update(observation, reference, decision.executed_control, perceived.guard)
                metrics.update(truth, decision, plan.diagnostics)
                viewer.sync()
                current_target = reference.target_at(truth.timestamp, truth.pose.as_array())
                goal_distance = math.hypot(
                    current_target.pose.x - truth.pose.x,
                    current_target.pose.y - truth.pose.y,
                )
                if goal_distance <= current_target.position_tolerance:
                    break
                if truth.collision and bool(experiment.get("terminate_on_collision", True)):
                    break
        finally:
            viewer.close()
            plant.close()
        summary = metrics.summary()
        metadata = dict(truth.metadata)
        metadata.update({
            "seed": seed,
            "plant_backend": self.config["plant"].get("backend"),
            "prediction_mode": self.components["prediction_mode"],
        })
        writer.write(metrics.records, summary, metadata)
        return ExperimentResult(summary, str(self.output_dir))
