"""Continuously display the qualified online V3 risk-aware closed loop."""

import argparse
import math
import time
from pathlib import Path

import numpy as np

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.runtime.factories import make_components
from mobile_robot_mppi.visualization.mujoco_viewer import MujocoViewer
from mobile_robot_mppi.visualization.probability_overlay import (
    MujocoProbabilityOverlay,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "configs/research/"
    "mujoco_v3_probabilistic_crossing_smoke_amendment17.yaml"
)
DEFAULT_SEEDS = (730100005, 730100003)


def show(config_path=DEFAULT_CONFIG, seeds=DEFAULT_SEEDS):
    config = load_yaml(Path(config_path).resolve())
    config["planner"]["probabilistic_obstacle_risk_enabled"] = True
    dt = float(config["experiment"]["control_dt"])
    max_steps = int(config["experiment"]["max_steps"])
    initial = np.asarray(
        config["experiment"]["initial_state"], dtype=np.float64
    )
    components = make_components(config, ROOT)
    plant = components["plant"]
    sensors = components["sensors"]
    perception = components["perception"]
    controller = components["controller"]
    reference = components["reference"]
    safety = components["safety"]
    memory = components["memory"]
    viewer = MujocoViewer(plant, enabled=True)
    if viewer.viewer is None:
        raise RuntimeError("MuJoCo viewer did not open")
    viewer.viewer.cam.lookat[:] = (-0.2, -0.2, 0.0)
    viewer.viewer.cam.distance = 11.0
    viewer.viewer.cam.azimuth = 90.0
    viewer.viewer.cam.elevation = -58.0
    overlay = MujocoProbabilityOverlay(
        plant.mujoco,
        viewer.viewer,
        start_xy=initial[:2],
        goal_xy=np.asarray(config["task"]["position"], dtype=np.float64),
    )

    episode = 0
    try:
        while viewer.viewer.is_running():
            seed = int(seeds[episode % len(seeds)])
            truth = plant.reset(seed, initial)
            observation = sensors.reset(truth, seed)
            perception.reset()
            controller.reset(seed=seed)
            safety_reset = getattr(safety, "reset", None)
            if callable(safety_reset):
                safety_reset()
            reference_reset = getattr(reference, "reset", None)
            if callable(reference_reset):
                reference_reset()
            if memory is not None:
                memory.reset()
            print(
                f"viewer episode={episode + 1} seed={seed} "
                "risk=enabled",
                flush=True,
            )
            for _ in range(max_steps):
                if not viewer.viewer.is_running():
                    return
                started = time.perf_counter()
                perceived = perception.process(observation)
                plan = controller.plan(
                    perceived.observation, reference
                )
                overlay.update(
                    perceived.observation.auxiliary.get(
                        "probabilistic_obstacle_forecasts", ()
                    ),
                    online_pose=perceived.observation.pose,
                    display_pose=truth.pose,
                    tracker_diagnostics=perceived.observation.auxiliary.get(
                        "dynamic_obstacle_tracker", {}
                    ),
                )
                decision = safety.arbitrate(
                    plan.proposed_control,
                    perceived.guard,
                    plan.diagnostics,
                )
                controller.observe_safety_decision(decision)
                step = plant.step(decision.executed_control, dt)
                truth = step.ground_truth
                observation = sensors.observe(truth)
                if memory is not None:
                    memory.update(
                        observation,
                        reference,
                        decision.executed_control,
                        perceived.guard,
                    )
                viewer.sync()
                remaining = dt - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
                target = reference.target_at(
                    truth.timestamp, truth.pose.as_array()
                )
                distance = math.hypot(
                    target.pose.x - truth.pose.x,
                    target.pose.y - truth.pose.y,
                )
                if truth.collision:
                    print("  reset_reason=collision", flush=True)
                    break
                if (
                    target.is_terminal
                    and distance <= target.position_tolerance
                ):
                    print("  reset_reason=goal_reached", flush=True)
                    break
            if viewer.viewer.is_running():
                time.sleep(1.0)
            episode += 1
    finally:
        viewer.close()
        plant.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    args = parser.parse_args(argv)
    show(args.config, tuple(args.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
