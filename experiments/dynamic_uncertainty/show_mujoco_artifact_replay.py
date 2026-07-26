"""Replay one saved MuJoCo episode without invoking its controller."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant
from mobile_robot_mppi.visualization.mujoco_viewer import MujocoViewer


def _controls(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty saved trajectory: {path}")
    return tuple(
        (float(row["executed_v"]), float(row["executed_omega"]))
        for row in rows
    )


def replay(run_dir: Path, playback_speed: float, loop: bool):
    run_dir = run_dir.resolve()
    config_path = run_dir / "config_resolved.yaml"
    trajectory_path = run_dir / "trajectory.csv"
    if not config_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(
            f"expected config_resolved.yaml and trajectory.csv in {run_dir}"
        )

    config = load_yaml(config_path)
    if str(config.get("plant", {}).get("backend")) != "mujoco_diff_drive":
        raise ValueError("saved episode is not a MuJoCo differential-drive run")
    controls = _controls(trajectory_path)
    dt = float(config["experiment"]["control_dt"])
    seed = int(config["experiment"]["seed"])
    initial = np.asarray(config["experiment"]["initial_state"], dtype=np.float64)
    plant = MujocoDiffDrivePlant(
        deepcopy(config["plant"]),
        deepcopy(config["scene"]),
    )
    viewer = MujocoViewer(plant, enabled=True)
    if viewer.viewer is None:
        plant.close()
        raise RuntimeError("MuJoCo viewer did not open")
    viewer.viewer.cam.lookat[:] = (0.0, -1.8, 0.0)
    viewer.viewer.cam.distance = 10.0
    viewer.viewer.cam.azimuth = 90.0
    viewer.viewer.cam.elevation = -58.0

    try:
        while viewer.viewer.is_running():
            plant.reset(seed, initial)
            viewer.sync()
            for step_index, values in enumerate(controls):
                if not viewer.viewer.is_running():
                    return
                started = time.perf_counter()
                plant.step(
                    ControlCommand(
                        np.asarray(values, dtype=np.float64),
                        step_index * dt,
                        "saved_artifact_replay",
                    ),
                    dt,
                )
                viewer.sync()
                remaining = (
                    dt / playback_speed
                    - (time.perf_counter() - started)
                )
                if remaining > 0.0:
                    time.sleep(remaining)
            if not loop:
                while viewer.viewer.is_running():
                    viewer.sync()
                    time.sleep(1.0 / 30.0)
                return
            time.sleep(1.0)
    finally:
        viewer.close()
        plant.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(argv)
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    replay(args.run_dir, args.playback_speed, bool(args.loop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
