"""Preview the frozen complex-static/three-dynamic MuJoCo scene.

This is a scene visualizer, not a controller evaluation.  The robot remains at
the configured start while the three seeded uncertain obstacle programs move.
The cyan overlay is the frozen global reference; yellow markers are the three
designed route-conflict points.
"""

import argparse
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant
from mobile_robot_mppi.visualization.mujoco_viewer import MujocoViewer


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_irregular_spiral_three_dynamic_v1.yaml"
)


def _next_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _draw_segment(mujoco, scene, start, end, radius, rgba):
    geom = _next_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_makeConnector(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        float(radius),
        float(start[0]),
        float(start[1]),
        float(start[2]),
        float(end[0]),
        float(end[1]),
        float(end[2]),
    )


def _draw_sphere(mujoco, scene, point, radius, rgba):
    geom = _next_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.asarray((radius, 0.0, 0.0), dtype=np.float64),
        np.asarray(point, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )


def _rebuild_overlay(viewer, config, mujoco):
    if viewer.viewer is None or viewer.viewer.user_scn is None:
        return
    points = config["task"]["points"]
    design = config.get("spiral_scene_design", config.get("complex_scene_design"))
    if design is None:
        raise KeyError("scene design block is required for preview")
    conflicts = design["route_conflict_points"]
    with viewer.viewer.lock():
        scene = viewer.viewer.user_scn
        scene.ngeom = 0
        for start, end in zip(points[:-1], points[1:]):
            _draw_segment(
                mujoco,
                scene,
                (start[0], start[1], 0.025),
                (end[0], end[1], 0.025),
                0.025,
                (0.10, 0.78, 0.92, 0.85),
            )
        _draw_sphere(
            mujoco,
            scene,
            (points[0][0], points[0][1], 0.06),
            0.11,
            (0.15, 0.90, 0.25, 0.95),
        )
        _draw_sphere(
            mujoco,
            scene,
            (points[-1][0], points[-1][1], 0.06),
            0.13,
            (0.15, 0.35, 1.00, 0.95),
        )
        for point in conflicts:
            _draw_sphere(
                mujoco,
                scene,
                (point[0], point[1], 0.055),
                0.09,
                (1.00, 0.90, 0.10, 0.92),
            )


def preview(config_path, seed, playback_speed, loop):
    config = load_yaml(config_path)
    plant = MujocoDiffDrivePlant(config["plant"], config["scene"])
    initial_state = np.asarray(
        config["experiment"]["initial_state"], dtype=np.float64
    )
    plant.reset(int(seed), initial_state)
    viewer = MujocoViewer(plant, enabled=True)
    dt = float(config["experiment"]["control_dt"])
    duration = max(
        float(obstacle["motion"]["period_s"])
        for obstacle in config["scene"]["obstacles"]
        if obstacle.get("motion")
    )
    static_count = sum(
        obstacle.get("motion") is None
        for obstacle in config["scene"]["obstacles"]
    )
    dynamic_count = sum(
        obstacle.get("motion") is not None
        for obstacle in config["scene"]["obstacles"]
    )
    print(
        "scene=%s seed=%d static=%d dynamic=%d reference_points=%d"
        % (
            config["scene"]["name"],
            int(seed),
            static_count,
            dynamic_count,
            len(config["task"]["points"]),
        ),
        flush=True,
    )
    try:
        if viewer.viewer is not None:
            with viewer.viewer.lock():
                viewer.viewer.cam.type = plant.mujoco.mjtCamera.mjCAMERA_FREE
                viewer.viewer.cam.lookat[:] = np.asarray(
                    (0.0, 0.0, 0.0), dtype=np.float64
                )
                visual = config["scene"].get("visual", {})
                viewer.viewer.cam.distance = float(
                    visual.get("camera_distance", 14.8)
                )
                viewer.viewer.cam.azimuth = 90.0
                viewer.viewer.cam.elevation = float(
                    visual.get("camera_elevation", -75.0)
                )
        while viewer.viewer is not None and viewer.viewer.is_running():
            plant.step(
                ControlCommand(
                    np.zeros(2, dtype=np.float64),
                    float(plant.time),
                    "scene_preview_hold",
                ),
                dt,
            )
            _rebuild_overlay(viewer, config, plant.mujoco)
            viewer.sync()
            time.sleep(max(0.0, dt / float(playback_speed)))
            if plant.time + 1.0e-9 >= duration:
                if not loop:
                    break
                plant.reset(int(seed), initial_state)
    finally:
        viewer.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preview complex static + three dynamic MuJoCo scene"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=781700001)
    parser.add_argument("--playback-speed", type=float, default=0.75)
    parser.add_argument("--loop", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.playback_speed <= 0.0:
        raise ValueError("playback speed must be positive")
    preview(args.config, args.seed, args.playback_speed, args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
