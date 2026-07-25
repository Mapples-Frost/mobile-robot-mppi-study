"""Show several V2 obstacle episodes in one visible MuJoCo review window."""

import argparse
import time
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.state_machine import (
    generate_state_machine_trajectory,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/research/dynamic_obstacle_process_v2.yaml"
DEFAULT_SEEDS = (730100001, 730100003, 730100005, 730100007)
TRAIL_STRIDE = 4
MAX_TRAIL_GEOMS = 81
MAX_EVENT_GEOMS = 8


def _load_config(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("V2 viewer config must be a mapping")
    return value


def build_multiepisode_xml(max_episodes=4):
    trails = "\n".join(
        '<geom name="trail_%03d" type="sphere" pos="0 0 -1" '
        'size="0.038" rgba="0.12 0.42 0.90 0" '
        'contype="0" conaffinity="0"/>' % index
        for index in range(MAX_TRAIL_GEOMS)
    )
    changes = "\n".join(
        '<geom name="event_%02d" type="cylinder" pos="0 0 -1" '
        'size="0.105 0.022" rgba="1.0 0.62 0.0 0" '
        'contype="0" conaffinity="0"/>' % index
        for index in range(MAX_EVENT_GEOMS)
    )
    indicators = "\n".join(
        '<geom name="episode_%02d" type="sphere" '
        'pos="%.3f -4.6 0.10" size="0.10" '
        'rgba="0.35 0.38 0.42 1" contype="0" conaffinity="0"/>'
        % (index, -0.45 * (max_episodes - 1) + 0.9 * index)
        for index in range(max_episodes)
    )
    return """
<mujoco model="%s">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.72 0.72 0.72"/>
  </visual>
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker"
             rgb1="0.89 0.91 0.93" rgb2="0.73 0.77 0.81"
             width="512" height="512"/>
    <material name="ground_mat" texture="ground_tex"
              texrepeat="12 12" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="-2 -2 8" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="floor" type="plane" size="9 9 0.1"
          material="ground_mat" friction="1 0.005 0.0001"/>
    %s
    %s
    %s
    <body name="robot_reference" pos="0 0 0.18">
      <geom name="robot_body" type="cylinder" size="0.27 0.18"
            rgba="0.10 0.66 0.29 1.0"/>
      <geom name="robot_heading" type="box" pos="0.24 0 0.03"
            size="0.13 0.038 0.038" rgba="0.02 0.20 0.07 1.0"/>
    </body>
    <body name="dynamic_obstacle" mocap="true" pos="-4 -2.2 0.25">
      <geom name="obstacle_body" type="cylinder" size="0.25 0.25"
            rgba="0.91 0.15 0.09 1.0"/>
      <geom name="obstacle_heading" type="box" pos="0.20 0 0.03"
            size="0.10 0.035 0.035" rgba="0.32 0.01 0.01 1.0"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""" % (
        escape("V2 P4 multi-episode review"),
        trails,
        changes,
        indicators,
    )


def _geom_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if value < 0:
        raise RuntimeError("missing viewer geom: %s" % name)
    return value


def _set_obstacle_pose(data, state):
    x, y, vx, vy = (float(value) for value in state)
    data.mocap_pos[0] = (x, y, 0.25)
    speed = float(np.hypot(vx, vy))
    yaw = float(np.arctan2(vy, vx)) if speed > 1.0e-9 else 0.0
    data.mocap_quat[0] = (
        float(np.cos(0.5 * yaw)),
        0.0,
        0.0,
        float(np.sin(0.5 * yaw)),
    )


def configure_episode_scene(model, trajectory, episode_index, episode_count):
    scene_minimum = np.minimum(trajectory.states[:, :2].min(axis=0), (0.0, 0.0))
    scene_maximum = np.maximum(trajectory.states[:, :2].max(axis=0), (0.0, 0.0))
    scene_center_x = 0.5 * (scene_minimum[0] + scene_maximum[0])
    indicator_y = scene_minimum[1] - 0.45
    sample_indices = list(range(0, trajectory.states.shape[0], TRAIL_STRIDE))
    if sample_indices[-1] != trajectory.states.shape[0] - 1:
        sample_indices.append(trajectory.states.shape[0] - 1)
    if len(sample_indices) > MAX_TRAIL_GEOMS:
        raise ValueError("trajectory exceeds fixed viewer trail capacity")
    for slot in range(MAX_TRAIL_GEOMS):
        geom_id = _geom_id(model, "trail_%03d" % slot)
        if slot < len(sample_indices):
            state_index = sample_indices[slot]
            x, y = trajectory.states[state_index, :2]
            model.geom_pos[geom_id] = (x, y, 0.035)
            if trajectory.observed_mask[state_index]:
                model.geom_rgba[geom_id] = (0.12, 0.42, 0.90, 0.55)
            else:
                model.geom_rgba[geom_id] = (0.80, 0.20, 0.66, 0.90)
        else:
            model.geom_pos[geom_id] = (0.0, 0.0, -1.0)
            model.geom_rgba[geom_id, 3] = 0.0
    event_steps = np.flatnonzero(trajectory.change_flags)
    if event_steps.size > MAX_EVENT_GEOMS:
        raise ValueError("trajectory exceeds fixed viewer event capacity")
    for slot in range(MAX_EVENT_GEOMS):
        geom_id = _geom_id(model, "event_%02d" % slot)
        if slot < event_steps.size:
            x, y = trajectory.states[int(event_steps[slot]), :2]
            model.geom_pos[geom_id] = (x, y, 0.025)
            model.geom_rgba[geom_id] = (1.0, 0.62, 0.0, 0.96)
        else:
            model.geom_pos[geom_id] = (0.0, 0.0, -1.0)
            model.geom_rgba[geom_id, 3] = 0.0
    for slot in range(episode_count):
        geom_id = _geom_id(model, "episode_%02d" % slot)
        model.geom_pos[geom_id] = (
            scene_center_x - 0.45 * (episode_count - 1) + 0.9 * slot,
            indicator_y,
            0.10,
        )
        if slot < episode_index:
            model.geom_rgba[geom_id] = (0.10, 0.66, 0.29, 0.55)
            model.geom_size[geom_id, 0] = 0.10
        elif slot == episode_index:
            model.geom_rgba[geom_id] = (0.10, 0.88, 0.34, 1.0)
            model.geom_size[geom_id, 0] = 0.145
        else:
            model.geom_rgba[geom_id] = (0.35, 0.38, 0.42, 1.0)
            model.geom_size[geom_id, 0] = 0.10


def configure_episode_label(
    scene, trajectory, episode_index, episode_count, seed
):
    """Add a world-space label so a viewer reset cannot look like teleportation."""
    if scene is None:
        return
    if scene.ngeom == 0:
        mujoco.mjv_initGeom(
            scene.geoms[0],
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.ones(4, dtype=np.float32),
        )
        scene.ngeom = 1
    scene_minimum = np.minimum(trajectory.states[:, :2].min(axis=0), (0.0, 0.0))
    scene_maximum = np.maximum(trajectory.states[:, :2].max(axis=0), (0.0, 0.0))
    scene.geoms[0].pos = (
        scene_minimum[0],
        scene_maximum[1] + 0.45,
        0.34,
    )
    scene.geoms[0].label = "Episode %d/%d | seed %d" % (
        episode_index + 1,
        episode_count,
        int(seed),
    )


def configure_camera(camera, trajectory):
    scene_minimum = np.minimum(trajectory.states[:, :2].min(axis=0), (0.0, 0.0))
    scene_maximum = np.maximum(trajectory.states[:, :2].max(axis=0), (0.0, 0.0))
    center = 0.5 * (scene_minimum + scene_maximum)
    span = float(np.max(scene_maximum - scene_minimum))
    camera.lookat[:] = (center[0], center[1], 0.0)
    camera.distance = max(6.5, 1.25 * span)
    camera.azimuth = 90.0
    camera.elevation = -62.0


def _interpolate(trajectory, episode_time):
    upper = int(
        np.searchsorted(trajectory.times, episode_time, side="right")
    )
    upper = min(max(upper, 1), trajectory.times.shape[0] - 1)
    lower = upper - 1
    interval = trajectory.times[upper] - trajectory.times[lower]
    alpha = (episode_time - trajectory.times[lower]) / interval
    return (
        (1.0 - alpha) * trajectory.states[lower]
        + alpha * trajectory.states[upper]
    )


def show_multiepisode(
    seeds=DEFAULT_SEEDS,
    config_path=DEFAULT_CONFIG,
    playback_speed=1.15,
):
    config = _load_config(config_path)
    profiles = noise_profiles_from_mapping(config["noise_profiles"])
    trajectories = [
        generate_state_machine_trajectory(
            "combined_change", int(seed), profiles["medium"], config
        )
        for seed in seeds
    ]
    model = mujoco.MjModel.from_xml_string(
        build_multiepisode_xml(len(trajectories))
    )
    data = mujoco.MjData(model)
    configure_episode_scene(model, trajectories[0], 0, len(trajectories))
    _set_obstacle_pose(data, trajectories[0].states[0])
    mujoco.mj_forward(model, data)
    duration = float(trajectories[0].times[-1])
    pause_duration = 1.8
    cycle_duration = duration + pause_duration

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        configure_camera(viewer.cam, trajectories[0])
        with viewer.lock():
            configure_episode_label(
                viewer.user_scn,
                trajectories[0],
                0,
                len(trajectories),
                seeds[0],
            )
        start = time.perf_counter()
        active_episode = -1
        while viewer.is_running():
            elapsed = (time.perf_counter() - start) * float(playback_speed)
            episode_index = int(elapsed // cycle_duration) % len(trajectories)
            episode_time = elapsed % cycle_duration
            trajectory = trajectories[episode_index]
            if episode_index != active_episode:
                with viewer.lock():
                    configure_episode_scene(
                        model, trajectory, episode_index, len(trajectories)
                    )
                    configure_episode_label(
                        viewer.user_scn,
                        trajectory,
                        episode_index,
                        len(trajectories),
                        seeds[episode_index],
                    )
                    configure_camera(viewer.cam, trajectory)
                active_episode = episode_index
            if episode_time >= duration:
                state = trajectory.states[-1]
            else:
                state = _interpolate(trajectory, episode_time)
            _set_obstacle_pose(data, state)
            data.time = min(episode_time, duration)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--playback-speed", type=float, default=1.15)
    return parser.parse_args()


def main():
    args = parse_args()
    show_multiepisode(args.seeds, args.config, args.playback_speed)


if __name__ == "__main__":
    main()
