"""Show recurrent V3 shuttle/patrol episodes in one MuJoCo review window."""

import argparse
import time
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import generate_patrol_trajectory


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"
DEFAULT_EPISODES = (
    ("stochastic_shuttle", 730100021),
    ("branching_patrol", 730100021),
    ("hybrid_patrol", 730100021),
    ("hybrid_patrol", 730100022),
)
TRAIL_STRIDE = 4
MAX_TRAIL_GEOMS = 226
MAX_EVENT_GEOMS = 64
MAX_WAYPOINT_GEOMS = 8
MAX_EPISODES = 4


def _load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("V3 viewer config must be a mapping")
    return value


def build_multiepisode_xml():
    trails = "\n".join(
        '<geom name="trail_%03d" type="sphere" pos="0 0 -1" '
        'size="0.032" rgba="0.12 0.42 0.90 0" '
        'contype="0" conaffinity="0"/>' % index
        for index in range(MAX_TRAIL_GEOMS)
    )
    events = "\n".join(
        '<geom name="event_%02d" type="cylinder" pos="0 0 -1" '
        'size="0.085 0.020" rgba="1.0 0.62 0.0 0" '
        'contype="0" conaffinity="0"/>' % index
        for index in range(MAX_EVENT_GEOMS)
    )
    waypoints = "\n".join(
        '<geom name="waypoint_%02d" type="cylinder" pos="0 0 -1" '
        'size="0.13 0.018" rgba="0.46 0.22 0.72 0" '
        'contype="0" conaffinity="0"/>' % index
        for index in range(MAX_WAYPOINT_GEOMS)
    )
    indicators = "\n".join(
        '<geom name="episode_%02d" type="sphere" pos="0 0 -1" '
        'size="0.10" rgba="0.35 0.38 0.42 1" '
        'contype="0" conaffinity="0"/>' % index
        for index in range(MAX_EPISODES)
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
    <geom name="floor" type="plane" size="12 12 0.1"
          material="ground_mat" friction="1 0.005 0.0001"/>
    %s
    %s
    %s
    %s
    <body name="robot_reference" pos="0 0 0.18">
      <geom name="robot_body" type="cylinder" size="0.27 0.18"
            rgba="0.10 0.66 0.29 1.0"/>
      <geom name="robot_heading" type="box" pos="0.24 0 0.03"
            size="0.13 0.038 0.038" rgba="0.02 0.20 0.07 1.0"/>
    </body>
    <body name="dynamic_obstacle" mocap="true" pos="-3.4 -1.4 0.25">
      <geom name="obstacle_body" type="cylinder" size="0.25 0.25"
            rgba="0.91 0.15 0.09 1.0"/>
      <geom name="obstacle_heading" type="box" pos="0.20 0 0.03"
            size="0.10 0.035 0.035" rgba="0.32 0.01 0.01 1.0"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""" % (
        escape("V3 recurrent obstacle review"),
        trails,
        events,
        waypoints,
        indicators,
    )


def _geom_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if value < 0:
        raise RuntimeError("missing viewer geom: %s" % name)
    return value


def _bounds(trajectory):
    points = trajectory.states[:, :2]
    waypoints = np.asarray(
        trajectory.metadata.get("waypoints_m", ()), dtype=np.float64
    )
    if waypoints.size:
        points = np.vstack((points, waypoints))
    minimum = np.minimum(points.min(axis=0), (0.0, 0.0))
    maximum = np.maximum(points.max(axis=0), (0.0, 0.0))
    return minimum, maximum


def configure_episode_scene(
    model, trajectory, episode_index, episode_count
):
    sample_indices = list(range(0, trajectory.states.shape[0], TRAIL_STRIDE))
    if sample_indices[-1] != trajectory.states.shape[0] - 1:
        sample_indices.append(trajectory.states.shape[0] - 1)
    if len(sample_indices) > MAX_TRAIL_GEOMS:
        raise ValueError("trajectory exceeds V3 viewer trail capacity")
    for slot in range(MAX_TRAIL_GEOMS):
        geom_id = _geom_id(model, "trail_%03d" % slot)
        if slot < len(sample_indices):
            state_index = sample_indices[slot]
            x, y = trajectory.states[state_index, :2]
            model.geom_pos[geom_id] = (x, y, 0.035)
            if trajectory.observed_mask[state_index]:
                model.geom_rgba[geom_id] = (0.12, 0.42, 0.90, 0.52)
            else:
                model.geom_rgba[geom_id] = (0.80, 0.20, 0.66, 0.92)
        else:
            model.geom_pos[geom_id] = (0.0, 0.0, -1.0)
            model.geom_rgba[geom_id, 3] = 0.0

    event_records = list(trajectory.metadata["events"])
    if len(event_records) > MAX_EVENT_GEOMS:
        raise ValueError("trajectory exceeds V3 viewer event capacity")
    for slot in range(MAX_EVENT_GEOMS):
        geom_id = _geom_id(model, "event_%02d" % slot)
        if slot < len(event_records):
            record = event_records[slot]
            step = int(record["step"])
            x, y = trajectory.states[step, :2]
            model.geom_pos[geom_id] = (x, y, 0.025)
            if record["kind"] in (
                "early_retarget",
                "hesitation_stop",
                "speed_replan",
            ):
                model.geom_rgba[geom_id] = (0.92, 0.18, 0.05, 0.98)
                model.geom_size[geom_id, 0] = 0.105
            else:
                model.geom_rgba[geom_id] = (1.0, 0.64, 0.0, 0.92)
                model.geom_size[geom_id, 0] = 0.080
        else:
            model.geom_pos[geom_id] = (0.0, 0.0, -1.0)
            model.geom_rgba[geom_id, 3] = 0.0

    waypoints = np.asarray(
        trajectory.metadata.get("waypoints_m", ()), dtype=np.float64
    )
    for slot in range(MAX_WAYPOINT_GEOMS):
        geom_id = _geom_id(model, "waypoint_%02d" % slot)
        if waypoints.size and slot < waypoints.shape[0]:
            model.geom_pos[geom_id] = (
                waypoints[slot, 0],
                waypoints[slot, 1],
                0.02,
            )
            model.geom_rgba[geom_id] = (0.46, 0.22, 0.72, 0.82)
        else:
            model.geom_pos[geom_id] = (0.0, 0.0, -1.0)
            model.geom_rgba[geom_id, 3] = 0.0

    minimum, maximum = _bounds(trajectory)
    center_x = 0.5 * (minimum[0] + maximum[0])
    indicator_y = minimum[1] - 0.42
    for slot in range(MAX_EPISODES):
        geom_id = _geom_id(model, "episode_%02d" % slot)
        if slot < episode_count:
            model.geom_pos[geom_id] = (
                center_x - 0.45 * (episode_count - 1) + 0.9 * slot,
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
        else:
            model.geom_pos[geom_id] = (0.0, 0.0, -1.0)
            model.geom_rgba[geom_id, 3] = 0.0


def configure_episode_label(
    scene, trajectory, process, episode_index, episode_count, seed
):
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
    minimum, maximum = _bounds(trajectory)
    scene.geoms[0].pos = (
        minimum[0],
        maximum[1] + 0.42,
        0.34,
    )
    scene.geoms[0].label = "Episode %d/%d | %s | seed %d" % (
        episode_index + 1,
        episode_count,
        process,
        int(seed),
    )


def configure_camera(camera, trajectory):
    minimum, maximum = _bounds(trajectory)
    center = 0.5 * (minimum + maximum)
    span = float(np.max(maximum - minimum))
    camera.lookat[:] = (center[0], center[1], 0.0)
    camera.distance = max(6.5, 1.25 * span)
    camera.azimuth = 90.0
    camera.elevation = -62.0


def _set_obstacle_pose(data, state, previous_yaw=0.0):
    x, y, vx, vy = (float(value) for value in state)
    data.mocap_pos[0] = (x, y, 0.25)
    speed = float(np.hypot(vx, vy))
    yaw = (
        float(np.arctan2(vy, vx)) if speed > 1.0e-4 else float(previous_yaw)
    )
    data.mocap_quat[0] = (
        float(np.cos(0.5 * yaw)),
        0.0,
        0.0,
        float(np.sin(0.5 * yaw)),
    )
    return yaw


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
    episodes=DEFAULT_EPISODES,
    config_path=DEFAULT_CONFIG,
    playback_speed=1.8,
):
    config = _load_config(config_path)
    profiles = noise_profiles_from_mapping(config["noise_profiles"])
    trajectories = [
        generate_patrol_trajectory(process, seed, profiles["medium"], config)
        for process, seed in episodes
    ]
    model = mujoco.MjModel.from_xml_string(build_multiepisode_xml())
    data = mujoco.MjData(model)
    configure_episode_scene(model, trajectories[0], 0, len(trajectories))
    last_yaw = _set_obstacle_pose(data, trajectories[0].states[0])
    mujoco.mj_forward(model, data)
    duration = float(trajectories[0].times[-1])
    pause_duration = 2.0
    cycle_duration = duration + pause_duration
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        configure_camera(viewer.cam, trajectories[0])
        with viewer.lock():
            configure_episode_label(
                viewer.user_scn,
                trajectories[0],
                episodes[0][0],
                0,
                len(trajectories),
                episodes[0][1],
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
                        model,
                        trajectory,
                        episode_index,
                        len(trajectories),
                    )
                    configure_episode_label(
                        viewer.user_scn,
                        trajectory,
                        episodes[episode_index][0],
                        episode_index,
                        len(trajectories),
                        episodes[episode_index][1],
                    )
                    configure_camera(viewer.cam, trajectory)
                active_episode = episode_index
                last_yaw = 0.0
            state = (
                trajectory.states[-1]
                if episode_time >= duration
                else _interpolate(trajectory, episode_time)
            )
            last_yaw = _set_obstacle_pose(data, state, last_yaw)
            data.time = min(episode_time, duration)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--playback-speed", type=float, default=1.8)
    return parser.parse_args()


def main():
    args = parse_args()
    show_multiepisode(
        config_path=args.config, playback_speed=args.playback_speed
    )


if __name__ == "__main__":
    main()
