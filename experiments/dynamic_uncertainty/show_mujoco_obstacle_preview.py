"""Open a visible MuJoCo window for the stochastic obstacle-process preview.

This is a visualization-only helper. It does not execute MPPI, use future truth
inside a controller, or write experiment metrics.
"""

import argparse
import time
from xml.sax.saxutils import escape

import mujoco
import mujoco.viewer
import numpy as np

from mobile_robot_mppi.obstacles.motion import (
    NOISE_PROFILES,
    PROCESS_NAMES,
    generate_obstacle_trajectory,
)


def _trail_geoms(trajectory):
    geoms = []
    for index in range(0, trajectory.states.shape[0], 4):
        x, y = trajectory.states[index, :2]
        geoms.append(
            '<geom name="trail_%03d" type="sphere" '
            'pos="%.7f %.7f 0.035" size="0.035" '
            'rgba="0.15 0.42 0.85 0.45" contype="0" conaffinity="0"/>'
            % (index, x, y)
        )
    for marker_index, index in enumerate(np.flatnonzero(trajectory.change_flags)):
        x, y = trajectory.states[index, :2]
        geoms.append(
            '<geom name="change_%02d" type="cylinder" '
            'pos="%.7f %.7f 0.025" size="0.09 0.025" '
            'rgba="1.0 0.65 0.0 0.95" contype="0" conaffinity="0"/>'
            % (marker_index, x, y)
        )
    return "\n".join(geoms)


def build_preview_xml(trajectory):
    return """
<mujoco model="%s">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75"/>
  </visual>
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker"
             rgb1="0.88 0.90 0.92" rgb2="0.72 0.76 0.80"
             width="512" height="512"/>
    <material name="ground_mat" texture="ground_tex"
              texrepeat="8 8" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="-1.5 -1.0 6.0" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="floor" type="plane" size="6 6 0.1"
          material="ground_mat" friction="1 0.005 0.0001"/>
    %s
    <body name="robot_reference" pos="-0.20 -2.60 0.18">
      <geom name="robot_body" type="cylinder" size="0.25 0.18"
            rgba="0.12 0.65 0.30 1.0"/>
      <geom name="robot_heading" type="box" pos="0.22 0 0.03"
            size="0.12 0.035 0.035" rgba="0.03 0.22 0.08 1.0"/>
    </body>
    <body name="dynamic_obstacle" mocap="true" pos="-3 -1.65 0.25">
      <geom name="obstacle_body" type="cylinder" size="0.25 0.25"
            rgba="0.90 0.16 0.10 1.0"/>
      <geom name="obstacle_heading" type="box" pos="0.20 0 0.03"
            size="0.10 0.035 0.035" rgba="0.35 0.02 0.01 1.0"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""" % (
        escape("stochastic_%s" % trajectory.process),
        _trail_geoms(trajectory),
    )


def _set_obstacle_pose(data, state):
    x, y, vx, vy = (float(value) for value in state)
    data.mocap_pos[0] = (x, y, 0.25)
    yaw = float(np.arctan2(vy, vx)) if np.hypot(vx, vy) > 1.0e-9 else 0.0
    data.mocap_quat[0] = (
        float(np.cos(0.5 * yaw)),
        0.0,
        0.0,
        float(np.sin(0.5 * yaw)),
    )


def show_preview(process, seed, noise_profile, playback_speed=0.65):
    trajectory = generate_obstacle_trajectory(
        process=process,
        seed=int(seed),
        noise_profile=NOISE_PROFILES[noise_profile],
        dt=0.1,
        steps=80,
    )
    model = mujoco.MjModel.from_xml_string(build_preview_xml(trajectory))
    data = mujoco.MjData(model)
    _set_obstacle_pose(data, trajectory.states[0])
    mujoco.mj_forward(model, data)

    duration = float(trajectory.times[-1])
    pause_duration = 1.2
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = (-1.45, -1.75, 0.0)
        viewer.cam.distance = 6.2
        viewer.cam.azimuth = 92.0
        viewer.cam.elevation = -46.0
        start = time.perf_counter()
        while viewer.is_running():
            cycle_time = (
                (time.perf_counter() - start) * float(playback_speed)
            ) % (duration + pause_duration)
            if cycle_time >= duration:
                state = trajectory.states[-1]
            else:
                upper = int(
                    np.searchsorted(trajectory.times, cycle_time, side="right")
                )
                upper = min(max(upper, 1), trajectory.times.shape[0] - 1)
                lower = upper - 1
                interval = trajectory.times[upper] - trajectory.times[lower]
                alpha = (cycle_time - trajectory.times[lower]) / interval
                state = (
                    (1.0 - alpha) * trajectory.states[lower]
                    + alpha * trajectory.states[upper]
                )
            _set_obstacle_pose(data, state)
            data.time = cycle_time
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", choices=PROCESS_NAMES, default="combined_change")
    parser.add_argument("--seed", type=int, default=730100001)
    parser.add_argument(
        "--noise-profile", choices=tuple(NOISE_PROFILES), default="medium"
    )
    parser.add_argument("--playback-speed", type=float, default=0.65)
    return parser.parse_args()


def main():
    args = parse_args()
    show_preview(
        process=args.process,
        seed=args.seed,
        noise_profile=args.noise_profile,
        playback_speed=args.playback_speed,
    )


if __name__ == "__main__":
    main()
