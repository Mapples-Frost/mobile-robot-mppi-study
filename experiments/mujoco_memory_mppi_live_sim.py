# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np


GOAL = (3.0, 3.0)
START_STATE = (0.0, 0.0, 0.0)
ROBOT_RADIUS = 0.25
BOUNDS = {
    "x_min": -0.5,
    "x_max": 3.6,
    "y_min": -0.5,
    "y_max": 3.6,
}

V_MIN = 0.0
V_MAX = 0.45
OMEGA_MAX = 1.2
V_STD = 0.18
OMEGA_STD = 0.35
MAX_PLANNER_OBSTACLES = 35
ESCAPE_WINDOW_STEPS = 20


BOUNDARY_WALL_RGBA = (0.35, 0.35, 0.35, 1.0)
LONG_BOARD_RGBA = (0.72, 0.43, 0.18, 1.0)
U_TRAP_RGBA = (0.30, 0.30, 0.32, 1.0)
MID_WALL_RGBA = (0.48, 0.48, 0.46, 1.0)
CYLINDER_RGBA = (0.75, 0.16, 0.12, 1.0)
DARK_CYLINDER_RGBA = (0.50, 0.08, 0.06, 1.0)


SCENE_CONFIGS = {
    "simple": {
        "name": "simple",
        "description": "Original compact live-sim scene for regression checks.",
        "bounds": dict(BOUNDS),
        "start_state": START_STATE,
        "goal": GOAL,
        "boxes": [
            {
                "name": "long_board",
                "center": (1.35, 1.00),
                "half_size": (0.08, 0.75),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": LONG_BOARD_RGBA,
            },
            {
                "name": "u_left_wall",
                "center": (1.80, 1.50),
                "half_size": (0.08, 0.55),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
            {
                "name": "u_top_wall",
                "center": (2.15, 2.00),
                "half_size": (0.45, 0.08),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
            {
                "name": "u_right_wall",
                "center": (2.50, 1.50),
                "half_size": (0.08, 0.55),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
        ],
        "cylinders": [
            {
                "name": "round_obstacle",
                "center": (2.35, 2.45),
                "radius": 0.18,
                "height": 0.18,
                "rgba": CYLINDER_RGBA,
            }
        ],
    },
    "lab_complex": {
        "name": "lab_complex",
        "description": "Dispersed lab-like map with staggered boards, gates, and sparse pillar obstacles.",
        "bounds": dict(BOUNDS),
        "start_state": START_STATE,
        "goal": GOAL,
        "boxes": [
            {
                "name": "low_board",
                "center": (0.85, 0.65),
                "half_size": (0.08, 0.45),
                "height": 0.16,
                "yaw": math.radians(15.0),
                "rgba": LONG_BOARD_RGBA,
            },
            {
                "name": "mid_bar",
                "center": (1.55, 1.35),
                "half_size": (0.45, 0.07),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": MID_WALL_RGBA,
            },
            {
                "name": "upper_gate_wall",
                "center": (1.65, 2.25),
                "half_size": (0.07, 0.42),
                "height": 0.15,
                "yaw": 0.0,
                "rgba": MID_WALL_RGBA,
            },
        ],
        "cylinders": [
            {
                "name": "lower_right_pillar",
                "center": (2.25, 1.05),
                "radius": 0.16,
                "height": 0.18,
                "rgba": CYLINDER_RGBA,
            },
            {
                "name": "mid_right_pillar",
                "center": (2.55, 1.75),
                "radius": 0.15,
                "height": 0.18,
                "rgba": DARK_CYLINDER_RGBA,
            },
            {
                "name": "near_goal_1",
                "center": (2.45, 2.55),
                "radius": 0.13,
                "height": 0.18,
                "rgba": CYLINDER_RGBA,
            },
            {
                "name": "near_goal_2",
                "center": (2.85, 2.35),
                "radius": 0.12,
                "height": 0.16,
                "rgba": DARK_CYLINDER_RGBA,
            },
            {
                "name": "left_upper_pillar",
                "center": (0.65, 2.25),
                "radius": 0.12,
                "height": 0.16,
                "rgba": CYLINDER_RGBA,
            },
        ],
    },
    "narrow_corridor": {
        "name": "narrow_corridor",
        "description": "Long staggered walls create a narrow passage slightly wider than the robot diameter.",
        "bounds": dict(BOUNDS),
        "start_state": START_STATE,
        "goal": GOAL,
        "boxes": [
            {
                "name": "corridor_lower_1",
                "center": (1.10, 0.78),
                "half_size": (0.70, 0.07),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": LONG_BOARD_RGBA,
            },
            {
                "name": "corridor_upper_1",
                "center": (1.20, 1.48),
                "half_size": (0.78, 0.07),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": LONG_BOARD_RGBA,
            },
            {
                "name": "corridor_lower_2",
                "center": (2.10, 1.56),
                "half_size": (0.72, 0.07),
                "height": 0.16,
                "yaw": math.radians(8.0),
                "rgba": MID_WALL_RGBA,
            },
            {
                "name": "corridor_upper_2",
                "center": (2.05, 2.30),
                "half_size": (0.72, 0.07),
                "height": 0.16,
                "yaw": math.radians(8.0),
                "rgba": MID_WALL_RGBA,
            },
            {
                "name": "corridor_exit_wall",
                "center": (2.92, 2.72),
                "half_size": (0.08, 0.42),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
        ],
        "cylinders": [
            {
                "name": "corridor_mid_cylinder",
                "center": (1.63, 1.14),
                "radius": 0.11,
                "height": 0.17,
                "rgba": CYLINDER_RGBA,
            },
            {
                "name": "corridor_goal_cylinder",
                "center": (2.55, 2.55),
                "radius": 0.12,
                "height": 0.17,
                "rgba": DARK_CYLINDER_RGBA,
            },
        ],
    },
    "u_trap_long_board": {
        "name": "u_trap_long_board",
        "description": "A U-shaped local trap plus nearby long boards, intended to provoke repeated low-progress turns.",
        "bounds": dict(BOUNDS),
        "start_state": START_STATE,
        "goal": GOAL,
        "boxes": [
            {
                "name": "trap_left",
                "center": (1.58, 1.38),
                "half_size": (0.08, 0.62),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
            {
                "name": "trap_top",
                "center": (1.95, 1.95),
                "half_size": (0.45, 0.08),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
            {
                "name": "trap_right",
                "center": (2.32, 1.38),
                "half_size": (0.08, 0.62),
                "height": 0.16,
                "yaw": 0.0,
                "rgba": U_TRAP_RGBA,
            },
            {
                "name": "approach_long_board",
                "center": (0.95, 1.05),
                "half_size": (0.08, 0.72),
                "height": 0.16,
                "yaw": math.radians(-12.0),
                "rgba": LONG_BOARD_RGBA,
            },
            {
                "name": "escape_board",
                "center": (2.65, 2.35),
                "half_size": (0.46, 0.07),
                "height": 0.16,
                "yaw": math.radians(-18.0),
                "rgba": LONG_BOARD_RGBA,
            },
            {
                "name": "upper_offset_wall",
                "center": (1.28, 2.55),
                "half_size": (0.42, 0.07),
                "height": 0.15,
                "yaw": 0.0,
                "rgba": MID_WALL_RGBA,
            },
        ],
        "cylinders": [
            {
                "name": "trap_cylinder_1",
                "center": (1.95, 1.18),
                "radius": 0.12,
                "height": 0.17,
                "rgba": CYLINDER_RGBA,
            },
            {
                "name": "trap_cylinder_2",
                "center": (2.55, 2.78),
                "radius": 0.13,
                "height": 0.17,
                "rgba": DARK_CYLINDER_RGBA,
            },
        ],
    },
}


def timer_now():
    try:
        return time.perf_counter()
    except AttributeError:
        return time.time()


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def ensure_project_on_path():
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_module_from_path(module_name, path):
    if sys.version_info[0] < 3:
        import imp
        return imp.load_source(module_name, path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_planner_helpers():
    ensure_project_on_path()
    helper_path = os.path.join(
        project_root(),
        "src",
        "planners",
        "mppi_mujoco_receding_horizon_experiment.py",
    )
    return load_module_from_path("_live_mppi_helpers", helper_path)


def load_bridge_module(module_name, file_name):
    module_path = os.path.join(
        project_root(),
        "mppi_hardware_bridge",
        "scripts",
        file_name,
    )
    return load_module_from_path(module_name, module_path)


def require_mujoco():
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError(
            "MuJoCo is required for experiments/mujoco_memory_mppi_live_sim.py. "
            "Install the mujoco Python package in this environment. Original error: %s"
            % exc
        )
    return mujoco


def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_control(control):
    return (
        clamp(float(control[0]), V_MIN, V_MAX),
        clamp(float(control[1]), -OMEGA_MAX, OMEGA_MAX),
    )


def rgba_to_xml(rgba):
    if isinstance(rgba, str):
        return rgba
    return "%.3f %.3f %.3f %.3f" % (
        float(rgba[0]),
        float(rgba[1]),
        float(rgba[2]),
        float(rgba[3]),
    )


def make_boundary_walls(bounds):
    wall_half_height = 0.14
    wall_thickness = 0.05
    x_min = bounds["x_min"]
    x_max = bounds["x_max"]
    y_min = bounds["y_min"]
    y_max = bounds["y_max"]
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_x = 0.5 * (x_max - x_min)
    half_y = 0.5 * (y_max - y_min)

    return [
        {
            "name": "wall_south",
            "center": (x_center, y_min - wall_thickness),
            "half_size": (half_x, wall_thickness),
            "height": wall_half_height,
            "yaw": 0.0,
            "rgba": BOUNDARY_WALL_RGBA,
        },
        {
            "name": "wall_north",
            "center": (x_center, y_max + wall_thickness),
            "half_size": (half_x, wall_thickness),
            "height": wall_half_height,
            "yaw": 0.0,
            "rgba": BOUNDARY_WALL_RGBA,
        },
        {
            "name": "wall_west",
            "center": (x_min - wall_thickness, y_center),
            "half_size": (wall_thickness, half_y),
            "height": wall_half_height,
            "yaw": 0.0,
            "rgba": BOUNDARY_WALL_RGBA,
        },
        {
            "name": "wall_east",
            "center": (x_max + wall_thickness, y_center),
            "half_size": (wall_thickness, half_y),
            "height": wall_half_height,
            "yaw": 0.0,
            "rgba": BOUNDARY_WALL_RGBA,
        },
    ]


def make_live_scene_primitives(scene_config):
    bounds = scene_config["bounds"]
    boxes = make_boundary_walls(bounds)
    boxes.extend([dict(box) for box in scene_config.get("boxes", [])])
    cylinders = [dict(cylinder) for cylinder in scene_config.get("cylinders", [])]
    return {"boxes": boxes, "cylinders": cylinders, "bounds": bounds}


def build_live_scene_xml(scene_config, primitives):
    bounds = scene_config["bounds"]
    ground_center_x = 0.5 * (bounds["x_min"] + bounds["x_max"])
    ground_center_y = 0.5 * (bounds["y_min"] + bounds["y_max"])
    ground_size_x = max(3.0, 0.75 * (bounds["x_max"] - bounds["x_min"]))
    ground_size_y = max(3.0, 0.75 * (bounds["y_max"] - bounds["y_min"]))
    goal_x, goal_y = scene_config["goal"]
    lines = [
        '<mujoco model="mppi_live_lab_scene">',
        '  <compiler angle="radian"/>',
        '  <option timestep="0.01" gravity="0 0 -9.81"/>',
        '  <visual>',
        '    <global offwidth="1280" offheight="900"/>',
        '  </visual>',
        '  <default>',
        '    <geom friction="1.0 0.005 0.0001"/>',
        '  </default>',
        '  <worldbody>',
        '    <light name="key_light" pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>',
        '    <geom name="ground" type="plane" pos="{cx:.4f} {cy:.4f} 0" size="{sx:.4f} {sy:.4f} 0.02" rgba="0.88 0.90 0.86 1" contype="0" conaffinity="0" group="3"/>'.format(
            cx=ground_center_x,
            cy=ground_center_y,
            sx=ground_size_x,
            sy=ground_size_y,
        ),
        '    <geom name="goal_marker" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.012" size="0.10 0.012" rgba="0.0 0.85 0.20 0.80" contype="0" conaffinity="0" group="2"/>'.format(
            gx=goal_x,
            gy=goal_y,
        ),
    ]

    for box in primitives["boxes"]:
        cx, cy = box["center"]
        hx, hy = box["half_size"]
        hz = box["height"]
        yaw = float(box.get("yaw", 0.0))
        lines.append(
            '    <geom name="{name}" type="box" pos="{cx:.4f} {cy:.4f} {hz:.4f}" size="{hx:.4f} {hy:.4f} {hz:.4f}" euler="0 0 {yaw:.6f}" rgba="{rgba}" group="1"/>'.format(
                name=box["name"],
                cx=cx,
                cy=cy,
                hz=hz,
                hx=hx,
                hy=hy,
                yaw=yaw,
                rgba=rgba_to_xml(box["rgba"]),
            )
        )

    for cyl in primitives["cylinders"]:
        cx, cy = cyl["center"]
        radius = cyl["radius"]
        hz = cyl["height"]
        lines.append(
            '    <geom name="{name}" type="cylinder" pos="{cx:.4f} {cy:.4f} {hz:.4f}" size="{radius:.4f} {hz:.4f}" rgba="{rgba}" group="1"/>'.format(
                name=cyl["name"],
                cx=cx,
                cy=cy,
                hz=hz,
                radius=radius,
                rgba=rgba_to_xml(cyl["rgba"]),
            )
        )

    lines.extend(
        [
            '    <body name="robot" pos="0 0 0.065">',
            '      <joint name="robot_x" type="slide" axis="1 0 0" damping="0"/>',
            '      <joint name="robot_y" type="slide" axis="0 1 0" damping="0"/>',
            '      <joint name="robot_yaw" type="hinge" axis="0 0 1" damping="0"/>',
            '      <geom name="robot_geom" type="cylinder" size="0.25 0.055" rgba="0.15 0.45 1.0 1" group="0"/>',
            '    </body>',
            '  </worldbody>',
            '</mujoco>',
        ]
    )
    return "\n".join(lines)


class MujocoLiveMppiEnv(object):
    def __init__(self, mujoco, scene_config, no_viewer=False):
        self.mujoco = mujoco
        self.scene_config = scene_config
        self.scene_name = scene_config["name"]
        self.description = scene_config.get("description", "")
        self.bounds = scene_config["bounds"]
        self.goal = scene_config["goal"]
        self.start_state = scene_config["start_state"]
        self.primitives = make_live_scene_primitives(scene_config)
        self.model = mujoco.MjModel.from_xml_string(
            build_live_scene_xml(scene_config, self.primitives)
        )
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.viewer_enabled = False
        self.no_viewer = bool(no_viewer)
        self.executed_trajectory = []
        self.scan_debug = None
        self.planner_obstacles = []
        self.best_predicted_trajectory = []
        self.top_predicted_trajectories = []
        self.memory_feature_positions = []
        self.escape_arrow = None
        self._geomgroup_scan = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        self._identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        self._zero_size = np.zeros(3, dtype=np.float64)
        self.robot_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.robot_geom_id = self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")
        self.x_qpos_adr, self.x_dof_adr = self._joint_addresses("robot_x")
        self.y_qpos_adr, self.y_dof_adr = self._joint_addresses("robot_y")
        self.yaw_qpos_adr, self.yaw_dof_adr = self._joint_addresses("robot_yaw")
        self.reset(self.start_state)
        if not self.no_viewer:
            self.launch_viewer()

    def _name_id(self, obj_type, name):
        value = self.mujoco.mj_name2id(self.model, obj_type, name)
        if value < 0:
            raise RuntimeError("MuJoCo object not found: %s" % name)
        return value

    def _joint_addresses(self, joint_name):
        joint_id = self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def launch_viewer(self):
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer_enabled = True
            self._configure_camera()
        except TypeError:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_enabled = True
            self._configure_camera()
        except Exception as exc:
            self.viewer = None
            self.viewer_enabled = False
            print("warning: MuJoCo viewer failed to start; running headless: %s" % exc)
        return self.viewer

    def _configure_camera(self):
        if self.viewer is None:
            return
        try:
            with self.viewer.lock():
                self.viewer.cam.type = self.mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat[:] = np.array(
                    [
                        0.5 * (self.bounds["x_min"] + self.bounds["x_max"]),
                        0.5 * (self.bounds["y_min"] + self.bounds["y_max"]),
                        0.05,
                    ],
                    dtype=float,
                )
                self.viewer.cam.distance = 5.4
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.elevation = -55.0
        except Exception:
            pass

    def reset(self, start_state):
        x_value, y_value, theta = start_state
        self.data.qpos[self.x_qpos_adr] = float(x_value)
        self.data.qpos[self.y_qpos_adr] = float(y_value)
        self.data.qpos[self.yaw_qpos_adr] = float(theta)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.executed_trajectory = [self.get_state()]
        return self.get_state()

    def get_state(self):
        return (
            float(self.data.qpos[self.x_qpos_adr]),
            float(self.data.qpos[self.y_qpos_adr]),
            wrap_angle(float(self.data.qpos[self.yaw_qpos_adr])),
        )

    def step(self, control, dt):
        v_value, omega_value = control
        substeps = max(1, int(round(float(dt) / float(self.model.opt.timestep))))
        for _ in range(substeps):
            x_value, y_value, theta = self.get_state()
            self.data.qvel[self.x_dof_adr] = float(v_value) * math.cos(theta)
            self.data.qvel[self.y_dof_adr] = float(v_value) * math.sin(theta)
            self.data.qvel[self.yaw_dof_adr] = float(omega_value)
            self.mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.yaw_qpos_adr] = wrap_angle(self.data.qpos[self.yaw_qpos_adr])
        self.mujoco.mj_forward(self.model, self.data)
        state = self.get_state()
        self.executed_trajectory.append(state)
        return state

    def raycast_laserscan(
        self,
        state=None,
        angle_min=-math.pi,
        angle_max=math.pi,
        num_beams=181,
        range_min=0.05,
        range_max=3.0,
    ):
        if state is None:
            state = self.get_state()
        x_value, y_value, theta = state
        if num_beams <= 1:
            num_beams = 2
        angle_increment = (float(angle_max) - float(angle_min)) / float(num_beams - 1)
        origin = np.array([x_value, y_value, 0.08], dtype=np.float64)
        ranges = []
        obstacle_ranges = []
        hit_points_world = []
        ray_segments_world = []
        hit_mask = []

        for beam_index in range(num_beams):
            local_angle = float(angle_min) + float(beam_index) * angle_increment
            world_angle = theta + local_angle
            direction = np.array(
                [math.cos(world_angle), math.sin(world_angle), 0.0],
                dtype=np.float64,
            )
            geomid = np.array([-1], dtype=np.int32)
            try:
                distance = self.mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    direction,
                    self._geomgroup_scan,
                    1,
                    self.robot_body_id,
                    geomid,
                )
            except TypeError:
                distance = self.mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    direction,
                    self._geomgroup_scan,
                    0,
                    self.robot_body_id,
                    geomid,
                )

            hit = (
                distance is not None
                and float(distance) >= float(range_min)
                and float(distance) <= float(range_max)
                and int(geomid[0]) >= 0
            )
            if hit:
                used_range = float(distance)
                hit_point = origin + used_range * direction
                hit_points_world.append((float(hit_point[0]), float(hit_point[1])))
                obstacle_ranges.append(used_range)
            else:
                used_range = float(range_max)
                obstacle_ranges.append(float("inf"))

            end_point = origin + used_range * direction
            ranges.append(used_range)
            hit_mask.append(bool(hit))
            ray_segments_world.append(
                (
                    (float(origin[0]), float(origin[1])),
                    (float(end_point[0]), float(end_point[1])),
                )
            )

        return {
            "ranges": ranges,
            "obstacle_ranges": obstacle_ranges,
            "angle_min": float(angle_min),
            "angle_increment": float(angle_increment),
            "range_min": float(range_min),
            "range_max": float(range_max),
            "hit_points_world": hit_points_world,
            "ray_segments_world": ray_segments_world,
            "hit_mask": hit_mask,
        }

    def has_robot_contact(self):
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if int(contact.geom1) == self.robot_geom_id or int(contact.geom2) == self.robot_geom_id:
                return True
        return False

    def update_debug_visuals(
        self,
        scan,
        planner_obstacles,
        best_predicted_trajectory,
        top_predicted_trajectories,
        memory_feature_positions,
        escape_arrow=None,
    ):
        self.scan_debug = scan
        self.planner_obstacles = list(planner_obstacles or [])
        self.best_predicted_trajectory = list(best_predicted_trajectory or [])
        self.top_predicted_trajectories = list(top_predicted_trajectories or [])
        self.memory_feature_positions = list(memory_feature_positions or [])
        self.escape_arrow = escape_arrow

    def _next_overlay_geom(self, user_scn):
        if user_scn.ngeom >= user_scn.maxgeom:
            return None
        geom = user_scn.geoms[user_scn.ngeom]
        user_scn.ngeom += 1
        return geom

    def _draw_capsule_segment(self, user_scn, p0, p1, radius, rgba):
        geom = self._next_overlay_geom(user_scn)
        if geom is None:
            return
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_CAPSULE),
            self._zero_size,
            np.zeros(3, dtype=np.float64),
            self._identity_mat,
            np.array(rgba, dtype=np.float32),
        )
        self.mujoco.mjv_makeConnector(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_CAPSULE),
            float(radius),
            float(p0[0]),
            float(p0[1]),
            float(p0[2]),
            float(p1[0]),
            float(p1[1]),
            float(p1[2]),
        )

    def _draw_sphere(self, user_scn, pos, radius, rgba):
        geom = self._next_overlay_geom(user_scn)
        if geom is None:
            return
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([float(radius), 0.0, 0.0], dtype=np.float64),
            np.array(pos, dtype=np.float64),
            self._identity_mat,
            np.array(rgba, dtype=np.float32),
        )

    def _draw_polyline(self, user_scn, trajectory, z_value, radius, rgba, max_points=160):
        points = list(trajectory or [])
        if len(points) < 2:
            return
        if len(points) > max_points:
            points = points[-max_points:]
        for index in range(len(points) - 1):
            p0 = (points[index][0], points[index][1], z_value)
            p1 = (points[index + 1][0], points[index + 1][1], z_value)
            self._draw_capsule_segment(user_scn, p0, p1, radius, rgba)

    def _rebuild_overlay(self):
        if self.viewer is None or self.viewer.user_scn is None:
            return
        with self.viewer.lock():
            user_scn = self.viewer.user_scn
            user_scn.ngeom = 0

            self._draw_polyline(
                user_scn,
                self.executed_trajectory,
                0.075,
                0.010,
                (0.60, 0.42, 0.20, 0.95),
            )

            if self.scan_debug is not None:
                for index, segment in enumerate(self.scan_debug.get("ray_segments_world", [])):
                    if index % 4 != 0:
                        continue
                    p0, p1 = segment
                    self._draw_capsule_segment(
                        user_scn,
                        (p0[0], p0[1], 0.045),
                        (p1[0], p1[1], 0.045),
                        0.0025,
                        (0.25, 0.55, 0.95, 0.18),
                    )
                for index, point in enumerate(self.scan_debug.get("hit_points_world", [])):
                    if index % 2 != 0:
                        continue
                    self._draw_sphere(
                        user_scn,
                        (point[0], point[1], 0.075),
                        0.018,
                        (1.0, 0.28, 0.12, 0.82),
                    )

            for obstacle in self.planner_obstacles[:90]:
                self._draw_sphere(
                    user_scn,
                    (obstacle[0], obstacle[1], 0.060),
                    max(0.018, min(0.07, float(obstacle[2]))),
                    (1.0, 0.85, 0.10, 0.45),
                )

            for trajectory in self.top_predicted_trajectories[:2]:
                self._draw_polyline(
                    user_scn,
                    trajectory,
                    0.105,
                    0.005,
                    (0.50, 0.80, 0.85, 0.42),
                    max_points=40,
                )

            self._draw_polyline(
                user_scn,
                self.best_predicted_trajectory,
                0.120,
                0.008,
                (0.00, 0.95, 1.0, 0.92),
                max_points=40,
            )

            for point in self.memory_feature_positions:
                self._draw_sphere(
                    user_scn,
                    (point[0], point[1], 0.145),
                    0.055,
                    (0.85, 0.05, 0.90, 0.82),
                )

            if self.escape_arrow is not None:
                p0, p1 = self.escape_arrow
                self._draw_capsule_segment(
                    user_scn,
                    (p0[0], p0[1], 0.165),
                    (p1[0], p1[1], 0.165),
                    0.014,
                    (0.25, 1.0, 0.35, 0.95),
                )
                self._draw_sphere(
                    user_scn,
                    (p1[0], p1[1], 0.165),
                    0.040,
                    (0.25, 1.0, 0.35, 0.95),
                )

    def render(self, sleep_dt=0.0):
        if self.viewer is None:
            return False
        try:
            self._rebuild_overlay()
            self.viewer.sync()
            if sleep_dt > 0.0:
                time.sleep(float(sleep_dt))
            return True
        except Exception as exc:
            print("warning: MuJoCo viewer render failed; disabling viewer: %s" % exc)
            self.close_viewer_only()
            return False

    def close_viewer_only(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.viewer = None
        self.viewer_enabled = False

    def close(self):
        self.close_viewer_only()


def box_clearance(point, center, half_size, yaw=0.0):
    px, py = point
    cx, cy = center
    hx, hy = half_size
    rel_x = float(px) - float(cx)
    rel_y = float(py) - float(cy)
    cos_yaw = math.cos(-float(yaw))
    sin_yaw = math.sin(-float(yaw))
    local_x = cos_yaw * rel_x - sin_yaw * rel_y
    local_y = sin_yaw * rel_x + cos_yaw * rel_y
    dx = abs(local_x) - float(hx)
    dy = abs(local_y) - float(hy)
    outside_dx = max(dx, 0.0)
    outside_dy = max(dy, 0.0)
    if outside_dx > 0.0 or outside_dy > 0.0:
        return math.hypot(outside_dx, outside_dy)
    return max(dx, dy)


def primitive_clearance(state, primitives, robot_radius):
    x_value, y_value, _ = state
    best = float("inf")
    for box in primitives["boxes"]:
        clearance = box_clearance(
            (x_value, y_value),
            box["center"],
            box["half_size"],
            yaw=box.get("yaw", 0.0),
        ) - robot_radius
        best = min(best, clearance)
    for cyl in primitives["cylinders"]:
        cx, cy = cyl["center"]
        clearance = math.hypot(x_value - cx, y_value - cy) - (robot_radius + cyl["radius"])
        best = min(best, clearance)
    return best


def run_scan_guard(scan, scan_guard_module):
    return scan_guard_module.analyze_scan_front_sector(
        ranges=scan["ranges"],
        angle_min=scan["angle_min"],
        angle_increment=scan["angle_increment"],
        range_min=scan["range_min"],
        range_max=scan["range_max"],
        front_stop_distance=0.34,
        front_slow_distance=0.85,
        front_angle_deg=35.0,
        hard_stop_distance=0.28,
        front_soft_block_distance=0.45,
        front_slow_min_scale=0.45,
        side_stop_distance=0.18,
        side_angle_deg=125.0,
        near_body_stop_radius=0.20,
    )


def run_local_obstacle_layer(scan, state, local_layer_module):
    ranges = scan.get("obstacle_ranges", scan["ranges"])
    try:
        obstacles, debug = local_layer_module.scan_to_experiment_obstacles_geometric(
            ranges=ranges,
            angle_min=scan["angle_min"],
            angle_increment=scan["angle_increment"],
            range_min=scan["range_min"],
            range_max=scan["range_max"],
            current_state_exp=state,
            max_radius=scan["range_max"],
            min_radius=0.08,
            angle_offset_rad=0.0,
            downsample_step=1,
            obstacle_radius=0.08,
            max_obstacle_count=80,
            obstacle_inflation=0.0,
            return_debug=True,
        )
        if obstacles:
            return obstacles, debug
    except Exception as exc:
        print("warning: geometric local obstacle layer failed, using point obstacles: %s" % exc)

    obstacles = local_layer_module.scan_to_experiment_obstacles(
        ranges=ranges,
        angle_min=scan["angle_min"],
        angle_increment=scan["angle_increment"],
        range_min=scan["range_min"],
        range_max=scan["range_max"],
        current_state_exp=state,
        max_radius=scan["range_max"],
        min_radius=0.08,
        angle_offset_rad=0.0,
        downsample_step=3,
        obstacle_radius=0.08,
    )
    return obstacles, {
        "mode": "point_fallback",
        "obstacle_count_after_limit": len(obstacles),
    }


def obstacle_to_robot_frame(obstacle, state):
    obs_x, obs_y = obstacle[0], obstacle[1]
    x_robot, y_robot, theta = state
    dx = float(obs_x) - float(x_robot)
    dy = float(obs_y) - float(y_robot)
    cos_yaw = math.cos(-float(theta))
    sin_yaw = math.sin(-float(theta))
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    return local_x, local_y


def filter_planner_obstacles(obstacles, state, max_count=MAX_PLANNER_OBSTACLES):
    filtered = []
    for obstacle in list(obstacles or []):
        local_x, local_y = obstacle_to_robot_frame(obstacle, state)
        distance = math.hypot(local_x, local_y)

        if local_x < -0.15 and distance > 0.60:
            continue
        if distance > 2.40 and float(obstacle[2]) < 0.11:
            continue

        forward_bonus = 0.0
        if local_x >= -0.05:
            forward_bonus += 0.55
        if local_x >= 0.0 and abs(local_y) < 1.25:
            forward_bonus += 0.45

        radius_bonus = 2.0 * float(obstacle[2])
        priority = distance - forward_bonus - radius_bonus
        filtered.append((priority, distance, -float(obstacle[2]), obstacle))

    filtered.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in filtered[: int(max_count)]]


def initialize_memory_field(args):
    memory_module = load_bridge_module("_live_memory_field", "mppi_memory_field.py")
    cfg = type("Cfg", (object,), {})()
    cfg.memory = type("MemoryCfg", (object,), {})()
    cfg.memory.enable = bool(args.memory_enabled)
    cfg.memory.max_features = 40
    cfg.memory.stuck_window_sec = max(2.0, 20.0 * float(args.dt))
    cfg.memory.stuck_pos_radius = 0.18
    cfg.memory.stuck_goal_progress_threshold = 0.12
    cfg.memory.spin_omega_threshold = 0.80
    cfg.memory.spin_v_threshold = 0.08
    cfg.memory.feature_merge_distance = 0.30
    cfg.memory.feature_radius = 0.60
    cfg.memory.feature_strength_initial = 1.5
    cfg.memory.feature_strength_max = 6.0
    cfg.memory.feature_decay = 0.997
    cfg.memory.local_min_cost_weight = 100.0
    cfg.memory.spin_trap_cost_weight = 90.0
    cfg.memory.low_progress_cost_weight = 80.0
    cfg.memory.escape_direction_cost_weight = 70.0
    cfg.memory.temperature_boost_max = 2.0
    return memory_module.MppiMemoryField(cfg)


def get_memory_feature_positions(memory_field):
    if memory_field is None:
        return []
    positions = []
    for feature in getattr(memory_field, "features", []):
        positions.append((float(feature.position[0]), float(feature.position[1])))
    return positions


def get_memory_debug(memory_field, state=None):
    if memory_field is None:
        return {
            "memory_feature_count": 0,
            "memory_nearest_type": "none",
            "memory_nearest_distance": None,
            "memory_nearest_strength": 0.0,
            "memory_cost_total": 0.0,
            "memory_cost_by_type": {},
            "memory_temperature_scale": 1.0,
        }
    try:
        return memory_field.debug_snapshot(state)
    except TypeError:
        return memory_field.debug_snapshot()


def format_memory_cost_by_type(by_type):
    if not by_type:
        return ""
    parts = []
    for key in sorted(by_type.keys()):
        value = float(by_type.get(key, 0.0))
        if abs(value) <= 1e-9:
            continue
        parts.append("%s:%.3f" % (key, value))
    return "|".join(parts)


def memory_breakdown_for_trajectory(memory_field, trajectory, controls=None):
    if memory_field is None:
        return {
            "total": 0.0,
            "by_type": {},
            "nearest_type": "none",
            "nearest_distance": None,
            "nearest_strength": 0.0,
            "active_feature_count": 0,
            "temperature_scale": 1.0,
        }
    if hasattr(memory_field, "memory_cost_breakdown_for_trajectory"):
        try:
            return memory_field.memory_cost_breakdown_for_trajectory(
                trajectory,
                controls=controls,
                stride=3,
            )
        except TypeError:
            return memory_field.memory_cost_breakdown_for_trajectory(trajectory, stride=3)
    return {
        "total": memory_cost_for_trajectory(memory_field, trajectory),
        "by_type": {},
        "nearest_type": "none",
        "nearest_distance": None,
        "nearest_strength": 0.0,
        "active_feature_count": 0,
        "temperature_scale": 1.0,
    }


def nearest_memory_feature(memory_field, state):
    if memory_field is None:
        return None, None
    if hasattr(memory_field, "nearest_feature"):
        feature, distance = memory_field.nearest_feature(state)
        if feature is not None:
            return (float(feature.position[0]), float(feature.position[1])), distance

    best_position = None
    best_distance = None
    for position in get_memory_feature_positions(memory_field):
        distance = math.hypot(float(state[0]) - position[0], float(state[1]) - position[1])
        if best_distance is None or distance < best_distance:
            best_position = position
            best_distance = distance
    return best_position, best_distance


def normalize_xy(vector):
    x_value = float(vector[0])
    y_value = float(vector[1])
    norm = math.hypot(x_value, y_value)
    if norm <= 1e-9:
        return None
    return (x_value / norm, y_value / norm)


def position_span(states):
    if not states:
        return 0.0
    xs = [float(state[0]) for state in states]
    ys = [float(state[1]) for state in states]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def detect_spin_and_stuck(recent_states, recent_controls, recent_goal_distances):
    repeated_spin = False
    stuck_low_progress = False

    if len(recent_controls) >= 15 and len(recent_goal_distances) >= 15:
        controls_window = recent_controls[-20:]
        goals_window = recent_goal_distances[-(len(controls_window) + 1):]
        mean_abs_omega = sum(abs(item[1]) for item in controls_window) / float(len(controls_window))
        mean_v = sum(abs(item[0]) for item in controls_window) / float(len(controls_window))
        if len(goals_window) >= 2:
            goal_improvement = float(goals_window[0]) - float(goals_window[-1])
        else:
            goal_improvement = 0.0
        high_spin_steps = sum(
            1 for control in controls_window[-8:]
            if abs(control[1]) > 0.8 and abs(control[0]) < 0.08
        )
        repeated_spin = (
            mean_abs_omega > 0.45
            and mean_v < 0.08
            and goal_improvement < 0.08
        ) or high_spin_steps >= 5

    if len(recent_states) >= ESCAPE_WINDOW_STEPS and len(recent_goal_distances) >= ESCAPE_WINDOW_STEPS:
        states_window = recent_states[-ESCAPE_WINDOW_STEPS:]
        goals_window = recent_goal_distances[-ESCAPE_WINDOW_STEPS:]
        stuck_low_progress = (
            position_span(states_window) < 0.18
            and (float(goals_window[0]) - float(goals_window[-1])) < 0.10
        )

    return repeated_spin, stuck_low_progress


def apply_escape_bias_to_nominal_sequence(
    nominal_sequence,
    state,
    goal,
    nearest_feature_position,
    front_clear,
    horizon_prefix=7,
    preferred_escape_dir=None,
):
    if nearest_feature_position is None and preferred_escape_dir is None:
        return nominal_sequence, False, None

    current_xy = (float(state[0]), float(state[1]))
    escape_from_memory = normalize_xy(preferred_escape_dir) if preferred_escape_dir is not None else None
    if escape_from_memory is None and nearest_feature_position is not None:
        escape_from_memory = normalize_xy(
            (
                current_xy[0] - float(nearest_feature_position[0]),
                current_xy[1] - float(nearest_feature_position[1]),
            )
        )
    goal_dir = normalize_xy((float(goal[0]) - current_xy[0], float(goal[1]) - current_xy[1]))
    if escape_from_memory is None:
        escape_from_memory = goal_dir
    if goal_dir is None:
        goal_dir = escape_from_memory
    if escape_from_memory is None or goal_dir is None:
        return nominal_sequence, False, None

    escape_dir = normalize_xy(
        (
            0.65 * escape_from_memory[0] + 0.35 * goal_dir[0],
            0.65 * escape_from_memory[1] + 0.35 * goal_dir[1],
        )
    )
    if escape_dir is None:
        return nominal_sequence, False, None

    desired_heading = math.atan2(escape_dir[1], escape_dir[0])
    heading_error = wrap_angle(desired_heading - float(state[2]))
    omega = clamp(heading_error / 0.35, -0.8, 0.8)
    if front_clear:
        v_value = 0.14
    else:
        v_value = 0.10
    v_value = clamp(v_value, 0.10, 0.18)

    biased = list(nominal_sequence)
    prefix = min(len(biased), int(horizon_prefix))
    for index in range(prefix):
        blend = 1.0 - 0.10 * float(index)
        old_v, old_omega = biased[index]
        biased[index] = (
            clamp(blend * v_value + (1.0 - blend) * old_v, V_MIN, V_MAX),
            clamp(blend * omega + (1.0 - blend) * old_omega, -OMEGA_MAX, OMEGA_MAX),
        )
    return biased, True, escape_dir


def memory_cost_for_trajectory(memory_field, trajectory):
    if memory_field is None:
        return 0.0
    if hasattr(memory_field, "memory_cost_for_trajectory"):
        return float(memory_field.memory_cost_for_trajectory(trajectory, stride=3))
    if hasattr(memory_field, "cost_for_trajectory"):
        return float(memory_field.cost_for_trajectory(trajectory, step_stride=3))
    return 0.0


def effective_temperature(args, memory_field, state):
    scale = 1.0
    if memory_field is not None:
        if hasattr(memory_field, "temperature_scale_for_state"):
            scale = memory_field.temperature_scale_for_state(state)
        elif hasattr(memory_field, "temperature_scale"):
            scale = memory_field.temperature_scale(state)
    return float(args.temperature) * clamp(float(scale), 1.0, 3.0)


def memory_temperature_scale(memory_field, state):
    if memory_field is None:
        return 1.0
    if hasattr(memory_field, "temperature_scale_for_state"):
        return clamp(float(memory_field.temperature_scale_for_state(state)), 1.0, 3.0)
    if hasattr(memory_field, "temperature_scale"):
        return clamp(float(memory_field.temperature_scale(state)), 1.0, 3.0)
    return 1.0


def compute_goal_distance(state, goal=GOAL):
    return math.hypot(goal[0] - state[0], goal[1] - state[1])


def compute_rollout_cost(
    helpers,
    trajectory,
    control_sequence,
    planner_obstacles,
    memory_field,
    goal,
    bounds,
):
    result = helpers.trajectory_cost(
        trajectory=trajectory,
        control_sequence=control_sequence,
        goal=goal,
        obstacles=planner_obstacles,
        robot_radius=ROBOT_RADIUS,
        bounds=bounds,
    )
    if isinstance(result, tuple):
        base_cost, collided = result
    else:
        base_cost, collided = result, False
    memory_cost = memory_cost_for_trajectory(memory_field, trajectory)
    return float(base_cost) + memory_cost, bool(collided), memory_cost


def plan_one_step(
    helpers,
    args,
    state,
    nominal_sequence,
    planner_obstacles,
    memory_field,
    goal,
    bounds,
):
    t0 = timer_now()
    sampled_sequences = helpers.sample_control_sequences(
        nominal_sequence=nominal_sequence,
        current_state=state,
        dt=args.dt,
        obstacles=planner_obstacles,
        robot_radius=ROBOT_RADIUS,
        num_samples=args.num_samples,
        v_std=V_STD,
        omega_std=OMEGA_STD,
        v_min=V_MIN,
        v_max=V_MAX,
        omega_max=OMEGA_MAX,
        goal=goal,
        bounds=bounds,
    )

    costs = []
    trajectories = []
    collisions = []
    memory_costs = []
    for control_sequence in sampled_sequences:
        trajectory = helpers.rollout_control_sequence(state, control_sequence, args.dt)
        cost, collided, memory_cost = compute_rollout_cost(
            helpers,
            trajectory,
            control_sequence,
            planner_obstacles,
            memory_field,
            goal,
            bounds,
        )
        costs.append(cost)
        trajectories.append(trajectory)
        collisions.append(collided)
        memory_costs.append(memory_cost)

    temperature = effective_temperature(args, memory_field, state)
    weights = helpers.compute_weights(costs, temperature)
    updated_sequence = helpers.weighted_update_sequence(sampled_sequences, weights)
    updated_trajectory = helpers.rollout_control_sequence(state, updated_sequence, args.dt)
    _, _, updated_memory_cost = compute_rollout_cost(
        helpers,
        updated_trajectory,
        updated_sequence,
        planner_obstacles,
        memory_field,
        goal,
        bounds,
    )
    updated_memory_breakdown = memory_breakdown_for_trajectory(
        memory_field,
        updated_trajectory,
        controls=updated_sequence,
    )

    sorted_indices = sorted(range(len(costs)), key=lambda index: costs[index])
    best_index = sorted_indices[0]
    top_indices = sorted_indices[1:3]
    t1 = timer_now()

    return {
        "updated_sequence": updated_sequence,
        "proposed_control": updated_sequence[0],
        "best_trajectory": trajectories[best_index],
        "top_trajectories": [trajectories[index] for index in top_indices],
        "best_cost": costs[best_index],
        "best_collision": collisions[best_index],
        "memory_cost": updated_memory_cost,
        "memory_cost_total": float(updated_memory_breakdown.get("total", updated_memory_cost)),
        "memory_cost_by_type": updated_memory_breakdown.get("by_type", {}),
        "memory_nearest_type": updated_memory_breakdown.get("nearest_type", "none"),
        "memory_nearest_distance": updated_memory_breakdown.get("nearest_distance"),
        "memory_nearest_strength": updated_memory_breakdown.get("nearest_strength", 0.0),
        "temperature_scale": memory_temperature_scale(memory_field, state),
        "effective_temperature": temperature,
        "planner_compute_ms": (t1 - t0) * 1000.0,
    }


def apply_scan_guard_control(proposed_control, scan_result):
    v_value, omega_value = clamp_control(proposed_control)
    if scan_result.get("emergency_stop"):
        return (0.0, 0.0)
    if scan_result.get("should_slow_down"):
        v_value *= float(scan_result.get("slow_scale", 0.45))
    if scan_result.get("front_stop_mode") == "front_soft_block":
        v_value = min(v_value, 0.05)
    return clamp_control((v_value, omega_value))


def apply_anti_spin_final_control(final_control, repeated_spin, scan_result):
    if not repeated_spin:
        return final_control
    if scan_result.get("emergency_stop"):
        return final_control
    if scan_result.get("front_stop_mode") in ("hard_stop", "front_soft_block", "near_body_hard_stop"):
        return final_control

    min_front_range = scan_result.get("min_front_range")
    if min_front_range is None or float(min_front_range) <= 0.70:
        return final_control

    v_value, omega_value = final_control
    if abs(omega_value) > 0.70 and abs(v_value) < 0.05:
        return (0.08, clamp(omega_value, -0.45, 0.45))
    return final_control


def update_memory(memory_field, state, final_control, scan_result, now):
    if memory_field is None:
        return None
    front_stop_mode = scan_result.get("front_stop_mode", "clear")
    if scan_result.get("emergency_stop"):
        avoidance_state = "HARD_STOP_RECOVERY"
    elif front_stop_mode in ("front_soft_block", "front_obstacle_slow", "side_obstacle_soft"):
        avoidance_state = "CREEP_ESCAPE"
    else:
        avoidance_state = "CLEAR"
    return memory_field.update(
        state=state,
        goal_distance=compute_goal_distance(state),
        control=final_control,
        min_front_range=scan_result.get("min_front_range"),
        avoidance_state=avoidance_state,
        now=now,
    )


CSV_FIELDS = [
    "step",
    "time",
    "x",
    "y",
    "theta",
    "v",
    "omega",
    "goal_distance",
    "min_front_range",
    "scan_reason",
    "front_stop_mode",
    "planner_obstacle_count",
    "memory_feature_count",
    "memory_nearest_type",
    "memory_nearest_distance",
    "memory_nearest_strength",
    "memory_cost",
    "memory_cost_total",
    "memory_cost_by_type",
    "temperature_scale",
    "effective_temperature",
    "escape_active",
    "repeated_spin",
    "stuck_low_progress",
    "planner_compute_ms",
    "collision",
]


def open_csv_for_write(path):
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


def write_csv(path, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open_csv_for_write(path) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_live_sim(args):
    mujoco = require_mujoco()
    helpers = load_planner_helpers()
    scan_guard_module = load_bridge_module("_live_scan_guard", "scan_guard.py")
    local_layer_module = load_bridge_module("_live_local_obstacle_layer", "local_obstacle_layer.py")
    scene_config = SCENE_CONFIGS[args.scene]
    if tuple(scene_config["goal"]) != tuple(GOAL):
        raise RuntimeError("live scene goal must remain fixed at %s" % (GOAL,))

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = MujocoLiveMppiEnv(
        mujoco=mujoco,
        scene_config=scene_config,
        no_viewer=args.no_viewer,
    )
    memory_field = initialize_memory_field(args) if args.memory_enabled else None
    nominal_sequence = helpers.initialize_control_sequence((0.25, 0.0), args.horizon)
    csv_rows = []
    recent_states = [env.get_state()]
    recent_controls = []
    recent_goal_distances = [compute_goal_distance(env.get_state(), scene_config["goal"])]

    print("=== MuJoCo live MPPI scene: %s ===" % scene_config["name"])
    print("description = %s" % scene_config.get("description", ""))
    print("goal = %s" % (scene_config["goal"],))
    print("memory_enabled = %s" % args.memory_enabled)
    print("viewer_enabled = %s" % env.viewer_enabled)
    print("max_steps = %d" % args.max_steps)

    try:
        for step_index in range(int(args.max_steps)):
            state = env.get_state()
            scan = env.raycast_laserscan(state)
            scan_result = run_scan_guard(scan, scan_guard_module)
            planner_obstacles, layer_debug = run_local_obstacle_layer(
                scan,
                state,
                local_layer_module,
            )
            raw_planner_obstacle_count = len(planner_obstacles)
            planner_obstacles = filter_planner_obstacles(
                planner_obstacles,
                state,
                max_count=MAX_PLANNER_OBSTACLES,
            )

            repeated_spin, stuck_low_progress = detect_spin_and_stuck(
                recent_states,
                recent_controls,
                recent_goal_distances,
            )
            nearest_memory_position, nearest_memory_distance = nearest_memory_feature(
                memory_field,
                state,
            )
            suggested_escape_dir = None
            escape_debug = {}
            if memory_field is not None and hasattr(memory_field, "suggest_escape_direction"):
                suggested_escape_dir, escape_debug = memory_field.suggest_escape_direction(
                    state,
                    scene_config["goal"],
                )
            memory_feature_count_before = len(get_memory_feature_positions(memory_field))
            min_front_range = scan_result.get("min_front_range")
            front_clear = (
                min_front_range is not None
                and float(min_front_range) > 0.70
                and not scan_result.get("emergency_stop")
            )
            escape_active = False
            escape_dir = None
            planning_nominal_sequence = nominal_sequence
            if (
                memory_feature_count_before > 0
                and (repeated_spin or stuck_low_progress)
                and not scan_result.get("emergency_stop")
            ):
                planning_nominal_sequence, escape_active, escape_dir = apply_escape_bias_to_nominal_sequence(
                    nominal_sequence,
                    state,
                    scene_config["goal"],
                    nearest_memory_position,
                    front_clear=front_clear,
                    preferred_escape_dir=suggested_escape_dir,
                )

            plan_info = plan_one_step(
                helpers,
                args,
                state,
                planning_nominal_sequence,
                planner_obstacles,
                memory_field,
                scene_config["goal"],
                scene_config["bounds"],
            )
            proposed_control = clamp_control(plan_info["proposed_control"])
            final_control = apply_scan_guard_control(proposed_control, scan_result)
            final_control = apply_anti_spin_final_control(
                final_control,
                repeated_spin,
                scan_result,
            )
            next_state = env.step(final_control, args.dt)
            collision = env.has_robot_contact() or (
                primitive_clearance(next_state, env.primitives, ROBOT_RADIUS) <= 0.0
            )

            memory_debug = update_memory(
                memory_field,
                next_state,
                final_control,
                scan_result,
                now=(step_index + 1) * args.dt,
            )
            if memory_debug is None:
                memory_debug = get_memory_debug(memory_field, next_state)
            if memory_debug is None:
                memory_feature_count = 0
            else:
                memory_feature_count = int(memory_debug.get("memory_feature_count", 0))
            memory_nearest_type = memory_debug.get(
                "memory_nearest_type",
                plan_info.get("memory_nearest_type", "none"),
            )
            memory_nearest_distance = memory_debug.get(
                "memory_nearest_distance",
                plan_info.get("memory_nearest_distance"),
            )
            memory_nearest_strength = memory_debug.get(
                "memory_nearest_strength",
                plan_info.get("memory_nearest_strength", 0.0),
            )
            memory_cost_total = float(plan_info.get("memory_cost_total", plan_info["memory_cost"]))
            memory_cost_by_type = plan_info.get("memory_cost_by_type", {})
            temperature_scale = float(plan_info.get("temperature_scale", 1.0))

            env.update_debug_visuals(
                scan=scan,
                planner_obstacles=planner_obstacles,
                best_predicted_trajectory=plan_info["best_trajectory"],
                top_predicted_trajectories=plan_info["top_trajectories"],
                memory_feature_positions=get_memory_feature_positions(memory_field),
                escape_arrow=(
                    (
                        state[0],
                        state[1],
                    ),
                    (
                        state[0] + 0.45 * escape_dir[0],
                        state[1] + 0.45 * escape_dir[1],
                    ),
                ) if escape_active and escape_dir is not None else None,
            )
            if not args.no_viewer and step_index % args.render_every == 0:
                env.render(sleep_dt=args.render_sleep)

            nominal_sequence = helpers.shift_sequence(
                plan_info["updated_sequence"],
                tail_control=plan_info["updated_sequence"][-1],
            )

            goal_distance = compute_goal_distance(next_state, scene_config["goal"])
            recent_states.append(next_state)
            recent_controls.append(final_control)
            recent_goal_distances.append(goal_distance)
            recent_states = recent_states[-30:]
            recent_controls = recent_controls[-30:]
            recent_goal_distances = recent_goal_distances[-31:]
            csv_rows.append(
                {
                    "step": step_index + 1,
                    "time": (step_index + 1) * args.dt,
                    "x": next_state[0],
                    "y": next_state[1],
                    "theta": next_state[2],
                    "v": final_control[0],
                    "omega": final_control[1],
                    "goal_distance": goal_distance,
                    "min_front_range": scan_result.get("min_front_range"),
                    "scan_reason": scan_result.get("reason"),
                    "front_stop_mode": scan_result.get("front_stop_mode"),
                    "planner_obstacle_count": len(planner_obstacles),
                    "memory_feature_count": memory_feature_count,
                    "memory_nearest_type": memory_nearest_type,
                    "memory_nearest_distance": memory_nearest_distance,
                    "memory_nearest_strength": memory_nearest_strength,
                    "memory_cost": plan_info["memory_cost"],
                    "memory_cost_total": memory_cost_total,
                    "memory_cost_by_type": format_memory_cost_by_type(memory_cost_by_type),
                    "temperature_scale": temperature_scale,
                    "effective_temperature": plan_info["effective_temperature"],
                    "escape_active": bool(escape_active),
                    "repeated_spin": bool(repeated_spin),
                    "stuck_low_progress": bool(stuck_low_progress),
                    "planner_compute_ms": plan_info["planner_compute_ms"],
                    "collision": bool(collision),
                }
            )

            if step_index % 10 == 0:
                print(
                    "step=%03d scene=%s goal=%.3f repeated_spin=%s stuck_low_progress=%s escape_active=%s "
                    "proposed_v=%.3f proposed_omega=%.3f final_v=%.3f final_omega=%.3f "
                    "front_stop_mode=%s scan_reason=%s obs=%d raw_obs=%d memory=%d "
                    "nearest_memory_type=%s nearest_memory_distance=%s nearest_memory_strength=%.3f "
                    "memory_cost_total=%.3f temperature_scale=%.3f effective_temperature=%.3f compute_ms=%.2f"
                    % (
                        step_index + 1,
                        scene_config["name"],
                        goal_distance,
                        repeated_spin,
                        stuck_low_progress,
                        escape_active,
                        proposed_control[0],
                        proposed_control[1],
                        final_control[0],
                        final_control[1],
                        scan_result.get("front_stop_mode"),
                        scan_result.get("reason"),
                        len(planner_obstacles),
                        raw_planner_obstacle_count,
                        memory_feature_count,
                        memory_nearest_type,
                        "%.3f" % memory_nearest_distance if memory_nearest_distance is not None else "none",
                        float(memory_nearest_strength or 0.0),
                        memory_cost_total,
                        temperature_scale,
                        plan_info["effective_temperature"],
                        plan_info["planner_compute_ms"],
                    )
                )
                print(
                    "          front=%s mode=%s obs=%d v=%.3f omega=%.3f memory_by_type=%s escape_source=%s"
                    % (
                        scan_result.get("min_front_range"),
                        scan_result.get("front_stop_mode"),
                        len(planner_obstacles),
                        final_control[0],
                        final_control[1],
                        format_memory_cost_by_type(memory_cost_by_type),
                        escape_debug.get("source", "none"),
                    )
                )

            if collision:
                print("collision detected at step %d" % (step_index + 1))
                break
            if goal_distance < 0.25:
                print("goal reached at step %d" % (step_index + 1))
                break
    finally:
        if args.save_csv:
            write_csv(args.csv_path, csv_rows)
            print("csv_path=%s" % args.csv_path)
        env.close()

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time MuJoCo Memory-Augmented MPPI live simulation")
    parser.set_defaults(memory_enabled=True)
    parser.add_argument("--memory-enable", dest="memory_enabled", action="store_true", help="enable memory field")
    parser.add_argument("--memory-disable", dest="memory_enabled", action="store_false", help="disable memory field")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_CONFIGS.keys()),
        default="lab_complex",
        help="MuJoCo live scene to load",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-sleep", type=float, default=0.02)
    parser.add_argument("--no-viewer", action="store_true", help="run MuJoCo simulation without viewer")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument(
        "--csv-path",
        default=os.path.join("experiments", "results", "mujoco_live_mppi_trajectory.csv"),
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.render_every = max(1, int(args.render_every))
    args.max_steps = max(1, int(args.max_steps))
    args.horizon = max(1, int(args.horizon))
    args.num_samples = max(1, int(args.num_samples))
    return args


def main():
    args = parse_args()
    try:
        return run_live_sim(args)
    except RuntimeError as exc:
        print("ERROR: %s" % exc)
        return 2
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
