"""Actuated differential-drive MuJoCo true plant.

No silent kinematic fallback is provided: asking for this backend without a
working MuJoCo installation is an experiment configuration error.
"""

import math
from collections import deque
from typing import Mapping

import numpy as np

from mobile_robot_mppi.core.types import ControlCommand, GroundTruth, PlantStep, Pose2D, Twist2D
from .model_factory import build_diff_drive_mjcf, model_hash


def _yaw_from_quaternion(quaternion):
    w, x, y, z = (float(v) for v in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MujocoDiffDrivePlant:
    def __init__(self, config: Mapping[str, object], scene: Mapping[str, object]):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("mujoco_diff_drive requires the optional MuJoCo dependency") from exc
        self.mujoco = mujoco
        self.config = dict(config)
        self.scene = dict(scene)
        self.xml = build_diff_drive_mjcf(config, scene)
        self.xml_hash = model_hash(self.xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.profile = str(config.get("actuator", {}).get("profile", "torque_pi"))
        robot = config.get("robot", {})
        actuator = config.get("actuator", {})
        self.wheel_radius = float(robot.get("wheel_radius", 0.08))
        self.track_width = float(robot.get("track_width", 0.32))
        self.kp = float(actuator.get("kp", 0.55))
        self.ki = float(actuator.get("ki", 1.5))
        self.integral_limit = float(actuator.get("integral_limit", 2.5))
        self.torque_limit = float(actuator.get("torque_limit", 2.2))
        self.deadband = float(actuator.get("deadband", 0.0))
        self.command_delay = float(actuator.get("command_delay", 0.0))
        self.torque_slew_rate = float(actuator.get("torque_slew_rate", 50.0))
        self.left_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        self.right_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
        self.base_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self.base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self.left_actuator_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_drive")
        self.right_actuator_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_drive")
        self.left_qvel = int(self.model.jnt_dofadr[self.left_joint_id])
        self.right_qvel = int(self.model.jnt_dofadr[self.right_joint_id])
        self.base_qpos = int(self.model.jnt_qposadr[self.base_joint_id])
        self.base_qvel = int(self.model.jnt_dofadr[self.base_joint_id])
        self._integral = np.zeros(2, dtype=np.float64)
        self._last_torque = np.zeros(2, dtype=np.float64)
        self._last_control = ControlCommand(np.zeros(2), 0.0, "reset")
        self._delay_queue = deque()
        self._seed = 0
        self._obstacle_geom_ids = {
            index for index in range(self.model.ngeom)
            if (self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("obstacle_")
        }
        self._robot_geom_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("chassis", "left_wheel_geom", "right_wheel_geom", "front_caster", "rear_caster")
        }

    def _id(self, object_type, name):
        value = int(self.mujoco.mj_name2id(self.model, object_type, name))
        if value < 0:
            raise RuntimeError("MuJoCo object not found: %s" % name)
        return value

    @property
    def time(self):
        return float(self.data.time)

    def reset(self, seed: int, initial_state: np.ndarray) -> GroundTruth:
        state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if state.size < 3 or not np.isfinite(state).all():
            raise ValueError("MuJoCo initial state requires at least x, y, theta")
        self.mujoco.mj_resetData(self.model, self.data)
        adr = self.base_qpos
        self.data.qpos[adr:adr + 3] = (state[0], state[1], float(self.config.get("robot", {}).get("base_height", 0.0)) or self.model.body_pos[self.base_body_id, 2])
        half = 0.5 * float(state[2])
        self.data.qpos[adr + 3:adr + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
        self._integral[:] = 0.0
        self._last_torque[:] = 0.0
        self._last_control = ControlCommand(np.zeros(2), 0.0, "reset")
        self._delay_queue.clear()
        self._seed = int(seed)
        self.mujoco.mj_forward(self.model, self.data)
        return self.ground_truth()

    def wheel_targets(self, command):
        left = (command.v - 0.5 * self.track_width * command.omega) / self.wheel_radius
        right = (command.v + 0.5 * self.track_width * command.omega) / self.wheel_radius
        return np.asarray((left, right), dtype=np.float64)

    def _delayed_command(self, command, dt):
        delay_steps = max(0, int(round(self.command_delay / float(dt))))
        self._delay_queue.append(command)
        while len(self._delay_queue) > delay_steps + 1:
            self._delay_queue.popleft()
        if len(self._delay_queue) <= delay_steps:
            return ControlCommand(np.zeros(2), command.timestamp, "actuator_delay")
        return self._delay_queue[0]

    def _apply_actuator(self, targets, physics_dt):
        if self.profile == "ideal_velocity":
            self.data.ctrl[self.left_actuator_id] = targets[0]
            self.data.ctrl[self.right_actuator_id] = targets[1]
            return
        wheel_speeds = np.asarray(
            (self.data.qvel[self.left_qvel], self.data.qvel[self.right_qvel]),
            dtype=np.float64,
        )
        error = targets - wheel_speeds
        self._integral = np.clip(
            self._integral + physics_dt * error,
            -self.integral_limit,
            self.integral_limit,
        )
        torque = self.kp * error + self.ki * self._integral
        torque[np.abs(targets) < self.deadband] = 0.0
        max_delta = self.torque_slew_rate * physics_dt
        torque = np.clip(torque, self._last_torque - max_delta, self._last_torque + max_delta)
        torque = np.clip(torque, -self.torque_limit, self.torque_limit)
        self.data.ctrl[self.left_actuator_id] = torque[0]
        self.data.ctrl[self.right_actuator_id] = torque[1]
        self._last_torque[:] = torque

    def _collision(self):
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self._obstacle_geom_ids and pair & self._robot_geom_ids:
                return True
        return False

    def _minimum_clearance(self, pose):
        values = []
        robot_radius = float(self.config.get("robot", {}).get("collision_radius", 0.25))
        for obstacle in self.scene.get("obstacles", ()):
            position = obstacle.get("position", (0.0, 0.0))
            if str(obstacle.get("type", "cylinder")) == "box":
                size = obstacle.get("size", (0.25, 0.25))
                yaw = float(obstacle.get("yaw", 0.0))
                world_dx = pose.x - float(position[0])
                world_dy = pose.y - float(position[1])
                local_x = math.cos(yaw) * world_dx + math.sin(yaw) * world_dy
                local_y = -math.sin(yaw) * world_dx + math.cos(yaw) * world_dy
                outside_x = max(abs(local_x) - float(size[0]), 0.0)
                outside_y = max(abs(local_y) - float(size[1]), 0.0)
                outside = math.hypot(outside_x, outside_y)
                inside = min(max(abs(local_x) - float(size[0]), abs(local_y) - float(size[1])), 0.0)
                values.append(outside + inside - robot_radius)
            else:
                values.append(
                    math.hypot(pose.x - float(position[0]), pose.y - float(position[1]))
                    - robot_radius - float(obstacle.get("radius", 0.25))
                )
        return min(values) if values else float("inf")

    def ground_truth(self) -> GroundTruth:
        adr = self.base_qpos
        pose = Pose2D(
            float(self.data.qpos[adr]),
            float(self.data.qpos[adr + 1]),
            _yaw_from_quaternion(self.data.qpos[adr + 3:adr + 7]),
        )
        linear_world = self.data.qvel[self.base_qvel:self.base_qvel + 3]
        angular = self.data.qvel[self.base_qvel + 3:self.base_qvel + 6]
        v_value = float(linear_world[0] * math.cos(pose.theta) + linear_world[1] * math.sin(pose.theta))
        omega = float(angular[2])
        wheel_speeds = (
            float(self.data.qvel[self.left_qvel]),
            float(self.data.qvel[self.right_qvel]),
        )
        expected_v = 0.5 * self.wheel_radius * (wheel_speeds[0] + wheel_speeds[1])
        slip = abs(expected_v - v_value) / max(abs(expected_v), abs(v_value), 0.05)
        effort = (
            float(self.data.actuator_force[self.left_actuator_id]),
            float(self.data.actuator_force[self.right_actuator_id]),
        )
        clearance = self._minimum_clearance(pose)
        return GroundTruth(
            timestamp=self.time,
            pose=pose,
            twist=Twist2D(v_value, omega),
            wheel_speeds=wheel_speeds,
            actuator_effort=effort,
            collision=bool(self._collision() or clearance <= 0.0),
            minimum_clearance=float(clearance),
            slip_ratio=float(slip),
            metadata={
                "backend": "mujoco_diff_drive",
                "mujoco_version": self.mujoco.__version__,
                "model_hash": self.xml_hash,
                "actuator_profile": self.profile,
                "seed": self._seed,
            },
        )

    def step(self, command: ControlCommand, dt: float) -> PlantStep:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        delayed = self._delayed_command(command, dt)
        targets = self.wheel_targets(delayed)
        physics_dt = float(self.model.opt.timestep)
        substeps = max(1, int(round(float(dt) / physics_dt)))
        for _ in range(substeps):
            self._apply_actuator(targets, physics_dt)
            self.mujoco.mj_step(self.model, self.data)
        self._last_control = delayed
        return PlantStep(self.ground_truth(), delayed, float(dt))

    def close(self):
        return None
