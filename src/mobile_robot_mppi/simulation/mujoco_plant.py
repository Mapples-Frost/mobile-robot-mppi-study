"""Actuated differential-drive MuJoCo true plant.

No silent kinematic fallback is provided: asking for this backend without a
working MuJoCo installation is an experiment configuration error.
"""

import math
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from mobile_robot_mppi.core.types import ControlCommand, GroundTruth, PlantStep, Pose2D, Twist2D
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import (
    V3_PROCESS_NAMES,
    generate_patrol_trajectory,
    validate_v3_config,
)
from .model_factory import build_diff_drive_mjcf, model_hash


def _yaw_from_quaternion(quaternion):
    w, x, y, z = (float(v) for v in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class MujocoPlantSnapshot:
    """Complete continuation state for deterministic counterfactual branches.

    MuJoCo's integration state captures the simulator quantities required to
    continue a trajectory.  The remaining fields capture this plant wrapper's
    PI actuator, delayed-command queue and seeded obstacle episode state.
    """

    model_hash: str
    time: float
    integration_state: np.ndarray
    integral: np.ndarray
    last_torque: np.ndarray
    last_control: ControlCommand
    delay_queue: tuple
    seed: int
    episode_dynamic_obstacles: tuple


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
        self._obstacle_geom_by_index = {
            index: self._id(mujoco.mjtObj.mjOBJ_GEOM, "obstacle_%d" % index)
            for index, _ in enumerate(self.scene.get("obstacles", ()))
        }
        self._obstacle_geom_ids = set(self._obstacle_geom_by_index.values())
        self._dynamic_obstacles = self._resolve_dynamic_obstacles()
        self._episode_dynamic_obstacles = self._dynamic_obstacles
        self._robot_geom_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("chassis", "left_wheel_geom", "right_wheel_geom", "front_caster", "rear_caster")
        }

    def _id(self, object_type, name):
        value = int(self.mujoco.mj_name2id(self.model, object_type, name))
        if value < 0:
            raise RuntimeError("MuJoCo object not found: %s" % name)
        return value

    @staticmethod
    def _finite_pair(value, name):
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError("%s must contain two finite values" % name)
        return array

    def _resolve_dynamic_obstacles(self):
        resolved = []
        for index, obstacle in enumerate(self.scene.get("obstacles", ())):
            motion = obstacle.get("motion")
            if motion is None:
                continue
            if not isinstance(motion, Mapping):
                raise TypeError("obstacle motion must be a mapping")
            motion_type = str(motion.get("type", "linear_ping_pong"))
            if motion_type not in (
                "linear_ping_pong",
                "recurrent_semimarkov_v3",
            ):
                raise ValueError("unknown obstacle motion type: %s" % motion_type)
            if motion_type == "recurrent_semimarkov_v3":
                config_path = Path(str(motion.get("config_path", "")))
                if not config_path.is_absolute() or not config_path.is_file():
                    raise ValueError(
                        "V3 obstacle config path must be an existing "
                        "absolute file"
                    )
                with config_path.open("r", encoding="utf-8") as handle:
                    obstacle_config = yaml.safe_load(handle)
                validate_v3_config(obstacle_config)
                process = str(motion.get("process", "hybrid_patrol"))
                if process not in V3_PROCESS_NAMES:
                    raise ValueError("unknown V3 obstacle process")
                profile = str(motion.get("noise_profile", "medium"))
                if profile not in obstacle_config["noise_profiles"]:
                    raise ValueError("unknown V3 obstacle noise profile")
                use_episode_seed = bool(
                    motion.get("use_episode_seed", True)
                )
                fixed_seed = int(motion.get("seed", 0))
                if not use_episode_seed and fixed_seed <= 0:
                    raise ValueError(
                        "fixed V3 obstacle seed must be positive"
                    )
                body_id = self._id(
                    self.mujoco.mjtObj.mjOBJ_BODY,
                    "dynamic_obstacle_%d" % index,
                )
                mocap_id = int(self.model.body_mocapid[body_id])
                if mocap_id < 0:
                    raise RuntimeError(
                        "dynamic obstacle body is not a mocap body"
                    )
                resolved.append(
                    {
                        "index": index,
                        "mocap_id": mocap_id,
                        "motion_type": motion_type,
                        "process": process,
                        "noise_profile": profile,
                        "use_episode_seed": use_episode_seed,
                        "fixed_seed": fixed_seed,
                        "obstacle_config": obstacle_config,
                        "config_path": str(config_path),
                        "yaw": float(obstacle.get("yaw", 0.0)),
                    }
                )
                continue
            start = self._finite_pair(
                motion.get("start", obstacle.get("position", (0.0, 0.0))),
                "dynamic obstacle start",
            )
            end = self._finite_pair(motion.get("end"), "dynamic obstacle end")
            period = float(motion.get("period_s", 0.0))
            phase = float(motion.get("phase_s", 0.0))
            phase_jitter = float(motion.get("phase_jitter_s", 0.0))
            period_scale = np.asarray(
                motion.get("period_scale_range", (1.0, 1.0)),
                dtype=np.float64,
            ).reshape(-1)
            endpoint_jitter = float(motion.get("endpoint_jitter_m", 0.0))
            yaw = float(obstacle.get("yaw", 0.0))
            if not math.isfinite(period) or period <= 0.0:
                raise ValueError("dynamic obstacle period_s must be positive")
            if not math.isfinite(phase) or not math.isfinite(yaw):
                raise ValueError("dynamic obstacle phase and yaw must be finite")
            if not math.isfinite(phase_jitter) or phase_jitter < 0.0:
                raise ValueError("dynamic obstacle phase_jitter_s must be non-negative")
            if (
                period_scale.shape != (2,)
                or not np.isfinite(period_scale).all()
                or np.any(period_scale <= 0.0)
                or period_scale[1] < period_scale[0]
            ):
                raise ValueError(
                    "dynamic obstacle period_scale_range must be two ordered "
                    "positive values"
                )
            if not math.isfinite(endpoint_jitter) or endpoint_jitter < 0.0:
                raise ValueError(
                    "dynamic obstacle endpoint_jitter_m must be non-negative"
                )
            body_id = self._id(
                self.mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle_%d" % index
            )
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError("dynamic obstacle body is not a mocap body")
            resolved.append({
                "index": index,
                "mocap_id": mocap_id,
                "motion_type": motion_type,
                "start": start,
                "end": end,
                "period_s": period,
                "phase_s": phase,
                "phase_jitter_s": phase_jitter,
                "period_scale_range": period_scale,
                "endpoint_jitter_m": endpoint_jitter,
                "yaw": yaw,
            })
        return tuple(resolved)

    def _sample_dynamic_obstacles(self, seed):
        """Resolve reproducible training-domain randomization at reset.

        Randomized motion remains part of the true plant only.  The planner
        observes its consequences through LaserScan and never receives these
        sampled parameters.
        """

        rng = np.random.RandomState((int(seed) ^ 0x5EEDC0DE) & 0xFFFFFFFF)
        sampled = []
        for base in self._dynamic_obstacles:
            item = dict(base)
            if base["motion_type"] == "recurrent_semimarkov_v3":
                trajectory_seed = (
                    int(seed)
                    if bool(base["use_episode_seed"])
                    else int(base["fixed_seed"])
                )
                profiles = noise_profiles_from_mapping(
                    base["obstacle_config"]["noise_profiles"]
                )
                item["trajectory_seed"] = trajectory_seed
                item["trajectory"] = generate_patrol_trajectory(
                    base["process"],
                    trajectory_seed,
                    profiles[base["noise_profile"]],
                    base["obstacle_config"],
                )
                item["current_yaw"] = float(base["yaw"])
                sampled.append(item)
                continue
            phase_jitter = float(base["phase_jitter_s"])
            item["phase_s"] = float(base["phase_s"]) + (
                rng.uniform(-phase_jitter, phase_jitter)
                if phase_jitter > 0.0 else 0.0
            )
            scale_low, scale_high = base["period_scale_range"]
            scale = (
                rng.uniform(scale_low, scale_high)
                if scale_high > scale_low else float(scale_low)
            )
            item["period_s"] = float(base["period_s"]) * float(scale)
            endpoint_jitter = float(base["endpoint_jitter_m"])
            if endpoint_jitter > 0.0:
                item["start"] = base["start"] + rng.uniform(
                    -endpoint_jitter, endpoint_jitter, size=2
                )
                item["end"] = base["end"] + rng.uniform(
                    -endpoint_jitter, endpoint_jitter, size=2
                )
            else:
                item["start"] = base["start"].copy()
                item["end"] = base["end"].copy()
            sampled.append(item)
        return tuple(sampled)

    @staticmethod
    def _ping_pong_fraction(time_value, period, phase):
        cycle = (float(time_value) + float(phase)) % float(period)
        half = 0.5 * float(period)
        return cycle / half if cycle <= half else (float(period) - cycle) / half

    def _set_dynamic_obstacles(self, time_value):
        for item in self._episode_dynamic_obstacles:
            if item["motion_type"] == "recurrent_semimarkov_v3":
                trajectory = item["trajectory"]
                timestamp = float(
                    np.clip(
                        float(time_value),
                        float(trajectory.times[0]),
                        float(trajectory.times[-1]),
                    )
                )
                upper = int(
                    np.searchsorted(
                        trajectory.times, timestamp, side="right"
                    )
                )
                if upper <= 0:
                    state = trajectory.states[0]
                elif upper >= trajectory.times.size:
                    state = trajectory.states[-1]
                else:
                    lower = upper - 1
                    interval = float(
                        trajectory.times[upper]
                        - trajectory.times[lower]
                    )
                    alpha = (
                        timestamp - float(trajectory.times[lower])
                    ) / interval
                    state = (
                        (1.0 - alpha) * trajectory.states[lower]
                        + alpha * trajectory.states[upper]
                    )
                position = state[:2]
                speed = float(np.linalg.norm(state[2:4]))
                yaw = (
                    float(np.arctan2(state[3], state[2]))
                    if speed > 1.0e-4
                    else float(item["current_yaw"])
                )
                item["current_yaw"] = yaw
                mocap_id = item["mocap_id"]
                self.data.mocap_pos[mocap_id] = (
                    position[0],
                    position[1],
                    0.0,
                )
                self.data.mocap_quat[mocap_id] = (
                    math.cos(0.5 * yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * yaw),
                )
                continue
            fraction = self._ping_pong_fraction(
                time_value, item["period_s"], item["phase_s"]
            )
            position = item["start"] + fraction * (item["end"] - item["start"])
            mocap_id = item["mocap_id"]
            self.data.mocap_pos[mocap_id] = (position[0], position[1], 0.0)
            half_yaw = 0.5 * item["yaw"]
            self.data.mocap_quat[mocap_id] = (
                math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)
            )

    def dynamic_obstacle_states(self):
        """Return current prescribed obstacle poses for metrics and auditing."""

        output = []
        for item in self._episode_dynamic_obstacles:
            geom_id = self._obstacle_geom_by_index[item["index"]]
            position = self.data.geom_xpos[geom_id]
            if item["motion_type"] == "recurrent_semimarkov_v3":
                output.append({
                    "index": int(item["index"]),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "motion_type": item["motion_type"],
                    "process": item["process"],
                    "noise_profile": item["noise_profile"],
                    "trajectory_seed": int(item["trajectory_seed"]),
                    "trajectory_duration_s": float(
                        item["trajectory"].times[-1]
                    ),
                })
                continue
            output.append({
                "index": int(item["index"]),
                "x": float(position[0]),
                "y": float(position[1]),
                "motion_type": item["motion_type"],
                "period_s": float(item["period_s"]),
                "phase_s": float(item["phase_s"]),
                "start": [float(value) for value in item["start"]],
                "end": [float(value) for value in item["end"]],
            })
        return output

    @property
    def time(self):
        return float(self.data.time)

    def snapshot(self) -> MujocoPlantSnapshot:
        """Capture an exact, model-bound state for offline branch rollouts."""

        state_spec = self.mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(
            self.mujoco.mj_stateSize(self.model, state_spec), dtype=np.float64
        )
        self.mujoco.mj_getState(self.model, self.data, state, state_spec)
        queue = tuple(
            (
                float(activation_time),
                ControlCommand(
                    command.values.copy(),
                    float(command.timestamp),
                    str(command.source),
                ),
            )
            for activation_time, command in self._delay_queue
        )
        return MujocoPlantSnapshot(
            model_hash=self.xml_hash,
            time=self.time,
            integration_state=state.copy(),
            integral=self._integral.copy(),
            last_torque=self._last_torque.copy(),
            last_control=ControlCommand(
                self._last_control.values.copy(),
                float(self._last_control.timestamp),
                str(self._last_control.source),
            ),
            delay_queue=queue,
            seed=int(self._seed),
            episode_dynamic_obstacles=deepcopy(
                tuple(self._episode_dynamic_obstacles)
            ),
        )

    def restore(self, snapshot: MujocoPlantSnapshot) -> GroundTruth:
        """Restore a snapshot and return the reconstructed ground truth.

        Snapshots cannot be moved between different MJCF models.  This strict
        check prevents an apparently valid but scientifically invalid branch
        when plant parameters or scene geometry differ.
        """

        if not isinstance(snapshot, MujocoPlantSnapshot):
            raise TypeError("snapshot must be a MujocoPlantSnapshot")
        if snapshot.model_hash != self.xml_hash:
            raise ValueError("snapshot belongs to a different MuJoCo model")
        state_spec = self.mujoco.mjtState.mjSTATE_INTEGRATION
        expected_size = self.mujoco.mj_stateSize(self.model, state_spec)
        state = np.asarray(snapshot.integration_state, dtype=np.float64).reshape(-1)
        if state.shape != (expected_size,) or not np.isfinite(state).all():
            raise ValueError("snapshot integration state is invalid")
        integral = np.asarray(snapshot.integral, dtype=np.float64).reshape(-1)
        torque = np.asarray(snapshot.last_torque, dtype=np.float64).reshape(-1)
        if (
            integral.shape != (2,)
            or torque.shape != (2,)
            or not np.isfinite(integral).all()
            or not np.isfinite(torque).all()
        ):
            raise ValueError("snapshot actuator state is invalid")

        self.mujoco.mj_setState(self.model, self.data, state, state_spec)
        self._integral[:] = integral
        self._last_torque[:] = torque
        self._last_control = ControlCommand(
            snapshot.last_control.values.copy(),
            float(snapshot.last_control.timestamp),
            str(snapshot.last_control.source),
        )
        self._delay_queue = deque(
            (
                float(activation_time),
                ControlCommand(
                    command.values.copy(),
                    float(command.timestamp),
                    str(command.source),
                ),
            )
            for activation_time, command in snapshot.delay_queue
        )
        self._seed = int(snapshot.seed)
        self._episode_dynamic_obstacles = deepcopy(
            tuple(snapshot.episode_dynamic_obstacles)
        )
        self.mujoco.mj_forward(self.model, self.data)
        return self.ground_truth()

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
        self._episode_dynamic_obstacles = self._sample_dynamic_obstacles(seed)
        self._set_dynamic_obstacles(0.0)
        self.mujoco.mj_forward(self.model, self.data)
        return self.ground_truth()

    def wheel_targets(self, command):
        left = (command.v - 0.5 * self.track_width * command.omega) / self.wheel_radius
        right = (command.v + 0.5 * self.track_width * command.omega) / self.wheel_radius
        return np.asarray((left, right), dtype=np.float64)

    def _schedule_command(self, command):
        activation_time = self.time + self.command_delay
        self._delay_queue.append((activation_time, command))

    def _activate_due_commands(self):
        # Evaluate delay at the MuJoCo physics clock, not at the slower outer
        # controller rate.  This preserves delays such as 40 ms when the MPPI
        # control period is 100 ms.
        tolerance = 1e-12
        while self._delay_queue and self._delay_queue[0][0] <= self.time + tolerance:
            _, self._last_control = self._delay_queue.popleft()
        return self._last_control

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
        for index, obstacle in enumerate(self.scene.get("obstacles", ())):
            geom_id = self._obstacle_geom_by_index[index]
            position = self.data.geom_xpos[geom_id]
            if str(obstacle.get("type", "cylinder")) == "box":
                size = obstacle.get("size", (0.25, 0.25))
                rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
                yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
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
        dynamic_states = self.dynamic_obstacle_states()
        nearest_dynamic_center = min(
            (
                math.hypot(pose.x - item["x"], pose.y - item["y"])
                for item in dynamic_states
            ),
            default=float("inf"),
        )
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
                "dynamic_obstacle_count": len(dynamic_states),
                "dynamic_obstacles": dynamic_states,
                "nearest_dynamic_obstacle_center_distance": (
                    float(nearest_dynamic_center)
                    if math.isfinite(nearest_dynamic_center)
                    else None
                ),
            },
        )

    def step(self, command: ControlCommand, dt: float) -> PlantStep:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")
        physics_dt = float(self.model.opt.timestep)
        ratio = float(dt) / physics_dt
        substeps = int(round(ratio))
        if substeps < 1 or not math.isclose(ratio, substeps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("dt must be a positive integer multiple of the MuJoCo timestep")
        self._schedule_command(command)
        applied_values = []
        for _ in range(substeps):
            delayed = self._activate_due_commands()
            applied_values.append(delayed.values.copy())
            targets = self.wheel_targets(delayed)
            self._apply_actuator(targets, physics_dt)
            self._set_dynamic_obstacles(self.time + physics_dt)
            self.mujoco.mj_step(self.model, self.data)
        return PlantStep(
            self.ground_truth(),
            self._last_control,
            float(dt),
            metadata={
                "average_applied_control": np.mean(applied_values, axis=0).tolist(),
                "command_delay": self.command_delay,
                "physics_substeps": substeps,
            },
        )

    def close(self):
        return None
