"""Seeded odometry and MuJoCo ray-cast LaserScan sensor suite."""

import math
from collections import deque
from dataclasses import replace
from typing import Mapping, Optional

import numpy as np

from mobile_robot_mppi.core.types import GroundTruth, LaserScan, Pose2D, RobotObservation, Twist2D


class SimulatedSensorSuite:
    def __init__(self, plant, config: Mapping[str, object], seed: int = 0):
        self.plant = plant
        self.config = dict(config)
        self.rng = np.random.RandomState(int(seed))
        self.wheel_radius = float(config.get("wheel_radius", getattr(plant, "wheel_radius", 0.08)))
        self.track_width = float(config.get("track_width", getattr(plant, "track_width", 0.32)))
        self.odom_noise = np.asarray(config.get("odom_noise_std", (0.0, 0.0, 0.0)), dtype=np.float64)
        self.twist_noise = np.asarray(config.get("twist_noise_std", (0.0, 0.0)), dtype=np.float64)
        self.lidar_noise = float(config.get("lidar_noise_std", 0.0))
        self.dropout = float(config.get("lidar_dropout_probability", 0.0))
        self.latency = float(config.get("latency", 0.0))
        self.pose_source = str(config.get("pose_source", "wheel_odometry"))
        self.twist_source = str(config.get("twist_source", "wheel_odometry"))
        valid_sources = (
            "wheel_odometry",
            "wheel_imu_localized",
            "ground_truth",
        )
        if self.pose_source not in valid_sources:
            raise ValueError(
                "sensor pose_source must be wheel_odometry or ground_truth"
            )
        if self.twist_source not in valid_sources:
            raise ValueError(
                "sensor twist_source must be wheel_odometry or ground_truth"
            )
        self.num_beams = int(config.get("lidar_beams", 181))
        self.angle_min = float(config.get("lidar_angle_min", -math.pi))
        self.angle_max = float(config.get("lidar_angle_max", math.pi))
        self.range_min = float(config.get("lidar_range_min", 0.05))
        self.range_max = float(config.get("lidar_range_max", 4.0))
        self._odom_pose = np.zeros(3, dtype=np.float64)
        self._localized_pose = np.zeros(3, dtype=np.float64)
        self.localization_update_period = float(
            config.get("localization_update_period_s", 0.5)
        )
        self.localization_pose_noise = np.asarray(
            config.get(
                "localization_pose_noise_std",
                (0.015, 0.015, math.radians(0.5)),
            ),
            dtype=np.float64,
        )
        self.localization_correction_gain = np.asarray(
            config.get("localization_correction_gain", (0.85, 0.85, 0.85)),
            dtype=np.float64,
        )
        self.imu_yaw_rate_noise = float(
            config.get("imu_yaw_rate_noise_std", 0.003)
        )
        self.imu_yaw_rate_bias_std = float(
            config.get("imu_yaw_rate_bias_std", 0.001)
        )
        if self.localization_update_period <= 0.0:
            raise ValueError("localization_update_period_s must be positive")
        if self.localization_pose_noise.shape != (3,) or np.any(
            self.localization_pose_noise < 0.0
        ):
            raise ValueError(
                "localization_pose_noise_std must contain three "
                "non-negative values"
            )
        if self.localization_correction_gain.shape != (3,) or np.any(
            (self.localization_correction_gain < 0.0)
            | (self.localization_correction_gain > 1.0)
        ):
            raise ValueError(
                "localization_correction_gain must contain three values "
                "in [0, 1]"
            )
        if self.imu_yaw_rate_noise < 0.0 or self.imu_yaw_rate_bias_std < 0.0:
            raise ValueError("IMU noise values must be non-negative")
        self._imu_yaw_rate_bias = 0.0
        self._last_localization_update_time = None
        self._last_time = None
        self._queue = deque()
        self._last_observation = None

    def reset(self, truth: GroundTruth, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.RandomState(int(seed))
        self._odom_pose[:] = truth.pose.as_array()
        self._localized_pose[:] = truth.pose.as_array()
        localized_enabled = (
            self.pose_source == "wheel_imu_localized"
            or self.twist_source == "wheel_imu_localized"
        )
        self._imu_yaw_rate_bias = (
            float(self.rng.normal(0.0, self.imu_yaw_rate_bias_std))
            if localized_enabled and self.imu_yaw_rate_bias_std > 0.0
            else 0.0
        )
        self._last_localization_update_time = float(truth.timestamp)
        self._last_time = float(truth.timestamp)
        self._queue.clear()
        self._last_observation = None
        return self.observe(truth)

    @staticmethod
    def _wrap(value):
        return math.atan2(math.sin(value), math.cos(value))

    def _integrate_odometry(self, truth: GroundTruth):
        if self._last_time is None:
            self._last_time = truth.timestamp
        dt = max(0.0, float(truth.timestamp) - float(self._last_time))
        left, right = truth.wheel_speeds
        if getattr(self.plant, "wheel_radius", None) is None:
            v_value, omega = truth.twist.v, truth.twist.omega
        else:
            v_value = 0.5 * self.wheel_radius * (left + right)
            omega = self.wheel_radius * (right - left) / self.track_width
        theta = self._odom_pose[2]
        self._odom_pose[0] += v_value * math.cos(theta) * dt
        self._odom_pose[1] += v_value * math.sin(theta) * dt
        self._odom_pose[2] = self._wrap(theta + omega * dt)
        self._last_time = float(truth.timestamp)
        return v_value, omega

    def _integrate_localized_odometry(
        self,
        truth: GroundTruth,
        v_value: float,
    ):
        """Fuse wheel speed, IMU yaw rate, and periodic external pose fixes.

        The external pose fix is a noisy measurement of the simulated chassis
        pose.  It represents a SLAM/mocap/GNSS-like localization interface and
        never exposes the exact simulator state to the controller.
        """

        if self._last_time is None:
            dt = 0.0
        else:
            dt = max(
                0.0,
                float(truth.timestamp) - float(self._last_time),
            )
        imu_omega = (
            float(truth.twist.omega)
            + self._imu_yaw_rate_bias
            + (
                float(self.rng.normal(0.0, self.imu_yaw_rate_noise))
                if self.imu_yaw_rate_noise > 0.0
                else 0.0
            )
        )
        theta = float(self._localized_pose[2])
        theta_mid = theta + 0.5 * imu_omega * dt
        self._localized_pose[0] += v_value * math.cos(theta_mid) * dt
        self._localized_pose[1] += v_value * math.sin(theta_mid) * dt
        self._localized_pose[2] = self._wrap(theta + imu_omega * dt)

        last_update = self._last_localization_update_time
        update_due = (
            last_update is None
            or float(truth.timestamp) - float(last_update)
            >= self.localization_update_period - 1.0e-12
        )
        if update_due:
            measurement = truth.pose.as_array().astype(
                np.float64, copy=True
            )
            if np.any(self.localization_pose_noise):
                measurement += self.rng.normal(
                    0.0, self.localization_pose_noise
                )
            innovation = measurement - self._localized_pose
            innovation[2] = self._wrap(float(innovation[2]))
            self._localized_pose += (
                self.localization_correction_gain * innovation
            )
            self._localized_pose[2] = self._wrap(
                float(self._localized_pose[2])
            )
            self._last_localization_update_time = float(truth.timestamp)
        return imu_omega

    def _raycast(self, truth: GroundTruth):
        if not hasattr(self.plant, "mujoco"):
            return self._raycast_analytic(truth)
        mujoco = self.plant.mujoco
        model = self.plant.model
        data = self.plant.data
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site"))
        origin = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        angle_increment = (self.angle_max - self.angle_min) / float(self.num_beams - 1)
        ranges = np.full(self.num_beams, self.range_max, dtype=np.float64)
        obstacle_ranges = np.full(self.num_beams, np.inf, dtype=np.float64)
        geom_group = np.asarray((0, 1, 0, 0, 0, 0), dtype=np.uint8)
        for index in range(self.num_beams):
            angle = truth.pose.theta + self.angle_min + index * angle_increment
            direction = np.asarray((math.cos(angle), math.sin(angle), 0.0), dtype=np.float64)
            geom_id = np.asarray((-1,), dtype=np.int32)
            distance = mujoco.mj_ray(
                model, data, origin, direction, geom_group, 1,
                int(self.plant.base_body_id), geom_id,
            )
            if distance is not None and self.range_min <= float(distance) <= self.range_max:
                value = float(distance)
                if self.lidar_noise > 0.0:
                    value += float(self.rng.normal(0.0, self.lidar_noise))
                value = float(np.clip(value, self.range_min, self.range_max))
                if self.dropout > 0.0 and self.rng.rand() < self.dropout:
                    value = self.range_max
                else:
                    obstacle_ranges[index] = value
                ranges[index] = value
        return LaserScan(
            ranges=ranges,
            obstacle_ranges=obstacle_ranges,
            angle_min=self.angle_min,
            angle_increment=angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
            timestamp=truth.timestamp,
            frame_id="lidar",
        )

    def _raycast_analytic(self, truth: GroundTruth):
        obstacles = getattr(self.plant, "obstacles", ())
        if not obstacles:
            return None
        increment = (self.angle_max - self.angle_min) / float(self.num_beams - 1)
        ranges = np.full(self.num_beams, self.range_max, dtype=np.float64)
        obstacle_ranges = np.full(self.num_beams, np.inf, dtype=np.float64)
        origin = np.asarray((truth.pose.x, truth.pose.y), dtype=np.float64)
        for index in range(self.num_beams):
            angle = truth.pose.theta + self.angle_min + index * increment
            direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
            best = self.range_max
            for obstacle in obstacles:
                center = np.asarray(obstacle[:2], dtype=np.float64)
                radius = float(obstacle[2]) if len(obstacle) >= 3 else 0.08
                offset = origin - center
                projection = float(np.dot(direction, offset))
                discriminant = projection * projection - (float(np.dot(offset, offset)) - radius * radius)
                if discriminant < 0.0:
                    continue
                root = -projection - math.sqrt(discriminant)
                if self.range_min <= root < best:
                    best = root
            if best < self.range_max:
                obstacle_ranges[index] = best
                ranges[index] = best
        return LaserScan(
            ranges=ranges,
            obstacle_ranges=obstacle_ranges,
            angle_min=self.angle_min,
            angle_increment=increment,
            range_min=self.range_min,
            range_max=self.range_max,
            timestamp=truth.timestamp,
            frame_id="synthetic_lidar",
        )

    def observe(self, truth: GroundTruth) -> RobotObservation:
        localized_enabled = (
            self.pose_source == "wheel_imu_localized"
            or self.twist_source == "wheel_imu_localized"
        )
        previous_time = self._last_time
        odom_v, odom_omega = self._integrate_odometry(truth)
        localized_omega = odom_omega
        if localized_enabled:
            self._last_time = previous_time
            localized_omega = self._integrate_localized_odometry(
                truth, odom_v
            )
            self._last_time = float(truth.timestamp)
        if self.pose_source == "ground_truth":
            pose_value = truth.pose.as_array()
        elif self.pose_source == "wheel_imu_localized":
            pose_value = self._localized_pose.copy()
        else:
            pose_value = self._odom_pose.copy()
        if self.twist_source == "ground_truth":
            twist_value = truth.twist.as_array()
        elif self.twist_source == "wheel_imu_localized":
            twist_value = np.asarray(
                (odom_v, localized_omega), dtype=np.float64
            )
        else:
            twist_value = np.asarray(
                (odom_v, odom_omega), dtype=np.float64
            )
        pose_noise = self.rng.normal(0.0, self.odom_noise) if np.any(self.odom_noise) else np.zeros(3)
        twist_noise = self.rng.normal(0.0, self.twist_noise) if np.any(self.twist_noise) else np.zeros(2)
        observation = RobotObservation(
            timestamp=truth.timestamp,
            pose=Pose2D(
                float(pose_value[0] + pose_noise[0]),
                float(pose_value[1] + pose_noise[1]),
                self._wrap(float(pose_value[2] + pose_noise[2])),
            ),
            twist=Twist2D(
                float(twist_value[0] + twist_noise[0]),
                float(twist_value[1] + twist_noise[1]),
            ),
            scan=self._raycast(truth),
            auxiliary={
                "wheel_left": truth.wheel_speeds[0],
                "wheel_right": truth.wheel_speeds[1],
                "pose_source": self.pose_source,
                "twist_source": self.twist_source,
                "localization_update_period_s": (
                    self.localization_update_period
                    if self.pose_source == "wheel_imu_localized"
                    else 0.0
                ),
            },
        )
        self._queue.append(observation)
        release_time = float(truth.timestamp) - self.latency
        while len(self._queue) > 1 and self._queue[1].timestamp <= release_time:
            self._queue.popleft()
        if self._queue and self._queue[0].timestamp <= release_time:
            self._last_observation = self._queue[0]
        elif self._last_observation is None:
            self._last_observation = observation
        return self._last_observation
