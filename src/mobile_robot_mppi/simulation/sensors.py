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
        self.num_beams = int(config.get("lidar_beams", 181))
        self.angle_min = float(config.get("lidar_angle_min", -math.pi))
        self.angle_max = float(config.get("lidar_angle_max", math.pi))
        self.range_min = float(config.get("lidar_range_min", 0.05))
        self.range_max = float(config.get("lidar_range_max", 4.0))
        self._odom_pose = np.zeros(3, dtype=np.float64)
        self._last_time = None
        self._queue = deque()
        self._last_observation = None

    def reset(self, truth: GroundTruth, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.RandomState(int(seed))
        self._odom_pose[:] = truth.pose.as_array()
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
        v_value, omega = self._integrate_odometry(truth)
        pose_noise = self.rng.normal(0.0, self.odom_noise) if np.any(self.odom_noise) else np.zeros(3)
        twist_noise = self.rng.normal(0.0, self.twist_noise) if np.any(self.twist_noise) else np.zeros(2)
        observation = RobotObservation(
            timestamp=truth.timestamp,
            pose=Pose2D(
                float(self._odom_pose[0] + pose_noise[0]),
                float(self._odom_pose[1] + pose_noise[1]),
                self._wrap(float(self._odom_pose[2] + pose_noise[2])),
            ),
            twist=Twist2D(float(v_value + twist_noise[0]), float(omega + twist_noise[1])),
            scan=self._raycast(truth),
            auxiliary={
                "wheel_left": truth.wheel_speeds[0],
                "wheel_right": truth.wheel_speeds[1],
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
