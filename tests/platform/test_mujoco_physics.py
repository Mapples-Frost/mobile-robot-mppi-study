import importlib.util

import numpy as np
import pytest

from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant
from mobile_robot_mppi.simulation.sensors import SimulatedSensorSuite


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None


def plant_config(profile="ideal_velocity"):
    return {
        "robot": {"wheel_radius": 0.08, "track_width": 0.32},
        "actuator": {"profile": profile, "torque_limit": 3.0, "velocity_gain": 8.0, "kp": 0.8, "ki": 2.0},
        "contact": {},
        "physics": {"timestep": 0.002, "integrator": "implicitfast"},
    }


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_model_has_actuators_and_sensors():
    plant = MujocoDiffDrivePlant(plant_config(), {"obstacles": []})
    try:
        assert plant.model.nu == 2
        assert plant.model.nsensor >= 8
    finally:
        plant.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_equal_wheel_targets_produce_forward_motion():
    plant = MujocoDiffDrivePlant(plant_config(), {"obstacles": []})
    try:
        plant.reset(1, np.zeros(3))
        for index in range(20):
            plant.step(ControlCommand(np.asarray((0.2, 0.0)), index * 0.05), 0.05)
        truth = plant.ground_truth()
        assert truth.pose.x > 0.02
        assert abs(truth.pose.y) < 0.05
    finally:
        plant.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_opposite_wheel_targets_rotate():
    plant = MujocoDiffDrivePlant(plant_config(), {"obstacles": []})
    try:
        plant.reset(2, np.zeros(3))
        for index in range(20):
            plant.step(ControlCommand(np.asarray((0.0, 0.5)), index * 0.05), 0.05)
        truth = plant.ground_truth()
        assert abs(truth.pose.theta) > 0.05
        assert np.hypot(truth.pose.x, truth.pose.y) < 0.10
    finally:
        plant.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_torque_pi_research_profile_moves_without_teleporting():
    plant = MujocoDiffDrivePlant(plant_config("torque_pi"), {"obstacles": []})
    try:
        plant.reset(3, np.zeros(3))
        for index in range(30):
            plant.step(ControlCommand(np.asarray((0.2, 0.0)), index * 0.05), 0.05)
        truth = plant.ground_truth()
        assert truth.pose.x > 0.02
        assert plant.model.nu == 2
        assert np.linalg.norm(truth.actuator_effort) > 0.0
    finally:
        plant.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_lidar_measures_known_cylinder_without_truth_leakage():
    scene = {"obstacles": [{"type": "cylinder", "position": [1.0, 0.0], "radius": 0.2, "height": 0.5}]}
    plant = MujocoDiffDrivePlant(plant_config(), scene)
    try:
        truth = plant.reset(4, np.zeros(3))
        sensors = SimulatedSensorSuite(
            plant,
            {"lidar_beams": 181, "lidar_angle_min": -np.pi / 2, "lidar_angle_max": np.pi / 2,
             "lidar_range_min": 0.05, "lidar_range_max": 3.0},
            4,
        )
        observation = sensors.reset(truth, 4)
        center = observation.scan.ranges[90]
        assert center == pytest.approx(0.7, abs=0.08)
        assert not hasattr(observation, "ground_truth")
    finally:
        plant.close()
