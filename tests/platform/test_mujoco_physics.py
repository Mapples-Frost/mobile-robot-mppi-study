import importlib.util

import numpy as np
import pytest

from mobile_robot_mppi.core.types import (
    ControlCommand,
    GroundTruth,
    Pose2D,
    Twist2D,
)
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant
from mobile_robot_mppi.simulation.sensors import SimulatedSensorSuite


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None


class FakePlant:
    wheel_radius = 0.08
    track_width = 0.32
    obstacles = ()


def truth_sample(time, x, body_v, wheel_speed):
    return GroundTruth(
        timestamp=float(time),
        pose=Pose2D(float(x), 0.0, 0.0),
        twist=Twist2D(float(body_v), 0.0),
        wheel_speeds=(float(wheel_speed), float(wheel_speed)),
    )


def test_sensor_state_source_is_explicit_and_backward_compatible():
    initial = truth_sample(0.0, 0.0, 0.0, 0.0)
    following = truth_sample(1.0, 0.10, 0.10, 2.5)  # wheel estimate is 0.20 m/s
    odometry = SimulatedSensorSuite(FakePlant(), {}, 0)
    odometry.reset(initial, 0)
    odom_observation = odometry.observe(following)
    assert odom_observation.pose.x == pytest.approx(0.20)
    assert odom_observation.twist.v == pytest.approx(0.20)
    assert odom_observation.auxiliary["pose_source"] == "wheel_odometry"

    localized = SimulatedSensorSuite(
        FakePlant(),
        {"pose_source": "ground_truth", "twist_source": "ground_truth"},
        0,
    )
    localized.reset(initial, 0)
    localized_observation = localized.observe(following)
    assert localized_observation.pose.x == pytest.approx(0.10)
    assert localized_observation.twist.v == pytest.approx(0.10)
    assert localized_observation.auxiliary["pose_source"] == "ground_truth"


@pytest.mark.parametrize(
    "config",
    ({"pose_source": "magic"}, {"twist_source": "magic"}),
)
def test_sensor_rejects_unknown_state_source(config):
    with pytest.raises(ValueError, match="source"):
        SimulatedSensorSuite(FakePlant(), config, 0)


def plant_config(profile="ideal_velocity"):
    return {
        "robot": {"wheel_radius": 0.08, "track_width": 0.32},
        "actuator": {"profile": profile, "torque_limit": 3.0, "velocity_gain": 8.0, "kp": 0.8, "ki": 2.0},
        "contact": {},
        "physics": {"timestep": 0.002, "integrator": "implicitfast"},
    }


def delayed_plant_config(delay):
    config = plant_config()
    config["actuator"]["command_delay"] = float(delay)
    return config


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


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_command_delay_is_resolved_at_physics_substep_rate():
    plant = MujocoDiffDrivePlant(delayed_plant_config(0.04), {"obstacles": []})
    try:
        plant.reset(5, np.zeros(3))
        recorded_targets = []
        original_apply = plant._apply_actuator

        def record_and_apply(targets, physics_dt):
            recorded_targets.append(np.asarray(targets).copy())
            original_apply(targets, physics_dt)

        plant._apply_actuator = record_and_apply
        command = ControlCommand(np.asarray((0.2, 0.0)), 0.0)
        step = plant.step(command, 0.10)
        zero_substeps = sum(np.allclose(targets, 0.0) for targets in recorded_targets)
        assert len(recorded_targets) == 50
        assert zero_substeps == 20
        np.testing.assert_allclose(step.executed_control.values, command.values)
        np.testing.assert_allclose(
            step.metadata["average_applied_control"], 0.6 * command.values
        )
    finally:
        plant.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo optional dependency is unavailable")
def test_command_delay_longer_than_control_period_keeps_zero_command_active():
    plant = MujocoDiffDrivePlant(delayed_plant_config(0.20), {"obstacles": []})
    try:
        plant.reset(6, np.zeros(3))
        step = plant.step(ControlCommand(np.asarray((0.2, 0.0)), 0.0), 0.05)
        np.testing.assert_allclose(step.executed_control.values, np.zeros(2))
    finally:
        plant.close()
