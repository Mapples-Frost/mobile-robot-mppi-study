from pathlib import Path

import numpy as np

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.runtime.factories import make_components


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs/research/"
    "mujoco_v3_probabilistic_crossing_smoke_amendment5.yaml"
)


def _components():
    return make_components(load_yaml(CONFIG_PATH), ROOT)


def _crossing_count(
    trajectory,
    route_start=(-4.6, -1.8),
    route_goal=(4.2, -1.8),
    duration_s=40.0,
):
    mask = trajectory.times <= float(duration_s) + 1.0e-12
    x = trajectory.states[mask, 0]
    y = trajectory.states[mask, 1]
    start = np.asarray(route_start, dtype=np.float64)
    goal = np.asarray(route_goal, dtype=np.float64)
    route = goal - start
    offset = route[0] * (y - start[1]) - route[1] * (x - start[0])
    crossing = offset[:-1] * offset[1:] < 0.0
    within = (
        np.maximum(x[:-1], x[1:]) >= float(min(start[0], goal[0]))
    ) & (
        np.minimum(x[:-1], x[1:]) <= float(max(start[0], goal[0]))
    )
    return int(np.sum(crossing & within))


def test_v3_mujoco_replay_is_seeded_recurrent_and_crosses_route():
    components = _components()
    plant = components["plant"]
    initial = np.asarray((-4.6, -1.8, 0.0, 0.0, 0.0))
    try:
        plant.reset(730100001, initial)
        item = plant._episode_dynamic_obstacles[0]
        first_trajectory = item["trajectory"].states.copy()
        first_state = plant.dynamic_obstacle_states()[0]
        assert first_state["motion_type"] == "recurrent_semimarkov_v3"
        assert first_state["process"] == "hybrid_patrol"
        assert first_state["trajectory_seed"] == 730100001
        assert _crossing_count(item["trajectory"]) >= 3

        command = ControlCommand(np.zeros(2), 0.0, "test")
        for _ in range(10):
            plant.step(command, 0.1)
        moved = plant.dynamic_obstacle_states()[0]
        assert np.hypot(
            moved["x"] - first_state["x"],
            moved["y"] - first_state["y"],
        ) > 0.05

        plant.reset(730100001, initial)
        np.testing.assert_array_equal(
            plant._episode_dynamic_obstacles[0]["trajectory"].states,
            first_trajectory,
        )
        plant.reset(730100003, initial)
        assert not np.array_equal(
            plant._episode_dynamic_obstacles[0]["trajectory"].states,
            first_trajectory,
        )
    finally:
        plant.close()


def test_real_mujoco_scan_feeds_tracker_forecast_and_risk_mppi():
    components = _components()
    plant = components["plant"]
    sensors = components["sensors"]
    perception = components["perception"]
    controller = components["controller"]
    reference = components["reference"]
    initial = np.asarray((-4.6, -1.8, 0.0, 0.0, 0.0))
    try:
        truth = plant.reset(730100001, initial)
        observation = sensors.reset(truth, 730100001)
        perception.reset()
        perceived = perception.process(observation)
        key = "probabilistic_obstacle_forecasts"
        assert key in perceived.observation.auxiliary
        assert len(perceived.observation.auxiliary[key]) == 1
        tracker = perceived.observation.auxiliary[
            "dynamic_obstacle_tracker"
        ]
        assert tracker["associated"]
        assert tracker["forecast_valid"]

        obstacle = truth.metadata["dynamic_obstacles"][0]
        error = np.hypot(
            tracker["measurement_x"] - obstacle["x"],
            tracker["measurement_y"] - obstacle["y"],
        )
        assert error <= 0.12

        controller.reset(seed=730100001)
        plan = controller.plan(perceived.observation, reference)
        assert plan.diagnostics["probabilistic_obstacle_risk_enabled"]
        assert plan.diagnostics["probabilistic_obstacle_forecast_count"] == 1
        assert plan.diagnostics["dynamic_obstacle_tracker_enabled"]
        assert plan.diagnostics[
            "dynamic_obstacle_tracker_forecast_valid"
        ]
        assert np.isfinite(
            plan.diagnostics[
                "probabilistic_obstacle_probability_mass"
            ]
        )
    finally:
        plant.close()
