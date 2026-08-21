from types import SimpleNamespace
import time

import numpy as np
import pytest

import real_robot_operator
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D


class _FakeLive:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self._index = 0

    def snapshot(self):
        index = min(self._index, len(self._snapshots) - 1)
        self._index += 1
        return self._snapshots[index]


class _FakeGatewayLive:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self._index = 0

    def gateway_snapshot(self):
        index = min(self._index, len(self._snapshots) - 1)
        self._index += 1
        return self._snapshots[index]


def _observation_result(timestamp):
    return (
        SimpleNamespace(
            timestamp=float(timestamp),
            scan=SimpleNamespace(timestamp=float(timestamp)),
        ),
        {"scan": 0.0},
        0,
    )


def test_rear_gateway_freshness_checks_both_sonars():
    assert real_robot_operator._fresh_rear_gateway(
        {"rear_right_age_s": 0.10, "rear_left_age_s": 0.20}
    )
    assert not real_robot_operator._fresh_rear_gateway(
        {"rear_right_age_s": 0.10, "rear_left_age_s": 0.50}
    )


def test_gateway_status_callback_burst_can_recover_while_zero_is_held():
    now = time.monotonic()
    live = _FakeGatewayLive((
        ({"armed": True}, now - 1.0),
        ({"armed": True}, now),
    ))
    status, received = (
        real_robot_operator._wait_for_fresh_gateway_status(
            live, timeout_s=0.10
        )
    )
    assert status == {"armed": True}
    assert received == now


def test_gateway_status_recovery_times_out_when_updates_stay_stale():
    now = time.monotonic()
    live = _FakeGatewayLive((({"armed": True}, now - 2.0),))
    assert real_robot_operator._wait_for_fresh_gateway_status(
        live, timeout_s=0.03
    ) == (None, None)


def test_fresh_observation_skips_cached_scan(monkeypatch):
    timestamps = iter((10.0, 10.0, 10.1))
    monkeypatch.setattr(
        real_robot_operator,
        "_observation",
        lambda snapshot, origin: _observation_result(next(timestamps)),
    )

    result = real_robot_operator._fresh_observation(
        _FakeLive((object(), object(), object())),
        origin=(0.0, 0.0, 0.0),
        timeout_s=0.20,
        after_timestamp=10.0,
    )

    assert result[0].timestamp == 10.1


def test_fresh_observation_reports_scan_not_advanced(monkeypatch):
    monkeypatch.setattr(
        real_robot_operator,
        "_observation",
        lambda snapshot, origin: _observation_result(10.0),
    )

    with pytest.raises(RuntimeError, match="scan_not_advanced"):
        real_robot_operator._fresh_observation(
            _FakeLive((object(),)),
            origin=(0.0, 0.0, 0.0),
            timeout_s=0.03,
            after_timestamp=10.0,
        )


def _forecast(x, y, radius=0.25):
    means = np.tile(np.asarray((x, y), dtype=np.float64), (36, 1))
    return _forecast_positions(means, radius=radius)


def _forecast_positions(means, radius=0.25):
    means = np.asarray(means, dtype=np.float64)
    return SimpleNamespace(
        component_means=means[:, None, :],
        component_weights=np.ones((36, 1), dtype=np.float64),
        component_covariances=np.tile(
            np.eye(2, dtype=np.float64)[None, None, :, :] * 0.01,
            (36, 1, 1, 1),
        ),
        radius_m=radius,
        dt=0.1,
        source="test",
    )


def _planner_observation(*forecasts):
    return RobotObservation(
        timestamp=1.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": forecasts},
    )


def test_path_relevance_retains_on_path_forecast():
    observation = _planner_observation(_forecast(1.5, 0.2))
    filtered, diagnostics = real_robot_operator._path_relevant_forecasts(
        observation, 3.0, 0.0
    )
    assert diagnostics["retained_count"] == 1
    candidate = diagnostics["collision_candidates"][0]
    assert candidate["closest_approach_time_s"] > 0.1
    assert "collision_clearance_m" in candidate
    assert filtered is observation


def test_path_relevance_drops_remote_off_path_forecast_only_from_planner():
    forecast = _forecast(1.5, -2.0)
    observation = _planner_observation(forecast)
    filtered, diagnostics = real_robot_operator._path_relevant_forecasts(
        observation, 3.0, 0.0
    )
    assert diagnostics["dropped_count"] == 1
    assert filtered.auxiliary["probabilistic_obstacle_forecasts"] == ()
    assert observation.auxiliary["probabilistic_obstacle_forecasts"] == (
        forecast,
    )


def test_spatial_path_crossing_at_wrong_time_is_not_a_collision():
    positions = np.tile(np.asarray((2.5, 3.0)), (36, 1))
    positions[0] = (2.5, 0.0)
    forecast = _forecast_positions(positions)
    observation = _planner_observation(forecast)
    filtered, diagnostics = real_robot_operator._path_relevant_forecasts(
        observation, 3.0, 0.0, maximum_linear_mps=0.50
    )
    assert diagnostics["retained_count"] == 0
    assert diagnostics["collision_candidates"] == ()
    assert filtered.auxiliary["probabilistic_obstacle_forecasts"] == ()


def test_turn_limited_projection_respects_heading_and_step_length():
    observation = RobotObservation(
        timestamp=1.0,
        pose=Pose2D(0.0, 0.0, np.pi / 2.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={},
    )
    positions = real_robot_operator._turn_limited_robot_positions(
        observation,
        3.0,
        0.0,
        np.asarray((0.1, 0.2, 0.3)),
        maximum_linear_mps=0.50,
        maximum_angular_radps=0.50,
    )
    steps = np.linalg.norm(
        np.diff(np.vstack((np.zeros((1, 2)), positions)), axis=0),
        axis=1,
    )
    assert positions.shape == (3, 2)
    assert positions[0, 1] > 0.04
    assert abs(positions[0, 0]) < 0.01
    assert np.all(steps <= 0.050001)
