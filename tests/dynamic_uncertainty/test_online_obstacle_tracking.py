import inspect
from pathlib import Path

import numpy as np
import pytest

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.types import (
    LaserScan,
    Pose2D,
    RobotObservation,
    Twist2D,
)
from mobile_robot_mppi.obstacles.online_tracking import (
    SingleObstacleChangeAwareTracker,
)
from mobile_robot_mppi.obstacles.multi_online_tracking import (
    MultiObstacleChangeAwareTracker,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs/research/"
    "mujoco_v3_probabilistic_crossing_smoke_amendment5.yaml"
)


def _tracker():
    config = load_yaml(CONFIG_PATH)
    values = config["perception"]["dynamic_obstacle_tracker"]
    return SingleObstacleChangeAwareTracker.from_mapping(ROOT, values)


def _ordinary_tracker():
    config = load_yaml(CONFIG_PATH)
    values = dict(config["perception"]["dynamic_obstacle_tracker"])
    values["predictor_mode"] = "ordinary"
    return SingleObstacleChangeAwareTracker.from_mapping(ROOT, values)


def _circle_scan(
    centers,
    pose=(0.0, 0.0, 0.0),
    radius=0.25,
    timestamp=0.0,
    beams=361,
    range_max=6.0,
    sensor_offset=0.10,
):
    x, y, theta = (float(value) for value in pose)
    origin = np.asarray(
        (
            x + sensor_offset * np.cos(theta),
            y + sensor_offset * np.sin(theta),
        )
    )
    angle_min = -np.pi
    increment = 2.0 * np.pi / (beams - 1)
    ranges = np.full(beams, range_max, dtype=np.float64)
    for index in range(beams):
        angle = theta + angle_min + index * increment
        direction = np.asarray((np.cos(angle), np.sin(angle)))
        best = range_max
        for center in centers:
            offset = origin - np.asarray(center, dtype=np.float64)
            projection = float(np.dot(direction, offset))
            discriminant = (
                projection ** 2
                - (float(np.dot(offset, offset)) - radius ** 2)
            )
            if discriminant < 0.0:
                continue
            root = -projection - np.sqrt(discriminant)
            if 0.05 <= root < best:
                best = float(root)
        ranges[index] = best
    scan = LaserScan(
        ranges=ranges,
        obstacle_ranges=np.where(
            ranges < range_max, ranges, np.inf
        ),
        angle_min=angle_min,
        angle_increment=increment,
        range_min=0.05,
        range_max=range_max,
        timestamp=timestamp,
    )
    return RobotObservation(
        timestamp=timestamp,
        pose=Pose2D(x, y, theta),
        twist=Twist2D(0.0, 0.0),
        scan=scan,
    )


def test_circle_cluster_recovers_center_without_truth_input():
    tracker = _tracker()
    truth_center = np.asarray((1.50, 0.0))
    observation = _circle_scan((truth_center,))
    clusters = tracker.scan_clusters(observation)
    assert len(clusters) == 1
    assert clusters[0].support_beams >= 5
    np.testing.assert_allclose(
        clusters[0].center, truth_center, atol=0.015, rtol=0.0
    )


def test_tracker_produces_planner_aligned_change_aware_forecast():
    tracker = _tracker()
    first = tracker.update(_circle_scan(((1.50, -0.05),), timestamp=0.0))
    second = tracker.update(_circle_scan(((1.50, 0.02),), timestamp=0.1))
    assert first.forecast is not None
    assert second.forecast is not None
    assert second.forecast.timestamp == 0.1
    assert second.forecast.dt == 0.1
    assert second.forecast.horizon == 36
    assert second.forecast.mode_count == 4
    assert second.forecast.source == "online_change_aware_imm"
    assert second.diagnostics["associated"]
    assert second.diagnostics["forecast_valid"]
    assert np.allclose(
        second.forecast.component_weights.sum(axis=1), 1.0
    )


def test_single_beam_candidate_requires_an_initialized_prediction_gate():
    tracker = _tracker()
    full = _circle_scan(((1.50, 0.0),), timestamp=0.0)
    ranges = np.full_like(full.scan.ranges, full.scan.range_max)
    hit = int(np.argmin(full.scan.ranges))
    ranges[hit] = full.scan.ranges[hit]
    one_beam_scan = LaserScan(
        ranges=ranges,
        obstacle_ranges=np.where(
            ranges < full.scan.range_max, ranges, np.inf
        ),
        angle_min=full.scan.angle_min,
        angle_increment=full.scan.angle_increment,
        range_min=full.scan.range_min,
        range_max=full.scan.range_max,
        timestamp=0.0,
    )
    one_beam = RobotObservation(
        timestamp=0.0,
        pose=full.pose,
        twist=full.twist,
        scan=one_beam_scan,
    )
    assert tracker.scan_clusters(one_beam) == ()

    tracker.update(full)
    tracked_one_beam = RobotObservation(
        timestamp=0.1,
        pose=full.pose,
        twist=full.twist,
        scan=LaserScan(
            ranges=one_beam_scan.ranges,
            obstacle_ranges=one_beam_scan.obstacle_ranges,
            angle_min=one_beam_scan.angle_min,
            angle_increment=one_beam_scan.angle_increment,
            range_min=one_beam_scan.range_min,
            range_max=one_beam_scan.range_max,
            timestamp=0.1,
        ),
    )
    update = tracker.update(tracked_one_beam)
    assert update.diagnostics["associated"]
    assert update.diagnostics["selected_support_beams"] == 1
    assert update.forecast is not None


def test_tracker_supports_frozen_ordinary_imm_ablation():
    tracker = _ordinary_tracker()
    assert tracker.predictor.name == "ordinary_imm"
    first = tracker.update(_circle_scan([(1.0, 0.0)], timestamp=0.0))
    second = tracker.update(_circle_scan([(1.02, 0.0)], timestamp=0.1))
    assert first.diagnostics["predictor_source"] == "ordinary_imm"
    assert second.forecast is not None
    assert second.forecast.source == "online_ordinary_imm"
    assert second.diagnostics["change_triggered"] is False


def test_prediction_gating_rejects_distractor_and_stale_track_fails_closed():
    tracker = _tracker()
    tracker.update(_circle_scan(((1.50, 0.0),), timestamp=0.0))
    associated = tracker.update(
        _circle_scan(
            ((1.50, 0.05), (-1.2, 0.0)),
            timestamp=0.1,
        )
    )
    assert associated.measurement is not None
    assert associated.measurement[0] > 1.0

    missing = tracker.update(_circle_scan((), timestamp=0.2))
    assert missing.measurement is None
    assert missing.forecast is not None

    stale = tracker.update(_circle_scan((), timestamp=1.7))
    assert stale.measurement is None
    assert stale.forecast is None
    assert stale.diagnostics["unobserved_duration_s"] > 1.5


def test_tracker_rejects_nonincreasing_time_and_reset_clears_history():
    tracker = _tracker()
    observation = _circle_scan(((1.5, 0.0),), timestamp=0.0)
    tracker.update(observation)
    with pytest.raises(ValueError, match="increase strictly"):
        tracker.update(observation)
    tracker.reset()
    assert tracker.predictor.state is None
    assert tracker.update_count == 0
    assert tracker.forecast_count == 0


def test_tracker_public_api_has_no_simulator_truth_or_future_arguments():
    forbidden = {
        "truth",
        "ground_truth",
        "trajectory",
        "future_states",
        "true_modes",
        "change_flags",
        "seed",
    }
    update_parameters = set(
        inspect.signature(
            SingleObstacleChangeAwareTracker.update
        ).parameters
    )
    cluster_parameters = set(
        inspect.signature(
            SingleObstacleChangeAwareTracker.scan_clusters
        ).parameters
    )
    assert forbidden.isdisjoint(update_parameters)
    assert forbidden.isdisjoint(cluster_parameters)


def test_multi_tracker_assigns_distinct_clusters_and_emits_three_forecasts():
    config = load_yaml(CONFIG_PATH)
    values = dict(config["perception"]["dynamic_obstacle_tracker"])
    values["maximum_tracks"] = 3
    tracker = MultiObstacleChangeAwareTracker.from_mapping(ROOT, values)
    first_centers = ((1.5, -0.8), (1.8, 0.0), (1.5, 0.8))
    second_centers = ((1.5, -0.74), (1.8, 0.04), (1.5, 0.86))

    first = tracker.update(_circle_scan(first_centers, timestamp=0.0))
    second = tracker.update(_circle_scan(second_centers, timestamp=0.1))

    assert first.diagnostics["track_count"] == 3
    assert second.diagnostics["associated_track_count"] == 3
    assert second.diagnostics["valid_forecast_count"] == 3
    assert len(second.forecast) == 3
    nearest_track = second.diagnostics["nearest_track_index"]
    nearest_forecast = second.diagnostics["nearest_forecast_index"]
    assert nearest_track in (0, 1, 2)
    assert nearest_forecast in (0, 1, 2)
    assert second.diagnostics["forecast_track_indices"][
        nearest_forecast
    ] == nearest_track
    assigned = {
        row["assigned_cluster_index"]
        for row in second.diagnostics["tracks"]
    }
    assert assigned == {0, 1, 2}


def test_multi_tracker_does_not_duplicate_one_cluster_across_free_tracks():
    config = load_yaml(CONFIG_PATH)
    values = dict(config["perception"]["dynamic_obstacle_tracker"])
    values["maximum_tracks"] = 3
    tracker = MultiObstacleChangeAwareTracker.from_mapping(ROOT, values)

    first = tracker.update(
        _circle_scan(((1.5, 0.0),), timestamp=0.0)
    )
    second = tracker.update(
        _circle_scan(((1.5, 0.06),), timestamp=0.1)
    )

    assert first.diagnostics["cluster_count"] == 1
    assert first.diagnostics["track_count"] == 1
    assert first.diagnostics["associated_track_count"] == 1
    assert first.diagnostics["valid_forecast_count"] == 1
    assert len(first.forecast) == 1
    assert second.diagnostics["track_count"] == 1
    assert second.diagnostics["associated_track_count"] == 1
    assert second.diagnostics["valid_forecast_count"] == 1
    assert len(second.forecast) == 1
    assert sum(
        row["assigned_cluster_index"] is not None
        for row in second.diagnostics["tracks"]
    ) == 1


def test_multi_tracker_publishes_only_motion_confirmed_clusters():
    config = load_yaml(CONFIG_PATH)
    values = dict(config["perception"]["dynamic_obstacle_tracker"])
    values.update({
        "maximum_tracks": 4,
        "association_gate_m": 0.40,
        "motion_confirmation_enabled": True,
        "motion_confirmation_required_observations": 4,
        "motion_confirmation_minimum_speed_mps": 0.45,
    })
    tracker = MultiObstacleChangeAwareTracker.from_mapping(ROOT, values)
    updates = []
    for step in range(4):
        updates.append(tracker.update(_circle_scan(
            (
                (1.45, -0.90 + 0.08 * step),
                (1.70, -0.20),
                (1.65, 0.45),
                (1.40, 0.95),
            ),
            timestamp=0.1 * step,
        )))

    assert all(update.forecast is None for update in updates[:3])
    final = updates[-1]
    assert final.forecast is not None
    assert len(final.forecast) == 1
    confirmed = [
        row for row in final.diagnostics["tracks"]
        if row["motion_confirmed"]
    ]
    assert len(confirmed) == 1
    assert confirmed[0]["measurement_speed_mps"] >= 0.45


def test_multi_tracker_public_api_is_causal():
    forbidden = {
        "truth",
        "ground_truth",
        "trajectory",
        "future_states",
        "true_modes",
        "change_flags",
        "seed",
    }
    assert forbidden.isdisjoint(
        inspect.signature(
            MultiObstacleChangeAwareTracker.update
        ).parameters
    )
