from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from complex_scene_tracker import MotionBootstrapMultiObstacleTracker
from mobile_robot_mppi.obstacles.online_tracking import ScanCluster


@dataclass
class _Config:
    association_gate_m: float = 0.8


class _Predictor:
    state = None


class _Track:
    predictor = _Predictor()

    def _association_reference(self, timestamp):
        del timestamp
        return None


def _cluster(x, y, beams=12, distance=1.0):
    return ScanCluster(
        center=np.asarray((x, y), dtype=np.float64),
        support_beams=beams,
        minimum_range_m=distance,
        beam_indices=tuple(range(beams)),
    )


def _selector():
    tracker = object.__new__(MotionBootstrapMultiObstacleTracker)
    tracker.trackers = (_Track(), _Track(), _Track())
    tracker.config = _Config()
    tracker.maximum_tracks = 3
    tracker.required_motion_intervals = 4
    tracker.minimum_speed_mps = 0.08
    tracker.maximum_speed_mps = 1.8
    tracker.maximum_support_beams = 35
    tracker.vehicle_minimum_support_beams = 12
    tracker.vehicle_maximum_support_beams = 80
    tracker.vehicle_candidate_gate_m = 0.65
    tracker.vehicle_maximum_speed_mps = 2.5
    tracker.maximum_range_m = 2.5
    tracker.candidate_gate_m = 0.35
    tracker.minimum_total_displacement_m = 0.10
    tracker.minimum_direction_cosine = 0.50
    tracker.provisional_motion_intervals = 1
    tracker.provisional_minimum_displacement_m = 0.025
    tracker.provisional_minimum_closing_speed_mps = 0.10
    tracker.provisional_horizon_s = 3.0
    tracker.provisional_closest_approach_m = 0.75
    tracker._bootstrap_previous_clusters = ()
    tracker._bootstrap_previous_timestamp = None
    tracker._bootstrap_candidates = ()
    return tracker


def test_static_room_clusters_do_not_fill_empty_track_slots():
    tracker = _selector()
    walls = (
        _cluster(2.0, 0.0, beams=45),
        _cluster(1.5, 1.0, beams=55),
        _cluster(1.5, -1.0, beams=50),
    )
    for index in range(8):
        selected = tracker._motion_candidates(
            walls, index * 0.1
        )
        assert selected == ()
        tracker._bootstrap_previous_clusters = walls
        tracker._bootstrap_previous_timestamp = index * 0.1
        tracker._bootstrap_candidates = selected


def test_persistent_compact_motion_reaches_confirmation_threshold():
    tracker = _selector()
    streaks = []
    for index in range(4):
        clusters = (
            _cluster(2.0, 0.0, beams=45),
            _cluster(1.0, -0.30 + 0.04 * index, beams=12),
        )
        selected = tracker._motion_candidates(
            clusters, index * 0.1
        )
        streaks.append(
            max(
                (
                    candidate["streak"]
                    for candidate in selected
                ),
                default=0,
            )
        )
        tracker._bootstrap_previous_clusters = clusters
        tracker._bootstrap_previous_timestamp = index * 0.1
        tracker._bootstrap_candidates = selected
    assert streaks == [0, 1, 2, 3]


def test_one_frame_jump_is_not_persistent_motion():
    tracker = _selector()
    positions = (0.0, 0.04, 0.04, 0.04)
    streaks = []
    for index, position in enumerate(positions):
        clusters = (_cluster(1.0, position, beams=12),)
        selected = tracker._motion_candidates(
            clusters, index * 0.1
        )
        streaks.append(
            max(
                (
                    candidate["streak"]
                    for candidate in selected
                ),
                default=0,
            )
        )
        tracker._bootstrap_previous_clusters = clusters
        tracker._bootstrap_previous_timestamp = index * 0.1
        tracker._bootstrap_candidates = selected
    assert max(streaks) == 1


def test_overwide_moving_cluster_is_rejected():
    tracker = _selector()
    for index in range(5):
        clusters = (_cluster(1.0, 0.04 * index, beams=90),)
        selected = tracker._motion_candidates(
            clusters, index * 0.1
        )
        assert selected == ()
        tracker._bootstrap_previous_clusters = clusters
        tracker._bootstrap_previous_timestamp = index * 0.1
        tracker._bootstrap_candidates = selected


def test_wide_fast_vehicle_cluster_is_retained_as_motion():
    tracker = _selector()
    streaks = []
    for index in range(4):
        clusters = (_cluster(1.0, 0.25 * index, beams=40),)
        selected = tracker._motion_candidates(clusters, 0.28 * index)
        streaks.append(
            max((item["streak"] for item in selected), default=0)
        )
        if selected:
            assert selected[0]["vehicle_shape"]
        tracker._bootstrap_previous_clusters = clusters
        tracker._bootstrap_previous_timestamp = 0.28 * index
        tracker._bootstrap_candidates = selected
    assert streaks == [0, 1, 2, 3]


def test_direction_reversal_resets_motion_identity():
    tracker = _selector()
    positions = (0.0, 0.04, 0.08, 0.04, 0.00)
    streaks = []
    for index, position in enumerate(positions):
        clusters = (_cluster(1.0, position, beams=12),)
        selected = tracker._motion_candidates(
            clusters, index * 0.1
        )
        streaks.append(
            max(
                (
                    candidate["streak"]
                    for candidate in selected
                ),
                default=0,
            )
        )
        tracker._bootstrap_previous_clusters = clusters
        tracker._bootstrap_previous_timestamp = index * 0.1
        tracker._bootstrap_candidates = selected
    assert streaks == [0, 1, 2, 0, 1]


def _observation(x=0.0, y=0.0, theta=0.0, v=0.0):
    return SimpleNamespace(
        pose=SimpleNamespace(x=x, y=y, theta=theta),
        twist=SimpleNamespace(v=v),
    )


def test_two_frame_approach_is_exposed_as_provisional_collision_threat():
    tracker = _selector()
    candidate = {
        "cluster_index": 2,
        "center": np.asarray((1.2, 0.2)),
        "streak": 1,
        "speed_mps": 0.5,
        "velocity_mps": np.asarray((-0.5, 0.0)),
        "origin": np.asarray((1.25, 0.2)),
        "total_displacement_m": 0.05,
        "support_beams": 12,
        "vehicle_shape": True,
    }
    threat = tracker._provisional_threat(candidate, _observation())
    assert threat is not None
    assert threat["cluster_index"] == 2
    assert threat["closing_speed_mps"] > 0.4
    assert threat["closest_approach_time_s"] < 3.0
    assert threat["lateral_m"] > 0.0
    assert threat["dynamic_shape"] == "wide_vehicle"


def test_receding_or_noncollision_motion_is_not_provisional_threat():
    tracker = _selector()
    receding = {
        "cluster_index": 0,
        "center": np.asarray((1.5, 0.0)),
        "streak": 1,
        "speed_mps": 0.5,
        "velocity_mps": np.asarray((0.5, 0.0)),
        "origin": np.asarray((1.45, 0.0)),
        "total_displacement_m": 0.05,
        "support_beams": 12,
    }
    crossing_wide = dict(receding)
    crossing_wide.update(
        {
            "center": np.asarray((1.5, 1.5)),
            "velocity_mps": np.asarray((-0.5, 0.0)),
        }
    )
    assert tracker._provisional_threat(receding, _observation()) is None
    assert (
        tracker._provisional_threat(crossing_wide, _observation())
        is None
    )
