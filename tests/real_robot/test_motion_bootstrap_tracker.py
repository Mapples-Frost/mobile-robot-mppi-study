from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from mobile_robot_mppi.real_robot.motion_bootstrap_tracker import (
    MotionBootstrapMultiObstacleTracker,
)
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


class _OccupiedTrack:
    def __init__(self, reference, clusters=()):
        self.reference = np.asarray(reference, dtype=np.float64)
        self.predictor = SimpleNamespace(state=object())
        self.clusters = tuple(clusters)
        self.reset_count = 0

    def _association_reference(self, timestamp):
        del timestamp
        if self.predictor.state is None:
            return None
        return self.reference

    def reset(self):
        self.predictor.state = None
        self.reset_count += 1

    def scan_clusters(self, observation):
        del observation
        return self.clusters


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
    tracker.vehicle_minimum_extent_m = 0.25
    tracker.vehicle_maximum_extent_m = 1.10
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
    tracker.temporal_flow_threat_preemption_enabled = True
    tracker.temporal_flow_threat_maximum_ttc_s = 2.0
    tracker.temporal_flow_threat_minimum_support_beams = 3
    tracker.temporal_flow_threat_angle_tolerance_rad = np.deg2rad(25.0)
    tracker.temporal_flow_threat_range_tolerance_m = 0.75
    tracker.temporal_flow_threat_hold_cycles = 6
    tracker._temporal_flow_threat_retained_indices = set()
    tracker._temporal_flow_threat_track_indices = set()
    tracker._temporal_flow_threat_hold = [0, 0, 0]
    tracker._bootstrap_previous_clusters = ()
    tracker._bootstrap_previous_timestamp = None
    tracker._bootstrap_candidates = ()
    tracker._bootstrap_cluster_extents_m = ()
    return tracker


def test_vehicle_geometry_rejects_wall_and_accepts_robot_sized_cluster():
    tracker = _selector()
    cluster = _cluster(1.0, 0.0, beams=20)
    tracker._bootstrap_cluster_extents_m = (0.62,)
    assert tracker._vehicle_shape(cluster, 0) == (True, 0.62)
    tracker._bootstrap_cluster_extents_m = (1.80,)
    assert tracker._vehicle_shape(cluster, 0) == (False, 1.80)


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
        auxiliary={},
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


def test_temporal_flow_requires_motion_angle_range_and_ttc_corroboration():
    tracker = _selector()
    observation = _observation()
    observation.auxiliary["temporal_scan_flow"] = {
        "valid": True,
        "ttc_s": 1.4,
        "support_beams": 4,
        "center_angle_rad": 0.10,
        "clearance_m": 1.15,
    }
    candidate = {
        "cluster_index": 2,
        "center": np.asarray((1.20, 0.12)),
        "minimum_range_m": 1.14,
    }
    matches = tracker._temporal_flow_matches((candidate,), observation)
    assert len(matches) == 1
    assert matches[0]["cluster_index"] == 2

    observation.auxiliary["temporal_scan_flow"]["center_angle_rad"] = 1.1
    assert tracker._temporal_flow_matches((candidate,), observation) == ()
    observation.auxiliary["temporal_scan_flow"]["center_angle_rad"] = 0.10
    observation.auxiliary["temporal_scan_flow"]["ttc_s"] = 2.1
    assert tracker._temporal_flow_matches((candidate,), observation) == ()


def test_temporal_flow_threat_gets_first_empty_slot_over_large_background():
    tracker = _selector()
    tracker._temporal_flow_threat_retained_indices = {1}
    clusters = (
        _cluster(2.0, 0.0, beams=30, distance=2.0),
        _cluster(1.1, 0.1, beams=4, distance=1.0),
    )
    assignments = tracker._assign(clusters, timestamp=0.2)
    assert assignments[0][0] == 1
    assert tracker._temporal_flow_threat_hold[0] == 6
    assert tracker._temporal_flow_threat_track_indices == {0}


def test_temporal_flow_threat_preempts_farthest_background_when_slots_full(
    monkeypatch,
):
    tracker = _selector()
    raw_clusters = (
        _cluster(2.0, 1.0, beams=30, distance=2.1),
        _cluster(3.0, 0.0, beams=30, distance=3.0),
        _cluster(4.0, 1.0, beams=30, distance=4.1),
        _cluster(1.1, 0.1, beams=4, distance=1.0),
    )
    tracker.trackers = (
        _OccupiedTrack((2.0, 1.0), raw_clusters),
        _OccupiedTrack((3.0, 0.0)),
        _OccupiedTrack((4.0, 1.0)),
    )
    threat = {
        "cluster_index": 3,
        "center": np.asarray((1.1, 0.1), dtype=np.float64),
        "minimum_range_m": 1.0,
        "streak": 1,
        "total_displacement_m": 0.05,
    }
    monkeypatch.setattr(
        tracker,
        "_motion_candidates",
        lambda clusters, timestamp, excluded_indices=(): (threat,),
    )
    monkeypatch.setattr(
        tracker,
        "_provisional_threat",
        lambda candidate, observation: None,
    )
    observation = SimpleNamespace(
        timestamp=0.2,
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        twist=SimpleNamespace(v=0.0),
        scan=SimpleNamespace(
            obstacle_ranges=None,
            ranges=np.ones(40, dtype=np.float64),
            angle_min=-0.2,
            angle_increment=0.01,
        ),
        auxiliary={
            "temporal_scan_flow": {
                "valid": True,
                "ttc_s": 1.4,
                "support_beams": 4,
                "center_angle_rad": float(np.arctan2(0.1, 1.1)),
                "clearance_m": 1.0,
            }
        },
    )

    retained = tracker.scan_clusters(observation)

    assert tracker.trackers[2].reset_count == 1
    assert tracker._temporal_flow_preempted_track_indices == (2,)
    assert tracker._temporal_flow_preempted_raw_cluster_indices == (2,)
    assert tuple(cluster.center.tolist() for cluster in retained) == (
        [2.0, 1.0],
        [3.0, 0.0],
        [1.1, 0.1],
    )

    assignments = tracker._assign(retained, timestamp=0.2)

    assert assignments[2][0] == 2
    assert tracker._temporal_flow_threat_track_indices == {2}
    assert tracker._temporal_flow_threat_hold == [0, 0, 6]
