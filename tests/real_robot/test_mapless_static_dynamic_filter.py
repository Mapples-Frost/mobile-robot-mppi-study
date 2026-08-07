from types import SimpleNamespace

import numpy as np

from mobile_robot_mppi.obstacles.online_tracking import OnlineTrackingUpdate

from mobile_robot_mppi.real_robot.mapless_static_dynamic_filter import (
    MaplessStaticDynamicFilter,
)
from mobile_robot_mppi.real_robot.human_point_cloud import (
    HumanPointCloudEvidenceExtractor,
)


class _RetirableTrack:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _Tracker:
    def __init__(self):
        self.config = SimpleNamespace()
        self.maximum_tracks = 3
        self.position = (0.0, 0.0)
        self.association_distance = 0.02
        self.selected_support_beams = 10
        self.measurement_speed_mps = 0.70
        self.vehicle_extent_m = None
        self.vehicle_geometry_confirmed = None
        self.temporal_flow_threat_hold_cycles = 0
        self.associated = True
        self.forecast_valid = True
        self.update_count = 0
        self.trackers = tuple(_RetirableTrack() for _ in range(3))

    def reset(self):
        self.update_count = 0

    def update(self, observation):
        self.update_count += 1
        track = {
            "update_count": self.update_count,
            "associated": self.associated,
            "association_distance_m": self.association_distance,
            "measurement_x": self.position[0],
            "measurement_y": self.position[1],
            "change_triggered": False,
            "recovery_active": False,
            "dropout_guard_triggered": False,
            "forecast_valid": self.forecast_valid,
            "measurement_speed_mps": self.measurement_speed_mps,
            "selected_support_beams": self.selected_support_beams,
            "vehicle_extent_m": self.vehicle_extent_m,
            "vehicle_geometry_confirmed": self.vehicle_geometry_confirmed,
            "temporal_flow_threat_hold_cycles": (
                self.temporal_flow_threat_hold_cycles
            ),
        }
        return OnlineTrackingUpdate(
            measurement=None,
            forecast=(("forecast",) if self.forecast_valid else None),
            diagnostics={
                "tracks": (track, {}, {}),
                "forecast_track_indices": (
                    (0,) if self.forecast_valid else ()
                ),
                "nearest_track_index": 0,
            },
        )


def _filter():
    tracker = _Tracker()
    value = object.__new__(MaplessStaticDynamicFilter)
    # Bypass the production type check only for this deterministic fake.
    value.tracker = tracker
    value.config = tracker.config
    value.maximum_tracks = 3
    value.history_size = 8
    value.minimum_samples = 3
    value.minimum_duration_s = 0.50
    value.static_speed_mps = 0.10
    value.dynamic_speed_mps = 0.20
    value.dynamic_displacement_m = 0.18
    value.minimum_direction_coherence = 0.80
    value.maximum_fit_residual_m = 0.06
    value.maximum_step_m = 0.20
    value.maximum_association_distance_m = 0.30
    value.maximum_history_gap_s = 0.70
    value.fast_minimum_duration_s = 0.27
    value.fast_minimum_speed_mps = 0.55
    value.fast_minimum_displacement_m = 0.18
    value.fast_minimum_direction_coherence = 0.90
    value.fast_maximum_fit_residual_m = 0.04
    value.fast_maximum_step_m = 0.22
    value.fast_minimum_tracker_speed_mps = 0.30
    value.persistent_minimum_tracker_speed_mps = 0.15
    value.minimum_dynamic_support_beams = 3
    value.vehicle_minimum_support_beams = 12
    value.vehicle_minimum_samples = 3
    value.vehicle_minimum_duration_s = 0.27
    value.vehicle_minimum_speed_mps = 0.45
    value.vehicle_minimum_displacement_m = 0.18
    value.vehicle_minimum_tracker_speed_mps = 0.25
    value.vehicle_minimum_direction_coherence = 0.92
    value.vehicle_maximum_fit_residual_m = 0.03
    value.vehicle_maximum_step_m = 0.60
    value.vehicle_maximum_extent_relative_span = 0.35
    value.allow_compact_dynamic = True
    value.vehicle_shape_hold_cycles = 4
    value.vehicle_provisional_minimum_streak = 2
    value.vehicle_provisional_minimum_displacement_m = 0.10
    value.vehicle_provisional_minimum_closing_speed_mps = 0.15
    value.vehicle_provisional_maximum_ttc_s = 3.00
    value.vehicle_provisional_maximum_closest_approach_m = 0.90
    value.dynamic_memory_ttl_s = 0.90
    value.dynamic_handoff_gate_m = 0.35
    value.dynamic_hold_cycles = 2
    value.static_track_eviction_cycles = 0
    value.allow_temporal_flow_provisional = False
    value.allow_recent_vehicle_fragment_dynamic = False
    value.recent_vehicle_minimum_direction_coherence = 0.45
    value.dynamic_classification_temporal_corroboration_enabled = False
    value.dynamic_classification_temporal_corroboration_maximum_ttc_s = 6.0
    value.dynamic_classification_temporal_corroboration_hold_cycles = 12
    value.dynamic_classification_temporal_corroboration_minimum_support_beams = 3
    value.dynamic_classification_collision_course_bypass_enabled = False
    value.allow_collision_course_provisional = False
    value.person_provisional_enabled = False
    value.person_provisional_minimum_streak = 2
    value.vehicle_shape_hold_cycles = 4
    value.person_provisional_minimum_samples = 4
    value.person_provisional_minimum_duration_s = 0.35
    value.person_provisional_minimum_speed_mps = 0.18
    value.person_provisional_maximum_speed_mps = 1.40
    value.person_provisional_minimum_displacement_m = 0.12
    value.person_provisional_minimum_direction_coherence = 0.55
    value.person_provisional_maximum_fit_residual_m = 0.08
    value.person_provisional_maximum_step_m = 0.28
    value.person_provisional_minimum_support_beams = 3
    value.person_provisional_maximum_extent_m = 1.20
    value.reset_filter_state()
    return value, tracker


def _observation(timestamp, robot_x=0.0, temporal_flow=None, point_cloud=None):
    auxiliary = (
        {}
        if temporal_flow is None
        else {"temporal_scan_flow": dict(temporal_flow)}
    )
    if point_cloud is not None:
        auxiliary["human_point_cloud_base"] = point_cloud
    return SimpleNamespace(
        timestamp=float(timestamp),
        pose=SimpleNamespace(x=float(robot_x), y=0.0, theta=0.0),
        twist=SimpleNamespace(v=0.0, omega=0.0),
        auxiliary=auxiliary,
    )


def test_world_fixed_measurement_is_static_while_robot_moves():
    value, tracker = _filter()
    result = None
    for index in range(6):
        tracker.position = (2.0, 0.005 * (index % 2))
        result = value.update(_observation(0.15 * index, robot_x=0.1 * index))
    assert result.forecast is None
    assert result.diagnostics["mapless_static_track_indices"] == (0,)
    assert result.diagnostics["tracks"][0]["mapless_classification"] == "static"


def test_static_background_track_is_evicted_only_when_explicitly_enabled():
    value, tracker = _filter()
    value.static_track_eviction_cycles = 3
    evicted = []
    for index in range(8):
        tracker.position = (2.0, 0.005 * (index % 2))
        result = value.update(_observation(0.15 * index))
        evicted.extend(
            result.diagnostics["mapless_static_evicted_track_indices"]
        )

    assert evicted == [0]
    assert tracker.trackers[0].reset_count == 1


def test_default_static_background_track_is_never_evicted():
    value, tracker = _filter()
    for index in range(8):
        tracker.position = (2.0, 0.005 * (index % 2))
        value.update(_observation(0.15 * index))

    assert tracker.trackers[0].reset_count == 0


def test_smooth_world_motion_becomes_dynamic_without_a_map():
    value, tracker = _filter()
    result = None
    for index in range(6):
        tracker.position = (1.5, 0.10 * index)
        result = value.update(_observation(0.15 * index))
    assert result.forecast == ("forecast",)
    assert result.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert result.diagnostics["valid_forecast_count"] == 1


def test_confirmed_dynamic_forecast_survives_a_brief_low_speed_fragment():
    value, tracker = _filter()
    value.dynamic_hold_cycles = 12
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.19)
    confirmed = value.update(_observation(0.28))
    assert confirmed.forecast == ("forecast",)

    # The low-level CA-IMM forecast remains available in the real profile
    # even when an instantaneous velocity estimate drops.  The mapless layer
    # must preserve the already-vetted dynamic identity rather than creating
    # a one-frame prediction hole and handing control back to another mode.
    tracker.measurement_speed_mps = 0.0
    tracker.position = (1.5, 0.19)
    fragmented = value.update(_observation(0.39))
    assert fragmented.forecast == ("forecast",)
    assert fragmented.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert fragmented.diagnostics["tracks"][0][
        "mapless_dynamic_hold_cycles"
    ] > 0


def test_confirmed_dynamic_forecast_survives_brief_cluster_dropout():
    value, tracker = _filter()
    value.dynamic_hold_cycles = 12
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.19)
    confirmed = value.update(_observation(0.28))
    assert confirmed.forecast == ("forecast",)

    # The physical lidar often loses both leg clusters for one control cycle.
    # CA-IMM still marks its prediction valid, so the mapless label must not
    # create a 1 -> 0 -> 1 forecast hole merely because association is absent.
    tracker.associated = False
    dropout = value.update(_observation(0.39))
    assert dropout.forecast == ("forecast",)
    assert dropout.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert dropout.diagnostics["tracks"][0][
        "mapless_dynamic_hold_cycles"
    ] > 0

    tracker.forecast_valid = False
    stale = value.update(_observation(0.50))
    assert stale.forecast is None


def test_two_frame_fast_motion_is_released_for_early_avoidance():
    value, tracker = _filter()
    tracker.position = (1.5, 0.0)
    first = value.update(_observation(0.0))
    tracker.position = (1.5, 0.19)
    second = value.update(_observation(0.28))
    assert first.forecast is None
    assert second.forecast == ("forecast",)
    assert second.diagnostics["mapless_dynamic_track_indices"] == (0,)


def test_vehicle_only_mode_rejects_compact_motion_fragments():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    tracker.selected_support_beams = 10
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.19)
    result = value.update(_observation(0.28))
    assert result.forecast is None
    assert result.diagnostics["mapless_dynamic_track_indices"] == ()


def test_recent_wide_shape_survives_fragmentation_and_one_center_jump():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    tracker.selected_support_beams = 15
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))

    tracker.selected_support_beams = 3
    tracker.association_distance = 0.50
    tracker.position = (1.5, 0.19)
    value.update(_observation(0.28))

    tracker.association_distance = 0.02
    tracker.position = (1.5, 0.39)
    value.update(_observation(0.56))
    tracker.position = (1.5, 0.59)
    result = value.update(_observation(0.84))
    track = result.diagnostics["tracks"][0]
    assert result.forecast == ("forecast",)
    assert track["mapless_dynamic_mode"] == "wide_vehicle"
    assert track["mapless_vehicle_shape_recent"]


def test_recent_human_shape_releases_smooth_fragment_before_close_ttc():
    value, tracker = _filter()
    value.allow_recent_vehicle_fragment_dynamic = True
    value.minimum_duration_s = 0.40
    value.dynamic_displacement_m = 0.14
    value.minimum_direction_coherence = 0.60
    value.maximum_fit_residual_m = 0.09
    value.maximum_step_m = 0.35
    tracker.measurement_speed_mps = 0.34
    samples = (
        (0.00, (3.41, 0.10), 13),
        (0.11, (3.40, 0.06), 1),
        (0.33, (3.40, 0.16), 13),
        (0.44, (3.44, 0.17), 1),
        (0.55, (3.43, 0.21), 1),
        (0.66, (3.34, 0.25), 23),
    )
    result = None
    for timestamp, position, support in samples:
        tracker.position = position
        tracker.selected_support_beams = support
        result = value.update(_observation(timestamp))

    assert result.forecast == ("forecast",)
    track = result.diagnostics["tracks"][0]
    assert track["mapless_dynamic_mode"] == "wide_vehicle"
    assert track["mapless_vehicle_shape_recent"]
    assert track["mapless_motion_evidence"][
        "direction_coherence"
    ] < value.minimum_direction_coherence


def test_wide_fast_vehicle_motion_has_a_dedicated_dynamic_mode():
    value, tracker = _filter()
    tracker.selected_support_beams = 38
    tracker.measurement_speed_mps = 0.52
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.25)
    value.update(_observation(0.28))
    tracker.position = (1.5, 0.50)
    result = value.update(_observation(0.56))
    assert result.forecast == ("forecast",)
    assert result.diagnostics["tracks"][0][
        "mapless_dynamic_mode"
    ] == "wide_vehicle"


def test_vehicle_extent_jump_is_not_misread_as_translation():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    tracker.selected_support_beams = 24
    tracker.vehicle_geometry_confirmed = True
    tracker.vehicle_extent_m = 0.30
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))

    tracker.vehicle_extent_m = 0.55
    tracker.position = (1.5, 0.20)
    result = value.update(_observation(0.28))
    evidence = result.diagnostics["tracks"][0][
        "mapless_motion_evidence"
    ]
    assert result.forecast is None
    assert not evidence["vehicle_extent_stable"]
    assert evidence["vehicle_extent_relative_span"] > 0.35


def test_stable_vehicle_extent_preserves_early_release():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    tracker.selected_support_beams = 24
    tracker.vehicle_geometry_confirmed = True
    tracker.vehicle_extent_m = 0.54
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))

    tracker.vehicle_extent_m = 0.56
    tracker.position = (1.5, 0.20)
    value.update(_observation(0.28))
    tracker.vehicle_extent_m = 0.55
    tracker.position = (1.5, 0.40)
    result = value.update(_observation(0.56))
    assert result.forecast == ("forecast",)
    assert result.diagnostics["tracks"][0][
        "mapless_motion_evidence"
    ]["vehicle_extent_stable"]


def test_two_frame_small_jitter_remains_unknown():
    value, tracker = _filter()
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.10)
    result = value.update(_observation(0.28))
    assert result.forecast is None
    assert result.diagnostics["mapless_dynamic_track_indices"] == ()


def test_temporal_corroboration_rejects_static_cluster_hopping_but_holds_motion():
    value, tracker = _filter()
    value.dynamic_classification_temporal_corroboration_enabled = True
    positions = ((0.0, 0.0), (0.28, 0.19), (0.56, 0.38))
    result = None
    for timestamp, lateral in positions:
        tracker.position = (1.5, lateral)
        result = value.update(_observation(timestamp))
    assert result.forecast is None
    assert result.diagnostics["mapless_dynamic_track_indices"] == ()

    value, tracker = _filter()
    value.dynamic_classification_temporal_corroboration_enabled = True
    tracker.temporal_flow_threat_hold_cycles = 6
    for index, (timestamp, lateral) in enumerate(positions):
        tracker.position = (1.5, lateral)
        flow = (
            {
                "valid": True,
                "ttc_s": 4.0,
                "support_beams": 4,
            }
            if index == 0 else None
        )
        result = value.update(_observation(timestamp, temporal_flow=flow))
    assert result.forecast == ("forecast",)
    assert result.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert result.diagnostics[
        "mapless_dynamic_classification_temporal_corroboration_remaining"
    ] > 0


def test_temporal_corroboration_is_scoped_to_the_matched_track():
    value, tracker = _filter()
    value.dynamic_classification_temporal_corroboration_enabled = True
    call_count = 0

    def update(_observation_value):
        nonlocal call_count
        call_count += 1
        lateral = 0.19 * (call_count - 1)

        def track(index, y_value, flow_hold):
            return {
                "track_index": index,
                "update_count": call_count,
                "associated": True,
                "association_distance_m": 0.02,
                "measurement_x": 1.5 + 0.4 * index,
                "measurement_y": y_value,
                "change_triggered": False,
                "recovery_active": False,
                "dropout_guard_triggered": False,
                "forecast_valid": True,
                "measurement_speed_mps": 0.70,
                "selected_support_beams": 10,
                "vehicle_extent_m": None,
                "vehicle_geometry_confirmed": None,
                "temporal_flow_threat_matched": flow_hold > 0,
                "temporal_flow_threat_hold_cycles": flow_hold,
            }

        return OnlineTrackingUpdate(
            measurement=None,
            forecast=("matched", "background"),
            diagnostics={
                "tracks": (
                    track(0, lateral, 6),
                    track(1, -lateral, 0),
                    {},
                ),
                "forecast_track_indices": (0, 1),
                "nearest_track_index": 0,
            },
        )

    tracker.update = update
    result = None
    for index in range(3):
        result = value.update(_observation(
            0.28 * index,
            temporal_flow=(
                {"valid": True, "ttc_s": 4.0, "support_beams": 4}
                if index == 0
                else None
            ),
        ))

    assert result.forecast == ("matched",)
    assert result.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert result.diagnostics["mapless_unknown_track_indices"] == (1, 2)
    assert result.diagnostics["tracks"][0][
        "mapless_temporal_flow_corroborated"
    ] is True
    assert result.diagnostics["tracks"][1][
        "mapless_temporal_flow_corroborated"
    ] is False


def test_explicit_synthetic_prime_warms_forecast_without_real_flow_match():
    value, tracker = _filter()
    value.dynamic_classification_temporal_corroboration_enabled = True
    result = None
    for index, lateral in enumerate((0.0, 0.19, 0.38)):
        tracker.position = (1.5, lateral)
        value.prime_dynamic_classification_temporal_corroboration()
        result = value.update(_observation(0.28 * index))

    assert result.forecast == ("forecast",)
    assert result.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert result.diagnostics["tracks"][0][
        "mapless_temporal_flow_corroborated"
    ] is True

    value.reset_filter_state()
    assert value.dynamic_classification_temporal_corroboration_primed is False


def test_strong_collision_course_publishes_before_radial_flow():
    value, tracker = _filter()
    value.dynamic_classification_temporal_corroboration_enabled = True
    value.dynamic_classification_collision_course_bypass_enabled = True
    result = None
    for index in range(6):
        tracker.position = (1.20, 1.00 - 0.16 * index)
        observation = _observation(0.12 * index)
        observation.twist.v = 0.50
        result = value.update(observation)

    assert result.forecast == ("forecast",)
    assert result.diagnostics["mapless_dynamic_track_indices"] == (0,)
    track = result.diagnostics["tracks"][0]
    assert track["mapless_temporal_flow_corroborated"] is False
    assert track["mapless_strong_motion_collision_course"] is True
    course = track[
        "mapless_strong_motion_collision_course_diagnostics"
    ]
    assert 0.0 < course["closest_approach_time_s"] <= 4.0
    assert course["closest_approach_distance_m"] <= 1.20


def test_collision_course_bypass_rejects_short_cluster_hop():
    value, tracker = _filter()
    value.dynamic_classification_temporal_corroboration_enabled = True
    value.dynamic_classification_collision_course_bypass_enabled = True
    result = None
    for index, lateral in enumerate((0.0, 0.19, 0.38)):
        tracker.position = (1.20, lateral)
        result = value.update(_observation(0.20 * index))

    assert result.forecast is None
    assert result.diagnostics["mapless_dynamic_track_indices"] == ()
    assert result.diagnostics[
        "mapless_strong_motion_collision_course_track_indices"
    ] == ()


def test_confirmed_dynamic_identity_survives_track_reset_handoff():
    value, tracker = _filter()
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.19)
    confirmed = value.update(_observation(0.28))
    assert confirmed.forecast == ("forecast",)

    tracker.update_count = 0
    tracker.position = (1.5, 0.38)
    inherited = value.update(_observation(0.56))
    assert inherited.forecast == ("forecast",)
    assert inherited.diagnostics["tracks"][0][
        "mapless_dynamic_inherited"
    ]
    assert inherited.diagnostics["mapless_dynamic_track_indices"] == (0,)


def test_association_jump_resets_motion_evidence():
    value, tracker = _filter()
    for index in range(4):
        tracker.position = (1.0, 0.05 * index)
        value.update(_observation(0.15 * index))
    tracker.position = (1.0, 0.65)
    tracker.association_distance = 0.50
    result = value.update(_observation(0.60))
    assert result.forecast is None
    assert result.diagnostics["mapless_unknown_track_indices"] == (0, 1, 2)
    evidence = result.diagnostics["tracks"][0]["mapless_motion_evidence"]
    assert evidence["sample_count"] == 1


def test_provisional_two_frame_threats_are_not_forwarded():
    value, tracker = _filter()
    original = tracker.update

    def update(observation):
        result = original(observation)
        diagnostics = dict(result.diagnostics)
        diagnostics["provisional_collision_candidates"] = ({"fake": True},)
        diagnostics["provisional_collision_candidate_count"] = 1
        return OnlineTrackingUpdate(
            result.measurement, result.forecast, diagnostics
        )

    tracker.update = update
    result = value.update(_observation(0.0))
    assert result.diagnostics["provisional_collision_candidates"] == ()
    assert result.diagnostics["provisional_collision_candidate_count"] == 0


def test_only_vetted_wide_vehicle_provisional_threat_is_forwarded():
    value, tracker = _filter()
    original = tracker.update

    def update(observation):
        result = original(observation)
        diagnostics = dict(result.diagnostics)
        diagnostics["provisional_collision_candidates"] = ({
            "dynamic_shape": "wide_vehicle",
            "streak": 2,
            "total_displacement_m": 0.12,
            "closing_speed_mps": 0.40,
            "closest_approach_time_s": 2.5,
            "closest_approach_distance_m": 0.70,
            "distance_m": 2.0,
            "lateral_m": 0.3,
        },)
        return OnlineTrackingUpdate(
            result.measurement, result.forecast, diagnostics
        )

    tracker.update = update
    result = value.update(_observation(0.0))
    candidates = result.diagnostics["provisional_collision_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["early_wide_vehicle"]


def test_flow_matched_ca_imm_forecast_is_admitted_during_bootstrap_hold():
    value, tracker = _filter()
    value.allow_temporal_flow_provisional = True
    tracker.temporal_flow_threat_hold_cycles = 6
    result = value.update(_observation(0.0))
    assert result.forecast == ("forecast",)
    assert result.diagnostics[
        "mapless_temporal_flow_provisional_track_indices"
    ] == (0,)
    assert result.diagnostics["tracks"][0][
        "mapless_temporal_flow_provisional"
    ] is True


def test_person_provisional_gate_releases_shape_backed_unknown_forecast():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    value.person_provisional_enabled = True
    value.person_provisional_minimum_streak = 2
    value.vehicle_shape_hold_cycles = 6
    value.person_provisional_minimum_samples = 4
    value.person_provisional_minimum_duration_s = 0.35
    value.person_provisional_minimum_direction_coherence = 0.55
    value.dynamic_classification_temporal_corroboration_enabled = True
    samples = (
        (0.00, 12, 0.00),
        (0.15, 3, 0.08),
        (0.30, 3, 0.16),
        (0.45, 3, 0.24),
        (0.60, 3, 0.32),
    )
    result = None
    for timestamp, support, lateral in samples:
        tracker.selected_support_beams = support
        tracker.position = (1.5, lateral)
        result = value.update(_observation(timestamp))
    assert result.diagnostics["tracks"][0][
        "mapless_classification"
    ] == "unknown"
    assert result.diagnostics[
        "mapless_person_provisional_track_indices"
    ] == (0,)
    assert result.diagnostics["person_forecast_candidate_track_indices"] == (0,)
    assert result.forecast == ("forecast",)


def test_person_provisional_gate_rejects_one_frame_unknown_motion():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    value.person_provisional_enabled = True
    value.person_provisional_minimum_streak = 2
    tracker.selected_support_beams = 12
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.selected_support_beams = 3
    tracker.position = (1.5, 0.20)
    result = value.update(_observation(0.20))
    assert result.diagnostics["mapless_person_provisional_track_indices"] == ()
    assert result.diagnostics["tracks"][0][
        "mapless_person_provisional_streak"
    ] == 0
    assert result.diagnostics["tracks"][0][
        "mapless_classification"
    ] == "unknown"


def test_3d_body_evidence_replaces_sparse_2d_shape_proxy_not_motion_gate():
    value, tracker = _filter()
    value.allow_compact_dynamic = False
    value.person_provisional_enabled = True
    value.person_provisional_minimum_streak = 2
    value.human_point_cloud = HumanPointCloudEvidenceExtractor({
        "enabled": True,
    })
    tracker.selected_support_beams = 1
    points = np.asarray([
        (1.5 + dx, lateral, height)
        for height in (0.20, 0.35, 0.75, 0.95, 1.35, 1.55)
        for dx in (-0.10, 0.0, 0.10)
        for lateral in (-0.12, 0.0, 0.12)
    ])
    result = None
    for timestamp, lateral in (
        (0.00, 0.00),
        (0.15, 0.08),
        (0.30, 0.16),
        (0.45, 0.24),
        (0.60, 0.32),
    ):
        tracker.position = (1.5, lateral)
        shifted = points.copy()
        shifted[:, 1] += lateral
        result = value.update(_observation(
            timestamp, point_cloud=shifted
        ))
    track = result.diagnostics["tracks"][0]
    assert track["point_cloud_human_candidate"] is True
    assert result.diagnostics[
        "human_point_cloud_candidate_track_indices"
    ] == (0,)
    assert result.diagnostics[
        "mapless_person_provisional_track_indices"
    ] == (0,)
    assert result.forecast == ("forecast",)
