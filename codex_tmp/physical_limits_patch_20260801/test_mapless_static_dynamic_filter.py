from types import SimpleNamespace

from mobile_robot_mppi.obstacles.online_tracking import OnlineTrackingUpdate

from mapless_static_dynamic_filter import MaplessStaticDynamicFilter


class _Tracker:
    def __init__(self):
        self.config = SimpleNamespace()
        self.maximum_tracks = 3
        self.position = (0.0, 0.0)
        self.association_distance = 0.02
        self.selected_support_beams = 10
        self.measurement_speed_mps = 0.70
        self.update_count = 0

    def reset(self):
        self.update_count = 0

    def update(self, observation):
        self.update_count += 1
        track = {
            "update_count": self.update_count,
            "associated": True,
            "association_distance_m": self.association_distance,
            "measurement_x": self.position[0],
            "measurement_y": self.position[1],
            "change_triggered": False,
            "recovery_active": False,
            "dropout_guard_triggered": False,
            "forecast_valid": True,
            "measurement_speed_mps": self.measurement_speed_mps,
            "selected_support_beams": self.selected_support_beams,
        }
        return OnlineTrackingUpdate(
            measurement=None,
            forecast=("forecast",),
            diagnostics={
                "tracks": (track, {}, {}),
                "forecast_track_indices": (0,),
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
    value.vehicle_minimum_duration_s = 0.27
    value.vehicle_minimum_speed_mps = 0.45
    value.vehicle_minimum_displacement_m = 0.18
    value.vehicle_minimum_tracker_speed_mps = 0.25
    value.vehicle_minimum_direction_coherence = 0.92
    value.vehicle_maximum_fit_residual_m = 0.03
    value.vehicle_maximum_step_m = 0.60
    value.allow_compact_dynamic = True
    value.vehicle_shape_hold_cycles = 4
    value.vehicle_provisional_minimum_streak = 2
    value.vehicle_provisional_minimum_displacement_m = 0.10
    value.vehicle_provisional_minimum_closing_speed_mps = 0.15
    value.vehicle_provisional_maximum_ttc_s = 3.00
    value.vehicle_provisional_maximum_closest_approach_m = 0.90
    value.dynamic_memory_ttl_s = 0.90
    value.dynamic_handoff_gate_m = 0.35
    value.reset_filter_state()
    return value, tracker


def _observation(timestamp, robot_x=0.0):
    return SimpleNamespace(
        timestamp=float(timestamp),
        pose=SimpleNamespace(x=float(robot_x), y=0.0, theta=0.0),
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


def test_smooth_world_motion_becomes_dynamic_without_a_map():
    value, tracker = _filter()
    result = None
    for index in range(6):
        tracker.position = (1.5, 0.10 * index)
        result = value.update(_observation(0.15 * index))
    assert result.forecast == ("forecast",)
    assert result.diagnostics["mapless_dynamic_track_indices"] == (0,)
    assert result.diagnostics["valid_forecast_count"] == 1


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
    result = value.update(_observation(0.56))
    track = result.diagnostics["tracks"][0]
    assert result.forecast == ("forecast",)
    assert track["mapless_dynamic_mode"] == "wide_vehicle"
    assert track["mapless_vehicle_shape_recent"]


def test_wide_fast_vehicle_motion_has_a_dedicated_dynamic_mode():
    value, tracker = _filter()
    tracker.selected_support_beams = 38
    tracker.measurement_speed_mps = 0.52
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.25)
    result = value.update(_observation(0.28))
    assert result.forecast == ("forecast",)
    assert result.diagnostics["tracks"][0][
        "mapless_dynamic_mode"
    ] == "wide_vehicle"


def test_two_frame_small_jitter_remains_unknown():
    value, tracker = _filter()
    tracker.position = (1.5, 0.0)
    value.update(_observation(0.0))
    tracker.position = (1.5, 0.10)
    result = value.update(_observation(0.28))
    assert result.forecast is None
    assert result.diagnostics["mapless_dynamic_track_indices"] == ()


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
