from mobile_robot_mppi.real_robot.person_track_manager import PersonTrackManager


def _track(index, x, y, vx, vy, *, dynamic=True, forecast=True, support=6):
    return {
        "track_index": index,
        "associated": True,
        "measurement_x": x,
        "measurement_y": y,
        "measurement_velocity_x_mps": vx,
        "measurement_velocity_y_mps": vy,
        "measurement_speed_mps": (vx * vx + vy * vy) ** 0.5,
        "selected_support_beams": support,
        "forecast_valid": forecast,
        "mapless_classification": "dynamic" if dynamic else "unknown",
        "mapless_temporal_flow_corroborated": dynamic,
    }


def test_person_identity_survives_low_level_slot_swap():
    manager = PersonTrackManager()
    first = manager.update(
        {
            "tracks": (_track(0, 1.5, 0.3, 0.0, 0.4),),
            "forecast_track_indices": (0,),
        },
        forecasts=(object(),),
        timestamp_s=1.0,
    )
    person_id = first["person_selected_id"]
    second = manager.update(
        {
            "tracks": (_track(2, 1.5, 0.34, 0.0, 0.4),),
            "forecast_track_indices": (2,),
        },
        forecasts=(object(),),
        timestamp_s=1.1,
    )
    assert second["person_selected_id"] == person_id
    assert second["person_identity_continuity"] is True
    assert second["tracks"][0]["person_id"] == person_id
    assert second["tracks"][0]["person_member_track_indices"] == (2,)


def test_two_leg_fragments_fuse_into_one_person():
    manager = PersonTrackManager()
    diagnostics = manager.update(
        {
            "tracks": (
                _track(0, 1.4, -0.18, 0.0, 0.35),
                _track(1, 1.4, 0.18, 0.02, 0.33),
            ),
            "forecast_track_indices": (0, 1),
        },
        forecasts=(object(), object()),
        timestamp_s=1.0,
    )
    assert diagnostics["person_active_count"] == 1
    assert diagnostics["person_selected_member_track_indices"] == (0, 1)
    assert diagnostics["tracks"][0]["person_id"] == diagnostics["tracks"][1][
        "person_id"
    ]


def test_forecast_qualification_distinguishes_valid_provisional_and_stale():
    manager = PersonTrackManager({"maximum_forecast_age_s": 0.30})
    valid = manager.update(
        {
            "tracks": (_track(0, 1.2, 0.0, -0.2, 0.0),),
            "forecast_track_indices": (0,),
        },
        forecasts=(object(),),
        timestamp_s=1.0,
    )
    assert valid["person_forecast_qualification"] == "VALID"

    provisional_track = _track(
        0, 1.18, 0.0, -0.2, 0.0, dynamic=False, forecast=True, support=2
    )
    provisional_track["mapless_temporal_flow_corroborated"] = True
    provisional = manager.update(
        {
            "tracks": (provisional_track,),
            "forecast_track_indices": (0,),
        },
        forecasts=(object(),),
        timestamp_s=1.1,
    )
    assert provisional["person_forecast_qualification"] == "PROVISIONAL"

    stale_track = dict(provisional_track)
    stale_track["forecast_valid"] = True
    stale = manager.update(
        {"tracks": (stale_track,), "forecast_track_indices": ()},
        forecasts=(),
        timestamp_s=1.2,
    )
    assert stale["person_forecast_qualification"] == "STALE"


def test_static_cluster_is_never_promoted_to_person():
    manager = PersonTrackManager()
    static = _track(0, 1.0, 0.0, 0.0, 0.0)
    static["mapless_classification"] = "static"
    diagnostics = manager.update(
        {"tracks": (static,), "forecast_track_indices": (0,)},
        forecasts=(object(),),
        timestamp_s=1.0,
    )
    assert diagnostics["person_active_count"] == 0
    assert diagnostics["person_forecast_qualification"] == "INVALID"


def test_unqualified_unknown_track_is_not_geometry_source():
    manager = PersonTrackManager()
    unknown = _track(
        0, 1.0, 0.0, 0.4, 0.0, dynamic=False, forecast=True, support=8
    )
    unknown["mapless_temporal_flow_corroborated"] = False
    diagnostics = manager.update(
        {
            "tracks": (unknown,),
            "forecast_track_indices": (),
        },
        forecasts=(),
        timestamp_s=1.0,
    )
    assert diagnostics["person_active_count"] == 1
    assert diagnostics["person_forecast_qualification"] == "INVALID"
    assert diagnostics["person_selected_id"] is None
    assert diagnostics["person_selected_position_x"] is None
