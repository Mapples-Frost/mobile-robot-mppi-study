import pytest
import numpy as np
from types import SimpleNamespace

from mobile_robot_mppi.rl.environment import (
    MppiPriorEnv,
    RewardConfig,
    _cross_track_error,
    _initialize_reference_progress,
    _path_tracking_state,
    _progress_signal,
    _validated_path_progress,
)
from mobile_robot_mppi.core.references import PolylineReference


def test_distance_delta_progress_has_no_stationary_reward():
    assert _progress_signal(4.0, 4.0, 0.99, "distance_delta") == pytest.approx(0.0)
    assert _progress_signal(4.0, 3.9, 0.99, "distance_delta") > 0.0
    assert _progress_signal(4.0, 4.1, 0.99, "distance_delta") < 0.0


def test_discounted_potential_mode_preserves_original_expression():
    assert _progress_signal(4.0, 4.0, 0.99, "discounted_potential") == pytest.approx(0.04)


def test_reward_config_parses_new_costs_and_rejects_unknown_mode():
    config = RewardConfig.from_mapping({
        "progress_mode": "distance_delta",
        "distance_penalty_weight": 0.2,
        "path_length_penalty_weight": 0.3,
        "cross_track_penalty_weight": 0.4,
        "path_progress_weight": 2.0,
        "path_progress_corridor": 0.5,
        "path_progress_step_cap": 0.1,
        "path_progress_hard_corridor": True,
        "heading_error_penalty_weight": 0.6,
        "cross_track_error_cap": 1.5,
        "reward_scale": 0.1,
    })
    config.validate()
    assert config.distance_penalty_weight == pytest.approx(0.2)
    assert config.path_length_penalty_weight == pytest.approx(0.3)
    assert config.cross_track_penalty_weight == pytest.approx(0.4)
    assert config.path_progress_weight == pytest.approx(2.0)
    assert config.path_progress_corridor == pytest.approx(0.5)
    assert config.path_progress_step_cap == pytest.approx(0.1)
    assert config.path_progress_hard_corridor
    assert config.heading_error_penalty_weight == pytest.approx(0.6)
    assert config.cross_track_error_cap == pytest.approx(1.5)
    assert config.reward_scale == pytest.approx(0.1)

    with pytest.raises(ValueError, match="progress_mode"):
        RewardConfig(progress_mode="unknown").validate()
    with pytest.raises(ValueError, match="reward_scale"):
        RewardConfig(reward_scale=0.0).validate()


def test_validated_progress_rejects_off_corridor_projection_jump():
    assert _validated_path_progress(1.0, 8.0, 2.0, 0.75) == pytest.approx(1.0)
    assert _validated_path_progress(1.0, 1.2, 0.2, 0.75) == pytest.approx(1.2)


def test_reference_progress_initializes_at_midpath_reset_before_windowing():
    reference = PolylineReference(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)),
        projection_forward_distance=0.25,
    )
    assert _initialize_reference_progress(reference, (2.0, 1.0)) == pytest.approx(3.0)
    projection = reference.project(
        np.asarray((2.0, 1.1)), minimum_progress=reference.progress
    )
    assert projection.progress == pytest.approx(3.1)


def test_reward_caps_false_progress_and_cross_track_scale():
    environment = MppiPriorEnv.__new__(MppiPriorEnv)
    environment.reward_config = RewardConfig(
        progress_mode="distance_delta",
        potential_progress_weight=0.0,
        path_progress_weight=14.0,
        path_progress_corridor=0.75,
        path_progress_step_cap=0.1,
        path_progress_hard_corridor=True,
        cross_track_penalty_weight=4.0,
        cross_track_error_cap=1.5,
        step_penalty=0.01,
        reward_scale=0.1,
    )
    environment.gamma = 0.99
    environment.config = {
        "experiment": {"control_dt": 0.1},
        "task": {"position_tolerance": 0.2},
    }
    environment.previous_distance = 5.0
    environment.previous_path_progress = 1.0
    environment.previous_control = np.zeros(2, dtype=np.float64)
    decision = SimpleNamespace(
        overridden=False,
        executed_control=SimpleNamespace(values=np.zeros(2, dtype=np.float64)),
    )
    truth = SimpleNamespace(
        twist=SimpleNamespace(v=0.1),
        collision=False,
        minimum_clearance=float("inf"),
    )
    reward, terms = environment._reward(
        5.0,
        decision,
        truth,
        False,
        path_state={
            "valid": True,
            "progress": 8.0,
            "cross_track_error": 3.0,
            "heading_error": 0.0,
        },
    )
    assert terms["path_progress"] == 0.0
    assert terms["cross_track"] == pytest.approx(-0.9)
    assert reward == pytest.approx(-0.901)


def test_cross_track_error_projects_onto_polyline_segments():
    reference = SimpleNamespace(
        points=np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    )
    assert _cross_track_error(
        reference, SimpleNamespace(x=0.4, y=0.3)
    ) == pytest.approx(0.3)
    assert _cross_track_error(
        reference, SimpleNamespace(x=1.2, y=0.6)
    ) == pytest.approx(0.2)
    assert _cross_track_error(
        SimpleNamespace(), SimpleNamespace(x=5.0, y=5.0)
    ) == 0.0


def test_path_tracking_state_uses_live_progress_floor_and_wrapped_heading():
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        lookahead_distance=0.2,
    )
    reference.target_at(0.0, np.asarray((0.5, 0.0, 0.0)))
    state = _path_tracking_state(
        reference, SimpleNamespace(x=0.7, y=0.2, theta=2.0 * np.pi - 0.1)
    )

    assert state["valid"]
    assert state["progress"] >= 0.5
    assert state["cross_track_error"] == pytest.approx(0.2)
    assert state["signed_cross_track_error"] > 0.0
    assert state["heading_error"] == pytest.approx(-0.1)

    neutral = _path_tracking_state(
        SimpleNamespace(), SimpleNamespace(x=1.0, y=2.0, theta=0.0)
    )
    assert not neutral["valid"]
    assert neutral["cross_track_error"] == 0.0
