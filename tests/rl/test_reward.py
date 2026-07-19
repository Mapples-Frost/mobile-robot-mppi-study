import pytest
import numpy as np
from types import SimpleNamespace

from mobile_robot_mppi.rl.environment import (
    RewardConfig,
    _cross_track_error,
    _path_tracking_state,
    _progress_signal,
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
        "heading_error_penalty_weight": 0.6,
    })
    config.validate()
    assert config.distance_penalty_weight == pytest.approx(0.2)
    assert config.path_length_penalty_weight == pytest.approx(0.3)
    assert config.cross_track_penalty_weight == pytest.approx(0.4)
    assert config.path_progress_weight == pytest.approx(2.0)
    assert config.path_progress_corridor == pytest.approx(0.5)
    assert config.heading_error_penalty_weight == pytest.approx(0.6)

    with pytest.raises(ValueError, match="progress_mode"):
        RewardConfig(progress_mode="unknown").validate()


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
