from types import SimpleNamespace

import numpy as np
import pytest

from mobile_robot_mppi.core.references import PointGoal, PolylineReference
from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    dynamic_unicycle_state,
)
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.planning.static_astar import (
    _point_clearance,
    plan_static_astar_path,
)


BOX = {
    "type": "box",
    "position": [2.0, 0.0],
    "size": [0.35, 0.75],
    "yaw": 0.0,
}


def _controller(**overrides):
    values = dict(
        horizon=6,
        num_samples=8,
        dt=0.1,
        noise_sigma=(0.05, 0.1),
        seed=7,
    )
    values.update(overrides)
    return MppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        body_velocity_action((0.0, 0.8), 1.2),
        MppiConfig(**values),
    )


def test_static_astar_routes_around_frozen_box_and_keeps_exact_endpoints():
    route = plan_static_astar_path(
        (0.0, 0.0),
        (4.0, 0.0),
        (BOX,),
        robot_radius=0.18,
        clearance_margin=0.08,
        resolution=0.10,
    )

    np.testing.assert_allclose(route[0], (0.0, 0.0))
    np.testing.assert_allclose(route[-1], (4.0, 0.0))
    assert len(route) >= 3
    assert np.max(np.abs(route[:, 1])) > 0.75


def test_static_astar_segment_thickness_matches_mujoco_full_width_contract():
    wall = {
        "type": "segment",
        "start": (-1.0, 0.0),
        "end": (1.0, 0.0),
        "thickness": 0.20,
    }

    clearance = _point_clearance(
        0.0,
        0.50,
        (wall,),
        robot_radius=0.25,
    )

    # MuJoCo represents the segment as a box with half-width
    # 0.5 * thickness, so 0.50 - 0.10 - 0.25 = 0.15 m.
    assert clearance == pytest.approx(0.15)


def test_polyline_replacement_preserves_contract_and_restarts_progress():
    reference = PolylineReference(
        ((0.0, 0.0), (4.0, 0.0)),
        tolerance=0.17,
        lookahead_distance=0.42,
        terminal_approach_distance=0.8,
    )
    reference.target_at(0.0, np.asarray((1.0, 0.0, 0.0)))
    assert reference.progress > 0.0

    reference.replace_points(((1.0, 0.0), (1.0, 1.0), (4.0, 1.0)))

    assert reference.progress == 0.0
    assert reference.tolerance == pytest.approx(0.17)
    assert reference.lookahead_distance == pytest.approx(0.42)
    np.testing.assert_allclose(reference.points[0], (1.0, 0.0))
    np.testing.assert_allclose(reference.points[-1], (4.0, 1.0))


def test_reference_authority_is_filtered_and_bounded():
    controller = _controller(
        probabilistic_obstacle_risk_enabled=True,
        probabilistic_reference_authority_enabled=True,
        probabilistic_reference_authority_filter_alpha=0.5,
        probabilistic_reference_authority_minimum=0.2,
    )
    risk = SimpleNamespace(
        maximum_step_probability=np.asarray((1.0, 0.8)),
        accumulated_probability_mass=np.asarray((1.0, 1.0)),
    )

    authority_first, raw_first = (
        controller._probabilistic_reference_authority(risk, 1)
    )
    authority_second, raw_second = (
        controller._probabilistic_reference_authority(risk, 1)
    )

    assert raw_first == raw_second == pytest.approx(0.8)
    assert authority_first == pytest.approx(0.6)
    assert authority_second == pytest.approx(0.4)
    for _ in range(20):
        authority_second, _ = (
            controller._probabilistic_reference_authority(risk, 1)
        )
    assert authority_second == pytest.approx(0.2)


def test_static_replan_uses_current_pose_and_static_geometry_only():
    controller = _controller(
        static_astar_replan_enabled=True,
        static_astar_replan_resolution_m=0.10,
        static_astar_replan_clearance_margin_m=0.08,
        static_astar_replan_deviation_m=0.50,
        static_astar_replan_stagnation_steps=30,
        static_astar_replan_minimum_progress_m=0.10,
        static_astar_replan_cooldown_steps=10,
    )
    reference = PolylineReference(((0.0, 0.0), (4.0, 0.0)))
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 1.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={
            "known_static_obstacles": (BOX,),
            # A dynamic forecast is intentionally present but is not an input
            # to the static-only replanner.
            "probabilistic_obstacles": ("dynamic-track-placeholder",),
        },
    )
    state = controller.state_from_observation(observation)

    diagnostics = controller._maybe_replan_static_reference(
        state, observation, reference
    )

    assert diagnostics["triggered"]
    assert diagnostics["reason"] == "deviation"
    np.testing.assert_allclose(reference.points[0], (0.0, 1.0))
    np.testing.assert_allclose(reference.points[-1], (4.0, 0.0))


def test_zero_stagnation_threshold_disables_redundant_static_replans():
    controller = _controller(
        static_astar_replan_enabled=True,
        static_astar_replan_stagnation_steps=0,
        static_astar_replan_deviation_m=0.50,
        static_astar_replan_minimum_progress_m=0.10,
    )
    reference = PolylineReference(((0.0, 0.0), (4.0, 0.0)))
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={"known_static_obstacles": (BOX,)},
    )
    state = controller.state_from_observation(observation)

    diagnostics = None
    for _ in range(60):
        diagnostics = controller._maybe_replan_static_reference(
            state, observation, reference
        )

    assert diagnostics is not None
    assert not diagnostics["triggered"]
    assert diagnostics["reason"] == "none"
    assert diagnostics["count"] == 0
    assert diagnostics["stagnation_steps"] == 60


def test_hard_static_candidate_filter_rejects_colliding_low_cost_path():
    obstacle = {
        "type": "box",
        "position": [0.40, 0.0],
        "size": [0.10, 0.30],
        "yaw": 0.0,
    }
    controller = _controller(
        horizon=5,
        num_samples=4,
        known_static_map_cost_enabled=True,
        known_static_map_influence_m=0.30,
        known_static_map_weight=10.0,
        known_static_map_collision_penalty=1000.0,
        known_static_map_candidate_filter_enabled=True,
    )
    straight = np.zeros((4, 5, 2), dtype=np.float64)
    straight[..., 0] = 0.8
    controller._sample = lambda prior, rng: straight.copy()
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={"known_static_obstacles": (obstacle,)},
    )

    result = controller.plan(observation, PointGoal(2.0, 0.0))

    assert result.proposed_control.values[0] == pytest.approx(0.0)
    assert result.diagnostics[
        "known_static_map_candidate_filter_enabled"
    ]
    assert result.diagnostics[
        "known_static_map_candidate_feasible_count"
    ] == 1
    assert result.diagnostics[
        "known_static_map_weighted_update_feasible"
    ]


def test_new_configuration_dependencies_fail_closed():
    with pytest.raises(ValueError, match="candidate filtering"):
        _controller(known_static_map_candidate_filter_enabled=True)
    with pytest.raises(ValueError, match="requires probabilistic"):
        _controller(probabilistic_reference_authority_enabled=True)
