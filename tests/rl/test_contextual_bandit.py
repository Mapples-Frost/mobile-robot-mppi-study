import json

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.rl.contextual_bandit import (
    ContextualBanditCovariancePrior,
    LinUCBCovarianceBandit,
    REFERENCE_GEOMETRY_FEATURE_NAMES,
    polyline_geometry_features,
    polyline_pose_at_progress,
    polyline_window,
    project_polyline_progress,
)


def test_polyline_features_are_rigid_transform_invariant():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.5, 0.5), (1.5, 1.0)))
    angle = 0.73
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    transformed = points @ rotation.T + np.asarray((4.0, -2.0))
    np.testing.assert_allclose(
        polyline_geometry_features(points),
        polyline_geometry_features(transformed),
        atol=1e-12,
    )


def test_polyline_progress_window_and_projection_are_consistent():
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (3.0, 2.0))
    point, theta = polyline_pose_at_progress(points, 2.0)
    np.testing.assert_allclose(point, (1.0, 1.0))
    assert theta == np.pi / 2.0
    window = polyline_window(points, 0.5, 2.0)
    np.testing.assert_allclose(window, ((0.5, 0.0), (1.0, 0.0), (1.0, 1.5)))
    projected = project_polyline_progress(
        points, (1.05, 1.2), minimum=0.5, maximum=3.5
    )
    assert abs(projected - 2.2) < 1e-12


def test_linucb_round_trip_and_greedy_decision():
    bandit = LinUCBCovarianceBandit(("narrow", "speed"), 2, ridge=0.2, exploration_alpha=1.0)
    for _ in range(8):
        bandit.update((1.0, 0.0), "narrow", -1.0)
        bandit.update((1.0, 0.0), "speed", 1.0)
        bandit.update((1.0, 1.0), "narrow", 2.0)
        bandit.update((1.0, 1.0), "speed", -2.0)
    assert bandit.decide((1.0, 0.0)).action == "speed"
    assert bandit.decide((1.0, 1.0)).action == "narrow"
    restored = LinUCBCovarianceBandit.from_state_dict(bandit.state_dict())
    assert restored.decide((1.0, 1.0)).action == "narrow"


def test_linucb_prediction_cache_is_invalidated_by_update():
    bandit = LinUCBCovarianceBandit(("left", "right"), 2, ridge=1.0)
    before = bandit.predict((1.0, 0.0))[0]["left"]
    cached_inverse = bandit._inverse_cache["left"]
    bandit.predict((1.0, 0.0))
    assert bandit._inverse_cache["left"] is cached_inverse
    bandit.update((1.0, 0.0), "left", 2.0)
    assert bandit._inverse_cache["left"] is None
    after = bandit.predict((1.0, 0.0))[0]["left"]
    assert after > before


def test_contextual_prior_checkpoint_uses_selected_covariance(tmp_path):
    bandit = LinUCBCovarianceBandit(
        ("narrow", "speed"), len(REFERENCE_GEOMETRY_FEATURE_NAMES), ridge=0.1
    )
    features = polyline_geometry_features(((0.0, 0.0), (1.0, 0.0)))
    for _ in range(5):
        bandit.update(features, "narrow", -2.0)
        bandit.update(features, "speed", 1.0)
    checkpoint = tmp_path / "bandit.json"
    checkpoint.write_text(json.dumps({
        "feature_names": list(REFERENCE_GEOMETRY_FEATURE_NAMES),
        "action_scales": {"narrow": [0.5, 0.5], "speed": [1.75, 0.75]},
        "bandit": bandit.state_dict(),
    }))
    action_spec = ActionSpec(
        names=("v_cmd", "omega_cmd"),
        lower=np.asarray((-0.2, -1.0)),
        upper=np.asarray((0.8, 1.0)),
    )
    prior = ContextualBanditCovariancePrior.from_checkpoint(
        checkpoint, GoalWarmStartPrior(), np.asarray((0.3, 0.4))
    )
    reference = PolylineReference(((0.0, 0.0), (1.0, 0.0)))
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
    )
    output = prior.propose(observation, reference, 4, action_spec)
    assert output.metadata["bandit_action"] == "speed"
    np.testing.assert_allclose(
        np.diag(output.covariance), np.square(np.asarray((0.3, 0.4)) * (1.75, 0.75))
    )
