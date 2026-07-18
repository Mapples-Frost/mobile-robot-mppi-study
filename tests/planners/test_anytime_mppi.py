import numpy as np

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.core.spaces import body_velocity_action, dynamic_unicycle_state
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.planning.anytime_mppi import AnytimeMppiController
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig
from mobile_robot_mppi.policies.priors import (
    FixedCovariancePrior,
    GoalWarmStartPrior,
)
from mobile_robot_mppi.rl.budget_bandit import PrimalDualBudgetBandit
from mobile_robot_mppi.rl.contextual_bandit import (
    REFERENCE_GEOMETRY_FEATURE_NAMES,
)


FEATURES = (
    "ess_fraction",
    "weight_entropy_fraction",
    "maximum_weight",
    "split_control_disagreement",
    "weighted_action_dispersion",
    "normalized_cost_spread",
    "sample_saturation_fraction",
    "weighted_perturbation_norm",
    "residual_reliability_alpha",
    "residual_reliability_last_relative_improvement",
) + tuple(
    "local_" + name for name in REFERENCE_GEOMETRY_FEATURE_NAMES[1:]
)


def _bandit(add):
    result = PrimalDualBudgetBandit(
        FEATURES,
        np.zeros(len(FEATURES)),
        np.ones(len(FEATURES)),
        ridge=1.0,
    )
    result._b[0] = 1.0 if add else -1.0
    return result


def _controller(add):
    action = body_velocity_action((0.0, 0.5), 1.0)
    config = MppiConfig(
        horizon=6,
        num_samples=10,
        dt=0.1,
        noise_sigma=(0.08, 0.20),
        seed=19,
    )
    prior = FixedCovariancePrior(
        GoalWarmStartPrior(), config.noise_sigma, (1.0, 1.0)
    )
    return AnytimeMppiController(
        DynamicUnicyclePrediction(),
        dynamic_unicycle_state(),
        action,
        config,
        sampling_prior=prior,
        anytime_config={
            "base_samples": 5,
            "local_route_window_m": 1.0,
            "checkpoint": "injected",
        },
        budget_bandit=_bandit(add),
    )


def _observation():
    return RobotObservation(
        0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(0.0, 0.0)
    )


def _reference():
    return PolylineReference(
        ((0.0, 0.0), (0.8, 0.0), (1.2, 0.4), (2.0, 0.4)),
        lookahead_distance=0.4,
    )


def test_anytime_stop_uses_only_byte_identical_prefix_budget():
    result = _controller(False).plan(_observation(), _reference())
    assert result.diagnostics["optimizer"] == "anytime_bandit"
    assert result.diagnostics["anytime_selected_samples"] == 5
    assert result.diagnostics["anytime_add_samples"] is False
    assert result.diagnostics["anytime_maximum_samples"] == 10
    assert np.isfinite(result.control_sequence).all()


def test_anytime_add_reweights_full_nested_batch():
    result = _controller(True).plan(_observation(), _reference())
    assert result.diagnostics["anytime_selected_samples"] == 10
    assert result.diagnostics["anytime_add_samples"] is True
    assert result.diagnostics["anytime_score"] > 0.0
    assert result.predicted_trajectory.shape == (7, 5)


def test_budget_bandit_prediction_cache_is_invalidated_by_update():
    bandit = PrimalDualBudgetBandit(
        ("feature",), np.zeros(1), np.ones(1), seed=3
    )
    features = {"feature": 0.25}
    first = bandit.predict(features)
    cached_inverse = bandit._inverse_cache
    cached_theta = bandit._theta_cache
    assert bandit.predict(features) == first
    assert bandit._inverse_cache is cached_inverse
    assert bandit._theta_cache is cached_theta

    bandit.update(features, True, observed_advantage=0.2)
    assert bandit._inverse_cache is None
    assert bandit._theta_cache is None
    updated = bandit.predict(features)
    assert np.isfinite(updated).all()
    assert bandit._inverse_cache is not cached_inverse


def test_latched_add_uses_one_full_batch_between_decision_steps():
    controller = _controller(True)
    controller.anytime_config = type(controller.anytime_config)(
        base_samples=5,
        local_route_window_m=1.0,
        bandit_checkpoint="injected",
        decision_interval_steps=2,
    )
    first = controller.plan(_observation(), _reference())
    second = controller.plan(_observation(), _reference())
    third = controller.plan(_observation(), _reference())
    assert first.diagnostics["anytime_decision_refreshed"] is True
    assert second.diagnostics["anytime_decision_refreshed"] is False
    assert third.diagnostics["anytime_decision_refreshed"] is True
    assert second.diagnostics["anytime_selected_samples"] == 10
