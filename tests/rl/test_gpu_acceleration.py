"""Regression tests for the opt-in CUDA planner hot paths."""

import numpy as np
import pytest
import torch

from mobile_robot_mppi.obstacles.collision_risk import (
    CollisionRiskConfig,
    GaussianMixtureObstacleForecast,
    evaluate_collision_risk,
    evaluate_collision_risk_cuda,
)
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


def _forecast(seed, radius):
    rng = np.random.RandomState(seed)
    horizon, modes = 7, 3
    means = rng.uniform(-0.4, 1.0, size=(horizon, modes, 2))
    covariances = np.zeros((horizon, modes, 2, 2), dtype=np.float64)
    for step in range(horizon):
        for mode in range(modes):
            scale = 0.01 + 0.03 * rng.rand()
            covariances[step, mode] = (
                (scale * np.asarray(((1.0, 0.002), (0.002, 1.3))))
            )
    weights = rng.uniform(0.1, 1.0, size=(horizon, modes))
    weights /= weights.sum(axis=1, keepdims=True)
    return GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=means,
        component_covariances=covariances,
        component_weights=weights,
        radius_m=radius,
        source="cuda_regression",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_collision_risk_matches_cpu_and_hard_mask():
    rng = np.random.RandomState(11)
    robot = rng.uniform(-0.8, 1.2, size=(9, 7, 2))
    forecasts = (_forecast(21, 0.12), _forecast(22, 0.15))
    config = CollisionRiskConfig(
        robot_radius_m=0.25,
        safety_margin_m=0.10,
        minimum_position_std_m=0.01,
        hard_probability_threshold=0.35,
    )
    cpu = evaluate_collision_risk(robot, forecasts, config)
    cuda = evaluate_collision_risk_cuda(robot, forecasts, config, device="cuda")
    np.testing.assert_allclose(
        cuda.step_probability_upper_bound,
        cpu.step_probability_upper_bound,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        cuda.accumulated_probability_mass,
        cpu.accumulated_probability_mass,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(cuda.horizon_union_bound, cpu.horizon_union_bound,
                               rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        cuda.maximum_step_probability,
        cpu.maximum_step_probability,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_array_equal(cuda.hard_violation, cpu.hard_violation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_graph_captures_frozen_correction_actor_with_identical_output():
    base = SACAgent(
        5, 2, SACConfig(hidden_sizes=(16, 16)), device="cuda", seed=101
    )
    correction = SACAgent(
        5,
        2,
        SACConfig(
            hidden_sizes=(16, 16),
            policy_mode="frozen_bc_correction",
            correction_scale=(0.20, 0.10),
        ),
        device="cuda",
        seed=102,
    )
    correction.initialize_frozen_base_actor(base.actor.state_dict())
    observations = torch.linspace(
        -0.4, 0.4, 15, device="cuda", dtype=torch.float32
    ).reshape(3, 5)
    with torch.inference_mode():
        eager = correction._evaluate_policy_gaussian_parameters_torch(
            observations
        )
    graphed = correction.policy_gaussian_parameters_batch_torch(observations)
    torch.cuda.synchronize()
    assert len(correction._policy_gaussian_cuda_graphs) == 1
    assert correction._last_policy_gaussian_backend == "cuda"
    assert correction._last_policy_gaussian_cuda_graph_used is True
    for eager_value, graph_value in zip(eager, graphed):
        if eager_value is None:
            assert graph_value is None
        else:
            torch.testing.assert_close(eager_value, graph_value, rtol=0.0, atol=0.0)
