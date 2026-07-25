import inspect
from pathlib import Path

import numpy as np
import yaml

from mobile_robot_mppi.obstacles.imm import (
    IMM_MODEL_NAMES,
    OrdinaryIMMPredictor,
    brake_transition,
    run_online_imm_forecasts,
    turn_transition,
)
from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import generate_patrol_trajectory
from mobile_robot_mppi.obstacles.prediction import cv_transition


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research/ordinary_imm_v3_development.yaml"
OBSTACLE_CONFIG_PATH = (
    ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"
)


def _configs():
    predictor = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    obstacle = yaml.safe_load(
        OBSTACLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    return predictor, obstacle


def _predictor(observation_std=0.075):
    config, _ = _configs()
    imm = config["ordinary_imm"]
    return OrdinaryIMMPredictor(
        observation_std=observation_std,
        initial_velocity_std=config["prediction"]["initial_velocity_std"],
        initial_mode_probabilities=imm["initial_mode_probabilities"],
        transition_matrix=imm["transition_matrix"],
        turn_rate_radps=imm["turn_rate_radps"],
        brake_decay_rate_per_s=imm["brake_decay_rate_per_s"],
        process_acceleration_std=imm["process_acceleration_std"],
    )


def test_turn_transition_zero_rate_matches_cv_and_preserves_speed():
    np.testing.assert_allclose(turn_transition(0.05, 0.0), cv_transition(0.05))
    state = np.asarray((0.0, 0.0, 0.7, -0.2))
    propagated = turn_transition(0.05, 0.65).dot(state)
    assert np.isclose(
        np.linalg.norm(propagated[2:]), np.linalg.norm(state[2:])
    )


def test_brake_transition_reduces_velocity_without_position_jump():
    state = np.asarray((1.0, -2.0, 0.8, 0.3))
    propagated = brake_transition(0.05, 1.8).dot(state)
    assert np.linalg.norm(propagated[2:]) < np.linalg.norm(state[2:])
    assert np.linalg.norm(propagated[:2] - state[:2]) < 0.05


def test_imm_mode_probabilities_and_covariances_remain_valid():
    predictor = _predictor()
    for index in range(80):
        timestamp = 0.05 * index
        observation = np.asarray((0.03 * index, 0.005 * index ** 1.2))
        predictor.update(observation, timestamp)
        assert np.all(predictor.mode_probabilities >= 0.0)
        assert np.isclose(predictor.mode_probabilities.sum(), 1.0)
        if predictor.covariance is not None:
            assert np.linalg.eigvalsh(predictor.covariance).min() >= -1.0e-9
    forecast = predictor.forecast(60, 0.05)
    forecast.validate()
    assert forecast.component_means.shape == (60, 4, 4)
    assert forecast.mode_probabilities.shape == (60, 4)


def test_imm_can_propagate_through_missing_observations():
    predictor = _predictor()
    predictor.update(np.asarray((0.0, 0.0)), 0.0)
    predictor.update(np.asarray((0.04, 0.0)), 0.05)
    before = predictor.covariance.copy()
    for index in range(2, 22):
        predictor.update(None, 0.05 * index)
    after = predictor.covariance
    assert np.trace(after) > np.trace(before)
    assert np.isclose(predictor.mode_probabilities.sum(), 1.0)


def test_online_imm_interface_cannot_receive_truth_or_latent_intent():
    parameters = tuple(inspect.signature(run_online_imm_forecasts).parameters)
    assert parameters == (
        "times",
        "observations",
        "observed_mask",
        "predictor",
        "horizon",
        "forecast_dt",
        "forecast_stride_steps",
        "warmup_steps",
    )
    forbidden = {
        "truth",
        "truth_states",
        "metadata",
        "true_modes",
        "events",
        "waypoints",
        "future_schedule",
    }
    assert not forbidden.intersection(parameters)


def test_v3_hybrid_observations_run_without_metadata_input():
    config, obstacle_config = _configs()
    profiles = noise_profiles_from_mapping(obstacle_config["noise_profiles"])
    trajectory = generate_patrol_trajectory(
        "hybrid_patrol", 730100041, profiles["medium"], obstacle_config
    )
    horizon = int(
        round(
            config["prediction"]["horizon_s"]
            / obstacle_config["trajectory"]["dt"]
        )
    )
    records = run_online_imm_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        _predictor(),
        horizon,
        obstacle_config["trajectory"]["dt"],
        forecast_stride_steps=config["prediction"][
            "forecast_stride_steps"
        ],
        warmup_steps=int(
            round(
                config["prediction"]["warmup_s"]
                / obstacle_config["trajectory"]["dt"]
            )
        ),
    )
    assert records
    assert any(not record.observation_available for record in records)
    assert all(
        tuple(record.future.model_names) == IMM_MODEL_NAMES
        for record in records
    )
