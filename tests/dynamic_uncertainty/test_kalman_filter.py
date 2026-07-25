import numpy as np
import pytest

from mobile_robot_mppi.obstacles.prediction import (
    GaussianCVKalmanPredictor,
    cv_process_covariance,
    cv_transition,
)


def test_cv_transition_matches_constant_velocity_kinematics():
    state = np.asarray((1.0, -2.0, 0.5, -0.25))
    predicted = cv_transition(0.2).dot(state)
    assert predicted == pytest.approx((1.1, -2.05, 0.5, -0.25))


def test_kalman_filter_tracks_finite_constant_velocity_observations():
    predictor = GaussianCVKalmanPredictor(
        process_acceleration_std=0.05,
        observation_std=0.02,
    )
    for index in range(20):
        time = 0.1 * index
        observation = np.asarray((0.6 * time, -0.2 * time))
        predictor.update(observation, time)
    forecast = predictor.forecast(horizon=8, dt=0.1)
    forecast.validate()
    assert forecast.means[-1, 0] == pytest.approx(0.6 * 2.7, abs=0.08)
    assert forecast.means[-1, 1] == pytest.approx(-0.2 * 2.7, abs=0.08)


def test_missing_observation_performs_prediction_without_update():
    predictor = GaussianCVKalmanPredictor(0.1, 0.05)
    predictor.update(np.asarray((0.0, 0.0)), 0.0)
    predictor.update(np.asarray((0.1, 0.0)), 0.1)
    before = np.trace(predictor.covariance)
    assert not predictor.update(None, 0.2)
    after = np.trace(predictor.covariance)
    assert after > before
    assert np.isfinite(predictor.state).all()


def test_process_covariance_is_symmetric_psd():
    covariance = cv_process_covariance(0.1, 0.2)
    assert np.allclose(covariance, covariance.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1.0e-12

