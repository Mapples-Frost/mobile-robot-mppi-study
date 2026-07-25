import numpy as np

from mobile_robot_mppi.obstacles.artifacts import covariance_audit
from mobile_robot_mppi.obstacles.motion import (
    NOISE_PROFILES,
    generate_obstacle_trajectory,
)
from mobile_robot_mppi.obstacles.prediction import (
    GaussianCVKalmanPredictor,
    run_online_forecasts,
)


def test_all_online_and_future_covariances_remain_psd_through_occlusion():
    trajectory = generate_obstacle_trajectory(
        "combined_change", 730100005, NOISE_PROFILES["medium"], steps=100
    )
    predictor = GaussianCVKalmanPredictor(
        process_acceleration_std=NOISE_PROFILES["medium"].acceleration_std,
        observation_std=NOISE_PROFILES["medium"].observation_std,
    )
    forecasts = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        predictor,
        horizon=20,
        forecast_dt=0.1,
    )
    audit = covariance_audit(forecasts)
    assert audit["finite"]
    assert audit["symmetric"]
    assert audit["positive_semidefinite"]
    assert audit["covariance_count"] == len(forecasts) * 20
    assert audit["minimum_eigenvalue"] >= -1.0e-9

