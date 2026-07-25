import numpy as np

from mobile_robot_mppi.obstacles.motion import (
    NOISE_PROFILES,
    generate_obstacle_trajectory,
)
from mobile_robot_mppi.obstacles.prediction import (
    GaussianCVKalmanPredictor,
    run_online_forecasts,
)


def _forecast_means(trajectory):
    predictor = GaussianCVKalmanPredictor(
        process_acceleration_std=0.075,
        observation_std=0.075,
    )
    records = run_online_forecasts(
        trajectory.times,
        trajectory.observations,
        trajectory.observed_mask,
        predictor,
        horizon=10,
        forecast_dt=0.1,
    )
    return np.stack([record.future.means for record in records])


def test_online_predictions_do_not_depend_on_truth_or_change_labels():
    trajectory = generate_obstacle_trajectory(
        "combined_change", 730100001, NOISE_PROFILES["medium"]
    )
    baseline = _forecast_means(trajectory)
    # The online runner receives only time, observation, and availability.
    # Corrupting offline labels must therefore have no effect on forecasts.
    object.__setattr__(
        trajectory,
        "states",
        np.full_like(trajectory.states, 12345.0),
    )
    object.__setattr__(
        trajectory,
        "change_flags",
        ~trajectory.change_flags,
    )
    object.__setattr__(
        trajectory,
        "true_modes",
        tuple("forbidden_truth" for _ in trajectory.true_modes),
    )
    repeated = _forecast_means(trajectory)
    assert np.array_equal(baseline, repeated)

