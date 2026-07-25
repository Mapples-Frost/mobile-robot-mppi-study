import numpy as np

from mobile_robot_mppi.obstacles.imm import (
    OrdinaryIMMPredictor,
    run_online_imm_forecasts,
)
from mobile_robot_mppi.obstacles.prediction import (
    DeterministicCVPredictor,
    GaussianCVKalmanPredictor,
    run_online_forecasts,
)
from mobile_robot_mppi.obstacles.predictor_evaluation import (
    evaluate_forecasts,
)


def _truth_and_observations():
    times = np.arange(101, dtype=np.float64) * 0.05
    states = np.zeros((101, 4), dtype=np.float64)
    states[:, 0] = 0.6 * times
    states[:, 1] = -0.2 * times
    states[:, 2] = 0.6
    states[:, 3] = -0.2
    observations = states[:, :2].copy()
    mask = np.ones(101, dtype=bool)
    mask[40:48] = False
    observations[~mask] = np.nan
    return times, states, observations, mask


def _imm():
    return OrdinaryIMMPredictor(
        observation_std=0.05,
        initial_velocity_std=1.0,
        initial_mode_probabilities=(0.55, 0.15, 0.15, 0.15),
        transition_matrix=(
            (0.94, 0.02, 0.02, 0.02),
            (0.04, 0.92, 0.01, 0.03),
            (0.04, 0.01, 0.92, 0.03),
            (0.05, 0.02, 0.02, 0.91),
        ),
        turn_rate_radps=0.65,
        brake_decay_rate_per_s=1.8,
        process_acceleration_std={
            "cv": 0.10,
            "turn_left": 0.14,
            "turn_right": 0.14,
            "brake_stop": 0.18,
        },
    )


def test_offline_evaluator_handles_all_three_predictor_types():
    times, truth, observations, mask = _truth_and_observations()
    deterministic = run_online_forecasts(
        times,
        observations,
        mask,
        DeterministicCVPredictor(),
        horizon=10,
        forecast_dt=0.05,
        forecast_stride_steps=2,
        warmup_steps=4,
    )
    gaussian = run_online_forecasts(
        times,
        observations,
        mask,
        GaussianCVKalmanPredictor(0.18, 0.05, 1.0),
        horizon=10,
        forecast_dt=0.05,
        forecast_stride_steps=2,
        warmup_steps=4,
    )
    imm = run_online_imm_forecasts(
        times,
        observations,
        mask,
        _imm(),
        horizon=10,
        forecast_dt=0.05,
        forecast_stride_steps=2,
        warmup_steps=4,
    )
    event = ({"step": 50, "time_s": 2.5, "kind": "test"},)
    deterministic_metrics = evaluate_forecasts(
        deterministic, truth, 0.05, event
    )
    gaussian_metrics = evaluate_forecasts(gaussian, truth, 0.05, event)
    imm_metrics = evaluate_forecasts(imm, truth, 0.05, event)
    assert deterministic_metrics["overall"]["nll"] is None
    assert gaussian_metrics["overall"]["nll"] is not None
    assert imm_metrics["overall"]["nll"] is not None
    assert imm_metrics["overall"]["mean_mixture_spread_m"] > 0.0
    assert (
        imm_metrics["event_windows"]["post_change_0_1"][
            "forecast_origins"
        ]
        > 0
    )
    assert (
        gaussian_metrics["observation_status"]["observation_missing"][
            "forecast_origins"
        ]
        > 0
    )
