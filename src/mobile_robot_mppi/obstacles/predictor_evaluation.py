"""Offline-only evaluation for deterministic, Gaussian, and IMM forecasts."""

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


_CHI2_POSITION = {
    "coverage_50": 1.3862943611198906,
    "coverage_90": 4.605170185988092,
    "coverage_95": 5.991464547107979,
}


def _gaussian_logpdf(position, mean, covariance):
    covariance = (
        np.asarray(covariance, dtype=np.float64)
        + 1.0e-10 * np.eye(2, dtype=np.float64)
    )
    error = np.asarray(position, dtype=np.float64) - np.asarray(
        mean, dtype=np.float64
    )
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise FloatingPointError("position covariance is not positive definite")
    mahalanobis = float(error.dot(np.linalg.solve(covariance, error)))
    logpdf = -0.5 * (
        2.0 * np.log(2.0 * np.pi) + log_determinant + mahalanobis
    )
    return float(logpdf), mahalanobis


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max())
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def _event_window(timestamp, event_times):
    candidates = []
    for event_time in event_times:
        delta = float(timestamp - event_time)
        if -2.0 <= delta < 0.0:
            candidates.append((abs(delta), "pre_change"))
        elif 0.0 <= delta < 1.0:
            candidates.append((abs(delta), "post_change_0_1"))
        elif 1.0 <= delta < 3.0:
            candidates.append((abs(delta), "post_change_1_3"))
    return min(candidates)[1] if candidates else "steady"


def _new_accumulator():
    return {
        "position_errors": [],
        "final_errors": [],
        "nll": [],
        "mahalanobis": [],
        "areas_95": [],
        "mixture_spread": [],
        "forecast_origins": 0,
        "future_points": 0,
    }


def _finalize(accumulator):
    errors = accumulator["position_errors"]
    if not errors:
        return {
            "forecast_origins": 0,
            "future_points": 0,
            "ade_m": None,
            "fde_3s_m": None,
            "nll": None,
            "coverage_50": None,
            "coverage_90": None,
            "coverage_95": None,
            "mean_95_area_m2": None,
            "mean_mixture_spread_m": None,
        }
    result = {
        "forecast_origins": int(accumulator["forecast_origins"]),
        "future_points": int(accumulator["future_points"]),
        "ade_m": float(np.mean(errors)),
        "fde_3s_m": float(np.mean(accumulator["final_errors"])),
        "nll": None,
        "coverage_50": None,
        "coverage_90": None,
        "coverage_95": None,
        "mean_95_area_m2": None,
        "mean_mixture_spread_m": None,
    }
    if accumulator["nll"]:
        result["nll"] = float(np.mean(accumulator["nll"]))
        mahalanobis = np.asarray(accumulator["mahalanobis"])
        for name, threshold in _CHI2_POSITION.items():
            result[name] = float(np.mean(mahalanobis <= threshold))
        result["mean_95_area_m2"] = float(
            np.mean(accumulator["areas_95"])
        )
    if accumulator["mixture_spread"]:
        result["mean_mixture_spread_m"] = float(
            np.mean(accumulator["mixture_spread"])
        )
    return result


def _truth_motion_mode(states, index, dt):
    velocity = states[index, 2:]
    speed = float(np.linalg.norm(velocity))
    if index <= 0:
        return "cv"
    previous_velocity = states[index - 1, 2:]
    previous_speed = float(np.linalg.norm(previous_velocity))
    longitudinal_acceleration = (speed - previous_speed) / dt
    if speed < 0.12 or longitudinal_acceleration < -0.25:
        return "brake_stop"
    if previous_speed < 0.15:
        return "cv"
    previous_heading = float(
        np.arctan2(previous_velocity[1], previous_velocity[0])
    )
    heading = float(np.arctan2(velocity[1], velocity[0]))
    yaw_rate = float(
        np.arctan2(
            np.sin(heading - previous_heading),
            np.cos(heading - previous_heading),
        )
        / dt
    )
    if yaw_rate > 0.15:
        return "turn_left"
    if yaw_rate < -0.15:
        return "turn_right"
    return "cv"


def evaluate_forecasts(
    forecasts: Sequence[object],
    truth_states,
    dt,
    events: Sequence[Mapping[str, object]] = (),
):
    """Evaluate already-produced forecasts using truth only offline."""
    truth_states = np.asarray(truth_states, dtype=np.float64)
    if truth_states.ndim != 2 or truth_states.shape[1] != 4:
        raise ValueError("truth states must have shape (T, 4)")
    event_times = [
        float(event["time_s"])
        for event in events
        if int(event["step"]) > 0
    ]
    buckets = defaultdict(_new_accumulator)
    mode_probability_sums = defaultdict(
        lambda: np.zeros(4, dtype=np.float64)
    )
    mode_counts = defaultdict(int)
    mode_top1_matches = defaultdict(int)
    used_records = 0
    for record in forecasts:
        horizon = int(record.future.means.shape[0])
        if record.time_index + horizon >= truth_states.shape[0]:
            continue
        used_records += 1
        labels = (
            "overall",
            _event_window(record.timestamp, event_times),
            (
                "observation_available"
                if record.observation_available
                else "observation_missing"
            ),
        )
        truth_future = truth_states[
            record.time_index + 1 : record.time_index + 1 + horizon, :2
        ]
        errors = record.future.means[:, :2] - truth_future
        distances = np.linalg.norm(errors, axis=1)
        mixture = hasattr(record.future, "component_means")
        gaussian = record.future.covariances is not None
        for label in labels:
            accumulator = buckets[label]
            accumulator["forecast_origins"] += 1
            accumulator["future_points"] += horizon
            accumulator["position_errors"].extend(distances.tolist())
            accumulator["final_errors"].append(float(distances[-1]))
        if gaussian:
            for horizon_index in range(horizon):
                truth_position = truth_future[horizon_index]
                covariance = record.future.covariances[
                    horizon_index, :2, :2
                ]
                _, mahalanobis = _gaussian_logpdf(
                    truth_position,
                    record.future.means[horizon_index, :2],
                    covariance,
                )
                if mixture:
                    terms = []
                    for mode in range(
                        record.future.component_means.shape[1]
                    ):
                        logpdf, _ = _gaussian_logpdf(
                            truth_position,
                            record.future.component_means[
                                horizon_index, mode, :2
                            ],
                            record.future.component_covariances[
                                horizon_index, mode, :2, :2
                            ],
                        )
                        weight = record.future.mode_probabilities[
                            horizon_index, mode
                        ]
                        terms.append(np.log(max(float(weight), 1.0e-300)) + logpdf)
                    nll = -_logsumexp(terms)
                    offsets = (
                        record.future.component_means[
                            horizon_index, :, :2
                        ]
                        - record.future.means[horizon_index, :2]
                    )
                    squared = np.sum(offsets ** 2, axis=1)
                    spread = float(
                        np.sqrt(
                            np.sum(
                                record.future.mode_probabilities[
                                    horizon_index
                                ]
                                * squared
                            )
                        )
                    )
                else:
                    logpdf, _ = _gaussian_logpdf(
                        truth_position,
                        record.future.means[horizon_index, :2],
                        covariance,
                    )
                    nll = -logpdf
                    spread = None
                area = float(
                    np.pi
                    * _CHI2_POSITION["coverage_95"]
                    * np.sqrt(max(0.0, np.linalg.det(covariance)))
                )
                for label in labels:
                    accumulator = buckets[label]
                    accumulator["nll"].append(float(nll))
                    accumulator["mahalanobis"].append(mahalanobis)
                    accumulator["areas_95"].append(area)
                    if spread is not None:
                        accumulator["mixture_spread"].append(spread)
        if hasattr(record, "filtered_mode_probabilities"):
            truth_mode = _truth_motion_mode(
                truth_states, int(record.time_index), float(dt)
            )
            probabilities = np.asarray(
                record.filtered_mode_probabilities, dtype=np.float64
            )
            mode_probability_sums[truth_mode] += probabilities
            mode_counts[truth_mode] += 1
            mode_top1_matches[truth_mode] += int(
                int(np.argmax(probabilities))
                == ("cv", "turn_left", "turn_right", "brake_stop").index(
                    truth_mode
                )
            )
    if used_records == 0:
        raise ValueError("no full-horizon forecast matches future truth")
    result = {
        "full_horizon_forecast_origins": used_records,
        "overall": _finalize(buckets["overall"]),
        "event_windows": {
            name: _finalize(buckets[name])
            for name in (
                "pre_change",
                "post_change_0_1",
                "post_change_1_3",
                "steady",
            )
        },
        "observation_status": {
            name: _finalize(buckets[name])
            for name in ("observation_available", "observation_missing")
        },
        "mode_probability_by_inferred_truth": {},
    }
    for truth_mode in ("cv", "turn_left", "turn_right", "brake_stop"):
        count = mode_counts[truth_mode]
        result["mode_probability_by_inferred_truth"][truth_mode] = {
            "count": int(count),
            "mean_mode_probabilities": (
                None
                if count == 0
                else (
                    mode_probability_sums[truth_mode] / count
                ).tolist()
            ),
            "top1_agreement": (
                None
                if count == 0
                else float(mode_top1_matches[truth_mode] / count)
            ),
        }
    return result


__all__ = ["evaluate_forecasts"]
