"""Deterministic CV and Gaussian CV Kalman prediction baselines."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


_H = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
)

_CHI2_POSITION = {
    "coverage_50": 1.3862943611198906,
    "coverage_90": 4.605170185988092,
    "coverage_95": 5.991464547107979,
}


def cv_transition(dt: float) -> np.ndarray:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return np.asarray(
        (
            (1.0, 0.0, dt, 0.0),
            (0.0, 1.0, 0.0, dt),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def cv_process_covariance(dt: float, acceleration_std: float) -> np.ndarray:
    if not np.isfinite(acceleration_std) or acceleration_std < 0.0:
        raise ValueError("acceleration_std must be finite and non-negative")
    cv_transition(dt)
    block = np.asarray(
        (
            (0.25 * dt ** 4, 0.5 * dt ** 3),
            (0.5 * dt ** 3, dt ** 2),
        ),
        dtype=np.float64,
    )
    covariance = np.zeros((4, 4), dtype=np.float64)
    covariance[np.ix_((0, 2), (0, 2))] = block
    covariance[np.ix_((1, 3), (1, 3))] = block
    return (float(acceleration_std) ** 2) * covariance


def _symmetrize_psd(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(np.min(eigenvalues)) < -1.0e-8:
        raise FloatingPointError("covariance lost positive semidefiniteness")
    clipped = np.maximum(eigenvalues, 0.0)
    result = (eigenvectors * clipped).dot(eigenvectors.T)
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class Prediction:
    means: np.ndarray
    covariances: Optional[np.ndarray]

    def validate(self) -> None:
        if self.means.ndim != 2 or self.means.shape[1] != 4:
            raise ValueError("prediction means must have shape (H, 4)")
        if not np.isfinite(self.means).all():
            raise FloatingPointError("prediction means contain NaN or Inf")
        if self.covariances is None:
            return
        expected = (self.means.shape[0], 4, 4)
        if self.covariances.shape != expected:
            raise ValueError("prediction covariance shape is invalid")
        for covariance in self.covariances:
            if not np.isfinite(covariance).all():
                raise FloatingPointError("prediction covariance contains NaN or Inf")
            if not np.allclose(covariance, covariance.T, atol=1.0e-10, rtol=0.0):
                raise FloatingPointError("prediction covariance is not symmetric")
            if float(np.linalg.eigvalsh(covariance).min()) < -1.0e-9:
                raise FloatingPointError("prediction covariance is not PSD")


@dataclass(frozen=True)
class OnlineForecast:
    time_index: int
    timestamp: float
    observation_available: bool
    filtered_mean: np.ndarray
    filtered_covariance: Optional[np.ndarray]
    future: Prediction


class DeterministicCVPredictor:
    """Online finite-difference CV estimate with no uncertainty output."""

    name = "deterministic_cv"

    def __init__(self) -> None:
        self.state: Optional[np.ndarray] = None
        self.timestamp: Optional[float] = None
        self.last_observation: Optional[np.ndarray] = None
        self.last_observation_time: Optional[float] = None

    def reset(self) -> None:
        self.__init__()

    @property
    def covariance(self) -> None:
        return None

    def update(
        self, observation: Optional[np.ndarray], timestamp: float
    ) -> bool:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.timestamp is not None:
            elapsed = timestamp - float(self.timestamp)
            if elapsed <= 0.0:
                raise ValueError("predictor timestamps must increase strictly")
            self.state[:2] += elapsed * self.state[2:]
        if observation is not None:
            observation = np.asarray(observation, dtype=np.float64).reshape(-1)
            if observation.shape != (2,) or not np.isfinite(observation).all():
                raise ValueError("available observation must be a finite 2-vector")
            velocity = np.zeros(2, dtype=np.float64)
            if self.last_observation is not None:
                observation_dt = timestamp - float(self.last_observation_time)
                if observation_dt <= 0.0:
                    raise ValueError("observation timestamps must increase")
                velocity = (observation - self.last_observation) / observation_dt
            elif self.state is not None:
                velocity = self.state[2:].copy()
            self.state = np.concatenate((observation, velocity))
            self.last_observation = observation.copy()
            self.last_observation_time = timestamp
            observed = True
        else:
            observed = False
        self.timestamp = timestamp
        return observed

    def forecast(self, horizon: int, dt: float) -> Prediction:
        if self.state is None:
            raise RuntimeError("deterministic CV requires an observation first")
        if int(horizon) <= 0:
            raise ValueError("forecast horizon must be positive")
        transition = cv_transition(dt)
        state = self.state.copy()
        means = np.empty((int(horizon), 4), dtype=np.float64)
        for index in range(int(horizon)):
            state = transition.dot(state)
            means[index] = state
        result = Prediction(means=means, covariances=None)
        result.validate()
        return result


class GaussianCVKalmanPredictor:
    """Linear Gaussian CV filter over ``[px, py, vx, vy]``."""

    name = "gaussian_cv_kalman"

    def __init__(
        self,
        process_acceleration_std: float,
        observation_std: float,
        initial_velocity_std: float = 1.0,
    ) -> None:
        values = (
            process_acceleration_std,
            observation_std,
            initial_velocity_std,
        )
        if not np.isfinite(values).all() or any(value < 0.0 for value in values):
            raise ValueError("Kalman standard deviations must be finite and non-negative")
        if observation_std <= 0.0 or initial_velocity_std <= 0.0:
            raise ValueError("observation and initial velocity std must be positive")
        self.process_acceleration_std = float(process_acceleration_std)
        self.observation_std = float(observation_std)
        self.initial_velocity_std = float(initial_velocity_std)
        self.state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self.timestamp: Optional[float] = None

    @property
    def covariance(self) -> Optional[np.ndarray]:
        return None if self._covariance is None else self._covariance.copy()

    def reset(self) -> None:
        self.state = None
        self._covariance = None
        self.timestamp = None

    def _predict(self, elapsed: float) -> None:
        transition = cv_transition(elapsed)
        process_covariance = cv_process_covariance(
            elapsed, self.process_acceleration_std
        )
        self.state = transition.dot(self.state)
        self._covariance = _symmetrize_psd(
            transition.dot(self._covariance).dot(transition.T)
            + process_covariance
        )

    def update(
        self, observation: Optional[np.ndarray], timestamp: float
    ) -> bool:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.timestamp is not None:
            elapsed = timestamp - float(self.timestamp)
            if elapsed <= 0.0:
                raise ValueError("predictor timestamps must increase strictly")
            self._predict(elapsed)
        if observation is None:
            self.timestamp = timestamp
            return False
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        if observation.shape != (2,) or not np.isfinite(observation).all():
            raise ValueError("available observation must be a finite 2-vector")
        if self.state is None:
            self.state = np.asarray(
                (observation[0], observation[1], 0.0, 0.0), dtype=np.float64
            )
            position_variance = self.observation_std ** 2
            velocity_variance = self.initial_velocity_std ** 2
            self._covariance = np.diag(
                (
                    position_variance,
                    position_variance,
                    velocity_variance,
                    velocity_variance,
                )
            )
        else:
            measurement_covariance = (
                self.observation_std ** 2
            ) * np.eye(2, dtype=np.float64)
            innovation = observation - _H.dot(self.state)
            innovation_covariance = (
                _H.dot(self._covariance).dot(_H.T) + measurement_covariance
            )
            gain = np.linalg.solve(
                innovation_covariance,
                _H.dot(self._covariance),
            ).T
            self.state = self.state + gain.dot(innovation)
            identity = np.eye(4, dtype=np.float64)
            update_matrix = identity - gain.dot(_H)
            # Joseph form is numerically safer than (I-KH)P.
            self._covariance = _symmetrize_psd(
                update_matrix.dot(self._covariance).dot(update_matrix.T)
                + gain.dot(measurement_covariance).dot(gain.T)
            )
        self.timestamp = timestamp
        if not np.isfinite(self.state).all():
            raise FloatingPointError("Kalman state contains NaN or Inf")
        return True

    def forecast(self, horizon: int, dt: float) -> Prediction:
        if self.state is None or self._covariance is None:
            raise RuntimeError("Gaussian CV requires an observation first")
        if int(horizon) <= 0:
            raise ValueError("forecast horizon must be positive")
        transition = cv_transition(dt)
        process_covariance = cv_process_covariance(
            dt, self.process_acceleration_std
        )
        state = self.state.copy()
        covariance = self._covariance.copy()
        means = np.empty((int(horizon), 4), dtype=np.float64)
        covariances = np.empty((int(horizon), 4, 4), dtype=np.float64)
        for index in range(int(horizon)):
            state = transition.dot(state)
            covariance = _symmetrize_psd(
                transition.dot(covariance).dot(transition.T)
                + process_covariance
            )
            means[index] = state
            covariances[index] = covariance
        result = Prediction(means=means, covariances=covariances)
        result.validate()
        return result


def run_online_forecasts(
    times: np.ndarray,
    observations: np.ndarray,
    observed_mask: np.ndarray,
    predictor,
    horizon: int,
    forecast_dt: float,
    forecast_stride_steps: int = 1,
    warmup_steps: int = 0,
) -> List[OnlineForecast]:
    """Run a predictor using observations only.

    Ground-truth state, mode, and change labels are intentionally absent from
    this function signature. Evaluation code receives truth only after these
    forecasts have been produced.
    """

    times = np.asarray(times, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    count = times.shape[0]
    if (
        observations.shape != (count, 2)
        or observed_mask.shape != (count,)
        or count < 2
    ):
        raise ValueError("online forecast inputs have inconsistent shapes")
    stride = int(forecast_stride_steps)
    warmup = int(warmup_steps)
    if stride <= 0 or warmup < 0:
        raise ValueError("online forecast stride/warmup is invalid")
    records: List[OnlineForecast] = []
    for index in range(count):
        observation = observations[index] if observed_mask[index] else None
        predictor.update(observation, float(times[index]))
        if (
            predictor.state is None
            or index < warmup
            or (index - warmup) % stride != 0
        ):
            continue
        future = predictor.forecast(int(horizon), float(forecast_dt))
        covariance = getattr(predictor, "covariance", None)
        records.append(
            OnlineForecast(
                time_index=index,
                timestamp=float(times[index]),
                observation_available=bool(observed_mask[index]),
                filtered_mean=np.asarray(predictor.state, dtype=np.float64).copy(),
                filtered_covariance=(
                    None
                    if covariance is None
                    else np.asarray(covariance, dtype=np.float64).copy()
                ),
                future=future,
            )
        )
    return records


def prediction_metrics(
    forecasts: Sequence[OnlineForecast], truth_states: np.ndarray
) -> Dict[str, Optional[float]]:
    truth_states = np.asarray(truth_states, dtype=np.float64)
    if truth_states.ndim != 2 or truth_states.shape[1] != 4:
        raise ValueError("truth states must have shape (T, 4)")
    position_errors: List[float] = []
    final_errors: List[float] = []
    nll_values: List[float] = []
    mahalanobis_values: List[float] = []
    ellipse_areas: List[float] = []
    gaussian = bool(forecasts and forecasts[0].future.covariances is not None)
    for record in forecasts:
        available = min(
            record.future.means.shape[0],
            truth_states.shape[0] - record.time_index - 1,
        )
        if available <= 0:
            continue
        errors = (
            record.future.means[:available, :2]
            - truth_states[
                record.time_index + 1 : record.time_index + 1 + available,
                :2,
            ]
        )
        distances = np.linalg.norm(errors, axis=1)
        position_errors.extend(float(value) for value in distances)
        final_errors.append(float(distances[-1]))
        if gaussian:
            for horizon_index in range(available):
                covariance = record.future.covariances[horizon_index, :2, :2]
                covariance = covariance + 1.0e-12 * np.eye(2)
                error = errors[horizon_index]
                sign, log_determinant = np.linalg.slogdet(covariance)
                if sign <= 0.0:
                    raise FloatingPointError("position covariance is not positive definite")
                mahalanobis = float(error.dot(np.linalg.solve(covariance, error)))
                nll_values.append(
                    0.5
                    * (
                        2.0 * np.log(2.0 * np.pi)
                        + log_determinant
                        + mahalanobis
                    )
                )
                mahalanobis_values.append(mahalanobis)
                ellipse_areas.append(
                    float(
                        np.pi
                        * _CHI2_POSITION["coverage_95"]
                        * np.sqrt(np.linalg.det(covariance))
                    )
                )
    if not position_errors:
        raise ValueError("no forecast has a matching future truth state")
    metrics: Dict[str, Optional[float]] = {
        "ade_m": float(np.mean(position_errors)),
        "fde_m": float(np.mean(final_errors)),
        "nll": None,
        "coverage_50": None,
        "coverage_90": None,
        "coverage_95": None,
        "mean_95_area_m2": None,
    }
    if gaussian:
        metrics["nll"] = float(np.mean(nll_values))
        for name, threshold in _CHI2_POSITION.items():
            metrics[name] = float(
                np.mean(np.asarray(mahalanobis_values) <= threshold)
            )
        metrics["mean_95_area_m2"] = float(np.mean(ellipse_areas))
    return metrics


__all__ = [
    "DeterministicCVPredictor",
    "GaussianCVKalmanPredictor",
    "OnlineForecast",
    "Prediction",
    "cv_process_covariance",
    "cv_transition",
    "prediction_metrics",
    "run_online_forecasts",
]
