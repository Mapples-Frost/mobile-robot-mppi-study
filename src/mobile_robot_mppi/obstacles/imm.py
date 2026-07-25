"""Ordinary and change-aware four-mode interacting multiple-model predictors."""

from collections import deque
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from .prediction import cv_process_covariance, cv_transition


IMM_MODEL_NAMES = ("cv", "turn_left", "turn_right", "brake_stop")
_H = np.asarray(
    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
    dtype=np.float64,
)


def _symmetrize_psd(covariance):
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues.min()) < -1.0e-7:
        raise FloatingPointError("IMM covariance lost positive semidefiniteness")
    eigenvalues = np.maximum(eigenvalues, 1.0e-12)
    result = (eigenvectors * eigenvalues).dot(eigenvectors.T)
    return 0.5 * (result + result.T)


def _symmetrize_fast(covariance):
    covariance = np.asarray(covariance, dtype=np.float64)
    return (
        0.5 * (covariance + covariance.T)
        + 1.0e-12 * np.eye(covariance.shape[0], dtype=np.float64)
    )


def turn_transition(dt, yaw_rate):
    dt = float(dt)
    yaw_rate = float(yaw_rate)
    if not np.isfinite((dt, yaw_rate)).all() or dt <= 0.0:
        raise ValueError("turn transition inputs must be finite and dt positive")
    if abs(yaw_rate) < 1.0e-10:
        return cv_transition(dt)
    angle = yaw_rate * dt
    sine = float(np.sin(angle))
    cosine = float(np.cos(angle))
    return np.asarray(
        (
            (1.0, 0.0, sine / yaw_rate, -(1.0 - cosine) / yaw_rate),
            (0.0, 1.0, (1.0 - cosine) / yaw_rate, sine / yaw_rate),
            (0.0, 0.0, cosine, -sine),
            (0.0, 0.0, sine, cosine),
        ),
        dtype=np.float64,
    )


def brake_transition(dt, decay_rate):
    dt = float(dt)
    decay_rate = float(decay_rate)
    if not np.isfinite((dt, decay_rate)).all() or dt <= 0.0:
        raise ValueError("brake transition inputs must be finite and dt positive")
    if decay_rate <= 0.0:
        raise ValueError("brake decay rate must be positive")
    decay = float(np.exp(-decay_rate * dt))
    position_gain = (1.0 - decay) / decay_rate
    return np.asarray(
        (
            (1.0, 0.0, position_gain, 0.0),
            (0.0, 1.0, 0.0, position_gain),
            (0.0, 0.0, decay, 0.0),
            (0.0, 0.0, 0.0, decay),
        ),
        dtype=np.float64,
    )


def _mixture_moments(
    component_means, component_covariances, weights, strict=True
):
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.sum(weights[:, None] * component_means, axis=0)
    covariance = np.zeros((4, 4), dtype=np.float64)
    for weight, component_mean, component_covariance in zip(
        weights, component_means, component_covariances
    ):
        offset = component_mean - mean
        covariance += float(weight) * (
            component_covariance + np.outer(offset, offset)
        )
    return mean, (
        _symmetrize_psd(covariance)
        if strict
        else _symmetrize_fast(covariance)
    )


@dataclass(frozen=True)
class MixturePrediction:
    means: np.ndarray
    covariances: np.ndarray
    component_means: np.ndarray
    component_covariances: np.ndarray
    mode_probabilities: np.ndarray
    model_names: Sequence[str] = IMM_MODEL_NAMES

    def validate(self, check_psd=True):
        horizon = self.means.shape[0]
        modes = len(self.model_names)
        if self.means.shape != (horizon, 4):
            raise ValueError("IMM mixture means must have shape (H, 4)")
        if self.covariances.shape != (horizon, 4, 4):
            raise ValueError("IMM mixture covariance shape is invalid")
        if self.component_means.shape != (horizon, modes, 4):
            raise ValueError("IMM component mean shape is invalid")
        if self.component_covariances.shape != (horizon, modes, 4, 4):
            raise ValueError("IMM component covariance shape is invalid")
        if self.mode_probabilities.shape != (horizon, modes):
            raise ValueError("IMM mode probability shape is invalid")
        arrays = (
            self.means,
            self.covariances,
            self.component_means,
            self.component_covariances,
            self.mode_probabilities,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise FloatingPointError("IMM prediction contains NaN or Inf")
        if np.any(self.mode_probabilities < -1.0e-12):
            raise FloatingPointError("IMM mode probability is negative")
        if not np.allclose(
            self.mode_probabilities.sum(axis=1),
            1.0,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise FloatingPointError("IMM mode probabilities do not sum to one")
        for covariance in np.concatenate(
            (
                self.covariances[:, None, :, :],
                self.component_covariances,
            ),
            axis=1,
        ).reshape(-1, 4, 4):
            if not np.allclose(
                covariance, covariance.T, atol=1.0e-10, rtol=0.0
            ):
                raise FloatingPointError("IMM covariance is asymmetric")
            if (
                check_psd
                and float(np.linalg.eigvalsh(covariance).min()) < -1.0e-9
            ):
                raise FloatingPointError("IMM covariance is not PSD")


@dataclass(frozen=True)
class OnlineIMMForecast:
    time_index: int
    timestamp: float
    observation_available: bool
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    filtered_mode_probabilities: np.ndarray
    future: MixturePrediction
    innovation_nis: Optional[float] = None
    change_triggered: bool = False
    dropout_guard_triggered: bool = False
    recovery_active: bool = False


class OrdinaryIMMPredictor:
    """Four-mode linear-Gaussian IMM using position measurements only."""

    name = "ordinary_imm"

    def __init__(
        self,
        observation_std,
        initial_velocity_std,
        initial_mode_probabilities,
        transition_matrix,
        turn_rate_radps,
        brake_decay_rate_per_s,
        process_acceleration_std: Mapping[str, float],
    ):
        self.observation_std = float(observation_std)
        self.initial_velocity_std = float(initial_velocity_std)
        self.initial_mode_probabilities = np.asarray(
            initial_mode_probabilities, dtype=np.float64
        )
        self.transition_matrix = np.asarray(
            transition_matrix, dtype=np.float64
        )
        self.turn_rate_radps = float(turn_rate_radps)
        self.brake_decay_rate_per_s = float(brake_decay_rate_per_s)
        self.process_acceleration_std = {
            str(name): float(value)
            for name, value in process_acceleration_std.items()
        }
        self._validate_config()
        self.reset()

    def _validate_config(self):
        mode_count = len(IMM_MODEL_NAMES)
        if self.observation_std <= 0.0 or self.initial_velocity_std <= 0.0:
            raise ValueError("IMM observation and initial velocity std must be positive")
        if self.initial_mode_probabilities.shape != (mode_count,):
            raise ValueError("IMM initial mode probability shape is invalid")
        if np.any(self.initial_mode_probabilities < 0.0) or not np.isclose(
            self.initial_mode_probabilities.sum(), 1.0
        ):
            raise ValueError("IMM initial mode probabilities are invalid")
        if self.transition_matrix.shape != (mode_count, mode_count):
            raise ValueError("IMM transition matrix shape is invalid")
        if np.any(self.transition_matrix < 0.0) or not np.allclose(
            self.transition_matrix.sum(axis=1), 1.0
        ):
            raise ValueError("IMM transition rows must be probabilities")
        if self.turn_rate_radps <= 0.0 or self.brake_decay_rate_per_s <= 0.0:
            raise ValueError("IMM turn and brake parameters must be positive")
        if tuple(self.process_acceleration_std.keys()) != IMM_MODEL_NAMES:
            raise ValueError("IMM process-noise map must follow model order")
        if any(value < 0.0 for value in self.process_acceleration_std.values()):
            raise ValueError("IMM process noise must be non-negative")

    def reset(self):
        self.component_states: Optional[np.ndarray] = None
        self.component_covariances: Optional[np.ndarray] = None
        self.mode_probabilities = self.initial_mode_probabilities.copy()
        self.timestamp: Optional[float] = None
        self.state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None

    @property
    def covariance(self):
        return (
            None
            if self._covariance is None
            else self._covariance.copy()
        )

    def _transition(self, model_name, dt):
        if model_name == "cv":
            return cv_transition(dt)
        if model_name == "turn_left":
            return turn_transition(dt, self.turn_rate_radps)
        if model_name == "turn_right":
            return turn_transition(dt, -self.turn_rate_radps)
        if model_name == "brake_stop":
            return brake_transition(dt, self.brake_decay_rate_per_s)
        raise RuntimeError("unknown IMM model")

    def _mix_predict(
        self,
        states,
        covariances,
        probabilities,
        dt,
        strict=True,
        process_noise_scale=1.0,
    ):
        process_noise_scale = float(process_noise_scale)
        if not np.isfinite(process_noise_scale) or process_noise_scale < 1.0:
            raise ValueError("IMM process-noise scale must be finite and >= 1")
        predicted_probabilities = probabilities.dot(self.transition_matrix)
        if np.any(predicted_probabilities <= 0.0):
            raise FloatingPointError("IMM predicted a zero-probability mode")
        mixing = (
            probabilities[:, None] * self.transition_matrix
        ) / predicted_probabilities[None, :]
        mode_count = len(IMM_MODEL_NAMES)
        predicted_states = np.empty_like(states)
        predicted_covariances = np.empty_like(covariances)
        for target in range(mode_count):
            mixed_state = np.sum(
                mixing[:, target, None] * states, axis=0
            )
            mixed_covariance = np.zeros((4, 4), dtype=np.float64)
            for source in range(mode_count):
                offset = states[source] - mixed_state
                mixed_covariance += mixing[source, target] * (
                    covariances[source] + np.outer(offset, offset)
                )
            transition = self._transition(IMM_MODEL_NAMES[target], dt)
            process_covariance = cv_process_covariance(
                dt,
                self.process_acceleration_std[IMM_MODEL_NAMES[target]],
            ) * process_noise_scale
            predicted_states[target] = transition.dot(mixed_state)
            raw_covariance = (
                transition.dot(mixed_covariance).dot(transition.T)
                + process_covariance
            )
            predicted_covariances[target] = (
                _symmetrize_psd(raw_covariance)
                if strict
                else _symmetrize_fast(raw_covariance)
            )
        return (
            predicted_states,
            predicted_covariances,
            predicted_probabilities,
        )

    def _set_moments(self):
        self.state, self._covariance = _mixture_moments(
            self.component_states,
            self.component_covariances,
            self.mode_probabilities,
        )

    def _filter_process_noise_scale(self, timestamp):
        return 1.0

    def _forecast_process_noise_scale(self, horizon_index, dt):
        return 1.0

    def _before_observation(self, observation, timestamp, elapsed):
        return None

    def _before_measurement_update(
        self, observation, timestamp, measurement_covariance
    ):
        return None

    def update(self, observation, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("IMM timestamp must be finite")
        if self.timestamp is not None:
            elapsed = timestamp - float(self.timestamp)
            if elapsed <= 0.0:
                raise ValueError("IMM timestamps must increase strictly")
            if self.component_states is not None:
                (
                    self.component_states,
                    self.component_covariances,
                    prior_probabilities,
                ) = self._mix_predict(
                    self.component_states,
                    self.component_covariances,
                    self.mode_probabilities,
                    elapsed,
                    process_noise_scale=self._filter_process_noise_scale(
                        timestamp
                    ),
                )
                self.mode_probabilities = prior_probabilities
        else:
            elapsed = None
        self._before_observation(observation, timestamp, elapsed)
        if observation is None:
            self.timestamp = timestamp
            if self.component_states is not None:
                self._set_moments()
            return False
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        if observation.shape != (2,) or not np.isfinite(observation).all():
            raise ValueError("IMM observation must be a finite 2-vector")
        if self.component_states is None:
            initial_state = np.asarray(
                (observation[0], observation[1], 0.0, 0.0),
                dtype=np.float64,
            )
            position_variance = self.observation_std ** 2
            velocity_variance = self.initial_velocity_std ** 2
            initial_covariance = np.diag(
                (
                    position_variance,
                    position_variance,
                    velocity_variance,
                    velocity_variance,
                )
            )
            self.component_states = np.repeat(
                initial_state[None, :], len(IMM_MODEL_NAMES), axis=0
            )
            self.component_covariances = np.repeat(
                initial_covariance[None, :, :],
                len(IMM_MODEL_NAMES),
                axis=0,
            )
            self.mode_probabilities = self.initial_mode_probabilities.copy()
        else:
            measurement_covariance = (
                self.observation_std ** 2
            ) * np.eye(2, dtype=np.float64)
            self._before_measurement_update(
                observation, timestamp, measurement_covariance
            )
            log_likelihoods = np.empty(len(IMM_MODEL_NAMES), dtype=np.float64)
            identity = np.eye(4, dtype=np.float64)
            for mode in range(len(IMM_MODEL_NAMES)):
                state = self.component_states[mode]
                covariance = self.component_covariances[mode]
                innovation = observation - _H.dot(state)
                innovation_covariance = (
                    _H.dot(covariance).dot(_H.T) + measurement_covariance
                )
                sign, log_determinant = np.linalg.slogdet(
                    innovation_covariance
                )
                if sign <= 0.0:
                    raise FloatingPointError("IMM innovation covariance invalid")
                mahalanobis = float(
                    innovation.dot(
                        np.linalg.solve(innovation_covariance, innovation)
                    )
                )
                log_likelihoods[mode] = -0.5 * (
                    2.0 * np.log(2.0 * np.pi)
                    + log_determinant
                    + mahalanobis
                )
                gain = np.linalg.solve(
                    innovation_covariance, _H.dot(covariance)
                ).T
                self.component_states[mode] = state + gain.dot(innovation)
                update_matrix = identity - gain.dot(_H)
                self.component_covariances[mode] = _symmetrize_psd(
                    update_matrix.dot(covariance).dot(update_matrix.T)
                    + gain.dot(measurement_covariance).dot(gain.T)
                )
            logits = np.log(self.mode_probabilities) + log_likelihoods
            logits -= float(logits.max())
            weights = np.exp(logits)
            self.mode_probabilities = weights / weights.sum()
        self.timestamp = timestamp
        self._set_moments()
        if not np.isfinite(self.state).all():
            raise FloatingPointError("IMM filtered state contains NaN or Inf")
        return True

    def forecast(self, horizon, dt):
        if self.component_states is None:
            raise RuntimeError("IMM requires an observation first")
        horizon = int(horizon)
        dt = float(dt)
        if horizon <= 0 or dt <= 0.0:
            raise ValueError("IMM forecast horizon and dt must be positive")
        states = self.component_states.copy()
        covariances = self.component_covariances.copy()
        probabilities = self.mode_probabilities.copy()
        component_means = np.empty(
            (horizon, len(IMM_MODEL_NAMES), 4), dtype=np.float64
        )
        component_covariances = np.empty(
            (horizon, len(IMM_MODEL_NAMES), 4, 4), dtype=np.float64
        )
        mode_probabilities = np.empty(
            (horizon, len(IMM_MODEL_NAMES)), dtype=np.float64
        )
        means = np.empty((horizon, 4), dtype=np.float64)
        mixture_covariances = np.empty((horizon, 4, 4), dtype=np.float64)
        for index in range(horizon):
            states, covariances, probabilities = self._mix_predict(
                states,
                covariances,
                probabilities,
                dt,
                strict=False,
                process_noise_scale=self._forecast_process_noise_scale(
                    index, dt
                ),
            )
            mean, covariance = _mixture_moments(
                states, covariances, probabilities, strict=False
            )
            component_means[index] = states
            component_covariances[index] = covariances
            mode_probabilities[index] = probabilities
            means[index] = mean
            mixture_covariances[index] = covariance
        result = MixturePrediction(
            means=means,
            covariances=mixture_covariances,
            component_means=component_means,
            component_covariances=component_covariances,
            mode_probabilities=mode_probabilities,
        )
        result.validate(check_psd=False)
        return result


class ChangeAwareIMMPredictor(OrdinaryIMMPredictor):
    """Ordinary IMM augmented with a causal NIS change detector."""

    name = "change_aware_imm"

    def __init__(
        self,
        observation_std,
        initial_velocity_std,
        initial_mode_probabilities,
        transition_matrix,
        turn_rate_radps,
        brake_decay_rate_per_s,
        process_acceleration_std: Mapping[str, float],
        nis_threshold,
        required_exceedances,
        window_observations,
        reset_mode_probabilities,
        state_covariance_inflation,
        recovery_process_noise_scale,
        recovery_duration_s,
        refractory_period_s,
        dropout_guard_after_s,
        dropout_covariance_inflation,
        single_exceedance_threshold=None,
    ):
        self.nis_threshold = float(nis_threshold)
        self.required_exceedances = int(required_exceedances)
        self.window_observations = int(window_observations)
        self.reset_mode_probabilities = np.asarray(
            reset_mode_probabilities, dtype=np.float64
        )
        self.state_covariance_inflation = float(
            state_covariance_inflation
        )
        self.recovery_process_noise_scale = float(
            recovery_process_noise_scale
        )
        self.recovery_duration_s = float(recovery_duration_s)
        self.refractory_period_s = float(refractory_period_s)
        self.dropout_guard_after_s = float(dropout_guard_after_s)
        self.dropout_covariance_inflation = float(
            dropout_covariance_inflation
        )
        self.single_exceedance_threshold = (
            None
            if single_exceedance_threshold is None
            else float(single_exceedance_threshold)
        )
        super().__init__(
            observation_std=observation_std,
            initial_velocity_std=initial_velocity_std,
            initial_mode_probabilities=initial_mode_probabilities,
            transition_matrix=transition_matrix,
            turn_rate_radps=turn_rate_radps,
            brake_decay_rate_per_s=brake_decay_rate_per_s,
            process_acceleration_std=process_acceleration_std,
        )

    def _validate_config(self):
        super()._validate_config()
        mode_count = len(IMM_MODEL_NAMES)
        if self.nis_threshold <= 0.0:
            raise ValueError("NIS threshold must be positive")
        if (
            self.single_exceedance_threshold is not None
            and (
                not np.isfinite(self.single_exceedance_threshold)
                or self.single_exceedance_threshold < self.nis_threshold
            )
        ):
            raise ValueError(
                "single-exceedance threshold must be finite and >= NIS threshold"
            )
        if (
            self.window_observations <= 0
            or self.required_exceedances <= 0
            or self.required_exceedances > self.window_observations
        ):
            raise ValueError("NIS persistence rule is invalid")
        if self.reset_mode_probabilities.shape != (mode_count,):
            raise ValueError("change-aware reset probability shape is invalid")
        if np.any(self.reset_mode_probabilities < 0.0) or not np.isclose(
            self.reset_mode_probabilities.sum(), 1.0
        ):
            raise ValueError("change-aware reset probabilities are invalid")
        positive = (
            self.state_covariance_inflation,
            self.recovery_process_noise_scale,
            self.recovery_duration_s,
            self.refractory_period_s,
            self.dropout_guard_after_s,
            self.dropout_covariance_inflation,
        )
        if not np.isfinite(positive).all() or any(
            value <= 0.0 for value in positive
        ):
            raise ValueError("change-aware response values must be positive")
        if (
            self.state_covariance_inflation < 1.0
            or self.recovery_process_noise_scale < 1.0
            or self.dropout_covariance_inflation < 1.0
        ):
            raise ValueError("change-aware inflation factors must be >= 1")

    def reset(self):
        super().reset()
        self.last_innovation_nis: Optional[float] = None
        self.last_change_triggered = False
        self.last_dropout_guard_triggered = False
        self.detection_events = []
        self._exceedance_history = deque(
            maxlen=self.window_observations
        )
        self._recovery_until = -np.inf
        self._refractory_until = -np.inf
        self._missing_duration_s = 0.0
        self._dropout_guard_active = False

    @property
    def recovery_active(self):
        return (
            self.timestamp is not None
            and float(self.timestamp) < float(self._recovery_until)
        )

    def _recovery_scale_at(self, timestamp):
        remaining = float(self._recovery_until) - float(timestamp)
        if remaining <= 0.0:
            return 1.0
        fraction = min(1.0, remaining / self.recovery_duration_s)
        return 1.0 + fraction * (
            self.recovery_process_noise_scale - 1.0
        )

    def _filter_process_noise_scale(self, timestamp):
        return self._recovery_scale_at(timestamp)

    def _forecast_process_noise_scale(self, horizon_index, dt):
        if self.timestamp is None:
            return 1.0
        future_time = float(self.timestamp) + (int(horizon_index) + 1) * float(
            dt
        )
        return self._recovery_scale_at(future_time)

    def _inflate_component_covariances(self, factor):
        if self.component_covariances is None:
            return
        self.component_covariances = np.asarray(
            [
                _symmetrize_psd(covariance * float(factor))
                for covariance in self.component_covariances
            ],
            dtype=np.float64,
        )

    def _activate_recovery(self, timestamp):
        self._recovery_until = max(
            float(self._recovery_until),
            float(timestamp) + self.recovery_duration_s,
        )

    def _before_observation(self, observation, timestamp, elapsed):
        self.last_innovation_nis = None
        self.last_change_triggered = False
        self.last_dropout_guard_triggered = False
        if observation is not None:
            self._missing_duration_s = 0.0
            self._dropout_guard_active = False
            return
        self._exceedance_history.clear()
        if elapsed is not None:
            self._missing_duration_s += float(elapsed)
        if (
            self.component_covariances is not None
            and not self._dropout_guard_active
            and self._missing_duration_s >= self.dropout_guard_after_s
        ):
            self._inflate_component_covariances(
                self.dropout_covariance_inflation
            )
            self._activate_recovery(timestamp)
            self._dropout_guard_active = True
            self.last_dropout_guard_triggered = True
            self.detection_events.append(
                {
                    "kind": "dropout_guard",
                    "timestamp": float(timestamp),
                    "nis": None,
                }
            )

    def _before_measurement_update(
        self, observation, timestamp, measurement_covariance
    ):
        mixture_state, mixture_covariance = _mixture_moments(
            self.component_states,
            self.component_covariances,
            self.mode_probabilities,
        )
        innovation = observation - _H.dot(mixture_state)
        innovation_covariance = (
            _H.dot(mixture_covariance).dot(_H.T)
            + measurement_covariance
        )
        nis = float(
            innovation.dot(
                np.linalg.solve(innovation_covariance, innovation)
            )
        )
        if not np.isfinite(nis) or nis < 0.0:
            raise FloatingPointError("change-aware NIS is invalid")
        self.last_innovation_nis = nis
        self._exceedance_history.append(nis >= self.nis_threshold)
        persistent = (
            len(self._exceedance_history) == self.window_observations
            and sum(self._exceedance_history)
            >= self.required_exceedances
        )
        severe = (
            self.single_exceedance_threshold is not None
            and nis >= self.single_exceedance_threshold
        )
        if (
            (persistent or severe)
            and float(timestamp) >= float(self._refractory_until)
        ):
            self.mode_probabilities = self.reset_mode_probabilities.copy()
            self._inflate_component_covariances(
                self.state_covariance_inflation
            )
            self._activate_recovery(timestamp)
            self._refractory_until = (
                float(timestamp) + self.refractory_period_s
            )
            self.last_change_triggered = True
            self.detection_events.append(
                {
                    "kind": "nis_change",
                    "timestamp": float(timestamp),
                    "nis": nis,
                    "trigger_rule": (
                        "single_severe" if severe else "persistent"
                    ),
                }
            )
            self._exceedance_history.clear()


def run_online_imm_forecasts(
    times,
    observations,
    observed_mask,
    predictor,
    horizon,
    forecast_dt,
    forecast_stride_steps=1,
    warmup_steps=0,
) -> List[OnlineIMMForecast]:
    """Run IMM with observation history only; no truth/intent argument exists."""
    times = np.asarray(times, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    count = times.shape[0]
    if (
        count < 2
        or observations.shape != (count, 2)
        or observed_mask.shape != (count,)
    ):
        raise ValueError("IMM online inputs have inconsistent shapes")
    stride = int(forecast_stride_steps)
    warmup = int(warmup_steps)
    if stride <= 0 or warmup < 0:
        raise ValueError("IMM stride/warmup is invalid")
    records: List[OnlineIMMForecast] = []
    for index in range(count):
        observation = observations[index] if observed_mask[index] else None
        predictor.update(observation, float(times[index]))
        if (
            predictor.state is None
            or index < warmup
            or (index - warmup) % stride != 0
        ):
            continue
        future = predictor.forecast(horizon, forecast_dt)
        records.append(
            OnlineIMMForecast(
                time_index=index,
                timestamp=float(times[index]),
                observation_available=bool(observed_mask[index]),
                filtered_mean=predictor.state.copy(),
                filtered_covariance=predictor.covariance,
                filtered_mode_probabilities=(
                    predictor.mode_probabilities.copy()
                ),
                future=future,
                innovation_nis=getattr(
                    predictor, "last_innovation_nis", None
                ),
                change_triggered=bool(
                    getattr(predictor, "last_change_triggered", False)
                ),
                dropout_guard_triggered=bool(
                    getattr(
                        predictor,
                        "last_dropout_guard_triggered",
                        False,
                    )
                ),
                recovery_active=bool(
                    getattr(predictor, "recovery_active", False)
                ),
            )
        )
    return records


__all__: Sequence[str] = (
    "IMM_MODEL_NAMES",
    "ChangeAwareIMMPredictor",
    "MixturePrediction",
    "OnlineIMMForecast",
    "OrdinaryIMMPredictor",
    "brake_transition",
    "run_online_imm_forecasts",
    "turn_transition",
)
