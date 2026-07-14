"""Low-dimensional RL action to full MPPI sampling-distribution parameters."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class PriorParameterizationConfig:
    kind: str = "control_knots"
    num_knots: int = 6
    mode: str = "delta"
    delta_scale: float = 0.65
    learn_covariance: bool = False
    covariance_min_scale: float = 0.5
    covariance_max_scale: float = 2.0
    subgoal_min_distance: float = 0.25
    subgoal_max_distance: float = 1.50
    subgoal_max_bearing: float = float(np.pi)
    subgoal_v_gain: float = 0.8
    subgoal_yaw_gain: float = 1.5
    subgoal_heading_gate_rad: float = float(np.pi / 3.0)
    subgoal_control_dt: float = 0.1
    subgoal_decoder: str = "kinematic"
    subgoal_velocity_time_constant: float = 0.18
    subgoal_yaw_time_constant: float = 0.12
    subgoal_command_delay: float = 0.0
    subgoal_max_integration_step: float = 0.02

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            kind=str(values.get("kind", "control_knots")),
            num_knots=int(values.get("num_knots", 6)),
            mode=str(values.get("mode", "delta")),
            delta_scale=float(values.get("delta_scale", 0.65)),
            learn_covariance=bool(values.get("learn_covariance", False)),
            covariance_min_scale=float(values.get("covariance_min_scale", 0.5)),
            covariance_max_scale=float(values.get("covariance_max_scale", 2.0)),
            subgoal_min_distance=float(values.get("subgoal_min_distance", 0.25)),
            subgoal_max_distance=float(values.get("subgoal_max_distance", 1.50)),
            subgoal_max_bearing=float(values.get("subgoal_max_bearing", np.pi)),
            subgoal_v_gain=float(values.get("subgoal_v_gain", 0.8)),
            subgoal_yaw_gain=float(values.get("subgoal_yaw_gain", 1.5)),
            subgoal_heading_gate_rad=float(
                values.get("subgoal_heading_gate_rad", np.pi / 3.0)
            ),
            subgoal_control_dt=float(values.get("subgoal_control_dt", 0.1)),
            subgoal_decoder=str(values.get("subgoal_decoder", "kinematic")),
            subgoal_velocity_time_constant=float(
                values.get("subgoal_velocity_time_constant", 0.18)
            ),
            subgoal_yaw_time_constant=float(
                values.get("subgoal_yaw_time_constant", 0.12)
            ),
            subgoal_command_delay=float(values.get("subgoal_command_delay", 0.0)),
            subgoal_max_integration_step=float(
                values.get("subgoal_max_integration_step", 0.02)
            ),
        )

    def validate(self):
        numeric_values = (
            self.delta_scale,
            self.covariance_min_scale,
            self.covariance_max_scale,
            self.subgoal_min_distance,
            self.subgoal_max_distance,
            self.subgoal_max_bearing,
            self.subgoal_v_gain,
            self.subgoal_yaw_gain,
            self.subgoal_heading_gate_rad,
            self.subgoal_control_dt,
            self.subgoal_velocity_time_constant,
            self.subgoal_yaw_time_constant,
            self.subgoal_command_delay,
            self.subgoal_max_integration_step,
        )
        if not np.isfinite(np.asarray(numeric_values, dtype=np.float64)).all():
            raise ValueError("RL prior parameterization values must be finite")
        if self.kind not in ("control_knots", "local_subgoal"):
            raise ValueError(
                "RL prior kind must be control_knots or local_subgoal"
            )
        if self.subgoal_decoder not in ("kinematic", "dynamic_first_order"):
            raise ValueError(
                "local-subgoal decoder must be kinematic or dynamic_first_order"
            )
        if self.num_knots < 2:
            raise ValueError("RL prior requires at least two sequence knots")
        if self.mode not in ("absolute", "delta"):
            raise ValueError("RL prior mode must be 'absolute' or 'delta'")
        if self.delta_scale <= 0.0:
            raise ValueError("RL prior delta_scale must be positive")
        if not 0.0 < self.covariance_min_scale <= self.covariance_max_scale:
            raise ValueError("invalid RL prior covariance scale bounds")
        if (
            self.subgoal_min_distance <= 0.0
            or self.subgoal_max_distance <= self.subgoal_min_distance
        ):
            raise ValueError(
                "local-subgoal maximum distance must exceed a positive minimum"
            )
        if not 0.0 < self.subgoal_max_bearing <= np.pi:
            raise ValueError("local-subgoal maximum bearing must be in (0, pi]")
        if min(
            self.subgoal_v_gain,
            self.subgoal_yaw_gain,
            self.subgoal_heading_gate_rad,
            self.subgoal_control_dt,
        ) <= 0.0:
            raise ValueError("local-subgoal gains, heading gate and dt must be positive")
        if self.subgoal_heading_gate_rad > np.pi:
            raise ValueError("local-subgoal heading gate cannot exceed pi")
        if min(
            self.subgoal_velocity_time_constant,
            self.subgoal_yaw_time_constant,
            self.subgoal_max_integration_step,
        ) <= 0.0:
            raise ValueError(
                "dynamic local-subgoal time constants and integration step must be positive"
            )
        if self.subgoal_command_delay < 0.0:
            raise ValueError("dynamic local-subgoal command delay cannot be negative")

    def to_dict(self):
        return asdict(self)


def _validated_local_subgoal_config(config):
    result = (
        config
        if isinstance(config, PriorParameterizationConfig)
        else PriorParameterizationConfig.from_mapping(config)
    )
    result.validate()
    if result.kind != "local_subgoal" or result.learn_covariance:
        raise ValueError(
            "local-subgoal action conversion requires kind=local_subgoal "
            "and learn_covariance=false"
        )
    return result


def encode_local_subgoal_action(config, distance, bearing):
    """Map a body-frame subgoal into the actor's normalized two-vector."""

    config = _validated_local_subgoal_config(config)
    distance = float(distance)
    bearing = float(bearing)
    if not np.isfinite((distance, bearing)).all():
        raise ValueError("local-subgoal distance and bearing must be finite")
    wrapped = float(np.arctan2(np.sin(bearing), np.cos(bearing)))
    clipped_distance = float(np.clip(
        distance, config.subgoal_min_distance, config.subgoal_max_distance
    ))
    clipped_bearing = float(np.clip(
        wrapped, -config.subgoal_max_bearing, config.subgoal_max_bearing
    ))
    normalized_distance = (
        2.0
        * (clipped_distance - config.subgoal_min_distance)
        / (config.subgoal_max_distance - config.subgoal_min_distance)
        - 1.0
    )
    result = np.asarray(
        (normalized_distance, clipped_bearing / config.subgoal_max_bearing),
        dtype=np.float32,
    )
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1.000001):
        raise FloatingPointError("local-subgoal encoding produced invalid values")
    return result


def decode_local_subgoal_action(config, action):
    """Decode a normalized actor action into body-frame distance/bearing."""

    config = _validated_local_subgoal_config(config)
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("normalized local-subgoal action must be finite [2]")
    if np.any(values < -1.000001) or np.any(values > 1.000001):
        raise ValueError("normalized local-subgoal action must lie in [-1, 1]")
    values = np.clip(values, -1.0, 1.0)
    unit_distance = 0.5 * (float(values[0]) + 1.0)
    distance = config.subgoal_min_distance + unit_distance * (
        config.subgoal_max_distance - config.subgoal_min_distance
    )
    bearing = float(values[1]) * config.subgoal_max_bearing
    return float(distance), float(bearing)


class PriorParameterization:
    """Decode normalized policy actions into a smooth mean sequence.

    With ``mode=delta`` the policy modifies the trusted goal warm-start prior;
    with ``mode=absolute`` it specifies the full mean.  Both remain bounded by
    the configured action space before MPPI sees them.
    """

    def __init__(self, config, action_spec, base_noise_sigma=None):
        self.config = (
            config
            if isinstance(config, PriorParameterizationConfig)
            else PriorParameterizationConfig.from_mapping(config)
        )
        self.config.validate()
        self.action_spec = action_spec
        if base_noise_sigma is None:
            base_noise_sigma = np.ones(action_spec.dimension, dtype=np.float64)
        self.base_noise_sigma = np.asarray(base_noise_sigma, dtype=np.float64).reshape(-1)
        if (
            self.base_noise_sigma.shape != (action_spec.dimension,)
            or not np.isfinite(self.base_noise_sigma).all()
            or np.any(self.base_noise_sigma <= 0.0)
        ):
            raise ValueError("base_noise_sigma must be positive per action dimension")

    @property
    def parameter_dimension(self):
        size = (
            2
            if self.config.kind == "local_subgoal"
            else self.config.num_knots * self.action_spec.dimension
        )
        if self.config.learn_covariance:
            size += self.action_spec.dimension
        return size

    @property
    def _mean_parameter_dimension(self):
        return self.parameter_dimension - (
            self.action_spec.dimension if self.config.learn_covariance else 0
        )

    def _interpolate(self, knots, horizon):
        source = np.linspace(0.0, 1.0, self.config.num_knots)
        target = np.linspace(0.0, 1.0, int(horizon))
        return np.stack(
            [np.interp(target, source, knots[:, index]) for index in range(knots.shape[1])],
            axis=-1,
        )

    @staticmethod
    def _wrap_angle(angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _local_subgoal_target(self, values):
        distance, bearing = decode_local_subgoal_action(self.config, values)
        return distance, bearing, distance * np.cos(bearing), distance * np.sin(bearing)

    def _local_subgoal_command(self, x, y, theta, target_x, target_y):
        dx = target_x - x
        dy = target_y - y
        remaining = float(np.hypot(dx, dy))
        desired = float(np.arctan2(dy, dx))
        yaw_error = self._wrap_angle(desired - theta)
        gate = self.config.subgoal_heading_gate_rad
        if abs(yaw_error) >= gate:
            alignment = 0.0
        else:
            gate_cosine = float(np.cos(gate))
            alignment = max(
                0.0,
                (float(np.cos(yaw_error)) - gate_cosine)
                / max(1.0 - gate_cosine, 1e-12),
            )
        return (
            self.config.subgoal_v_gain * remaining * alignment,
            self.config.subgoal_yaw_gain * yaw_error,
            remaining,
            yaw_error,
        )

    def _initial_local_subgoal_mean(self, horizon, baseline_mean):
        expected = (int(horizon), self.action_spec.dimension)
        if baseline_mean is None:
            return np.zeros(expected, dtype=np.float64)
        mean = np.asarray(baseline_mean, dtype=np.float64).copy()
        if mean.shape != expected or not np.isfinite(mean).all():
            raise ValueError("RL prior baseline has an invalid mean sequence")
        return mean

    def _decode_local_subgoal_kinematic(self, values, horizon, baseline_mean):
        distance, bearing, target_x, target_y = self._local_subgoal_target(values)
        mean = self._initial_local_subgoal_mean(horizon, baseline_mean)
        x = 0.0
        y = 0.0
        theta = 0.0
        v_index = self.action_spec.index("v_cmd")
        omega_index = self.action_spec.index("omega_cmd")
        dt = self.config.subgoal_control_dt
        remaining = distance
        for step in range(int(horizon)):
            v_command, omega_command, remaining, _ = self._local_subgoal_command(
                x, y, theta, target_x, target_y
            )
            mean[step, v_index] = v_command
            mean[step, omega_index] = omega_command
            mean[step] = self.action_spec.clip(mean[step])
            v_command = float(mean[step, v_index])
            omega_command = float(mean[step, omega_index])
            x += dt * v_command * np.cos(theta)
            y += dt * v_command * np.sin(theta)
            theta = self._wrap_angle(theta + dt * omega_command)
        return mean, {
            "parameterization": "local_subgoal",
            "subgoal_decoder": "kinematic",
            "subgoal_distance": float(distance),
            "subgoal_bearing": float(bearing),
            "subgoal_x_body": float(target_x),
            "subgoal_y_body": float(target_y),
            "predicted_terminal_distance": float(remaining),
        }

    def _advance_first_order(self, state, control, duration):
        """Advance local pose/twist under a constant delayed command.

        Speed and yaw-rate responses are integrated analytically.  Translation
        uses their interval averages and a midpoint heading, which is stable at
        zero rate and sufficiently accurate for sampling-prior construction.
        """
        x, y, theta, v_value, omega = state
        v_command, omega_command = control
        remaining = float(duration)
        maximum_step = self.config.subgoal_max_integration_step
        tau_v = self.config.subgoal_velocity_time_constant
        tau_omega = self.config.subgoal_yaw_time_constant
        while remaining > 1e-12:
            dt = min(maximum_step, remaining)
            velocity_decay = float(np.exp(-dt / tau_v))
            yaw_decay = float(np.exp(-dt / tau_omega))
            next_v = v_command + (v_value - v_command) * velocity_decay
            next_omega = omega_command + (omega - omega_command) * yaw_decay
            average_v = v_command + (v_value - v_command) * (
                tau_v * (1.0 - velocity_decay) / dt
            )
            delta_theta = omega_command * dt + (omega - omega_command) * (
                tau_omega * (1.0 - yaw_decay)
            )
            midpoint_theta = theta + 0.5 * delta_theta
            x += dt * average_v * np.cos(midpoint_theta)
            y += dt * average_v * np.sin(midpoint_theta)
            theta = self._wrap_angle(theta + delta_theta)
            v_value = next_v
            omega = next_omega
            remaining -= dt
        return np.asarray((x, y, theta, v_value, omega), dtype=np.float64)

    def _decode_local_subgoal_dynamic(
        self,
        values,
        horizon,
        baseline_mean,
        current_twist,
        previous_control,
    ):
        twist = np.asarray(current_twist, dtype=np.float64).reshape(-1)
        if twist.shape != (2,) or not np.isfinite(twist).all():
            raise ValueError(
                "dynamic local-subgoal decoder requires finite current [v, omega]"
            )
        previous = np.asarray(previous_control, dtype=np.float64).reshape(-1)
        expected_control = (self.action_spec.dimension,)
        if previous.shape != expected_control or not np.isfinite(previous).all():
            raise ValueError(
                "dynamic local-subgoal decoder requires the previous control"
            )
        previous = self.action_spec.clip(previous)
        distance, bearing, target_x, target_y = self._local_subgoal_target(values)
        mean = self._initial_local_subgoal_mean(horizon, baseline_mean)
        v_index = self.action_spec.index("v_cmd")
        omega_index = self.action_spec.index("omega_cmd")
        state = np.asarray((0.0, 0.0, 0.0, twist[0], twist[1]), dtype=np.float64)
        active_control = np.asarray(
            (previous[v_index], previous[omega_index]), dtype=np.float64
        )
        pending = []
        time_value = 0.0
        control_dt = self.config.subgoal_control_dt
        command_delay = self.config.subgoal_command_delay
        remaining = distance
        for step in range(int(horizon)):
            v_command, omega_command, remaining, _ = self._local_subgoal_command(
                state[0], state[1], state[2], target_x, target_y
            )
            mean[step, v_index] = v_command
            mean[step, omega_index] = omega_command
            mean[step] = self.action_spec.clip(mean[step])
            scheduled = np.asarray(
                (mean[step, v_index], mean[step, omega_index]), dtype=np.float64
            )
            pending.append((time_value + command_delay, scheduled))
            interval_end = time_value + control_dt
            while pending and pending[0][0] <= interval_end + 1e-12:
                activation_time, activated = pending.pop(0)
                if activation_time > time_value + 1e-12:
                    state = self._advance_first_order(
                        state, active_control, activation_time - time_value
                    )
                    time_value = activation_time
                active_control = activated
            if interval_end > time_value + 1e-12:
                state = self._advance_first_order(
                    state, active_control, interval_end - time_value
                )
            time_value = interval_end
        remaining = float(np.hypot(target_x - state[0], target_y - state[1]))
        return mean, {
            "parameterization": "local_subgoal",
            "subgoal_decoder": "dynamic_first_order",
            "subgoal_distance": float(distance),
            "subgoal_bearing": float(bearing),
            "subgoal_x_body": float(target_x),
            "subgoal_y_body": float(target_y),
            "initial_v": float(twist[0]),
            "initial_omega": float(twist[1]),
            "previous_v_cmd": float(previous[v_index]),
            "previous_omega_cmd": float(previous[omega_index]),
            "velocity_time_constant": float(
                self.config.subgoal_velocity_time_constant
            ),
            "yaw_time_constant": float(self.config.subgoal_yaw_time_constant),
            "command_delay": float(command_delay),
            "predicted_terminal_distance": remaining,
            "predicted_terminal_v": float(state[3]),
            "predicted_terminal_omega": float(state[4]),
        }

    def _decode_local_subgoal(
        self,
        values,
        horizon,
        baseline_mean,
        current_twist,
        previous_control,
    ):
        required = ("v_cmd", "omega_cmd")
        if any(name not in self.action_spec.names for name in required):
            raise ValueError(
                "local_subgoal prior requires v_cmd and omega_cmd actions"
            )
        if self.config.subgoal_decoder == "kinematic":
            return self._decode_local_subgoal_kinematic(
                values, horizon, baseline_mean
            )
        return self._decode_local_subgoal_dynamic(
            values,
            horizon,
            baseline_mean,
            current_twist,
            previous_control,
        )

    def decode(
        self,
        parameters,
        horizon,
        baseline_mean=None,
        current_twist=None,
        previous_control=None,
    ):
        values = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if values.shape != (self.parameter_dimension,) or not np.isfinite(values).all():
            raise ValueError("RL prior parameters have an invalid shape or non-finite values")
        values = np.clip(values, -1.0, 1.0)
        action_dim = self.action_spec.dimension
        mean_parameter_count = self._mean_parameter_dimension
        if self.config.kind == "local_subgoal":
            mean, metadata = self._decode_local_subgoal(
                values[:mean_parameter_count],
                horizon,
                baseline_mean,
                current_twist,
                previous_control,
            )
        else:
            normalized_knots = values[:mean_parameter_count].reshape(
                self.config.num_knots, action_dim
            )
            normalized_sequence = self._interpolate(normalized_knots, horizon)
            if self.config.mode == "absolute":
                center = 0.5 * (self.action_spec.upper + self.action_spec.lower)
                half_range = 0.5 * (self.action_spec.upper - self.action_spec.lower)
                mean = center[None, :] + normalized_sequence * half_range[None, :]
            else:
                if baseline_mean is None:
                    raise ValueError("delta RL prior requires a baseline mean sequence")
                baseline_mean = np.asarray(baseline_mean, dtype=np.float64)
                expected = (int(horizon), action_dim)
                if baseline_mean.shape != expected or not np.isfinite(baseline_mean).all():
                    raise ValueError("RL prior baseline has an invalid mean sequence")
                action_range = self.action_spec.upper - self.action_spec.lower
                mean = baseline_mean + (
                    self.config.delta_scale * normalized_sequence * action_range[None, :]
                )
            metadata = {
                "parameterization": self.config.mode,
                "num_knots": self.config.num_knots,
            }
        mean = np.clip(mean, self.action_spec.lower, self.action_spec.upper)
        covariance = None
        covariance_scale = np.ones(action_dim, dtype=np.float64)
        if self.config.learn_covariance:
            raw = values[mean_parameter_count:]
            unit = 0.5 * (raw + 1.0)
            low = self.config.covariance_min_scale
            high = self.config.covariance_max_scale
            covariance_scale = np.exp(np.log(low) + unit * (np.log(high) - np.log(low)))
            covariance = np.diag((self.base_noise_sigma * covariance_scale) ** 2)
        metadata.update({
            "kind": self.config.kind,
            "learn_covariance": self.config.learn_covariance,
            "covariance_scale": covariance_scale.tolist(),
        })
        return mean, covariance, metadata
