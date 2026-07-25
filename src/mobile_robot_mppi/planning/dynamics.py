import numpy as np


class LegacyUnicyclePrediction:
    state_dim = 3
    control_dim = 2

    def derivative(self, state, control, time=None):
        del time
        state = np.asarray(state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)
        theta = state[..., 2]
        v_value = control[..., 0]
        omega = control[..., 1]
        return np.stack((v_value * np.cos(theta), v_value * np.sin(theta), omega), axis=-1)


class DynamicUnicyclePrediction:
    state_dim = 5
    control_dim = 2

    def __init__(self, velocity_time_constant=0.18, yaw_time_constant=0.12):
        self.velocity_time_constant = float(velocity_time_constant)
        self.yaw_time_constant = float(yaw_time_constant)
        if self.velocity_time_constant <= 0.0 or self.yaw_time_constant <= 0.0:
            raise ValueError("time constants must be positive")

    def derivative(self, state, control, time=None):
        del time
        state = np.asarray(state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)
        theta = state[..., 2]
        v_value = state[..., 3]
        omega = state[..., 4]
        return np.stack(
            (
                v_value * np.cos(theta),
                v_value * np.sin(theta),
                omega,
                (control[..., 0] - v_value) / self.velocity_time_constant,
                (control[..., 1] - omega) / self.yaw_time_constant,
            ),
            axis=-1,
        )


class ResidualPrediction:
    def __init__(self, nominal, residual):
        self.nominal = nominal
        self.residual = residual
        self.state_dim = int(nominal.state_dim)
        self.control_dim = int(nominal.control_dim)

    def derivative(self, state, control, time=None):
        nominal = self.nominal.derivative(state, control, time)
        residual = self.residual.derivative(state, control, time)
        value = np.asarray(nominal) + np.asarray(residual)
        if not np.isfinite(value).all():
            raise FloatingPointError("combined prediction produced NaN or Inf")
        return value

    @property
    def supports_rollout_batch(self):
        return bool(
            getattr(self.residual, "supports_combined_rollout", False)
        )

    def rollout_batch(
        self, initial_state, controls, dt, state_spec, method="euler"
    ):
        if not self.supports_rollout_batch:
            raise RuntimeError("residual dynamics has no batch-rollout fast path")
        return self.residual.combined_rollout(
            self.nominal,
            initial_state,
            controls,
            dt,
            state_spec,
            method,
        )


class OracleResidualPrediction:
    """Exact derivative difference for explicit low-order oracle ablations."""

    def __init__(self, true_dynamics, nominal_dynamics):
        self.true_dynamics = true_dynamics
        self.nominal_dynamics = nominal_dynamics
        self.state_dim = int(nominal_dynamics.state_dim)
        self.control_dim = int(nominal_dynamics.control_dim)

    def derivative(self, state, control, time=None):
        return (
            np.asarray(self.true_dynamics.derivative(state, control, time), dtype=np.float64)
            - np.asarray(self.nominal_dynamics.derivative(state, control, time), dtype=np.float64)
        )


class GainBiasedUnicyclePrediction(LegacyUnicyclePrediction):
    def __init__(self, velocity_gain=1.0, yaw_gain=1.0, yaw_bias=0.0):
        self.velocity_gain = float(velocity_gain)
        self.yaw_gain = float(yaw_gain)
        self.yaw_bias = float(yaw_bias)

    def derivative(self, state, control, time=None):
        del time
        state = np.asarray(state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)
        theta = state[..., 2]
        v_value = self.velocity_gain * control[..., 0]
        omega = self.yaw_gain * control[..., 1] + self.yaw_bias
        return np.stack((v_value * np.cos(theta), v_value * np.sin(theta), omega), axis=-1)


class BatchCompatibleResidual:
    """Lift a validated single-transition residual adapter to batch rollouts."""

    def __init__(self, residual):
        self.residual = residual
        self.state_dim = int(residual.state_dim)
        self.control_dim = int(residual.control_dim)

    def derivative(self, state, control, time=None):
        state_value = np.asarray(state, dtype=np.float64)
        control_value = np.asarray(control, dtype=np.float64)
        if state_value.ndim == 1:
            return self.residual.derivative(state_value, control_value, time)
        leading = state_value.shape[:-1]
        flat_state = state_value.reshape(-1, state_value.shape[-1])
        flat_control = control_value.reshape(-1, control_value.shape[-1])
        values = [
            self.residual.derivative(flat_state[index], flat_control[index], time)
            for index in range(flat_state.shape[0])
        ]
        return np.asarray(values, dtype=np.float64).reshape(leading + (self.state_dim,))
