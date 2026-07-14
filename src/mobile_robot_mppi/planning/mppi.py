"""Dimension-agnostic MPPI controller with plugin cost and prior ports."""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec, StateSpec
from mobile_robot_mppi.core.types import ControlCommand, PlanResult, RobotObservation
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior, PriorOutput


def _wrap_periodic(state: np.ndarray, state_spec: StateSpec) -> np.ndarray:
    for index in state_spec.periodic_indices:
        state[..., index] = np.arctan2(np.sin(state[..., index]), np.cos(state[..., index]))
    return state


def integrate_batch(dynamics, state, control, dt, state_spec, method="euler"):
    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if dt <= 0.0 or not np.isfinite(state).all() or not np.isfinite(control).all():
        raise ValueError("integration inputs must be finite and dt positive")
    if method == "euler":
        result = state + dt * dynamics.derivative(state, control)
    elif method == "rk4":
        k1 = dynamics.derivative(state, control)
        k2 = dynamics.derivative(state + 0.5 * dt * k1, control)
        k3 = dynamics.derivative(state + 0.5 * dt * k2, control)
        k4 = dynamics.derivative(state + dt * k3, control)
        result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    else:
        raise ValueError("unknown integration method: %s" % method)
    if not np.isfinite(result).all():
        raise FloatingPointError("rollout integration produced NaN or Inf")
    return _wrap_periodic(result, state_spec)


@dataclass(frozen=True)
class MppiConfig:
    horizon: int = 25
    num_samples: int = 400
    dt: float = 0.1
    temperature: float = 4.0
    noise_sigma: Tuple[float, ...] = (0.12, 0.35)
    integrator: str = "rk4"
    goal_running_weight: float = 1.0
    goal_terminal_weight: float = 12.0
    heading_weight: float = 0.2
    terminal_velocity_weight: float = 0.0
    terminal_yaw_rate_weight: float = 0.0
    control_weight: float = 0.05
    control_rate_weight: float = 0.08
    obstacle_weight: float = 30.0
    obstacle_influence: float = 0.55
    robot_radius: float = 0.25
    collision_penalty: float = 5000.0
    importance_sampling_correction: bool = False
    previous_sequence_blend: float = 0.5
    safety_recovery_prefix_steps: int = 0
    seed: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], action_dim: int):
        sigma = values.get("noise_sigma", (0.12, 0.35))
        if np.isscalar(sigma):
            sigma = (float(sigma),) * int(action_dim)
        return cls(
            horizon=int(values.get("horizon", 25)),
            num_samples=int(values.get("num_samples", 400)),
            dt=float(values.get("dt", values.get("control_dt", 0.1))),
            temperature=float(values.get("temperature", 4.0)),
            noise_sigma=tuple(float(v) for v in sigma),
            integrator=str(values.get("integrator", "rk4")),
            goal_running_weight=float(values.get("goal_running_weight", 1.0)),
            goal_terminal_weight=float(values.get("goal_terminal_weight", 12.0)),
            heading_weight=float(values.get("heading_weight", 0.2)),
            terminal_velocity_weight=float(values.get("terminal_velocity_weight", 0.0)),
            terminal_yaw_rate_weight=float(values.get("terminal_yaw_rate_weight", 0.0)),
            control_weight=float(values.get("control_weight", 0.05)),
            control_rate_weight=float(values.get("control_rate_weight", 0.08)),
            obstacle_weight=float(values.get("obstacle_weight", 30.0)),
            obstacle_influence=float(values.get("obstacle_influence", 0.55)),
            robot_radius=float(values.get("robot_radius", 0.25)),
            collision_penalty=float(values.get("collision_penalty", 5000.0)),
            importance_sampling_correction=bool(
                values.get("importance_sampling_correction", False)
            ),
            previous_sequence_blend=float(values.get("previous_sequence_blend", 0.5)),
            safety_recovery_prefix_steps=int(
                values.get("safety_recovery_prefix_steps", 0)
            ),
            seed=int(values.get("seed", 0)),
        )

    def validate(self, action_dim: int) -> None:
        if self.horizon <= 0 or self.num_samples <= 0 or self.dt <= 0.0:
            raise ValueError("MPPI horizon, samples, and dt must be positive")
        if self.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive")
        if len(self.noise_sigma) != action_dim or any(value <= 0.0 for value in self.noise_sigma):
            raise ValueError("noise_sigma must contain one positive value per action dimension")
        numeric_costs = (
            self.goal_running_weight,
            self.goal_terminal_weight,
            self.heading_weight,
            self.terminal_velocity_weight,
            self.terminal_yaw_rate_weight,
            self.control_weight,
            self.control_rate_weight,
            self.obstacle_weight,
            self.obstacle_influence,
            self.robot_radius,
            self.collision_penalty,
        )
        if not np.isfinite(numeric_costs).all() or any(value < 0.0 for value in numeric_costs):
            raise ValueError("MPPI cost and geometry parameters must be finite and non-negative")
        if self.integrator not in ("euler", "rk4"):
            raise ValueError("MPPI integrator must be 'euler' or 'rk4'")
        if not 0.0 <= self.previous_sequence_blend <= 1.0:
            raise ValueError("previous_sequence_blend must be between zero and one")
        if not 0 <= self.safety_recovery_prefix_steps <= self.horizon:
            raise ValueError("safety_recovery_prefix_steps must be within the horizon")


class MppiController:
    def __init__(
        self,
        dynamics,
        state_spec: StateSpec,
        action_spec: ActionSpec,
        config: MppiConfig,
        sampling_prior=None,
        memory_cost: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        config.validate(action_spec.dimension)
        if int(dynamics.state_dim) != state_spec.dimension:
            raise ValueError("prediction dynamics and state spec dimensions differ")
        if int(dynamics.control_dim) != action_spec.dimension:
            raise ValueError("prediction dynamics and action spec dimensions differ")
        self.dynamics = dynamics
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.config = config
        self.sampling_prior = sampling_prior or GoalWarmStartPrior()
        self.memory_cost = memory_cost
        self.rng = np.random.RandomState(config.seed)
        self.previous_sequence = None
        self.previous_action = np.zeros(action_spec.dimension, dtype=np.float64)
        self._safety_blocked = False

    def reset(self):
        self.rng = np.random.RandomState(self.config.seed)
        self.previous_sequence = None
        self.previous_action = np.zeros(self.action_spec.dimension, dtype=np.float64)
        self._safety_blocked = False
        reset = getattr(self.sampling_prior, "reset", None)
        if callable(reset):
            reset()

    def state_from_observation(self, observation: RobotObservation) -> np.ndarray:
        values = {
            "x": observation.pose.x,
            "y": observation.pose.y,
            "theta": observation.pose.theta,
            "v": observation.twist.v,
            "omega": observation.twist.omega,
        }
        values.update(observation.auxiliary)
        try:
            state = np.asarray([values[name] for name in self.state_spec.names], dtype=np.float64)
        except KeyError as exc:
            raise ValueError("observation does not provide state channel %s" % exc)
        return self.state_spec.validate(state)

    def _prior(self, observation, reference) -> PriorOutput:
        output = self.sampling_prior.propose(
            observation, reference, self.config.horizon, self.action_spec
        )
        mean = np.asarray(output.mean, dtype=np.float64)
        expected = (self.config.horizon, self.action_spec.dimension)
        if mean.shape != expected or not np.isfinite(mean).all():
            raise ValueError("sampling prior returned an invalid mean")
        metadata = dict(output.metadata)
        if self.previous_sequence is not None:
            shifted = np.empty_like(self.previous_sequence)
            shifted[:-1] = self.previous_sequence[1:]
            shifted[-1] = self.previous_sequence[-1]
            blend = self.config.previous_sequence_blend
            mean = (1.0 - blend) * mean + blend * shifted
        if (
            self._safety_blocked
            and self.config.safety_recovery_prefix_steps > 0
            and "v_cmd" in self.action_spec.names
        ):
            v_index = self.action_spec.index("v_cmd")
            mean[:self.config.safety_recovery_prefix_steps, v_index] = 0.0
            metadata["safety_recovery"] = True
            metadata["safety_recovery_prefix_steps"] = int(
                self.config.safety_recovery_prefix_steps
            )
        else:
            metadata["safety_recovery"] = False
        return PriorOutput(
            np.clip(mean, self.action_spec.lower, self.action_spec.upper),
            output.covariance,
            metadata,
        )

    def observe_safety_decision(self, decision) -> None:
        """Close the loop between MPPI's proposal and the command actually sent.

        A forward stop must not erase angular control.  When scan_guard blocks
        translation, the next proposal therefore starts with a zero-translation
        prefix while retaining the angular sequence, allowing an in-place turn.
        """
        executed = np.asarray(decision.executed_control.values, dtype=np.float64).reshape(-1)
        proposed = np.asarray(decision.proposed_control.values, dtype=np.float64).reshape(-1)
        expected = (self.action_spec.dimension,)
        if (
            executed.shape != expected
            or proposed.shape != expected
            or not np.isfinite(executed).all()
            or not np.isfinite(proposed).all()
        ):
            raise ValueError("safety feedback actions must be finite and match action dimension")
        self.previous_action = executed.copy()
        blocked = False
        if "v_cmd" in self.action_spec.names:
            v_index = self.action_spec.index("v_cmd")
            blocked = bool(
                decision.overridden
                and proposed[v_index] > executed[v_index] + 1e-12
            )
            if blocked and self.previous_sequence is not None:
                prefix = max(1, self.config.safety_recovery_prefix_steps)
                self.previous_sequence[:prefix, v_index] = executed[v_index]
        self._safety_blocked = blocked
        prior_observer = getattr(self.sampling_prior, "observe_safety_decision", None)
        if callable(prior_observer):
            prior_observer(decision)

    def _sample(self, prior: PriorOutput) -> np.ndarray:
        shape = (self.config.num_samples, self.config.horizon, self.action_spec.dimension)
        if prior.covariance is None:
            noise = self.rng.normal(size=shape) * np.asarray(self.config.noise_sigma)[None, None, :]
        else:
            covariance = np.asarray(prior.covariance, dtype=np.float64)
            if covariance.shape == (self.action_spec.dimension, self.action_spec.dimension):
                noise = self.rng.multivariate_normal(
                    np.zeros(self.action_spec.dimension), covariance,
                    size=(self.config.num_samples, self.config.horizon),
                )
            else:
                raise ValueError("prior covariance must have shape [action_dim, action_dim]")
        samples = prior.mean[None, :, :] + noise
        samples = np.clip(samples, self.action_spec.lower, self.action_spec.upper)
        samples[0] = prior.mean
        return samples

    def _sampling_covariance(self, prior: PriorOutput) -> np.ndarray:
        if prior.covariance is None:
            sigma = np.asarray(self.config.noise_sigma, dtype=np.float64)
            return np.diag(sigma ** 2)
        covariance = np.asarray(prior.covariance, dtype=np.float64)
        expected = (self.action_spec.dimension, self.action_spec.dimension)
        if covariance.shape != expected or not np.isfinite(covariance).all():
            raise ValueError("prior covariance must be a finite [action_dim, action_dim] matrix")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("prior covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("prior covariance must be positive definite") from exc
        return covariance

    def _importance_sampling_cost(self, nominal, perturbations, covariance):
        """Return the MPPI likelihood-ratio correction for each rollout.

        For a Gaussian proposal with mean ``nominal`` and covariance Sigma,
        the practical MPPI correction is

            lambda * R * sum_t nominal_t^T Sigma^-1 epsilon_t,

        where this implementation uses ``control_weight`` as the scalar
        control-cost matrix R.  Keeping R explicit prevents a small sampling
        variance from unintentionally collapsing all weight onto one rollout.

        It is intentionally optional so archived experiments can reproduce the
        repository's former weighted-sampling behavior exactly.
        """
        if not self.config.importance_sampling_correction:
            return np.zeros(perturbations.shape[0], dtype=np.float64)
        inverse_covariance = np.linalg.inv(covariance)
        return self.config.temperature * self.config.control_weight * np.einsum(
            "ha,ab,kha->k", nominal, inverse_covariance, perturbations
        )

    def rollout(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 2:
            controls = controls[None, ...]
        batch = controls.shape[0]
        trajectory = np.empty((batch, self.config.horizon + 1, self.state_spec.dimension), dtype=np.float64)
        trajectory[:, 0, :] = np.repeat(initial_state[None, :], batch, axis=0)
        for step in range(self.config.horizon):
            trajectory[:, step + 1, :] = integrate_batch(
                self.dynamics,
                trajectory[:, step, :],
                controls[:, step, :],
                self.config.dt,
                self.state_spec,
                self.config.integrator,
            )
        return trajectory

    def _cost(self, trajectories, controls, target, obstacles):
        xy_indices = self.state_spec.position_indices
        xy = trajectories[..., list(xy_indices)]
        target_xy = np.asarray((target.pose.x, target.pose.y), dtype=np.float64)
        distance_sq = np.sum((xy - target_xy) ** 2, axis=-1)
        costs = self.config.goal_running_weight * np.sum(distance_sq[:, 1:-1], axis=1)
        costs += self.config.goal_terminal_weight * distance_sq[:, -1]
        if target.heading_tolerance is not None and "theta" in self.state_spec.names:
            theta = trajectories[:, -1, self.state_spec.index("theta")]
            error = np.arctan2(np.sin(theta - target.pose.theta), np.cos(theta - target.pose.theta))
            costs += self.config.heading_weight * error ** 2
        if "v" in self.state_spec.names:
            terminal_v = trajectories[:, -1, self.state_spec.index("v")]
            costs += self.config.terminal_velocity_weight * terminal_v ** 2
        if "omega" in self.state_spec.names:
            terminal_omega = trajectories[:, -1, self.state_spec.index("omega")]
            costs += self.config.terminal_yaw_rate_weight * terminal_omega ** 2
        costs += self.config.control_weight * np.sum(controls ** 2, axis=(1, 2))
        previous = np.broadcast_to(
            self.previous_action[None, None, :],
            (controls.shape[0], 1, controls.shape[2]),
        )
        deltas = np.diff(controls, axis=1, prepend=previous)
        costs += self.config.control_rate_weight * np.sum(deltas ** 2, axis=(1, 2))
        for obstacle in obstacles:
            ox, oy = float(obstacle[0]), float(obstacle[1])
            radius = float(obstacle[2]) if len(obstacle) >= 3 else 0.08
            clearance = np.sqrt((xy[..., 0] - ox) ** 2 + (xy[..., 1] - oy) ** 2)
            clearance -= radius + self.config.robot_radius
            collision = np.any(clearance <= 0.0, axis=1)
            influence = np.maximum(0.0, self.config.obstacle_influence - clearance)
            costs += self.config.obstacle_weight * np.sum(influence ** 2, axis=1)
            costs += self.config.collision_penalty * collision.astype(np.float64)
        if self.memory_cost is not None:
            for index in range(trajectories.shape[0]):
                costs[index] += float(self.memory_cost(trajectories[index], controls[index]))
        return costs

    def plan(self, observation: RobotObservation, reference) -> PlanResult:
        started = time.perf_counter()
        state = self.state_from_observation(observation)
        target = reference.target_at(observation.timestamp, state)
        prior = self._prior(observation, reference)
        samples = self._sample(prior)
        # Effective perturbations include actuator-bound clipping.  Expressing
        # the update this way makes the MPPI control law explicit while
        # remaining numerically equivalent to the historical weighted average.
        perturbations = samples - prior.mean[None, :, :]
        covariance = self._sampling_covariance(prior)
        trajectories = self.rollout(state, samples)
        costs = self._cost(trajectories, samples, target, observation.local_obstacles)
        correction = self._importance_sampling_cost(prior.mean, perturbations, covariance)
        costs = costs + correction
        beta = float(np.min(costs))
        exponent = np.clip(-(costs - beta) / self.config.temperature, -700.0, 0.0)
        weights = np.exp(exponent)
        weights /= max(float(np.sum(weights)), 1e-12)
        weighted_perturbation = np.sum(weights[:, None, None] * perturbations, axis=0)
        sequence = prior.mean + weighted_perturbation
        sequence = np.clip(sequence, self.action_spec.lower, self.action_spec.upper)
        action = self.action_spec.clip(
            sequence[0], self.previous_action, self.config.dt
        )
        sequence[0] = action
        updated_trajectory = self.rollout(state, sequence)[0]
        self.previous_sequence = sequence.copy()
        self.previous_action = action.copy()
        setter = getattr(self.sampling_prior, "set_previous", None)
        if callable(setter):
            setter(sequence)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        return PlanResult(
            proposed_control=ControlCommand(action, observation.timestamp, "mppi"),
            control_sequence=sequence,
            predicted_trajectory=updated_trajectory,
            diagnostics={
                "compute_ms": elapsed_ms,
                "cost_min": float(costs.min()),
                "cost_mean": float(costs.mean()),
                "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
                "importance_sampling_correction": bool(
                    self.config.importance_sampling_correction
                ),
                "importance_cost_mean": float(correction.mean()),
                "weighted_perturbation_norm": float(np.linalg.norm(weighted_perturbation)),
                "sample_saturation_fraction": float(
                    np.mean(
                        (samples <= self.action_spec.lower[None, None, :])
                        | (samples >= self.action_spec.upper[None, None, :])
                    )
                ),
                "reference_id": target.reference_id,
                "prior": dict(prior.metadata),
            },
        )
