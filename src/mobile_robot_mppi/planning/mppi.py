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
    command_delay_s: float = 0.0
    temperature: float = 4.0
    noise_sigma: Tuple[float, ...] = (0.12, 0.35)
    integrator: str = "rk4"
    goal_running_weight: float = 1.0
    goal_terminal_weight: float = 12.0
    heading_weight: float = 0.2
    path_preview_enabled: bool = False
    path_preview_speed_mps: float = 0.45
    path_preview_heading_weight: float = 0.0
    terminal_velocity_weight: float = 0.0
    terminal_yaw_rate_weight: float = 0.0
    terminal_bearing_weight: float = 0.0
    terminal_translation_speed_limit: Optional[float] = None
    terminal_translation_heading_gate_rad: Optional[float] = None
    terminal_alignment_yaw_gain: Optional[float] = None
    terminal_control_radius: Optional[float] = None
    control_weight: float = 0.05
    control_rate_weight: float = 0.08
    obstacle_weight: float = 30.0
    obstacle_influence: float = 0.55
    robot_radius: float = 0.25
    collision_penalty: float = 5000.0
    importance_sampling_correction: bool = False
    previous_sequence_blend: float = 0.5
    safety_recovery_prefix_steps: int = 0
    profile_components: bool = False
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
            command_delay_s=float(values.get("command_delay_s", 0.0)),
            temperature=float(values.get("temperature", 4.0)),
            noise_sigma=tuple(float(v) for v in sigma),
            integrator=str(values.get("integrator", "rk4")),
            goal_running_weight=float(values.get("goal_running_weight", 1.0)),
            goal_terminal_weight=float(values.get("goal_terminal_weight", 12.0)),
            heading_weight=float(values.get("heading_weight", 0.2)),
            path_preview_enabled=bool(values.get("path_preview_enabled", False)),
            path_preview_speed_mps=float(
                values.get("path_preview_speed_mps", 0.45)
            ),
            path_preview_heading_weight=float(
                values.get("path_preview_heading_weight", 0.0)
            ),
            terminal_velocity_weight=float(values.get("terminal_velocity_weight", 0.0)),
            terminal_yaw_rate_weight=float(values.get("terminal_yaw_rate_weight", 0.0)),
            terminal_bearing_weight=float(values.get("terminal_bearing_weight", 0.0)),
            terminal_translation_speed_limit=(
                None
                if values.get("terminal_translation_speed_limit") is None
                else float(values["terminal_translation_speed_limit"])
            ),
            terminal_translation_heading_gate_rad=(
                None
                if values.get("terminal_translation_heading_gate_rad") is None
                else float(values["terminal_translation_heading_gate_rad"])
            ),
            terminal_alignment_yaw_gain=(
                None
                if values.get("terminal_alignment_yaw_gain") is None
                else float(values["terminal_alignment_yaw_gain"])
            ),
            terminal_control_radius=(
                None
                if values.get("terminal_control_radius") is None
                else float(values["terminal_control_radius"])
            ),
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
            profile_components=bool(
                values.get("profile_components", False)
            ),
            seed=int(values.get("seed", 0)),
        )

    def validate(self, action_dim: int) -> None:
        if self.horizon <= 0 or self.num_samples <= 0 or self.dt <= 0.0:
            raise ValueError("MPPI horizon, samples, and dt must be positive")
        if (
            not np.isfinite(self.command_delay_s)
            or self.command_delay_s < 0.0
            or self.command_delay_s > self.dt + 1e-12
        ):
            raise ValueError(
                "command_delay_s must be finite and within one control interval"
            )
        if self.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive")
        if len(self.noise_sigma) != action_dim or any(value <= 0.0 for value in self.noise_sigma):
            raise ValueError("noise_sigma must contain one positive value per action dimension")
        numeric_costs = (
            self.goal_running_weight,
            self.goal_terminal_weight,
            self.heading_weight,
            self.path_preview_heading_weight,
            self.terminal_velocity_weight,
            self.terminal_yaw_rate_weight,
            self.terminal_bearing_weight,
            self.control_weight,
            self.control_rate_weight,
            self.obstacle_weight,
            self.obstacle_influence,
            self.robot_radius,
            self.collision_penalty,
        )
        if not np.isfinite(numeric_costs).all() or any(value < 0.0 for value in numeric_costs):
            raise ValueError("MPPI cost and geometry parameters must be finite and non-negative")
        if (
            not np.isfinite(self.path_preview_speed_mps)
            or self.path_preview_speed_mps <= 0.0
        ):
            raise ValueError("path_preview_speed_mps must be finite and positive")
        if self.integrator not in ("euler", "rk4"):
            raise ValueError("MPPI integrator must be 'euler' or 'rk4'")
        if not 0.0 <= self.previous_sequence_blend <= 1.0:
            raise ValueError("previous_sequence_blend must be between zero and one")
        if self.terminal_translation_speed_limit is not None and (
            not np.isfinite(self.terminal_translation_speed_limit)
            or self.terminal_translation_speed_limit <= 0.0
        ):
            raise ValueError(
                "terminal_translation_speed_limit must be finite and positive"
            )
        if self.terminal_translation_heading_gate_rad is not None and (
            not np.isfinite(self.terminal_translation_heading_gate_rad)
            or not 0.0 < self.terminal_translation_heading_gate_rad <= np.pi
        ):
            raise ValueError(
                "terminal_translation_heading_gate_rad must be in (0, pi]"
            )
        if self.terminal_alignment_yaw_gain is not None and (
            not np.isfinite(self.terminal_alignment_yaw_gain)
            or self.terminal_alignment_yaw_gain <= 0.0
        ):
            raise ValueError(
                "terminal_alignment_yaw_gain must be finite and positive"
            )
        if self.terminal_control_radius is not None and (
            not np.isfinite(self.terminal_control_radius)
            or self.terminal_control_radius <= 0.0
        ):
            raise ValueError(
                "terminal_control_radius must be finite and positive"
            )
        if not 0 <= self.safety_recovery_prefix_steps <= self.horizon:
            raise ValueError("safety_recovery_prefix_steps must be within the horizon")


@dataclass(frozen=True)
class SequenceEvaluation:
    """Predicted trajectories and base costs for fixed control sequences.

    This public result contract is deliberately independent of MPPI sampling
    weights.  It supports paired model diagnostics where nominal, learned and
    physical rollouts must score the exact same commanded sequences.
    """

    trajectories: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class SequenceWeights:
    """Importance-adjusted costs and normalized MPPI sequence weights."""

    adjusted_costs: np.ndarray
    weights: np.ndarray
    effective_sample_size: float


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
        self._reliability_previous_state = None
        self._reliability_pending_control = None
        self._delay_preceding_action = self.previous_action.copy()

    def reset(self, seed=None):
        """Reset receding-horizon state and optionally reseed sampling.

        The optional episode seed is essential for paired experiments: plant,
        sensors and MPPI perturbations must all belong to the same replicate.
        Existing runtime callers that omit it preserve the configured seed.
        """

        resolved_seed = self.config.seed if seed is None else int(seed)
        if not 0 <= resolved_seed <= 2 ** 32 - 1:
            raise ValueError("MPPI reset seed must be in [0, 2**32 - 1]")
        self.rng = np.random.RandomState(resolved_seed)
        self.previous_sequence = None
        self.previous_action = np.zeros(self.action_spec.dimension, dtype=np.float64)
        self._safety_blocked = False
        self._reliability_previous_state = None
        self._reliability_pending_control = None
        self._delay_preceding_action = self.previous_action.copy()
        residual_reset = getattr(
            getattr(self.dynamics, "residual", None), "reset", None
        )
        if callable(residual_reset):
            residual_reset()
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
            output.proposals,
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
        delay_fraction = float(self.config.command_delay_s / self.config.dt)
        self._reliability_pending_control = (
            delay_fraction * self._delay_preceding_action
            + (1.0 - delay_fraction) * executed
        )
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

    def _observe_residual_reliability(self, current_state):
        residual = getattr(self.dynamics, "residual", None)
        ungated = getattr(residual, "ungated_derivative", None)
        nominal = getattr(self.dynamics, "nominal", None)
        if residual is None or nominal is None:
            return
        if (
            self._reliability_previous_state is None
            or self._reliability_pending_control is None
        ):
            return

        previous = self._reliability_previous_state
        control = self._reliability_pending_control
        self.observe_completed_transition(
            previous,
            control,
            current_state,
            residual_derivative=(ungated if callable(ungated) else None),
        )
        self._reliability_pending_control = None

    def observe_completed_transition(
        self,
        previous_state,
        applied_control,
        current_state,
        residual_derivative=None,
    ):
        """Update residual diagnostics from one already completed transition.

        This public hook is shared by online MPPI execution and offline direct
        Actor training.  It is intentionally called only after ``current_state``
        has been measured, so no future plant information can leak into the
        command that generated the transition.
        """

        residual = getattr(self.dynamics, "residual", None)
        nominal = getattr(self.dynamics, "nominal", None)
        if residual is None or nominal is None:
            return False
        observers = []
        seen = set()
        current = residual
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            observer = getattr(current, "observe_prediction_errors", None)
            if callable(observer):
                observers.append(observer)
            child = getattr(current, "residual", None)
            current = None if child is current else child
        if not observers:
            return False
        previous = self.state_spec.validate(previous_state)
        control = np.asarray(applied_control, dtype=np.float64).reshape(-1)
        if (
            control.shape != (self.action_spec.dimension,)
            or not np.isfinite(control).all()
        ):
            raise ValueError("completed-transition control is invalid")
        observed = self.state_spec.validate(current_state)
        derivative = residual_derivative
        if derivative is None:
            derivative = getattr(residual, "ungated_derivative", None)
        if not callable(derivative):
            derivative = residual.derivative

        class _ResidualCombined:
            state_dim = int(self.dynamics.state_dim)
            control_dim = int(self.dynamics.control_dim)

            def derivative(inner_self, state, action, time=None):
                return np.asarray(
                    nominal.derivative(state, action, time), dtype=np.float64
                ) + np.asarray(derivative(state, action, time), dtype=np.float64)

        nominal_prediction = integrate_batch(
            nominal, previous, control, self.config.dt,
            self.state_spec, self.config.integrator,
        )
        residual_prediction = integrate_batch(
            _ResidualCombined(), previous, control, self.config.dt,
            self.state_spec, self.config.integrator,
        )
        nominal_error = nominal_prediction - observed
        residual_error = residual_prediction - observed
        for index in self.state_spec.periodic_indices:
            nominal_error[index] = np.arctan2(
                np.sin(nominal_error[index]), np.cos(nominal_error[index])
            )
            residual_error[index] = np.arctan2(
                np.sin(residual_error[index]), np.cos(residual_error[index])
            )
        for observer in observers:
            observer(nominal_error.copy(), residual_error.copy())
        return True

    def _sample(self, prior: PriorOutput, rng=None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        shape = (self.config.num_samples, self.config.horizon, self.action_spec.dimension)
        if prior.covariance is None:
            noise = rng.normal(size=shape) * np.asarray(self.config.noise_sigma)[None, None, :]
        else:
            covariance = np.asarray(prior.covariance, dtype=np.float64)
            if covariance.shape == (self.action_spec.dimension, self.action_spec.dimension):
                noise = rng.multivariate_normal(
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
        initial_state = self.state_spec.validate(
            np.asarray(initial_state, dtype=np.float64).reshape(-1)
        )
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 2:
            controls = controls[None, ...]
        expected_tail = (self.config.horizon, self.action_spec.dimension)
        if controls.ndim != 3 or controls.shape[1:] != expected_tail:
            raise ValueError(
                "rollout controls must have shape [K,%d,%d]"
                % expected_tail
            )
        if controls.shape[0] < 1 or not np.isfinite(controls).all():
            raise ValueError("rollout controls must contain a finite batch")
        prediction_controls = self._prediction_controls(controls)
        batch = controls.shape[0]
        trajectory = np.empty((batch, self.config.horizon + 1, self.state_spec.dimension), dtype=np.float64)
        trajectory[:, 0, :] = np.repeat(initial_state[None, :], batch, axis=0)
        for step in range(self.config.horizon):
            trajectory[:, step + 1, :] = integrate_batch(
                self.dynamics,
                trajectory[:, step, :],
                prediction_controls[:, step, :],
                self.config.dt,
                self.state_spec,
                self.config.integrator,
            )
        return trajectory

    def evaluate_control_sequences(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        target,
        obstacles: Iterable[Sequence[float]] = (),
        reference=None,
    ) -> SequenceEvaluation:
        """Roll out and score a fixed, bounded batch without sampling it.

        The method is intended for counterfactual and model-ranking studies.
        It neither mutates receding-horizon state nor adds the proposal-density
        importance correction used internally by MPPI.
        """

        values = self._validate_evaluation_controls(controls)
        trajectories = self.rollout(initial_state, values)
        costs = self.cost_trajectories(
            trajectories, values, target, obstacles, reference=reference
        )
        return SequenceEvaluation(trajectories=trajectories, costs=costs)

    def cost_trajectories(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        target,
        obstacles: Iterable[Sequence[float]] = (),
        reference=None,
    ) -> np.ndarray:
        """Apply the controller's base cost contract to fixed trajectories."""

        values = self._validate_evaluation_controls(controls)
        paths = np.asarray(trajectories, dtype=np.float64)
        expected = (
            values.shape[0],
            self.config.horizon + 1,
            self.state_spec.dimension,
        )
        if paths.shape != expected:
            raise ValueError(
                "evaluation trajectories must have shape [K,%d,%d]"
                % (self.config.horizon + 1, self.state_spec.dimension)
            )
        if not np.isfinite(paths).all():
            raise ValueError("evaluation trajectories must be finite")
        costs = np.asarray(
            self._cost(
                paths, values, target, tuple(obstacles), reference=reference
            ),
            dtype=np.float64,
        )
        if costs.shape != (values.shape[0],) or not np.isfinite(costs).all():
            raise FloatingPointError("trajectory cost produced NaN, Inf, or invalid shape")
        return costs

    def _validate_evaluation_controls(self, controls: np.ndarray) -> np.ndarray:
        values = np.asarray(controls, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, ...]
        expected_tail = (self.config.horizon, self.action_spec.dimension)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                "evaluation controls must have shape [K,%d,%d]"
                % expected_tail
            )
        if values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError("evaluation controls must contain a finite batch")
        tolerance = 1e-12
        if (
            np.any(values < self.action_spec.lower[None, None, :] - tolerance)
            or np.any(values > self.action_spec.upper[None, None, :] + tolerance)
        ):
            raise ValueError("evaluation controls exceed action bounds")
        return values

    def importance_weights(
        self,
        nominal: np.ndarray,
        candidates: np.ndarray,
        base_costs: np.ndarray,
        covariance: Optional[np.ndarray] = None,
    ) -> SequenceWeights:
        """Reweight an externally fixed proposal batch using MPPI semantics."""

        values = self._validate_evaluation_controls(candidates)
        nominal = np.asarray(nominal, dtype=np.float64)
        expected_nominal = (self.config.horizon, self.action_spec.dimension)
        if nominal.shape != expected_nominal or not np.isfinite(nominal).all():
            raise ValueError("nominal sequence has invalid shape or values")
        if (
            np.any(nominal < self.action_spec.lower[None, :] - 1e-12)
            or np.any(nominal > self.action_spec.upper[None, :] + 1e-12)
        ):
            raise ValueError("nominal sequence exceeds action bounds")
        costs = np.asarray(base_costs, dtype=np.float64).reshape(-1)
        if costs.shape != (values.shape[0],) or not np.isfinite(costs).all():
            raise ValueError("base costs must be one finite value per candidate")
        covariance = self._sampling_covariance(
            PriorOutput(nominal, covariance, {})
        )
        correction = self._importance_sampling_cost(
            nominal, values - nominal[None, :, :], covariance
        )
        adjusted = costs + correction
        beta = float(np.min(adjusted))
        exponent = np.clip(
            -(adjusted - beta) / self.config.temperature, -700.0, 0.0
        )
        weights = np.exp(exponent)
        weights /= max(float(np.sum(weights)), 1e-12)
        effective_sample_size = float(1.0 / np.sum(weights ** 2))
        if not np.isfinite(weights).all() or not np.isfinite(effective_sample_size):
            raise FloatingPointError("MPPI importance weights are not finite")
        return SequenceWeights(adjusted, weights, effective_sample_size)

    def _prediction_controls(self, controls: np.ndarray) -> np.ndarray:
        """Map commanded sequences to interval-average delayed commands.

        MuJoCo schedules each zero-order-held command after a known delay. For
        delays no longer than one controller interval, the interval average is
        exactly the convex combination of the preceding and current command.
        The preceding value for the first rollout step is the last command that
        passed safety arbitration.
        """

        values = np.asarray(controls, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != self.action_spec.dimension:
            raise ValueError("prediction controls must have shape [K,H,action_dim]")
        fraction = float(self.config.command_delay_s / self.config.dt)
        if fraction <= 1e-12:
            return values
        preceding = np.empty_like(values)
        preceding[:, 0, :] = self.previous_action
        preceding[:, 1:, :] = values[:, :-1, :]
        return fraction * preceding + (1.0 - fraction) * values

    def _cost(self, trajectories, controls, target, obstacles, reference=None):
        xy_indices = self.state_spec.position_indices
        xy = trajectories[..., list(xy_indices)]
        target_xy = np.asarray((target.pose.x, target.pose.y), dtype=np.float64)
        preview = getattr(reference, "preview_poses", None)
        path_preview_active = bool(
            self.config.path_preview_enabled and callable(preview)
        )
        if path_preview_active:
            initial_offset = float(getattr(reference, "lookahead_distance", 0.0))
            offsets = initial_offset + np.arange(
                self.config.horizon + 1, dtype=np.float64
            ) * self.config.dt * self.config.path_preview_speed_mps
            reference_poses = np.asarray(preview(offsets), dtype=np.float64)
            if reference_poses.shape != (self.config.horizon + 1, 3):
                raise ValueError("path preview must have shape [H+1,3]")
            distance_sq = np.sum(
                (xy - reference_poses[None, :, :2]) ** 2, axis=-1
            )
        else:
            reference_poses = None
            distance_sq = np.sum((xy - target_xy) ** 2, axis=-1)
        costs = self.config.goal_running_weight * np.sum(distance_sq[:, 1:-1], axis=1)
        costs += self.config.goal_terminal_weight * distance_sq[:, -1]
        if (
            path_preview_active
            and self.config.path_preview_heading_weight > 0.0
            and "theta" in self.state_spec.names
        ):
            theta = trajectories[:, 1:, self.state_spec.index("theta")]
            heading_error = np.arctan2(
                np.sin(theta - reference_poses[None, 1:, 2]),
                np.cos(theta - reference_poses[None, 1:, 2]),
            )
            costs += self.config.path_preview_heading_weight * np.sum(
                heading_error ** 2, axis=1
            )
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
        if (
            self.config.terminal_bearing_weight > 0.0
            and target.phase in ("terminal_approach", "terminal")
            and "theta" in self.state_spec.names
        ):
            final_dx = target_xy[0] - xy[:, -1, 0]
            final_dy = target_xy[1] - xy[:, -1, 1]
            final_distance = np.hypot(final_dx, final_dy)
            desired_heading = np.arctan2(final_dy, final_dx)
            final_theta = trajectories[:, -1, self.state_spec.index("theta")]
            bearing_error = np.arctan2(
                np.sin(final_theta - desired_heading),
                np.cos(final_theta - desired_heading),
            )
            # At the exact target the bearing is undefined; position success
            # should not impose an arbitrary world-frame orientation.
            bearing_error = np.where(final_distance > 1e-12, bearing_error, 0.0)
            costs += self.config.terminal_bearing_weight * bearing_error ** 2
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

    def _solve_plan(
        self, state, prior, target, obstacles, rng, observation=None,
        reference=None,
    ):
        del observation
        profiling = self.config.profile_components
        solve_started = time.perf_counter() if profiling else None
        stage_started = solve_started
        profile = {}

        def mark(name):
            nonlocal stage_started
            if not profiling:
                return
            now = time.perf_counter()
            profile["profile_mppi_%s_ms" % name] = 1000.0 * (
                now - stage_started
            )
            stage_started = now

        samples = self._sample(prior, rng=rng)
        terminal_dx = float(
            target.pose.x - state[self.state_spec.position_indices[0]]
        )
        terminal_dy = float(
            target.pose.y - state[self.state_spec.position_indices[1]]
        )
        terminal_distance = float(np.hypot(terminal_dx, terminal_dy))
        terminal_control_region_active = bool(
            self.config.terminal_control_radius is None
            or terminal_distance <= self.config.terminal_control_radius
        )
        terminal_heading_gate_active = bool(
            self.config.terminal_translation_heading_gate_rad is not None
            and target.phase in ("terminal_approach", "terminal")
            and terminal_control_region_active
            and "v_cmd" in self.action_spec.names
            and "theta" in self.state_spec.names
        )
        terminal_bearing_error = 0.0
        terminal_translation_scale = 1.0
        if terminal_heading_gate_active:
            theta = float(state[self.state_spec.index("theta")])
            dx = terminal_dx
            dy = terminal_dy
            if np.hypot(dx, dy) > 1e-12:
                desired_heading = float(np.arctan2(dy, dx))
                terminal_bearing_error = float(np.arctan2(
                    np.sin(desired_heading - theta),
                    np.cos(desired_heading - theta),
                ))
                gate = float(self.config.terminal_translation_heading_gate_rad)
                gate_cosine = float(np.cos(gate))
                if abs(terminal_bearing_error) >= gate:
                    terminal_translation_scale = 0.0
                else:
                    # Smooth recovery avoids a stop/go discontinuity as the
                    # body aligns with the terminal target bearing.
                    terminal_translation_scale = max(
                        0.0,
                        (float(np.cos(terminal_bearing_error)) - gate_cosine)
                        / max(1.0 - gate_cosine, 1e-12),
                    )
        terminal_speed_limit_active = bool(
            self.config.terminal_translation_speed_limit is not None
            and target.phase in ("terminal_approach", "terminal")
            and terminal_control_region_active
            and "v_cmd" in self.action_spec.names
        )
        if terminal_speed_limit_active:
            v_index = self.action_spec.index("v_cmd")
            samples[..., v_index] = np.minimum(
                samples[..., v_index],
                self.config.terminal_translation_speed_limit,
            )
        if terminal_heading_gate_active:
            v_index = self.action_spec.index("v_cmd")
            # The bearing gate is a current-step feasibility constraint.  The
            # remaining horizon stays free so MPPI can plan translation after
            # the predicted in-place alignment rather than becoming blind to
            # the value of turning.
            samples[:, 0, v_index] *= terminal_translation_scale
        # Effective perturbations include actuator-bound clipping.  Expressing
        # the update this way makes the MPPI control law explicit while
        # remaining numerically equivalent to the historical weighted average.
        perturbations = samples - prior.mean[None, :, :]
        covariance = self._sampling_covariance(prior)
        mark("sampling")
        trajectories = self.rollout(state, samples)
        mark("batch_rollout")
        costs = self._cost(
            trajectories, samples, target, obstacles, reference=reference
        )
        mark("cost")
        correction = self._importance_sampling_cost(prior.mean, perturbations, covariance)
        costs = costs + correction
        beta = float(np.min(costs))
        exponent = np.clip(-(costs - beta) / self.config.temperature, -700.0, 0.0)
        weights = np.exp(exponent)
        weights /= max(float(np.sum(weights)), 1e-12)
        weighted_perturbation = np.sum(weights[:, None, None] * perturbations, axis=0)
        sequence = prior.mean + weighted_perturbation
        sequence = np.clip(sequence, self.action_spec.lower, self.action_spec.upper)
        if terminal_speed_limit_active:
            sequence[:, v_index] = np.minimum(
                sequence[:, v_index],
                self.config.terminal_translation_speed_limit,
            )
        if terminal_heading_gate_active:
            sequence[0, v_index] *= terminal_translation_scale
        terminal_alignment_active = bool(
            terminal_heading_gate_active
            and self.config.terminal_alignment_yaw_gain is not None
            and "omega_cmd" in self.action_spec.names
        )
        terminal_alignment_omega = 0.0
        if terminal_alignment_active:
            omega_index = self.action_spec.index("omega_cmd")
            terminal_alignment_omega = (
                float(self.config.terminal_alignment_yaw_gain)
                * terminal_bearing_error
            )
            # Keep the local bearing controller authoritative throughout the
            # terminal phase.  Blending back to unconstrained MPPI at small
            # errors caused a limit cycle because position-only rollout costs
            # do not make the instantaneous turn direction identifiable.
            sequence[0, omega_index] = terminal_alignment_omega
        action = self.action_spec.clip(
            sequence[0], self.previous_action, self.config.dt
        )
        if terminal_speed_limit_active:
            action[v_index] = min(
                action[v_index], self.config.terminal_translation_speed_limit
            )
        if terminal_heading_gate_active:
            # Like an emergency translation stop, terminal alignment may need
            # to decelerate faster than the nominal command slew limit.  It
            # never changes angular control and therefore preserves the
            # rotate-in-place degree of freedom.
            action[v_index] *= terminal_translation_scale
        sequence[0] = action
        mark("weighting_update")
        updated_trajectory = self.rollout(state, sequence)[0]
        mark("final_rollout")
        diagnostics = {
            "cost_min": float(costs.min()),
            "cost_mean": float(costs.mean()),
            "cost_std": float(costs.std()),
            "cost_q10": float(np.quantile(costs, 0.10)),
            "cost_q50": float(np.quantile(costs, 0.50)),
            "cost_q90": float(np.quantile(costs, 0.90)),
            "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
            "effective_sample_fraction": float(
                (1.0 / np.sum(weights ** 2)) / self.config.num_samples
            ),
            "importance_sampling_correction": bool(
                self.config.importance_sampling_correction
            ),
            "importance_cost_mean": float(correction.mean()),
            "weighted_perturbation_norm": float(
                np.linalg.norm(weighted_perturbation)
            ),
            "sample_saturation_fraction": float(
                np.mean(
                    (samples <= self.action_spec.lower[None, None, :])
                    | (samples >= self.action_spec.upper[None, None, :])
                )
            ),
            "reference_id": target.reference_id,
            "target_x": float(target.pose.x),
            "target_y": float(target.pose.y),
            "target_theta": float(target.pose.theta),
            "target_is_terminal": bool(target.is_terminal),
            "target_phase": str(target.phase),
            "terminal_speed_limit_active": terminal_speed_limit_active,
            "terminal_translation_speed_limit": (
                0.0
                if self.config.terminal_translation_speed_limit is None
                else float(self.config.terminal_translation_speed_limit)
            ),
            "terminal_heading_gate_active": terminal_heading_gate_active,
            "terminal_translation_heading_gate_rad": (
                0.0
                if self.config.terminal_translation_heading_gate_rad is None
                else float(self.config.terminal_translation_heading_gate_rad)
            ),
            "terminal_bearing_error": terminal_bearing_error,
            "terminal_translation_scale": terminal_translation_scale,
            "terminal_alignment_active": terminal_alignment_active,
            "terminal_alignment_yaw_gain": (
                0.0
                if self.config.terminal_alignment_yaw_gain is None
                else float(self.config.terminal_alignment_yaw_gain)
            ),
            "terminal_alignment_omega": terminal_alignment_omega,
            "terminal_control_radius": (
                0.0
                if self.config.terminal_control_radius is None
                else float(self.config.terminal_control_radius)
            ),
            "terminal_control_distance": terminal_distance,
            "terminal_control_region_active": terminal_control_region_active,
            "prior": dict(prior.metadata),
        }
        residual = getattr(self.dynamics, "residual", None)
        confidence = getattr(residual, "confidence", None)
        if callable(confidence):
            support_controls = self._prediction_controls(
                sequence[None, :, :]
            )[0]
            support = np.asarray(
                confidence(updated_trajectory[:-1], support_controls), dtype=np.float64
            ).reshape(-1)
            diagnostics.update({
                "residual_support_confidence_mean": float(np.mean(support)),
                "residual_support_confidence_min": float(np.min(support)),
                "residual_support_reduced_fraction": float(np.mean(support < 1.0 - 1e-12)),
                "residual_support_disabled_fraction": float(np.mean(support <= 1e-12)),
            })
        else:
            diagnostics.update({
                "residual_support_confidence_mean": 1.0,
                "residual_support_confidence_min": 1.0,
                "residual_support_reduced_fraction": 0.0,
                "residual_support_disabled_fraction": 0.0,
            })
        reliability = getattr(residual, "diagnostics", None)
        if callable(reliability):
            diagnostics.update(reliability())
        else:
            diagnostics.update({
                "residual_reliability_enabled": False,
                "residual_reliability_alpha": 1.0 if residual is not None else 0.0,
                "residual_reliability_evidence_alpha": 1.0 if residual is not None else 0.0,
                "residual_reliability_context_alpha": 1.0,
                "residual_reliability_context_value": 0.0,
                "residual_reliability_samples": 0,
                "residual_reliability_mean_improvement": 0.0,
                "residual_reliability_lcb": 0.0,
                "residual_reliability_last_relative_improvement": 0.0,
                "residual_reliability_nominal_error": 0.0,
                "residual_reliability_residual_error": 0.0,
            })
        if profiling:
            profile["profile_mppi_solve_total_ms"] = 1000.0 * (
                time.perf_counter() - solve_started
            )
            diagnostics.update(profile)
        return action, sequence, updated_trajectory, diagnostics

    def preview_plan(self, observation: RobotObservation, reference) -> PlanResult:
        """Evaluate the next MPPI plan without advancing controller state.

        A cloned RNG supplies common random numbers for paired candidate-prior
        comparisons.  The real controller RNG, receding-horizon sequence and
        previous action are not changed.  Stateful prior/reference restoration
        is handled by the offline environment wrapper that owns those objects.
        """

        started = time.perf_counter()
        state = self.state_from_observation(observation)
        target = reference.target_at(observation.timestamp, state)
        state_reference_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        prior = self._prior(observation, reference)
        prior_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        preview_rng = np.random.RandomState()
        preview_rng.set_state(self.rng.get_state())
        action, sequence, trajectory, diagnostics = self._solve_plan(
            state,
            prior,
            target,
            observation.local_obstacles,
            preview_rng,
            observation,
            reference,
        )
        diagnostics["compute_ms"] = 1000.0 * (
            time.perf_counter() - started
        )
        if self.config.profile_components:
            diagnostics.update({
                "profile_planner_state_reference_ms": 1000.0 * (
                    state_reference_finished - started
                ),
                "profile_planner_prior_ms": 1000.0 * (
                    prior_finished - state_reference_finished
                ),
            })
        diagnostics["preview"] = True
        return PlanResult(
            proposed_control=ControlCommand(action, observation.timestamp, "mppi_preview"),
            control_sequence=sequence,
            predicted_trajectory=trajectory,
            diagnostics=diagnostics,
        )

    def plan(self, observation: RobotObservation, reference) -> PlanResult:
        started = time.perf_counter()
        state = self.state_from_observation(observation)
        self._observe_residual_reliability(state)
        self._reliability_previous_state = state.copy()
        target = reference.target_at(observation.timestamp, state)
        state_reference_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        prior = self._prior(observation, reference)
        prior_finished = (
            time.perf_counter()
            if self.config.profile_components else None
        )
        action, sequence, updated_trajectory, diagnostics = self._solve_plan(
            state,
            prior,
            target,
            observation.local_obstacles,
            self.rng,
            observation,
            reference,
        )
        self.previous_sequence = sequence.copy()
        self._delay_preceding_action = self.previous_action.copy()
        self.previous_action = action.copy()
        setter = getattr(self.sampling_prior, "set_previous", None)
        if callable(setter):
            setter(sequence)
        diagnostics["compute_ms"] = 1000.0 * (
            time.perf_counter() - started
        )
        if self.config.profile_components:
            diagnostics.update({
                "profile_planner_state_reference_ms": 1000.0 * (
                    state_reference_finished - started
                ),
                "profile_planner_prior_ms": 1000.0 * (
                    prior_finished - state_reference_finished
                ),
            })
        diagnostics["preview"] = False
        return PlanResult(
            proposed_control=ControlCommand(action, observation.timestamp, "mppi"),
            control_sequence=sequence,
            predicted_trajectory=updated_trajectory,
            diagnostics=diagnostics,
        )
