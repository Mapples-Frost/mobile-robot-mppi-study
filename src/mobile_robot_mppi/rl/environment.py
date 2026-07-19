"""MuJoCo environment where RL steers MPPI sampling, never final control."""

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from mobile_robot_mppi.core.spaces import action_spec_from_config
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.runtime.factories import make_components
from .intrinsic import EpisodicPoseCountBonus
from .observation import ObservationEncoder, ObservationEncoderConfig
from .parameterization import PriorParameterization, PriorParameterizationConfig
from .prior import ExternalActionPrior


@dataclass(frozen=True)
class RewardConfig:
    progress_mode: str = "discounted_potential"
    potential_progress_weight: float = 8.0
    distance_penalty_weight: float = 0.0
    path_length_penalty_weight: float = 0.0
    cross_track_penalty_weight: float = 0.0
    path_progress_weight: float = 0.0
    path_progress_corridor: float = 0.0
    heading_error_penalty_weight: float = 0.0
    step_penalty: float = 0.02
    goal_bonus: float = 100.0
    collision_penalty: float = 100.0
    clearance_margin: float = 0.45
    clearance_penalty_weight: float = 1.5
    safety_override_penalty: float = 0.15
    control_effort_weight: float = 0.02
    control_rate_weight: float = 0.04
    stuck_penalty: float = 0.03

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        numeric = {
            name: float(values.get(name, field.default))
            for name, field in cls.__dataclass_fields__.items()
            if name != "progress_mode"
        }
        return cls(
            progress_mode=str(values.get("progress_mode", "discounted_potential")),
            **numeric
        )

    def validate(self):
        if self.progress_mode not in ("discounted_potential", "distance_delta"):
            raise ValueError(
                "RL progress_mode must be discounted_potential or distance_delta"
            )
        values = tuple(
            value for name, value in asdict(self).items() if name != "progress_mode"
        )
        if not np.isfinite(values).all() or any(value < 0.0 for value in values):
            raise ValueError("RL reward weights and margins must be finite and non-negative")


def _progress_signal(previous_distance, distance, gamma, mode):
    """Return the configured dense goal-progress signal before weighting.

    ``discounted_potential`` preserves the original potential-based shaping
    expression.  ``distance_delta`` is intentionally more literal for finite
    robotics experiments: standing still earns zero, approaching is positive,
    and moving away is negative.  Keeping both modes makes old checkpoints and
    experiment configurations reproducible.
    """
    if mode == "discounted_potential":
        return float(previous_distance - gamma * distance)
    if mode == "distance_delta":
        return float(previous_distance - distance)
    raise ValueError("unknown RL progress mode: %s" % mode)


def _final_target(reference, initial_pose):
    pose_type = type(initial_pose)
    if hasattr(reference, "waypoints"):
        values = reference.waypoints[-1]
        theta = values[2] if len(values) >= 3 else initial_pose.theta
        return pose_type(values[0], values[1], theta)
    if hasattr(reference, "points"):
        values = reference.points[-1]
        return pose_type(values[0], values[1], initial_pose.theta)
    if hasattr(reference, "poses"):
        values = reference.poses[-1]
        return pose_type(values[0], values[1], values[2])
    return reference.target_at(0.0, initial_pose.as_array()).pose


def _cross_track_error(reference, pose):
    """Return distance to a configured polyline, or zero when unavailable."""

    points = getattr(reference, "points", None)
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("reference points must contain at least two x/y pairs")
    position = np.asarray((pose.x, pose.y), dtype=np.float64)
    starts = points[:-1, :2]
    segments = points[1:, :2] - starts
    squared_lengths = np.sum(segments ** 2, axis=1)
    if np.any(squared_lengths <= 1e-18):
        raise ValueError("reference polyline contains a zero-length segment")
    fractions = np.sum((position[None, :] - starts) * segments, axis=1)
    fractions = np.clip(fractions / squared_lengths, 0.0, 1.0)
    projections = starts + fractions[:, None] * segments
    return float(
        np.min(np.linalg.norm(projections - position[None, :], axis=1))
    )


def _path_tracking_state(reference, pose):
    """Return read-only path metrics, or neutral values for point goals."""

    project = getattr(reference, "project", None)
    if not callable(project):
        return {
            "valid": False,
            "progress": 0.0,
            "remaining": 0.0,
            "cross_track_error": 0.0,
            "signed_cross_track_error": 0.0,
            "heading_error": 0.0,
            "curvature": 0.0,
        }
    projection = project(
        np.asarray((pose.x, pose.y), dtype=np.float64),
        minimum_progress=getattr(reference, "progress", None),
    )
    heading_error = float(np.arctan2(
        np.sin(float(pose.theta) - projection.tangent_heading),
        np.cos(float(pose.theta) - projection.tangent_heading),
    ))
    return {
        "valid": True,
        "progress": float(projection.progress),
        "remaining": float(projection.remaining),
        "cross_track_error": float(projection.cross_track_error),
        "signed_cross_track_error": float(
            projection.signed_cross_track_error
        ),
        "heading_error": heading_error,
        "curvature": float(projection.curvature),
    }


def _resolved_prior_mapping(values, plant_config, control_dt):
    """Resolve decoder timing from the same plant config used by MuJoCo.

    The resolved numeric values are written back into the environment config
    and checkpoint.  This prevents a learned prior from silently using timing
    constants that differ from the nominal five-state MPPI model.
    """
    resolved = dict(values or {})
    if str(resolved.get("kind", "control_knots")) != "local_subgoal":
        return resolved
    if resolved.get("subgoal_control_dt") is None:
        resolved["subgoal_control_dt"] = float(control_dt)
    if str(resolved.get("subgoal_decoder", "kinematic")) != "dynamic_first_order":
        return resolved
    plant = dict(plant_config or {})
    actuator = dict(plant.get("actuator", {}))
    inherited = {
        "subgoal_velocity_time_constant": plant.get(
            "nominal_velocity_time_constant", 0.18
        ),
        "subgoal_yaw_time_constant": plant.get(
            "nominal_yaw_time_constant", 0.12
        ),
        "subgoal_command_delay": actuator.get("command_delay", 0.0),
    }
    for name, value in inherited.items():
        if resolved.get(name) is None:
            resolved[name] = float(value)
    return resolved


class MppiPriorEnv:
    """Small Gym-like API for SAC training around the complete control stack.

    ``step(policy_action)`` decodes the policy action into an MPPI prior,
    executes MPPI, then the unchanged scan guard, then MuJoCo.  Observations
    come from odometry and LaserScan; simulator obstacle truth is used only for
    reward/metrics such as collision and clearance.
    """

    def __init__(self, config, project_root, seed=None):
        self.config = copy.deepcopy(dict(config))
        self.project_root = project_root
        self.seed = int(
            self.config["experiment"].get("seed", 0) if seed is None else seed
        )
        rl_config = dict(self.config.get("rl", {}))
        if not bool(rl_config.get("enabled", False)):
            raise ValueError("RL training environment requires rl.enabled=true")
        sensor_overrides = dict(rl_config.get("sensor_overrides", {}))
        if sensor_overrides:
            # Keep localization as an explicit experimental variable.  This
            # override is stored in the resolved config and never changes the
            # LaserScan/local-obstacle data path.
            self.config.setdefault("sensors", {}).update(sensor_overrides)
        self.action_spec = action_spec_from_config(self.config["action_space"])
        planner = dict(self.config["planner"])
        observation_config = ObservationEncoderConfig.from_mapping(
            rl_config.get("observation", {})
        )
        self.encoder = ObservationEncoder(observation_config, self.action_spec)
        parameter_mapping = _resolved_prior_mapping(
            rl_config.get("prior", {}),
            self.config.get("plant", {}),
            self.config["experiment"]["control_dt"],
        )
        parameter_config = PriorParameterizationConfig.from_mapping(
            parameter_mapping
        )
        self.config.setdefault("rl", {})["prior"] = parameter_config.to_dict()
        self.parameterization = PriorParameterization(
            parameter_config,
            self.action_spec,
            planner.get("noise_sigma", (0.12, 0.35)),
        )
        fallback = GoalWarmStartPrior(
            planner.get("prior_v_gain", 0.8),
            planner.get("prior_yaw_gain", 1.2),
            planner.get("prior_translation_heading_gate_rad"),
            planner.get("prior_translation_heading_gate_terminal_only", False),
            planner.get("prior_terminal_max_speed"),
        )
        training_config = dict(rl_config.get("training", {}))
        self.external_prior = ExternalActionPrior(
            self.parameterization,
            fallback_prior=fallback,
            gate_alpha=float(training_config.get("rl_prior_alpha", 1.0)),
            gate_config=rl_config.get("gate", {}),
        )
        self.config["planner"]["sampling_prior"] = "rl"
        self.components = make_components(
            self.config, self.project_root, rl_policy=self.external_prior
        )
        self.residual_context = None
        if self.encoder.config.include_residual_context:
            from .residual_context import ResidualContextEncoder

            context_mapping = dict(rl_config.get("residual_context", {}))
            context_mapping["enabled"] = True
            self.residual_context = ResidualContextEncoder(
                self.components["controller"].dynamics,
                self.components["state_spec"],
                context_mapping,
            )
            if (
                self.residual_context.dimension
                != self.encoder.config.residual_context_dimension
            ):
                raise ValueError(
                    "RL observation and residual context dimensions differ"
                )
        self.reward_config = RewardConfig.from_mapping(rl_config.get("reward", {}))
        self.reward_config.validate()
        self.intrinsic_exploration = EpisodicPoseCountBonus(
            rl_config.get("intrinsic_exploration", {})
        )
        self.config.setdefault("rl", {})[
            "intrinsic_exploration"
        ] = self.intrinsic_exploration.config.to_dict()
        self.gamma = float(dict(rl_config.get("sac", {})).get("gamma", 0.99))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("RL environment gamma must be in [0, 1]")
        self.initial_state_noise = np.asarray(
            training_config.get(
                "initial_state_noise",
                np.zeros(len(self.config["experiment"].get("initial_state", (0, 0, 0)))),
            ),
            dtype=np.float64,
        ).reshape(-1)
        self.rng = np.random.RandomState(self.seed)
        self.observation = None
        self.perceived = None
        self.truth = None
        self.final_target = None
        self.steps = 0
        self.previous_distance = None
        self.previous_path_progress = None
        self.previous_control = np.zeros(self.action_spec.dimension, dtype=np.float64)
        self.last_safety_override = False
        self.action_mode = "mppi_prior"

    @property
    def observation_dim(self):
        return self.encoder.dimension

    @property
    def policy_action_dim(self):
        return self.parameterization.parameter_dimension

    def _encoded_observation(self):
        residual_context = None
        if self.residual_context is not None:
            state = self.components["controller"].state_from_observation(
                self.perceived.observation
            )
            residual_context = self.residual_context.features(
                state, self.previous_control
            )
        return self.encoder.encode(
            self.perceived.observation,
            self.components["reference"],
            previous_action=self.previous_control,
            safety_override=self.last_safety_override,
            residual_context=residual_context,
        )

    def _distance_to_final(self, truth):
        return float(math.hypot(
            self.final_target.x - truth.pose.x,
            self.final_target.y - truth.pose.y,
        ))

    def reset(self, seed=None, initial_state=None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.RandomState(self.seed)
        experiment = self.config["experiment"]
        configured_initial = experiment.get("initial_state", (0.0, 0.0, 0.0))
        initial = np.asarray(
            configured_initial if initial_state is None else initial_state,
            dtype=np.float64,
        ).reshape(-1)
        if initial.shape != np.asarray(configured_initial, dtype=np.float64).reshape(-1).shape:
            raise ValueError(
                "RL reset initial_state must match the configured state dimension"
            )
        if not np.isfinite(initial).all():
            raise ValueError("RL reset initial_state must be finite")
        if self.initial_state_noise.size not in (0, initial.size):
            raise ValueError("RL initial_state_noise must match initial_state dimension")
        if self.initial_state_noise.size:
            initial = initial + self.rng.uniform(-1.0, 1.0, size=initial.size) * self.initial_state_noise
        self.truth = self.components["plant"].reset(self.seed, initial)
        self.observation = self.components["sensors"].reset(self.truth, self.seed)
        intrinsic_diagnostics = self.intrinsic_exploration.reset(
            self.observation.pose
        )
        reset_reference = getattr(self.components["reference"], "reset", None)
        if callable(reset_reference):
            reset_reference()
        # The episode seed owns every stochastic component, including MPPI
        # perturbation sampling.  Previously the controller silently reset to
        # the training-config seed, so nominally fixed validation seeds did
        # not reproduce the standard evaluation path.
        self.components["controller"].reset(seed=self.seed)
        if self.residual_context is not None:
            self.residual_context.reset()
        self.encoder.reset()
        perception_reset = getattr(self.components["perception"], "reset", None)
        if callable(perception_reset):
            perception_reset()
        memory = self.components.get("memory")
        if memory is not None:
            memory.reset()
        self.perceived = self.components["perception"].process(self.observation)
        self.final_target = _final_target(self.components["reference"], self.truth.pose)
        self.steps = 0
        self.previous_distance = self._distance_to_final(self.truth)
        self.previous_control.fill(0.0)
        self.last_safety_override = False
        encoded_observation = self._encoded_observation()
        path_state = _path_tracking_state(
            self.components["reference"], self.truth.pose
        )
        self.previous_path_progress = float(path_state["progress"])
        return encoded_observation, {
            "seed": self.seed,
            "scene": self.config.get("scene", {}).get("name", "unknown"),
            "goal_distance": self.previous_distance,
            "path_tracking": path_state,
            "intrinsic_exploration": intrinsic_diagnostics,
        }

    def _reward(
        self,
        distance,
        decision,
        truth,
        terminated_success,
        intrinsic_bonus=0.0,
        path_state=None,
    ):
        cfg = self.reward_config
        progress = _progress_signal(
            self.previous_distance, distance, self.gamma, cfg.progress_mode
        )
        control_dt = float(self.config["experiment"]["control_dt"])
        if path_state is None:
            path_state = _path_tracking_state(
                self.components["reference"], truth.pose
            )
        cross_track_error = float(path_state["cross_track_error"])
        path_progress = 0.0
        if bool(path_state["valid"]) and self.previous_path_progress is not None:
            path_progress = float(
                path_state["progress"] - self.previous_path_progress
            )
        corridor_weight = 1.0
        if cfg.path_progress_corridor > 0.0:
            corridor_weight = float(np.exp(
                -(cross_track_error / cfg.path_progress_corridor) ** 2
            ))
        terms = {
            "potential_progress": cfg.potential_progress_weight
            * progress,
            "distance": -cfg.distance_penalty_weight * distance,
            "path_length": -cfg.path_length_penalty_weight
            * abs(float(truth.twist.v))
            * control_dt,
            "cross_track": -cfg.cross_track_penalty_weight
            * cross_track_error ** 2,
            "path_progress": cfg.path_progress_weight
            * path_progress
            * corridor_weight,
            "heading_error": -cfg.heading_error_penalty_weight
            * float(path_state["heading_error"]) ** 2,
            "step": -cfg.step_penalty,
            "goal": cfg.goal_bonus if terminated_success else 0.0,
            "collision": -cfg.collision_penalty if truth.collision else 0.0,
            "clearance": 0.0,
            "safety_override": -cfg.safety_override_penalty if decision.overridden else 0.0,
            "control_effort": -cfg.control_effort_weight
            * float(np.dot(decision.executed_control.values, decision.executed_control.values)),
            "control_rate": -cfg.control_rate_weight
            * float(np.dot(
                decision.executed_control.values - self.previous_control,
                decision.executed_control.values - self.previous_control,
            )),
            "stuck": 0.0,
            "intrinsic_exploration": float(intrinsic_bonus),
        }
        if math.isfinite(truth.minimum_clearance):
            violation = max(0.0, cfg.clearance_margin - truth.minimum_clearance)
            terms["clearance"] = -cfg.clearance_penalty_weight * violation ** 2
        if abs(truth.twist.v) < 0.02 and distance > float(self.config["task"].get("position_tolerance", 0.2)):
            terms["stuck"] = -cfg.stuck_penalty
        reward = float(sum(terms.values()))
        if not np.isfinite(reward):
            raise FloatingPointError("RL reward produced NaN or Inf")
        return reward, terms

    def preview_policy_action(self, policy_action):
        """Return a side-effect-free MPPI plan for one latent prior action.

        This is an offline research interface used to compare BC and learned
        candidate trajectories at the same physical state with common MPPI
        random numbers.  It does not execute control, consume the controller
        RNG, advance the receding-horizon sequence, or change the reference.
        """

        if self.perceived is None:
            raise RuntimeError("RL environment must be reset before preview")
        controller = self.components["controller"]
        reference = self.components["reference"]
        previous_parameters = self.external_prior.parameters.copy()
        reference_state = copy.deepcopy(getattr(reference, "__dict__", {}))
        try:
            self.external_prior.set_parameters(policy_action)
            return controller.preview_plan(
                self.perceived.observation, reference
            )
        finally:
            self.external_prior.parameters = previous_parameters
            if hasattr(reference, "__dict__"):
                reference.__dict__.clear()
                reference.__dict__.update(reference_state)

    def _advance_control(
        self,
        proposed_control,
        planner_compute_ms=0.0,
        prior_diagnostics=None,
        notify_controller=False,
    ):
        """Apply one proposed command through safety and the true plant.

        This shared execution path is deliberately below both policy action
        modes. The legacy mode obtains ``proposed_control`` from MPPI; the
        paper-faithful offline Actor mode obtains it directly from the policy.
        Both retain the same scan guard, safety arbitration, MuJoCo plant,
        sensor, perception, memory, reward, and termination semantics.
        """

        if self.perceived is None:
            raise RuntimeError("RL environment must be reset before step")
        if not isinstance(proposed_control, ControlCommand):
            raise TypeError("proposed_control must be a ControlCommand")
        controller = self.components["controller"]
        reference = self.components["reference"]
        previous_prediction_state = controller.state_from_observation(
            self.perceived.observation
        )
        decision = self.components["safety"].arbitrate(
            proposed_control, self.perceived.guard
        )
        if notify_controller:
            controller.observe_safety_decision(decision)
        plant_step = self.components["plant"].step(
            decision.executed_control,
            float(self.config["experiment"]["control_dt"]),
        )
        self.truth = plant_step.ground_truth
        self.observation = self.components["sensors"].observe(self.truth)
        intrinsic_bonus, intrinsic_diagnostics = self.intrinsic_exploration.observe(
            self.observation.pose
        )
        memory = self.components.get("memory")
        if memory is not None:
            memory.update(
                self.observation, reference, decision.executed_control, self.perceived.guard
            )
        self.perceived = self.components["perception"].process(self.observation)
        self.steps += 1
        distance = self._distance_to_final(self.truth)
        state = self.truth.pose.as_array()
        current_target = reference.target_at(self.truth.timestamp, state)
        path_state = _path_tracking_state(reference, self.truth.pose)
        current_distance = math.hypot(
            current_target.pose.x - self.truth.pose.x,
            current_target.pose.y - self.truth.pose.y,
        )
        success = bool(
            current_target.is_terminal
            and current_distance <= current_target.position_tolerance
            and not self.truth.collision
        )
        terminate_collision = bool(
            self.truth.collision
            and self.config["experiment"].get("terminate_on_collision", True)
        )
        terminated = bool(success or terminate_collision)
        truncated = bool(
            self.steps >= int(self.config["experiment"]["max_steps"])
            and not terminated
        )
        reward, reward_terms = self._reward(
            distance,
            decision,
            self.truth,
            success,
            intrinsic_bonus=intrinsic_bonus,
            path_state=path_state,
        )
        applied = np.asarray(
            plant_step.metadata.get(
                "average_applied_control", decision.executed_control.values
            ),
            dtype=np.float64,
        )
        delay_fraction = float(
            controller.config.command_delay_s / controller.config.dt
        )
        prediction_control = (
            delay_fraction * self.previous_control
            + (1.0 - delay_fraction) * decision.executed_control.values
        )
        if not notify_controller:
            controller.observe_completed_transition(
                previous_prediction_state,
                prediction_control,
                controller.state_from_observation(self.perceived.observation),
            )
        self.previous_distance = distance
        self.previous_path_progress = float(path_state["progress"])
        self.previous_control = decision.executed_control.values.copy()
        self.last_safety_override = bool(decision.overridden)
        next_observation = self._encoded_observation()
        info = {
            "success": success,
            "collision": bool(self.truth.collision),
            "goal_distance": distance,
            "minimum_clearance": float(self.truth.minimum_clearance),
            "safety_override": bool(decision.overridden),
            "safety_reason": decision.reason,
            "proposed_control": proposed_control.values.copy(),
            "executed_control": decision.executed_control.values.copy(),
            "applied_control": applied,
            "planner_compute_ms": float(planner_compute_ms),
            "cross_track_error": float(path_state["cross_track_error"]),
            "signed_cross_track_error": float(
                path_state["signed_cross_track_error"]
            ),
            "path_progress": float(path_state["progress"]),
            "path_remaining": float(path_state["remaining"]),
            "path_heading_error": float(path_state["heading_error"]),
            "path_curvature": float(path_state["curvature"]),
            "prior": dict(prior_diagnostics or {}),
            "action_mode": self.action_mode,
            "reward_terms": reward_terms,
            "intrinsic_exploration": intrinsic_diagnostics,
            "scene": self.config.get("scene", {}).get("name", "unknown"),
        }
        return next_observation, reward, terminated, truncated, info

    def step(self, policy_action):
        if self.perceived is None:
            raise RuntimeError("RL environment must be reset before step")
        self.external_prior.set_parameters(policy_action)
        controller = self.components["controller"]
        reference = self.components["reference"]
        plan = controller.plan(self.perceived.observation, reference)
        return self._advance_control(
            plan.proposed_control,
            planner_compute_ms=float(plan.diagnostics.get("compute_ms", 0.0)),
            prior_diagnostics=plan.diagnostics.get("prior", {}),
            notify_controller=True,
        )

    def close(self):
        self.components["plant"].close()


class DirectControlEnv(MppiPriorEnv):
    """Offline low-level Actor environment for paper-faithful RL-Driven MPPI.

    The SAC action remains normalized to ``[-1, 1]`` internally. It is mapped
    bijectively to the configured physical action space and rate-limited before
    the unchanged safety arbiter and true plant. The Actor is trained here;
    during online evaluation it only proposes sequences to MPPI and never
    bypasses MPPI or safety.
    """

    def __init__(self, config, project_root, seed=None):
        super().__init__(config, project_root, seed=seed)
        self.action_mode = "direct_control"
        self.config.setdefault("rl", {}).setdefault("training", {})[
            "action_mode"
        ] = self.action_mode

    @property
    def policy_action_dim(self):
        return self.action_spec.dimension

    def normalized_to_physical_action(self, policy_action):
        values = np.asarray(policy_action, dtype=np.float64).reshape(-1)
        if (
            values.shape != (self.action_spec.dimension,)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "direct-control policy action must be finite with shape (%d,)"
                % self.action_spec.dimension
            )
        if np.any(values < -1.000001) or np.any(values > 1.000001):
            raise ValueError("direct-control policy action must lie in [-1, 1]")
        values = np.clip(values, -1.0, 1.0)
        center = 0.5 * (self.action_spec.lower + self.action_spec.upper)
        half_range = 0.5 * (self.action_spec.upper - self.action_spec.lower)
        physical = center + half_range * values
        return self.action_spec.clip(
            physical,
            previous=self.previous_control,
            dt=float(self.config["experiment"]["control_dt"]),
        )

    def preview_policy_action(self, policy_action):
        raise RuntimeError(
            "direct-control Actor actions do not invoke MPPI during offline training"
        )

    def step(self, policy_action):
        if self.perceived is None:
            raise RuntimeError("RL environment must be reset before step")
        physical = self.normalized_to_physical_action(policy_action)
        proposed = ControlCommand(
            physical,
            timestamp=float(self.perceived.observation.timestamp),
            source="offline_rl_actor",
        )
        return self._advance_control(
            proposed,
            planner_compute_ms=0.0,
            prior_diagnostics={
                "kind": "direct_control",
                "normalized_action": np.asarray(
                    policy_action, dtype=np.float64
                ).reshape(-1).copy(),
            },
            notify_controller=False,
        )
