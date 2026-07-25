from copy import deepcopy
from pathlib import Path

from mobile_robot_mppi.core.references import reference_from_config
from mobile_robot_mppi.core.spaces import action_spec_from_config, state_spec_from_config
from mobile_robot_mppi.memory.legacy_adapter import LegacyMemoryAdapter
from mobile_robot_mppi.perception.legacy_pipeline import LegacyScanPipeline
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    BatchCompatibleResidual,
    GainBiasedUnicyclePrediction,
    LegacyUnicyclePrediction,
    OracleResidualPrediction,
    ResidualPrediction,
)
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.policies.priors import (
    GoalWarmStartPrior,
    HybridBaselinePrior,
    PreviousSequencePrior,
    RLPolicyPrior,
    ZeroPrior,
)
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter
from mobile_robot_mppi.simulation.kinematic import KinematicPlant
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant
from mobile_robot_mppi.simulation.sensors import SimulatedSensorSuite


def _build_frozen_hss_sidecar(mapping, project_root, state_spec, action_spec):
    """Load an explicit residual ensemble used only as causal HSS evidence."""

    values = dict(mapping or {})
    if not values:
        return None
    allowed = {
        "contract",
        "checkpoints",
        "device",
        "torch_num_threads",
        "use_torchscript",
        "ensemble",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            "unknown frozen HSS sidecar fields: %s" % ", ".join(unknown)
        )
    if str(values.get("contract", "")) != "frozen_l217_value_hss_v1":
        raise ValueError(
            "paper HSS sidecar requires contract=frozen_l217_value_hss_v1"
        )
    checkpoints = values.get("checkpoints")
    if not isinstance(checkpoints, (list, tuple)) or len(checkpoints) < 2:
        raise ValueError("paper HSS sidecar requires at least two checkpoints")

    import torch
    from mobile_robot_mppi.learning.models import (
        PlatformResidualDynamics,
        PlatformResidualEnsemble,
    )

    requested_device = str(values.get("device", "cpu"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device == "cpu":
        thread_count = int(values.get("torch_num_threads", 1))
        if thread_count <= 0:
            raise ValueError("HSS sidecar torch_num_threads must be positive")
        torch.set_num_threads(thread_count)
    paths = []
    for checkpoint in checkpoints:
        path = Path(str(checkpoint))
        if not path.is_absolute():
            path = Path(project_root) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError("HSS sidecar checkpoint missing: %s" % path)
        paths.append(path)
    members = [
        PlatformResidualDynamics.from_checkpoint(
            path,
            device=device,
            use_torchscript=bool(values.get("use_torchscript", True)),
        )
        for path in paths
    ]
    if any(member.model.model_type != "icode_residual" for member in members):
        raise ValueError("HSS sidecar checkpoints must be ICODE residual models")
    if any(
        int(member.state_dim) != state_spec.dimension
        or int(member.control_dim) != action_spec.dimension
        for member in members
    ):
        raise ValueError("HSS sidecar checkpoint dimensions do not match")
    ensemble = dict(values.get("ensemble", {}))
    return PlatformResidualEnsemble(
        members,
        state_scales=ensemble.get(
            "state_scales", [1.0] * state_spec.dimension
        ),
        disagreement_scales=ensemble.get("disagreement_scales"),
        innovation_scales=ensemble.get("innovation_scales"),
        support_soft_z=float(ensemble.get("support_soft_z", 3.0)),
        support_hard_z=float(ensemble.get("support_hard_z", 7.0)),
        innovation_decay=float(ensemble.get("innovation_decay", 0.9)),
        member_paths=paths,
    )


def make_components(config, project_root, rl_policy=None):
    state_spec = state_spec_from_config(config["state_space"])
    action_spec = action_spec_from_config(config["action_space"])
    reference = reference_from_config(config["task"])
    plant_cfg = dict(config["plant"])
    scene = deepcopy(dict(config.get("scene", {})))
    for obstacle in scene.get("obstacles", ()):
        motion = obstacle.get("motion")
        if (
            isinstance(motion, dict)
            and str(motion.get("type"))
            == "recurrent_semimarkov_v3"
        ):
            config_path = Path(str(motion["config_path"]))
            if not config_path.is_absolute():
                config_path = Path(project_root) / config_path
            motion["config_path"] = str(config_path.resolve())
    backend = str(plant_cfg.get("backend", "legacy_kinematic"))
    if backend == "legacy_kinematic":
        kinematic_cfg = dict(plant_cfg)
        kinematic_cfg["dynamic"] = state_spec.dimension >= 5
        obstacles = []
        for item in scene.get("obstacles", ()):
            if str(item.get("type", "cylinder")) == "cylinder":
                obstacles.append((item["position"][0], item["position"][1], item.get("radius", 0.25)))
        kinematic_cfg["obstacles"] = obstacles
        plant = KinematicPlant(kinematic_cfg)
    elif backend == "mujoco_diff_drive":
        plant = MujocoDiffDrivePlant(plant_cfg, scene)
    else:
        raise ValueError("unknown plant backend: %s" % backend)
    sensor_cfg = dict(config.get("sensors", {}))
    sensor_cfg.setdefault("wheel_radius", plant_cfg.get("robot", {}).get("wheel_radius", 0.08))
    sensor_cfg.setdefault("track_width", plant_cfg.get("robot", {}).get("track_width", 0.32))
    sensors = SimulatedSensorSuite(plant, sensor_cfg, int(config["experiment"].get("seed", 0)))
    if state_spec.dimension == 3:
        dynamics = LegacyUnicyclePrediction()
    elif state_spec.dimension in (5, 7):
        dynamics = DynamicUnicyclePrediction(
            plant_cfg.get("nominal_velocity_time_constant", 0.18),
            plant_cfg.get("nominal_yaw_time_constant", 0.12),
        )
        if state_spec.dimension == 7:
            raise NotImplementedError("wheel_augmented_7 requires an explicit seven-state prediction model")
    else:
        raise ValueError("no default prediction model for state dimension %d" % state_spec.dimension)
    nominal_dynamics = dynamics
    planner_cfg = dict(config["planner"])
    prediction_mode = str(planner_cfg.get("prediction_mode", "nominal"))
    if prediction_mode == "oracle_residual":
        if backend != "legacy_kinematic" or state_spec.dimension != 3:
            raise ValueError(
                "oracle_residual is only honest for the configured low-order kinematic plant; "
                "MuJoCo is a hidden-state plant and cannot be exposed to ordinary MPPI"
            )
        true_dynamics = GainBiasedUnicyclePrediction(
            plant_cfg.get("velocity_gain", 1.0),
            plant_cfg.get("yaw_gain", 1.0),
            plant_cfg.get("yaw_bias", 0.0),
        )
        dynamics = ResidualPrediction(
            dynamics, OracleResidualPrediction(true_dynamics, dynamics)
        )
    elif prediction_mode in ("mlp_residual", "icode_residual"):
        import torch
        requested_device = str(planner_cfg.get("device", "cpu"))
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device
        planner_cfg["device"] = device
        if device == "cpu":
            thread_count = int(planner_cfg.get("residual_torch_num_threads", 1))
            if thread_count <= 0:
                raise ValueError("residual_torch_num_threads must be positive")
            torch.set_num_threads(thread_count)
        import sys
        root_text = str(Path(project_root).resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        checkpoint_values = planner_cfg.get("checkpoints")
        if checkpoint_values is not None:
            if (
                not isinstance(checkpoint_values, (list, tuple))
                or len(checkpoint_values) < 2
            ):
                raise ValueError(
                    "planner.checkpoints requires at least two paths"
                )
            checkpoints = list(checkpoint_values)
        else:
            checkpoint = planner_cfg.get("checkpoint")
            if not checkpoint:
                raise ValueError(
                    "%s requires planner.checkpoint" % prediction_mode
                )
            checkpoints = [checkpoint]
        checkpoint_paths = []
        for checkpoint in checkpoints:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(project_root) / checkpoint_path
            checkpoint_paths.append(checkpoint_path.resolve())
        checkpoint_path = checkpoint_paths[0]
        try:
            from mobile_robot_mppi.learning.models import (
                CausalStallGatedResidualDynamics,
                InnovationGatedResidualDynamics,
                CanonicalizedStateResidualDynamics,
                NormalizedSupportGatedResidualDynamics,
                PlatformResidualDynamics,
                PlatformResidualEnsemble,
                ResidualComponentMaskedDynamics,
            )
            residual_members = [
                PlatformResidualDynamics.from_checkpoint(
                    path,
                    device=device,
                    use_torchscript=bool(
                        planner_cfg.get("residual_torchscript", False)
                    ),
                    device_rollout_enabled=bool(
                        planner_cfg.get(
                            "residual_device_rollout_enabled", False
                        )
                    ),
                    cuda_graph_enabled=bool(
                        planner_cfg.get(
                            "residual_cuda_graph_enabled", False
                        )
                    ),
                )
                for path in checkpoint_paths
            ]
            for member in residual_members:
                if member.model.model_type != prediction_mode:
                    raise ValueError(
                        "checkpoint model type %s does not match %s"
                        % (member.model.model_type, prediction_mode)
                    )
            if len(residual_members) == 1:
                residual = residual_members[0]
            else:
                ensemble = dict(
                    planner_cfg.get("residual_ensemble", {})
                )
                residual = PlatformResidualEnsemble(
                    residual_members,
                    state_scales=ensemble.get(
                        "state_scales",
                        [1.0] * state_spec.dimension,
                    ),
                    disagreement_scales=ensemble.get(
                        "disagreement_scales"
                    ),
                    innovation_scales=ensemble.get(
                        "innovation_scales"
                    ),
                    support_soft_z=float(
                        ensemble.get("support_soft_z", 3.0)
                    ),
                    support_hard_z=float(
                        ensemble.get("support_hard_z", 7.0)
                    ),
                    innovation_decay=float(
                        ensemble.get("innovation_decay", 0.9)
                    ),
                    member_paths=checkpoint_paths,
                )
            canonicalization = dict(
                planner_cfg.get("residual_state_canonicalization", {})
            )
            if bool(canonicalization.get("enabled", False)):
                state_names = tuple(
                    str(value)
                    for value in canonicalization.get("state_names", ())
                )
                canonical_values = tuple(
                    float(value)
                    for value in canonicalization.get("values", ())
                )
                if (
                    not state_names
                    or len(state_names) != len(canonical_values)
                    or len(set(state_names)) != len(state_names)
                ):
                    raise ValueError(
                        "residual state canonicalization requires aligned "
                        "unique state_names and values"
                    )
                unknown = [
                    name for name in state_names
                    if name not in state_spec.names
                ]
                if unknown:
                    raise ValueError(
                        "canonical residual state names are unavailable: %s"
                        % unknown
                    )
                residual = CanonicalizedStateResidualDynamics(
                    residual,
                    [state_spec.index(name) for name in state_names],
                    canonical_values,
                )
            component_mask = planner_cfg.get("residual_component_mask")
            if component_mask is not None:
                residual = ResidualComponentMaskedDynamics(
                    residual, component_mask
                )
            support_gate = dict(planner_cfg.get("residual_support_gate", {}))
            if bool(support_gate.get("enabled", False)):
                residual = NormalizedSupportGatedResidualDynamics(
                    residual,
                    soft_z=float(support_gate.get("soft_z", 3.0)),
                    hard_z=float(support_gate.get("hard_z", 5.0)),
                )
            reliability_gate = dict(
                planner_cfg.get("residual_reliability_gate", {})
            )
            if bool(reliability_gate.get("enabled", False)):
                state_names = tuple(
                    reliability_gate.get("state_names", ("v", "omega"))
                )
                unknown = [name for name in state_names if name not in state_spec.names]
                if unknown:
                    raise ValueError(
                        "residual reliability state names are unavailable: %s"
                        % unknown
                    )
                state_indices = [state_spec.index(name) for name in state_names]
                state_scales = reliability_gate.get("state_scales", (0.25, 0.6))
                delay_context = dict(
                    reliability_gate.get("actuation_delay_context", {})
                )
                context_value = None
                if bool(delay_context.get("enabled", False)):
                    context_value = float(
                        planner_cfg.get("command_delay_s", 0.0)
                    ) / float(
                        planner_cfg.get(
                            "dt", config["experiment"]["control_dt"]
                        )
                    )
                residual = InnovationGatedResidualDynamics(
                    residual,
                    state_indices=state_indices,
                    state_scales=state_scales,
                    forgetting_factor=float(
                        reliability_gate.get("forgetting_factor", 0.95)
                    ),
                    minimum_samples=int(
                        reliability_gate.get("minimum_samples", 8)
                    ),
                    confidence_z=float(
                        reliability_gate.get("confidence_z", 1.0)
                    ),
                    off_threshold=float(
                        reliability_gate.get("off_threshold", 0.0)
                    ),
                    on_threshold=float(
                        reliability_gate.get("on_threshold", 0.15)
                    ),
                    rise_rate=float(reliability_gate.get("rise_rate", 0.25)),
                    fall_rate=float(reliability_gate.get("fall_rate", 0.5)),
                    context_value=context_value,
                    context_off_threshold=float(
                        delay_context.get("off_threshold", 0.5)
                    ),
                    context_on_threshold=float(
                        delay_context.get("on_threshold", 0.8)
                    ),
                )
            stall_guard = dict(
                planner_cfg.get("residual_stall_guard", {})
            )
            if bool(stall_guard.get("enabled", False)):
                required_names = ("x", "y", "v")
                unknown = [
                    name
                    for name in required_names
                    if name not in state_spec.names
                ]
                if unknown:
                    raise ValueError(
                        "residual stall guard state names are unavailable: %s"
                        % unknown
                    )
                residual = CausalStallGatedResidualDynamics(
                    residual,
                    position_indices=(
                        state_spec.index("x"),
                        state_spec.index("y"),
                    ),
                    speed_index=state_spec.index("v"),
                    speed_threshold_mps=float(
                        stall_guard.get("speed_threshold_mps", 0.02)
                    ),
                    maximum_low_risk_probability=float(
                        stall_guard.get(
                            "maximum_low_risk_probability", 0.05
                        )
                    ),
                    consecutive_steps=int(
                        stall_guard.get("consecutive_steps", 10)
                    ),
                    goal_exclusion_distance_m=float(
                        stall_guard.get(
                            "goal_exclusion_distance_m", 0.45
                        )
                    ),
                    latch_for_episode=bool(
                        stall_guard.get("latch_for_episode", True)
                    ),
                )
            dynamics = ResidualPrediction(dynamics, residual)
        except ValueError as platform_error:
            from src.planners.mppi_dynamics_adapter import LearnedResidualDynamics
            try:
                if len(checkpoint_paths) != 1:
                    raise platform_error
                residual = LearnedResidualDynamics.from_checkpoint(
                    checkpoint_path,
                    device=device,
                    expected_model_type=prediction_mode.replace("_residual", ""),
                )
            except Exception:
                raise platform_error
            dynamics = ResidualPrediction(dynamics, BatchCompatibleResidual(residual))
    elif prediction_mode != "nominal":
        raise ValueError("unknown prediction mode: %s" % prediction_mode)
    planner_cfg.setdefault("dt", config["experiment"]["control_dt"])
    planner_cfg.setdefault("seed", config["experiment"].get("seed", 0))
    mppi_config = MppiConfig.from_mapping(planner_cfg, action_spec.dimension)
    prior_kind = str(planner_cfg.get("sampling_prior", "goal_warm_start"))
    if prior_kind == "zero":
        prior = ZeroPrior()
    elif prior_kind == "previous_sequence":
        prior = PreviousSequencePrior()
    elif prior_kind == "goal_warm_start":
        prior = GoalWarmStartPrior(
            planner_cfg.get("prior_v_gain", 0.8),
            planner_cfg.get("prior_yaw_gain", 1.2),
            planner_cfg.get("prior_translation_heading_gate_rad"),
            planner_cfg.get("prior_translation_heading_gate_terminal_only", False),
            planner_cfg.get("prior_terminal_max_speed"),
            planner_cfg.get("prior_path_rollout_enabled", False),
            mppi_config.dt,
        )
    elif prior_kind == "hybrid_baseline":
        prior = HybridBaselinePrior(GoalWarmStartPrior(
            planner_cfg.get("prior_v_gain", 0.8),
            planner_cfg.get("prior_yaw_gain", 1.2),
            planner_cfg.get("prior_translation_heading_gate_rad"),
            planner_cfg.get(
                "prior_translation_heading_gate_terminal_only", False
            ),
            planner_cfg.get("prior_terminal_max_speed"),
        ))
    elif prior_kind == "fixed_covariance":
        from mobile_robot_mppi.policies.priors import FixedCovariancePrior

        fallback = GoalWarmStartPrior(
            planner_cfg.get("prior_v_gain", 0.8),
            planner_cfg.get("prior_yaw_gain", 1.2),
            planner_cfg.get("prior_translation_heading_gate_rad"),
            planner_cfg.get(
                "prior_translation_heading_gate_terminal_only", False
            ),
            planner_cfg.get("prior_terminal_max_speed"),
            planner_cfg.get("prior_path_rollout_enabled", False),
            mppi_config.dt,
        )
        prior = FixedCovariancePrior(
            fallback,
            mppi_config.noise_sigma,
            planner_cfg.get(
                "fixed_covariance_scale",
                [1.0] * action_spec.dimension,
            ),
        )
    elif prior_kind == "contextual_bandit_covariance":
        rl_cfg = dict(config.get("rl", {}))
        if not bool(rl_cfg.get("enabled", False)):
            raise ValueError(
                "sampling_prior=contextual_bandit_covariance requires rl.enabled=true"
            )
        checkpoint = rl_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("contextual bandit inference requires rl.checkpoint")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(project_root) / checkpoint_path
        fallback = GoalWarmStartPrior(
            planner_cfg.get("prior_v_gain", 0.8),
            planner_cfg.get("prior_yaw_gain", 1.2),
            planner_cfg.get("prior_translation_heading_gate_rad"),
            planner_cfg.get(
                "prior_translation_heading_gate_terminal_only", False
            ),
            planner_cfg.get("prior_terminal_max_speed"),
            planner_cfg.get("prior_path_rollout_enabled", False),
            mppi_config.dt,
        )
        from mobile_robot_mppi.rl.contextual_bandit import (
            ContextualBanditCovariancePrior,
        )

        prior = ContextualBanditCovariancePrior.from_checkpoint(
            checkpoint_path,
            fallback,
            mppi_config.noise_sigma,
        )
    elif prior_kind == "rl":
        rl_cfg = dict(config.get("rl", {}))
        if not bool(rl_cfg.get("enabled", False)):
            raise ValueError("sampling_prior=rl requires rl.enabled=true")
        if rl_policy is not None:
            # Training injects a stateful prior object.  Legacy integrations may
            # still inject a plain callable through the framework-free adapter.
            prior = (
                rl_policy
                if callable(getattr(rl_policy, "propose", None))
                else RLPolicyPrior(
                    rl_policy, str(rl_cfg.get("policy_id", "rl_policy"))
                )
            )
        else:
            checkpoint = rl_cfg.get("checkpoint")
            if not checkpoint:
                raise ValueError(
                    "RL inference requires rl.checkpoint or an injected training prior"
                )
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(project_root) / checkpoint_path
            fallback = GoalWarmStartPrior(
                planner_cfg.get("prior_v_gain", 0.8),
                planner_cfg.get("prior_yaw_gain", 1.2),
                planner_cfg.get("prior_translation_heading_gate_rad"),
                planner_cfg.get(
                    "prior_translation_heading_gate_terminal_only", False
                ),
                planner_cfg.get("prior_terminal_max_speed"),
            )
            from mobile_robot_mppi.rl.prior import TorchSACPrior

            rl_device = str(rl_cfg.get("device", "cpu"))
            if rl_device == "auto":
                import torch
                rl_device = "cuda" if torch.cuda.is_available() else "cpu"
            if rl_device == "cpu":
                import torch
                thread_count = int(rl_cfg.get("torch_num_threads", 1))
                if thread_count <= 0:
                    raise ValueError("rl.torch_num_threads must be positive")
                torch.set_num_threads(thread_count)

            prior = TorchSACPrior.from_checkpoint(
                checkpoint_path,
                action_spec,
                mppi_config.noise_sigma,
                device=rl_device,
                gate_config=rl_cfg.get("gate", {}),
                fallback_prior=fallback,
                policy_id=str(rl_cfg.get("policy_id", "sac_mppi_prior")),
            )
    elif prior_kind == "paper_direct_rl":
        rl_cfg = dict(config.get("rl", {}))
        if not bool(rl_cfg.get("enabled", False)):
            raise ValueError(
                "sampling_prior=paper_direct_rl requires rl.enabled=true"
            )
        fallback = GoalWarmStartPrior(
            planner_cfg.get("prior_v_gain", 0.8),
            planner_cfg.get("prior_yaw_gain", 1.2),
            planner_cfg.get("prior_translation_heading_gate_rad"),
            planner_cfg.get(
                "prior_translation_heading_gate_terminal_only", False
            ),
            planner_cfg.get("prior_terminal_max_speed"),
            planner_cfg.get("prior_path_rollout_enabled", False),
            mppi_config.dt,
        )
        if rl_policy is not None:
            prior = rl_policy
        else:
            checkpoint = rl_cfg.get("checkpoint")
            if not checkpoint:
                raise ValueError(
                    "paper direct RL inference requires rl.checkpoint"
                )
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(project_root) / checkpoint_path
            rl_device = str(rl_cfg.get("device", "cpu"))
            if rl_device == "auto":
                import torch
                rl_device = "cuda" if torch.cuda.is_available() else "cpu"
            if rl_device == "cpu":
                import torch
                thread_count = int(rl_cfg.get("torch_num_threads", 1))
                if thread_count <= 0:
                    raise ValueError("rl.torch_num_threads must be positive")
                torch.set_num_threads(thread_count)
            from mobile_robot_mppi.rl.paper_policy import (
                PaperDirectControlPolicy,
            )
            residual_context = None
            residual_context_mapping = dict(
                rl_cfg.get("residual_context", {})
            )
            if bool(residual_context_mapping.get("enabled", False)):
                from mobile_robot_mppi.rl.residual_context import (
                    ResidualContextEncoder,
                )

                residual_context = ResidualContextEncoder(
                    dynamics, state_spec, residual_context_mapping
                )

            prior = PaperDirectControlPolicy.from_checkpoint(
                checkpoint_path,
                action_spec,
                device=rl_device,
                fallback_prior=fallback,
                residual_context=residual_context,
                residual_correction_authority=rl_cfg.get(
                    "residual_correction_authority", {}
                ),
                allow_controller_action_superset=bool(
                    rl_cfg.get("allow_actor_action_subspace", False)
                ),
            )
    else:
        raise ValueError("unknown built-in sampling prior: %s" % prior_kind)
    memory_cfg = dict(config.get("memory", {}))
    memory = LegacyMemoryAdapter(project_root, memory_cfg) if memory_cfg.get("enable", False) else None
    optimizer = str(planner_cfg.get("optimizer", "standard"))
    controller_type = MppiController
    controller_kwargs = {}
    if optimizer == "rl_driven":
        if prior_kind not in (
            "rl",
            "hybrid_baseline",
            "fixed_covariance",
            "contextual_bandit_covariance",
        ):
            raise ValueError(
                "planner.optimizer=rl_driven requires an explicit mixture prior"
            )
        from mobile_robot_mppi.planning.rl_driven_mppi import (
            RLDrivenMppiController,
        )

        controller_type = RLDrivenMppiController
        controller_kwargs["rl_driven_config"] = planner_cfg.get(
            "rl_driven", {}
        )
    elif optimizer == "paper_rl_driven":
        if prior_kind != "paper_direct_rl":
            raise ValueError(
                "planner.optimizer=paper_rl_driven requires "
                "sampling_prior=paper_direct_rl"
            )
        from mobile_robot_mppi.planning.rl_driven_mppi import (
            PaperRLDrivenMppiController,
        )

        controller_type = PaperRLDrivenMppiController
        paper_mapping = dict(planner_cfg.get("paper_rl_driven", {}))
        controller_kwargs["paper_rl_driven_config"] = paper_mapping
        sidecar = _build_frozen_hss_sidecar(
            paper_mapping.get("reliability_sidecar", {}),
            project_root,
            state_spec,
            action_spec,
        )
        if sidecar is not None:
            controller_kwargs["reliability_residual"] = sidecar
            controller_kwargs["reliability_nominal_dynamics"] = (
                nominal_dynamics
            )
    elif optimizer == "anytime_bandit":
        if prior_kind not in (
            "fixed_covariance", "contextual_bandit_covariance"
        ):
            raise ValueError(
                "planner.optimizer=anytime_bandit requires a Gaussian "
                "fixed or contextual covariance prior"
            )
        from mobile_robot_mppi.planning.anytime_mppi import (
            AnytimeMppiController,
        )

        controller_type = AnytimeMppiController
        anytime = dict(planner_cfg.get("anytime_bandit", {}))
        checkpoint = anytime.get("checkpoint")
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(project_root) / checkpoint_path
            anytime["checkpoint"] = str(checkpoint_path.resolve())
        controller_kwargs["anytime_config"] = anytime
    elif optimizer != "standard":
        raise ValueError("unknown MPPI optimizer: %s" % optimizer)
    controller = controller_type(
        dynamics=dynamics,
        state_spec=state_spec,
        action_spec=action_spec,
        config=mppi_config,
        sampling_prior=prior,
        memory_cost=(None if memory is None else memory.trajectory_cost),
        **controller_kwargs
    )
    shield_mapping = dict(
        planner_cfg.get("residual_safety_shield", {})
    )
    if bool(shield_mapping.get("enabled", False)):
        if prediction_mode not in ("mlp_residual", "icode_residual"):
            raise ValueError(
                "residual safety shield requires learned residual dynamics"
            )
        integration_contract = str(
            shield_mapping.get("rl_hss_integration", "")
        )
        rl_enabled = bool(config.get("rl", {}).get("enabled", False))
        matched_rl_hss = bool(
            optimizer == "paper_rl_driven"
            and rl_enabled
            and integration_contract == "matched_controllers_v1"
        )
        if optimizer != "standard" and not matched_rl_hss:
            raise ValueError(
                "residual safety shield requires standard MPPI or the "
                "explicit matched_controllers_v1 RL/HSS contract"
            )
        if rl_enabled and not matched_rl_hss:
            raise ValueError(
                "residual safety shield RL requires matched paper RL/HSS "
                "controllers"
            )
        if integration_contract and not matched_rl_hss:
            raise ValueError("residual shield RL/HSS contract is not active")
        from mobile_robot_mppi.planning.residual_shield import (
            ResidualSafetyShieldController,
        )

        nominal_controller = controller_type(
            dynamics=nominal_dynamics,
            state_spec=state_spec,
            action_spec=action_spec,
            config=mppi_config,
            sampling_prior=deepcopy(prior),
            memory_cost=(
                None if memory is None else memory.trajectory_cost
            ),
            **deepcopy(controller_kwargs)
        )
        controller = ResidualSafetyShieldController(
            residual_controller=controller,
            nominal_controller=nominal_controller,
            shield_config=shield_mapping,
        )
    perception = LegacyScanPipeline(project_root, config.get("perception", {}))
    safety = ScanGuardArbiter(
        action_spec,
        config.get("perception", {}).get("scan_guard", {}),
    )
    return {
        "state_spec": state_spec,
        "action_spec": action_spec,
        "reference": reference,
        "plant": plant,
        "sensors": sensors,
        "controller": controller,
        "perception": perception,
        "safety": safety,
        "memory": memory,
        "prediction_mode": prediction_mode,
    }
