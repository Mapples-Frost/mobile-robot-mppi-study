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
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior, PreviousSequencePrior, RLPolicyPrior, ZeroPrior
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter
from mobile_robot_mppi.simulation.kinematic import KinematicPlant
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant
from mobile_robot_mppi.simulation.sensors import SimulatedSensorSuite


def make_components(config, project_root, rl_policy=None):
    state_spec = state_spec_from_config(config["state_space"])
    action_spec = action_spec_from_config(config["action_space"])
    reference = reference_from_config(config["task"])
    plant_cfg = dict(config["plant"])
    scene = dict(config.get("scene", {}))
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
        if str(planner_cfg.get("device", "cpu")) == "cpu":
            import torch
            thread_count = int(planner_cfg.get("residual_torch_num_threads", 1))
            if thread_count <= 0:
                raise ValueError("residual_torch_num_threads must be positive")
            torch.set_num_threads(thread_count)
        import sys
        root_text = str(Path(project_root).resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        checkpoint = planner_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("%s requires planner.checkpoint" % prediction_mode)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(project_root) / checkpoint_path
        device = str(planner_cfg.get("device", "cpu"))
        try:
            from mobile_robot_mppi.learning.models import PlatformResidualDynamics
            residual = PlatformResidualDynamics.from_checkpoint(
                checkpoint_path,
                device=device,
                use_torchscript=bool(planner_cfg.get("residual_torchscript", False)),
            )
            if residual.model.model_type != prediction_mode:
                raise ValueError(
                    "checkpoint model type %s does not match %s"
                    % (residual.model.model_type, prediction_mode)
                )
            dynamics = ResidualPrediction(dynamics, residual)
        except ValueError as platform_error:
            from src.planners.mppi_dynamics_adapter import LearnedResidualDynamics
            try:
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
    else:
        raise ValueError("unknown built-in sampling prior: %s" % prior_kind)
    memory_cfg = dict(config.get("memory", {}))
    memory = LegacyMemoryAdapter(project_root, memory_cfg) if memory_cfg.get("enable", False) else None
    controller = MppiController(
        dynamics=dynamics,
        state_spec=state_spec,
        action_spec=action_spec,
        config=mppi_config,
        sampling_prior=prior,
        memory_cost=(None if memory is None else memory.trajectory_cost),
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
