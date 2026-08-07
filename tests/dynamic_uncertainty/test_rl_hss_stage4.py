from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.dynamic_uncertainty.run_residual_dynamics_stage1 import (
    configure_condition as configure_stage3_condition,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    DEFAULT_PROTOCOL,
    _mapping,
    _resolve,
    build_schedule,
    configure_job,
    run,
    validate_job_config,
    validate_protocol,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import (
    action_spec_from_config,
    body_velocity_action,
    dynamic_unicycle_state,
    unicycle_state,
)
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D
from mobile_robot_mppi.obstacles.collision_risk import (
    GaussianMixtureObstacleForecast,
)
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    LegacyUnicyclePrediction,
)
from mobile_robot_mppi.planning.mppi import MppiConfig
from mobile_robot_mppi.planning.residual_shield import (
    ResidualSafetyShieldController,
)
from mobile_robot_mppi.planning.rl_driven_mppi import (
    PaperRLDrivenMppiController,
)
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.rl.paper_policy import PaperDirectControlPolicy


ROOT = Path(__file__).resolve().parents[2]


def _inputs():
    protocol = _mapping(DEFAULT_PROTOCOL)
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    base = load_yaml(_resolve(protocol["base_config"]))
    return protocol, stage3, base


def test_stage4_schedule_is_reproducible_complete_and_blocked():
    protocol, stage3, _ = _inputs()
    first = build_schedule(protocol, stage3)
    second = build_schedule(protocol, stage3)
    assert first == second
    assert len(first) == 24
    assert [row["run_order"] for row in first] == list(range(24))
    assert len({row["experimental_key"] for row in first}) == 24
    for seed in protocol["design"]["obstacle_process_seeds"]:
        block = [row for row in first if row["episode_seed"] == seed]
        assert len(block) == 8
        assert {row["rl_hss_enabled"] for row in block} == {False, True}
        assert {
            (row["condition"], row["model_block"], row["rl_hss_enabled"])
            for row in block
        } == {
            ("nominal", -1, False),
            ("nominal", -1, True),
            *{
                ("icode_residual", model_block, enabled)
                for model_block in range(3)
                for enabled in (False, True)
            },
        }


def test_rl_off_cells_are_exact_stage3_configurations():
    protocol, stage3, base = _inputs()
    for job in build_schedule(protocol, stage3):
        if job["rl_hss_enabled"]:
            continue
        expected = configure_stage3_condition(
            base,
            {
                "condition": job["condition"],
                "episode_seed": job["episode_seed"],
                "model_block": job["model_block"],
            },
            stage3,
        )
        assert configure_job(base, job, protocol, stage3) == expected


def test_rl_on_changes_only_registered_controller_treatment():
    protocol, stage3, base = _inputs()
    frozen_sections = (
        "task",
        "state_space",
        "action_space",
        "plant",
        "scene",
        "sensors",
        "perception",
    )
    for job in build_schedule(protocol, stage3):
        config = configure_job(base, job, protocol, stage3)
        validate_job_config(config, job, protocol, stage3, base)
        for section in frozen_sections:
            assert config[section] == base[section]
        if not job["rl_hss_enabled"]:
            continue
        planner = config["planner"]
        assert planner["optimizer"] == "paper_rl_driven"
        assert planner["sampling_prior"] == "paper_direct_rl"
        assert planner["num_samples"] == 300
        assert planner["paper_rl_driven"]["iterations"] == 2
        assert planner["num_samples"] * planner["paper_rl_driven"]["iterations"] == 600
        assert planner["paper_rl_driven"]["reliability"]["enabled"] is True
        assert planner["paper_rl_driven"]["reliability_sidecar"]["contract"] == (
            "frozen_l217_value_hss_v1"
        )
        assert config["rl"]["checkpoint"] == protocol["actor"]["checkpoint"]
        assert config["rl"]["allow_actor_action_subspace"] is True
        assert config["perception"]["scan_guard"] == base["perception"]["scan_guard"]
        if job["condition"] == "icode_residual":
            assert planner["residual_safety_shield"]["enabled"] is True
            assert planner["residual_safety_shield"]["parallel_planning_enabled"] is True
            assert planner["residual_safety_shield"]["rl_hss_integration"] == (
                "matched_controllers_v1"
            )


def test_protocol_preflight_verifies_hashes_without_starting_matrix():
    protocol, _, _ = _inputs()
    audit = validate_protocol(protocol)
    assert audit["status"] == "preflight_passed_matrix_not_started"
    assert audit["episode_count"] == 24
    assert audit["sealed_seeds_opened"] is False
    output_existed = _resolve(protocol["output_dir"]).exists()
    result = run(DEFAULT_PROTOCOL, execute=False)
    assert result["formal_matrix_started"] is False
    assert len(result["schedule"]) == 24
    assert _resolve(protocol["output_dir"]).exists() is output_existed


class _SidecarResidual:
    state_dim = 5
    control_dim = 2

    def __init__(self):
        self.updates = 0

    def derivative(self, state, control, time=None):
        del control, time
        return np.zeros_like(np.asarray(state, dtype=np.float64))

    def disagreement(self, states, controls):
        del controls
        return np.zeros(np.asarray(states).shape[:-1], dtype=np.float64)

    def support_confidence(self, states, controls):
        del controls
        return np.ones(np.asarray(states).shape[:-1], dtype=np.float64)

    def observe_prediction_errors(self, nominal_error, residual_error):
        assert nominal_error.shape == (5,)
        assert residual_error.shape == (5,)
        self.updates += 1

    def reset(self):
        self.updates = 0


class _DirectPolicy:
    def __init__(self):
        self.fallback = GoalWarmStartPrior()

    def propose(self, *args, **kwargs):
        return self.fallback.propose(*args, **kwargs)

    def action_distribution(self, *args, **kwargs):
        raise AssertionError("construction/transition test must not run Actor")

    def sample_actions(self, *args, **kwargs):
        raise AssertionError("construction/transition test must not run Actor")

    def terminal_value(self, *args, **kwargs):
        raise AssertionError("construction/transition test must not run critic")


class _PlanningPolicy:
    def __init__(self):
        self.fallback = GoalWarmStartPrior()

    def propose(self, *args, **kwargs):
        return self.fallback.propose(*args, **kwargs)

    def action_distribution(
        self, states, previous, observation, reference, state_spec, time_offset
    ):
        del previous, observation, reference, state_spec, time_offset
        batch = np.asarray(states).shape[0]
        return {
            "physical_mean": np.zeros((batch, 2), dtype=np.float64),
            "physical_std": np.full((batch, 2), 0.05, dtype=np.float64),
            "raw_observation": np.zeros((batch, 1), dtype=np.float64),
        }

    def sample_actions(
        self,
        states,
        previous,
        observation,
        reference,
        state_spec,
        time_offset,
        rng,
    ):
        del previous, observation, reference, state_spec, time_offset, rng
        return np.zeros((np.asarray(states).shape[0], 2)), {}

    def terminal_value(self, *args, **kwargs):
        raise AssertionError("zero terminal weight must not run critic")


def _paper_risk_controller(horizon=4):
    return PaperRLDrivenMppiController(
        dynamics=LegacyUnicyclePrediction(),
        state_spec=unicycle_state(),
        action_spec=body_velocity_action((0.0, 0.5), 1.0),
        config=MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            seed=19,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_risk_weight=75.0,
            probabilistic_obstacle_hard_threshold=0.20,
            probabilistic_obstacle_hard_penalty=10000.0,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_missing_forecast_action="stop",
        ),
        sampling_prior=_PlanningPolicy(),
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.0,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )


def _paper_risk_forecast(horizon):
    means = np.repeat([[[-0.49, 0.0]]], horizon, axis=0)
    covariances = np.repeat(
        (1.0e-4 * np.eye(2))[None, None, :, :],
        horizon,
        axis=0,
    )
    return GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=means,
        component_covariances=covariances,
        component_weights=np.ones((horizon, 1)),
        radius_m=0.15,
        source="stage4_amendment1_test",
    )


def test_paper_rl_missing_forecast_uses_frozen_fail_closed_stop():
    controller = _paper_risk_controller()
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
    )
    plan = controller.plan(observation, PointGoal(2.0, 0.0))
    np.testing.assert_array_equal(plan.proposed_control.values, np.zeros(2))
    np.testing.assert_array_equal(plan.control_sequence, np.zeros((4, 2)))
    assert plan.diagnostics["probabilistic_obstacle_fail_closed"] is True
    assert plan.diagnostics["probabilistic_obstacle_forecast_count"] == 0


def test_paper_rl_preserves_risk_filter_and_active_avoidance_contract():
    horizon = 4
    controller = _paper_risk_controller(horizon=horizon)

    def fixed_gaussian(mean, variance, count, rng):
        del mean, variance, rng
        samples = np.zeros((count, horizon, 2), dtype=np.float64)
        samples[0, :, 0] = 0.5
        return samples

    controller._gaussian_samples = fixed_gaussian
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={
            "probabilistic_obstacle_forecasts": (
                _paper_risk_forecast(horizon),
            ),
            "dynamic_obstacle_tracker": {
                "enabled": True,
                "forecast_valid": True,
                "forecast_availability": 1.0,
            },
        },
    )
    plan = controller.plan(observation, PointGoal(2.0, 0.0))
    assert plan.proposed_control.values[0] > 0.45
    assert plan.diagnostics["paper_total_rollouts"] == 16
    assert plan.diagnostics[
        "probabilistic_obstacle_candidate_filter_enabled"
    ] is True
    assert plan.diagnostics[
        "probabilistic_obstacle_candidate_feasible_fraction"
    ] < 1.0
    assert plan.diagnostics[
        "probabilistic_obstacle_active_avoidance_enabled"
    ] is True
    assert plan.diagnostics["probabilistic_obstacle_hard_violation"] is False
    assert plan.diagnostics["dynamic_obstacle_tracker_forecast_valid"] is True


def test_paper_rl_reverse_candidate_filter_matches_standard_mppi(monkeypatch):
    horizon = 4
    controller = PaperRLDrivenMppiController(
        dynamics=LegacyUnicyclePrediction(),
        state_spec=unicycle_state(),
        action_spec=body_velocity_action((-0.30, 0.50), 0.60),
        config=MppiConfig(
            horizon=horizon,
            num_samples=8,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            seed=23,
            probabilistic_obstacle_risk_enabled=True,
            probabilistic_obstacle_candidate_filter_enabled=True,
            probabilistic_obstacle_hard_violation_action="active_avoidance",
            probabilistic_obstacle_missing_forecast_action="stop",
            probabilistic_obstacle_reverse_candidate_filter_enabled=True,
            optimizer_diagnostics_enabled=True,
        ),
        sampling_prior=_PlanningPolicy(),
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.0,
            "elite_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
    )

    def fixed_gaussian(mean, variance, count, rng):
        del mean, variance, rng
        samples = np.zeros((count, horizon, 2), dtype=np.float64)
        samples[2, :, 0] = -0.30
        samples[3, :, 0] = 0.30
        return samples

    def prefer_reverse(_trajectories, controls, *_args, **_kwargs):
        first_v = np.asarray(controls, dtype=np.float64)[:, 0, 0]
        return np.where(first_v < 0.0, 0.0, np.where(
            first_v > 0.0, 20.0, 10.0
        ))

    def reverse_is_hard(trajectories, _forecasts):
        values = np.asarray(trajectories, dtype=np.float64)
        reverse = values[:, 1, 0] < -1.0e-9
        count = values.shape[0]
        return SimpleNamespace(
            hard_violation=reverse,
            maximum_step_probability=np.where(reverse, 0.90, 0.05),
            accumulated_probability_mass=np.where(reverse, 1.20, 0.10),
            horizon_union_bound=np.where(reverse, 1.0, 0.10),
            step_probability_upper_bound=np.repeat(
                np.where(reverse, 0.90, 0.05)[:, None],
                horizon,
                axis=1,
            ).reshape(count, horizon),
        )

    monkeypatch.setattr(controller, "_gaussian_samples", fixed_gaussian)
    monkeypatch.setattr(controller, "_cost", prefer_reverse)
    monkeypatch.setattr(
        controller, "_probabilistic_collision_risk", reverse_is_hard
    )
    observation = RobotObservation(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        auxiliary={
            "probabilistic_obstacle_forecasts": (
                _paper_risk_forecast(horizon),
            ),
            "dynamic_obstacle_tracker": {
                "enabled": True,
                "forecast_valid": True,
                "forecast_availability": 1.0,
            },
        },
    )

    plan = controller.plan(observation, PointGoal(2.0, 0.0))

    assert plan.proposed_control.v >= 0.0
    assert plan.diagnostics[
        "optimizer_reverse_candidate_filter_enabled"
    ]
    assert plan.diagnostics["optimizer_reverse_candidate_count"] == 1
    assert plan.diagnostics[
        "optimizer_reverse_candidate_prediction_rejected_count"
    ] == 1


def test_nominal_controller_updates_explicit_hss_sidecar_causally():
    sidecar = _SidecarResidual()
    nominal = DynamicUnicyclePrediction()
    controller = PaperRLDrivenMppiController(
        dynamics=nominal,
        state_spec=dynamic_unicycle_state(),
        action_spec=body_velocity_action((0.0, 0.5), 1.0),
        config=MppiConfig(
            horizon=4,
            num_samples=8,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            seed=17,
        ),
        sampling_prior=_DirectPolicy(),
        paper_rl_driven_config={
            "iterations": 2,
            "guided_fraction": 0.25,
            "terminal_value_weight": 0.0,
        },
        reliability_residual=sidecar,
        reliability_nominal_dynamics=DynamicUnicyclePrediction(),
    )
    assert controller._residual_for_reliability() is sidecar
    controller._reliability_previous_state = np.zeros(5)
    controller._reliability_pending_control = np.zeros(2)
    controller._observe_residual_reliability(np.zeros(5))
    assert sidecar.updates == 1
    assert controller._reliability_pending_control is None
    controller.reset(seed=17)
    assert sidecar.updates == 0


def test_frozen_actor_subspace_does_not_narrow_stage3_controller():
    protocol, _, base = _inputs()
    controller_action = action_spec_from_config(base["action_space"])
    checkpoint = _resolve(protocol["actor"]["checkpoint"])
    with pytest.raises(ValueError, match="physical action bounds"):
        PaperDirectControlPolicy.from_checkpoint(
            checkpoint, controller_action, device="cpu"
        )
    policy = PaperDirectControlPolicy.from_checkpoint(
        checkpoint,
        controller_action,
        device="cpu",
        allow_controller_action_superset=True,
    )
    assert policy.actor_action_subspace is True
    np.testing.assert_array_equal(policy.action_spec.lower, (0.0, -0.9))
    np.testing.assert_array_equal(
        policy.controller_action_spec.lower, (-0.35, -0.9)
    )


def test_factory_builds_matched_rl_hss_shield_controllers(monkeypatch):
    import mobile_robot_mppi.learning.models as models
    import mobile_robot_mppi.runtime.factories as factories

    class FakeResidual:
        state_dim = 3
        control_dim = 2
        model = SimpleNamespace(model_type="icode_residual")

        def derivative(self, state, control, time=None):
            del control, time
            return np.zeros_like(np.asarray(state, dtype=np.float64))

        def reset(self):
            return None

    class FakeSidecar(FakeResidual):
        def disagreement(self, states, controls):
            del controls
            return np.zeros(np.asarray(states).shape[:-1])

        def support_confidence(self, states, controls):
            del controls
            return np.ones(np.asarray(states).shape[:-1])

    monkeypatch.setattr(
        models.PlatformResidualDynamics,
        "from_checkpoint",
        lambda *args, **kwargs: FakeResidual(),
    )
    monkeypatch.setattr(
        factories,
        "_build_frozen_hss_sidecar",
        lambda *args, **kwargs: FakeSidecar(),
    )
    config = load_yaml(ROOT / "configs/research/legacy_kinematic.yaml")
    config = deepcopy(config)
    config["planner"].update({
        "prediction_mode": "icode_residual",
        "checkpoint": "unused-test-checkpoint.pt",
        "probabilistic_obstacle_risk_enabled": True,
        "optimizer": "paper_rl_driven",
        "sampling_prior": "paper_direct_rl",
        "importance_sampling_correction": False,
        "paper_rl_driven": {
            "iterations": 2,
            "guided_fraction": 0.25,
            "terminal_value_weight": 0.0,
            "reliability_sidecar": {"contract": "test"},
        },
        "residual_safety_shield": {
            "enabled": True,
            "rl_hss_integration": "matched_controllers_v1",
        },
    })
    config["rl"] = {"enabled": True}
    invalid = deepcopy(config)
    invalid["planner"]["residual_safety_shield"].pop(
        "rl_hss_integration"
    )
    with pytest.raises(ValueError, match="explicit matched_controllers_v1"):
        factories.make_components(invalid, ROOT, rl_policy=_DirectPolicy())
    components = factories.make_components(
        config, ROOT, rl_policy=_DirectPolicy()
    )
    try:
        shield = components["controller"]
        assert isinstance(shield, ResidualSafetyShieldController)
        assert isinstance(
            shield.residual_controller, PaperRLDrivenMppiController
        )
        assert isinstance(
            shield.nominal_controller, PaperRLDrivenMppiController
        )
        assert shield.residual_controller.sampling_prior is not (
            shield.nominal_controller.sampling_prior
        )
        assert shield.residual_controller.reliability_residual is not (
            shield.nominal_controller.reliability_residual
        )
        shield.reset(seed=91)
        assert shield.residual_controller.rng is not shield.nominal_controller.rng
        np.testing.assert_array_equal(
            shield.residual_controller.rng.get_state()[1],
            shield.nominal_controller.rng.get_state()[1],
        )
    finally:
        components["controller"].close()
        components["plant"].close()
