"""Compatibility gates for adding composable dynamics to the trusted MPPI."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np

from src.dynamics import (
    CombinedDynamics,
    DisturbanceConfig,
    DisturbedUnicycle,
    NominalUnicycle,
)
from src.dynamics.residual import OracleResidual, ZeroResidual
from src.planners import mppi_mujoco_receding_horizon_experiment as planner
from src.planners.mppi_dynamics_adapter import MppiDynamicsAdapter
from src.planners.sampling_prior import (
    GoalWarmStartPrior,
    PreviousSequencePrior,
    SamplingPrior,
)


ROOT = Path(__file__).resolve().parents[2]
START_STATE = (0.2, -0.1, 0.3)
CONTROLS = [(0.5, 0.2), (1.0, -0.4), (0.25, 0.0)]
DT = 0.1

# Captured before the optional dynamics argument was introduced.  This is
# intentionally literal rather than recomputed through another implementation.
LEGACY_GOLDEN = [
    (0.2, -0.1, 0.3),
    (0.24776682445628032, -0.08522398966693302, 0.32),
    (0.3426903662645244, -0.053767333605321245, 0.28),
    (0.3667167522222937, -0.0468584423912184, 0.28),
]


def test_default_rollout_is_the_exact_pre_change_golden_and_keeps_types():
    three_positional = planner.rollout_control_sequence(START_STATE, CONTROLS, DT)
    explicit_none = planner.rollout_control_sequence(
        START_STATE,
        CONTROLS,
        DT,
        dynamics_model=None,
    )

    assert three_positional == LEGACY_GOLDEN
    assert explicit_none == LEGACY_GOLDEN
    assert isinstance(three_positional, list)
    assert all(isinstance(state, tuple) for state in three_positional)


def test_explicit_nominal_euler_adapter_matches_the_legacy_rollout():
    adapter = MppiDynamicsAdapter(
        NominalUnicycle(),
        integration_method="euler",
    )

    adapted = planner.rollout_control_sequence(
        START_STATE,
        CONTROLS,
        DT,
        dynamics_model=adapter,
    )

    assert isinstance(adapted, list)
    assert all(isinstance(state, tuple) for state in adapted)
    np.testing.assert_allclose(adapted, LEGACY_GOLDEN, rtol=0.0, atol=2e-16)


def test_rk4_adapter_follows_the_constant_control_unicycle_arc():
    state = (0.2, -0.1, 0.3)
    velocity, yaw_rate = (0.8, -0.4)
    dt = 0.2
    adapter = MppiDynamicsAdapter(
        NominalUnicycle(),
        integration_method="rk4",
    )

    trajectory = adapter.rollout(state, [(velocity, yaw_rate)], dt)
    final_heading = state[2] + yaw_rate * dt
    expected = (
        state[0]
        + (velocity / yaw_rate)
        * (math.sin(final_heading) - math.sin(state[2])),
        state[1]
        - (velocity / yaw_rate)
        * (math.cos(final_heading) - math.cos(state[2])),
        final_heading,
    )

    assert isinstance(trajectory, list)
    assert all(isinstance(item, tuple) for item in trajectory)
    np.testing.assert_allclose(trajectory[-1], expected, rtol=0.0, atol=3e-9)


def test_zero_residual_rollout_equals_nominal_for_every_state():
    nominal_adapter = MppiDynamicsAdapter(
        NominalUnicycle(),
        integration_method="rk4",
        start_time=0.35,
    )
    combined_adapter = MppiDynamicsAdapter(
        CombinedDynamics(NominalUnicycle(), ZeroResidual()),
        integration_method="rk4",
        start_time=0.35,
    )

    nominal = nominal_adapter.rollout(START_STATE, CONTROLS, DT)
    combined = combined_adapter.rollout(START_STATE, CONTROLS, DT)

    assert combined == nominal


def test_combined_oracle_rollout_equals_the_disturbed_true_rollout():
    config = DisturbanceConfig(
        velocity_gain=1.15,
        yaw_gain=0.8,
        yaw_bias=0.12,
        world_disturbance_amplitude=(0.08, -0.04, 0.03),
        world_disturbance_frequency=(0.7, 1.1, 0.5),
        world_disturbance_phase=(0.2, -0.1, 0.4),
        state_disturbance_gain=(0.03, -0.02, 0.06),
    )
    true_model = DisturbedUnicycle(config)
    nominal = NominalUnicycle()
    combined = CombinedDynamics(
        nominal,
        OracleResidual(true_dynamics=true_model, nominal_dynamics=nominal),
    )
    start_time = 0.45

    true_trajectory = MppiDynamicsAdapter(
        true_model,
        integration_method="rk4",
        start_time=start_time,
    ).rollout(START_STATE, CONTROLS, DT)
    oracle_trajectory = MppiDynamicsAdapter(
        combined,
        integration_method="rk4",
        start_time=start_time,
    ).rollout(START_STATE, CONTROLS, DT)

    np.testing.assert_allclose(
        oracle_trajectory,
        true_trajectory,
        rtol=1e-13,
        atol=1e-13,
    )


def test_stateful_dynamics_are_cloned_independently_for_each_rollout():
    stateful_model = DisturbedUnicycle(
        DisturbanceConfig(control_delay_steps=1)
    )
    adapter = MppiDynamicsAdapter(
        stateful_model,
        integration_method="euler",
        clone_per_rollout=True,
    )
    controls = [(1.0, 0.2), (0.5, -0.1), (0.2, 0.0)]

    first = adapter.rollout((0.0, 0.0, 0.0), controls, 0.1)
    second = adapter.rollout((0.0, 0.0, 0.0), controls, 0.1)

    assert second == first
    np.testing.assert_array_equal(stateful_model.last_applied_control, [0.0, 0.0])
    assert len(stateful_model.pending_controls) == 1
    np.testing.assert_array_equal(stateful_model.pending_controls[0], [0.0, 0.0])


def test_planner_optional_dynamics_parameters_are_appended_and_forwarded(
    monkeypatch,
):
    rollout_parameters = list(
        inspect.signature(planner.rollout_control_sequence).parameters.values()
    )
    sample_parameters = list(
        inspect.signature(planner.sample_control_sequences).parameters.values()
    )

    assert rollout_parameters[-1].name == "dynamics_model"
    assert rollout_parameters[-1].default is None
    assert sample_parameters[-1].name == "dynamics_model"
    assert sample_parameters[-1].default is None

    sentinel = object()
    captured = {}

    def fake_rollout(start_state, control_sequence, dt, dynamics_model=None):
        captured["dynamics_model"] = dynamics_model
        return [start_state] * (len(control_sequence) + 1)

    monkeypatch.setattr(planner, "rollout_control_sequence", fake_rollout)
    result = planner.sample_control_sequences(
        nominal_sequence=[(0.5, 0.0)],
        current_state=(0.0, 0.0, 0.0),
        dt=0.1,
        obstacles=[],
        robot_radius=0.2,
        num_samples=0,
        v_std=0.1,
        omega_std=0.1,
        v_min=0.0,
        v_max=1.0,
        omega_max=1.0,
        dynamics_model=sentinel,
    )

    assert result == []
    assert captured["dynamics_model"] is sentinel


def test_previous_sequence_prior_shifts_pads_clips_and_updates():
    prior = PreviousSequencePrior(
        [(2.0, -2.0), (1.5, 0.5), (-1.0, 3.0)],
        v_limits=(0.0, 1.0),
        omega_limits=(-1.0, 1.0),
    )

    assert isinstance(prior, SamplingPrior)
    assert prior.mean_control_sequence((9.0, 8.0, 7.0), 4) == [
        (1.0, 0.5),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]

    prior.update([(0.4, -0.3)])
    assert prior.mean_control_sequence((0.0, 0.0, 0.0), 2) == [
        (0.4, -0.3),
        (0.4, -0.3),
    ]


def test_goal_warm_start_prior_matches_trusted_helper_and_requires_goal():
    signature = inspect.signature(GoalWarmStartPrior)
    assert signature.parameters["goal"].default is inspect.Parameter.empty

    kwargs = {
        "goal": (3.0, 3.0),
        "dt": 0.1,
        "v_max": 1.4,
        "omega_max": 1.0,
        "warm_start_prefix_steps": 4,
        "nominal_tail_control": (0.6, -0.1),
    }
    prior = GoalWarmStartPrior(**kwargs)

    expected = planner.build_goal_warm_start_sequence(
        current_state=START_STATE,
        horizon=7,
        **kwargs,
    )
    assert prior.mean_control_sequence(START_STATE, 7) == expected


def test_sampling_prior_module_has_no_torch_import():
    source_path = ROOT / "src" / "planners" / "sampling_prior.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert "torch" not in imported_roots


def test_protected_live_sim_goal_remains_fixed_at_three_three():
    live_sim = ROOT / "experiments" / "mujoco_memory_mppi_live_sim.py"
    tree = ast.parse(live_sim.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GOAL" for target in node.targets)
    ]

    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == (3.0, 3.0)
