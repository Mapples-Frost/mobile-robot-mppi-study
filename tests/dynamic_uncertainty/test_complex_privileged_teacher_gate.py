from pathlib import Path

import numpy as np
import yaml

from experiments.dynamic_uncertainty.audit_complex_privileged_teacher_gate import (
    _candidate_lattice,
    _icode_controller,
    _load_protocol,
)
from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import Pose2D, RobotObservation, Twist2D


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_actor_teacher_gate_v1.yaml"
)


def test_privileged_teacher_gate_freezes_scope_and_training_authority():
    protocol = _load_protocol(PROTOCOL)

    assert protocol["frozen_components"] == {
        "change_aware": True,
        "collision_risk": True,
        "icode": True,
        "mppi": True,
        "safety": True,
        "hss": True,
        "map_physics": True,
        "action_bounds": True,
        "proposal_actor_replaced": False,
    }
    assert set(protocol["saved_conflict_states"]) == {
        "chapter1",
        "chapter2",
        "chapter3",
    }
    assert protocol["gate"]["actor_training_authorized_only_on_pass"]
    assert not protocol["gate"]["formal_server_launch_authorized"]


def test_teacher_lattice_is_bounded_deterministic_and_multimodal():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    action_spec = ActionSpec(
        names=("v", "omega"),
        lower=np.asarray([-0.35, -0.9]),
        upper=np.asarray([0.7, 0.9]),
    )

    labels, first = _candidate_lattice(
        action_spec, protocol["teacher"]
    )
    repeated_labels, second = _candidate_lattice(
        action_spec, protocol["teacher"]
    )

    assert labels == repeated_labels
    assert np.array_equal(first, second)
    assert first.shape[1:] == (80, 2)
    assert np.all(first >= action_spec.lower[None, None, :])
    assert np.all(first <= action_spec.upper[None, None, :])
    assert any("left" in label for label in labels)
    assert any("right" in label for label in labels)
    assert any(label.startswith("wait_") for label in labels)
    assert any(label.startswith("reverse_") for label in labels)
    assert "stop" in labels
    assert "reverse" in labels


def test_teacher_gate_unwraps_residual_safety_shield():
    class Planner:
        prediction_mode = "icode_residual"

        def rollout(self):
            pass

    class Shield:
        residual_controller = Planner()

    planner = _icode_controller(Shield())

    assert planner is Shield.residual_controller


def test_forecast_contract_is_carried_in_observation_auxiliary():
    marker = object()
    observation = RobotObservation(
        timestamp=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        auxiliary={"probabilistic_obstacle_forecasts": (marker,)},
    )

    assert observation.auxiliary[
        "probabilistic_obstacle_forecasts"
    ] == (marker,)
