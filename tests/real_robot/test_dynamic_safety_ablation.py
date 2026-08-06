"""Focused tests for the physical perception/control ablation switches."""

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import ControlCommand
from mobile_robot_mppi.safety.arbiter import ScanGuardArbiter


ACTION = ActionSpec(
    ("v_cmd", "omega_cmd"),
    np.asarray((-0.30, -0.60), dtype=np.float64),
    np.asarray((0.50, 0.60), dtype=np.float64),
)


def test_dynamic_arbitration_disabled_preserves_mppi_proposal():
    arbiter = ScanGuardArbiter(
        ACTION,
        {"dynamic_safety_arbitration_enabled": False},
    )
    proposed = ControlCommand(np.asarray((0.42, -0.31)))
    decision = arbiter.arbitrate(
        proposed,
        {
            "reason": "temporal_slowdown",
            "should_slow_down": True,
            "slow_scale": 0.1,
            "emergency_stop": False,
        },
    )
    assert np.allclose(decision.executed_control.values, proposed.values)
    assert decision.overridden is False
    assert decision.reason == "dynamic_safety_arbitration_disabled"
    assert decision.diagnostics["final_motion_owner"] == "mppi"
    assert decision.diagnostics["hard_safety_only"] is True


def test_dynamic_arbitration_disabled_keeps_emergency_translation_stop():
    arbiter = ScanGuardArbiter(
        ACTION,
        {"dynamic_safety_arbitration_enabled": False},
    )
    proposed = ControlCommand(np.asarray((0.42, 0.31)))
    decision = arbiter.arbitrate(
        proposed,
        {
            "reason": "near_body_hard_stop",
            "emergency_stop": True,
        },
    )
    assert decision.executed_control.v == 0.0
    assert decision.executed_control.omega == proposed.omega
    assert decision.overridden is True
    assert decision.reason == "near_body_hard_stop"
    assert decision.diagnostics["hard_safety_emergency_stop_applied"] is True

