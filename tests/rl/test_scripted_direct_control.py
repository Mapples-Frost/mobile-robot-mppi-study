from types import SimpleNamespace

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.rl.scripted_direct_control import (
    ScriptedDirectControlConfig,
    ScriptedPolylineDirectControl,
)


def _spec():
    return ActionSpec(
        names=("v_cmd", "omega_cmd"),
        lower=np.asarray((0.0, -0.9)),
        upper=np.asarray((0.35, 0.9)),
        rate_limits=np.asarray((0.6, 2.0)),
    )


def test_direct_teacher_emits_finite_normalized_controls_and_diagnostics():
    teacher = ScriptedPolylineDirectControl(
        np.asarray(((0.0, 0.0), (2.0, 0.0))),
        _spec(),
        ScriptedDirectControlConfig(cruise_speed=0.28),
    )

    action, diagnostics = teacher.action(
        SimpleNamespace(x=0.0, y=0.0, theta=0.0)
    )

    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert diagnostics["desired_v"] == 0.28
    assert abs(diagnostics["desired_omega"]) < 1e-12


def test_direct_teacher_stops_translation_for_large_heading_error():
    teacher = ScriptedPolylineDirectControl(
        np.asarray(((0.0, 0.0), (2.0, 0.0))), _spec()
    )

    action, diagnostics = teacher.action(
        SimpleNamespace(x=0.0, y=0.0, theta=np.pi / 2.0)
    )

    assert diagnostics["desired_v"] == 0.0
    assert diagnostics["desired_omega"] < 0.0
    assert action[0] == -1.0
