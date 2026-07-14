import numpy as np
import pytest

from mobile_robot_mppi.core.types import Pose2D
from mobile_robot_mppi.rl.scripted_subgoal import (
    ScriptedPolylineSubgoal,
    ScriptedSubgoalConfig,
)


def _prior():
    return {
        "kind": "local_subgoal",
        "subgoal_min_distance": 0.25,
        "subgoal_max_distance": 1.25,
        "subgoal_max_bearing": np.pi,
    }


def test_scripted_subgoal_emits_exact_local_subgoal_action_shape():
    policy = ScriptedPolylineSubgoal(
        ((0.0, 0.0), (2.0, 0.0)),
        _prior(),
        ScriptedSubgoalConfig(lookahead_distance=0.75),
    )
    action, metadata = policy.action(Pose2D(0.0, 0.0, 0.0))
    assert action.shape == (2,)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert metadata["target_x"] == pytest.approx(0.75)
    assert metadata["route_progress"] == pytest.approx(0.0)


def test_scripted_subgoal_uses_body_frame_bearing_and_monotonic_progress():
    policy = ScriptedPolylineSubgoal(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)), _prior()
    )
    first, _ = policy.action(Pose2D(0.5, 0.0, np.pi / 2.0))
    assert first[1] == pytest.approx(-0.5)
    progress = policy.progress
    policy.action(Pose2D(0.2, 0.0, 0.0))
    assert policy.progress >= progress
    policy.reset()
    assert policy.progress == 0.0


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 0.0),),
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (float("nan"), 1.0)),
    ],
)
def test_scripted_subgoal_rejects_invalid_polyline(points):
    with pytest.raises(ValueError):
        ScriptedPolylineSubgoal(points, _prior())


def test_scripted_subgoal_rejects_non_local_prior():
    with pytest.raises(ValueError, match="requires local_subgoal"):
        ScriptedPolylineSubgoal(((0.0, 0.0), (1.0, 0.0)), {})
