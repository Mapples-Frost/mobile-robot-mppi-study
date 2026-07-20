import numpy as np

from mobile_robot_mppi.core.references import PolylineReference


def test_terminal_polyline_target_is_endpoint_with_nonzero_tolerance():
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0)),
        tolerance=0.25,
        lookahead_distance=0.45,
    )

    target = reference.target_at(1.0, np.asarray((1.9, 0.95, 0.0)))

    assert target.is_terminal
    assert target.phase == "terminal"
    assert np.isclose(target.pose.x, 2.0)
    assert np.isclose(target.pose.y, 1.0)
    assert np.isclose(target.position_tolerance, 0.25)


def test_previewing_terminal_action_does_not_advance_live_progress():
    reference = PolylineReference(((0.0, 0.0), (2.0, 0.0)))

    target = reference.preview_target_at(
        0.0, np.asarray((1.9, 0.0, 0.0)), progress_floor=1.8
    )

    assert target.is_terminal
    assert reference.progress == 0.0
