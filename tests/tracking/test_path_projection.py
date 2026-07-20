import numpy as np

from mobile_robot_mppi.core.references import PolylineReference


def test_projection_reports_arc_length_and_remaining_distance():
    reference = PolylineReference(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)))

    projection = reference.project(np.asarray((2.0, 1.0)))

    assert np.isclose(projection.progress, 3.0)
    assert np.isclose(projection.remaining, 1.0)
    assert np.isclose(projection.normalized_progress, 0.75)
    assert projection.segment_index == 1


def test_projection_progress_floor_prevents_self_crossing_regression():
    reference = PolylineReference(
        ((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
    )
    floor = 0.8 * reference.total_length

    projection = reference.project(np.asarray((1.0, 1.0)), minimum_progress=floor)

    assert projection.progress >= floor


def test_projection_window_preserves_branch_identity_at_self_crossing():
    reference = PolylineReference(
        ((-2.0, -2.0), (2.0, 2.0), (-2.0, 2.0), (2.0, -2.0)),
        projection_backtrack_distance=0.5,
        projection_forward_distance=1.0,
    )
    floor = 1.0

    projection = reference.project(np.asarray((0.0, 0.0)), minimum_progress=floor)

    assert floor - 0.5 <= projection.progress <= floor + 1.0
