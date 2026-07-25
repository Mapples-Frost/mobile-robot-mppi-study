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


def test_batched_projection_matches_scalar_geometry_and_progress_windows():
    reference = PolylineReference(
        ((-2.0, -2.0), (2.0, 2.0), (-2.0, 2.0), (2.0, -2.0)),
        projection_backtrack_distance=0.5,
        projection_forward_distance=1.0,
    )
    positions = np.asarray((
        (0.0, 0.0),
        (1.7, 1.8),
        (-0.5, 2.3),
        (1.0, -0.8),
    ))
    floors = np.asarray((1.0, 2.0, 6.0, 11.0))

    batched = reference.project_batch(positions, minimum_progress=floors)
    scalar = [
        reference.project(position, minimum_progress=floor)
        for position, floor in zip(positions, floors)
    ]

    np.testing.assert_allclose(
        batched.point, np.asarray([value.point for value in scalar])
    )
    for name in (
        "progress",
        "remaining",
        "normalized_progress",
        "tangent_heading",
        "cross_track_error",
        "signed_cross_track_error",
        "curvature",
    ):
        np.testing.assert_allclose(
            getattr(batched, name),
            np.asarray([getattr(value, name) for value in scalar]),
            rtol=0.0,
            atol=1.0e-12,
        )
    np.testing.assert_array_equal(
        batched.segment_index,
        np.asarray([value.segment_index for value in scalar]),
    )


def test_batched_target_poses_match_scalar_preview_targets():
    reference = PolylineReference(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0)),
        lookahead_distance=0.35,
    )
    positions = np.asarray(((0.2, 0.1), (0.9, 0.4), (1.1, 1.7)))
    projections = reference.project_batch(positions, minimum_progress=0.0)

    batched = reference.target_poses_at_progress(projections.progress)
    scalar = np.asarray([
        reference.preview_target_at(
            0.0,
            np.asarray((position[0], position[1], 0.0)),
            progress_floor=0.0,
        ).pose.as_array()
        for position in positions
    ])

    np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=1.0e-12)
