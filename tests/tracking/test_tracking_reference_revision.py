import numpy as np
import pytest

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.evaluation.tracking import (
    FootprintCorridor,
    TrackingEventMonitor,
    tracking_sample,
)


def _corridor():
    return FootprintCorridor(half_width=1.0, footprint_radius=0.2)


def test_polyline_revision_changes_only_after_successful_replacement():
    reference = PolylineReference(((0.0, 0.0), (2.0, 0.0)))

    assert reference.revision == 0
    reference.reset()
    assert reference.revision == 0

    with pytest.raises(ValueError):
        reference.replace_points(((0.0, 0.0), (0.0, 0.0)))
    assert reference.revision == 0

    reference.replace_points(((0.0, 0.0), (0.0, 3.0)))
    assert reference.revision == 1
    reference.replace_points(((0.0, 0.0), (4.0, 0.0)))
    assert reference.revision == 2


def test_monitor_without_replacement_matches_side_effect_free_sample():
    reference = PolylineReference(((0.0, 0.0), (5.0, 0.0)))
    monitor = TrackingEventMonitor(reference, _corridor())
    state = np.asarray([1.25, 0.1, 0.0])

    expected = tracking_sample(reference, state, _corridor(), 0.0)
    actual = monitor.update(state, timestamp=0.1)

    assert reference.revision == 0
    assert actual.path_arc_length == expected.path_arc_length
    assert actual.signed_cross_track_error == (
        expected.signed_cross_track_error
    )
    assert actual.tangent_heading_error == expected.tangent_heading_error


@pytest.mark.parametrize(
    "replacement,state",
    [
        (((0.0, 0.0), (2.0, 0.0)), (0.5, 0.0, 0.0)),
        (((0.0, 0.0), (12.0, 0.0)), (0.5, 0.0, 0.0)),
    ],
)
def test_monitor_resets_route_state_after_shorter_or_longer_replacement(
    replacement,
    state,
):
    reference = PolylineReference(((0.0, 0.0), (10.0, 0.0)))
    monitor = TrackingEventMonitor(
        reference,
        _corridor(),
        obstacle_positions=((8.0, 0.0),),
    )
    monitor.update((9.0, 0.0, 0.0), timestamp=0.1)
    monitor.recovery_start_time = 0.1
    monitor.recovery_start_progress = monitor.progress

    reference.replace_points(replacement)
    sample = monitor.update(state, timestamp=0.2)

    assert np.isfinite(tuple(sample.to_dict().values())[:9]).all()
    assert monitor._last_reference_revision == reference.revision
    assert monitor.progress <= reference.total_length
    assert monitor.next_obstacle == 0
    assert monitor.recovery_start_time is None
    assert monitor.recovery_start_progress is None


def test_monitor_reprojects_obstacles_on_replacement_route():
    reference = PolylineReference(((0.0, 0.0), (5.0, 0.0)))
    monitor = TrackingEventMonitor(
        reference,
        _corridor(),
        obstacle_positions=((3.0, 0.0),),
    )
    assert monitor.obstacle_progress == (3.0,)

    reference.replace_points(((0.0, 0.0), (0.0, 5.0)))
    monitor.update((0.0, 0.5, np.pi / 2.0), timestamp=0.1)

    assert monitor.obstacle_progress == (0.0,)


def test_route_replacement_preserves_world_coordinate_center_crossing_state():
    reference = PolylineReference(((0.0, 0.0), (5.0, 0.0)))
    monitor = TrackingEventMonitor(
        reference,
        _corridor(),
        center_crossing=(1.0, 0.0),
        center_crossing_radius=0.2,
    )
    first = monitor.update((1.0, 0.0, 0.0), timestamp=0.1)
    assert first.center_crossing_count == 1

    reference.replace_points(((1.0, 0.0), (1.0, 3.0)))
    second = monitor.update((1.0, 0.5, np.pi / 2.0), timestamp=0.2)

    assert second.center_crossing_count == 1
    assert monitor._inside_center_crossing is False
