import numpy as np

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.evaluation.tracking import (
    FootprintCorridor,
    TrackingEventMonitor,
)


def test_recovery_time_and_distance_start_at_obstacle_pass():
    reference = PolylineReference(((0.0, 0.0), (10.0, 0.0)))
    monitor = TrackingEventMonitor(
        reference,
        FootprintCorridor(1.0, 0.25),
        obstacle_positions=((3.0, 0.0),),
        recovery_error=0.10,
    )

    passed = monitor.update((3.1, 0.40, 0.0), 2.0)
    recovering = monitor.update((3.6, 0.20, 0.0), 2.5)
    recovered = monitor.update((4.1, 0.05, 0.0), 3.0)

    assert passed.obstacle_pass_event and not passed.recovery_event
    assert np.isclose(recovering.recovery_time, 0.5)
    assert np.isclose(recovering.recovery_distance, 0.5)
    assert recovered.recovery_event
    assert np.isclose(recovered.recovery_time, 1.0)
    assert np.isclose(recovered.recovery_distance, 1.0)


def test_recovery_state_clears_after_event():
    reference = PolylineReference(((0.0, 0.0), (5.0, 0.0)))
    monitor = TrackingEventMonitor(
        reference,
        FootprintCorridor(1.0, 0.25),
        obstacle_positions=((1.0, 0.0),),
    )
    monitor.update((1.1, 0.4, 0.0), 1.0)
    monitor.update((1.5, 0.0, 0.0), 2.0)

    stable = monitor.update((2.0, 0.0, 0.0), 3.0)

    assert not stable.recovery_event
    assert stable.recovery_time == 0.0
    assert stable.recovery_distance == 0.0
