from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.evaluation.tracking import (
    FootprintCorridor,
    TrackingEventMonitor,
)


def test_obstacle_pass_event_fires_once_after_projected_obstacle_progress():
    reference = PolylineReference(((0.0, 0.0), (10.0, 0.0)))
    monitor = TrackingEventMonitor(
        reference,
        FootprintCorridor(1.0, 0.25),
        obstacle_positions=((3.0, 0.0),),
        pass_distance=0.2,
    )

    assert not monitor.update((3.1, 0.4, 0.0), 1.0).obstacle_pass_event
    passed = monitor.update((3.3, 0.4, 0.0), 2.0)
    after = monitor.update((4.0, 0.3, 0.0), 3.0)

    assert passed.obstacle_pass_event
    assert passed.obstacle_pass_index == 0
    assert not after.obstacle_pass_event
