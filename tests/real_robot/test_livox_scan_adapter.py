import numpy as np
import pytest

from mobile_robot_mppi.real_robot.livox_scan_adapter import (
    LivoxPointCloudFrame,
    LivoxScanAdapter,
    LivoxScanAdapterConfig,
)


def _frame(points, timestamp_ns=2_500_000_000):
    points = np.asarray(points, dtype=np.float32)
    return LivoxPointCloudFrame(
        timestamp_ns=timestamp_ns,
        points=points,
        reflectivity=np.full(points.shape[0], 20, dtype=np.uint8),
        tags=np.zeros(points.shape[0], dtype=np.uint8),
    )


def test_rejects_mismatched_raw_fields():
    with pytest.raises(ValueError, match="counts must match"):
        LivoxPointCloudFrame(
            timestamp_ns=0,
            points=np.zeros((2, 3), dtype=np.float32),
            reflectivity=np.zeros(1, dtype=np.uint8),
            tags=np.zeros(2, dtype=np.uint8),
        )

    with pytest.raises(ValueError, match="timestamp count"):
        LivoxPointCloudFrame(
            timestamp_ns=10,
            points=np.zeros((2, 3), dtype=np.float32),
            reflectivity=np.zeros(2, dtype=np.uint8),
            tags=np.zeros(2, dtype=np.uint8),
            point_timestamps_ns=np.zeros(1, dtype=np.int64),
        )


def test_invalid_self_and_height_points_are_removed():
    adapter = LivoxScanAdapter(LivoxScanAdapterConfig(beam_count=360))
    scan, diagnostics = adapter.convert(_frame([
        (0.0, 0.0, 0.0),       # explicit invalid return
        (0.25, 0.0, 0.20),      # robot body
        (2.00, 0.0, -0.30),     # below configured height ROI
        (2.00, 0.0, 0.20),      # valid obstacle
    ]))
    assert diagnostics.input_points == 4
    assert diagnostics.finite_nonzero_points == 3
    assert diagnostics.height_points == 2
    assert diagnostics.nonself_points == 1
    assert diagnostics.populated_beams == 1
    assert np.min(scan.ranges) == pytest.approx(2.0)
    assert scan.timestamp == pytest.approx(2.5)


def test_nearest_point_wins_within_an_angular_beam():
    adapter = LivoxScanAdapter(LivoxScanAdapterConfig(beam_count=360))
    scan, diagnostics = adapter.convert(_frame([
        (3.0, 0.01, 0.25),
        (1.5, 0.01, 0.25),
    ]))
    occupied = np.flatnonzero(np.isfinite(scan.obstacle_ranges))
    assert diagnostics.populated_beams == 1
    assert occupied.size == 1
    assert scan.obstacle_ranges[occupied[0]] == pytest.approx(
        np.hypot(1.5, 0.01), rel=1e-6
    )


def test_extrinsic_translation_is_applied_before_self_filtering():
    transform = (
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    adapter = LivoxScanAdapter(LivoxScanAdapterConfig(
        lidar_to_base=transform,
        beam_count=360,
    ))
    scan, diagnostics = adapter.convert(_frame([(1.0, 0.0, 0.2)]))
    assert diagnostics.populated_beams == 1
    assert np.min(scan.ranges) == pytest.approx(2.0)


def test_convert_with_points_preserves_filtered_vertical_geometry():
    adapter = LivoxScanAdapter(LivoxScanAdapterConfig(beam_count=360))
    scan, diagnostics, base_points = adapter.convert_with_points(_frame([
        (1.5, -0.1, 0.20),
        (1.5, 0.0, 0.80),
        (1.5, 0.1, 1.40),
        (0.2, 0.0, 0.20),  # self return
    ]))
    assert diagnostics.nonself_points == 3
    assert base_points.shape == (3, 3)
    assert np.ptp(base_points[:, 2]) == pytest.approx(1.20)
    assert base_points.flags.writeable is False
    assert diagnostics.populated_beams > 0
    assert scan.timestamp == pytest.approx(2.5)
