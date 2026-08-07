import numpy as np
import pytest

from mobile_robot_mppi.core.types import LaserScan
from mobile_robot_mppi.perception.scan_flow import (
    RobustScanFlowEstimator,
    ScanFlowConfig,
    TemporalSafetyHysteresis,
)


def _scan(ranges, timestamp):
    values = np.asarray(ranges, dtype=np.float64)
    return LaserScan(
        ranges=values,
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / float(values.size - 1),
        range_min=0.05,
        range_max=4.0,
        timestamp=float(timestamp),
    )


def _config(**updates):
    values = {
        "enabled": True,
        "window_beams": 7,
        "minimum_support_beams": 4,
        "hold_s": 0.0,
    }
    values.update(updates)
    return values


def test_contiguous_beam_consensus_reports_rate_ttc_and_support():
    estimator = RobustScanFlowEstimator(_config())
    previous = np.full(181, 4.0)
    current = previous.copy()
    previous[86:95] = 1.00
    current[86:95] = 0.96

    warmup = estimator.update(_scan(previous, 0.0))
    estimate = estimator.update(_scan(current, 0.1))

    assert warmup.valid is False
    assert warmup.reason == "warmup"
    assert estimate.valid is True
    assert estimate.reason == "closing_consensus"
    assert estimate.closing_rate_mps == pytest.approx(0.4)
    assert estimate.clearance_m == pytest.approx(0.96)
    assert estimate.ttc_s == pytest.approx(2.4)
    assert estimate.risk_alpha == pytest.approx(1.0)
    assert estimate.support_beams >= 4
    assert abs(estimate.center_angle_rad) < 0.2


def test_single_beam_reassociation_jump_is_rejected_not_clamped():
    estimator = RobustScanFlowEstimator(_config())
    previous = np.full(181, 4.0)
    current = previous.copy()
    current[90] = 0.20

    estimator.update(_scan(previous, 0.0))
    estimate = estimator.update(_scan(current, 0.1))

    assert estimate.valid is False
    assert estimate.reason == "insufficient_consensus"
    assert estimate.risk_alpha == 0.0
    assert estimate.support_beams == 0
    assert estimate.rejected_jump_fraction > 0.0


def test_hold_prevents_one_frame_risk_chatter_and_reset_clears_history():
    estimator = RobustScanFlowEstimator(_config(hold_s=0.3))
    previous = np.full(181, 4.0)
    approaching = previous.copy()
    previous[86:95] = 1.0
    approaching[86:95] = 0.96
    stationary = approaching.copy()

    estimator.update(_scan(previous, 0.0))
    active = estimator.update(_scan(approaching, 0.1))
    held = estimator.update(_scan(stationary, 0.2))
    estimator.reset()
    reset = estimator.update(_scan(stationary, 0.3))

    assert active.valid is True
    assert held.valid is True
    assert held.held is True
    assert held.reason == "held"
    assert held.risk_alpha == active.risk_alpha
    assert reset.valid is False
    assert reset.reason == "warmup"


def test_temporal_safety_hysteresis_enters_immediately_and_releases_confirmed():
    gate = TemporalSafetyHysteresis(
        enabled=True, release_clear_frames=2
    )

    slow, slow_diagnostics = gate.update("slow")
    stop, stop_diagnostics = gate.update("stop")
    held_stop, first_release = gate.update("slow")
    released_slow, second_release = gate.update("slow")
    held_slow, first_clear = gate.update("clear")
    released_clear, second_clear = gate.update("clear")

    assert slow == "slow"
    assert slow_diagnostics["temporal_scan_safety_entered_immediately"]
    assert stop == "stop"
    assert stop_diagnostics["temporal_scan_safety_entered_immediately"]
    assert held_stop == "stop"
    assert first_release["temporal_scan_safety_release_pending"]
    assert released_slow == "slow"
    assert second_release["temporal_scan_safety_release_confirmed"]
    assert held_slow == "slow"
    assert first_clear["temporal_scan_safety_release_pending"]
    assert released_clear == "clear"
    assert second_clear["temporal_scan_safety_release_confirmed"]


def test_temporal_safety_hysteresis_resets_pending_release_on_new_risk():
    gate = TemporalSafetyHysteresis(
        enabled=True, release_clear_frames=2
    )
    gate.update("stop")
    gate.update("clear")

    stop, diagnostics = gate.update("stop")

    assert stop == "stop"
    assert diagnostics["temporal_scan_safety_clear_streak"] == 0
    assert not diagnostics["temporal_scan_safety_release_pending"]


def _front_wall_scan(robot_x, timestamp, beams=721):
    angles = np.linspace(-np.pi, np.pi, beams)
    ranges = np.full(beams, np.inf)
    visible = np.abs(angles) <= 0.35
    ranges[visible] = (1.0 - float(robot_x)) / np.cos(angles[visible])
    return LaserScan(
        ranges=ranges,
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / float(beams - 1),
        range_min=0.05,
        range_max=4.0,
        timestamp=float(timestamp),
    )


def test_ego_motion_compensation_rejects_static_wall_closing_rate():
    uncompensated = RobustScanFlowEstimator(_config())
    compensated = RobustScanFlowEstimator(_config(
        ego_motion_compensation_enabled=True
    ))
    previous = _front_wall_scan(0.0, 0.0)
    current = _front_wall_scan(0.05, 0.1)

    uncompensated.update(previous, pose=(0.0, 0.0, 0.0))
    false_closing = uncompensated.update(
        current, pose=(0.05, 0.0, 0.0)
    )
    compensated.update(previous, pose=(0.0, 0.0, 0.0))
    static_result = compensated.update(
        current, pose=(0.05, 0.0, 0.0)
    )

    assert false_closing.valid is True
    assert false_closing.closing_rate_mps > 0.45
    assert static_result.valid is False
    assert static_result.reason == "insufficient_consensus"


def test_ego_motion_compensation_retains_independent_object_closing_rate():
    estimator = RobustScanFlowEstimator(_config(
        ego_motion_compensation_enabled=True
    ))
    previous = _front_wall_scan(0.0, 0.0)
    # The robot advances 0.05 m and the observed surface independently moves
    # 0.05 m toward it, so compensation should retain only the latter 0.5 m/s.
    current = _front_wall_scan(0.10, 0.1)

    estimator.update(previous, pose=(0.0, 0.0, 0.0))
    moving_result = estimator.update(
        current, pose=(0.05, 0.0, 0.0)
    )

    assert moving_result.valid is True
    assert moving_result.closing_rate_mps > 0.45


def test_continuous_temporal_slowdown_has_no_threshold_step():
    config = ScanFlowConfig.from_mapping({
        "safety_continuous_slowdown_enabled": True,
        "safety_hard_stop_ttc_s": 0.8,
        "safety_slow_ttc_s": 2.0,
        "safety_slow_scale": 0.25,
    })

    assert config.safety_slowdown_scale(2.0) == pytest.approx(1.0)
    assert config.safety_slowdown_scale(1.99) == pytest.approx(
        (1.99 - 0.8) / 1.2
    )
    assert config.safety_slowdown_scale(1.4) == pytest.approx(0.5)
    assert config.safety_slowdown_scale(0.81) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "mapping,match",
    [
        ({"window_beams": 4}, "odd"),
        ({"window_beams": 5, "minimum_support_beams": 6}, "support"),
        ({"rate_soft_mps": 0.3, "rate_hard_mps": 0.2}, "hard rate"),
        ({"ttc_hard_s": 2.0, "ttc_soft_s": 1.0}, "soft TTC"),
        ({"safety_slow_scale": 1.1}, "scale"),
    ],
)
def test_scan_flow_configuration_rejects_invalid_contracts(mapping, match):
    with pytest.raises(ValueError, match=match):
        ScanFlowConfig.from_mapping(mapping).validate()
