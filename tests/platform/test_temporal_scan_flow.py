import numpy as np
import pytest

from mobile_robot_mppi.core.types import LaserScan
from mobile_robot_mppi.perception.scan_flow import (
    RobustScanFlowEstimator,
    ScanFlowConfig,
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
