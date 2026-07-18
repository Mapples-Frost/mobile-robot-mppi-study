from types import SimpleNamespace

import pytest

from mobile_robot_mppi.evaluation.metrics import EpisodeMetrics


def _truth(time_value, x_value):
    return SimpleNamespace(
        timestamp=time_value,
        pose=SimpleNamespace(x=x_value, y=0.0, theta=0.0),
        twist=SimpleNamespace(v=0.1, omega=0.0),
        collision=False,
        minimum_clearance=0.5,
        slip_ratio=0.0,
    )


def _decision():
    control = SimpleNamespace(v=0.1, omega=0.0)
    return SimpleNamespace(
        proposed_control=control,
        executed_control=control,
        overridden=False,
        reason="none",
    )


def test_episode_metrics_record_and_summarize_policy_correction():
    metrics = EpisodeMetrics(1.0, 0.0, 0.2, control_dt=0.1)
    for index, correction in enumerate((0.02, 0.06)):
        metrics.update(
            _truth(0.1 * (index + 1), 0.4 * (index + 1)),
            _decision(),
            {
                "prior": {
                    "type": "rl_sac",
                    "policy_mode": "frozen_bc_correction",
                    "correction_gate_alpha": 1.0,
                    "base_action_abs_mean": 0.3,
                    "unit_correction_abs_mean": correction / 0.1,
                    "applied_correction_abs_mean": correction,
                    "applied_correction_abs_max": 2.0 * correction,
                    "raw_applied_correction_abs_mean": 1.5 * correction,
                    "correction_advantage_gate_mode": "hard",
                    "correction_advantage_critic_source": "target",
                    "correction_advantage_threshold": 0.0,
                    "correction_advantage_gate_alpha": float(index),
                    "correction_support_gate_enabled": True,
                    "correction_support_confidence": 0.25 + 0.5 * index,
                    "correction_effective_gate_alpha": 0.25 * index,
                    "online_conservative_advantage": -0.1 + 0.3 * index,
                    "target_conservative_advantage": -0.2 + 0.3 * index,
                }
            },
        )

    result = metrics.summary()

    assert metrics.records[0]["rl_policy_mode"] == "frozen_bc_correction"
    assert result["rl_policy_mode"] == "frozen_bc_correction"
    assert result["rl_correction_gate_alpha_mean"] == 1.0
    assert result["rl_base_action_abs_mean"] == pytest.approx(0.3)
    assert result["rl_applied_correction_abs_mean"] == pytest.approx(0.04)
    assert result["rl_applied_correction_abs_max"] == pytest.approx(0.12)
    assert result["rl_raw_applied_correction_abs_mean"] == pytest.approx(0.06)
    assert result["rl_correction_advantage_gate_mode"] == "hard"
    assert result["rl_correction_advantage_critic_source"] == "target"
    assert result["rl_correction_advantage_gate_alpha_mean"] == 0.5
    assert result["rl_correction_support_gate_enabled"] is True
    assert result["rl_correction_support_confidence_mean"] == 0.5
    assert result["rl_correction_support_confidence_min"] == 0.25
    assert result["rl_correction_effective_gate_alpha_mean"] == 0.125
    assert result["rl_online_conservative_advantage_mean"] == pytest.approx(0.05)
    assert result["rl_online_conservative_advantage_min"] == pytest.approx(-0.1)
    assert result["rl_online_conservative_advantage_max"] == pytest.approx(0.2)
    assert result["rl_online_positive_advantage_fraction"] == 0.5
    assert result["rl_target_conservative_advantage_mean"] == pytest.approx(-0.05)
    assert result["rl_target_positive_advantage_fraction"] == 0.5


def test_episode_metrics_reports_polyline_cross_track_error():
    metrics = EpisodeMetrics(
        2.0,
        1.0,
        0.2,
        control_dt=0.1,
        reference_points=((0.0, 1.0), (2.0, 1.0)),
    )
    for index in range(2):
        metrics.update(
            _truth(0.1 * (index + 1), 0.5 + index),
            _decision(),
            {},
        )

    result = metrics.summary()
    assert result["cross_track_rmse"] == pytest.approx(1.0)
    assert result["cross_track_mean"] == pytest.approx(1.0)
    assert result["cross_track_max"] == pytest.approx(1.0)


def test_episode_metrics_summarizes_hss_reliability_levels():
    metrics = EpisodeMetrics(1.0, 0.0, 0.2, control_dt=0.1)
    for index, level in enumerate(("low", "high")):
        metrics.update(
            _truth(0.1 * (index + 1), 0.2 * (index + 1)),
            _decision(),
            {
                "reliability_hss_enabled": True,
                "reliability_level": level,
                "reliability_authority": 0.2 + 0.6 * index,
                "reliability_guided_fraction_applied": 0.3 * index,
                "reliability_guided_fraction_next": 0.6 * index,
            },
        )

    result = metrics.summary()

    assert result["reliability_hss_enabled_fraction"] == 1.0
    assert result["reliability_low_fraction"] == 0.5
    assert result["reliability_high_fraction"] == 0.5
    assert result["reliability_guided_fraction_applied_mean"] == 0.15
