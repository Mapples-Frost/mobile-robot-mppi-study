from types import SimpleNamespace

import pytest

from mobile_robot_mppi.evaluation.metrics import EpisodeMetrics


def _truth(timestamp):
    return SimpleNamespace(
        timestamp=float(timestamp),
        pose=SimpleNamespace(x=0.1 * timestamp, y=0.0, theta=0.0),
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


def test_episode_metrics_preserve_proposal_execution_diagnostics():
    metrics = EpisodeMetrics(1.0, 0.0, 0.2, control_dt=0.1)
    diagnostics = {
        "supervised_elite_weight_sum_final_iteration": 0.25,
        "supervised_counterfactual_available": True,
        "supervised_counterfactual_first_action_delta_norm": 0.4,
        "supervised_counterfactual_sequence_delta_norm": 1.2,
        "supervised_pre_guard_counterfactual_first_action_delta_norm": 0.5,
        "supervised_pre_guard_counterfactual_sequence_delta_norm": 1.3,
        "supervised_post_guard_counterfactual_first_action_delta_norm": 0.3,
        "supervised_post_guard_counterfactual_sequence_delta_norm": 1.0,
        "supervised_pre_guard_action": "0.2,0.1",
        "supervised_post_guard_action": "0.1,0.0",
        "supervised_post_guard_action_delta_norm": 0.15,
        "supervised_post_guard_replacement_reason": "geometry_guard",
        "supervised_influence_survived_guard": True,
        "supervised_influence_survival_ratio": 0.75,
    }
    metrics.update(_truth(1.0), _decision(), diagnostics)
    metrics.update(
        _truth(2.0),
        _decision(),
        {"supervised_counterfactual_available": False},
    )

    first = metrics.records[0]
    summary = metrics.summary()

    assert first["supervised_counterfactual_available"] == 1.0
    assert first["supervised_pre_guard_action"] == "0.2,0.1"
    assert first["supervised_post_guard_action"] == "0.1,0.0"
    assert first["supervised_influence_survived_guard"] == 1.0
    assert summary[
        "supervised_elite_weight_sum_final_iteration_mean"
    ] == pytest.approx(0.125)
    assert summary[
        "supervised_counterfactual_available_fraction"
    ] == pytest.approx(0.5)
    assert summary[
        "supervised_counterfactual_first_action_delta_norm_mean_active"
    ] == pytest.approx(0.4)
    assert summary[
        "supervised_counterfactual_sequence_delta_norm_mean_active"
    ] == pytest.approx(1.2)
    assert summary[
        "supervised_influence_survived_guard_fraction_active"
    ] == pytest.approx(1.0)
    assert summary[
        "supervised_influence_survival_ratio_mean_active"
    ] == pytest.approx(0.75)
    assert summary[
        "supervised_post_guard_action_delta_norm_mean_active"
    ] == pytest.approx(0.15)
    assert summary[
        "supervised_post_guard_replacement_reason_counts"
    ] == {"geometry_guard": 1}


def test_episode_metrics_use_zero_defaults_without_counterfactual():
    metrics = EpisodeMetrics(1.0, 0.0, 0.2, control_dt=0.1)
    metrics.update(_truth(1.0), _decision(), {})

    record = metrics.records[0]
    summary = metrics.summary()

    assert record["supervised_counterfactual_available"] == 0.0
    assert record["supervised_post_guard_replacement_reason"] == "none"
    assert summary["supervised_counterfactual_available_fraction"] == 0.0
    assert summary[
        "supervised_counterfactual_first_action_delta_norm_mean_active"
    ] == 0.0
    assert summary[
        "supervised_influence_survived_guard_fraction_active"
    ] == 0.0
    assert summary["supervised_post_guard_replacement_reason_counts"] == {}
