import inspect
from pathlib import Path

import numpy as np
import yaml

from mobile_robot_mppi.obstacles.change_detection_evaluation import (
    evaluate_change_detections,
)
from mobile_robot_mppi.obstacles.imm import (
    ChangeAwareIMMPredictor,
    run_online_imm_forecasts,
)


ROOT = Path(__file__).resolve().parents[2]
CHANGE_CONFIG_PATH = (
    ROOT / "configs/research/change_aware_imm_v3_development.yaml"
)
ORDINARY_CONFIG_PATH = (
    ROOT / "configs/research/ordinary_imm_v3_development.yaml"
)


def _predictor(
    candidate="single_99",
    observation_std=0.075,
    change_config_path=CHANGE_CONFIG_PATH,
):
    change = yaml.safe_load(
        Path(change_config_path).read_text(encoding="utf-8")
    )
    ordinary = yaml.safe_load(
        ORDINARY_CONFIG_PATH.read_text(encoding="utf-8")
    )
    imm = ordinary["ordinary_imm"]
    response = change["change_response"]
    detector = change["pilot_candidates"][candidate]
    return ChangeAwareIMMPredictor(
        observation_std=observation_std,
        initial_velocity_std=ordinary["prediction"][
            "initial_velocity_std"
        ],
        initial_mode_probabilities=imm["initial_mode_probabilities"],
        transition_matrix=imm["transition_matrix"],
        turn_rate_radps=imm["turn_rate_radps"],
        brake_decay_rate_per_s=imm["brake_decay_rate_per_s"],
        process_acceleration_std=imm["process_acceleration_std"],
        nis_threshold=detector["nis_threshold"],
        required_exceedances=detector["required_exceedances"],
        window_observations=detector["window_observations"],
        single_exceedance_threshold=detector.get(
            "single_exceedance_threshold"
        ),
        **response,
    )


def test_large_online_innovation_triggers_without_truth_input():
    predictor = _predictor()
    predictor.update(np.asarray((0.0, 0.0)), 0.0)
    predictor.update(np.asarray((0.04, 0.0)), 0.05)
    predictor.update(np.asarray((3.0, -2.0)), 0.10)
    assert predictor.last_change_triggered
    assert predictor.last_innovation_nis > predictor.nis_threshold
    assert predictor.detection_events[-1]["kind"] == "nis_change"
    np.testing.assert_allclose(
        predictor.mode_probabilities.sum(), 1.0, atol=1.0e-12
    )
    forecast = predictor.forecast(60, 0.05)
    forecast.validate()


def test_persistence_rule_rejects_one_isolated_exceedance():
    predictor = _predictor("persistent_95")
    predictor.update(np.asarray((0.0, 0.0)), 0.0)
    predictor.update(np.asarray((0.04, 0.0)), 0.05)
    predictor.update(np.asarray((3.0, 0.0)), 0.10)
    assert not predictor.last_change_triggered
    predictor.update(np.asarray((-3.0, 0.0)), 0.15)
    assert predictor.last_change_triggered


def test_extreme_single_nis_can_trigger_dual_rule():
    config_path = (
        ROOT
        / "configs/research/change_aware_imm_v3_development_amendment2.yaml"
    )
    predictor = _predictor(
        "dual_99_999", change_config_path=config_path
    )
    predictor.update(np.asarray((0.0, 0.0)), 0.0)
    predictor.update(np.asarray((0.04, 0.0)), 0.05)
    predictor.update(np.asarray((4.0, -3.0)), 0.10)
    assert predictor.last_change_triggered
    assert (
        predictor.detection_events[-1]["trigger_rule"]
        == "single_severe"
    )


def test_dropout_guard_is_separate_from_motion_change_trigger():
    predictor = _predictor()
    predictor.update(np.asarray((0.0, 0.0)), 0.0)
    for index in range(1, 7):
        predictor.update(None, 0.05 * index)
    kinds = [event["kind"] for event in predictor.detection_events]
    assert kinds == ["dropout_guard"]
    assert not predictor.last_change_triggered


def test_change_aware_runner_has_no_truth_or_event_parameter():
    parameters = tuple(inspect.signature(run_online_imm_forecasts).parameters)
    forbidden = {
        "truth",
        "truth_states",
        "metadata",
        "events",
        "change_flags",
        "future_schedule",
    }
    assert not forbidden.intersection(parameters)


def test_change_detection_matching_is_one_to_one_and_excludes_dropout():
    times = np.arange(0.0, 5.05, 0.05)
    change_flags = np.zeros(times.shape, dtype=bool)
    change_flags[40] = True
    change_flags[60] = True
    events = [
        {"kind": "dropout_guard", "timestamp": 2.1, "nis": None},
        {"kind": "nis_change", "timestamp": 2.4, "nis": 12.0},
        {"kind": "nis_change", "timestamp": 4.8, "nis": 15.0},
    ]
    result = evaluate_change_detections(
        events, times, change_flags, maximum_match_delay_s=1.5
    )
    assert result["true_change_count"] == 2
    assert result["matched_trigger_count"] == 1
    assert result["false_trigger_count"] == 1
    assert result["dropout_guard_count"] == 1
    assert np.isclose(result["true_change_recall"], 0.5)


def test_two_histories_with_same_prefix_produce_identical_prefix_records():
    times = np.arange(0.0, 1.05, 0.05)
    prefix = np.column_stack((0.4 * times, np.zeros(times.shape)))
    first = prefix.copy()
    second = prefix.copy()
    first[15:, 1] += 2.0
    second[15:, 1] -= 2.0
    mask = np.ones(times.shape, dtype=bool)
    records_first = run_online_imm_forecasts(
        times, first, mask, _predictor(), 4, 0.05
    )
    records_second = run_online_imm_forecasts(
        times, second, mask, _predictor(), 4, 0.05
    )
    for left, right in zip(records_first[:15], records_second[:15]):
        np.testing.assert_array_equal(
            left.filtered_mean, right.filtered_mean
        )
        np.testing.assert_array_equal(
            left.future.means, right.future.means
        )
