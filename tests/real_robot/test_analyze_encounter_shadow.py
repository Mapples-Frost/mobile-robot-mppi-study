import json

import pytest

from deploy.raspberry_pi5_scout.analyze_encounter_shadow import (
    analyze_rows,
    latest_run,
    load_cycles,
)


def _row(cycle, **values):
    diagnostics = {
        "encounter_candidate_mode": "unknown",
        "encounter_confirmed_mode": "unknown",
        "encounter_phase": "idle",
        "encounter_track_index": None,
        "encounter_track_continuous": False,
        "encounter_change_reason": "no_track_history",
        "encounter_change_detected": False,
        "encounter_confidence": 0.0,
        "encounter_strategy": "none",
        "encounter_locked_steering_side": 0,
        "encounter_line_crossed": False,
    }
    diagnostics.update(values)
    return {
        "cycle": cycle,
        "timestamp": 10.0 + 0.1 * cycle,
        "encounter_mode_shadow": diagnostics,
        "timing_ms": {"encounter": 0.08 + 0.01 * cycle},
    }


def test_left_to_right_audit_reports_confirmation_and_direction_consistency():
    rows = [
        _row(0),
        _row(
            1,
            encounter_candidate_mode="straight_crossing",
            encounter_track_index=2,
            encounter_change_reason="no_track_history",
            encounter_human_lateral_velocity_mps=-0.5,
        ),
        _row(
            2,
            encounter_candidate_mode="straight_crossing",
            encounter_track_index=2,
            encounter_track_continuous=True,
            encounter_change_reason="continuous",
            encounter_human_lateral_velocity_mps=-0.5,
        ),
        _row(
            3,
            encounter_candidate_mode="straight_crossing",
            encounter_confirmed_mode="straight_crossing",
            encounter_phase="straight_crossing",
            encounter_track_index=2,
            encounter_track_continuous=True,
            encounter_change_reason="continuous",
            encounter_confidence=0.92,
            encounter_human_lateral_velocity_mps=-0.5,
            encounter_strategy="behind_pass",
            encounter_locked_steering_side=1,
        ),
    ]

    result = analyze_rows(rows, expected="left_to_right")

    assert result["usable"] is True
    assert result["confirmation_delay_cycles"] == 2
    assert result["confirmation_delay_s"] == pytest.approx(0.2)
    assert result["direction_consistency"]["consistent_count"] == 1
    assert result["checks"]["observed_left_to_right_velocity"] is True
    assert result["all_checks_passed"] is True


def test_crossing_yield_is_directionally_consistent_without_a_locked_side():
    rows = [
        _row(
            0,
            encounter_candidate_mode="straight_crossing",
            encounter_track_index=2,
            encounter_change_reason="no_track_history",
            encounter_human_lateral_velocity_mps=-0.5,
        ),
        _row(
            1,
            encounter_candidate_mode="straight_crossing",
            encounter_confirmed_mode="straight_crossing",
            encounter_phase="straight_crossing",
            encounter_track_index=2,
            encounter_track_continuous=True,
            encounter_change_reason="continuous",
            encounter_confidence=0.92,
            encounter_human_lateral_velocity_mps=-0.5,
            encounter_strategy="yield",
            encounter_locked_steering_side=0,
        ),
    ]

    result = analyze_rows(rows, expected="left_to_right")

    assert result["direction_consistency"]["sample_count"] == 1
    assert result["direction_consistency"]["consistent_count"] == 1
    assert result["checks"]["observed_left_to_right_velocity"] is True


def test_cross_to_frontal_requires_real_change_event():
    rows = [
        _row(
            0,
            encounter_candidate_mode="straight_crossing",
            encounter_confirmed_mode="straight_crossing",
            encounter_phase="straight_crossing",
            encounter_track_index=1,
            encounter_change_reason="no_track_history",
        ),
        _row(
            1,
            encounter_candidate_mode="frontal_approach",
            encounter_confirmed_mode="straight_crossing",
            encounter_phase="straight_crossing",
            encounter_track_index=1,
            encounter_track_continuous=True,
            encounter_change_detected=True,
            encounter_change_score=0.8,
            encounter_change_reason="ca_imm+velocity_direction",
        ),
        _row(
            2,
            encounter_candidate_mode="frontal_approach",
            encounter_confirmed_mode="frontal_approach",
            encounter_phase="frontal_approach",
            encounter_track_index=1,
            encounter_track_continuous=True,
            encounter_change_reason="continuous",
        ),
    ]

    result = analyze_rows(rows, expected="cross_to_frontal")

    assert result["checks"]["confirmed_crossing"] is True
    assert result["checks"]["confirmed_frontal"] is True
    assert result["checks"]["detected_behaviour_change"] is True
    assert result["change_events"][0]["cycle"] == 1


def test_identity_jump_is_reported_and_fails_continuity_check():
    rows = [
        _row(
            0,
            encounter_track_index=1,
            encounter_change_reason="no_track_history",
        ),
        _row(
            1,
            encounter_track_index=4,
            encounter_track_continuous=False,
            encounter_change_reason="track_index_changed",
            encounter_track_residual_m=0.0,
        ),
    ]

    result = analyze_rows(rows)

    assert result["track_discontinuities"][0]["cycle"] == 1
    assert result["checks"]["no_track_identity_jump"] is False


def test_load_and_latest_run_ignore_empty_failed_start(tmp_path):
    older = tmp_path / "20260805_100000"
    newer_failed = tmp_path / "20260805_100100"
    older.mkdir()
    newer_failed.mkdir()
    row = _row(0)
    (older / "cycles.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (newer_failed / "cycles.jsonl").write_text("", encoding="utf-8")

    selected = latest_run(tmp_path)
    loaded = load_cycles(selected / "cycles.jsonl")

    assert selected == older
    assert loaded[0]["cycle"] == 0


def test_missing_shadow_rows_are_rejected():
    result = analyze_rows([{"cycle": 0, "timestamp": 1.0}])

    assert result["usable"] is False
    assert result["reason"] == "no_encounter_mode_shadow_rows"
