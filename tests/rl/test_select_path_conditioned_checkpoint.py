import pytest

from experiments.rl.select_path_conditioned_checkpoint import (
    aggregate_validation_rows,
)


def _row(step, seed, collision, success, cross_track, completion, reward):
    return {
        "global_step": str(step),
        "scene": "path",
        "seed": str(seed),
        "collision": str(bool(collision)),
        "success": str(bool(success)),
        "cross_track_rmse": str(cross_track),
        "path_completion_ratio": str(completion),
        "return": str(reward),
    }


def test_path_checkpoint_selection_is_lexicographic_not_scalar_reward():
    rows = [
        _row(10, 1, False, False, 0.20, 0.8, 1000.0),
        _row(10, 2, False, False, 0.22, 0.8, 1000.0),
        _row(20, 1, False, True, 0.30, 1.0, 10.0),
        _row(20, 2, False, False, 0.30, 0.9, 10.0),
        _row(30, 1, True, True, 0.01, 1.0, 5000.0),
        _row(30, 2, False, True, 0.01, 1.0, 5000.0),
    ]
    result = aggregate_validation_rows(rows, expected_steps=(10, 20, 30))
    assert result["selected_global_step"] == 20


def test_path_checkpoint_selection_uses_cross_track_after_safety_and_success():
    rows = [
        _row(10, 1, False, True, 0.20, 1.0, 20.0),
        _row(20, 1, False, True, 0.10, 0.8, 10.0),
    ]
    result = aggregate_validation_rows(rows, expected_steps=(10, 20))
    assert result["selected_global_step"] == 20


def test_path_checkpoint_selection_rejects_incomplete_or_duplicate_blocks():
    rows = [_row(10, 1, False, True, 0.1, 1.0, 1.0)]
    with pytest.raises(ValueError, match="missing"):
        aggregate_validation_rows(rows, expected_steps=(10, 20))

    rows.append(_row(10, 1, False, True, 0.1, 1.0, 1.0))
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_validation_rows(rows, expected_steps=(10,))
