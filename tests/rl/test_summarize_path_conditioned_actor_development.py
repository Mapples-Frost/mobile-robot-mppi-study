import csv

import pytest

from experiments.rl.summarize_path_conditioned_actor_development import (
    parse_candidate,
    summarize,
)


FIELDS = (
    "scene",
    "physics_domain",
    "seed",
    "success",
    "collision",
    "return",
    "cross_track_rmse",
    "cross_track_max",
    "heading_rmse",
    "path_completion_ratio",
    "control_jerk",
)


def write_rows(directory, rows):
    directory.mkdir()
    with (directory / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(seed, cross_track, success):
    return {
        "scene": "path",
        "physics_domain": "plant",
        "seed": seed,
        "success": success,
        "collision": False,
        "return": -cross_track,
        "cross_track_rmse": cross_track,
        "cross_track_max": cross_track * 2.0,
        "heading_rmse": cross_track,
        "path_completion_ratio": 0.9 if success else 0.4,
        "control_jerk": cross_track,
    }


def test_summary_uses_paired_seed_clusters(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_rows(baseline, [row(1, 1.0, False), row(2, 2.0, False)])
    write_rows(candidate, [row(1, 0.5, True), row(2, 1.0, True)])

    result = summarize(
        baseline,
        [("new", candidate)],
        bootstrap_samples=20,
        seed=7,
    )

    effects = result["candidates"]["new"]["paired_effects"]
    assert effects["independent_clusters"] == 2
    assert effects["metrics"]["success"]["favorable_effect"] == 1.0
    assert effects["metrics"]["cross_track_rmse"][
        "favorable_effect"
    ] == pytest.approx(0.75)


def test_candidate_parser_requires_label_and_directory():
    assert parse_candidate("run=/tmp/run") == ("run", "/tmp/run")
    with pytest.raises(ValueError, match="LABEL=DIRECTORY"):
        parse_candidate("missing_separator")
    with pytest.raises(ValueError, match="label"):
        parse_candidate(" =/tmp/run")
