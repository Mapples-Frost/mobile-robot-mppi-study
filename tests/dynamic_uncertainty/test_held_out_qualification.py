import inspect
from pathlib import Path

import numpy as np
import yaml

from experiments.dynamic_uncertainty.run_held_out_predictor_qualification import (
    build_schedule,
)
from mobile_robot_mppi.obstacles.patrol import validate_v3_config
from mobile_robot_mppi.obstacles.prediction import run_online_forecasts
from mobile_robot_mppi.obstacles.qualification import (
    deep_merge_ood_config,
    exact_sign_flip_pvalue,
    holm_adjust,
    paired_seed_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_ood_overlay_preserves_generator_and_physical_limits():
    base = _yaml(
        ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"
    )
    override = _yaml(
        ROOT
        / "configs/research/dynamic_obstacle_process_v3_ood.yaml"
    )
    merged = deep_merge_ood_config(base, override)
    validate_v3_config(merged)
    assert merged["physical_limits"] == base["physical_limits"]
    assert tuple(merged["noise_profiles"]) == (
        "process_shift",
        "sensor_shift",
    )
    assert (
        merged["navigation"]["maximum_cruise_speed_mps"]
        > base["navigation"]["maximum_cruise_speed_mps"]
    )
    assert base["noise_profiles"] == _yaml(
        ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"
    )["noise_profiles"]


def test_exact_sign_flip_uses_only_seed_level_vector():
    assert np.isclose(exact_sign_flip_pvalue([-1.0, -1.0]), 0.5)
    assert exact_sign_flip_pvalue([-1.0] * 10) < 0.01


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"id": 0.01, "ood": 0.04, "secondary": 0.03})
    assert adjusted["id"] <= adjusted["secondary"]
    assert adjusted["secondary"] <= adjusted["ood"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_paired_summary_bootstraps_whole_seed_differences_reproducibly():
    differences = np.asarray((-3.0, -2.5, -2.0, -1.5))
    first = paired_seed_summary(differences, 123, 1000)
    second = paired_seed_summary(differences, 123, 1000)
    assert first == second
    assert first["seed_count"] == 4
    assert first["bootstrap_95_ci"][1] < 0.0


def test_held_out_schedule_is_complete_paired_and_seed_disjoint():
    config = _yaml(
        ROOT / "configs/research/held_out_predictor_qualification.yaml"
    )
    registry = _yaml(
        ROOT / "configs/seeds/dynamic_uncertainty_splits.yaml"
    )
    sealed = _yaml(ROOT / registry["sealed_registry"]["path"])
    jobs = build_schedule(config)

    assert len(jobs) == 160
    assert len({job["experimental_key"] for job in jobs}) == 160
    assert sorted(job["run_order"] for job in jobs) == list(range(160))

    development = set(
        range(
            registry["splits"]["development"]["seed_start"],
            registry["splits"]["development"]["seed_start"]
            + registry["splits"]["development"]["count"],
        )
    )
    sealed_seeds = set(sealed["seeds"])
    expected = {
        "held_out_id": set(range(730200001, 730200011)),
        "held_out_ood": set(range(730300001, 730300011)),
    }
    for split, expected_seeds in expected.items():
        selected = [job for job in jobs if job["split"] == split]
        used_seeds = {job["seed"] for job in selected}
        assert len(selected) == 80
        assert used_seeds == expected_seeds
        assert used_seeds.isdisjoint(development)
        assert used_seeds.isdisjoint(sealed_seeds)
        assert all(
            sum(job["seed"] == seed for job in selected) == 8
            for seed in expected_seeds
        )


def test_online_prediction_api_cannot_receive_truth_or_change_labels():
    parameters = tuple(inspect.signature(run_online_forecasts).parameters)
    assert parameters == (
        "times",
        "observations",
        "observed_mask",
        "predictor",
        "horizon",
        "forecast_dt",
        "forecast_stride_steps",
        "warmup_steps",
    )
    forbidden = {"truth", "states", "modes", "change_flags", "events"}
    assert forbidden.isdisjoint(parameters)
