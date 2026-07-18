import copy

import numpy as np

from experiments.rl.run_contextual_covariance_jerk_screening import _blocked_schedule
from experiments.rl.run_cross_layer_sample_efficiency_factorial import (
    METRICS,
    OPTIONAL_METRICS,
    _audit_rows,
    _configure_arm,
    _difference_in_differences,
    _dynamics_gate,
    _final_package_gate,
    _half_budget_gate,
)


def _arms():
    result = []
    for dynamics in ("nominal", "icode_residual"):
        for sampler, budget in (("fixed", 50), ("contextual_bandit", 50), ("fixed", 100)):
            result.append({
                "name": "%s_%s_%d" % (dynamics, sampler, budget),
                "prediction_mode": dynamics,
                "sampler_kind": sampler,
                "num_samples": budget,
                "fixed_covariance_scale": [1.75, 0.75],
            })
    return result


def _spec():
    return {
        "scenes": ["a.yaml", "b.yaml"],
        "physics_domains": [{"name": "anchor", "plant_override": {}}],
        "seeds": [1, 2, 3],
        "schedule_seed": 17,
        "arms": _arms(),
        "icode_checkpoint": "icode.pt",
        "bandit_checkpoint": "bandit.json",
        "shared_override": {
            "action_space": {"rate_limits": [1.1, 2.5]},
            "planner": {"control_rate_weight": 0.16},
        },
    }


def test_six_arm_schedule_balances_every_run_position():
    schedule = _blocked_schedule(_spec())
    assert len(schedule) == 36
    counts = {
        arm["name"]: [0] * 6 for arm in _spec()["arms"]
    }
    for item in schedule:
        counts[item["arm"]["name"]][item["run_position"]] += 1
    assert all(values == [1] * 6 for values in counts.values())


def test_configure_arm_separates_dynamics_and_sampler():
    base = {
        "planner": {"checkpoint": "old.pt"},
        "rl": {"enabled": True, "checkpoint": "old.json"},
        "action_space": {},
    }
    spec = _spec()
    nominal = _configure_arm(base, spec["arms"][0], spec)
    assert nominal["planner"]["prediction_mode"] == "nominal"
    assert "checkpoint" not in nominal["planner"]
    assert nominal["planner"]["sampling_prior"] == "fixed_covariance"
    assert nominal["rl"]["enabled"] is False
    assert nominal["action_space"]["rate_limits"] == [1.1, 2.5]

    contextual_arm = copy.deepcopy(spec["arms"][4])
    learned = _configure_arm(base, contextual_arm, spec)
    assert learned["planner"]["prediction_mode"] == "icode_residual"
    assert learned["planner"]["checkpoint"] == "icode.pt"
    assert learned["planner"]["sampling_prior"] == "contextual_bandit_covariance"
    assert learned["rl"]["checkpoint"] == "bandit.json"


def test_factorial_audit_requires_complete_finite_balanced_blocks():
    spec = _spec()
    rows = []
    for item in _blocked_schedule(spec):
        row = {
            "scene": item["scene_path"],
            "physics_domain": item["physics_domain"]["name"],
            "seed": item["seed"],
            "condition": item["arm"]["name"],
            "run_position": item["run_position"],
            "success": True,
            "collision": False,
        }
        row.update({metric: 1.0 for metric in METRICS})
        row.update({metric: None for metric in OPTIONAL_METRICS})
        rows.append(row)
    audit = _audit_rows(rows, spec)
    assert audit["passed"] is True
    assert audit["optional_metric_missing_counts"]["minimum_clearance"] == len(rows)
    broken = copy.deepcopy(rows)
    broken[0]["cross_track_rmse"] = np.nan
    assert _audit_rows(broken, spec)["passed"] is False


def _contrast():
    result = {
        "success_delta_mean": 0.0,
        "collision_delta_mean": 0.0,
        "cross_track_rmse_delta_ci95": [-0.004, -0.001],
        "elapsed_s_delta_ci95": [-2.0, -0.3],
        "planner_compute_ms_mean_delta_ci95": [-12.0, -8.0],
    }
    return result


def test_primary_gates_encode_superiority_and_noninferiority_contracts():
    assert _dynamics_gate(_contrast(), 38.0, 50.0)["gate_passed"] is True
    assert _half_budget_gate(_contrast(), 0.002)["gate_passed"] is True
    assert _final_package_gate(_contrast(), 38.0, 50.0)["gate_passed"] is True

    inconclusive = _contrast()
    inconclusive["cross_track_rmse_delta_ci95"] = [-0.001, 0.001]
    assert _dynamics_gate(inconclusive, 38.0, 50.0)["gate_passed"] is False
    assert _half_budget_gate(inconclusive, 0.002)["gate_passed"] is True


def test_absolute_compute_deadline_is_not_a_relative_nominal_claim():
    contrast = _contrast()
    contrast["planner_compute_ms_mean_delta_ci95"] = [10.0, 20.0]
    assert _final_package_gate(contrast, 49.9, 50.0)["gate_passed"] is True
    assert _final_package_gate(contrast, 50.1, 50.0)["gate_passed"] is False


def test_difference_in_differences_uses_complete_episode_blocks():
    rows = []
    values = {
        "a1": 7.0,
        "a0": 10.0,
        "b1": 9.0,
        "b0": 10.0,
    }
    for seed in (1, 2, 3):
        for condition, value in values.items():
            row = {
                "scene": "route",
                "physics_domain": "plant",
                "seed": seed,
                "condition": condition,
            }
            row.update({metric: value for metric in METRICS})
            rows.append(row)
    result = _difference_in_differences(rows, "a1", "a0", "b1", "b0", 9)
    assert result["paired_episodes"] == 3
    assert result["cross_track_rmse_mean"] == -2.0
    assert result["cross_track_rmse_ci95"] == [-2.0, -2.0]
