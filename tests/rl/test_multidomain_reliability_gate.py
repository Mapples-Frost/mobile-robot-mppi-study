import copy

import pytest

from experiments.rl.analyze_multidomain_reliability_gate import analyze


ARMS = (
    "ordinary_fixed",
    "value_fixed",
    "ordinary_adaptive",
    "full_proposed",
)
DOMAINS = (
    "high_mass_seen",
    "low_friction_seen",
    "long_delay_seen",
    "combined_unseen",
)


def _rows():
    rows = []
    for domain_index, domain in enumerate(DOMAINS):
        for seed in (569, 570, 571):
            for arm in ARMS:
                ordinary_error = 0.05 + 0.005 * domain_index
                full_error = ordinary_error * (
                    0.90 if domain == "combined_unseen" else 0.97
                )
                error = {
                    "ordinary_fixed": ordinary_error,
                    "value_fixed": ordinary_error * 0.98,
                    "ordinary_adaptive": ordinary_error * 0.99,
                    "full_proposed": full_error,
                }[arm]
                adaptive = arm in (
                    "ordinary_adaptive", "full_proposed"
                )
                rows.append({
                    "factorial_arm": arm,
                    "physics_domain": domain,
                    "seed": seed,
                    "success": "True",
                    "collision": "False",
                    "cross_track_rmse": error,
                    "control_jerk": (
                        0.205 if arm == "full_proposed" else 0.20
                    ),
                    "rollout_budget_per_decision": 100,
                    "paper_iterations": 2,
                    "reliability_dynamics_confidence_mean": (
                        0.45
                        if domain == "combined_unseen" and adaptive
                        else 0.80
                    ),
                    "reliability_actor_competence_mean": (
                        0.55 if adaptive else 1.0
                    ),
                    "reliability_actor_competence_min": (
                        0.35 if adaptive else 1.0
                    ),
                    "reliability_actor_competence_max": (
                        0.75 if adaptive else 1.0
                    ),
                    "reliability_guided_fraction_raw_applied_mean": (
                        0.40 if adaptive else 0.30
                    ),
                })
    return rows


def test_multidomain_gate_passes_complete_directional_design():
    result = analyze(
        _rows(), [569, 570, 571], DOMAINS
    )
    assert result["gate_passed"] is True
    assert result["episode_count"] == 48
    assert result["checks"]["unseen_dynamics_confidence_lower"]


def test_multidomain_gate_rejects_unseen_regression():
    rows = _rows()
    for row in rows:
        if (
            row["physics_domain"] == "combined_unseen"
            and row["factorial_arm"] == "full_proposed"
        ):
            row["cross_track_rmse"] = 0.20
    result = analyze(rows, [569, 570, 571], DOMAINS)
    assert result["gate_passed"] is False
    assert result["checks"]["unseen_tracking_directional"] is False


def test_multidomain_gate_rejects_constant_competence():
    rows = _rows()
    for row in rows:
        if row["factorial_arm"] in (
            "ordinary_adaptive", "full_proposed"
        ):
            row["reliability_actor_competence_min"] = 0.5
            row["reliability_actor_competence_max"] = 0.5
    result = analyze(rows, [569, 570, 571], DOMAINS)
    assert result["gate_passed"] is False
    assert result["checks"]["adaptive_mechanism_exercised"] is False


def test_multidomain_gate_rejects_duplicate_block():
    rows = _rows()
    changed = copy.deepcopy(rows)
    changed[-1] = dict(changed[0])
    with pytest.raises(ValueError, match="duplicate"):
        analyze(changed, [569, 570, 571], DOMAINS)
