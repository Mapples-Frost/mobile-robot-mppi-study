import numpy as np

from experiments.rl.calibrate_residual_reliability_gate import (
    _candidate_grid,
    _evaluate_candidate,
)


def _calibration():
    return {
        "state_names": ["v", "omega"],
        "state_scales": [1.0, 1.0],
        "rise_rate": 1.0,
        "fall_rate": 1.0,
        "active_alpha_threshold": 0.5,
        "matched_domain": "matched",
        "beneficial_domain": "beneficial",
        "parameter_grid": {
            "forgetting_factor": [0.9, 0.95],
            "minimum_samples": [2],
            "confidence_z": [0.0],
            "off_threshold": [0.0],
            "on_threshold": [0.2],
        },
        "selection_gate": {
            "minimum_beneficial_domain_mean_alpha": 0.5,
            "maximum_matched_domain_mean_alpha": 0.1,
            "minimum_worst_model_block_alpha_separation": 0.4,
            "minimum_beneficial_domain_active_fraction": 0.5,
            "maximum_matched_domain_active_fraction": 0.1,
        },
    }


def _sequence(block, domain, nominal_magnitude, residual_magnitude):
    nominal = np.asarray([0.0, 0.0, 0.0, nominal_magnitude, nominal_magnitude])
    residual = np.asarray([0.0, 0.0, 0.0, residual_magnitude, residual_magnitude])
    return {
        "model_block": block,
        "physics_domain": domain,
        "state_indices": [3, 4],
        "state_scales": np.ones(2),
        "errors": [(nominal, residual)] * 10,
    }


def test_grid_is_cartesian_and_causal_candidate_separates_domains():
    calibration = _calibration()
    assert len(_candidate_grid(calibration)) == 2
    sequences = []
    for block in range(3):
        sequences.append(_sequence(block, "matched", 0.1, 0.2))
        sequences.append(_sequence(block, "beneficial", 0.3, 0.05))
    result = _evaluate_candidate(
        _candidate_grid(calibration)[0], sequences, calibration, 0
    )
    assert result["passed"]
    assert result["cold_start_violations"] == 0
    # One fail-closed cold-start action remains in the episode average.
    assert result["worst_block_alpha_separation"] > 0.85


def test_candidate_fails_when_innovations_do_not_separate_domains():
    calibration = _calibration()
    sequences = []
    for block in range(3):
        sequences.append(_sequence(block, "matched", 0.3, 0.05))
        sequences.append(_sequence(block, "beneficial", 0.3, 0.05))
    result = _evaluate_candidate(
        _candidate_grid(calibration)[0], sequences, calibration, 0
    )
    assert not result["passed"]
    assert result["worst_block_alpha_separation"] == 0.0
