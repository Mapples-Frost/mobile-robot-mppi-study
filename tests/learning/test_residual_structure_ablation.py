from pathlib import Path

import numpy as np
import pytest

from experiments.rl.summarize_residual_structure_offline import (
    _relative_advantage,
)
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.learning.models import ResidualNetwork


ROOT = Path(__file__).resolve().parents[2]
ICODE_CONFIG = ROOT / "configs/research/icode_high_dynamic_h36_l57.yaml"
MLP_CONFIG = ROOT / "configs/research/mlp_high_dynamic_h36_l60.yaml"


def _statistics(state_dim=5, control_dim=2, angle_indices=(2,)):
    feature_dim = state_dim + len(angle_indices)
    return {
        "feature_mean": np.zeros(feature_dim, dtype=np.float32),
        "feature_scale": np.ones(feature_dim, dtype=np.float32),
        "control_mean": np.zeros(control_dim, dtype=np.float32),
        "control_scale": np.ones(control_dim, dtype=np.float32),
        "residual_mean": np.zeros(state_dim, dtype=np.float32),
        "residual_scale": np.ones(state_dim, dtype=np.float32),
    }


def test_l60_mlp_is_parameter_matched_to_l57_icode():
    icode_config = load_yaml(ICODE_CONFIG)
    mlp_config = load_yaml(MLP_CONFIG)
    statistics = _statistics()
    icode = ResidualNetwork(5, 2, icode_config["model"], statistics)
    mlp = ResidualNetwork(5, 2, mlp_config["model"], statistics)
    relative_difference = abs(
        mlp.parameter_count() - icode.parameter_count()
    ) / float(icode.parameter_count())

    assert icode.model_type == "icode_residual"
    assert mlp.model_type == "mlp_residual"
    assert tuple(mlp_config["model"]["hidden_sizes"]) == (93, 93)
    assert relative_difference <= mlp_config["structure_ablation"][
        "maximum_parameter_count_relative_difference"
    ]


def test_l60_reuses_frozen_seeds_horizons_and_dataset_hashes():
    icode = load_yaml(ICODE_CONFIG)
    mlp = load_yaml(MLP_CONFIG)
    design = mlp["structure_ablation"]
    assert design["required_training_seeds"] == icode["offline_gate"][
        "required_training_seeds"
    ]
    assert design["required_horizons"] == icode["offline_gate"][
        "required_horizons"
    ]
    assert mlp["offline_gate"]["expected_dataset_sha256"] == icode[
        "offline_gate"
    ]["expected_dataset_sha256"]


def test_relative_advantage_is_positive_only_for_lower_first_error():
    assert _relative_advantage(0.8, 1.0) == pytest.approx(0.2)
    assert _relative_advantage(1.2, 1.0) == pytest.approx(-0.2)
    assert _relative_advantage(1.0, 1.0) == pytest.approx(0.0)
