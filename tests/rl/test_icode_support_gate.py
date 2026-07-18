from experiments.rl.run_cross_layer_factorial import CONDITION_SPECS, _condition_config
from mobile_robot_mppi.core.config import load_yaml


def test_l42_support_gate_thresholds_are_frozen_and_seeds_disjoint():
    config = load_yaml("configs/rl/icode_support_gate_l42.yaml")
    design = config["rl"]["cross_layer_factorial"]
    assert design["support_gate"] == {"soft_z": 3.0, "hard_z": 5.0}
    assert set(design["development_episode_seeds"]).isdisjoint(design["sealed_confirmation_episode_seeds"])


def test_support_gate_condition_is_registered():
    spec = CONDITION_SPECS["traditional_icode_support_gate"]
    assert spec["policy"] == "traditional"
    assert spec["residual_support_gate"] is True
