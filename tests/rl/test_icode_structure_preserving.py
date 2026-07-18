from experiments.rl.run_cross_layer_factorial import CONDITION_SPECS
from mobile_robot_mppi.core.config import load_yaml


def test_l44_freezes_mask_gate_and_disjoint_seeds():
    config = load_yaml("configs/rl/icode_structure_preserving_l44.yaml")
    design = config["rl"]["cross_layer_factorial"]
    assert design["residual_component_mask"] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert design["support_gate"] == {"soft_z": 3.0, "hard_z": 5.0}
    assert set(design["development_episode_seeds"]).isdisjoint(
        design["sealed_confirmation_episode_seeds"]
    )
    assert CONDITION_SPECS["traditional_icode_dynamic_mask"]["residual_component_mask"]
    primary = CONDITION_SPECS["traditional_icode_dynamic_mask_support_gate"]
    assert primary["residual_component_mask"] and primary["residual_support_gate"]

