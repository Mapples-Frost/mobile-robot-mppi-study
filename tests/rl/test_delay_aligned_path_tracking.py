from mobile_robot_mppi.core.config import load_yaml


def test_l47_freezes_delay_aware_design_and_disjoint_seeds():
    config = load_yaml("configs/rl/icode_delay_aligned_path_tracking_l47.yaml")
    design = config["rl"]["cross_layer_factorial"]
    assert design["planner_known_command_delay"] is True
    assert design["conditions"] == [
        "traditional_nominal", "traditional_icode", "traditional_icode_support_gate"
    ]
    assert [block["icode_seed"] for block in design["model_blocks"]] == [
        20261001, 20261002, 20261003
    ]
    assert set(design["development_episode_seeds"]).isdisjoint(
        design["sealed_confirmation_episode_seeds"]
    )

