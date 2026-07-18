from mobile_robot_mppi.core.config import load_yaml


def test_l48_confirmation_has_new_seeds_and_frozen_gate():
    config = load_yaml("configs/rl/icode_efficiency_confirmation_l48.yaml")
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == ["traditional_nominal", "traditional_icode"]
    assert len(design["development_episode_seeds"]) == 10
    assert set(design["development_episode_seeds"]).isdisjoint(
        design["sealed_confirmation_episode_seeds"]
    )
    gate = design["confirmation_gate"]
    assert gate["minimum_positive_path_length_model_blocks"] == 3
    assert gate["minimum_positive_control_jerk_model_blocks"] == 3
    assert gate["minimum_mean_path_length_reduction_m"] == 0.03
    assert gate["maximum_relative_cross_track_rmse_increase"] == 0.05

