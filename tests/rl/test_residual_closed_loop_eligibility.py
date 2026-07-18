from mobile_robot_mppi.core.config import load_yaml


def test_l41_uses_new_disjoint_episode_seeds():
    config = load_yaml("configs/rl/icode_closed_loop_eligibility_l41.yaml")
    design = config["rl"]["cross_layer_factorial"]
    development = set(design["development_episode_seeds"])
    sealed = set(design["sealed_confirmation_episode_seeds"])
    assert development.isdisjoint(sealed)
    assert development == set(range(20860731, 20860736))


def test_l41_keeps_rl_out_of_conditions():
    config = load_yaml("configs/rl/icode_closed_loop_eligibility_l41.yaml")
    assert config["rl"]["cross_layer_factorial"]["conditions"] == [
        "traditional_nominal", "traditional_icode"
    ]
