from mobile_robot_mppi.core.config import load_yaml


def test_l49_dataset_splits_are_episode_disjoint_and_cover_source_seeds():
    config = load_yaml("configs/rl/icode_iterative_path_data_l49.yaml")
    design = config["rl"]["cross_layer_factorial"]
    splits = config["residual_dataset"]["splits"]
    flattened = [seed for values in splits.values() for seed in values]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(design["development_episode_seeds"])
    assert config["residual_dataset"]["source_condition"] == "traditional_nominal"
    assert config["residual_dataset"]["control_source"] == "applied"

