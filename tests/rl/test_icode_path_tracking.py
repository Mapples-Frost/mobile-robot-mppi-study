import numpy as np

from experiments.rl.summarize_icode_path_tracking import project_polyline
from mobile_robot_mppi.core.config import load_yaml


def test_project_polyline_has_exact_distance_heading_and_completion():
    points = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    xy = np.asarray([[0.0, 0.0], [1.0, 0.3], [2.0, 0.0]])
    theta = np.asarray([0.0, 0.2, 0.0])
    distance, heading, completion = project_polyline(points, xy, theta)
    np.testing.assert_allclose(distance, [0.0, 0.3, 0.0], atol=1e-12)
    np.testing.assert_allclose(heading, [0.0, 0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(completion, [0.0, 0.5, 1.0], atol=1e-12)


def test_l43_design_is_residual_only_and_seeds_are_disjoint():
    config = load_yaml("configs/rl/icode_path_tracking_eligibility_l43.yaml")
    design = config["rl"]["cross_layer_factorial"]
    assert design["conditions"] == ["traditional_nominal", "traditional_icode"]
    assert set(design["development_episode_seeds"]).isdisjoint(
        design["sealed_confirmation_episode_seeds"]
    )
    assert len(design["model_blocks"]) == 3
    assert len(design["scenes"]) == 3
    for item in design["scenes"]:
        scene = load_yaml(item["path"])
        assert scene["task"]["type"] == "polyline"
        assert scene["scene"]["obstacles"] == []

