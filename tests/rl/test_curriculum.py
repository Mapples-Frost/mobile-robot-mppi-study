import numpy as np

from mobile_robot_mppi.rl.trainer import SceneCurriculum


def test_scene_curriculum_returns_normalized_piecewise_weights():
    curriculum = SceneCurriculum(
        {
            "enabled": True,
            "phases": [
                {"name": "easy", "until_step": 10, "scene_weights": [2.0, 0.0]},
                {"name": "mixed", "until_step": 20, "scene_weights": [1.0, 3.0]},
            ],
        },
        scene_count=2,
    )
    name, weights = curriculum.at(0)
    assert name == "easy"
    np.testing.assert_allclose(weights, (1.0, 0.0))
    name, weights = curriculum.at(10)
    assert name == "mixed"
    np.testing.assert_allclose(weights, (0.25, 0.75))
    name, weights = curriculum.at(100)
    assert name == "mixed"
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_disabled_curriculum_is_uniform():
    curriculum = SceneCurriculum({}, scene_count=3)
    name, weights = curriculum.at(123)
    assert name == "uniform"
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))
