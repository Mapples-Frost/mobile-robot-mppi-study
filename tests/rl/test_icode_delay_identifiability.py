import copy

from experiments.rl.summarize_icode_delay_identifiability import summarize
from mobile_robot_mppi.core.config import load_yaml


def _row(block, scene, domain, seed, condition, success, collision, distance):
    return {
        "model_block": str(block), "scene": scene, "physics_domain": domain,
        "episode_seed": str(seed), "condition": condition,
        "success": str(bool(success)), "collision": str(bool(collision)),
        "final_goal_distance": str(distance), "planner_compute_ms_mean": "1.0",
    }


def test_l38_config_keeps_confirmation_seeds_disjoint():
    config = load_yaml("configs/rl/icode_delay_identifiability_l38.yaml")
    design = config["rl"]["cross_layer_factorial"]
    assert set(design["development_episode_seeds"]).isdisjoint(
        design["sealed_confirmation_episode_seeds"]
    )
    assert [domain["plant_override"]["actuator"]["command_delay"] for domain in design["physics_domains"]] == [0.04, 0.10]


def test_l38_pair_direction_uses_nominal_minus_icode_distance():
    from experiments.rl.summarize_icode_delay_identifiability import _paired_effects

    rows = [
        _row(0, "s", "combined_matched_delay", 1, "traditional_nominal", 0, 1, 2.0),
        _row(0, "s", "combined_matched_delay", 1, "traditional_icode", 1, 0, 1.5),
    ]
    effect = _paired_effects(rows)[0]
    assert effect["success_difference"] == 1
    assert effect["collision_difference"] == -1
    assert effect["goal_distance_improvement_m"] == 0.5
