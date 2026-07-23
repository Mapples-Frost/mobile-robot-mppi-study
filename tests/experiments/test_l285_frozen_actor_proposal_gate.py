from pathlib import Path

import yaml

from experiments.rl.run_l285_frozen_actor_proposal_gate import (
    L285_ARMS,
    evaluate,
    frozen_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/research/tracking_l285_frozen_actor_proposal_gate.yaml"
PROTOCOL = ROOT / "docs/experiments/post_l217/l285_frozen_actor_proposal_gate_protocol.md"


def _row(arm, seed, scene, completion, cte, goal, collision=False, boundary=0):
    return {
        "experimental_arm": arm,
        "seed": seed,
        "scene": scene,
        "physics_domain": "nominal_seen",
        "path_completion_ratio": completion,
        "cross_track_rmse": cte,
        "final_goal_distance": goal,
        "return": completion * 10.0 - cte,
        "collision": collision,
        "boundary_violation_steps": boundary,
        "boundary_safe_success": not collision and boundary == 0,
        "reliability_proposal_authority_mean": 0.2 if arm != "icode_value_control" else 0.0,
        "reliability_proposal_fallback_fraction_mean": 0.1 if arm != "icode_value_control" else 0.0,
        "rl_elite_fraction_mean": 0.1 if arm != "icode_value_control" else 0.0,
    }


def _positive_rows():
    rows = []
    for seed in (1, 2, 3):
        for scene in ("wave", "switchback", "loop_exit"):
            rows.extend([
                _row("icode_value_control", seed, scene, 0.40, 1.00, 5.0),
                _row("source_full", seed, scene, 0.40, 1.00, 5.0),
                _row("l276_full", seed, scene, 0.46, 0.80, 4.6),
                _row("l281_full", seed, scene, 0.45, 0.82, 4.7),
                _row("l284_full", seed, scene, 0.44, 0.85, 4.8),
            ])
    return rows


def test_l285_contract_is_frozen_and_final_maps_are_absent():
    raw = CONFIG.read_text(encoding="utf-8").lower()
    protocol = PROTOCOL.read_text(encoding="utf-8")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["final_benchmark"]
    assert config["status"] == "preregistered"
    assert tuple(config["l285_arms"]) == L285_ARMS
    assert config["study_mode"] == "resource_limited_direction_screen"
    assert config["development_seeds"] == [20264511]
    assert len(config["paired_blocks"]) == 1
    assert all("l285_frozen_actor_proposal" in path for path in config["scene_configs"])
    for path in config["scene_configs"]:
        scene = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        assert scene["task"]["corridor_half_width"] == 0.80
        assert scene["task"]["footprint_radius"] == 0.20
        assert scene["task"]["completion_corridor"] == 0.75
    assert "final Hairpin, S-Chicane, and Infinity" in protocol
    assert "mujoco_tracking_grand" not in raw
    assert "20264311" not in raw


def test_l285_schedule_is_blocked_randomized_and_complete():
    scenes = [{"name": name} for name in ("a", "b", "c")]
    domains = [{"name": "nominal_seen"}]
    first = frozen_schedule((1, 2, 3), scenes, domains, 77)
    second = frozen_schedule((1, 2, 3), scenes, domains, 77)
    assert first == second
    assert len(first) == 45
    for seed in (1, 2, 3):
        for scene in ("a", "b", "c"):
            arms = {
                row["experimental_arm"]
                for row in first
                if row["seed"] == seed and row["scene"] == scene
            }
            assert arms == set(L285_ARMS)


def test_l285_positive_gate_uses_l276_not_descriptive_arms():
    frozen = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["final_benchmark"]
    frozen["study_mode"] = "confirmatory_gate"
    frozen["development_seeds"] = [1, 2, 3]
    result = evaluate(_positive_rows(), frozen, {"contract": True})
    assert result["gate_pass"]
    assert result["decision"] == "frozen_l276_actor_proposal_gate_pass"
    assert result["descriptive_arms_cannot_override_primary_gate"]


def test_l285_l281_cannot_rescue_failed_l276_primary_contrast():
    frozen = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["final_benchmark"]
    frozen["study_mode"] = "confirmatory_gate"
    frozen["development_seeds"] = [1, 2, 3]
    rows = _positive_rows()
    for row in rows:
        if row["experimental_arm"] == "l276_full":
            row["path_completion_ratio"] = 0.30
            row["cross_track_rmse"] = 1.40
            row["final_goal_distance"] = 5.5
    result = evaluate(rows, frozen, {"contract": True})
    assert not result["gate_pass"]
    assert result["early_stopping_probe_authorized"]
    assert not result["larger_validation_preregistration_authorized"]


def test_l285_resource_limited_screen_never_claims_confirmatory_gate():
    frozen = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["final_benchmark"]
    frozen["development_seeds"] = [1]
    frozen["gate"]["minimum_seeds_with_cte_or_goal_improvement"] = 1
    rows = [row for row in _positive_rows() if row["seed"] == 1]
    result = evaluate(rows, frozen, {"contract": True})
    assert result["screen_pass"]
    assert not result["gate_pass"]
    assert result["decision"] == "frozen_l276_actor_proposal_screen_promising"
