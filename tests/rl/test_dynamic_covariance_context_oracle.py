from experiments.rl.run_dynamic_covariance_context_oracle import (
    _choose,
    evaluate_mapping,
)


def test_dynamic_choice_prioritizes_success_then_time():
    candidates = {
        "safe": {"success_rate": 1.0, "collision_rate": 0.0, "elapsed_s": 10.0, "final_goal_distance": 0.1},
        "fast": {"success_rate": 0.8, "collision_rate": 0.0, "elapsed_s": 5.0, "final_goal_distance": 0.1},
    }
    assert _choose(candidates) == "safe"


def test_dynamic_mapping_gate_passes_safe_time_improvement():
    rows = []
    mapping = {
        "clean": "speed",
        "dynamic_a": "turn",
        "dynamic_b": "turn",
        "dynamic_c": "turn",
    }
    for scene, selected in mapping.items():
        for seed in range(5):
            for candidate in ("speed", "turn"):
                elapsed = 8.0 if candidate == selected else 10.0
                rows.append({
                    "scene": scene,
                    "seed": seed,
                    "candidate": candidate,
                    "elapsed_s": elapsed,
                    "final_goal_distance": 0.1,
                    "control_jerk": 0.2,
                    "safety_interventions": 0,
                    "minimum_clearance": 0.3,
                    "success": True,
                    "collision": False,
                })
    result = evaluate_mapping(rows, mapping, "speed", 23)
    assert result["heterogeneity_gate_passed"]
    assert result["safety_gate_passed"]
    assert result["time_gate_passed"]
    assert result["primary_gate_passed"]


def test_dynamic_safety_confirmation_uses_binary_outcome_intervals():
    rows = []
    mapping = {
        "dynamic_%d" % index: ("adaptive_a" if index < 3 else "adaptive_b")
        for index in range(6)
    }
    for scene, selected in mapping.items():
        for seed in range(6):
            rows.extend([
                {
                    "scene": scene,
                    "seed": seed,
                    "candidate": selected,
                    "elapsed_s": 12.0,
                    "final_goal_distance": 0.1,
                    "control_jerk": 0.2,
                    "safety_interventions": 0,
                    "minimum_clearance": 0.4,
                    "success": True,
                    "collision": False,
                },
                {
                    "scene": scene,
                    "seed": seed,
                    "candidate": "global",
                    "elapsed_s": 10.0,
                    "final_goal_distance": 1.0,
                    "control_jerk": 0.3,
                    "safety_interventions": 1,
                    "minimum_clearance": 0.2,
                    "success": False,
                    "collision": True,
                },
            ])
    result = evaluate_mapping(
        rows,
        mapping,
        "global",
        29,
        gate_mode="safety_confirmation",
    )
    assert result["success_superiority_gate_passed"]
    assert result["collision_noninferiority_gate_passed"]
    assert result["primary_gate_passed"]
