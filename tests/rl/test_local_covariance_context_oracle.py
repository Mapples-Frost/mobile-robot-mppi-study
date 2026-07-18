from experiments.rl.run_local_covariance_context_oracle import (
    _choose,
    evaluate_mapping,
)


def test_local_choice_is_lexicographic_precision_then_progress():
    candidates = {
        "narrow": {"collision_rate": 0.0, "cross_track_rmse": 0.040, "local_progress_m": 1.0},
        "speed": {"collision_rate": 0.0, "cross_track_rmse": 0.041, "local_progress_m": 1.2},
    }
    assert _choose(candidates, 0.002) == "speed"
    assert _choose(candidates, 0.0005) == "narrow"


def test_local_mapping_gate_requires_heterogeneity_and_progress():
    rows = []
    mapping = {
        ("route", "a"): "narrow",
        ("route", "b"): "narrow",
        ("route", "c"): "narrow",
        ("route", "d"): "speed",
    }
    for anchor, selected in (
        ("a", "narrow"),
        ("b", "narrow"),
        ("c", "narrow"),
        ("d", "speed"),
    ):
        for seed in (1, 2, 3):
            for candidate in ("narrow", "speed"):
                progress = 1.2 if candidate == selected else 1.0
                rows.append({
                    "scene": "route",
                    "anchor_id": anchor,
                    "physics_domain": "domain",
                    "seed": seed,
                    "candidate": candidate,
                    "local_progress_m": progress,
                    "cross_track_rmse": 0.04,
                    "control_jerk": 0.1,
                    "collision": False,
                })
    result = evaluate_mapping(rows, mapping, "speed", 0.002, 17)
    assert result["heterogeneity_gate_passed"]
    assert result["progress_gate_passed"]
    assert result["precision_gate_passed"]
    assert result["primary_gate_passed"]
