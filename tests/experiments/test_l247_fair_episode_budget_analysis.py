from experiments.rl.analyze_l247_fair_episode_budget import (
    ARMS,
    BUDGETS,
    SCENES,
    SCENE_NAMES,
    SEED,
    evaluate_design_gate,
    validate_l247_rows,
)


def _rows():
    rows = []
    for scene in SCENES:
        for arm in ARMS:
            rows.append({
                "scene_key": scene,
                "scene": SCENE_NAMES[scene],
                "benchmark_arm": arm,
                "seed": str(SEED),
                "physics_domain": "nominal_seen",
                "qualification": "1",
                "max_steps_budget": str(BUDGETS[scene]),
                "rollout_budget_per_decision": "100",
                "paper_iterations": "2" if arm == "full_proposed" else "1",
                "termination_reason": (
                    "boundary_violation" if arm == "full_proposed" else "max_steps"
                ),
                "steps": (
                    "885" if arm == "full_proposed" else str(BUDGETS[scene])
                ),
            })
    return rows


def test_l247_integrity_contract_accepts_the_complete_matrix():
    assert validate_l247_rows(_rows()) == []
    gate = evaluate_design_gate(_rows())
    assert gate["gate_passed"] is True
    assert gate["old_700_step_limit_failures"] == []


def test_l247_integrity_contract_rejects_missing_or_wrong_budget_rows():
    rows = _rows()
    rows.pop()
    assert validate_l247_rows(rows)
    rows = _rows()
    rows[0]["max_steps_budget"] = "700"
    assert any("budget mismatch" in item for item in validate_l247_rows(rows))


def test_l247_gate_rejects_a_remaining_old_limit_failure():
    rows = _rows()
    row = next(
        item for item in rows
        if item["scene_key"] == "hairpin" and item["benchmark_arm"] == "icode_mppi"
    )
    row["steps"] = "700"
    row["termination_reason"] = "max_steps"
    gate = evaluate_design_gate(rows)
    assert gate["gate_passed"] is False
    assert gate["old_700_step_limit_failures"] == ["hairpin/icode_mppi"]
