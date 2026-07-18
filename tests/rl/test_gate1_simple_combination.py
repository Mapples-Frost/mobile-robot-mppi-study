import pytest

from experiments.rl.run_gate1_simple_combination import (
    method_config,
    randomized_block_schedule,
)


def _base():
    return {
        "experiment": {"seed": 0},
        "planner": {
            "prediction_mode": "nominal",
            "num_samples": 20,
            "importance_sampling_correction": True,
        },
        "memory": {"enable": True},
    }


def test_gate1_factorial_has_equal_rollout_budget_and_correct_semantics():
    traditional = method_config(
        _base(), "traditional_mppi", False, False,
        "actor.pt", "icode.pt", 40, 2, 7,
    )
    combination = method_config(
        _base(), "simple_combination", True, True,
        "actor.pt", "icode.pt", 40, 2, 7,
    )

    assert traditional["planner"]["num_samples"] == 40
    assert traditional["planner"]["optimizer"] == "standard"
    assert traditional["planner"]["prediction_mode"] == "nominal"
    assert not traditional["rl"]["enabled"]
    assert not traditional["memory"]["enable"]

    assert combination["planner"]["num_samples"] == 20
    assert combination["planner"]["paper_rl_driven"]["iterations"] == 2
    assert combination["planner"]["optimizer"] == "paper_rl_driven"
    assert combination["planner"]["prediction_mode"] == "icode_residual"
    assert combination["planner"]["checkpoint"] == "icode.pt"
    assert combination["rl"]["checkpoint"] == "actor.pt"
    assert combination["experiment"]["rollout_budget_per_decision"] == 40


def test_gate1_factorial_rejects_unbalanced_paper_budget():
    with pytest.raises(ValueError, match="divisible"):
        method_config(
            _base(), "rl_driven_mppi", False, True,
            "actor.pt", "icode.pt", 41, 2, 7,
        )


def test_gate1_schedule_is_seeded_blocked_and_balanced():
    first = randomized_block_schedule((11, 12), 91)
    second = randomized_block_schedule((11, 12), 91)

    assert first == second
    assert len(first) == 8
    for seed in (11, 12):
        block = [row for row in first if row["seed"] == seed]
        assert {row["method"] for row in block} == {
            "traditional_mppi",
            "icode_mppi",
            "rl_driven_mppi",
            "simple_combination",
        }
        assert {row["run_order_within_block"] for row in block} == {
            0, 1, 2, 3,
        }


def test_gate1_schedule_blocks_each_seed_by_physics_domain():
    domains = (
        {"name": "seen", "role": "seen", "plant_override": {}},
        {
            "name": "unseen",
            "role": "unseen",
            "plant_override": {"actuator": {"command_delay": 0.1}},
        },
    )
    schedule = randomized_block_schedule((11, 12), 91, domains)

    assert len(schedule) == 16
    assert {row["block"] for row in schedule} == {
        "embedded__seen__seed_11",
        "embedded__seen__seed_12",
        "embedded__unseen__seed_11",
        "embedded__unseen__seed_12",
    }
    for block_name in {row["block"] for row in schedule}:
        block = [row for row in schedule if row["block"] == block_name]
        assert len(block) == 4
        assert {row["method"] for row in block} == {
            "traditional_mppi",
            "icode_mppi",
            "rl_driven_mppi",
            "simple_combination",
        }


def test_gate1_method_applies_domain_without_changing_factorial_cell():
    domain = {
        "name": "unseen_delay",
        "role": "unseen",
        "plant_override": {"actuator": {"command_delay": 0.1}},
    }
    result = method_config(
        _base(), "simple_combination", True, True,
        "actor.pt", "icode.pt", 40, 2, 7, domain,
    )

    assert result["plant"]["actuator"]["command_delay"] == 0.1
    assert result["experiment"]["physics_domain"] == "unseen_delay"
    assert result["experiment"]["physics_domain_role"] == "unseen"
    assert result["planner"]["prediction_mode"] == "icode_residual"


def test_gate1_schedule_blocks_scenes_independently():
    scenes = (
        {"name": "simple", "source": "simple.yaml"},
        {"name": "corridor", "source": "corridor.yaml"},
    )
    schedule = randomized_block_schedule(
        (11,), 91, scenes=scenes
    )

    assert len(schedule) == 8
    assert {row["block"] for row in schedule} == {
        "simple__embedded__seed_11",
        "corridor__embedded__seed_11",
    }
    for block_name in {row["block"] for row in schedule}:
        block = [row for row in schedule if row["block"] == block_name]
        assert len(block) == 4
