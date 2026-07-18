from experiments.rl.run_contextual_covariance_half_budget_confirmation import (
    _paired_schedule,
)


def test_confirmation_schedule_is_paired_and_order_balanced():
    spec = {
        "schedule_seed": 3,
        "scenes": ["a", "b"],
        "physics_domains": [{"name": "d"}],
        "seeds": [1, 2],
        "arms": [{"name": "learned"}, {"name": "fixed"}],
    }
    schedule = _paired_schedule(spec)
    assert len(schedule) == 8
    for index in range(0, len(schedule), 2):
        assert schedule[index]["pair_index"] == schedule[index + 1]["pair_index"]
        assert {schedule[index]["arm"]["name"], schedule[index + 1]["arm"]["name"]} == {
            "learned", "fixed"
        }
    assert schedule[0]["arm"]["name"] != schedule[2]["arm"]["name"]
