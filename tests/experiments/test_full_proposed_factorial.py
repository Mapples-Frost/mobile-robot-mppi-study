from experiments.rl.run_full_proposed_factorial import (
    ARMS,
    factorial_schedule,
)


def test_factorial_schedule_is_complete_reproducible_and_blocked():
    domains = [{"name": "nominal"}, {"name": "shift"}]
    scenes = [{"name": "clean"}]
    first = factorial_schedule((48, 49), domains, scenes, 20260736)
    second = factorial_schedule((48, 49), domains, scenes, 20260736)

    assert first == second
    assert len(first) == 2 * 2 * 1 * 4
    blocks = {}
    for job in first:
        blocks.setdefault(job["block"], []).append(job)
    assert len(blocks) == 4
    for jobs in blocks.values():
        assert {job["arm"] for job in jobs} == set(ARMS)
        assert sorted(
            job["run_order_within_block"] for job in jobs
        ) == [0, 1, 2, 3]
