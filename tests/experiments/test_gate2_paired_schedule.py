from mobile_robot_mppi.evaluation.paired_checkpoint import (
    paired_checkpoint_schedule,
)


def test_paired_schedule_contains_both_variants_once_per_block():
    domains = [{"name": "seen"}, {"name": "unseen"}]
    scenes = [{"name": "simple"}, {"name": "corridor"}]
    schedule = paired_checkpoint_schedule((21, 22), domains, scenes, 7)

    assert len(schedule) == 16
    blocks = {}
    for row in schedule:
        blocks.setdefault(row["block"], []).append(row)
    assert len(blocks) == 8
    for rows in blocks.values():
        assert {row["variant"] for row in rows} == {
            "control", "value_aligned"
        }
        assert {row["run_order_within_block"] for row in rows} == {0, 1}


def test_paired_schedule_is_reproducible():
    domains = [{"name": "seen"}]
    scenes = [{"name": "simple"}]

    assert paired_checkpoint_schedule(
        (1, 2, 3), domains, scenes, 11
    ) == (
        paired_checkpoint_schedule((1, 2, 3), domains, scenes, 11)
    )
