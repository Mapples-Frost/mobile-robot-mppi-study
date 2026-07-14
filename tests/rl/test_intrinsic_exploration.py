import pytest

from mobile_robot_mppi.rl.intrinsic import (
    EpisodicPoseCountBonus,
    EpisodicPoseCountConfig,
)


def test_pose_count_bonus_rewards_new_bins_and_decays_revisits():
    novelty = EpisodicPoseCountBonus({
        "enabled": True,
        "weight": 0.08,
        "maximum_bonus": 0.1,
        "position_bin_size": 0.25,
        "heading_bins": 8,
        "count_exponent": 0.5,
    })
    reset = novelty.reset([0.0, 0.0, 0.0])
    same_bonus, same = novelty.observe([0.02, 0.03, 0.01])
    new_bonus, new = novelty.observe([0.30, 0.0, 0.0])

    assert reset["unique_bins"] == 1
    assert same["bin_count"] == 2
    assert same_bonus == pytest.approx(0.08 / (2.0 ** 0.5))
    assert new["bin_count"] == 1
    assert new["unique_bins"] == 2
    assert new_bonus == pytest.approx(0.08)


def test_pose_count_bonus_wraps_heading_and_resets_episode_counts():
    novelty = EpisodicPoseCountBonus({"enabled": True, "heading_bins": 8})
    first = novelty.reset([0.0, 0.0, 3.141592653589793])
    _, wrapped = novelty.observe([0.0, 0.0, -3.141592653589793])
    second = novelty.reset([1.0, 1.0, 0.0])

    assert first["bin"] == wrapped["bin"]
    assert wrapped["bin_count"] == 2
    assert second["unique_bins"] == 1
    assert second["bin_count"] == 1


def test_disabled_pose_count_bonus_is_exactly_zero():
    novelty = EpisodicPoseCountBonus()
    reset = novelty.reset([0.0, 0.0, 0.0])
    bonus, diagnostics = novelty.observe([1.0, 1.0, 1.0])

    assert reset["enabled"] is False
    assert bonus == 0.0
    assert diagnostics["unique_bins"] == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"position_bin_size": 0.0}, "position_bin_size"),
        ({"heading_bins": 0}, "heading_bins"),
        ({"count_exponent": 0.0}, "count_exponent"),
        ({"weight": -1.0}, "non-negative"),
    ],
)
def test_pose_count_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EpisodicPoseCountConfig(**kwargs).validate()
