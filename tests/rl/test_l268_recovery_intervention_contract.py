import hashlib
from pathlib import Path

import yaml

from experiments.rl.generate_l267_recovery_dataset import _schedule_for_scene


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l268_recovery_balanced_intervention.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l268"]


def test_l268_source_and_budget_are_frozen():
    config = _config()
    checkpoint = ROOT / config["source_checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        config["source_checkpoint_sha256"]
    )
    assert config["collection"]["maximum_target_cte_m"] == 3.5
    assert config["collection"]["maximum_steps"] == 600
    assert config["collection"]["maximum_attempts_per_stratum"] == 16
    assert config["collection"]["boundary_inset_beyond_radius_m"] == 0.35
    assert config["collection"]["accepted_chains_per_scene"] == 18
    assert config["replay"]["total_transitions_per_arm"] == 6000
    assert config["critic_only"]["updates"] == 6000


def test_l268_schedule_and_seeds_are_isolated():
    config = _config()
    for scene_index in range(6):
        schedule = _schedule_for_scene(scene_index, config)
        assert len(schedule) == 18
        assert len({
            (row["side"], row["severity"], row["heading_class"])
            for row in schedule if row["split"] == "train"
        }) == 12
        smoke = _schedule_for_scene(scene_index, config, smoke=True)
        assert len(smoke) == 3
        assert {row["severity"] for row in smoke} == {
            "mild", "moderate", "severe"
        }
    assert set(config["critic_only"]["seeds"]).isdisjoint(
        config["conditional_sac"]["seeds"]
    )
    references = "\n".join(
        [config["source_checkpoint"], *config["scene_configs"]]
    ).lower()
    assert not any(token.lower() in references for token in config["forbidden_tokens"])


def test_l268_actor_stage_is_fail_closed():
    config = _config()
    assert config["conditional_sac"]["enabled_only_after_critic_gate"] is True
    assert config["critic_gate"]["minimum_aggregate_in_support_spearman"] > 0
    assert config["critic_gate"]["minimum_improving_seed_blocks"] == 2
