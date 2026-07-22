import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.rl.generate_l267_recovery_dataset import (
    _schedule_for_scene,
    _severity_matches,
    _student_npz,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rl/l267_recovery_balanced_intervention.yaml"


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["l267"]


def test_l267_frozen_design_arithmetic_and_isolation():
    config = _config()
    assert len(config["scene_configs"]) == 6
    assert config["collection"]["accepted_chains_per_scene"] == 18
    assert sum(config["collection"]["split_counts"].values()) == 18
    assert config["replay"]["transitions_per_scene"] == 1000
    assert config["replay"]["recovery_transitions_per_scene"] == 300
    assert config["replay"]["original_transitions_per_scene"] == 700
    assert 6 * config["replay"]["transitions_per_scene"] == 6000
    assert config["critic_only"]["updates"] == 6000
    assert len(set(config["critic_only"]["seeds"])) == 3
    assert set(config["critic_only"]["seeds"]).isdisjoint(
        config["conditional_sac"]["seeds"]
    )
    references = "\n".join(
        [config["source_checkpoint"], *config["scene_configs"]]
    ).lower()
    assert not any(token.lower() in references for token in config["forbidden_tokens"])
    checkpoint = ROOT / config["source_checkpoint"]
    assert checkpoint.is_file()
    assert _sha256(checkpoint) == config["source_checkpoint_sha256"]


def test_l267_schedule_is_deterministic_balanced_and_complete():
    config = _config()
    for scene_index in range(6):
        first = _schedule_for_scene(scene_index, config)
        second = _schedule_for_scene(scene_index, config)
        assert first == second
        assert len(first) == 18
        assert [row["split"] for row in first].count("train") == 12
        assert [row["split"] for row in first].count("validation") == 3
        assert [row["split"] for row in first].count("test") == 3
        train_cells = {
            (row["side"], row["severity"], row["heading_class"])
            for row in first if row["split"] == "train"
        }
        assert len(train_cells) == 12
        assert {row["side"] for row in first} == {-1, 1}
        assert {row["severity"] for row in first} == {
            "mild", "moderate", "severe"
        }
        assert {row["heading_class"] for row in first} == {"small", "large"}
        assert {row["anchor_kind"] for row in first} == {
            "low_abs_curvature", "left", "right", "reversal"
        }


@pytest.mark.parametrize(
    ("target", "cte_m", "tolerance", "expected"),
    [
        (0.8, 0.8, 0.08, True),
        (0.8, 0.87, 0.08, True),
        (0.8, 0.89, 0.08, False),
        (2.0, 1.93, 0.08, True),
        (2.0, 2.09, 0.08, False),
    ],
)
def test_l267_realized_cte_severity_bins(target, cte_m, tolerance, expected):
    assert _severity_matches(target, cte_m, tolerance) is expected


def test_l267_student_shard_excludes_privileged_truth(tmp_path):
    rows = [
        {
            "observation": np.zeros(69, dtype=np.float32),
            "action": np.asarray((0.25, -0.5), dtype=np.float32),
            "reward": 1.25,
            "next_observation": np.ones(69, dtype=np.float32),
            "terminated": False,
            "cte": 1.0,
        }
    ]
    target = tmp_path / "chain.npz"
    _student_npz(target, rows, group=2, chain_id=7)
    with np.load(target, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "observations", "actions", "rewards", "constraint_costs",
            "next_observations", "dones", "groups", "chain_ids", "steps",
        }
        assert payload["observations"].shape == (1, 69)
        assert payload["actions"].shape == (1, 2)
        assert int(payload["groups"][0]) == 2
        assert int(payload["chain_ids"][0]) == 7


def test_l267_protocol_requires_conditional_actor_unfreeze():
    config = _config()
    gate = config["critic_gate"]
    assert config["conditional_sac"]["enabled_only_after_critic_gate"] is True
    assert gate["minimum_aggregate_in_support_spearman"] > 0.0
    assert gate["minimum_paired_median_spearman_improvement"] >= 0.20
    assert gate["minimum_improving_seed_blocks"] >= 2
    assert gate["minimum_recovery_forward_accuracy"] > 0.5
