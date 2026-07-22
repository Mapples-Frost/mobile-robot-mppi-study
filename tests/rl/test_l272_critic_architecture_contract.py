from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _config():
    return yaml.safe_load((
        ROOT / "configs/rl/l272_critic_architecture_causal_probe.yaml"
    ).read_text(encoding="utf-8"))["l272"]


def test_l272_is_critic_only_paired_and_hash_pinned():
    config = _config()
    assert config["actor_training_authorized"] is False
    assert config["l271"]["required_gate_pass"] is True
    assert config["design"]["paired_seeds"] == [20263071, 20263072, 20263073]
    assert len(config["l268"]["shared_checkpoints"]) == 3
    assert all(len(row["sha256"]) == 64 for row in config["l268"]["shared_checkpoints"])


def test_l272_equalizes_optimizer_steps_and_sample_exposure():
    design = _config()["design"]
    batch = design["batch_size"]
    shared_steps = design["shared"]["updates"]
    per_scene_steps = (
        design["per_scene"]["critics"]
        * design["per_scene"]["updates_per_critic"]
    )
    conditioned_steps = design["scene_conditioned"]["updates"]
    assert shared_steps == per_scene_steps == conditioned_steps == 6000
    assert shared_steps * batch == per_scene_steps * batch
    assert design["scene_conditioned"]["new_input_initialization"] == "zeros"


def test_l272_uses_only_frozen_heldout_recovery_gate():
    config = _config()
    assert config["evaluation"]["states"] == 36
    assert config["evaluation"]["actions_per_state"] == 3
    assert config["evaluation"]["horizon"] == 40
    assert config["selection_order"] == [
        "scene_conditioned_equal_compute", "per_scene_equal_compute",
    ]


def test_l272_forbids_l258_final_tracking_and_actor_training():
    config = _config()
    forbidden = set(config["forbidden_tokens"])
    assert {"l258_tracking_curriculum", "l258_heldout_tracking_gate"} <= forbidden
    assert {"hairpin_v1", "s_chicane_v1", "infinity_v1"} <= forbidden
    assert config["actor_training_authorized"] is False
