from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_l270_contract_is_diagnostic_and_fail_closed():
    config = yaml.safe_load((
        ROOT / "configs/rl/l270_observation_aliasing_diagnosis.yaml"
    ).read_text(encoding="utf-8"))
    section = config["l270"]
    assert section["conditional_training"] is False
    assert section["l269"]["required_status"] == "horizon_gate_fail"
    assert section["recovery_probe"]["samples_per_chain"] == 12
    assert section["recovery_probe"]["minimum_history_step"] >= 2
    assert section["recovery_probe"]["feature_arms"] == [
        "current_69d", "far_preview", "three_frame_history",
        "progress_segment", "all_privileged",
    ]
    assert section["selection_order"] == [
        "far_preview", "three_frame_history", "progress_segment",
        "all_privileged",
    ]


def test_l270_forbids_l258_and_final_tracking_scenes():
    config = yaml.safe_load((
        ROOT / "configs/rl/l270_observation_aliasing_diagnosis.yaml"
    ).read_text(encoding="utf-8"))
    forbidden = set(config["l270"]["forbidden_tokens"])
    assert {"l258_tracking_curriculum", "l258_heldout_tracking_gate"} <= forbidden
    assert {"hairpin_v1", "s_chicane_v1", "infinity_v1"} <= forbidden
