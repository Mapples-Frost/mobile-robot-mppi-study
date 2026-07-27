from pathlib import Path

from experiments.dynamic_uncertainty.generate_complex_supervised_maneuver_dataset import (
    _dataset_gate,
    _load_protocol,
    _source_items,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_maneuver_actor_dataset_v1.yaml"
)


def _teacher(family):
    return {"family": family}


def _job(split, scenario, anchor, families):
    map_name, seed_text = scenario.rsplit("_seed", 1)
    return {
        "split": split,
        "scenario_id": scenario,
        "map": map_name,
        "seed": int(seed_text),
        "anchor_index": int(anchor),
        "rows": [_teacher(family) for family in families],
    }


def test_protocol_freezes_privileged_and_formal_authority():
    protocol = _load_protocol(PROTOCOL)
    assert protocol["student_input"]["causal_only"]
    assert not protocol["student_input"]["include_future_truth"]
    assert not protocol["student_input"][
        "include_future_obstacle_trajectory"
    ]
    assert not protocol["student_input"]["include_privileged_change_label"]
    assert not protocol["frozen_contract"]["formal_server_launch_authorized"]
    assert protocol["post_bootstrap_rule"][
        "current_three_maps_are_final_test_maps"
    ] is False


def test_dataset_gate_requires_disjoint_complete_scenarios():
    protocol = _load_protocol(PROTOCOL)
    jobs = []
    for scenario in ("chapter1_seed1", "chapter3_seed3"):
        for anchor in range(4):
            jobs.append(_job(
                "train", scenario, anchor, ("left", "right")
            ))
    for anchor in range(3):
        jobs.append(_job(
            "validation", "chapter2_seed2", anchor, ("yield",)
        ))
    result = _dataset_gate(protocol, jobs)
    assert result["status"] == "pass"
    assert result["engineering_training_authorized"]
    assert result["checks"]["scenario_split_disjoint"]
    assert not result["formal_claim_authorized"]
    assert not result["closed_loop_gate_d_authorized"]


def test_dataset_gate_rejects_scenario_leakage_even_with_enough_rows():
    protocol = _load_protocol(PROTOCOL)
    jobs = []
    for anchor in range(8):
        jobs.append(_job(
            "train", "chapter1_seed1", anchor, ("left", "right")
        ))
    for anchor in range(4):
        jobs.append(_job(
            "train", "chapter3_seed3", anchor, ("left", "yield")
        ))
    for anchor in range(3):
        jobs.append(_job(
            "validation", "chapter1_seed1", anchor, ("yield",)
        ))
    result = _dataset_gate(protocol, jobs)
    assert result["status"] == "fail"
    assert not result["engineering_training_authorized"]
    assert not result["checks"]["scenario_split_disjoint"]


def test_dataset_sources_support_multiple_frozen_seeds_per_map():
    protocol = {
        "sources": [
            {"map": "chapter1", "artifact": "one", "split": "train"},
            {"map": "chapter1", "artifact": "two", "split": "train"},
            {
                "map": "chapter2",
                "artifact": "validation",
                "split": "validation",
            },
        ]
    }
    items = _source_items(protocol)
    assert [map_name for map_name, _ in items] == [
        "chapter1",
        "chapter1",
        "chapter2",
    ]


def test_dataset_sources_reject_duplicate_artifacts():
    protocol = {
        "sources": [
            {"map": "chapter1", "artifact": "same", "split": "train"},
            {"map": "chapter1", "artifact": "same", "split": "train"},
        ]
    }
    import pytest

    with pytest.raises(ValueError, match="duplicated"):
        _source_items(protocol)
