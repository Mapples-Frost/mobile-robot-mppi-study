from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.dynamic_uncertainty.run_rl_hss_stage5_proposal_advantage import (
    ARMS,
    build_schedule,
    configure_job,
    validate_job_config,
    validate_protocol,
)
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/research/"
    "dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development.yaml"
)


def _mapping(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_stage5_protocol_and_blocked_schedule_are_frozen():
    protocol = _mapping(CONFIG)
    audit = validate_protocol(protocol)
    first = build_schedule(protocol)
    second = build_schedule(protocol)

    assert audit["protocol_valid"]
    assert not audit["sealed_seeds_opened"]
    assert first == second
    assert len(first) == 9
    assert [job["run_order"] for job in first] == list(range(9))
    for seed in protocol["design"]["obstacle_process_seeds"]:
        assert {
            job["arm"] for job in first if job["episode_seed"] == seed
        } == set(ARMS)


def test_stage5_arms_change_only_the_registered_treatment():
    protocol = _mapping(CONFIG)
    base = load_yaml(ROOT / protocol["base_config"])
    stage3 = _mapping(ROOT / protocol["stage3_protocol"])
    stage4 = _mapping(ROOT / protocol["stage4_protocol"])
    seed = int(protocol["design"]["obstacle_process_seeds"][0])
    configs = {}
    for arm in ARMS:
        job = {"arm": arm, "episode_seed": seed}
        config = configure_job(base, job, protocol, stage3, stage4)
        assert validate_job_config(config, job, protocol, base)
        configs[arm] = config

    off = configs["rl_hss_off"]
    assert off["planner"].get("optimizer", "standard") == "standard"
    assert off["planner"]["num_samples"] == 600
    assert not off.get("rl", {}).get("enabled", False)

    shadow = configs["rl_hss_shadow"]["planner"]["paper_rl_driven"]
    active = configs["rl_hss_advantage_veto"]["planner"][
        "paper_rl_driven"
    ]
    assert shadow["proposal_advantage_gate"]["mode"] == "shadow"
    assert active["proposal_advantage_gate"]["mode"] == (
        "episode_latched_veto"
    )
    for paper in (shadow, active):
        assert paper["proposal_advantage_gate"][
            "consecutive_disadvantages"
        ] == 3
        assert paper["reliability"]["enabled"]
        assert not paper["reliability"].get(
            "source_competence_enabled", False
        )


def test_stage5_protocol_fails_closed_on_sealed_or_reused_seeds():
    protocol = _mapping(CONFIG)
    sealed = deepcopy(protocol)
    sealed["design"]["sealed_seeds_authorized"] = True
    with pytest.raises(ValueError, match="sealed"):
        validate_protocol(sealed)

    reused = deepcopy(protocol)
    reused["design"]["obstacle_process_seeds"][0] = 730100006
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol(reused)


def test_stage5_protocol_fails_closed_on_gate_drift():
    protocol = _mapping(CONFIG)
    protocol["proposal_advantage_gate"]["active"][
        "consecutive_disadvantages"
    ] = 4
    with pytest.raises(ValueError, match="active gate"):
        validate_protocol(protocol)
