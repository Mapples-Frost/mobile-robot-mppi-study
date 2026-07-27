from copy import deepcopy

import pytest

from experiments.dynamic_uncertainty.run_complex_supervised_maneuver_actor_gate_d import (
    DEFAULT_PROTOCOL,
    build_gate_d_config,
)


def _without_arm_specific_fields(config):
    value = deepcopy(config)
    value["planner"]["paper_rl_driven"].pop(
        "supervised_maneuver_actor"
    )
    value["experiment"].pop("name")
    value.pop("gate_d_contract")
    return value


def test_gate_d_pair_differs_only_by_proposal_actor_interface():
    control = build_gate_d_config(
        DEFAULT_PROTOCOL, "chapter1", 791101019, "frozen_full"
    )
    treatment = build_gate_d_config(
        DEFAULT_PROTOCOL,
        "chapter1",
        791101019,
        "full_plus_bootstrap_actor",
    )

    assert _without_arm_specific_fields(control) == (
        _without_arm_specific_fields(treatment)
    )
    assert not control["planner"]["paper_rl_driven"][
        "supervised_maneuver_actor"
    ]["enabled"]
    assert treatment["planner"]["paper_rl_driven"][
        "supervised_maneuver_actor"
    ]["enabled"]
    for config in (control, treatment):
        assert (
            config["planner"]["num_samples"]
            * config["planner"]["paper_rl_driven"]["iterations"]
            == 600
        )


def test_gate_d_rejects_nonreserved_seed():
    with pytest.raises(ValueError, match="not a frozen Gate D paired block"):
        build_gate_d_config(
            DEFAULT_PROTOCOL,
            "chapter1",
            791101011,
            "full_plus_bootstrap_actor",
        )
