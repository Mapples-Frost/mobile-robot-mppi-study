import json

import numpy as np

from experiments.rl.run_l269_horizon_diagnostic import (
    ROOT,
    _discounted_prefix,
    _horizon_gate,
    _resolved_environment,
)
from experiments.rl.run_l263_counterfactual_actor_diagnosis import _prepare_reset
from mobile_robot_mppi.core.config import load_yaml


GATE = {
    "minimum_h40_recovery_advantage_fraction": 0.60,
    "minimum_positive_fraction_gain_h1_to_h40": 0.15,
    "require_median_h40_greater_than_h1": True,
    "minimum_late_first_positive_fraction": 0.20,
    "late_first_positive_minimum_horizon": 10,
    "minimum_full_chain_recovery_advantage_fraction": 0.60,
    "minimum_scenes_with_h40_majority_positive": 4,
}


def test_l269_discounted_prefix_is_exact():
    rewards = np.asarray((1.0, 2.0, 3.0))
    assert np.isclose(_discounted_prefix(rewards, 0.5, 2), 2.0)
    assert np.isclose(_discounted_prefix(rewards, 0.5, 10), 2.75)


def _passing_rows():
    rows = []
    for state in range(10):
        scene = "scene_%d" % (state % 5)
        for horizon in (1, 5, 10, 20, 40):
            advantage = -1.0 if horizon < 10 else 1.0 + 0.1 * state
            rows.append({
                "state_id": "state_%d" % state,
                "scene": scene,
                "horizon": horizon,
                "recovery_advantage": advantage,
            })
    return rows


def test_l269_horizon_gate_recognizes_late_credit_signature():
    full = [{"recovery_advantage": 1.0} for _ in range(10)]
    checks, metrics = _horizon_gate(
        _passing_rows(), full, (1, 5, 10, 20, 40), GATE
    )
    assert all(checks.values())
    assert metrics["late_first_positive_fraction"] == 1.0


def test_l269_horizon_gate_fails_when_recovery_never_wins():
    rows = _passing_rows()
    for row in rows:
        row["recovery_advantage"] = -1.0
    full = [{"recovery_advantage": -1.0} for _ in range(10)]
    checks, _ = _horizon_gate(rows, full, (1, 5, 10, 20, 40), GATE)
    assert not checks["h40_recovery_advantage_fraction"]
    assert not checks["full_chain_recovery_advantage_fraction"]


def test_l269_l263_and_l268_reset_contracts_replay_exactly():
    base = load_yaml(ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml")
    sources = (
        (
            "l263",
            ROOT / "results/research_platform/rl/l263_counterfactual_actor_diagnosis/state_manifest.json",
        ),
        (
            "l268",
            ROOT / "results/research_platform/rl/l268_recovery_balanced_intervention/heldout_recovery_diagnostic/state_manifest.json",
        ),
    )
    for contract, path in sources:
        state = json.loads(path.read_text(encoding="utf-8"))[0]
        environment = _resolved_environment(
            base, state["scene_config"], 600, state["seed"], contract
        )
        try:
            observation, _, _ = _prepare_reset(
                environment, np.asarray(state["initial_state"], dtype=np.float64),
                int(state["seed"]),
            )
        finally:
            environment.close()
        assert np.max(np.abs(
            observation - np.asarray(state["raw_observation"], dtype=np.float32)
        )) <= 1e-6
