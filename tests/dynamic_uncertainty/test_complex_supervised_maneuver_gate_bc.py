from pathlib import Path

from experiments.dynamic_uncertainty.audit_complex_supervised_maneuver_actor_gate_bc import (
    _load_protocol,
    _summarize,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "configs/research/complex_supervised_maneuver_actor_gate_bc_v1.yaml"
)


def _state(full=(True, False, False), safe=(True, False, False)):
    return {
        "candidate_count": 3,
        "true_safe": list(safe),
        "true_progressing": list(safe),
        "risk_accepted": list(safe),
        "safety_nonstop": list(safe),
        "true_collision": [False, True, True],
        "full_chain": list(full),
    }


def test_gate_bc_protocol_freezes_the_online_stack():
    protocol = _load_protocol(PROTOCOL)
    assert protocol["actor"]["execution_authority"] == "proposal_only"
    assert protocol["online_audit"]["risk_icode_mppi_safety_frozen"]
    assert not protocol["frozen_contract"]["future_truth_used_by_actor"]
    assert not protocol["frozen_contract"]["formal_server_launch_authorized"]


def test_gate_bc_summary_passes_safe_covered_states():
    protocol = _load_protocol(PROTOCOL)
    metrics, checks = _summarize(protocol, [_state() for _ in range(6)])
    assert all(checks.values())
    assert metrics["state_full_chain_coverage_fraction"] == 1.0
    assert metrics["risk_accepted_true_collisions"] == 0


def test_gate_bc_summary_rejects_risk_accepted_true_collision():
    protocol = _load_protocol(PROTOCOL)
    states = [_state() for _ in range(6)]
    states[0]["risk_accepted"][1] = True
    states[0]["safety_nonstop"][1] = True
    metrics, checks = _summarize(protocol, states)
    assert metrics["risk_accepted_true_collisions"] == 1
    assert not checks["risk_accepted_true_collisions"]
    assert not checks["risk_accepted_physical_safety_precision"]
