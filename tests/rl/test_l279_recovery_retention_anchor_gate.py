import pytest

from experiments.rl.run_l279_recovery_retention_anchor_sac import _combined_gate


def _config(tmp_path):
    return {
        "l278_summary": {"path": "l278.json"},
        "gate": {"minimum_relative_test_rmse_improvement_vs_l277": 0.20},
    }


def test_l279_combined_gate_requires_retention_validation_and_control_improvement(tmp_path, monkeypatch):
    import experiments.rl.run_l279_recovery_retention_anchor_sac as module
    control = {
        "seed_metrics": [
            {"seed": seed, "relative_test_rmse_increase_6k": 1.0}
            for seed in (1, 2, 3)
        ]
    }
    path = tmp_path / "l278.json"
    path.write_text(__import__("json").dumps(control))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    retention = {
        "retention_gate_pass": True,
        "seed_metrics": [
            {"seed": seed, "relative_test_rmse_increase_6k": 0.1}
            for seed in (1, 2, 3)
        ],
    }
    checks, metrics = _combined_gate(
        _config(tmp_path), {"validation": True}, retention, True
    )
    assert all(checks.values())
    assert metrics["median_relative_test_rmse_improvement_vs_l277"] == pytest.approx(0.45)
    retention["retention_gate_pass"] = False
    checks, _ = _combined_gate(
        _config(tmp_path), {"validation": True}, retention, True
    )
    assert checks["recovery_retention_gate"] is False
