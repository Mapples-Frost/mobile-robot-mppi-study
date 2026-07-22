import csv
from pathlib import Path

import torch

from experiments.rl.run_l281_component_separated_anchor_sac import (
    _component_engineering,
)


def test_l281_component_engineering_requires_exact_batch_contract(tmp_path, monkeypatch):
    import experiments.rl.run_l281_component_separated_anchor_sac as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    torch.save({
        "resolved_config": {"rl": {"training": {"bc_anchor": {
            "source_kind_balanced": True,
            "recovery_action_mean_weights": [1.0, 1.0],
            "source_action_mean_weights": [0.0, 1.0],
        }}}},
    }, output / "checkpoints/step_000006000.pt")
    with (output / "updates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "actor_update_applied",
            "bc_anchor_recovery_fraction",
            "bc_anchor_velocity_weight_mean",
            "bc_anchor_angular_weight_mean",
        ])
        writer.writeheader()
        writer.writerow({
            "actor_update_applied": 1,
            "bc_anchor_recovery_fraction": 0.5,
            "bc_anchor_velocity_weight_mean": 0.5,
            "bc_anchor_angular_weight_mean": 1.0,
        })
    config = {
        "runs": [{"seed": 7, "output_dir": "run"}],
        "training": {
            "expected_anchor_update_steps": 1,
            "recovery_action_mean_weights": [1.0, 1.0],
            "source_action_mean_weights": [0.0, 1.0],
        },
    }
    complete, rows = _component_engineering(config)
    assert complete is True
    assert rows[0]["complete"] is True
