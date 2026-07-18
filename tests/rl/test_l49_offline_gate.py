import json

from experiments.rl.summarize_l49_offline_gate import summarize


def _metrics(scale):
    return {
        "rollout_rmse_h36": 0.8 * scale,
        "nominal_rollout_rmse_h36": scale,
        "endpoint_h36": {"position_rmse": 0.8 * scale, "heading_rmse": 0.9 * scale},
        "nominal_endpoint_h36": {"position_rmse": scale, "heading_rmse": scale},
        "rollout_windows_h36": 10,
    }


def test_l49_gate_requires_all_total_and_two_joint_blocks(tmp_path):
    model_dirs = []
    for block in range(3):
        model_dir = tmp_path / ("model_%d" % block)
        model_dir.mkdir()
        for split in ("test", "unseen"):
            (model_dir / ("%s_metrics.json" % split)).write_text(
                json.dumps(_metrics(1.0 + block)), encoding="utf-8"
            )
        model_dirs.append(model_dir)
    assert summarize(model_dirs)["gate"]["passed"]

    broken = _metrics(1.0)
    broken["rollout_rmse_h36"] = 1.1
    (model_dirs[0] / "unseen_metrics.json").write_text(json.dumps(broken), encoding="utf-8")
    result = summarize(model_dirs)
    assert not result["gate"]["passed"]
    assert result["gate"]["unseen_total_improvement_blocks"] == 2
