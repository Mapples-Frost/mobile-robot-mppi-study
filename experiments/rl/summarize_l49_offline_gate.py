#!/usr/bin/env python3
"""Audit the frozen L49 offline gate before starting L50 closed-loop trials."""

import argparse
import json
import math
from pathlib import Path


SPLITS = ("test", "unseen")
HORIZON = 36


def _load_metrics(model_dir, split):
    path = Path(model_dir) / ("%s_metrics.json" % split)
    return json.loads(path.read_text(encoding="utf-8"))


def _improvement(candidate, nominal):
    return (float(nominal) - float(candidate)) / max(float(nominal), 1e-12)


def summarize(model_dirs):
    rows = []
    for model_block, model_dir in enumerate(model_dirs):
        for split in SPLITS:
            metrics = _load_metrics(model_dir, split)
            endpoint = metrics["endpoint_h%d" % HORIZON]
            nominal_endpoint = metrics["nominal_endpoint_h%d" % HORIZON]
            row = {
                "model_block": model_block,
                "model_dir": str(Path(model_dir).resolve()),
                "split": split,
                "rollout_rmse": float(metrics["rollout_rmse_h%d" % HORIZON]),
                "nominal_rollout_rmse": float(metrics["nominal_rollout_rmse_h%d" % HORIZON]),
                "position_rmse": float(endpoint["position_rmse"]),
                "nominal_position_rmse": float(nominal_endpoint["position_rmse"]),
                "heading_rmse": float(endpoint["heading_rmse"]),
                "nominal_heading_rmse": float(nominal_endpoint["heading_rmse"]),
                "rollout_windows": int(metrics["rollout_windows_h%d" % HORIZON]),
            }
            row.update({
                "rollout_relative_improvement": _improvement(row["rollout_rmse"], row["nominal_rollout_rmse"]),
                "position_relative_improvement": _improvement(row["position_rmse"], row["nominal_position_rmse"]),
                "heading_relative_improvement": _improvement(row["heading_rmse"], row["nominal_heading_rmse"]),
            })
            rows.append(row)

    finite = all(
        math.isfinite(value)
        for row in rows
        for key, value in row.items()
        if key not in ("model_dir", "split")
    )
    total_pass = {
        split: sum(
            row["rollout_rmse"] < row["nominal_rollout_rmse"]
            for row in rows if row["split"] == split
        )
        for split in SPLITS
    }
    position_heading_pass = {
        split: sum(
            row["position_rmse"] < row["nominal_position_rmse"]
            and row["heading_rmse"] < row["nominal_heading_rmse"]
            for row in rows if row["split"] == split
        )
        for split in SPLITS
    }
    gate = {
        "finite_metrics": finite,
        "model_blocks": len(model_dirs),
        "test_total_improvement_blocks": total_pass["test"],
        "unseen_total_improvement_blocks": total_pass["unseen"],
        "test_position_and_heading_improvement_blocks": position_heading_pass["test"],
        "unseen_position_and_heading_improvement_blocks": position_heading_pass["unseen"],
    }
    gate["passed"] = bool(
        finite
        and len(model_dirs) == 3
        and total_pass["test"] == 3
        and total_pass["unseen"] == 3
        and position_heading_pass["test"] >= 2
        and position_heading_pass["unseen"] >= 2
    )
    return {
        "design_id": "l49_iterative_path_offline_gate_v1",
        "horizon": HORIZON,
        "rows": rows,
        "gate": gate,
        "interpretation_guard": (
            "Passing this offline gate only permits L50 closed-loop evaluation; "
            "it is not evidence of improved control performance."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = summarize(args.model_dirs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
