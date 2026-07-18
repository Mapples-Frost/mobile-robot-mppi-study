#!/usr/bin/env python3
"""Audit the preregistered L80 group-robust actor training blocks."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.summarize_conservative_schedule_training import (  # noqa: E402
    _resolved,
    summarize,
)
from experiments.rl.summarize_static_multigeometry_training import (  # noqa: E402
    _write_csv,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = summarize(
        args.config,
        args.run_dir,
        study_label="L80",
        expected_training_seeds=(20260777, 20260778, 20260779),
        expected_validation_seed_base=22310801,
        expected_group_robust=True,
        interpretation_guard=(
            "L80 changes one factor from L78: the residual actor aggregates "
            "scene-wise losses with a preregistered smooth worst-group "
            "objective. This remains a development-training gate."
        ),
    )
    group_diagnostics_exact = True
    group_errors = []
    for index, raw in enumerate(args.run_dir):
        path = _resolved(raw) / "updates.csv"
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError("update log is empty")
            for row in rows:
                values = (
                    float(row["actor_group_robust_loss"]),
                    float(row["actor_group_loss_min"]),
                    float(row["actor_group_loss_max"]),
                )
                if (
                    float(row["actor_group_robust_enabled"]) != 1.0
                    or float(row["actor_group_count"]) != 3.0
                    or not all(math.isfinite(value) for value in values)
                    or values[2] < values[1]
                ):
                    raise ValueError(
                        "group-robust update diagnostics violate contract"
                    )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            group_diagnostics_exact = False
            group_errors.append("block %d: %s" % (index, error))
    result["checks"]["group_robust_update_diagnostics_exact"] = (
        group_diagnostics_exact
    )
    result["artifact_errors"].extend(group_errors)
    result["gate_passed"] = bool(
        not result["artifact_errors"] and all(result["checks"].values())
    )
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "l80_training_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "l80_training_audit_blocks.csv", result["blocks"])
    _write_csv(
        output / "l80_training_audit_eligibility.csv",
        result["selected_block_eligibility"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
