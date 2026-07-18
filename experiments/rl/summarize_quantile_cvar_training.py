#!/usr/bin/env python3
"""Audit the preregistered L81 quantile/CVaR critic training blocks."""

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
        study_label="L81",
        expected_training_seeds=(20260784, 20260785, 20260786),
        expected_validation_seed_base=22320801,
        expected_group_robust=False,
        expected_critic_distribution="quantile",
        expected_num_quantiles=25,
        expected_cvar_fraction=0.20,
        interpretation_guard=(
            "L81 changes the return representation from scalar mean-Q to 25 "
            "quantiles and trains/selects the actor with lower-tail 20% CVaR. "
            "This remains a development-training gate."
        ),
    )
    diagnostics_exact = True
    errors = []
    for index, raw in enumerate(args.run_dir):
        path = _resolved(raw) / "updates.csv"
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError("update log is empty")
            positive_spreads = 0
            for row in rows:
                values = (
                    float(row["critic_quantile_spread_mean"]),
                    float(row["q_mean"]),
                    float(row["q_cvar_mean"]),
                    float(row["policy_q_cvar_mean"]),
                )
                if (
                    float(row["critic_quantile_enabled"]) != 1.0
                    or float(row["critic_quantile_count"]) != 25.0
                    or float(row["critic_cvar_fraction"]) != 0.20
                    or not all(math.isfinite(value) for value in values)
                    or values[0] < 0.0
                ):
                    raise ValueError(
                        "quantile/CVaR update diagnostics violate contract"
                    )
                positive_spreads += int(values[0] > 0.0)
            if positive_spreads != len(rows):
                raise ValueError("quantile spread collapsed to zero")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            diagnostics_exact = False
            errors.append("block %d: %s" % (index, error))
    result["checks"]["quantile_cvar_update_diagnostics_exact"] = (
        diagnostics_exact
    )
    result["artifact_errors"].extend(errors)
    result["gate_passed"] = bool(
        not result["artifact_errors"] and all(result["checks"].values())
    )
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "l81_training_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "l81_training_audit_blocks.csv", result["blocks"])
    _write_csv(
        output / "l81_training_audit_eligibility.csv",
        result["selected_block_eligibility"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

