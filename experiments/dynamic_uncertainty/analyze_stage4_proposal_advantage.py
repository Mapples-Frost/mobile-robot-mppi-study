"""Audit Stage 4 per-decision Actor-versus-Gaussian proposal advantage.

This is a retrospective development analysis.  It reads only completed Stage 4
RL/HSS-on trajectories and never opens sealed seeds.  The output is descriptive
evidence for the Stage 5 mechanism probe, not an independent confirmation set.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "research_artifacts"
    / "dynamic_uncertainty_rl_hss_stage4_amendment1_development"
)


def _first_streak(values, length):
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if bool(value) else 0
        if run >= int(length):
            return int(index - length + 1)
    return -1


def analyze(input_dir=DEFAULT_INPUT):
    input_dir = Path(input_dir).resolve()
    paths = sorted(input_dir.glob("runs/**/rl_hss_on/seed_*/trajectory.csv"))
    if not paths:
        raise FileNotFoundError(
            "no completed Stage 4 RL/HSS-on trajectories: %s" % input_dir
        )
    episode_rows = []
    pooled_delta = []
    pooled_relative = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            "paper_guided_cost_observed",
            "paper_gaussian_cost_observed",
            "paper_guided_minus_gaussian_cost_min",
            "paper_gaussian_cost_min",
            "reliability_proposal_authority",
            "reliability_guided_fraction_applied",
        }
        missing = required.difference(rows[0] if rows else {})
        if missing:
            raise ValueError(
                "trajectory lacks proposal diagnostics %s: %s"
                % (sorted(missing), path)
            )
        observed = [
            row
            for row in rows
            if float(row["paper_guided_cost_observed"]) > 0.5
            and float(row["paper_gaussian_cost_observed"]) > 0.5
        ]
        delta = np.asarray([
            float(row["paper_guided_minus_gaussian_cost_min"])
            for row in observed
        ], dtype=np.float64)
        gaussian = np.asarray([
            float(row["paper_gaussian_cost_min"])
            for row in observed
        ], dtype=np.float64)
        relative = delta / np.maximum(np.abs(gaussian), 1.0)
        disadvantage = delta > 0.0
        authority = np.asarray([
            float(row["reliability_proposal_authority"])
            for row in rows
        ], dtype=np.float64)
        guided_fraction = np.asarray([
            float(row["reliability_guided_fraction_applied"])
            for row in rows
        ], dtype=np.float64)
        episode_rows.append({
            "cell": str(path.parent.relative_to(input_dir / "runs")).replace(
                "\\", "/"
            ),
            "steps": len(rows),
            "comparable_decisions": int(delta.size),
            "disadvantage_fraction": float(np.mean(disadvantage)),
            "guided_minus_gaussian_cost_min_median": float(np.median(delta)),
            "relative_disadvantage_median": float(np.median(relative)),
            "first_disadvantage_streak_1": _first_streak(disadvantage, 1),
            "first_disadvantage_streak_3": _first_streak(disadvantage, 3),
            "first_disadvantage_streak_5": _first_streak(disadvantage, 5),
            "proposal_authority_mean": float(np.mean(authority)),
            "guided_fraction_mean": float(np.mean(guided_fraction)),
        })
        pooled_delta.extend(delta.tolist())
        pooled_relative.extend(relative.tolist())
    pooled_delta = np.asarray(pooled_delta, dtype=np.float64)
    pooled_relative = np.asarray(pooled_relative, dtype=np.float64)
    return {
        "schema_version": 1,
        "analysis": "retrospective_stage4_proposal_advantage_eda",
        "input_dir": str(input_dir.relative_to(ROOT)).replace("\\", "/"),
        "sealed_seeds_opened": False,
        "episode_count": len(episode_rows),
        "episodes": episode_rows,
        "pooled": {
            "comparable_decisions": int(pooled_delta.size),
            "disadvantage_count": int(np.sum(pooled_delta > 0.0)),
            "disadvantage_fraction": float(np.mean(pooled_delta > 0.0)),
            "guided_minus_gaussian_cost_min_quantiles": {
                str(value): float(np.quantile(pooled_delta, value))
                for value in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
            },
            "relative_disadvantage_quantiles": {
                str(value): float(np.quantile(pooled_relative, value))
                for value in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
            },
            "episodes_with_initial_three_disadvantages": int(sum(
                row["first_disadvantage_streak_3"] == 0
                for row in episode_rows
            )),
            "episodes_with_initial_five_disadvantages": int(sum(
                row["first_disadvantage_streak_5"] == 0
                for row in episode_rows
            )),
        },
        "interpretation_limit": (
            "Threshold-selection data; not an independent Stage 5 outcome."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.input_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = Path(args.output).resolve()
        if ROOT not in output.parents:
            raise ValueError("EDA output must remain inside the repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
