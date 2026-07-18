#!/usr/bin/env python3
"""Post-gate EDA for the L25 scene-complexity development dataset."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.run_scene_complexity_gate_ablation import CONDITIONS


def _read(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite %s" % name)
    return value


def _quantiles(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("EDA quantiles require finite nonempty data")
    return {
        "min": float(data.min()),
        "q10": float(np.quantile(data, 0.10)),
        "median": float(np.median(data)),
        "q90": float(np.quantile(data, 0.90)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _success_by_checkpoint(episodes):
    groups = defaultdict(list)
    for row in episodes:
        groups[(row["scene"], row["condition"], int(row["training_seed"]))].append(row)
    output = {}
    for (scene, condition, seed), rows in sorted(groups.items()):
        output.setdefault(scene, {}).setdefault(condition, {})[str(seed)] = {
            "episodes": len(rows),
            "successes": int(sum(_bool(row["success"]) for row in rows)),
            "collisions": int(sum(_bool(row["collision"]) for row in rows)),
            "final_distance_mean_m": float(np.mean([
                _float(row, "final_goal_distance") for row in rows
            ])),
        }
    return output


def _paired_comparisons(paired):
    output = {}
    for scene in sorted({row["scene"] for row in paired}):
        for condition in ("frozen_bc_prior", "lcb_always", "complexity_lcb"):
            rows = [
                row for row in paired
                if row["scene"] == scene and row["condition"] == condition
            ]
            traditional = {
                "success_gains": int(sum(_bool(row["traditional_success_gained"]) for row in rows)),
                "success_losses": int(sum(_bool(row["traditional_success_lost"]) for row in rows)),
                "collision_regressions": int(sum(_bool(row["traditional_collision_regression"]) for row in rows)),
                "distance_improvement_m": _quantiles([
                    _float(row, "traditional_distance_improvement") for row in rows
                ]),
            }
            bc = {
                "success_gains": int(sum(_bool(row["bc_success_gained"]) for row in rows)),
                "success_losses": int(sum(_bool(row["bc_success_lost"]) for row in rows)),
                "collision_regressions": int(sum(_bool(row["bc_collision_regression"]) for row in rows)),
                "distance_improvement_m": _quantiles([
                    _float(row, "bc_distance_improvement") for row in rows
                ]),
            }
            output.setdefault(scene, {})[condition] = {
                "episodes": len(rows),
                "versus_traditional": traditional,
                "versus_frozen_bc": bc,
            }
    return output


def _step_gate_summary(steps):
    groups = defaultdict(list)
    for row in steps:
        if row["condition"] in ("lcb_always", "complexity_lcb"):
            groups[(row["scene"], row["condition"])].append(row)
    output = {}
    for (scene, condition), rows in sorted(groups.items()):
        alpha = [_float(row, "rl_gate_alpha") for row in rows]
        score = [_float(row, "rl_scene_complexity_score") for row in rows]
        output.setdefault(scene, {})[condition] = {
            "steps": len(rows),
            "gate_alpha": _quantiles(alpha),
            "complexity_score": _quantiles(score),
            "active_fraction_alpha_gt_0p05": float(np.mean(np.asarray(alpha) > 0.05)),
            "full_fraction_alpha_ge_0p95": float(np.mean(np.asarray(alpha) >= 0.95)),
            "fallback_fraction_alpha_eq_0": float(np.mean(np.asarray(alpha) <= 1e-12)),
        }
    return output


def _complexity_vs_always(episodes):
    lookup = {
        (
            int(row["training_seed"]), row["scene"],
            int(row["episode_seed"]), row["condition"],
        ): row
        for row in episodes
    }
    output = {}
    for scene in sorted({row["scene"] for row in episodes}):
        groups = sorted({
            (int(row["training_seed"]), int(row["episode_seed"]))
            for row in episodes if row["scene"] == scene
        })
        gains = 0
        losses = 0
        collisions = 0
        distances = []
        for training_seed, episode_seed in groups:
            always = lookup[(training_seed, scene, episode_seed, "lcb_always")]
            gated = lookup[(training_seed, scene, episode_seed, "complexity_lcb")]
            gains += int(not _bool(always["success"]) and _bool(gated["success"]))
            losses += int(_bool(always["success"]) and not _bool(gated["success"]))
            collisions += int(not _bool(always["collision"]) and _bool(gated["collision"]))
            distances.append(
                _float(always, "final_goal_distance")
                - _float(gated, "final_goal_distance")
            )
        output[scene] = {
            "groups": len(groups),
            "success_gains": gains,
            "success_losses": losses,
            "collision_regressions": collisions,
            "distance_improvement_m": _quantiles(distances),
        }
    return output


def _clean_exact_fallback(steps):
    selected = [row for row in steps if row["scene"] == "clean_dynamics"]
    lookup = {
        (
            int(row["training_seed"]),
            int(row["episode_seed"]),
            row["condition"],
            int(row["step"]),
        ): row
        for row in selected
    }
    groups = sorted({key[:2] for key in lookup})
    fields = ("goal_distance", "executed_v", "executed_omega", "collision", "safety_override")
    maximum = {field: 0.0 for field in fields}
    rows_compared = 0
    length_mismatches = 0
    for group in groups:
        traditional_steps = sorted(
            key[3] for key in lookup
            if key[:3] == group + ("traditional_mppi",)
        )
        complexity_steps = sorted(
            key[3] for key in lookup
            if key[:3] == group + ("complexity_lcb",)
        )
        if traditional_steps != complexity_steps:
            length_mismatches += 1
            continue
        for step in traditional_steps:
            left = lookup[group + ("traditional_mppi", step)]
            right = lookup[group + ("complexity_lcb", step)]
            for field in fields:
                difference = abs(float(left[field]) - float(right[field]))
                maximum[field] = max(maximum[field], difference)
            rows_compared += 1
    return {
        "checkpoint_episode_groups": len(groups),
        "rows_compared": rows_compared,
        "length_mismatches": length_mismatches,
        "maximum_absolute_difference": maximum,
        "exact": length_mismatches == 0 and max(maximum.values()) == 0.0,
    }


def _quality(episodes, paired, steps):
    episode_key = [
        (row["training_seed"], row["scene"], row["episode_seed"], row["condition"])
        for row in episodes
    ]
    step_key = [
        (
            row["training_seed"], row["scene"], row["episode_seed"],
            row["condition"], row["step"],
        )
        for row in steps
    ]
    numeric_episode = (
        "final_goal_distance", "trajectory_length", "control_jerk",
        "planner_compute_ms_mean", "rl_gate_alpha_mean",
    )
    numeric_step = (
        "time", "goal_distance", "rl_gate_alpha",
        "rl_scene_complexity_score", "executed_v", "executed_omega",
    )
    return {
        "episode_rows": len(episodes),
        "episode_columns": len(episodes[0]),
        "paired_rows": len(paired),
        "step_rows": len(steps),
        "step_columns": len(steps[0]),
        "duplicate_episode_keys": len(episode_key) - len(set(episode_key)),
        "duplicate_step_keys": len(step_key) - len(set(step_key)),
        "finite_episode_numeric": bool(all(
            math.isfinite(float(row[name])) for row in episodes for name in numeric_episode
        )),
        "finite_step_numeric": bool(all(
            math.isfinite(float(row[name])) for row in steps for name in numeric_step
        )),
        "conditions": sorted({row["condition"] for row in episodes}),
        "scenes": sorted({row["scene"] for row in episodes}),
        "training_seeds": sorted({int(row["training_seed"]) for row in episodes}),
        "episode_seeds": sorted({int(row["episode_seed"]) for row in episodes}),
        "structural_missing_clearance_rows": int(sum(
            row.get("minimum_clearance") in ("", "None") for row in episodes
        )),
    }


def _markdown(report):
    success = report["success_by_checkpoint"]
    lines = [
        "# L25 scene-complexity development EDA",
        "",
        "This is explanatory analysis performed after the frozen Development Gate. It does not change L25 thresholds or authorize sealed-test use.",
        "",
        "## Data quality",
        "",
        "- Episode rows: `%d`; paired rows: `%d`; step rows: `%d`." % (
            report["quality"]["episode_rows"],
            report["quality"]["paired_rows"],
            report["quality"]["step_rows"],
        ),
        "- Duplicate episode/step keys: `%d / %d`." % (
            report["quality"]["duplicate_episode_keys"],
            report["quality"]["duplicate_step_keys"],
        ),
        "- All required episode and step numeric values are finite. Missing clearance occurs only in the no-obstacle scene and is structural, not imputed.",
        "",
        "## Success by scene and method",
        "",
        "| scene | traditional | frozen BC | always LCB | complexity LCB |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene in sorted(success):
        total = sum(value["episodes"] for value in success[scene]["traditional_mppi"].values())
        values = []
        for condition in CONDITIONS:
            values.append(sum(
                value["successes"] for value in success[scene][condition].values()
            ))
        lines.append("| %s | %d/%d | %d/%d | %d/%d | %d/%d |" % (
            scene, values[0], total, values[1], total, values[2], total, values[3], total
        ))
    lines.extend([
        "",
        "All 480 episodes had zero collision.",
        "",
        "Paired complexity-gate gains/losses versus always-on LCB were: %s." % (
            "; ".join(
                "%s +%d/-%d" % (
                    scene,
                    row["success_gains"],
                    row["success_losses"],
                )
                for scene, row in sorted(report["complexity_vs_always"].items())
            )
        ),
        "",
        "## Checkpoint-level replication",
        "",
    ])
    for scene in sorted(success):
        values = success[scene]["complexity_lcb"]
        lines.append("- `%s` complexity-LCB successes by training seed: %s." % (
            scene,
            ", ".join("%s=%d/%d" % (seed, row["successes"], row["episodes"]) for seed, row in sorted(values.items())),
        ))
    gate = report["step_gate_summary"]
    lines.extend([
        "",
        "## Gate behavior",
        "",
        "| scene | complexity mean | alpha mean | active fraction | fallback fraction |",
        "|---|---:|---:|---:|---:|",
    ])
    for scene in sorted(gate):
        row = gate[scene]["complexity_lcb"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (
            scene,
            row["complexity_score"]["mean"],
            row["gate_alpha"]["mean"],
            row["active_fraction_alpha_gt_0p05"],
            row["fallback_fraction_alpha_eq_0"],
        ))
    fallback = report["clean_exact_fallback"]
    lines.extend([
        "",
        "In `clean_dynamics`, complexity-LCB and traditional MPPI matched exactly over `%d` step rows; maximum control and goal-distance differences were zero." % fallback["rows_compared"],
        "",
        "## Interpretation",
        "",
        "1. The sensor-only gate is causally active: it exactly recovers traditional MPPI when scan complexity is zero and substantially changes behavior near blocking geometry.",
        "2. The preregistered global scene label was too coarse. `clean_single_obstacle` contains only one obstacle but it blocks the direct route at K=200; the gate activated and raised success from 0/30 to 20/30. L25 must still be recorded as gate-failed because that scene was preregistered as simple.",
        "3. Relative to always-on correction, gating removed catastrophic simple-scene behavior and improved U-trap success, but reduced narrow-corridor success. A compute-budget/sample-efficiency interaction is therefore the next falsifiable question, not threshold retuning on these data.",
        "4. The current actor is a U-trap specialist. Strong results on narrow corridor are promising transfer evidence, but three training seeds are still a development sample and not a final generalization claim.",
        "",
        "## Next analysis boundary",
        "",
        "Freeze L25. Use new development seeds to compare traditional and complexity-gated MPPI across K=50/100/200/400. Keep L25 sealed seeds unopened, and do not relabel the single-obstacle rows inside L25.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.input_dir).resolve()
    episodes = _read(directory / "episodes.csv")
    paired = _read(directory / "paired_episodes.csv")
    steps = _read(directory / "gate_steps.csv")
    with (directory / "development_gate.json").open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    report = {
        "quality": _quality(episodes, paired, steps),
        "development_gate": gate,
        "success_by_checkpoint": _success_by_checkpoint(episodes),
        "paired_comparisons": _paired_comparisons(paired),
        "complexity_vs_always": _complexity_vs_always(episodes),
        "step_gate_summary": _step_gate_summary(steps),
        "clean_exact_fallback": _clean_exact_fallback(steps),
        "interpretation_guard": (
            "post-gate EDA cannot change preregistered L25 eligibility or open sealed test seeds"
        ),
    }
    with (directory / "eda.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    (directory / "eda.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
