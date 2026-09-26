"""Aggregate the audited paper-default runs and the completed fixed-H grid."""
import json
from pathlib import Path
from statistics import mean, stdev

from runtime import ART

ROOT = ART / "results"
OLD = ROOT / "paper_defaults"
NEW = ROOT / "paper_exact_grid_2026-09-23"
DEST = ART / "report" / "paper_exact_grid_2026-09-23"


def read(path):
    return json.loads(path.read_text())


def row(folder, mode="holdout_value"):
    spec = read(folder / "manifest.json")
    summary = read(folder / mode / "summary.json")
    values = [float(e["total_cost"]) for e in summary["episodes"]]
    return {
        "run": folder.name,
        "task": spec["task"],
        "seed": spec["seed"],
        "fixed_horizon": spec["fixed_horizon"],
        "terminal_value": mode == "holdout_value",
        "episodes": len(values),
        "mean_total_cost": mean(values),
        "sd_total_cost": stdev(values) if len(values) > 1 else 0.0,
        "constraint_episodes": summary["constraint_episodes"],
        "goal_episodes": summary["goal_episodes"],
        "mean_horizon": mean(e["mean_horizon"] for e in summary["episodes"]),
    }


def folders():
    old = [p for p in OLD.iterdir() if p.is_dir() and (p / "completed.json").exists()]
    new = [p for p in NEW.iterdir() if p.is_dir() and (p / "completed.json").exists()]
    return old + new


def main():
    all_folders = folders()
    rows = [row(f) for f in all_folders if (f / "holdout_value" / "summary.json").exists()]
    rows += [row(f, "holdout_no_value") for f in all_folders if (f / "holdout_no_value" / "summary.json").exists()]
    value_rows = [r for r in rows if r["terminal_value"]]
    comparisons = []
    for task in ["vehicle", "pendulum"]:
        rl = [r for r in value_rows if r["task"] == task and r["fixed_horizon"] is None]
        fixed = [r for r in value_rows if r["task"] == task and r["fixed_horizon"] is not None]
        if len(rl) != 3 or len(fixed) != 10:
            continue
        best = min(fixed, key=lambda r: r["mean_total_cost"])
        rl_mean = mean(r["mean_total_cost"] for r in rl)
        comparisons.append({
            "task": task,
            "rl_seed_mean": rl_mean,
            "rl_seed_sd": stdev(r["mean_total_cost"] for r in rl),
            "best_fixed_h": best["fixed_horizon"],
            "best_fixed_cost": best["mean_total_cost"],
            "best_fixed_sd_episodes": best["sd_total_cost"],
            "rl_minus_best_fixed": rl_mean - best["mean_total_cost"],
            "rl_relative_change_percent": 100.0 * (rl_mean - best["mean_total_cost"]) / abs(best["mean_total_cost"]),
        })
    DEST.mkdir(parents=True, exist_ok=True)
    out = {"rows": rows, "comparisons": comparisons,
           "counts": {"models": len(all_folders), "holdout_rows": len(rows),
                      "fixed_value_rows": sum(r["fixed_horizon"] is not None and r["terminal_value"] for r in rows),
                      "rl_value_rows": sum(r["fixed_horizon"] is None and r["terminal_value"] for r in rows)}}
    (DEST / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# Bøhn 2021 paper-configuration reconstruction: complete fixed-H grid", "",
             "This report combines the previously audited `paper_defaults` models with the fourteen continuation runs in `paper_exact_grid_2026-09-23`.",
             "All models use 15,000 transitions, actor 32x32, critic 256x256, gamma=rho=0.97, batch 256, replay 1e6, fixed entropy coefficient 1.0, aligned/scaled observations, and joint 32-step terminal-value learning.",
             "", "## Holdout results with terminal value", "", "|Task|Method|Seed|Mean total cost|SD across 20|Constraint episodes|Goal episodes|Mean H|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(value_rows, key=lambda x: (x["task"], x["fixed_horizon"] is None, x["fixed_horizon"] if x["fixed_horizon"] is not None else x["seed"], x["seed"])):
        method = "RL" if r["fixed_horizon"] is None else "H%d" % r["fixed_horizon"]
        lines.append("|%s|%s|%d|%.3f|%.3f|%d|%d|%.2f|" % (r["task"], method, r["seed"], r["mean_total_cost"], r["sd_total_cost"], r["constraint_episodes"], r["goal_episodes"], r["mean_horizon"]))
    lines += ["", "## RL versus the complete fixed-H grid", "", "|Task|RL mean (3 seeds)|Best fixed H|Best fixed mean|RL minus fixed|Relative change|", "|---|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append("|%s|%.3f|H%d|%.3f|%+.3f|%+.2f%%|" % (c["task"], c["rl_seed_mean"], c["best_fixed_h"], c["best_fixed_cost"], c["rl_minus_best_fixed"], c["rl_relative_change_percent"]))
    lines += ["", "A negative RL-minus-fixed value means the adaptive policy has lower holdout cost than the best fixed-H seed-0 model. This is a reconstruction comparison, not a claim of exact numeric reproduction: the authors' original experiment configuration and test bank were not recovered.", "", "The fixed-H models are retained for every H=5,10,...,50; no holdout value was used to select a checkpoint or discard a failed episode. The no-terminal-value holdout is stored beside each model for the terminal-value ablation."]
    (DEST / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DEST / "report.md")


if __name__ == "__main__":
    main()
