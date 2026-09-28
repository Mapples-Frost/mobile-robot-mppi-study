#!/usr/bin/env python3
"""Metadata-only postdiagnostic for vehicle fixed-H opportunity probe V0.

Parses the fresh fixed-H opportunity probe outputs and quantifies whether the
observed fixed-H variation is (a) a true oracle upper bound, (b) plausibly
predictable from currently source-supported scenario metadata, or (c) mostly
noise/terminal-quality variation that should not trigger another blind adaptive
campaign.  No simulation, no training, no historical validation64 bank access,
and no sealed-test access.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/raw.json"
COMPLETED_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/completed.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-fixed-h-opportunity-probe-v0-postdiagnostic-20260928T1030Z"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
FAILURE_PENALTY = 1_000_000.0
SOLVER_STEP_PENALTY = 1_000.0


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def values(xs: Iterable[float]) -> Dict[str, Any]:
    arr = list(float(x) for x in xs)
    if not arr:
        return {"count": 0, "sum": 0.0, "mean": None, "min": None, "max": None}
    arr_sorted = sorted(arr)
    n = len(arr)
    return {
        "count": n,
        "sum": float(math.fsum(arr)),
        "mean": float(math.fsum(arr) / n),
        "min": float(arr_sorted[0]),
        "max": float(arr_sorted[-1]),
    }


def eligible(row: Mapping[str, Any], strict_solver: bool = True) -> bool:
    if not bool(row.get("success")) or bool(row.get("constraint")):
        return False
    if strict_solver and (int(row.get("initial_failed_steps", 0)) > 0 or int(row.get("solver_failure_steps", 0)) > 0):
        return False
    return True


def risk_score(row: Mapping[str, Any], primary: str = "physical_constraint_cost", time_weight: float = 0.0) -> float:
    fail = 0.0
    if not bool(row.get("success")):
        fail += FAILURE_PENALTY
    if bool(row.get("constraint")):
        fail += FAILURE_PENALTY
    fail += SOLVER_STEP_PENALTY * (int(row.get("initial_failed_steps", 0)) + int(row.get("solver_failure_steps", 0)))
    return float(row.get(primary, 0.0)) + time_weight * float(row["decision_timing_s"]["sum"]) + fail


def by_case_h(episodes: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, Mapping[str, Any]]]:
    out: Dict[int, Dict[int, Mapping[str, Any]]] = {}
    for row in episodes:
        out.setdefault(int(row["case"]), {})[int(row["horizon"])] = row
    return out


def choose(rows: Sequence[Mapping[str, Any]], objective: str) -> Mapping[str, Any]:
    strict = [r for r in rows if eligible(r, strict_solver=True)]
    cand = strict or [r for r in rows if eligible(r, strict_solver=False)] or list(rows)
    if objective == "physical":
        return min(cand, key=lambda r: (float(r["physical_constraint_cost"]), float(r["decision_timing_s"]["sum"]), int(r["horizon"])))
    if objective == "total":
        return min(cand, key=lambda r: (float(r["total_cost"]), float(r["decision_timing_s"]["sum"]), int(r["horizon"])))
    if objective == "fastest":
        return min(cand, key=lambda r: (float(r["decision_timing_s"]["sum"]), float(r["physical_constraint_cost"]), int(r["horizon"])))
    raise ValueError(objective)


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "success_count": sum(1 for r in rows if r.get("success")),
        "constraint_count": sum(1 for r in rows if r.get("constraint")),
        "initial_failed_steps": sum(int(r.get("initial_failed_steps", 0)) for r in rows),
        "solver_failure_steps": sum(int(r.get("solver_failure_steps", 0)) for r in rows),
        "steps": sum(int(r.get("steps", 0)) for r in rows),
        "physical_constraint_cost_sum": float(math.fsum(float(r.get("physical_constraint_cost", 0.0)) for r in rows)),
        "total_cost_sum": float(math.fsum(float(r.get("total_cost", 0.0)) for r in rows)),
        "decision_total_s": float(math.fsum(float(r["decision_timing_s"]["sum"]) for r in rows)),
        "horizon_counts": {str(h): sum(1 for r in rows if int(r["horizon"]) == h) for h in HORIZONS if any(int(r["horizon"]) == h for r in rows)},
        "risk_physical_score": float(math.fsum(risk_score(r, "physical_constraint_cost", 0.0) for r in rows)),
    }


def meta_vec(meta: Mapping[str, Any]) -> List[float]:
    return [float(abs(meta.get("theta_r", 0.0))), float(meta.get("traj_steps", 0.0)), float(meta.get("min_reference_obstacle_clearance", 0.0))]


def standardize(features: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    cols = list(zip(*features))
    means = [sum(c) / len(c) for c in cols]
    scales = []
    for c, m in zip(cols, means):
        var = sum((x - m) ** 2 for x in c) / max(1, len(c) - 1)
        scales.append(math.sqrt(var) if var > 1e-12 else 1.0)
    z = [[(x - m) / s for x, m, s in zip(row, means, scales)] for row in features]
    return z, means, scales


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def constant_choice(train_case_ids: Sequence[int], table: Mapping[int, Mapping[int, Mapping[str, Any]]], objective: str) -> int:
    best_h = None
    best_score = None
    for h in HORIZONS:
        rows = [table[c][h] for c in train_case_ids]
        if objective == "physical_risk":
            score = math.fsum(risk_score(r, "physical_constraint_cost", 0.0) for r in rows)
        elif objective == "total_risk":
            score = math.fsum(risk_score(r, "total_cost", 0.0) for r in rows)
        else:
            raise ValueError(objective)
        tie = (score, h)
        if best_score is None or tie < best_score:
            best_score = tie
            best_h = h
    assert best_h is not None
    return int(best_h)


def threshold_rule_choice(train_ids: Sequence[int], test_id: int, features: Mapping[int, Sequence[float]],
                          table: Mapping[int, Mapping[int, Mapping[str, Any]]], objective: str) -> Dict[str, Any]:
    # Fit a one-split, two-leaf selector on seven cases.  This is diagnostic only;
    # it is included to see whether the currently available metadata has any
    # out-of-sample signal beyond a constant H.
    candidates: List[Dict[str, Any]] = []
    const_h = constant_choice(train_ids, table, objective)
    candidates.append({"kind": "constant", "h_left": const_h, "h_right": const_h, "feature": None, "threshold": None})
    for feat_idx, feat_name in enumerate(["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"]):
        vals = sorted(set(float(features[c][feat_idx]) for c in train_ids))
        thresholds = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
        for thr in thresholds:
            left = [c for c in train_ids if float(features[c][feat_idx]) <= thr]
            right = [c for c in train_ids if float(features[c][feat_idx]) > thr]
            if not left or not right:
                continue
            h_left = constant_choice(left, table, objective)
            h_right = constant_choice(right, table, objective)
            candidates.append({"kind": "threshold", "feature": feat_name, "feature_index": feat_idx, "threshold": float(thr), "h_left": h_left, "h_right": h_right})
    def train_score(model: Mapping[str, Any]) -> Tuple[float, int, str]:
        score = 0.0
        for c in train_ids:
            if model["kind"] == "constant" or float(features[c][int(model.get("feature_index", 0))]) <= float(model.get("threshold", 0.0)):
                h = int(model["h_left"])
            else:
                h = int(model["h_right"])
            r = table[c][h]
            score += risk_score(r, "physical_constraint_cost" if objective == "physical_risk" else "total_cost", 0.0)
        distinct = 1 if model["h_left"] == model["h_right"] else 2
        return (float(score), distinct, json.dumps(model, sort_keys=True))
    best = min(candidates, key=train_score)
    if best["kind"] == "constant" or float(features[test_id][int(best.get("feature_index", 0))]) <= float(best.get("threshold", 0.0)):
        pred_h = int(best["h_left"])
    else:
        pred_h = int(best["h_right"])
    out = dict(best)
    out["predicted_horizon"] = pred_h
    out["train_score"] = train_score(best)[0]
    return out


def loocv_selectors(table: Mapping[int, Mapping[int, Mapping[str, Any]]], metas: Mapping[int, Mapping[str, Any]],
                    objective: str) -> Dict[str, Any]:
    case_ids = sorted(table)
    raw_features = {c: meta_vec(metas[c]) for c in case_ids}
    z_list, means, scales = standardize([raw_features[c] for c in case_ids])
    z = {c: z_list[i] for i, c in enumerate(case_ids)}
    oracle_best = {c: int(choose(list(table[c].values()), "physical" if objective == "physical_risk" else "total")["horizon"]) for c in case_ids}
    nn_rows = []
    tree_rows = []
    const_rows = []
    for test in case_ids:
        train = [c for c in case_ids if c != test]
        nearest = min(train, key=lambda c: (euclidean(z[test], z[c]), c))
        nn_h = oracle_best[nearest]
        nn_rows.append(table[test][nn_h])
        tree = threshold_rule_choice(train, test, raw_features, table, objective)
        tree_rows.append(table[test][int(tree["predicted_horizon"])])
        const_h = constant_choice(train, table, objective)
        const_rows.append(table[test][const_h])
    return {
        "objective": objective,
        "feature_order": ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"],
        "feature_means": means,
        "feature_scales": scales,
        "nearest_neighbor_oracle_label_loocv": aggregate_rows(nn_rows),
        "one_split_threshold_loocv": aggregate_rows(tree_rows),
        "constant_H_refit_loocv": aggregate_rows(const_rows),
        "nearest_neighbor_horizons": [int(r["horizon"]) for r in nn_rows],
        "threshold_horizons": [int(r["horizon"]) for r in tree_rows],
        "constant_refit_horizons": [int(r["horizon"]) for r in const_rows],
        "limits": "Eight-case development-only LOOCV over outcomes already measured in the fixed-H probe; detects gross metadata signal only and is not validation evidence.",
    }


def write_summary(result: Mapping[str, Any]) -> None:
    cmp = result["comparisons"]
    lines = [
        "# Vehicle fixed-H opportunity probe V0 postdiagnostic",
        "",
        f"UTC: `{result['created_utc']}`. Metadata-only; no new rollouts/training/validation64/test access.",
        "",
        "## What the fixed-H probe means",
        "",
        "- The fresh fixed-H probe found real development-only horizon variation, but it is an upper-bound opportunity map, not an adaptive-policy result.",
        f"- Global fixed H15 remains the strongest aggregate physical/total-cost arm in this 8-case bank; H10 is the fastest all-success arm but has one initial solver-failure step and much larger physical cost.",
        f"- Oracle best-physical per case uses horizons `{cmp['oracle_best_physical']['horizon_counts']}` and saves `{cmp['oracle_best_physical_vs_H15']['physical_delta_positive_means_oracle_lower']:.6g}` physical+constraint cost versus H15 on this same bank, with decision-time delta `{cmp['oracle_best_physical_vs_H15']['decision_time_delta_positive_means_oracle_lower_s']:.6g}` s.",
        f"- Oracle best-total per case uses horizons `{cmp['oracle_best_total']['horizon_counts']}` and changes total cost versus H15 by `{cmp['oracle_best_total_vs_H15']['total_delta_positive_means_oracle_lower']:.6g}`.",
        "",
        "## Predictability from currently available scenario metadata",
        "",
        "| selector diagnostic | objective | horizons used | physical+constraint | total | decision s | risk score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name in ["H15_global", "H10_fast_safe", "oracle_best_physical", "oracle_best_total"]:
        row = cmp[name]
        lines.append(f"| {name} | same-bank | `{row['horizon_counts']}` | {row['physical_constraint_cost_sum']:.6g} | {row['total_cost_sum']:.6g} | {row['decision_total_s']:.6g} | {row['risk_physical_score']:.6g} |")
    for obj, diag in result["loocv_diagnostics"].items():
        for name in ["constant_H_refit_loocv", "nearest_neighbor_oracle_label_loocv", "one_split_threshold_loocv"]:
            row = diag[name]
            lines.append(f"| {name} | {obj} | `{row['horizon_counts']}` | {row['physical_constraint_cost_sum']:.6g} | {row['total_cost_sum']:.6g} | {row['decision_total_s']:.6g} | {row['risk_physical_score']:.6g} |")
    lines += [
        "",
        "## Interpretation and next action",
        "",
    ]
    for item in result["diagnostic_conclusions"]:
        lines.append(f"- {item}")
    lines += [
        "",
        f"Backup request before any further simulation: `{result['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backup_request(created: str) -> str:
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V0_POSTDIAGNOSTIC_20260928T1030Z.json")
    write_json(path, {
        "requested_utc": created,
        "reason": "backup fixed-H opportunity V0 postdiagnostic before any further selector/training simulation",
        "backup_required_before_more_simulations": True,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve())],
    })
    return rel(path)


def append_docs(result: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 fixed-H opportunity V0 postdiagnostic\n\n"
        f"UTC: {result['created_utc']}. Metadata-only postdiagnostic parsed the fresh fixed-H opportunity probe. "
        f"No validation64/test access. Oracle best-physical same-bank physical delta vs H15="
        f"{result['comparisons']['oracle_best_physical_vs_H15']['physical_delta_positive_means_oracle_lower']:.6g}; "
        f"LOOCV one-split physical-risk selector risk delta vs H15="
        f"{result['comparisons']['loocv_one_split_physical_risk_vs_H15']['risk_delta_positive_means_selector_lower']:.6g}. "
        f"Conclusion: {result['next_action']}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`.\n"
    )
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_json(RAW_PATH)
    completed = read_json(COMPLETED_PATH)
    if completed.get("passed") is not True or raw.get("sealed_test_accessed") or raw.get("historical_validation64_bank_opened"):
        raise RuntimeError("Input access/completion flags invalid")
    episodes = raw["episodes"]
    table = by_case_h(episodes)
    metas = {int(i): m for i, m in enumerate(raw["bank_selection"]["selected_metadata"])}
    h15_rows = [table[c][15] for c in sorted(table)]
    h10_rows = [table[c][10] for c in sorted(table)]
    oracle_phys_rows = [choose(list(table[c].values()), "physical") for c in sorted(table)]
    oracle_total_rows = [choose(list(table[c].values()), "total") for c in sorted(table)]
    oracle_fast_rows = [choose(list(table[c].values()), "fastest") for c in sorted(table)]
    comp: Dict[str, Any] = {
        "H15_global": aggregate_rows(h15_rows),
        "H10_fast_safe": aggregate_rows(h10_rows),
        "oracle_best_physical": aggregate_rows(oracle_phys_rows),
        "oracle_best_total": aggregate_rows(oracle_total_rows),
        "oracle_fastest": aggregate_rows(oracle_fast_rows),
    }
    comp["oracle_best_physical_vs_H15"] = {
        "physical_delta_positive_means_oracle_lower": comp["H15_global"]["physical_constraint_cost_sum"] - comp["oracle_best_physical"]["physical_constraint_cost_sum"],
        "total_delta_positive_means_oracle_lower": comp["H15_global"]["total_cost_sum"] - comp["oracle_best_physical"]["total_cost_sum"],
        "decision_time_delta_positive_means_oracle_lower_s": comp["H15_global"]["decision_total_s"] - comp["oracle_best_physical"]["decision_total_s"],
    }
    comp["oracle_best_total_vs_H15"] = {
        "physical_delta_positive_means_oracle_lower": comp["H15_global"]["physical_constraint_cost_sum"] - comp["oracle_best_total"]["physical_constraint_cost_sum"],
        "total_delta_positive_means_oracle_lower": comp["H15_global"]["total_cost_sum"] - comp["oracle_best_total"]["total_cost_sum"],
        "decision_time_delta_positive_means_oracle_lower_s": comp["H15_global"]["decision_total_s"] - comp["oracle_best_total"]["decision_total_s"],
    }
    loocv = {
        "physical_risk": loocv_selectors(table, metas, "physical_risk"),
        "total_risk": loocv_selectors(table, metas, "total_risk"),
    }
    tree_phys = loocv["physical_risk"]["one_split_threshold_loocv"]
    comp["loocv_one_split_physical_risk_vs_H15"] = {
        "physical_delta_positive_means_selector_lower": comp["H15_global"]["physical_constraint_cost_sum"] - tree_phys["physical_constraint_cost_sum"],
        "total_delta_positive_means_selector_lower": comp["H15_global"]["total_cost_sum"] - tree_phys["total_cost_sum"],
        "decision_time_delta_positive_means_selector_lower_s": comp["H15_global"]["decision_total_s"] - tree_phys["decision_total_s"],
        "risk_delta_positive_means_selector_lower": comp["H15_global"]["risk_physical_score"] - tree_phys["risk_physical_score"],
    }
    # Materiality flags: same-bank oracle must be nontrivial, and crude LOOCV
    # metadata should not be worse than H15 before spending a long adaptive campaign.
    oracle_phys_delta = comp["oracle_best_physical_vs_H15"]["physical_delta_positive_means_oracle_lower"]
    tree_risk_delta = comp["loocv_one_split_physical_risk_vs_H15"]["risk_delta_positive_means_selector_lower"]
    h15_phys = comp["H15_global"]["physical_constraint_cost_sum"]
    oracle_material = bool(oracle_phys_delta > max(5.0, 0.05 * h15_phys))
    metadata_signal_nonnegative = bool(tree_risk_delta >= 0.0)
    conclusions = [
        "Scenario layer: the fixed-H grid contains development-only state/scenario-dependent opportunity; per-case oracle best-physical horizons are not constant, while H5 is unsafe on most selected cases.",
        "Training/selection layer: current reused gated policies were not trained/reselected on this per-stratum oracle surface, so a new selector/search objective is justified only as IMPROVED development work, not ORIGINAL reproduction.",
        "Implementation/timing layer: actual decision timing is non-monotone across H on this AWS run, so any adaptive method must use measured whole-decision timing and should include timing-noise checks rather than assuming shorter H is faster.",
    ]
    if oracle_material and metadata_signal_nonnegative:
        next_action = "after backup, freeze a small IMPROVED metadata-aware selector smoke on fresh development cases, with H15/H10/H25/H40 fixed baselines and no validation64/test access"
    elif oracle_material:
        next_action = "after backup, run a bounded feature/transition diagnostic before adaptive training because oracle opportunity exists but simple metadata predictability is weak"
    else:
        next_action = "document weak material opportunity and avoid another adaptive campaign until scenario design is revised under a new protocol"
    result: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "metadata_only_postdiagnostic_development_not_validation_not_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_raw": {"path": rel(RAW_PATH), "sha256": sha256(RAW_PATH)},
        "input_completed": {"path": rel(COMPLETED_PATH), "sha256": sha256(COMPLETED_PATH)},
        "comparisons": comp,
        "loocv_diagnostics": loocv,
        "diagnostic_flags": {
            "oracle_material_development_only": oracle_material,
            "metadata_signal_nonnegative_loocv_physical_risk": metadata_signal_nonnegative,
            "validation64_or_test_accessed": False,
        },
        "diagnostic_conclusions": conclusions,
        "next_action": next_action,
    }
    result["backup_request"] = backup_request(result["created_utc"])
    write_json(OUT_DIR / "raw.json", result)
    write_summary(result)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle fixed-H opportunity V0 postdiagnostic state\n\nUTC: {result['created_utc']}. No new rollouts/training/validation/test access. "
        f"Oracle material={oracle_material}; LOOCV metadata nonnegative={metadata_signal_nonnegative}. Next action: {next_action}. "
        "Backup required before any further simulation.\n",
        encoding="utf-8",
    )
    append_docs(result)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, ROOT / result["backup_request"], Path(__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "next_action": next_action,
        "headline": {
            "oracle_material_development_only": oracle_material,
            "metadata_signal_nonnegative_loocv_physical_risk": metadata_signal_nonnegative,
            "oracle_physical_delta_vs_H15": oracle_phys_delta,
            "loocv_tree_risk_delta_vs_H15": tree_risk_delta,
        },
        "backup_request": result["backup_request"],
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "oracle_material_development_only": oracle_material,
        "metadata_signal_nonnegative_loocv_physical_risk": metadata_signal_nonnegative,
        "oracle_physical_delta_vs_H15": oracle_phys_delta,
        "loocv_tree_risk_delta_vs_H15": tree_risk_delta,
        "next_action": next_action,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": result["backup_request"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
