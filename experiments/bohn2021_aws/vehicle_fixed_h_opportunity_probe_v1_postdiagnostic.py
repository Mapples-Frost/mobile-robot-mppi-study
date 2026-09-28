#!/usr/bin/env python3
"""Metadata-only postdiagnostic for vehicle fixed-H opportunity probe V1.

This script parses the enlarged fresh fixed-H opportunity probe V1 and applies
its frozen materiality/predictability rules.  It performs no simulation, no
training/refit, no historical validation64-bank access, and no sealed-test
access.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
COMPLETED_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
PROTOCOL_PATH = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.json"
SUMMARY_IN = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/summary.md"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z.md"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V1_POSTDIAGNOSTIC_20260928T1150Z.json"
MARKER = "vehicle-fixed-h-opportunity-probe-v1-postdiagnostic-20260928T1150Z"
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not RAW_PATH.exists() or not COMPLETED_PATH.exists() or not PROTOCOL_PATH.exists():
        raise RuntimeError("V1 raw/completed/protocol input missing")
    completed = read_json(COMPLETED_PATH)
    if completed.get("passed") is not True or completed.get("hard_pass") is not True:
        raise RuntimeError("V1 completed marker did not pass")
    if completed.get("historical_validation64_bank_opened") is not False or completed.get("sealed_test_accessed") is not False:
        raise RuntimeError("V1 completed marker access flags invalid")
    # Do not re-hash the entire 160-episode tree here; the completed marker
    # already records it and the postdiagnostic is metadata-only.  Verify the
    # core top-level files that drive the postdiagnostic.
    raw = read_json(RAW_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("V1 raw access flags invalid")
    if raw.get("budget_actual", {}).get("episodes") != protocol.get("budget", {}).get("rollout_episodes_exact"):
        raise RuntimeError("V1 episode budget mismatch between raw and protocol")
    if raw.get("budget_actual", {}).get("control_steps", 0) > protocol.get("budget", {}).get("control_step_upper_bound", -1):
        raise RuntimeError("V1 control-step budget exceeded")
    if protocol.get("protocol_id") != "vehicle_fixed_h_opportunity_probe_v1_frozen_20260928":
        raise RuntimeError("unexpected V1 protocol id")
    if protocol.get("access_rules", {}).get("sealed_test_accessed") is not False:
        raise RuntimeError("unexpected protocol access flag")
    return raw, completed, protocol


def horizons_from(raw: Mapping[str, Any], protocol: Mapping[str, Any]) -> List[int]:
    hs = protocol.get("rollout_design", {}).get("horizons") or raw.get("protocol_full", {}).get("rollout_design", {}).get("horizons")
    if not hs:
        hs = sorted({int(e["horizon"]) for e in raw["episodes"]})
    return [int(h) for h in hs]


def table_by_case_h(episodes: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, Mapping[str, Any]]]:
    out: Dict[int, Dict[int, Mapping[str, Any]]] = {}
    for row in episodes:
        c = int(row["case"])
        h = int(row["horizon"])
        if h in out.get(c, {}):
            raise RuntimeError("duplicate case/horizon row: case=%s H=%s" % (c, h))
        out.setdefault(c, {})[h] = row
    return out


def is_success_no_constraint(row: Mapping[str, Any]) -> bool:
    return bool(row.get("success")) and not bool(row.get("constraint"))


def is_strict(row: Mapping[str, Any]) -> bool:
    return is_success_no_constraint(row) and int(row.get("initial_failed_steps", 0)) == 0 and int(row.get("solver_failure_steps", 0)) == 0


def solver_steps(row: Mapping[str, Any]) -> int:
    return int(row.get("initial_failed_steps", 0)) + int(row.get("solver_failure_steps", 0))


def decision_sum(row: Mapping[str, Any]) -> float:
    timing = row.get("decision_timing_s") or {}
    return safe_float(timing.get("sum"))


def solver_sum(row: Mapping[str, Any]) -> float:
    timing = row.get("solver_timing_s") or row.get("solver_attempt_timing_s") or {}
    return safe_float(timing.get("sum"))


def risk_score(row: Mapping[str, Any], primary: str = "physical_constraint_cost", time_weight: float = 0.0,
               solver_penalty: float = SOLVER_STEP_PENALTY) -> float:
    fail = 0.0
    if not bool(row.get("success")):
        fail += FAILURE_PENALTY
    if bool(row.get("constraint")):
        fail += FAILURE_PENALTY
    fail += solver_penalty * solver_steps(row)
    return safe_float(row.get(primary)) + time_weight * decision_sum(row) + fail


def aggregate(rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "success_count": int(sum(1 for r in rows if r.get("success"))),
        "constraint_count": int(sum(1 for r in rows if r.get("constraint"))),
        "initial_failed_steps": int(sum(int(r.get("initial_failed_steps", 0)) for r in rows)),
        "solver_failure_steps": int(sum(int(r.get("solver_failure_steps", 0)) for r in rows)),
        "steps": int(sum(int(r.get("steps", 0)) for r in rows)),
        "physical_constraint_cost_sum": float(math.fsum(safe_float(r.get("physical_constraint_cost")) for r in rows)),
        "total_cost_sum": float(math.fsum(safe_float(r.get("total_cost")) for r in rows)),
        "performance_cost_sum": float(math.fsum(safe_float(r.get("performance_cost")) for r in rows)),
        "h_penalty_sum": float(math.fsum(safe_float(r.get("h_penalty")) for r in rows)),
        "decision_total_s": float(math.fsum(decision_sum(r) for r in rows)),
        "solver_total_s": float(math.fsum(solver_sum(r) for r in rows)),
        "risk_physical_solver1000": float(math.fsum(risk_score(r, "physical_constraint_cost", 0.0, SOLVER_STEP_PENALTY) for r in rows)),
        "risk_total_solver1000": float(math.fsum(risk_score(r, "total_cost", 0.0, SOLVER_STEP_PENALTY) for r in rows)),
        "risk_physical_no_solver_penalty": float(math.fsum(risk_score(r, "physical_constraint_cost", 0.0, 0.0) for r in rows)),
        "horizon_counts": {str(h): int(sum(1 for r in rows if int(r["horizon"]) == h)) for h in horizons if any(int(r["horizon"]) == h for r in rows)},
    }


def choose(rows: Sequence[Mapping[str, Any]], objective: str, strict: bool = False) -> Mapping[str, Any]:
    candidates = [r for r in rows if is_strict(r)] if strict else [r for r in rows if is_success_no_constraint(r)]
    if not candidates:
        candidates = [r for r in rows if is_success_no_constraint(r)] or list(rows)
    if objective == "physical":
        return min(candidates, key=lambda r: (safe_float(r.get("physical_constraint_cost")), decision_sum(r), int(r["horizon"])))
    if objective == "total":
        return min(candidates, key=lambda r: (safe_float(r.get("total_cost")), decision_sum(r), int(r["horizon"])))
    if objective == "fastest":
        return min(candidates, key=lambda r: (decision_sum(r), safe_float(r.get("physical_constraint_cost")), int(r["horizon"])))
    if objective == "physical_risk":
        return min(candidates, key=lambda r: (risk_score(r, "physical_constraint_cost"), int(r["horizon"])))
    raise ValueError(objective)


def meta_features(meta: Mapping[str, Any]) -> List[float]:
    return [
        abs(safe_float(meta.get("theta_r"))),
        safe_float(meta.get("traj_steps")),
        safe_float(meta.get("min_reference_obstacle_clearance")),
        safe_float(meta.get("path_length")),
        safe_float(meta.get("goal_distance")),
    ]


def feature_names() -> List[str]:
    return ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "path_length", "goal_distance"]


def standardize(feature_rows: Sequence[Sequence[float]], train_ids: Sequence[int]) -> Tuple[Dict[int, List[float]], List[float], List[float]]:
    means: List[float] = []
    scales: List[float] = []
    n_features = len(feature_rows[0])
    for j in range(n_features):
        col = [feature_rows[i][j] for i in train_ids]
        m = math.fsum(col) / len(col)
        var = math.fsum((x - m) ** 2 for x in col) / max(1, len(col) - 1)
        means.append(float(m))
        scales.append(float(math.sqrt(var) if var > 1e-12 else 1.0))
    z = {i: [float((feature_rows[i][j] - means[j]) / scales[j]) for j in range(n_features)] for i in range(len(feature_rows))}
    return z, means, scales


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def row_objective(row: Mapping[str, Any], objective: str) -> float:
    if objective == "physical_cost":
        return risk_score(row, "physical_constraint_cost", 0.0, 0.0)
    if objective == "physical_risk":
        return risk_score(row, "physical_constraint_cost", 0.0, SOLVER_STEP_PENALTY)
    if objective == "total_cost":
        return risk_score(row, "total_cost", 0.0, 0.0)
    if objective == "total_risk":
        return risk_score(row, "total_cost", 0.0, SOLVER_STEP_PENALTY)
    raise ValueError(objective)


def constant_choice(train_ids: Sequence[int], table: Mapping[int, Mapping[int, Mapping[str, Any]]],
                    horizons: Sequence[int], objective: str) -> int:
    best: Optional[Tuple[float, int]] = None
    best_h: Optional[int] = None
    for h in horizons:
        score = math.fsum(row_objective(table[c][h], objective) for c in train_ids)
        key = (float(score), int(h))
        if best is None or key < best:
            best = key
            best_h = int(h)
    assert best_h is not None
    return best_h


def fit_stump(train_ids: Sequence[int], feature_rows: Mapping[int, Sequence[float]],
              table: Mapping[int, Mapping[int, Mapping[str, Any]]], horizons: Sequence[int], objective: str,
              min_leaf: int = 2) -> Dict[str, Any]:
    best_model: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int, str]] = None
    const_h = constant_choice(train_ids, table, horizons, objective)
    models: List[Dict[str, Any]] = [{"kind": "constant", "h_left": const_h, "h_right": const_h, "feature_index": None, "threshold": None}]
    for j, name in enumerate(feature_names()):
        vals = sorted(set(float(feature_rows[c][j]) for c in train_ids))
        for a, b in zip(vals[:-1], vals[1:]):
            thr = (a + b) / 2.0
            left = [c for c in train_ids if float(feature_rows[c][j]) <= thr]
            right = [c for c in train_ids if float(feature_rows[c][j]) > thr]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            h_left = constant_choice(left, table, horizons, objective)
            h_right = constant_choice(right, table, horizons, objective)
            models.append({"kind": "stump", "feature": name, "feature_index": j, "threshold": float(thr), "h_left": h_left, "h_right": h_right})
    for m in models:
        score = 0.0
        distinct = {int(m["h_left"]), int(m["h_right"])}
        for c in train_ids:
            h = predict_tree(m, feature_rows[c])
            score += row_objective(table[c][h], objective)
        key = (float(score), len(distinct), json.dumps(m, sort_keys=True))
        if best_key is None or key < best_key:
            best_key = key
            best_model = dict(m)
            best_model["train_score"] = float(score)
    assert best_model is not None
    return best_model


def predict_tree(model: Mapping[str, Any], features: Sequence[float]) -> int:
    if model.get("kind") == "constant" or model.get("feature_index") is None:
        return int(model["h_left"])
    j = int(model["feature_index"])
    return int(model["h_left"] if float(features[j]) <= float(model["threshold"]) else model["h_right"])


def fit_depth2(train_ids: Sequence[int], feature_rows: Mapping[int, Sequence[float]],
               table: Mapping[int, Mapping[int, Mapping[str, Any]]], horizons: Sequence[int], objective: str,
               min_leaf: int = 3) -> Dict[str, Any]:
    # Exhaustive two-level tree with constant-H leaves.  Kept deliberately small
    # and LOOCV-only; this is a predictability diagnostic, not selected policy.
    best_model: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int, str]] = None
    base = fit_stump(train_ids, feature_rows, table, horizons, objective, min_leaf=min_leaf)
    models: List[Dict[str, Any]] = [{"kind": "stump_as_depth2", "root": base}]
    for j, name in enumerate(feature_names()):
        vals = sorted(set(float(feature_rows[c][j]) for c in train_ids))
        for a, b in zip(vals[:-1], vals[1:]):
            thr = (a + b) / 2.0
            left = [c for c in train_ids if float(feature_rows[c][j]) <= thr]
            right = [c for c in train_ids if float(feature_rows[c][j]) > thr]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            left_model = fit_stump(left, feature_rows, table, horizons, objective, min_leaf=max(1, min_leaf // 2))
            right_model = fit_stump(right, feature_rows, table, horizons, objective, min_leaf=max(1, min_leaf // 2))
            models.append({"kind": "depth2", "feature": name, "feature_index": j, "threshold": float(thr), "left": left_model, "right": right_model})
    for m in models:
        score = 0.0
        used = set()
        for c in train_ids:
            h = predict_depth2(m, feature_rows[c])
            used.add(h)
            score += row_objective(table[c][h], objective)
        key = (float(score), len(used), json.dumps(m, sort_keys=True))
        if best_key is None or key < best_key:
            best_key = key
            best_model = dict(m)
            best_model["train_score"] = float(score)
            best_model["train_horizons_used"] = sorted(used)
    assert best_model is not None
    return best_model


def predict_depth2(model: Mapping[str, Any], features: Sequence[float]) -> int:
    if model.get("kind") == "stump_as_depth2":
        return predict_tree(model["root"], features)
    j = int(model["feature_index"])
    child = model["left"] if float(features[j]) <= float(model["threshold"]) else model["right"]
    return predict_tree(child, features)


def loocv_predictors(table: Mapping[int, Mapping[int, Mapping[str, Any]]], metas: Sequence[Mapping[str, Any]],
                     horizons: Sequence[int], objective: str) -> Dict[str, Any]:
    case_ids = sorted(table)
    raw_features = [meta_features(metas[c]) for c in case_ids]
    # Map from case id to feature vector with stable case-id indexing; case ids are 0..N-1 here.
    feat_by_case = {c: raw_features[c] for c in case_ids}
    oracle_label = {c: int(choose(list(table[c].values()), "physical" if objective.startswith("physical") else "total", strict=False)["horizon"]) for c in case_ids}
    outputs: Dict[str, List[Mapping[str, Any]]] = {"constant_refit": [], "nearest_oracle_label": [], "stump": [], "depth2": []}
    model_records: Dict[str, List[Dict[str, Any]]] = {"constant_refit": [], "nearest_oracle_label": [], "stump": [], "depth2": []}
    for test in case_ids:
        train = [c for c in case_ids if c != test]
        h_const = constant_choice(train, table, horizons, objective)
        outputs["constant_refit"].append(table[test][h_const])
        model_records["constant_refit"].append({"test_case": test, "predicted_horizon": h_const})

        z, means, scales = standardize(raw_features, train)
        nearest = min(train, key=lambda c: (euclidean(z[test], z[c]), c))
        h_nn = int(oracle_label[nearest])
        outputs["nearest_oracle_label"].append(table[test][h_nn])
        model_records["nearest_oracle_label"].append({"test_case": test, "nearest_case": nearest, "predicted_horizon": h_nn, "means": means, "scales": scales})

        stump = fit_stump(train, feat_by_case, table, horizons, objective, min_leaf=2)
        h_stump = predict_tree(stump, feat_by_case[test])
        outputs["stump"].append(table[test][h_stump])
        st_rec = dict(stump)
        st_rec.update({"test_case": test, "predicted_horizon": h_stump})
        model_records["stump"].append(st_rec)

        depth2 = fit_depth2(train, feat_by_case, table, horizons, objective, min_leaf=3)
        h_depth2 = predict_depth2(depth2, feat_by_case[test])
        outputs["depth2"].append(table[test][h_depth2])
        d_rec = dict(depth2)
        d_rec.update({"test_case": test, "predicted_horizon": h_depth2})
        model_records["depth2"].append(d_rec)
    return {
        "objective": objective,
        "feature_names": feature_names(),
        "oracle_label_counts": {str(h): int(sum(1 for c in case_ids if oracle_label[c] == h)) for h in horizons if any(oracle_label[c] == h for c in case_ids)},
        "aggregates": {name: aggregate(rows, horizons) for name, rows in outputs.items()},
        "predicted_horizon_counts": {name: {str(h): int(sum(1 for r in rows if int(r["horizon"]) == h)) for h in horizons if any(int(r["horizon"]) == h for r in rows)} for name, rows in outputs.items()},
        "model_records": model_records,
        "limits": "Development-only LOOCV on the same V1 fixed-H outcome bank; checks crude scenario-metadata signal and cannot establish final performance.",
    }


def percent_delta(ref: float, candidate: float) -> float:
    if abs(ref) <= 1e-12:
        return 0.0
    return 100.0 * (ref - candidate) / abs(ref)


def find_fixed_references(by_h: Mapping[int, Dict[str, Any]], n_cases: int) -> Dict[str, Any]:
    safe = {h: a for h, a in by_h.items() if a["success_count"] == n_cases and a["constraint_count"] == 0}
    strict = {h: a for h, a in safe.items() if a["initial_failed_steps"] == 0 and a["solver_failure_steps"] == 0}
    if not safe:
        safe = by_h
    return {
        "safe_all_success_no_constraint_horizons": sorted(safe),
        "strict_no_solver_failure_horizons": sorted(strict),
        "best_safe_physical_horizon": min(safe, key=lambda h: (safe[h]["physical_constraint_cost_sum"], safe[h]["total_cost_sum"], h)),
        "best_safe_total_horizon": min(safe, key=lambda h: (safe[h]["total_cost_sum"], safe[h]["physical_constraint_cost_sum"], h)),
        "fastest_safe_horizon": min(safe, key=lambda h: (safe[h]["decision_total_s"], safe[h]["physical_constraint_cost_sum"], h)),
        "best_strict_physical_horizon": min(strict, key=lambda h: (strict[h]["physical_constraint_cost_sum"], strict[h]["total_cost_sum"], h)) if strict else None,
        "best_strict_total_horizon": min(strict, key=lambda h: (strict[h]["total_cost_sum"], strict[h]["physical_constraint_cost_sum"], h)) if strict else None,
        "best_solver_risk_horizon": min(by_h, key=lambda h: (by_h[h]["risk_physical_solver1000"], by_h[h]["physical_constraint_cost_sum"], h)),
    }


def write_summary(result: Mapping[str, Any]) -> None:
    comp = result["comparisons"]
    refs = result["fixed_references"]
    material = result["materiality"]
    lines = [
        "# Vehicle fixed-H opportunity probe V1 postdiagnostic",
        "",
        f"UTC: `{result['created_utc']}`. Metadata-only; no simulations, no training/refit, no historical validation64-bank access, no sealed-test access.",
        "",
        "## Access and budgets checked",
        "",
        f"- V1 input episodes/control steps: `{result['input_budget']['episodes']}` episodes, `{result['input_budget']['control_steps']}` control steps.",
        f"- historical_validation64_bank_opened: `{result['historical_validation64_bank_opened']}`; sealed_test_accessed: `{result['sealed_test_accessed']}`.",
        f"- Input raw hash: `{result['input_raw']['sha256']}`; completed hash: `{result['input_completed']['sha256']}`.",
        "",
        "## Aggregate fixed-H references",
        "",
        f"- Best all-success/no-constraint physical fixed H: `H{refs['best_safe_physical_horizon']}`.",
        f"- Best all-success/no-constraint total fixed H: `H{refs['best_safe_total_horizon']}`.",
        f"- Fastest all-success/no-constraint fixed H: `H{refs['fastest_safe_horizon']}`.",
        f"- Strict no-solver-failure horizons: `{refs['strict_no_solver_failure_horizons']}`; best strict physical H: `{refs['best_strict_physical_horizon']}`.",
        "",
        "| policy/oracle | horizons used | success | constraints | init-fail | final-fail | physical+constraint | total | decision s | risk phys solver1000 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["reference_best_safe_physical", "reference_best_safe_total", "reference_fastest_safe", "reference_best_solver_risk", "oracle_best_physical_safety_only", "oracle_best_physical_strict", "oracle_best_total_safety_only", "oracle_best_total_strict", "oracle_fastest_strict"]:
        row = comp[name]
        lines.append("| %s | `%s` | %d | %d | %d | %d | %.6g | %.6g | %.6g | %.6g |" % (
            name,
            row["horizon_counts"],
            row["success_count"],
            row["constraint_count"],
            row["initial_failed_steps"],
            row["solver_failure_steps"],
            row["physical_constraint_cost_sum"],
            row["total_cost_sum"],
            row["decision_total_s"],
            row["risk_physical_solver1000"],
        ))
    lines += [
        "",
        "## Frozen materiality-rule evaluation",
        "",
        "| rule | pass | evidence |",
        "|---|---:|---|",
    ]
    for key, item in material.items():
        lines.append("| %s | `%s` | %s |" % (key, item["pass"], item["evidence"].replace("|", "/")))
    lines += [
        "",
        "## LOOCV metadata predictability diagnostics",
        "",
        "Reference for predictor gate is reported both against the H15 physical reference and the best solver-risk fixed-H reference because V1 had a single initial solver-recovery step for H10-H25. This distinction is comparison/reward-design relevant: counting the recovery as a 1000-point risk penalty makes constant H30 a strong baseline even though H15 has lower physical and total cost.",
        "",
        "| objective | selector | horizons used | physical+constraint | total | decision s | risk phys solver1000 | delta phys vs H15 | delta risk vs solver-risk ref |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    h15 = comp["reference_best_safe_physical"]
    risk_ref = comp["reference_best_solver_risk"]
    for obj, diag in result["loocv_diagnostics"].items():
        for selector, row in diag["aggregates"].items():
            lines.append("| %s | %s | `%s` | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g |" % (
                obj,
                selector,
                row["horizon_counts"],
                row["physical_constraint_cost_sum"],
                row["total_cost_sum"],
                row["decision_total_s"],
                row["risk_physical_solver1000"],
                h15["physical_constraint_cost_sum"] - row["physical_constraint_cost_sum"],
                risk_ref["risk_physical_solver1000"] - row["risk_physical_solver1000"],
            ))
    lines += [
        "",
        "## Four-axis causal diagnosis update",
        "",
        "| Axis | Verified findings | Competing hypotheses | Missing evidence | Discriminating experiment |",
        "|---|---|---|---|---|",
    ]
    for axis, row in result["four_axis_evidence"].items():
        lines.append("| %s | %s | %s | %s | %s |" % (axis, row["verified"], row["hypotheses"], row["missing"], row["experiment"]))
    lines += [
        "",
        "## Decision",
        "",
        f"- `next_action`: {result['next_action']}",
        f"- `backup_required_before_more_simulations`: `{result['backup_required_before_more_simulations']}`; request `{result['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(result: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 fixed-H opportunity V1 postdiagnostic\n\n"
        f"UTC: {result['created_utc']}. Metadata-only postdiagnostic applied frozen V1 materiality and LOOCV predictor gates. "
        f"No simulation/training/validation64/test access. Physical oracle pass={result['materiality']['physical_oracle_material']['pass']}; "
        f"total oracle pass={result['materiality']['total_oracle_material']['pass']}; predictor gate pass={result['materiality']['metadata_predictability_gate']['pass']}. "
        f"Decision: {result['next_action']}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`.\n"
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
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    raw, completed, protocol = verify_inputs()
    horizons = horizons_from(raw, protocol)
    episodes = raw["episodes"]
    table = table_by_case_h(episodes)
    case_ids = sorted(table)
    if any(sorted(table[c]) != horizons for c in case_ids):
        raise RuntimeError("V1 case/horizon grid incomplete")
    metas = list(raw["bank_selection"]["selected_metadata"])
    by_h = {h: aggregate([table[c][h] for c in case_ids], horizons) for h in horizons}
    refs = find_fixed_references(by_h, len(case_ids))
    ref_phys_h = int(refs["best_safe_physical_horizon"])
    ref_total_h = int(refs["best_safe_total_horizon"])
    ref_fast_h = int(refs["fastest_safe_horizon"])
    ref_risk_h = int(refs["best_solver_risk_horizon"])

    oracle_phys_safe_rows = [choose(list(table[c].values()), "physical", strict=False) for c in case_ids]
    oracle_phys_strict_rows = [choose(list(table[c].values()), "physical", strict=True) for c in case_ids]
    oracle_total_safe_rows = [choose(list(table[c].values()), "total", strict=False) for c in case_ids]
    oracle_total_strict_rows = [choose(list(table[c].values()), "total", strict=True) for c in case_ids]
    oracle_fast_strict_rows = [choose(list(table[c].values()), "fastest", strict=True) for c in case_ids]

    comp: Dict[str, Any] = {
        "by_horizon": {str(h): by_h[h] for h in horizons},
        "reference_best_safe_physical": by_h[ref_phys_h],
        "reference_best_safe_total": by_h[ref_total_h],
        "reference_fastest_safe": by_h[ref_fast_h],
        "reference_best_solver_risk": by_h[ref_risk_h],
        "oracle_best_physical_safety_only": aggregate(oracle_phys_safe_rows, horizons),
        "oracle_best_physical_strict": aggregate(oracle_phys_strict_rows, horizons),
        "oracle_best_total_safety_only": aggregate(oracle_total_safe_rows, horizons),
        "oracle_best_total_strict": aggregate(oracle_total_strict_rows, horizons),
        "oracle_fastest_strict": aggregate(oracle_fast_strict_rows, horizons),
    }
    for oracle_name in ["oracle_best_physical_safety_only", "oracle_best_physical_strict", "oracle_best_total_safety_only", "oracle_best_total_strict", "oracle_fastest_strict"]:
        ref = comp["reference_best_safe_physical"]
        cand = comp[oracle_name]
        comp[oracle_name + "_vs_reference_best_safe_physical"] = {
            "physical_delta_positive_means_candidate_lower": ref["physical_constraint_cost_sum"] - cand["physical_constraint_cost_sum"],
            "physical_delta_percent": percent_delta(ref["physical_constraint_cost_sum"], cand["physical_constraint_cost_sum"]),
            "total_delta_positive_means_candidate_lower": ref["total_cost_sum"] - cand["total_cost_sum"],
            "total_delta_percent": percent_delta(ref["total_cost_sum"], cand["total_cost_sum"]),
            "decision_time_delta_positive_means_candidate_lower_s": ref["decision_total_s"] - cand["decision_total_s"],
            "initial_failed_step_delta_positive_means_candidate_lower": ref["initial_failed_steps"] - cand["initial_failed_steps"],
            "solver_failure_step_delta_positive_means_candidate_lower": ref["solver_failure_steps"] - cand["solver_failure_steps"],
        }

    loocv = {
        "physical_cost": loocv_predictors(table, metas, horizons, "physical_cost"),
        "physical_risk": loocv_predictors(table, metas, horizons, "physical_risk"),
        "total_cost": loocv_predictors(table, metas, horizons, "total_cost"),
        "total_risk": loocv_predictors(table, metas, horizons, "total_risk"),
    }
    # Predictor gate: at least one deliberately simple metadata selector must
    # beat H15 on physical cost AND must not lose to the best solver-risk fixed-H
    # reference when risk penalties are used.  We also record weaker partial wins.
    best_selector = None
    best_selector_key = None
    partial_wins: List[Dict[str, Any]] = []
    h15_ref = comp["reference_best_safe_physical"]
    risk_ref = comp["reference_best_solver_risk"]
    for obj, diag in loocv.items():
        for selector, agg in diag["aggregates"].items():
            phys_delta = h15_ref["physical_constraint_cost_sum"] - agg["physical_constraint_cost_sum"]
            total_delta = h15_ref["total_cost_sum"] - agg["total_cost_sum"]
            risk_delta = risk_ref["risk_physical_solver1000"] - agg["risk_physical_solver1000"]
            if phys_delta > 0 or total_delta > 0 or risk_delta > 0:
                partial_wins.append({"objective": obj, "selector": selector, "physical_delta_vs_H15": phys_delta, "total_delta_vs_H15": total_delta, "risk_delta_vs_solver_ref": risk_delta, "horizon_counts": agg["horizon_counts"]})
            key = (agg["risk_physical_solver1000"], agg["physical_constraint_cost_sum"], agg["total_cost_sum"], obj, selector)
            if best_selector_key is None or key < best_selector_key:
                best_selector_key = key
                best_selector = {"objective": obj, "selector": selector, "aggregate": agg}
    assert best_selector is not None
    best_vs_h15_phys = h15_ref["physical_constraint_cost_sum"] - best_selector["aggregate"]["physical_constraint_cost_sum"]
    best_vs_risk_ref = risk_ref["risk_physical_solver1000"] - best_selector["aggregate"]["risk_physical_solver1000"]

    phys_delta = comp["oracle_best_physical_safety_only_vs_reference_best_safe_physical"]["physical_delta_positive_means_candidate_lower"]
    phys_delta_pct = comp["oracle_best_physical_safety_only_vs_reference_best_safe_physical"]["physical_delta_percent"]
    phys_total_delta = comp["oracle_best_physical_safety_only_vs_reference_best_safe_physical"]["total_delta_positive_means_candidate_lower"]
    total_delta = comp["oracle_best_total_safety_only_vs_reference_best_safe_physical"]["total_delta_positive_means_candidate_lower"]
    total_delta_pct = comp["oracle_best_total_safety_only_vs_reference_best_safe_physical"]["total_delta_percent"]
    total_solver_regression = comp["oracle_best_total_safety_only"]["initial_failed_steps"] > h15_ref["initial_failed_steps"] or comp["oracle_best_total_safety_only"]["solver_failure_steps"] > h15_ref["solver_failure_steps"] or comp["oracle_best_total_safety_only"]["constraint_count"] > h15_ref["constraint_count"]

    physical_oracle_pass = bool(phys_delta >= max(5.0, 0.05 * h15_ref["physical_constraint_cost_sum"]) and phys_total_delta >= -0.05 * h15_ref["total_cost_sum"])
    total_oracle_pass = bool(total_delta >= max(3.0, 0.03 * h15_ref["total_cost_sum"]) and not total_solver_regression)
    predictor_gate_pass = bool(best_vs_h15_phys > 0.0 and best_vs_risk_ref >= 0.0)
    # A weaker gate is useful for deciding between selector smoke and continuation
    # diagnostics: if metadata beats H15 but loses to H30 risk only because of the
    # single initial solver-recovery event, we need continuation/value evidence
    # rather than immediate broad training.
    predictor_partial_h15_cost_win = any(x["physical_delta_vs_H15"] > 0.0 and x["total_delta_vs_H15"] >= -0.05 * h15_ref["total_cost_sum"] for x in partial_wins)

    materiality = {
        "physical_oracle_material": {
            "pass": physical_oracle_pass,
            "evidence": "oracle safety-only best physical improves %.6g physical units (%.2f%%) vs H%d and changes total by %.6g; threshold is max(5,5%% of %.6g)=%.6g and total-loss limit is %.6g" % (phys_delta, phys_delta_pct, ref_phys_h, phys_total_delta, h15_ref["physical_constraint_cost_sum"], max(5.0, 0.05 * h15_ref["physical_constraint_cost_sum"]), -0.05 * h15_ref["total_cost_sum"]),
        },
        "total_oracle_material": {
            "pass": total_oracle_pass,
            "evidence": "oracle safety-only best total improves %.6g total units (%.2f%%) vs H%d; solver/constraint regression=%s; threshold is max(3,3%% of %.6g)=%.6g" % (total_delta, total_delta_pct, ref_total_h, total_solver_regression, h15_ref["total_cost_sum"], max(3.0, 0.03 * h15_ref["total_cost_sum"])),
        },
        "metadata_predictability_gate": {
            "pass": predictor_gate_pass,
            "evidence": "best LOOCV metadata selector by risk is %s/%s with physical delta vs H%d %.6g and risk delta vs best solver-risk H%d %.6g; partial H15-cost win=%s" % (best_selector["objective"], best_selector["selector"], ref_phys_h, best_vs_h15_phys, ref_risk_h, best_vs_risk_ref, predictor_partial_h15_cost_win),
        },
    }

    if physical_oracle_pass and total_oracle_pass and predictor_gate_pass:
        next_action = "after verified backup, freeze a small IMPROVED metadata/state selector smoke on fresh development-only cases, with H15 and solver-risk H%d fixed baselines and measured actual timing; do not use validation64/test" % ref_risk_h
    elif physical_oracle_pass or total_oracle_pass:
        next_action = "after verified backup, run a bounded controlled continuation/value-and-transition diagnostic on representative V1 states because oracle opportunity is material but metadata predictability is not yet robust enough for broad retraining/refit"
    else:
        next_action = "after verified backup, document weak canonical-distribution opportunity and freeze a versioned scenario-redesign/continuation protocol rather than another unchanged adaptive validation campaign"

    four_axis = {
        "SCENARIOS": {
            "verified": "V1 enlarged the source-supported canonical bank to 16 cases/160 fixed-H episodes. H15 remains aggregate best physical and total, but per-case best physical horizons span %s and oracle materiality=%s." % (comp["oracle_best_physical_safety_only"]["horizon_counts"], physical_oracle_pass),
            "hypotheses": "Canonical straight-line cases contain some horizon opportunity, but simple case metadata may not identify it; within-episode state features or redesigned stress cases may be required.",
            "missing": "Continuation comparisons from identical intermediate states and independent confirmation cases; no sealed test accessed.",
            "experiment": "Use representative V1 states/cases to compare H choices from identical states with matched terminal sources and timing, or freeze a versioned stress scenario if continuation still shows weak opportunity.",
        },
        "REWARD": {
            "verified": "Postdiagnostic separates physical+constraint cost, total cost including h_penalty, solver penalties used only for diagnostic risk, and measured decision time. Synthetic horizon penalty is not treated as runtime.",
            "hypotheses": "A single initial solver-recovery event can dominate risk if penalized at 1000/step; total cost may favor H15 while actual timing favors H10 on some cases.",
            "missing": "Sensitivity to solver-penalty scaling and explicit success/safety accounting under any learned selector objective.",
            "experiment": "Before selector/refit, freeze objective weights that report both physical/total and risk/timing, with H15 and H%d risk baselines." % ref_risk_h,
        },
        "TRAINING": {
            "verified": "V1 is fixed-H only with zero training/gradient steps. Current gated policies are finite re-selections; V1 does not prove neural under-training but can provide labels for an IMPROVED selector if predictability is adequate.",
            "hypotheses": "Training/refit may help only if state-level features predict material opportunity; otherwise broad retraining will chase noise or H15 dominance.",
            "missing": "Out-of-sample selector performance from richer state/continuation features and terminal-value mismatch diagnostics.",
            "experiment": "If continuation labels are predictive, freeze one small selector/value-refit smoke before any 3-seed campaign; otherwise do not retrain yet.",
        },
        "COMPARISONS": {
            "verified": "Strong same-bank fixed-H grid over H5..H50 with independent terminal sources is available; H15 is strongest aggregate cost and H%d is best solver-risk reference under diagnostic penalties." % ref_risk_h,
            "hypotheses": "Adaptive gains may vanish against tuned H15/H30 references and actual timing noise; weak fixed-H comparators would overstate adaptive value.",
            "missing": "Fresh confirmation after any method/scenario change, paired seeds, blocked runtime timing, and sealed final test only after gate.",
            "experiment": "Any selector smoke must compare against H15, H10 fastest-safe, and H%d solver-risk fixed references with actual whole-decision timing." % ref_risk_h,
        },
    }

    result: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_metadata_only",
        "classification": "metadata_only_development_postdiagnostic_not_validation_not_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_raw": {"path": rel(RAW_PATH), "sha256": sha256(RAW_PATH)},
        "input_completed": {"path": rel(COMPLETED_PATH), "sha256": sha256(COMPLETED_PATH)},
        "input_protocol": {"path": rel(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "input_summary": {"path": rel(SUMMARY_IN), "sha256": sha256(SUMMARY_IN) if SUMMARY_IN.exists() else None},
        "input_budget": raw.get("budget_actual", {}),
        "fixed_references": refs,
        "comparisons": comp,
        "loocv_diagnostics": loocv,
        "best_loocv_selector_by_risk": best_selector,
        "loocv_partial_wins": partial_wins,
        "materiality": materiality,
        "four_axis_evidence": four_axis,
        "next_action": next_action,
        "backup_required_before_more_simulations": True,
        "backup_request": rel(BACKUP_PATH),
        "limitations": [
            "V1 is development-only and not final validation/test evidence",
            "Episode-level fixed-H oracle is not an upper bound on within-episode adaptive switching",
            "LOOCV metadata diagnostics are small-sample and use measured V1 outcomes for development only",
            "Actual timing on t3a.medium is noisy and nonmonotonic, so timing claims require blocked confirmation",
        ],
    }
    write_json(OUT_DIR / "raw.json", result)
    write_summary(result)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        "# Vehicle fixed-H opportunity V1 postdiagnostic state\n\n"
        f"UTC: {created}. Metadata-only; no simulation/training/validation64/test access. "
        f"Physical oracle material={physical_oracle_pass}; total oracle material={total_oracle_pass}; predictor gate={predictor_gate_pass}. "
        f"Next action: {next_action}. Backup required before any further simulation.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup fixed-H opportunity V1 postdiagnostic before continuation/selector/scenario simulations",
        "backup_required_before_more_simulations": True,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_PATH), rel(Path(__file__).resolve())],
    })
    append_docs(result)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, BACKUP_PATH, Path(__file__).resolve(), RAW_PATH, COMPLETED_PATH, PROTOCOL_PATH]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_required_before_more_simulations": True,
        "backup_request": rel(BACKUP_PATH),
        "headline": {
            "physical_oracle_material": physical_oracle_pass,
            "total_oracle_material": total_oracle_pass,
            "metadata_predictability_gate": predictor_gate_pass,
            "next_action": next_action,
            "best_safe_physical_horizon": ref_phys_h,
            "best_solver_risk_horizon": ref_risk_h,
            "best_loocv_selector": {"objective": best_selector["objective"], "selector": best_selector["selector"], "horizon_counts": best_selector["aggregate"]["horizon_counts"]},
        },
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "physical_oracle_material": physical_oracle_pass,
        "total_oracle_material": total_oracle_pass,
        "metadata_predictability_gate": predictor_gate_pass,
        "next_action": next_action,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "backup_request": rel(BACKUP_PATH),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
