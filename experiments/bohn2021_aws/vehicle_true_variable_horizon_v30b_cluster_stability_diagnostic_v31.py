#!/usr/bin/env python3
"""v31 cluster-stability diagnostic for v30b feature separability.

Development analysis only. This script reads already-opened v29/v30b/postdiagnostic
JSON artifacts and performs no MPC simulation, no training/refit, no validation64
or sealed-test access. It quantifies whether the v30b/postdiagnostic feature rule
is stable when correlated opened rows are grouped by source family rather than
counted as independent cases.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31"
STAMP = "20260930T0405Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_OUT = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v31_cluster_stability_diagnostic.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_{STAMP}.json"
INPUT_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T035819_from_user_context_after_v30b_postdiagnostic_and_gate_source.json"
V29_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json"
V30B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json"
POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/raw.json"
POST_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/completed.json"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
DOC_MARKER = f"<!-- {NAME}-{STAMP} -->"
FEATURES = ["abs_obs_00", "abs_obs_07"]
HORIZONS = [12, 15, 35]
EXPECTED_HASHES = {
    V29_RAW: "62817a9373dcc02d79b39bb08c1bfc01bf1e7b242221edc743f06e83c6651a73",
    V30B_RAW: "f8ad34b335beb6f6d4b4f44cfc85a74691f51ccfd63ac7d57e7efa5a8f379639",
    POST_RAW: "e183aeb0e1e3972fb28f6df5306af41bfedd7f95173a0676c17501a0ceaa20a7",
}


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(BASE.resolve()).as_posix()
        except Exception:
            return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def git(args: List[str]) -> Dict[str, Any]:
    try:
        p = subprocess.run(["git", *args], cwd=str(ROOT), text=True, capture_output=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:4000], "stderr": p.stderr.strip()[:1000]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def group_key(row: Mapping[str, Any]) -> str:
    cat = str(row.get("category_for_audit_only") or row.get("category") or "unknown")
    label = str(row.get("state_label") or row.get("base_state_id") or "unknown")
    base = str(row.get("base_state_id") or "")
    if cat == "v19_h12_only_failure_h15_safe" or base.startswith("v15c"):
        return "v19_case05_H15_risk_family"
    if cat == "v27_both_fail_h12_h15" or label.startswith("v27_case09"):
        return "v27_case09_H35_rescue_family"
    if label.startswith("v27_case"):
        # v27_caseXX_slotY_* -> group both slots from the same source case.
        parts = label.split("_slot", 1)
        return parts[0] + "_H12_control_family"
    return cat + "::" + label.split("_slot", 1)[0]


def horizon_bad(row: Mapping[str, Any], h: int) -> bool:
    key = f"H{h}_bad"
    if key in row:
        return bool(row[key])
    per = row.get("per_horizon") or {}
    rec = per.get(str(h)) or per.get(h) or {}
    if isinstance(rec, Mapping) and "safe_success_no_solver_fail" in rec:
        return not bool(rec["safe_success_no_solver_fail"])
    return True


def horizon_metric(row: Mapping[str, Any], h: int, metric: str, fallback: float = math.nan) -> float:
    per = row.get("per_horizon") or {}
    rec = per.get(str(h)) or per.get(h) or {}
    if isinstance(rec, Mapping):
        value = rec.get(metric, fallback)
        try:
            return float(value)
        except Exception:
            return fallback
    return fallback


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(rel(path))
        got = sha256(path)
        if got != expected:
            raise RuntimeError(f"Input hash mismatch for {rel(path)}: got {got}, expected {expected}")
    v29 = read_json(V29_RAW)
    v30b = read_json(V30B_RAW)
    post = read_json(POST_RAW)
    v29_rows = ((v29.get("analysis") or {}).get("state_rows") or [])
    v29_by_label: Dict[str, Mapping[str, Any]] = {}
    for r in v29_rows:
        if not isinstance(r, Mapping):
            continue
        for k in ("state_label", "base_state_id"):
            if r.get(k):
                v29_by_label[str(r[k])] = r
    rows: List[Dict[str, Any]] = []
    for r in post.get("state_table", []):
        if not isinstance(r, Mapping):
            continue
        rr = dict(r)
        src = v29_by_label.get(str(rr.get("state_label"))) or v29_by_label.get(str(rr.get("base_state_id")))
        if src:
            rr["per_horizon"] = src.get("per_horizon") or {}
            rr["source_campaign"] = src.get("source_campaign")
            rr["source_candidate_index"] = src.get("source_candidate_index")
            rr["role"] = src.get("role")
        rr["family_group"] = group_key(rr)
        rows.append(rr)
    if len(rows) != 11:
        raise RuntimeError(f"Expected 11 postdiagnostic rows, found {len(rows)}")
    meta = {"v29_headline": (v29.get("analysis") or {}).get("headline"), "v30b_headline": (v30b.get("headline") or (v30b.get("analysis") or {}).get("headline")), "post_headline": post.get("headline")}
    return rows, meta


def condition_candidates(train: Sequence[Mapping[str, Any]]) -> List[Optional[Dict[str, Any]]]:
    out: List[Optional[Dict[str, Any]]] = [None]
    for feat in FEATURES:
        vals = sorted({float(r[feat]) for r in train if feat in r and r[feat] is not None})
        if not vals:
            continue
        thresholds: List[float] = [vals[0] - 1e-9, vals[-1] + 1e-9]
        thresholds += [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:]) if a != b]
        # Use sorted unique to keep deterministic order.
        for t in sorted(set(thresholds)):
            for op in ("<=", ">"):
                out.append({"feature": feat, "op": op, "threshold": float(t), "repr": f"{feat} {op} {t:.12g}"})
    return out


def match(cond: Optional[Mapping[str, Any]], row: Mapping[str, Any]) -> bool:
    if cond is None:
        return False
    val = float(row[cond["feature"]])
    thr = float(cond["threshold"])
    return val <= thr if cond["op"] == "<=" else val > thr


def predict_rule(rule: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    order = rule.get("order")
    h35 = rule.get("h35_condition")
    h15 = rule.get("h15_condition")
    if order == "H15_first":
        if match(h15, row):
            return 15
        if match(h35, row):
            return 35
    else:
        if match(h35, row):
            return 35
        if match(h15, row):
            return 15
    return 12


def evaluate_policy(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, int]) -> Dict[str, Any]:
    bad_rows = []
    h_counts = {"12": 0, "15": 0, "35": 0}
    decision_sum = 0.0
    solver_sum = 0.0
    physical_sum = 0.0
    for i, row in enumerate(rows):
        key = str(row.get("base_state_id") or row.get("state_label") or i)
        h = int(predictions[key])
        h_counts[str(h)] += 1
        bad = horizon_bad(row, h)
        decision = horizon_metric(row, h, "decision_sum_s", 0.0)
        solver = horizon_metric(row, h, "solver_sum_s", 0.0)
        physical = horizon_metric(row, h, "physical_constraint_cost", 0.0)
        decision_sum += decision
        solver_sum += solver
        physical_sum += physical
        if bad:
            best_safe_phys = min([horizon_metric(row, hh, "physical_constraint_cost", math.inf) for hh in HORIZONS if not horizon_bad(row, hh)] or [math.inf])
            bad_rows.append({
                "state_label": row.get("state_label"),
                "base_state_id": row.get("base_state_id"),
                "family_group": row.get("family_group"),
                "predicted_horizon": h,
                "oracle_h": row.get("oracle_h"),
                "decision_sum_s": decision,
                "physical_constraint_cost": physical,
                "best_safe_physical": best_safe_phys,
                "bad_flags": {str(hh): horizon_bad(row, hh) for hh in HORIZONS},
            })
    fixed_h35 = sum(horizon_metric(r, 35, "decision_sum_s", 0.0) for r in rows)
    saving = None if fixed_h35 <= 0 else 1.0 - decision_sum / fixed_h35
    return {
        "rows": len(rows),
        "bad_count": len(bad_rows),
        "bad_rows": bad_rows,
        "h_counts": h_counts,
        "decision_sum_s": decision_sum,
        "solver_sum_s": solver_sum,
        "physical_sum": physical_sum,
        "decision_saving_vs_fixed_H35": saving,
        "saving_valid_only_if_bad_zero": len(bad_rows) == 0,
    }


def evaluate_rule(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> Dict[str, Any]:
    preds = {str(r.get("base_state_id") or r.get("state_label") or i): predict_rule(rule, r) for i, r in enumerate(rows)}
    metric = evaluate_policy(rows, preds)
    metric["predictions"] = preds
    return metric


def complexity(rule: Mapping[str, Any]) -> int:
    return int(rule.get("h35_condition") is not None) + int(rule.get("h15_condition") is not None)


def fit_rule(train: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    conds = condition_candidates(train)
    best_rule: Optional[Dict[str, Any]] = None
    best_metric: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[Any, ...]] = None
    evaluated = 0
    for h35_cond, h15_cond, order in itertools.product(conds, conds, ("H35_first", "H15_first")):
        rule = {"default_horizon": 12, "h35_condition": h35_cond, "h15_condition": h15_cond, "order": order}
        metric = evaluate_rule(train, rule)
        evaluated += 1
        # Primary: no bad rows. Secondary: lower decision time. Tertiary: simpler, deterministic repr.
        repr_key = json.dumps(rule, sort_keys=True)
        score = (metric["bad_count"], metric["decision_sum_s"], complexity(rule), repr_key)
        if best_score is None or score < best_score:
            best_score = score
            best_rule = rule
            best_metric = metric
    assert best_rule is not None and best_metric is not None
    return {"rule": best_rule, "train_metric": best_metric, "rules_evaluated": evaluated, "score": best_score}


def row_key(row: Mapping[str, Any], idx: int) -> str:
    return str(row.get("base_state_id") or row.get("state_label") or idx)


def oracle_policy(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {row_key(r, i): int(r.get("oracle_h")) for i, r in enumerate(rows)}


def fixed_policy(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {row_key(r, i): h for i, r in enumerate(rows)}


def run_cv(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    folds = []
    if mode == "loo":
        fold_defs = [(f"row::{row_key(r, i)}", [i]) for i, r in enumerate(rows)]
    elif mode == "logo":
        groups = sorted({str(r["family_group"]) for r in rows})
        fold_defs = [(g, [i for i, r in enumerate(rows) if r["family_group"] == g]) for g in groups]
    else:
        raise ValueError(mode)
    all_preds: Dict[str, int] = {}
    bad_total = 0
    for name, test_idx in fold_defs:
        train = [r for i, r in enumerate(rows) if i not in set(test_idx)]
        test = [r for i, r in enumerate(rows) if i in set(test_idx)]
        fit = fit_rule(train)
        preds = {row_key(r, i): predict_rule(fit["rule"], r) for i, r in enumerate(rows) if i in set(test_idx)}
        metric = evaluate_policy(test, preds)
        train_classes = sorted({int(r.get("oracle_h")) for r in train})
        test_classes = sorted({int(r.get("oracle_h")) for r in test})
        folds.append({
            "fold": name,
            "test_rows": len(test),
            "train_oracle_classes": train_classes,
            "test_oracle_classes": test_classes,
            "missing_test_classes_from_train": [h for h in test_classes if h not in train_classes],
            "rule": fit["rule"],
            "train_metric": fit["train_metric"],
            "test_metric": metric,
            "predictions": preds,
        })
        bad_total += int(metric["bad_count"])
        all_preds.update(preds)
    all_metric = evaluate_policy(rows, all_preds)
    return {"mode": mode, "folds": folds, "aggregate_metric": all_metric, "bad_total_from_folds": bad_total}


def threshold_fragility(rows: Sequence[Mapping[str, Any]], post: Mapping[str, Any]) -> Dict[str, Any]:
    sep = post.get("separation_diagnostics") or {}
    t0 = float(sep.get("H35_threshold_midpoint"))
    t7 = float(sep.get("H15_threshold_midpoint"))
    records = []
    for r in rows:
        d0 = float(r["abs_obs_00"]) - t0
        d7 = float(r["abs_obs_07"]) - t7
        records.append({
            "state_label": r.get("state_label"),
            "family_group": r.get("family_group"),
            "oracle_h": r.get("oracle_h"),
            "abs_obs_00": r.get("abs_obs_00"),
            "signed_margin_to_H35_threshold_abs_obs_00": d0,
            "abs_margin_H35_feature": abs(d0),
            "abs_obs_07": r.get("abs_obs_07"),
            "signed_margin_to_H15_threshold_abs_obs_07": d7,
            "abs_margin_H15_feature": abs(d7),
        })
    return {
        "thresholds": {"abs_obs_00_H35_le": t0, "abs_obs_07_H15_gt": t7},
        "min_abs_margin_H35_feature": min(r["abs_margin_H35_feature"] for r in records),
        "min_abs_margin_H15_feature": min(r["abs_margin_H15_feature"] for r in records),
        "rows_within_0p02_H35_threshold": [r for r in records if r["abs_margin_H35_feature"] <= 0.02],
        "rows_within_0p05_H35_threshold": [r for r in records if r["abs_margin_H35_feature"] <= 0.05],
        "rows_within_0p10_H15_threshold": [r for r in records if r["abs_margin_H15_feature"] <= 0.10],
        "records": records,
    }


def family_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    families: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        g = str(r["family_group"])
        rec = families.setdefault(g, {"rows": 0, "labels": {}, "state_labels": [], "abs_obs_00_values": [], "abs_obs_07_values": []})
        rec["rows"] += 1
        rec["labels"][str(r.get("oracle_h"))] = rec["labels"].get(str(r.get("oracle_h")), 0) + 1
        rec["state_labels"].append(r.get("state_label"))
        rec["abs_obs_00_values"].append(float(r["abs_obs_00"]))
        rec["abs_obs_07_values"].append(float(r["abs_obs_07"]))
    label_to_fams: Dict[str, List[str]] = {}
    for g, rec in families.items():
        for lab in rec["labels"]:
            label_to_fams.setdefault(lab, []).append(g)
    exact_pairs: Dict[str, List[str]] = {}
    for r in rows:
        key = f"{float(r['abs_obs_00']):.12f}|{float(r['abs_obs_07']):.12f}"
        exact_pairs.setdefault(key, []).append(str(r.get("state_label")))
    duplicate_feature_pairs = {k: v for k, v in exact_pairs.items() if len(v) > 1}
    return {"families": families, "oracle_label_to_family_count": {k: len(v) for k, v in label_to_fams.items()}, "oracle_label_to_families": label_to_fams, "duplicate_feature_pairs": duplicate_feature_pairs}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-analysis-only-opened-development-rows", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_analysis_only_opened_development_rows:
        raise SystemExit("missing explicit analysis-only acknowledgement")

    created = now()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "run_started.json", {
        "started_utc": created.isoformat(),
        "method": NAME,
        "classification": "development_analysis_only_no_sim_no_validation_no_test_no_training",
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    rows, meta = load_rows()
    post = read_json(POST_RAW)
    input_hashes = {rel(p): sha256(p) for p in [Path(__file__).resolve(), V29_RAW, V30B_RAW, POST_RAW, POST_COMPLETED, INPUT_BACKUP_PROOF] if p.exists()}
    family = family_summary(rows)
    in_sample_fit = fit_rule(rows)
    in_sample_metric = evaluate_rule(rows, in_sample_fit["rule"])
    loo = run_cv(rows, "loo")
    logo = run_cv(rows, "logo")
    oracle_metric = evaluate_policy(rows, oracle_policy(rows))
    fixed_metrics = {str(h): evaluate_policy(rows, fixed_policy(rows, h)) for h in HORIZONS}
    fragility = threshold_fragility(rows, post)

    headline = {
        "rows": len(rows),
        "families": len(family["families"]),
        "oracle_label_to_family_count": family["oracle_label_to_family_count"],
        "in_sample_bad_two_feature_rule": in_sample_metric["bad_count"],
        "in_sample_saving_vs_fixed_H35": in_sample_metric["decision_saving_vs_fixed_H35"],
        "loo_bad_two_feature_rule": loo["aggregate_metric"]["bad_count"],
        "loo_saving_vs_fixed_H35": loo["aggregate_metric"]["decision_saving_vs_fixed_H35"],
        "logo_bad_two_feature_rule": logo["aggregate_metric"]["bad_count"],
        "logo_saving_vs_fixed_H35": logo["aggregate_metric"]["decision_saving_vs_fixed_H35"],
        "logo_bad_folds": [(f["fold"], f["test_metric"]["bad_count"], f["missing_test_classes_from_train"]) for f in logo["folds"] if f["test_metric"]["bad_count"]],
        "min_abs_margin_H35_feature": fragility["min_abs_margin_H35_feature"],
        "min_abs_margin_H15_feature": fragility["min_abs_margin_H15_feature"],
        "duplicate_feature_pairs": family["duplicate_feature_pairs"],
    }

    diagnosis = (
        "Two-feature row-level LOO stability can be achieved on the opened 11 rows, "
        "but grouped-by-source-family CV fails for the non-default H15 and H35 regimes because each has only one independent opened source family. "
        "This is a diagnostic of evidence dependence/label coverage, not a deployable selector validation."
    )
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_analysis_only_no_sim_no_validation_no_test_no_training",
        "input_hashes": input_hashes,
        "input_backup_proof": rel(INPUT_BACKUP_PROOF) if INPUT_BACKUP_PROOF.exists() else None,
        "input_backup_proof_exists": INPUT_BACKUP_PROOF.exists(),
        "meta": meta,
        "headline": headline,
        "diagnosis_for_astra": diagnosis,
        "family_summary": family,
        "oracle_metric": oracle_metric,
        "fixed_metrics": fixed_metrics,
        "in_sample_fit_two_feature": in_sample_fit,
        "in_sample_metric_two_feature": in_sample_metric,
        "loo_two_feature": loo,
        "logo_two_feature": logo,
        "threshold_fragility": fragility,
        "rows": rows,
        "budgets_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "git": {"head": git(["rev-parse", "HEAD"]), "status_subset": git(["status", "--short", "--", "experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py", "research_artifacts/aws_diagnostics", "docs/bohn2021_takeover/astra_reviews", "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"])},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)

    logo_lines = []
    for fold_name, bad, missing in headline["logo_bad_folds"]:
        logo_lines.append(f"- `{fold_name}`: bad rows `{bad}`, missing heldout oracle class(es) from train `{missing}`.")
    if not logo_lines:
        logo_lines.append("- none")
    summary = "\n".join([
        "# v31 v30b cluster-stability diagnostic",
        "",
        f"UTC: `{created.isoformat()}`. Development analysis only over already-opened v29/v30b/postdiagnostic artifacts; simulations=0, control steps=0, training/refit=0, validation64 closed, sealed test closed.",
        "",
        "## Headline",
        f"- Rows: `{headline['rows']}`; source-family groups: `{headline['families']}`; oracle-label family counts: `{headline['oracle_label_to_family_count']}`.",
        f"- Two-feature in-sample rule bad rows: `{headline['in_sample_bad_two_feature_rule']}`, saving vs fixed H35: `{headline['in_sample_saving_vs_fixed_H35']:.4%}`.",
        f"- Two-feature row-level LOO bad rows: `{headline['loo_bad_two_feature_rule']}`, saving vs fixed H35: `{headline['loo_saving_vs_fixed_H35']:.4%}`.",
        f"- Two-feature leave-one-source-family-out bad rows: `{headline['logo_bad_two_feature_rule']}`, nominal saving vs fixed H35: `{headline['logo_saving_vs_fixed_H35']:.4%}` (not a valid speed claim when bad>0).",
        f"- Minimum absolute margin to H35 threshold: `{headline['min_abs_margin_H35_feature']:.6f}`; to H15 threshold: `{headline['min_abs_margin_H15_feature']:.6f}`.",
        "",
        "## Leave-one-source-family-out failures",
        *logo_lines,
        "",
        "## Diagnostic conclusion for Astra",
        diagnosis,
        "",
        "## Limits",
        "This diagnostic adds no independent labels and uses outcome-opened development rows. Grouping uses audit-only source-family/category information to test dependence; it is not a deployable feature. Do not validate or final-test a selector from this result alone.",
        "",
        f"Raw: `{rel(OUT / 'raw.json')}`. Completed: `{rel(OUT / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]) + "\n"
    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    write_json(BACKUP_REQUEST, {
        "requested_utc": created.isoformat(),
        "reason": "v31 cluster-stability diagnostic produced new analysis artifacts and updated Astra handoff/docs; back up before simulations/refits/validation",
        "must_cover": [
            rel(Path(__file__).resolve()),
            rel(OUT / "raw.json"),
            rel(OUT / "summary.md"),
            rel(OUT / "completed.json"),
            rel(STATE_OUT),
            rel(NEXT_REVIEW),
            rel(RESPONSE_LOG),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
        ],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
    })

    review = {
        "created": created.isoformat(),
        "request_id": "v31-cluster-stability-diagnostic-20260930T0405Z",
        "experiment_id": NAME,
        "status": "analysis_requested",
        "trigger": "meaningful_cluster_stability_diagnostic_after_v30b_postdiagnostic",
        "question": "Astra should incorporate v31 cluster-stability evidence into the pending v29/v30b direction decision. The same opened rows are separable by two features at row-level LOO, but leave-one-source-family-out fails because non-default H15 and H35 regimes each have only one independent opened source family. Decide whether next execution should acquire fresh source-independent triage labels near the H35/H15 boundaries, perform terminal-risk/value refit/training, or change scenario/comparison design. GPT-5.5 should not treat v31 as deployable validation.",
        "evidence_paths": [
            rel(OUT / "summary.md"),
            rel(OUT / "raw.json"),
            rel(OUT / "completed.json"),
            rel(POST_RAW),
            rel(V30B_RAW),
            rel(V29_RAW),
            rel(RESPONSE_LOG),
        ],
    }
    write_json(NEXT_REVIEW, review)

    response_block = f"""## Follow-up v31 cluster-stability diagnostic

Updated by GPT-5.5 executor at `{created.isoformat()}`. v31 is analysis-only over already-opened v29/v30b/postdiagnostic outputs; no simulations/training/refit/validation64/sealed-test access.

| linked recommendation(s) | disposition after v31 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; strengthened | v31 groups the 11 opened rows into {len(family['families'])} source-family clusters. Oracle H15 and H35 each have only one independent opened source-family cluster: `{family['oracle_label_to_family_count']}`. | Do not treat row-level LOO or same-row separability as population/deployable evidence. Await Astra direction for fresh labels/refit/scenario decision. |
| `A11_training_failure_modes_need_separation` | accepted; refined | Two-feature row-level LOO bad={headline['loo_bad_two_feature_rule']} but leave-one-source-family-out bad={headline['logo_bad_two_feature_rule']}; failing folds: `{headline['logo_bad_folds']}`. | Evidence separates feature-form overfit/tiny-family coverage from complete feature absence. GPT-5.5 will execute Astra-selected next action. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; unchanged | v31 uses success-sensitive bad flags inherited from v29/v30b and reports fixed H12/H15/H35 comparators in raw. | Preserve fixed-H35 and shorter-H comparison; no validation/test selector claim. |
| `A12_registry_backup_schema_contract` | accepted; active | Backup request `{rel(BACKUP_REQUEST)}` written for v31 source/results/docs/handoff. | Require verified backup before any new simulation/refit/validation. |
"""
    append_once(RESPONSE_LOG, DOC_MARKER, response_block)

    doc_block = f"""## 2026-09-30 v31 cluster-stability diagnostic

UTC: {created.isoformat()}. Analysis-only over already-opened development artifacts; no simulations/control/training/refit/validation64/sealed-test access. Row-level two-feature LOO bad={headline['loo_bad_two_feature_rule']}, but leave-one-source-family-out bad={headline['logo_bad_two_feature_rule']} with non-default oracle H15/H35 represented by one source-family each. This is not deployable validation; it is evidence for Astra's pending direction choice. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, DOC_MARKER, doc_block)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    row = f"{created.isoformat()},{NAME},development_analysis_cluster_stability,no_validation_no_test,0,0,0,0,0,not_deployable,{rel(OUT / 'completed.json')}\n"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if NAME not in old[-50000:]:
        reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")

    STATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATE_OUT.write_text(summary, encoding="utf-8")

    completed_files = [Path(__file__).resolve(), OUT / "run_started.json", OUT / "raw.json", OUT / "summary.md", STATE_OUT, BACKUP_REQUEST, NEXT_REVIEW, RESPONSE_LOG]
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw["classification"],
        "headline": headline,
        "diagnosis_for_astra": diagnosis,
        "next_review_request": rel(NEXT_REVIEW),
        "backup_request": rel(BACKUP_REQUEST),
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "raw": rel(OUT / "raw.json"),
        "summary": rel(OUT / "summary.md"),
        "hashes": {rel(p): sha256(p) for p in sorted(set(completed_files), key=lambda p: rel(p)) if p.exists() and p.is_file()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "headline": headline,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_review_request": rel(NEXT_REVIEW),
        "backup_request": rel(BACKUP_REQUEST),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
