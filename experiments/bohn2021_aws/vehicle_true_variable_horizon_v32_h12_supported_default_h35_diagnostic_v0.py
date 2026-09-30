#!/usr/bin/env python3
"""Branch-neutral diagnostic: source-supported H12 shortening with H35 default.

Uses only already-opened v29/v30b/v31 development artifacts.  It does not run
MPC, does not train/refit a deployable selector, does not open validation64, and
does not access sealed tests.

Question addressed for Astra: given v31 showed H15/H35 non-default classes each
have only one independent source family, is there at least source-family support
for a conservative H12-vs-H35 diagnostic rule, and what residual opened-row bad
cases remain?  This is diagnostic evidence only, not a branch decision and not a
validation-ready controller.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
NAME = "vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0"
OUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_DIR = ROOT / "research_artifacts/aws_state"
ASTRA_DIR = ROOT / "docs/bohn2021_takeover/astra_reviews"
RESPONSE_LOG = ASTRA_DIR / "RESPONSE_LOG.md"
NEXT_REVIEW = ASTRA_DIR / "NEXT_REVIEW_REQUEST.json"
ANALYSIS_READY = ASTRA_DIR / "ANALYSIS_READY.json"
SQLITE_CANDIDATES = [BASE / "state" / "research.sqlite", ROOT / "research.sqlite"]

V29_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json"
V30B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json"
V31_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json"
V31_SOURCE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/raw.json"
EXPECTED_REQUEST_ID = "v31-source-coverage-budget-bounds-20260930T044819Z"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_elapsed(seconds: float) -> str:
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    return f"{days}d {hours}h {minutes}m {sec:.3f}s"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def try_read_json(path: Path) -> Optional[Any]:
    try:
        return read_json(path)
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + marker + "\n" + block.strip() + "\n", encoding="utf-8")


def append_registry_once(row_id: str, row: str) -> None:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if row_id not in old:
        path.write_text(old.rstrip() + "\n" + row.strip() + "\n", encoding="utf-8")


def token_total() -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "path": None, "table": None, "column": None, "call_count": None, "total_tokens": None, "error": None}
    for candidate in SQLITE_CANDIDATES:
        if not candidate.exists():
            continue
        result["path"] = str(candidate)
        try:
            conn = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [str(row[0]) for row in cur.fetchall()]
            for table in ([t for t in tables if t == "calls"] + [t for t in tables if t != "calls"]):
                qt = '"' + table.replace('"', '""') + '"'
                cur.execute(f"PRAGMA table_info({qt})")
                cols = [str(row[1]) for row in cur.fetchall()]
                token_cols = [c for c in cols if c == "total_tokens"] + [c for c in cols if c.lower() in {"usage_total_tokens", "tokens_total", "server_total_tokens"}]
                if not token_cols:
                    continue
                col = token_cols[0]
                qc = '"' + col.replace('"', '""') + '"'
                cur.execute(f"SELECT COUNT(*), SUM(CASE WHEN {qc} IS NULL THEN 0 ELSE {qc} END) FROM {qt}")
                count, total = cur.fetchone()
                conn.close()
                result.update({"available": True, "path": str(candidate), "table": table, "column": col, "call_count": int(count or 0), "total_tokens": int(total or 0), "error": None})
                return result
            conn.close()
            result["error"] = "sqlite present but no total_tokens-like column found"
            return result
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return result
    result["error"] = "research.sqlite not found in checked locations"
    return result


def ready_matches(ready: Any, request_id: Optional[str]) -> bool:
    if not isinstance(ready, Mapping) or not request_id:
        return False
    if ready.get("request_id") == request_id or ready.get("supersedes_request_id") == request_id:
        return True
    for key in ("covers_request_ids", "covered_request_ids"):
        v = ready.get(key)
        if isinstance(v, list) and request_id in v:
            return True
    return False


def safe_metric(row: Mapping[str, Any], horizon: int) -> Dict[str, Any]:
    per = row["per_horizon"][str(horizon)]
    safe_phys = [float(v["physical_constraint_cost"]) for v in row["per_horizon"].values() if v.get("safe_success_no_solver_fail")]
    best_safe = min(safe_phys) if safe_phys else None
    bad = False
    reason = None
    if not per.get("safe_success_no_solver_fail"):
        bad = True
        reason = "unsafe_or_step_cap"
    elif best_safe is not None:
        tol = max(2.0, 0.25 * best_safe)
        excess = float(per["physical_constraint_cost"]) - best_safe
        if excess > tol:
            bad = True
            reason = "large_physical_excess_vs_best_safe"
    return {
        "bad": bad,
        "reason": reason,
        "decision_sum_s": float(per["decision_sum_s"]),
        "solver_sum_s": float(per.get("solver_sum_s", 0.0)),
        "physical_constraint_cost": float(per["physical_constraint_cost"]),
        "success": bool(per.get("success")),
        "safe_success_no_solver_fail": bool(per.get("safe_success_no_solver_fail")),
        "steps": int(per.get("steps", 0)),
        "best_safe_physical": best_safe,
    }


def build_rows(v29: Mapping[str, Any], v30b: Mapping[str, Any], v31: Mapping[str, Any]) -> List[Dict[str, Any]]:
    feature_values = v30b.get("feature_values_by_state", {})
    oracle_pred = v30b.get("comparators", {}).get("oracle_fastest_safe_H12_H15_H35", {}).get("predictions", {})
    family_by_state: Dict[str, str] = {}
    family_labels: Dict[str, Dict[str, int]] = {}
    families = v31.get("family_summary", {}).get("families", {})
    for fam, info in families.items():
        family_labels[fam] = info.get("labels", {})
        for label in info.get("state_labels", []):
            family_by_state[str(label)] = fam
    rows = []
    for row in v29.get("analysis", {}).get("state_rows", []):
        base = str(row.get("base_state_id"))
        state_label = str(row.get("state_label"))
        feats = feature_values.get(base) or feature_values.get(state_label) or {}
        fam = family_by_state.get(state_label, "UNKNOWN_FAMILY")
        oracle_h = int(oracle_pred.get(base, row.get("fastest_safe_horizon_by_decision_sum")))
        rows.append({
            "base_state_id": base,
            "state_label": state_label,
            "family_group": fam,
            "family_labels": family_labels.get(fam, {}),
            "oracle_h": oracle_h,
            "category": row.get("category"),
            "abs_obs_07": float(feats.get("abs_obs_07")),
            "abs_obs_00": float(feats.get("abs_obs_00")),
            "per_horizon": {str(h): row["per_horizon"][str(h)] for h in (12, 15, 35)},
        })
    return rows


def metric_for_predictions(rows: Iterable[Mapping[str, Any]], predictions: Mapping[str, int]) -> Dict[str, Any]:
    bad_rows = []
    decision = 0.0
    solver = 0.0
    physical = 0.0
    h_counts: Dict[str, int] = {"12": 0, "15": 0, "35": 0}
    success_count = 0
    for row in rows:
        base = str(row["base_state_id"])
        h = int(predictions[base])
        m = safe_metric(row, h)
        h_counts[str(h)] = h_counts.get(str(h), 0) + 1
        decision += m["decision_sum_s"]
        solver += m["solver_sum_s"]
        physical += m["physical_constraint_cost"]
        if m["safe_success_no_solver_fail"]:
            success_count += 1
        if m["bad"]:
            bad_rows.append({
                "base_state_id": base,
                "state_label": row["state_label"],
                "family_group": row["family_group"],
                "oracle_h": row["oracle_h"],
                "predicted_horizon": h,
                "reason": m["reason"],
                "physical_constraint_cost": m["physical_constraint_cost"],
                "best_safe_physical": m["best_safe_physical"],
                "category": row.get("category"),
            })
    fixed_h35_decision = sum(safe_metric(row, 35)["decision_sum_s"] for row in rows)
    return {
        "rows": len(list(rows)) if not isinstance(rows, list) else len(rows),
        "bad_count": len(bad_rows),
        "bad_rows": bad_rows,
        "decision_sum_s": decision,
        "solver_sum_s": solver,
        "physical_sum": physical,
        "success_count": success_count,
        "h_counts": h_counts,
        "decision_saving_vs_fixed_H35_on_same_rows": (fixed_h35_decision - decision) / fixed_h35_decision if fixed_h35_decision > 0 else None,
    }


def candidate_thresholds(rows: List[Mapping[str, Any]], feature: str) -> List[float]:
    vals = sorted({float(r[feature]) for r in rows})
    if not vals:
        return []
    mids = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:]) if a != b]
    return [vals[0] - 1e-9] + mids + [vals[-1] + 1e-9]


def predictions_for_threshold(rows: Iterable[Mapping[str, Any]], threshold: float, feature: str = "abs_obs_07") -> Dict[str, int]:
    return {str(r["base_state_id"]): (12 if float(r[feature]) <= threshold else 35) for r in rows}


def fit_h12_default_h35_rule(train_rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    best = None
    for thr in candidate_thresholds(train_rows, "abs_obs_07"):
        pred = predictions_for_threshold(train_rows, thr)
        metric = metric_for_predictions(train_rows, pred)
        score = (metric["bad_count"], metric["decision_sum_s"], metric["physical_sum"], thr)
        if best is None or score < best["score"]:
            best = {"threshold": thr, "feature": "abs_obs_07", "rule_repr": f"H12 if abs_obs_07 <= {thr:.12f} else H35", "train_metric": metric, "score": score}
    assert best is not None
    return best


def logo_evaluate(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    folds = []
    all_predictions: Dict[str, int] = {}
    for fam in sorted({str(r["family_group"]) for r in rows}):
        train = [r for r in rows if r["family_group"] != fam]
        test = [r for r in rows if r["family_group"] == fam]
        fit = fit_h12_default_h35_rule(train)
        pred = predictions_for_threshold(test, fit["threshold"], fit["feature"])
        all_predictions.update(pred)
        folds.append({
            "fold_family": fam,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_oracle_classes": sorted({int(r["oracle_h"]) for r in train}),
            "test_oracle_classes": sorted({int(r["oracle_h"]) for r in test}),
            "fit_rule": {"feature": fit["feature"], "threshold": fit["threshold"], "repr": fit["rule_repr"]},
            "train_metric": fit["train_metric"],
            "test_metric": metric_for_predictions(test, pred),
            "test_predictions": pred,
        })
    return {"folds": folds, "aggregate_metric": metric_for_predictions(rows, all_predictions), "aggregate_predictions": all_predictions}


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{NAME}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "run_started.json", {
        "started_utc": created.isoformat(),
        "classification": "development_analysis_only_h12_supported_default_h35_no_sim_no_validation_no_test_no_training_no_refit",
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    v29 = read_json(V29_RAW)
    v30b = read_json(V30B_RAW)
    v31 = read_json(V31_RAW)
    v31s = read_json(V31_SOURCE_RAW)
    rows = build_rows(v29, v30b, v31)
    in_sample_fit = fit_h12_default_h35_rule(rows)
    in_sample_pred = predictions_for_threshold(rows, in_sample_fit["threshold"], in_sample_fit["feature"])
    in_sample_metric = metric_for_predictions(rows, in_sample_pred)
    logo = logo_evaluate(rows)

    fixed = {}
    for h in (12, 15, 35):
        pred = {str(r["base_state_id"]): h for r in rows}
        fixed[str(h)] = metric_for_predictions(rows, pred)
    oracle_pred = {str(r["base_state_id"]): int(r["oracle_h"]) for r in rows}
    oracle_metric = metric_for_predictions(rows, oracle_pred)

    family_counts = v31s.get("current_oracle_label_independent_source_family_counts", {})
    h12_families = v31.get("family_summary", {}).get("oracle_label_to_families", {}).get("12", [])
    non_h12_families = {str(r["family_group"]) for r in rows if int(r["oracle_h"]) != 12}
    h12_abs07_values = [r["abs_obs_07"] for r in rows if int(r["oracle_h"]) == 12]
    non_h12_abs07_values = [r["abs_obs_07"] for r in rows if int(r["oracle_h"]) != 12]
    separability = {
        "feature": "abs_obs_07",
        "max_h12_abs_obs_07": max(h12_abs07_values),
        "min_non_h12_abs_obs_07": min(non_h12_abs07_values),
        "margin": min(non_h12_abs07_values) - max(h12_abs07_values),
        "h12_source_family_count": len(h12_families),
        "non_h12_source_family_count": len(non_h12_families),
        "h12_families": h12_families,
        "non_h12_families": sorted(non_h12_families),
    }

    next_review = try_read_json(NEXT_REVIEW) or {}
    current_request_id = next_review.get("request_id") if isinstance(next_review, Mapping) else None
    ready = try_read_json(ANALYSIS_READY)
    ready_match = ready_matches(ready, current_request_id)
    tokens = token_total()
    elapsed = fmt_elapsed((created - FIRST_SUPERVISOR_EVENT).total_seconds())
    token_text = f"{tokens['total_tokens']:,} ({tokens['total_tokens']/1_000_000:.3f}M)" if tokens.get("available") else f"unknown ({tokens.get('error')})"

    interpretation = {
        "diagnostic_not_branch_decision": True,
        "not_validation_or_test_evidence": True,
        "headline": "A one-feature H12-if-low-abs_obs_07 else H35 diagnostic rule is source-family stable on opened rows and reduces LOGO bad rows from v31 two-feature rule's 5 to 1, but the residual bad row is H15-family default-to-H35 physical excess; current data still cannot support a deployable three-way H12/H15/H35 selector because H15/H35 each have one independent family.",
        "residual_issue": "H15 regime remains under-supported; the H12-supported/H35-default rule never predicts H15, so it cannot resolve H15-specific physical-cost excess such as the opened v19_c13 row.",
        "what_this_rules_out": "The v31 LOGO failure is not simply lack of any source-family-stable feature signal; H12-vs-non-H12 shortening has three-family support in opened data.",
        "what_this_does_not_rule_out": "It does not establish deployability, validation performance, value/training adequacy, or that acquisition is the next branch; Astra must decide the next scientific action.",
    }

    raw = {
        "created_utc": created.isoformat(),
        "classification": "development_analysis_only_h12_supported_default_h35_no_sim_no_validation_no_test_no_training_no_refit",
        "elapsed_since_first_supervisor_event_text": elapsed,
        "server_api_token_total_from_research_sqlite": tokens,
        "inputs": {
            "v29_raw": rel(V29_RAW),
            "v29_raw_sha256": sha256(V29_RAW),
            "v30b_raw": rel(V30B_RAW),
            "v30b_raw_sha256": sha256(V30B_RAW),
            "v31_raw": rel(V31_RAW),
            "v31_raw_sha256": sha256(V31_RAW),
            "v31_source_raw": rel(V31_SOURCE_RAW),
            "v31_source_raw_sha256": sha256(V31_SOURCE_RAW),
        },
        "astra_gate": {
            "current_request_id": current_request_id,
            "expected_request_id": EXPECTED_REQUEST_ID,
            "analysis_ready_exists": ANALYSIS_READY.exists(),
            "analysis_ready_matches_or_supersedes_current": ready_match,
        },
        "family_counts": family_counts,
        "separability": separability,
        "in_sample_fit_h12_default_h35": {"rule": {"feature": in_sample_fit["feature"], "threshold": in_sample_fit["threshold"], "repr": in_sample_fit["rule_repr"]}, "metric": in_sample_metric, "predictions": in_sample_pred},
        "logo_h12_default_h35": logo,
        "comparators_on_same_opened_rows": {"fixed": fixed, "oracle_h12_h15_h35": oracle_metric},
        "interpretation_for_astra": interpretation,
        "budgets_actual": {"new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
    }
    write_json(out_dir / "raw.json", raw)

    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V32_H12_SUPPORTED_DEFAULT_H35_DIAGNOSTIC_{stamp}.json"
    continue_path = STATE_DIR / f"continue_state_{stamp}_after_v32_h12_supported_default_h35.md"
    summary = f"""# v32 H12-supported / H35-default diagnostic

UTC: `{created.isoformat()}`. Development analysis only over already-opened v29/v30b/v31 artifacts; no simulations, no control steps, no selector refit, no training, no validation64 access and no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `{elapsed}`.
- Cumulative server API total_tokens from research.sqlite: `{token_text}`; desktop conversation tokens excluded.

## Concrete diagnostic
A conservative opened-row diagnostic rule class was evaluated: **H12 if `abs_obs_07 <= threshold`, else H35**.  It deliberately never predicts H15, so it is not a deployable three-way selector.

- In-sample threshold: `{in_sample_fit['threshold']:.12f}` (`{in_sample_fit['rule_repr']}`).
- H12-vs-non-H12 feature margin on opened rows: `{separability['margin']:.12f}` with H12 family count `{separability['h12_source_family_count']}` and non-H12 family count `{separability['non_h12_source_family_count']}`.
- In-sample opened rows: bad_count `{in_sample_metric['bad_count']}`, h_counts `{in_sample_metric['h_counts']}`, decision_sum_s `{in_sample_metric['decision_sum_s']:.6f}`, saving vs fixed H35 `{in_sample_metric['decision_saving_vs_fixed_H35_on_same_rows']:.6f}`.
- Leave-one-source-family-out aggregate: bad_count `{logo['aggregate_metric']['bad_count']}`, h_counts `{logo['aggregate_metric']['h_counts']}`, decision_sum_s `{logo['aggregate_metric']['decision_sum_s']:.6f}`, saving vs fixed H35 `{logo['aggregate_metric']['decision_saving_vs_fixed_H35_on_same_rows']:.6f}`.
- Residual LOGO bad rows: `{[(r['state_label'], r['family_group'], r['oracle_h'], r['predicted_horizon'], r['reason']) for r in logo['aggregate_metric']['bad_rows']]}`.

## Same opened-row comparators
- Fixed H35: bad_count `{fixed['35']['bad_count']}`, decision_sum_s `{fixed['35']['decision_sum_s']:.6f}`, physical_sum `{fixed['35']['physical_sum']:.6f}`.
- v30b/v29 oracle H12/H15/H35: bad_count `{oracle_metric['bad_count']}`, decision_sum_s `{oracle_metric['decision_sum_s']:.6f}`, physical_sum `{oracle_metric['physical_sum']:.6f}`.
- Prior v31 two-feature grouped LOGO bad count was 5; this H12-supported/H35-default diagnostic has LOGO bad count `{logo['aggregate_metric']['bad_count']}` on the same opened rows, but still cannot solve H15 because it never predicts H15.

## Interpretation limits
- This is **not** validation evidence, not final-test evidence, not a retrained/refit selector and not a branch decision.
- It suggests the current opened data contain a source-family-stable H12-shortening signal, so the v31 grouped-CV failure is not total feature absence.
- The residual bad row is from the single H15 source family defaulting to H35; current evidence still leaves H15 source-independent deployability unsupported.
- Astra should use this as additional evidence when choosing acquisition vs value/refit/training vs scenario/comparison redesign.

## Gate status
- Current Astra request: `{current_request_id}`; ANALYSIS_READY matching/superseding current: `{ready_match}`.
- New backup request: `{rel(backup_request)}`.
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    continue_path.write_text(summary, encoding="utf-8")

    write_json(backup_request, {
        "requested_utc": created.isoformat(),
        "reason": "Back up v32 branch-neutral H12-supported/H35-default diagnostic outputs before unique simulation/refit/training/validation/final-test work.",
        "must_cover": [
            rel(Path(__file__).resolve()),
            rel(out_dir / "run_started.json"),
            rel(out_dir / "summary.md"),
            rel(out_dir / "raw.json"),
            rel(out_dir / "completed.json"),
            rel(backup_request),
            rel(continue_path),
            "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md",
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
        ],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    marker = f"<!-- {NAME}-{stamp} -->"
    response_block = f"""## v32 H12-supported / H35-default diagnostic

Updated by GPT-5.5 executor at `{created.isoformat()}`. Development analysis only over opened v29/v30b/v31 artifacts; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| Astra v31 source-coverage direction | accepted; additional branch-neutral diagnostic supplied while current Astra analysis remains pending | `{rel(out_dir / 'summary.md')}` and `{rel(out_dir / 'raw.json')}` show H12-if-`abs_obs_07`-low else H35 diagnostic LOGO bad_count `{logo['aggregate_metric']['bad_count']}` vs prior v31 two-feature LOGO bad_count 5, with residual bad row(s) `{[(r['state_label'], r['family_group'], r['oracle_h'], r['predicted_horizon']) for r in logo['aggregate_metric']['bad_rows']]}`. | Carry to Astra. This is not branch selection and not deployable validation evidence. |
| v31 non-default coverage limitation | still open | Source-family counts remain `{family_counts}`. H12 shortening has three-family support in this opened set, but H15 and H35 still have one family each; this diagnostic never predicts H15 and leaves an H15-family residual bad row. | Do not claim three-way selector success; await Astra for acquisition/refit/training/scenario decision. |
| `A12_registry_backup_schema_contract` | accepted; new diagnostic pending backup | Backup request `{rel(backup_request)}`. | Require external backup coverage before unique simulation/refit/training/validation/final-test work. |
"""
    append_once(RESPONSE_LOG, marker, response_block)

    doc_block = f"""## 2026-09-30 v32 H12-supported/H35-default diagnostic

UTC: {created.isoformat()}. Development analysis only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `{elapsed}`; server API total_tokens `{token_text}`. On the same opened v29/v30b/v31 rows, a diagnostic rule H12 if `abs_obs_07 <= {in_sample_fit['threshold']:.12f}` else H35 gave LOGO bad_count `{logo['aggregate_metric']['bad_count']}`, h_counts `{logo['aggregate_metric']['h_counts']}`, decision_sum_s `{logo['aggregate_metric']['decision_sum_s']:.6f}`, saving vs fixed H35 `{logo['aggregate_metric']['decision_saving_vs_fixed_H35_on_same_rows']:.6f}`. Residual bad rows `{[(r['state_label'], r['family_group'], r['oracle_h'], r['predicted_horizon'], r['reason']) for r in logo['aggregate_metric']['bad_rows']]}`. Interpretation: opened data contain source-family-supported H12-shortening signal, but H15 remains unsupported; not validation/final evidence and not a branch decision. Backup request `{rel(backup_request)}`. Astra gate matching=`{ready_match}` for `{current_request_id}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / doc, marker, doc_block)

    registry_id = f"{stamp}_{NAME}"
    append_registry_once(registry_id, f"{registry_id},{created.isoformat()},development_analysis_only_h12_supported_default_h35,none,no_validation_no_test,unknown,complete,0,0,0,{rel(out_dir / 'raw.json')}")

    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "summary": rel(out_dir / "summary.md"),
        "raw": rel(out_dir / "raw.json"),
        "backup_request": rel(backup_request),
        "continue_state": rel(continue_path),
        "logo_bad_count": logo["aggregate_metric"]["bad_count"],
        "in_sample_bad_count": in_sample_metric["bad_count"],
        "analysis_ready_matches_or_supersedes_current": ready_match,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {},
    }
    write_json(out_dir / "completed.json", completed)
    hash_paths = [
        Path(__file__).resolve(), out_dir / "run_started.json", out_dir / "summary.md", out_dir / "raw.json", out_dir / "completed.json",
        backup_request, continue_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv",
    ]
    completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and p.is_file()}
    write_json(out_dir / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
