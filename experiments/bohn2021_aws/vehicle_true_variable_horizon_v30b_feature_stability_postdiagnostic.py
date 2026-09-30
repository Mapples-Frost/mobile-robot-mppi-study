#!/usr/bin/env python3
"""v30b feature-stability postdiagnostic.

Analysis-only over already opened v29/v30b artifacts.  No MPC simulation,
validation64 access, sealed-test access, training, refit, or model selection for
claiming validation.  Purpose: distinguish observable separability in the opened
v29 triage rows from tiny-sample rule-selection instability, and preserve an
Astra handoff with concrete numerical evidence.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic"
STAMP = "20260930T0355Z"
MARKER = f"vehicle-v30b-feature-stability-postdiagnostic-{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v30b_feature_stability_postdiagnostic.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_{STAMP}.json"
V29_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json"
V30B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json"
V30B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/completed.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
HORIZONS = [12, 15, 35]


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sf(x: Any) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else 0.0
    except Exception:
        return 0.0


def append_if_missing(p: Path, marker: str, block: str) -> None:
    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    if marker not in old:
        p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def safe_success(e: Mapping[str, Any]) -> bool:
    return bool(e.get("safe_success_no_solver_fail")) and not bool(e.get("hit_step_cap"))


def row_lookup(v29: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(r["base_state_id"]): r for r in v29["analysis"]["state_rows"]}


def per_h(row: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    return row["per_horizon"][str(h)]


def row_bad(row: Mapping[str, Any], h: int) -> Tuple[bool, str, float | None]:
    safe_phys = [sf(per_h(row, hh).get("physical_constraint_cost")) for hh in [12, 15, 25, 35] if safe_success(per_h(row, hh))]
    if not safe_success(per_h(row, h)):
        return True, "unsafe_or_step_cap", None
    if not safe_phys:
        return True, "no_safe_reference", None
    best = min(safe_phys)
    excess = sf(per_h(row, h).get("physical_constraint_cost")) - best
    tol = max(2.0, 0.25 * abs(best))
    return (excess > tol), ("large_physical_excess_vs_best_safe" if excess > tol else "ok"), excess


def evaluate(pred: Mapping[str, int], rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    bad_rows: List[Dict[str, Any]] = []
    counts = {"12": 0, "15": 0, "35": 0}
    decision = solver = phys = 0.0
    safe_count = 0
    for bid, h in pred.items():
        row = rows[bid]
        e = per_h(row, h)
        counts[str(h)] += 1
        decision += sf(e.get("decision_sum_s"))
        solver += sf(e.get("solver_sum_s"))
        phys += sf(e.get("physical_constraint_cost"))
        if safe_success(e):
            safe_count += 1
        bad, reason, excess = row_bad(row, h)
        if bad:
            bad_rows.append({"base_state_id": bid, "state_label": row.get("state_label"), "category_for_audit_only": row.get("category"), "predicted_horizon": h, "reason": reason, "physical_excess_vs_best_safe": excess})
    return {"states": len(pred), "h_counts": counts, "bad_count": len(bad_rows), "bad_rows": bad_rows, "safe_success_count": safe_count, "decision_sum_s": decision, "solver_sum_s": solver, "physical_sum": phys, "predictions": dict(pred)}


def threshold_mid(max_low: float, min_high: float) -> float:
    return (max_low + min_high) / 2.0


def main(argv: Sequence[str] | None = None) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v29 = read_json(V29_RAW)
    v30 = read_json(V30B_RAW)
    done = read_json(V30B_DONE)
    if done.get("validation64_bank_opened") or done.get("sealed_test_accessed") or v30.get("validation64_bank_opened") or v30.get("sealed_test_accessed"):
        raise RuntimeError("unexpected validation/test access marker in inputs")
    rows = row_lookup(v29)
    feats: Mapping[str, Mapping[str, float]] = v30["feature_values_by_state"]
    oracle_pred: Mapping[str, int] = {k: int(v) for k, v in v30["comparators"]["oracle_fastest_safe_H12_H15_H35"]["predictions"].items()}
    ins_pred: Mapping[str, int] = {k: int(v) for k, v in v30["learned_in_sample"]["predictions"].items()}
    loo_pred: Mapping[str, int] = {k: int(v) for k, v in v30["leave_one_out"]["predictions"].items()}

    ids = list(oracle_pred.keys())
    h35_ids = [bid for bid in ids if oracle_pred[bid] == 35]
    non_h35_ids = [bid for bid in ids if oracle_pred[bid] != 35]
    h15_ids = [bid for bid in ids if oracle_pred[bid] == 15]
    h12_ids = [bid for bid in ids if oracle_pred[bid] == 12]

    h35_max = max(sf(feats[b]["abs_obs_00"]) for b in h35_ids)
    non_h35_min = min(sf(feats[b]["abs_obs_00"]) for b in non_h35_ids)
    h35_thr = threshold_mid(h35_max, non_h35_min)
    h15_min = min(sf(feats[b]["abs_obs_07"]) for b in h15_ids)
    h12_max = max(sf(feats[b]["abs_obs_07"]) for b in h12_ids)
    h15_thr = threshold_mid(h12_max, h15_min)
    margin_h35 = non_h35_min - h35_max
    margin_h15 = h15_min - h12_max

    margin_rule: Dict[str, int] = {}
    for bid in ids:
        if sf(feats[bid]["abs_obs_00"]) <= h35_thr:
            margin_rule[bid] = 35
        elif sf(feats[bid]["abs_obs_07"]) > h15_thr:
            margin_rule[bid] = 15
        else:
            margin_rule[bid] = 12
    margin_metric = evaluate(margin_rule, rows)

    fixed_h35_decision = sf(v30["comparators"]["fixed_H35"]["decision_sum_s"])
    saving_margin_vs_h35 = (fixed_h35_decision - margin_metric["decision_sum_s"]) / fixed_h35_decision
    saving_loo_vs_h35 = sf(v30["headline"]["loo_decision_saving_vs_fixed_H35"])
    saving_oracle_vs_h35 = sf(v30["headline"]["oracle_decision_saving_vs_fixed_H35"])

    folds = v30["leave_one_out"].get("folds", [])
    h15_feature_counts: Dict[str, int] = {}
    h35_feature_counts: Dict[str, int] = {}
    bad_folds: List[Dict[str, Any]] = []
    for f in folds:
        r = f.get("rule", {})
        h15f = str((r.get("h15_condition") or {}).get("feature"))
        h35f = str((r.get("h35_condition") or {}).get("feature"))
        h15_feature_counts[h15f] = h15_feature_counts.get(h15f, 0) + 1
        h35_feature_counts[h35f] = h35_feature_counts.get(h35f, 0) + 1
        if int(f.get("heldout_bad_count") or 0) > 0:
            bad_folds.append(f)

    state_table = []
    for bid in ids:
        row = rows[bid]
        state_table.append({
            "state_label": row.get("state_label"),
            "base_state_id": bid,
            "category_for_audit_only": row.get("category"),
            "oracle_h": oracle_pred[bid],
            "in_sample_h": ins_pred[bid],
            "loo_h": loo_pred[bid],
            "margin_rule_h": margin_rule[bid],
            "abs_obs_00": sf(feats[bid]["abs_obs_00"]),
            "abs_obs_07": sf(feats[bid]["abs_obs_07"]),
            "branch_step": sf(feats[bid]["branch_step"]),
            "H12_bad": row_bad(row, 12)[0],
            "H15_bad": row_bad(row, 15)[0],
            "H35_bad": row_bad(row, 35)[0],
        })

    if margin_metric["bad_count"] == 0 and v30["headline"]["loo_bad"] == 1:
        diagnosis = "opened features are separable in-sample with a low-complexity margin rule, but leave-one-out model selection is unstable on the tiny risk cluster; this is not yet deployable evidence."
    else:
        diagnosis = "feature separability remains ambiguous; do not validate a selector without fresh labels or a stronger/refit representation."

    created = now()
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_analysis_only_feature_stability_postdiagnostic_no_sim_no_validation_no_test_no_training",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "inputs": {"v29_raw": rel(V29_RAW), "v29_raw_sha256": sha256(V29_RAW), "v30b_raw": rel(V30B_RAW), "v30b_raw_sha256": sha256(V30B_RAW), "v30b_completed": rel(V30B_DONE), "v30b_completed_sha256": sha256(V30B_DONE)},
        "separation_diagnostics": {"oracle_counts": {"H12": len(h12_ids), "H15": len(h15_ids), "H35": len(h35_ids)}, "H35_rule_feature": "abs_obs_00", "H35_max_oracle_H35": h35_max, "H35_min_non_H35": non_h35_min, "H35_margin": margin_h35, "H35_threshold_midpoint": h35_thr, "H15_rule_feature_after_H35_guard": "abs_obs_07", "H15_min_oracle_H15": h15_min, "H12_max_oracle_H12": h12_max, "H15_margin": margin_h15, "H15_threshold_midpoint": h15_thr},
        "margin_rule": {"order": "if abs_obs_00 <= threshold choose H35; elif abs_obs_07 > threshold choose H15; else H12", "thresholds": {"abs_obs_00_H35_le": h35_thr, "abs_obs_07_H15_gt": h15_thr}, "metric": margin_metric, "decision_saving_vs_fixed_H35": saving_margin_vs_h35},
        "v30b_reference": {"oracle_bad": v30["headline"]["oracle_bad"], "oracle_saving_vs_fixed_H35": saving_oracle_vs_h35, "in_sample_bad": v30["headline"]["in_sample_bad"], "loo_bad": v30["headline"]["loo_bad"], "loo_saving_vs_fixed_H35": saving_loo_vs_h35, "loo_bad_rows": v30["leave_one_out"].get("bad_rows", [])},
        "loo_rule_instability": {"h15_condition_feature_counts": h15_feature_counts, "h35_condition_feature_counts": h35_feature_counts, "bad_folds": bad_folds, "unique_h15_features": sorted(h15_feature_counts), "unique_h35_features": sorted(h35_feature_counts)},
        "state_table": state_table,
        "diagnosis": diagnosis,
        "executor_non_directional_next_note": "Await Astra scientific direction. Operationally, a post-v30b backup is required before new simulations/refits/validation. If Astra requests label acquisition, focus fresh source-independent pre-outcome states near the low-margin H35 abs_obs_00 boundary and v19 H15-risk family; if Astra requests refit/training, use this diagnostic as the separability/overfit baseline.",
        "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), V29_RAW, V30B_RAW, V30B_DONE]},
        "backup_request": rel(BACKUP_REQUEST),
    }
    write_json(RUN_DIR / "raw.json", raw)

    lines = [
        "# v30b feature-stability postdiagnostic",
        "",
        f"UTC: `{raw['created_utc']}`. Analysis-only over v29/v30b; simulations=0, control steps=0, validation64 closed, sealed test closed, training/refit=0.",
        "",
        "## Headline",
        "",
        f"- v30b oracle H12/H15/H35 triage: bad={raw['v30b_reference']['oracle_bad']}, saving vs fixed H35={100*saving_oracle_vs_h35:.2f}% on 11 opened states.",
        f"- v30b leave-one-out feature rule: bad={raw['v30b_reference']['loo_bad']}, nominal saving vs fixed H35={100*saving_loo_vs_h35:.2f}% (not valid as a safe deployable claim).",
        f"- Margin-rule postdiagnostic on the same opened rows: bad={margin_metric['bad_count']}, H counts={margin_metric['h_counts']}, saving vs fixed H35={100*saving_margin_vs_h35:.2f}%.",
        f"- Diagnosis: {diagnosis}",
        "",
        "## Separability margins",
        "",
        f"- H35 rescue feature `abs_obs_00`: max oracle-H35={h35_max:.6g}, min non-H35={non_h35_min:.6g}, margin={margin_h35:.6g}, midpoint threshold={h35_thr:.6g}. This margin is small and based on only {len(h35_ids)} H35 rows.",
        f"- H15 risk feature after H35 guard `abs_obs_07`: min oracle-H15={h15_min:.6g}, max oracle-H12={h12_max:.6g}, margin={margin_h15:.6g}, midpoint threshold={h15_thr:.6g}. This is more separated but still based on only {len(h15_ids)} H15 rows.",
        "",
        "## LOO instability",
        "",
        f"- H15-condition feature counts across LOO folds: `{h15_feature_counts}`.",
        f"- H35-condition feature counts across LOO folds: `{h35_feature_counts}`.",
        f"- Bad LOO folds: `{[(f.get('heldout_state'), f.get('predicted_horizon')) for f in bad_folds]}`.",
        "",
        "## State table",
        "",
        "| state | category | oracle | LOO | margin rule | abs_obs_00 | abs_obs_07 | H12_bad | H15_bad | H35_bad |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in state_table:
        lines.append(f"| `{s['state_label']}` | `{s['category_for_audit_only']}` | {s['oracle_h']} | {s['loo_h']} | {s['margin_rule_h']} | {s['abs_obs_00']:.6g} | {s['abs_obs_07']:.6g} | {s['H12_bad']} | {s['H15_bad']} | {s['H35_bad']} |")
    lines += [
        "",
        "## Limits and handoff",
        "",
        "This does not add independent evidence. It is a numerical postdiagnostic of already opened development rows selected after earlier outcomes. It supports Astra review by showing that the v30b failure is dominated by tiny-sample model-selection instability rather than complete lack of observable feature signal, while also showing the H35 boundary is low-margin. No validation64/final-test access is authorized by this result.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(BACKUP_REQUEST, {"request": "backup_after_v30b_feature_stability_postdiagnostic", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v30b feature-stability postdiagnostic and updated handoff/docs", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})

    doc_block = f"""<!-- {MARKER} -->
## 2026-09-30 v30b feature-stability postdiagnostic

UTC: {raw['created_utc']}. Analysis-only over opened v29/v30b; simulations=0, control_steps=0, training/refit=0, validation64 closed, sealed test closed. Diagnostic result: {diagnosis} Margin-rule same-opened-row metric bad={margin_metric['bad_count']}, H counts={margin_metric['h_counts']}, decision saving vs fixed H35={100*saving_margin_vs_h35:.2f}%; v30b LOO remained bad={raw['v30b_reference']['loo_bad']}. H35 margin on `abs_obs_00` is {margin_h35:.6g} from only {len(h35_ids)} rows; H15 margin on `abs_obs_07` after H35 guard is {margin_h15:.6g} from {len(h15_ids)} rows. This is not independent confirmation; it is a handoff diagnostic for Astra. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, doc_block)
    response_block = f"""
<!-- {MARKER} -->
## Follow-up v30b feature-stability postdiagnostic

Updated by GPT-5.5 executor at `{raw['created_utc']}`. This is a numerical postdiagnostic over already-opened v29/v30b outputs; validation64 and sealed test remained closed.

| linked recommendation(s) | disposition after postdiagnostic | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A11_training_failure_modes_need_separation` | accepted; sharpened for Astra decision | Same-opened-row margin rule has bad={margin_metric['bad_count']} and saving={100*saving_margin_vs_h35:.2f}% vs fixed H35, while v30b LOO has bad={raw['v30b_reference']['loo_bad']} because of a v19 H12-risk heldout. H15 condition features vary across LOO folds: `{h15_feature_counts}`. | Evidence suggests tiny-sample model-selection instability despite observable in-sample signal; await Astra to choose fresh label acquisition vs terminal-risk/value refit/training. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; no final comparator claim | Fixed-H35 is safe-successful but has physical-excess bad rows; oracle triage bad=0; all are opened development rows with possible horizon/terminal confounds from v29. | Do not validate selector; preserve fixed H12/H15/H25/H35 comparators and terminal-confound caveat. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | Only 11 opened v29 states; thresholds use outcome-informed labels. | Require fresh independent confirmation before validation/final claims. |
| `A12_registry_backup_schema_contract` | accepted; active | Backup request `{rel(BACKUP_REQUEST)}` written. | Require verified backup before unique simulations/refits/validation. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response_block)

    next_request = {
        "request_id": "v30b-feature-stability-postdiagnostic-20260930T0355Z",
        "trigger": "meaningful_postdiagnostic_after_v30b_and_user_role_correction",
        "created": created.isoformat(),
        "experiment_id": os.environ.get("EXPERIMENT_ID", "postdiagnostic-run"),
        "status": "analysis_requested",
        "question": "Astra should lead scientific interpretation and choose the next research direction after v29/v30b plus this feature-stability postdiagnostic. Evidence now shows oracle H12/H15/H35 triage is safe on opened rows; the v30b LOO learned rule has one bad row; a same-opened-row margin rule is separable but not independent and the H35 boundary is low-margin. Decide whether the next controlled action should be fresh source-independent triage label acquisition, bounded terminal-risk/value refit/training, or scenario/comparison redesign/stratification. GPT-5.5 should execute the selected plan after backup.",
        "evidence_paths": [rel(V29_RAW), rel(V30B_RAW), rel(V30B_DONE), rel(RUN_DIR / "summary.md"), rel(RUN_DIR / "raw.json"), rel(RUN_DIR / "completed.json"), "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"],
    }
    write_json(ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json", next_request)

    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "deterministic_existing_evidence", "opened_development_v29_v30b_only", 0, 0, 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v30b feature-stability postdiagnostic\n\nUTC: {created.isoformat()}\n\nResult: {diagnosis}\n\nNo simulations/training/validation/test. Backup required: {rel(BACKUP_REQUEST)}\n\nAstra handoff updated: docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json\n\nNext executor action: verify postdiagnostic backup and check ANALYSIS_READY.json before starting fresh labels/refit/scenario branch.\n", encoding="utf-8")

    completed = {"status": "complete", "hard_pass": True, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "new_simulation_episodes": 0, "new_control_steps": 0, "new_training_or_gradient_steps": 0, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "diagnosis": diagnosis, "headline": {"margin_rule_bad": margin_metric["bad_count"], "margin_rule_saving_vs_fixed_H35": saving_margin_vs_h35, "loo_bad": raw["v30b_reference"]["loo_bad"], "oracle_bad": raw["v30b_reference"]["oracle_bad"], "H35_margin_abs_obs_00": margin_h35, "H15_margin_abs_obs_07": margin_h15}, "backup_request": rel(BACKUP_REQUEST)}
    write_json(RUN_DIR / "completed.json", completed)
    raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
    write_json(RUN_DIR / "raw.json", raw)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": completed["headline"], "diagnosis": diagnosis, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
