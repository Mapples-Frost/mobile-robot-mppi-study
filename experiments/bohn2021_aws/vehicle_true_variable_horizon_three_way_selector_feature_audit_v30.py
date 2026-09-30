#!/usr/bin/env python3
"""v30 three-way selector feature/separability audit after v29.

Development-only analysis.  This script reads the v29 H12/H15/H25/H35
identical-state outcomes and asks a narrow question before any new simulation or
training: are the three observed action regimes (H35 for source242 rescue,
H15 for v19 H12-only risk, H12 for fresh safe controls) even separable by
pre-outcome deployable state/observation features without using source IDs,
roles, category labels, or outcome fields at prediction time?

It runs no MPC, no validation64, no sealed test, no refit/training.  Because it
uses v29 outputs as inputs, it requires an external backup proof postdating v29.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_three_way_selector_feature_audit_v30"
STAMP = "20260930T0355Z"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0355_after_v30_three_way_feature_audit.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30_THREE_WAY_SELECTOR_FEATURE_AUDIT_{STAMP}.json"
MARKER = f"vehicle-three-way-selector-feature-audit-v30-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V29_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json"
V29_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/completed.json"
V29_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/summary.md"
HORIZONS = [12, 15, 25, 35]
SELECTOR_HORIZONS = [12, 15, 35]

class ContractError(RuntimeError):
    pass

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
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
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default

def parse_time(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str):
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def verify_completed(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing v29 completion marker")
    obj = read_json(path)
    if obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError("v29 completion marker is not complete")
    if obj.get("validation64_bank_opened") is True or obj.get("sealed_test_accessed") is True:
        raise ContractError("v29 marker unexpectedly reports validation/test access")
    return obj

def verify_backup(path: Path, after_time: Optional[dt.datetime]) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("backup proof missing: " + rel(path))
    obj = read_json(path)
    if obj.get("status") != "verified" or obj.get("backup_verified") is not True:
        raise ContractError("backup proof is not verified")
    if obj.get("remaining_changed_files") not in (0, "0"):
        raise ContractError("backup proof has remaining changed files")
    if not obj.get("commit") or not obj.get("packages_this_run"):
        raise ContractError("backup proof lacks commit/package evidence")
    if after_time is not None:
        t = parse_time(obj.get("time"))
        if t is None or t <= after_time:
            raise ContractError("backup proof does not postdate v29")
    return obj

def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def outcome(row: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    return row["per_horizon"][str(h)]

def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("safe_success_no_solver_fail")) and not bool(e.get("hit_step_cap"))

def row_min_safe_physical(row: Mapping[str, Any]) -> float:
    vals = [sf(outcome(row, h).get("physical_constraint_cost")) for h in HORIZONS if is_safe(outcome(row, h))]
    return min(vals) if vals else float("inf")

def row_bad_for_h(row: Mapping[str, Any], h: int) -> Tuple[bool, str, float]:
    e = outcome(row, h)
    if not is_safe(e):
        return True, "unsafe_or_step_cap", float("inf")
    min_phys = row_min_safe_physical(row)
    phys = sf(e.get("physical_constraint_cost"))
    # Success-sensitive physical guard: tolerate small numerical/control-cost
    # variation, but mark horizon choices with very large continuation-cost
    # excess as bad even if the episode eventually reaches the goal.
    tol = max(2.0, 0.25 * abs(min_phys))
    excess = phys - min_phys
    if excess > tol:
        return True, "large_physical_excess_vs_best_safe", excess
    return False, "ok", excess

def total_metrics(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, int]) -> Dict[str, Any]:
    bad_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {str(h): 0 for h in HORIZONS}
    decision = solver = physical = 0.0
    safe_count = 0
    for row in rows:
        bid = str(row["base_state_id"])
        h = int(predictions[bid])
        counts[str(h)] = counts.get(str(h), 0) + 1
        e = outcome(row, h)
        decision += sf(e.get("decision_sum_s"))
        solver += sf(e.get("solver_sum_s"))
        physical += sf(e.get("physical_constraint_cost"))
        safe = is_safe(e)
        if safe:
            safe_count += 1
        bad, reason, excess = row_bad_for_h(row, h)
        if bad:
            bad_rows.append({
                "base_state_id": bid,
                "state_label": row.get("state_label"),
                "category_for_audit_only": row.get("category"),
                "predicted_horizon": h,
                "reason": reason,
                "physical_excess_vs_best_safe": excess if math.isfinite(excess) else None,
            })
    return {
        "states": len(rows),
        "h_counts": counts,
        "safe_success_count": safe_count,
        "bad_count": len(bad_rows),
        "bad_rows": bad_rows,
        "decision_sum_s": decision,
        "solver_sum_s": solver,
        "physical_sum": physical,
    }

def fixed_predictions(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {str(r["base_state_id"]): h for r in rows}

def fastest_safe_predictions(rows: Sequence[Mapping[str, Any]], allowed: Sequence[int]) -> Dict[str, int]:
    preds: Dict[str, int] = {}
    for row in rows:
        safe = [h for h in allowed if is_safe(outcome(row, h))]
        if not safe:
            # Keep deterministic fallback; metric will mark bad.
            preds[str(row["base_state_id"])] = int(allowed[0])
        else:
            preds[str(row["base_state_id"])] = min(safe, key=lambda h: sf(outcome(row, h).get("decision_sum_s")))
    return preds

def build_features(raw: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    state_specs = {str(s.get("base_state_id")): s for s in raw.get("selected_states") or []}
    out: Dict[str, Dict[str, float]] = {}
    for row in raw["analysis"]["state_rows"]:
        bid = str(row["base_state_id"])
        spec = state_specs.get(bid)
        if not spec:
            raise ContractError("missing selected-state spec for " + bid)
        feats: Dict[str, float] = {}
        obs = spec.get("initial_observation") or []
        for i, val in enumerate(obs):
            feats[f"obs_{i:02d}"] = sf(val)
            feats[f"abs_obs_{i:02d}"] = abs(sf(val))
        prev = spec.get("branch_previous_state") or {}
        for k in ("x", "y", "theta"):
            feats[f"prev_{k}"] = sf(prev.get(k))
            feats[f"abs_prev_{k}"] = abs(sf(prev.get(k)))
        feats["branch_step"] = sf(row.get("branch_step"))
        vals = [sf(v) for v in obs]
        feats["obs_l2"] = math.sqrt(sum(v * v for v in vals)) if vals else 0.0
        feats["obs_linf"] = max([abs(v) for v in vals] or [0.0])
        feats["prev_xy_radius"] = math.sqrt(feats.get("prev_x", 0.0) ** 2 + feats.get("prev_y", 0.0) ** 2)
        # Intentionally excluded from deployable features: source_candidate_index,
        # campaign/category/role labels, base_state_id, and any branch outcome.
        out[bid] = feats
    return out

def condition_candidates(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]) -> List[Tuple[str, str, float]]:
    names = sorted(next(iter(features.values())).keys())
    conds: List[Tuple[str, str, float]] = [("__false__", "false", 0.0)]
    for name in names:
        vals = sorted({features[str(r["base_state_id"])][name] for r in rows if math.isfinite(features[str(r["base_state_id"])][name])})
        if len(vals) <= 1:
            continue
        thresholds = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:]) if a != b]
        for thr in thresholds:
            conds.append((name, "<=", thr))
            conds.append((name, ">", thr))
    return conds

def cond_true(cond: Tuple[str, str, float], feats: Mapping[str, float]) -> bool:
    name, op, thr = cond
    if name == "__false__":
        return False
    val = sf(feats.get(name))
    return val <= thr if op == "<=" else val > thr

def predict_rule(rule: Mapping[str, Any], feats: Mapping[str, float]) -> int:
    c35 = tuple(rule["h35_condition"])
    c15 = tuple(rule["h15_condition"])
    if rule["order"] == "H35_first":
        if cond_true(c35, feats):
            return 35
        if cond_true(c15, feats):
            return 15
        return 12
    if cond_true(c15, feats):
        return 15
    if cond_true(c35, feats):
        return 35
    return 12

def eval_rule(rule: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    preds = {str(r["base_state_id"]): predict_rule(rule, features[str(r["base_state_id"])]) for r in rows}
    m = total_metrics(rows, preds)
    m["predictions"] = preds
    return m

def score_key(metric: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(metric["bad_count"]),
        -float(metric["safe_success_count"]),
        float(metric["decision_sum_s"]),
        float(metric["physical_sum"]),
    )

def learn_rule(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    conds = condition_candidates(rows, features)
    best_rule: Optional[Dict[str, Any]] = None
    best_metric: Optional[Dict[str, Any]] = None
    for c35, c15, order in itertools.product(conds, conds, ("H35_first", "H15_first")):
        rule = {"default_horizon": 12, "h35_condition": list(c35), "h15_condition": list(c15), "order": order}
        metric = eval_rule(rule, rows, features)
        if best_metric is None or score_key(metric) < score_key(best_metric):
            best_rule, best_metric = rule, metric
    assert best_rule is not None and best_metric is not None
    return best_rule, best_metric

def leave_one_out(rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    heldout_predictions: Dict[str, int] = {}
    folds: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        rule, train_metric = learn_rule(train, features)
        bid = str(row["base_state_id"])
        pred = predict_rule(rule, features[bid])
        heldout_predictions[bid] = pred
        held_metric = total_metrics([row], {bid: pred})
        folds.append({
            "heldout_state": bid,
            "heldout_category_for_audit_only": row.get("category"),
            "predicted_horizon": pred,
            "heldout_bad_count": held_metric["bad_count"],
            "rule": rule,
            "train_bad_count": train_metric["bad_count"],
            "train_decision_sum_s": train_metric["decision_sum_s"],
        })
    metric = total_metrics(rows, heldout_predictions)
    metric["folds"] = folds
    metric["predictions"] = heldout_predictions
    return metric

def pct_saving(a: float, b: float) -> float:
    return 0.0 if b <= 0 else (b - a) / b

def write_summary(raw: Mapping[str, Any]) -> None:
    c = raw["comparators"]
    ins = raw["learned_in_sample"]
    loo = raw["leave_one_out"]
    lines = [
        "# v30 three-way selector feature/separability audit",
        "",
        f"UTC: `{raw['created_utc']}`. Analysis-only over v29; no MPC simulations, no validation64, no sealed test, no training/refit.",
        "",
        "## Question",
        "",
        "Can a simple deployable pre-outcome feature rule separate the v29 regimes: H35 for source242 H12/H15/H25 failures, H15 for v19 H12-only high-cost rows, and H12 for fresh safe controls? Source/campaign/role/category labels and outcome fields are not predictor features.",
        "",
        "## Comparator metrics on the 11 opened v29 states",
        "",
        "| policy | H counts | bad rows | safe successes | decision sum s | physical sum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["fixed_H12", "fixed_H15", "fixed_H25", "fixed_H35", "oracle_fastest_safe_H12_H15_H35"]:
        m = c[name]
        lines.append(f"| `{name}` | `{m['h_counts']}` | {m['bad_count']} | {m['safe_success_count']} | {m['decision_sum_s']:.6g} | {m['physical_sum']:.6g} |")
    lines += [
        "",
        "## Simple two-threshold rule audit",
        "",
        f"- Best in-sample rule: `{raw['learned_in_sample_rule']}`",
        f"- In-sample bad rows: `{ins['bad_count']}`, decision sum `{ins['decision_sum_s']:.6g}` s, H counts `{ins['h_counts']}`.",
        f"- Leave-one-out bad rows: `{loo['bad_count']}`, decision sum `{loo['decision_sum_s']:.6g}` s, H counts `{loo['h_counts']}`.",
        f"- LOO saving vs fixed H35: `{100.0 * raw['headline']['loo_decision_saving_vs_fixed_H35']:.2f}%` (only meaningful if LOO bad rows are zero).",
        "",
        "## Decision",
        "",
        raw["scientific_decision"],
        "",
        "## Limits",
        "",
        "This is a tiny opened development audit after outcome-discovering v29. Passing this audit would only justify a fresh development confirmation protocol, not validation64/final claims. Failing this audit would not prove impossible adaptivity; it would indicate that the current hand-engineered observable feature set is insufficient or overfit.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def update_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v30 three-way selector feature/separability audit

UTC: {raw['created_utc']}. Analysis-only over v29; validation64 closed, sealed test closed, no simulations/training/refit. Fixed baselines on the 11 opened v29 states: H12 bad={raw['comparators']['fixed_H12']['bad_count']}, H15 bad={raw['comparators']['fixed_H15']['bad_count']}, H35 bad={raw['comparators']['fixed_H35']['bad_count']}. Oracle fastest safe among H12/H15/H35 bad={raw['comparators']['oracle_fastest_safe_H12_H15_H35']['bad_count']}, decision_sum={raw['comparators']['oracle_fastest_safe_H12_H15_H35']['decision_sum_s']:.6g}s. In-sample two-threshold rule bad={raw['learned_in_sample']['bad_count']}; leave-one-out bad={raw['leave_one_out']['bad_count']}; LOO saving vs fixed H35={100.0*h['loo_decision_saving_vs_fixed_H35']:.2f}% if safety holds. Decision: {raw['scientific_decision']} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required before further unique science: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
## Follow-up through v30 three-way selector feature/separability audit

Updated by GPT-5.5 executor at `{raw['created_utc']}`. v30 is analysis-only over v29 and did not access validation64 or sealed test.

| linked recommendation(s) | disposition after v30 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; broadened to H12/H15/H35 triage evidence | v30 compares fixed H12/H15/H25/H35 and oracle/feature rules on v29 states; fixed H35 bad={raw['comparators']['fixed_H35']['bad_count']}, fixed H12 bad={raw['comparators']['fixed_H12']['bad_count']}, fixed H15 bad={raw['comparators']['fixed_H15']['bad_count']}. | If a selector is pursued, compare against fixed H35 and fixed H12/H15, not only H15. |
| `A11_training_failure_modes_need_separation` | accepted; refit/training gate updated | Leave-one-out two-threshold audit bad={raw['leave_one_out']['bad_count']}; LOO saving vs fixed H35={100.0*raw['headline']['loo_decision_saving_vs_fixed_H35']:.2f}% conditional on safety. | Use this to decide whether a fresh triage-selector confirmation, terminal-risk/value refit, or scenario redesign is more informative. |
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; preserved | Source242 rows require H35 in v29; failed H12/H15/H25 timing is excluded from success claims. | Continue absolute success-sensitive accounting. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v30 uses only the 11 selected opened v29 states. | Development-only mechanism audit; require fresh independent confirmation. |
| `A12_registry_backup_schema_contract` | accepted; active | v30 wrote analysis/docs/state/registry and backup request `{rel(BACKUP_REQUEST)}`. | Require verified backup before more unique simulation/refit/validation. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], "deterministic_v29_analysis", "development_analysis_no_validation64_no_test", 0, 0, 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v30 three-way feature audit\n\nUTC: {raw['created_utc']}\n\nDecision: {raw['scientific_decision']}\n\nHeadline: {raw['headline']}\n\nNext: require verified backup covering v30. Then follow the decision: fresh triage-selector development confirmation if LOO safety/separability is good, otherwise collect more source-independent labels or pivot to terminal-risk/value refit or scenario redesign.\n", encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-proof", required=True)
    ap.add_argument("--i-accept-development-v30", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if (RUN_DIR / "completed.json").exists():
            done = read_json(RUN_DIR / "completed.json")
            print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
            return 0
        v29_done = verify_completed(V29_DONE)
        v29_time = parse_time(v29_done.get("created_utc"))
        backup = verify_backup(Path(args.backup_proof), v29_time)
        raw29 = read_json(V29_RAW)
        if raw29.get("validation64_bank_opened") is True or raw29.get("sealed_test_accessed") is True:
            raise ContractError("v29 raw unexpectedly reports validation/test access")
        rows = list(raw29["analysis"]["state_rows"])
        if len(rows) != 11:
            raise ContractError(f"unexpected v29 state count {len(rows)}")
        features = build_features(raw29)
        comparators: Dict[str, Any] = {}
        for h in HORIZONS:
            comparators[f"fixed_H{h}"] = total_metrics(rows, fixed_predictions(rows, h))
        oracle_preds = fastest_safe_predictions(rows, SELECTOR_HORIZONS)
        comparators["oracle_fastest_safe_H12_H15_H35"] = total_metrics(rows, oracle_preds)
        rule, ins_metric = learn_rule(rows, features)
        loo_metric = leave_one_out(rows, features)
        fixed_h35 = comparators["fixed_H35"]
        oracle = comparators["oracle_fastest_safe_H12_H15_H35"]
        ins_saving_vs_h35 = pct_saving(ins_metric["decision_sum_s"], fixed_h35["decision_sum_s"])
        loo_saving_vs_h35 = pct_saving(loo_metric["decision_sum_s"], fixed_h35["decision_sum_s"])
        oracle_saving_vs_h35 = pct_saving(oracle["decision_sum_s"], fixed_h35["decision_sum_s"])
        if loo_metric["bad_count"] == 0 and loo_saving_vs_h35 >= 0.05:
            decision = "Feature separability is promising on this tiny opened audit. After backup, freeze a fresh development-only source-independent H12/H15/H35 triage-selector confirmation before any validation; do not retrain yet."
        elif oracle["bad_count"] == 0 and oracle_saving_vs_h35 >= 0.05:
            decision = "Oracle triage opportunity is real on v29, but the simple deployable feature rule did not pass leave-one-out. After backup, acquire more pre-outcome source-independent labels or run a bounded terminal-risk/value refit rather than validating the current selector."
        else:
            decision = "v29 does not provide enough deployable feature-separable triage value beyond strong fixed H35; prioritize scenario/comparison redesign or a negative adaptive-opportunity conclusion for this stress-v1 slice."
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_analysis_only_three_way_selector_feature_audit_no_sim_no_validation_no_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "new_simulation_episodes": 0,
            "new_control_steps": 0,
            "new_training_or_gradient_steps": 0,
            "backup_proof_used": {"path": rel(Path(args.backup_proof)), "sha256": sha256(Path(args.backup_proof)), "commit": backup.get("commit")},
            "v29_inputs": {"raw": rel(V29_RAW), "raw_sha256": sha256(V29_RAW), "completed": rel(V29_DONE), "completed_sha256": sha256(V29_DONE), "summary": rel(V29_SUMMARY), "summary_sha256": sha256(V29_SUMMARY)},
            "features_excluded_to_avoid_leakage": ["source_candidate_index", "source_campaign", "category", "role", "base_state_id", "state_label", "per_horizon_outcomes", "success/cost/timing fields"],
            "features_used": sorted(next(iter(features.values())).keys()),
            "comparators": comparators,
            "learned_in_sample_rule": rule,
            "learned_in_sample": ins_metric,
            "leave_one_out": loo_metric,
            "headline": {
                "states": len(rows),
                "fixed_H12_bad": comparators["fixed_H12"]["bad_count"],
                "fixed_H15_bad": comparators["fixed_H15"]["bad_count"],
                "fixed_H35_bad": comparators["fixed_H35"]["bad_count"],
                "oracle_bad": oracle["bad_count"],
                "oracle_decision_saving_vs_fixed_H35": oracle_saving_vs_h35,
                "in_sample_bad": ins_metric["bad_count"],
                "in_sample_decision_saving_vs_fixed_H35": ins_saving_vs_h35,
                "loo_bad": loo_metric["bad_count"],
                "loo_decision_saving_vs_fixed_H35": loo_saving_vs_h35,
            },
            "scientific_decision": decision,
            "interpretation_limits": ["opened v29 development states only", "tiny sample", "rules selected after observing v29 outcomes", "not validation64", "not sealed test", "not population estimate", "not ORIGINAL SAC"],
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
            "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), V29_RAW, V29_DONE, V29_SUMMARY, Path(args.backup_proof)] if p.exists()},
            "backup_request": rel(BACKUP_REQUEST),
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v30_three_way_selector_feature_audit", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v30 development analysis/docs/state/registry/response-log", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "new_control_steps": 0, "new_simulation_episodes": 0, "new_training_or_gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        update_docs(raw)
        completed = {"status": "complete", "hard_pass": True, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "new_control_steps": 0, "new_simulation_episodes": 0, "new_training_or_gradient_steps": 0, "headline": raw["headline"], "scientific_decision": decision, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "backup_request": rel(BACKUP_REQUEST), "hashes": {}}
        files = [Path(__file__).resolve(), RUN_DIR / "raw.json", RUN_DIR / "summary.md", V29_RAW, V29_DONE, V29_SUMMARY, STATE, BACKUP_REQUEST, ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
        completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
        write_json(RUN_DIR / "completed.json", completed)
        raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": raw["headline"], "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_analysis_only_three_way_selector_feature_audit_no_sim_no_validation_no_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
