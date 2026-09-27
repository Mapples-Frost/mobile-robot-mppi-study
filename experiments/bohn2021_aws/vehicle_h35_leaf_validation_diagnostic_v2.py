#!/usr/bin/env python3
"""Targeted H35-leaf diagnostic for vehicle learned_s2 validation behavior, v2.

Development analysis only. Reads existing validation64 rollout artifacts and the
case43 instrumented replay output; does not reopen scenario banks, run
simulation, train, or touch the sealed test bank.

v2 supersedes the failed v1 diagnostic by (1) preserving v1 failure artifacts,
(2) using actual rollout time indices for H35 events when trace rows omit a
``step`` field, and (3) carrying first_h35 details into the raw per-case table
used by the Markdown summary.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
EPISODE_TABLE = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/episode_table.csv"
POLICY_PATH = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train/vehicle_s2/policy.json"
INSTRUMENTED_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/raw.json"
V1_FAILURE = ROOT / "research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v1/failure.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v1/raw.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v2"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_h35_leaf_validation_diagnostic_v2.py"
DOC_MARKER = "vehicle-h35-leaf-validation-diagnostic-v2-20260927"
LEARNED_KEY = "learned_s2"
FIXED_KEY = "fixed_seed2_terminal25_controllerH25"
FEATURE_NAMES = [
    "tracking_error", "heading_error_5", "heading_error_15", "heading_error_30",
    "reference_turn_30", "current_clearance", "preview_clearance_15", "preview_clearance_30",
    "speed", "abs_yaw_input", "remaining", "previous_initial_failure", "previous_final_failure",
]


class DiagnosticError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnum(x: Any) -> float:
    if x is None or x == "":
        return math.nan
    return float(x)


def inum(x: Any) -> int:
    if x is None or x == "":
        return 0
    return int(float(x))


def bval(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes")


def stats(values: Iterable[float]) -> Dict[str, Any]:
    data = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not data:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(data) == 1:
            return data[0]
        i = (len(data) - 1) * p / 100.0
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        if lo == hi:
            return data[lo]
        return data[lo] * (hi - i) + data[hi] * (i - lo)

    return {"count": len(data), "sum": float(math.fsum(data)), "mean": float(math.fsum(data) / len(data)),
            "median": float(pct(50)), "p95": float(pct(95)), "min": data[0], "max": data[-1]}


def load_episode_table() -> List[Dict[str, Any]]:
    with EPISODE_TABLE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise DiagnosticError("empty episode table")
    return rows


def numeric_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for k in ("shard", "execution_index", "seed", "case", "steps", "initial_failed_steps", "solver_failure_steps", "retries", "deadline_exceed_steps", "switches"):
        out[k] = inum(row.get(k))
    for k in ("total_cost", "physical_constraint_cost", "performance_cost", "h_penalty", "decision_mean_s", "decision_sum_s", "decision_p95_s", "solver_mean_s", "solver_sum_s", "solver_p95_s", "construction_s", "reset_gross_s", "episode_wall_s"):
        out[k] = fnum(row.get(k))
    out["success"] = bval(row.get("success"))
    out["episode_failure"] = bval(row.get("episode_failure"))
    return out


def trace_path(summary_path: str) -> Path:
    p = ROOT / summary_path
    candidate = p.parent / "trace.json"
    if not candidate.exists():
        raise DiagnosticError(f"missing trace for {summary_path}")
    return candidate


def get_features(row: Mapping[str, Any]) -> Optional[List[float]]:
    vals = row.get("tree_features") or (row.get("decision") or {}).get("features")
    if vals is None:
        return None
    return [float(v) for v in vals]


def h35_detail(trace: List[Mapping[str, Any]], policy: Mapping[str, Any]) -> Dict[str, Any]:
    h35 = []
    leaf_counts: Dict[str, int] = {}
    feature_values: Dict[str, List[float]] = {name: [] for name in FEATURE_NAMES}
    for idx, row in enumerate(trace):
        if int(row.get("horizon")) != 35:
            continue
        # Validation trace rows may omit a ``step`` key. v1 incorrectly used the
        # ordinal H35-event count in that case; v2 uses the true rollout index.
        step = int(row.get("step", idx))
        decision = row.get("decision") or {}
        leaf = decision.get("leaf")
        leaf_counts[str(leaf)] = leaf_counts.get(str(leaf), 0) + 1
        feats = get_features(row)
        item: Dict[str, Any] = {"step": step, "trace_index": idx, "leaf": leaf}
        if feats and len(feats) == len(FEATURE_NAMES):
            for name, val in zip(FEATURE_NAMES, feats):
                feature_values[name].append(val)
            root = policy["nodes"][0]
            side = int(feats[int(root["feature"])] > float(root["threshold"]))
            node = policy["nodes"][1 + side]
            item.update({
                "root_feature": FEATURE_NAMES[int(root["feature"])],
                "root_value": feats[int(root["feature"])],
                "root_threshold": float(root["threshold"]),
                "root_side": side,
                "node_feature": FEATURE_NAMES[int(node["feature"])],
                "node_value": feats[int(node["feature"])],
                "node_threshold": float(node["threshold"]),
                "node_side": int(feats[int(node["feature"])] > float(node["threshold"])),
                "heading_error_5": feats[1],
                "abs_yaw_input": feats[9],
                "tracking_error": feats[0],
                "remaining": feats[10],
            })
        h35.append(item)
    feature_stats = {name: stats(vals) for name, vals in feature_values.items() if vals}
    return {"h35_count": len(h35), "first_h35": h35[0] if h35 else None, "h35_steps": [x["step"] for x in h35],
            "leaf_counts_at_h35": leaf_counts, "h35_feature_stats": feature_stats}


def append_once(path: Path, marker: str, body: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = f"<!-- {marker} -->"
    if token in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")
    return True


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    a = raw["aggregate"]
    case43 = raw["per_case_by_case"].get("43", {})
    lines = [
        "# Vehicle learned_s2 H35-leaf validation diagnostic v2",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development diagnostic only: reads existing validation64 rollout artifacts and the prior case43 replay; no new simulation, no training, no scenario-bank reopen, no sealed-test access/hash.",
        "",
        "v2 note: the failed v1 diagnostic wrote partial artifacts but used H35-event ordinal rather than rollout step when trace rows omitted `step`; v2 preserves v1 and corrects that step-index bug.",
        "",
        "## Policy logic under diagnosis",
        "",
        f"- learned_s2 policy leaves: `{raw['policy']['leaves']}`.",
        f"- H35 is leaf 0 and occurs when `{a['h35_rule_human']}` under the stored depth-2 tree.",
        "",
        "## Validation64 H35 usage vs same-seed fixed H25",
        "",
        f"- Cases with any H35: `{a['cases_with_h35_count']}/64`; total H35 steps: `{a['total_h35_steps']}`.",
        f"- H35 cases success/failure: `{a['h35_cases_success_count']}/{a['h35_cases_failure_count']}`; no-H35 cases success/failure: `{a['no_h35_cases_success_count']}/{a['no_h35_cases_failure_count']}`.",
        f"- First-H35 step stats over H35 cases: `{a['first_h35_step_stats']}`.",
        f"- Early H35 cases (first H35 step <=5): `{a['early_h35_cases']}`.",
        f"- Singleton-H35 cases: `{a['singleton_h35_cases']}`.",
        f"- Failure cases: `{a['failed_cases']}`.",
        f"- Learned_s2 minus fixed seed2 H25 physical cost, all cases: `{a['delta_physical_all_stats']}`.",
        f"- Same delta excluding case43: `{a['delta_physical_without_case43_stats']}`.",
        f"- Learned_s2 minus fixed seed2 H25 decision mean seconds, all cases: `{a['delta_decision_mean_all_stats']}`.",
        "",
        "## Case43 causal context",
        "",
        f"- Instrumented replay diagnosis: {raw.get('instrumented_case43_diagnosis')}",
        f"- Case43 H35 detail: `{case43.get('first_h35')}`.",
        f"- Case43 H35 steps: `{case43.get('h35_steps')}`.",
        f"- Case43 learned/fixed physical costs: `{case43.get('learned_physical')}` vs `{case43.get('fixed_h25_physical')}`.",
        "",
        "## Top H35-count cases",
        "",
        "| case | h35_count | first_h35_step | learned_success | fixed_success | delta_physical | learned_steps | fixed_steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in raw["top_h35_cases"]:
        lines.append(f"| {r['case']} | {r['h35_count']} | {r.get('first_h35_step')} | {r['learned_success']} | {r['fixed_h25_success']} | {r['delta_physical']:.6g} | {r['learned_steps']} | {r['fixed_h25_steps']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The H35 branch is not a runtime dispatch bug: it is the stored learned_s2 tree's leaf-0 rule. Existing validation plus the deterministic case43 ablation show that a single early H35 can directly put this vehicle case onto the catastrophic trajectory, even though the solver reports accepted solves and no retries/failures. H35 also lengthens rather than shortens the H25 baseline, and the full validation aggregate showed no timing advantage. The next IMPROVED revision should therefore test a safe-shortening/risk-sensitive policy-extraction rule rather than rewarding occasional longer horizons for training-set cost improvements.",
        "",
        "## Artifacts",
        "",
        f"- Raw: `{raw['artifacts']['raw']}`",
        f"- Per-case CSV: `{raw['artifacts']['per_case_csv']}`",
        f"- Completed: `{raw['artifacts']['completed']}`",
        f"- Backup request: `{raw['artifacts']['backup_request']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            for name, expected in done.get("hashes", {}).items():
                if sha256(ROOT / name) != expected:
                    raise DiagnosticError(f"existing completed hash mismatch: {name}")
            print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
            return 0
        raise DiagnosticError("prior completed marker exists but did not pass")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise DiagnosticError(f"partial output exists; preserve before retry: {rel(OUT_DIR)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    policy = read_json(POLICY_PATH)
    if policy.get("task") != "vehicle" or policy.get("kind") != "tree":
        raise DiagnosticError("unexpected learned_s2 policy schema")
    rows = [numeric_row(r) for r in load_episode_table()]
    learned = {r["case"]: r for r in rows if r.get("rollout_key") == LEARNED_KEY}
    fixed = {r["case"]: r for r in rows if r.get("rollout_key") == FIXED_KEY}
    if len(learned) != 64 or len(fixed) != 64 or set(learned) != set(range(64)) or set(fixed) != set(range(64)):
        raise DiagnosticError(f"missing learned/fixed rows: learned={len(learned)} fixed={len(fixed)}")

    per_case: List[Dict[str, Any]] = []
    all_trace_hashes: Dict[str, str] = {}
    total_h35_steps = 0
    for case in range(64):
        lr, fr = learned[case], fixed[case]
        tp = trace_path(lr["summary_path"])
        trace = read_json(tp)
        all_trace_hashes[rel(tp)] = sha256(tp)
        detail = h35_detail(trace, policy)
        total_h35_steps += detail["h35_count"]
        first = detail["first_h35"] or {}
        per_case.append({
            "case": case,
            "learned_success": bool(lr["success"]),
            "fixed_h25_success": bool(fr["success"]),
            "learned_steps": int(lr["steps"]),
            "fixed_h25_steps": int(fr["steps"]),
            "learned_physical": float(lr["physical_constraint_cost"]),
            "fixed_h25_physical": float(fr["physical_constraint_cost"]),
            "delta_physical": float(lr["physical_constraint_cost"] - fr["physical_constraint_cost"]),
            "learned_total": float(lr["total_cost"]),
            "fixed_h25_total": float(fr["total_cost"]),
            "delta_total": float(lr["total_cost"] - fr["total_cost"]),
            "learned_decision_mean_s": float(lr["decision_mean_s"]),
            "fixed_h25_decision_mean_s": float(fr["decision_mean_s"]),
            "delta_decision_mean_s": float(lr["decision_mean_s"] - fr["decision_mean_s"]),
            "h35_count": int(detail["h35_count"]),
            "first_h35": detail["first_h35"],
            "first_h35_step": first.get("step"),
            "first_h35_trace_index": first.get("trace_index"),
            "first_h35_leaf": first.get("leaf"),
            "first_h35_heading_error_5": first.get("heading_error_5"),
            "first_h35_abs_yaw_input": first.get("abs_yaw_input"),
            "first_h35_tracking_error": first.get("tracking_error"),
            "first_h35_remaining": first.get("remaining"),
            "leaf_counts_at_h35": detail["leaf_counts_at_h35"],
            "h35_steps": detail["h35_steps"],
            "h35_feature_stats": detail["h35_feature_stats"],
            "learned_summary_path": lr["summary_path"],
            "fixed_summary_path": fr["summary_path"],
        })

    cases_with_h35 = [r for r in per_case if r["h35_count"] > 0]
    cases_without_h35 = [r for r in per_case if r["h35_count"] == 0]
    failed_cases = [r["case"] for r in per_case if not r["learned_success"]]
    early_h35_cases = [r["case"] for r in cases_with_h35 if r["first_h35_step"] is not None and int(r["first_h35_step"]) <= 5]
    singleton_h35_cases = [r["case"] for r in cases_with_h35 if int(r["h35_count"]) == 1]
    root = policy["nodes"][0]
    left_node = policy["nodes"][1]
    h35_rule_human = (
        f"{FEATURE_NAMES[int(root['feature'])]} <= {float(root['threshold']):.12g} and "
        f"{FEATURE_NAMES[int(left_node['feature'])]} <= {float(left_node['threshold']):.12g}"
    )
    top_h35 = sorted(cases_with_h35, key=lambda r: (-r["h35_count"], r["case"]))[:12]
    instrumented_diag = None
    if INSTRUMENTED_RAW.exists():
        inst = read_json(INSTRUMENTED_RAW)
        instrumented_diag = inst.get("programmatic_diagnosis")
    per_case_csv = OUT_DIR / "per_case_h35.csv"
    fields = ["case", "learned_success", "fixed_h25_success", "learned_steps", "fixed_h25_steps", "learned_physical", "fixed_h25_physical", "delta_physical", "learned_total", "fixed_h25_total", "delta_total", "learned_decision_mean_s", "fixed_h25_decision_mean_s", "delta_decision_mean_s", "h35_count", "first_h35_step", "first_h35_trace_index", "first_h35_leaf", "first_h35_heading_error_5", "first_h35_abs_yaw_input", "first_h35_tracking_error", "first_h35_remaining", "h35_steps"]
    with per_case_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_case:
            row = {k: r.get(k) for k in fields}
            row["h35_steps"] = json.dumps(row["h35_steps"], sort_keys=True)
            w.writerow(row)

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%z")
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_H35_LEAF_VALIDATION_DIAGNOSTIC_V2_{stamp}.json"
    aggregate = {
        "h35_rule_human": h35_rule_human,
        "cases_with_h35_count": len(cases_with_h35),
        "cases_with_h35": [r["case"] for r in cases_with_h35],
        "total_h35_steps": int(total_h35_steps),
        "h35_cases_success_count": int(sum(1 for r in cases_with_h35 if r["learned_success"])),
        "h35_cases_failure_count": int(sum(1 for r in cases_with_h35 if not r["learned_success"])),
        "no_h35_cases_success_count": int(sum(1 for r in cases_without_h35 if r["learned_success"])),
        "no_h35_cases_failure_count": int(sum(1 for r in cases_without_h35 if not r["learned_success"])),
        "early_h35_cases": early_h35_cases,
        "singleton_h35_cases": singleton_h35_cases,
        "failed_cases": failed_cases,
        "delta_physical_all_stats": stats(r["delta_physical"] for r in per_case),
        "delta_physical_h35_cases_stats": stats(r["delta_physical"] for r in cases_with_h35),
        "delta_physical_no_h35_cases_stats": stats(r["delta_physical"] for r in cases_without_h35),
        "delta_physical_without_case43_stats": stats(r["delta_physical"] for r in per_case if r["case"] != 43),
        "delta_decision_mean_all_stats": stats(r["delta_decision_mean_s"] for r in per_case),
        "h35_count_stats_over_h35_cases": stats(r["h35_count"] for r in cases_with_h35),
        "first_h35_step_stats": stats(r["first_h35_step"] for r in cases_with_h35 if r["first_h35_step"] is not None),
    }
    input_hashes = {rel(EPISODE_TABLE): sha256(EPISODE_TABLE), rel(POLICY_PATH): sha256(POLICY_PATH), rel(THIS_SCRIPT): sha256(THIS_SCRIPT), **all_trace_hashes}
    if V1_FAILURE.exists():
        input_hashes[rel(V1_FAILURE)] = sha256(V1_FAILURE)
    if V1_RAW.exists():
        input_hashes[rel(V1_RAW)] = sha256(V1_RAW)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "IMPROVED_latency_tree_vehicle_h35_leaf_validation_diagnostic_v2_existing_outputs_no_sim_not_original_SAC",
        "supersedes_failed_v1": {"failure": rel(V1_FAILURE) if V1_FAILURE.exists() else None, "raw": rel(V1_RAW) if V1_RAW.exists() else None,
                                  "reason": "v1 failed summary KeyError and used H35-event ordinal rather than rollout time index when trace rows omitted step"},
        "passed": True,
        "validation_accessed_existing_outputs": True,
        "validation_bank_reopened": False,
        "validation_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "formal_scientific_evidence_created": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "inputs": {"episode_table": rel(EPISODE_TABLE), "learned_policy": rel(POLICY_PATH), "instrumented_case43_raw": rel(INSTRUMENTED_RAW)},
        "input_hashes": input_hashes,
        "policy": policy,
        "feature_names": FEATURE_NAMES,
        "aggregate": aggregate,
        "top_h35_cases": top_h35,
        "per_case": per_case,
        "per_case_by_case": {str(r["case"]): r for r in per_case},
        "instrumented_case43_diagnosis": instrumented_diag,
        "artifacts": {"raw": rel(raw_path), "summary": rel(summary_path), "per_case_csv": rel(per_case_csv), "completed": rel(OUT_DIR / "completed.json"), "backup_request": rel(backup_request)},
        "next_action_recommendation": "Version an IMPROVED safe-shortening/risk-sensitive extraction protocol that forbids H>baseline for vehicle unless independently justified, adds no-per-case-catastrophe gates, and validates on fresh non-test cases before any final test gate.",
    }
    write_json(raw_path, raw)
    write_summary(summary_path, raw)
    write_json(backup_request, {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "purpose": "backup after vehicle H35 leaf validation diagnostic v2",
                                "validation_accessed_existing_outputs": True, "validation_bank_reopened": False, "test_accessed": False,
                                "new_simulations": 0, "new_control_steps": 0,
                                "artifacts_to_cover": [rel(OUT_DIR), rel(THIS_SCRIPT), rel(V1_FAILURE) if V1_FAILURE.exists() else None],
                                "pre_completed_hashes": {rel(p): sha256(p) for p in [raw_path, summary_path, per_case_csv, THIS_SCRIPT]}})
    doc_body = f"""
## 2026-09-27 vehicle H35 leaf validation diagnostic v2

UTC: {raw['created_utc']}. Existing validation64 rollout outputs only; no new simulation/control steps/gradient steps, no validation bank reopen, sealed test closed. v2 preserves failed v1 and fixes its H35 step-index bug.

Finding: learned_s2's H35 branch is leaf 0 with rule `{h35_rule_human}`. It appeared in {aggregate['cases_with_h35_count']}/64 validation cases ({aggregate['total_h35_steps']} total H35 steps). First-H35 rollout-step stats: `{aggregate['first_h35_step_stats']}`. Case43 remains the only learned_s2 validation failure, with H35 steps `{raw['per_case_by_case']['43']['h35_steps']}`; prior deterministic replay showed that forcing H25 at the singleton H35 rescues it while forcing H35 into the constant-H25 path reproduces the failure. Excluding case43, learned_s2 vs same-seed fixed H25 physical deltas are summarized by `{aggregate['delta_physical_without_case43_stats']}`.

Decision: next IMPROVED vehicle revision should test safe-shortening/risk-sensitive extraction rather than allowing H>25 cost-seeking branches to compete as adaptive-horizon acceleration. Artifacts: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(per_case_csv)}`. Backup requested: `{rel(backup_request)}`.
"""
    docs_updated: List[str] = []
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        p = ROOT / doc
        if p.exists() and append_once(p, DOC_MARKER, doc_body):
            docs_updated.append(doc)
    raw["docs_updated"] = docs_updated
    write_json(raw_path, raw)
    write_summary(summary_path, raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [backup_request, THIS_SCRIPT]
    if V1_FAILURE.exists():
        files.append(V1_FAILURE)
    files.extend(ROOT / d for d in docs_updated)
    write_json(OUT_DIR / "completed.json", {"passed": True, "validation_accessed_existing_outputs": True, "validation_bank_reopened": False,
                                             "validation_bank_content_opened": False, "test_accessed": False,
                                             "sealed_test_bank_content_opened": False, "sealed_test_bank_hashed": False,
                                             "formal_scientific_evidence_created": False, "new_simulations": 0,
                                             "new_control_steps": 0, "new_gradient_steps": 0,
                                             "supersedes_failed_v1": raw["supersedes_failed_v1"],
                                             "cases_with_h35_count": len(cases_with_h35), "total_h35_steps": int(total_h35_steps),
                                             "first_h35_step_stats": aggregate["first_h35_step_stats"],
                                             "failed_cases": failed_cases, "case43_h35_steps": raw["per_case_by_case"]["43"]["h35_steps"],
                                             "next_action_recommendation": raw["next_action_recommendation"],
                                             "backup_request": rel(backup_request),
                                             "hashes": {rel(p): sha256(p) for p in sorted(set(files))}})
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(summary_path), "cases_with_h35": len(cases_with_h35),
                      "total_h35_steps": int(total_h35_steps), "first_h35_step_stats": aggregate["first_h35_step_stats"],
                      "failed_cases": failed_cases, "case43_h35_steps": raw["per_case_by_case"]["43"]["h35_steps"],
                      "test_accessed": False, "next_action": raw["next_action_recommendation"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            write_json(OUT_DIR / "failure.json", {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": repr(exc),
                                                   "test_accessed": False, "validation_bank_reopened": False, "new_simulations": 0,
                                                   "new_control_steps": 0, "new_gradient_steps": 0})
        except Exception:
            pass
        raise
