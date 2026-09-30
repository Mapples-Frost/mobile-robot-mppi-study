#!/usr/bin/env python3
"""v27 fixed-H12-primary postdiagnostic audit.

Zero-simulation audit of the completed v27 development confirmation.  It checks
budget/access invariants and separates two different interpretations that the
v27 summary intentionally kept scoped but did not fully spell out:

* H15-referenced tradeoff: fixed true H12 is no worse than H15 under the row-wise
  relative physical tolerance and saves decision/solver time.
* absolute deployability/success: rows where both H12 and H15 fail at the step
  cap are not successful control episodes and must not be treated as evidence
  that a deployable fixed-H12 controller is safe on the scenario distribution.

No validation64 or sealed test content is opened; no training/refit/simulation is
performed.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic"
STAMP = "20260930T0300Z"
MARKER = f"vehicle-fixed-h12-primary-v27-postdiagnostic-{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/raw.json"
DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/completed.json"
SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/summary.md"
V26B_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_corrected_fixed_H12_primary_confirmation_amendment_20260930T0235Z.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0300_after_v27_postdiagnostic.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_POSTDIAGNOSTIC_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MIN_SAVE = 0.05


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
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe(summary: Mapping[str, Any]) -> bool:
    if "safe_all" in summary:
        return bool(summary.get("safe_all"))
    return bool(summary.get("success_all")) and not bool(summary.get("constraint_any")) and si(summary.get("solver_failure_steps_sum"), 0) == 0 and si(summary.get("initial_failed_steps_sum"), 0) == 0 and si(summary.get("final_failed_steps_sum"), 0) == 0


def pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "NA"


def aggregate(rows: Sequence[Mapping[str, Any]], choice_map: Mapping[str, int], overhead_per_state_s: float = 0.0) -> Dict[str, Any]:
    h15_phys = math.fsum(sf((r.get("h15") or {}).get("physical")) for r in rows)
    h15_dec = math.fsum(sf((r.get("h15") or {}).get("decision_sum_s")) for r in rows)
    h15_sol = math.fsum(sf((r.get("h15") or {}).get("solver_sum_s")) for r in rows)
    tol_sum = math.fsum(sf(r.get("row_tolerance_vs_H15"), 2.0) for r in rows)
    phys = dec = sol = 0.0
    unsafe = []
    bad = []
    counts: Dict[str, int] = {}
    for r in rows:
        sid = str(r.get("base_state_id"))
        h = int(choice_map.get(sid, 15))
        counts[str(h)] = counts.get(str(h), 0) + 1
        hs = r.get("h12") if h == 12 else r.get("h15")
        assert isinstance(hs, Mapping)
        phys += sf(hs.get("physical"))
        dec += sf(hs.get("decision_sum_s"))
        sol += sf(hs.get("solver_sum_s"))
        if not safe(hs):
            unsafe.append({"base_state_id": sid, "source_candidate_index": si(r.get("source_candidate_index"), -1), "selected_h": h, "steps": hs.get("steps"), "success_all": hs.get("success_all"), "safe_all": hs.get("safe_all"), "physical": hs.get("physical")})
        if h == 12 and bool(r.get("h12_catastrophic_vs_h15")):
            bad.append({"base_state_id": sid, "source_candidate_index": si(r.get("source_candidate_index"), -1), "phys_delta_h12_minus_h15": sf(r.get("physical_delta_h12_minus_h15"))})
    dec += overhead_per_state_s * len(rows)
    return {
        "rows": len(rows),
        "chosen_counts": counts,
        "physical_sum": phys,
        "fixed_H15_physical_sum": h15_phys,
        "physical_delta_vs_H15": phys - h15_phys,
        "physical_tolerance_sum": tol_sum,
        "physical_gate_vs_H15": (phys - h15_phys) <= tol_sum,
        "decision_sum_s": dec,
        "fixed_H15_decision_sum_s": h15_dec,
        "decision_relative_saving_vs_H15": (h15_dec - dec) / h15_dec if h15_dec > 0 else 0.0,
        "solver_sum_s": sol,
        "fixed_H15_solver_sum_s": h15_sol,
        "solver_relative_saving_vs_H15": (h15_sol - sol) / h15_sol if h15_sol > 0 else 0.0,
        "catastrophic_false_positive_count": len(bad),
        "absolute_unsafe_count": len(unsafe),
        "absolute_unsafe_rows": unsafe,
        "success_sensitive_pass5_zero_cat_physical": bool((h15_dec - dec) / h15_dec >= MIN_SAVE and (phys - h15_phys) <= tol_sum and not bad and not unsafe) if h15_dec > 0 else False,
        "h15_referenced_pass5_zero_cat_physical": bool((h15_dec - dec) / h15_dec >= MIN_SAVE and (phys - h15_phys) <= tol_sum and not bad) if h15_dec > 0 else False,
    }


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_json(RAW)
    done = read_json(DONE)
    v26b = read_json(V26B_PROTOCOL)
    if raw.get("validation64_bank_opened") is True or raw.get("sealed_test_accessed") is True or done.get("validation64_bank_opened") is True or done.get("sealed_test_accessed") is True:
        raise RuntimeError("v27 access flags unexpectedly show validation/test access")
    rows = list((((raw.get("analysis") or {}).get("label_analysis") or {}).get("state_rows")) or [])
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 v27 state rows, got {len(rows)}")
    budget = raw.get("budget_actual") or {}
    if budget.get("episodes") != 60 or budget.get("control_steps", 999999) > 9000:
        raise RuntimeError("v27 budget mismatch")
    h12_unsafe = [r for r in rows if not safe(r.get("h12") or {})]
    h15_unsafe = [r for r in rows if not safe(r.get("h15") or {})]
    both_unsafe = [r for r in rows if not safe(r.get("h12") or {}) and not safe(r.get("h15") or {})]
    h12_only_unsafe = [r for r in rows if not safe(r.get("h12") or {}) and safe(r.get("h15") or {})]
    h15_only_unsafe = [r for r in rows if safe(r.get("h12") or {}) and not safe(r.get("h15") or {})]
    safe_both = [r for r in rows if safe(r.get("h12") or {}) and safe(r.get("h15") or {})]
    choice_map = {str(c.get("base_state_id")): int(c.get("selected_h_preoutcome", 15)) for c in ((raw.get("preoutcome_selector_choices") or {}).get("choices") or [])}
    overhead = sf(((raw.get("preoutcome_selector_choices") or {}).get("overhead_summary") or {}).get("mean_s"))
    fixed_h12 = aggregate(rows, {str(r.get("base_state_id")): 12 for r in rows})
    fixed_h15 = aggregate(rows, {str(r.get("base_state_id")): 15 for r in rows})
    selector = aggregate(rows, choice_map, overhead_per_state_s=overhead)
    fixed_h12_safe_both_only = aggregate(safe_both, {str(r.get("base_state_id")): 12 for r in safe_both}) if safe_both else {}
    fresh_before = (v26b.get("corrected_before_evidence") or {}).get("fresh_v21_v23_v25") or {}
    all_before = (v26b.get("corrected_before_evidence") or {}).get("all_opened_v19_v21_v23_v25") or {}
    combined_fresh = {
        "rows": si(fresh_before.get("rows")) + len(rows),
        "h12_bad_relative": si(fresh_before.get("h12_bad")) + si(((raw.get("analysis") or {}).get("label_analysis") or {}).get("aggregate", {}).get("catastrophic_H12_count")),
        "h12_beneficial": si(fresh_before.get("h12_beneficial")) + si(((raw.get("analysis") or {}).get("label_analysis") or {}).get("aggregate", {}).get("beneficial_H12_count")),
        "h15_or_both_unsafe_known_from_v27": len(h15_unsafe),
        "h12_decision_sum_s": sf(fresh_before.get("h12_decision_sum_s")) + fixed_h12["decision_sum_s"],
        "h15_decision_sum_s": sf(fresh_before.get("h15_decision_sum_s")) + fixed_h15["decision_sum_s"],
        "h12_solver_sum_s": sf(fresh_before.get("h12_solver_sum_s")) + fixed_h12["solver_sum_s"],
        "h15_solver_sum_s": sf(fresh_before.get("h15_solver_sum_s")) + fixed_h15["solver_sum_s"],
        "physical_delta_H12_minus_H15": sf(fresh_before.get("physical_delta_H12_minus_H15")) + fixed_h12["physical_delta_vs_H15"],
    }
    combined_fresh["decision_saving_vs_H15"] = (combined_fresh["h15_decision_sum_s"] - combined_fresh["h12_decision_sum_s"]) / combined_fresh["h15_decision_sum_s"]
    combined_fresh["solver_saving_vs_H15"] = (combined_fresh["h15_solver_sum_s"] - combined_fresh["h12_solver_sum_s"]) / combined_fresh["h15_solver_sum_s"]
    combined_all = {
        "rows": si(all_before.get("rows")) + len(rows),
        "h12_bad_relative": si(all_before.get("h12_bad")) + si(((raw.get("analysis") or {}).get("label_analysis") or {}).get("aggregate", {}).get("catastrophic_H12_count")),
        "h12_decision_sum_s": sf(all_before.get("h12_decision_sum_s")) + fixed_h12["decision_sum_s"],
        "h15_decision_sum_s": sf(all_before.get("h15_decision_sum_s")) + fixed_h15["decision_sum_s"],
        "physical_delta_H12_minus_H15": sf(all_before.get("physical_delta_H12_minus_H15")) + fixed_h12["physical_delta_vs_H15"],
    }
    combined_all["decision_saving_vs_H15"] = (combined_all["h15_decision_sum_s"] - combined_all["h12_decision_sum_s"]) / combined_all["h15_decision_sum_s"]
    decision = (
        "v27 strengthens the comparison-design conclusion: on fresh stress-pool rows, true fixed H12 is the dominant H12/H15 relative-compute baseline and the current adaptive selector is too conservative. "
        "However two source242 rows are absolute failures for both H12 and H15, so the v27 pass is H15-referenced rather than a deployable success/safety pass. "
        "After external backup, do not validate this adaptive H12/H15 selector; freeze a success-aware scenario/comparison audit/protocol that separates feasible states, both-fail hard cases, and the opened v19 H12-negative cluster before deciding on scenario redesign or terminal-risk/value training."
    )
    result = {
        "created_utc": now_utc().isoformat(),
        "classification": "development_analysis_only_v27_postdiagnostic_no_sim_no_validation_no_test",
        "source_v27_raw": rel(RAW),
        "source_v27_completed": rel(DONE),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_training_or_gradient_steps": 0,
        "budget_checks": {"v27_episodes": budget.get("episodes"), "v27_control_steps": budget.get("control_steps"), "within_declared_cap": budget.get("episodes") == 60 and si(budget.get("control_steps"), 999999) <= 9000},
        "v27_absolute_safety_counts": {
            "rows": len(rows),
            "h12_unsafe_count": len(h12_unsafe),
            "h15_unsafe_count": len(h15_unsafe),
            "both_h12_h15_unsafe_count": len(both_unsafe),
            "h12_only_unsafe_count": len(h12_only_unsafe),
            "h15_only_unsafe_count": len(h15_only_unsafe),
            "both_unsafe_rows": [{"base_state_id": str(r.get("base_state_id")), "source_candidate_index": si(r.get("source_candidate_index"), -1), "h12_steps": (r.get("h12") or {}).get("steps"), "h15_steps": (r.get("h15") or {}).get("steps"), "h12_physical": (r.get("h12") or {}).get("physical"), "h15_physical": (r.get("h15") or {}).get("physical")} for r in both_unsafe],
        },
        "v27_relative_policy_audit": {"fixed_H12": fixed_h12, "fixed_H15": fixed_h15, "selector_preoutcome": selector, "fixed_H12_on_safe_both_rows_only_diagnostic_not_a_gate": fixed_h12_safe_both_only},
        "combined_corrected_development_context": {"fresh_v21_v23_v25_v27": combined_fresh, "all_opened_v19_v21_v23_v25_v27": combined_all},
        "decision": decision,
        "input_hashes": {rel(p): sha256(p) for p in [RAW, DONE, SUMMARY, V26B_PROTOCOL, Path(__file__).resolve()] if p.exists()},
    }
    write_json(RUN_DIR / "raw.json", result)
    lines = [
        "# v27 fixed-H12-primary postdiagnostic",
        "",
        f"UTC: `{result['created_utc']}`. Analysis-only audit of v27; no new simulation, no validation64, no sealed test, no training/refit.",
        "",
        "## Key check",
        "",
        f"- v27 budget check: `{result['budget_checks']['v27_episodes']}` episodes, `{result['budget_checks']['v27_control_steps']}` control steps, within cap `{result['budget_checks']['within_declared_cap']}`.",
        f"- H15-referenced fixed-H12 result: decision saving `{pct(fixed_h12['decision_relative_saving_vs_H15'])}`, solver saving `{pct(fixed_h12['solver_relative_saving_vs_H15'])}`, bad `{fixed_h12['catastrophic_false_positive_count']}`, physical gate `{fixed_h12['physical_gate_vs_H15']}`, relative pass5 `{fixed_h12['h15_referenced_pass5_zero_cat_physical']}`.",
        f"- Success-sensitive fixed-H12 result: absolute unsafe rows `{fixed_h12['absolute_unsafe_count']}`, success-sensitive pass5 `{fixed_h12['success_sensitive_pass5_zero_cat_physical']}`.",
        f"- Selector success-sensitive result: H counts `{selector['chosen_counts']}`, decision saving `{pct(selector['decision_relative_saving_vs_H15'])}`, unsafe rows `{selector['absolute_unsafe_count']}`, success-sensitive pass5 `{selector['success_sensitive_pass5_zero_cat_physical']}`.",
        f"- Both H12 and H15 unsafe rows: `{len(both_unsafe)}`; source indices `{sorted(set(si(r.get('source_candidate_index'), -1) for r in both_unsafe))}`.",
        f"- Diagnostic safe-both subset (not a new gate): fixed H12 saving `{pct(fixed_h12_safe_both_only.get('decision_relative_saving_vs_H15'))}` over `{fixed_h12_safe_both_only.get('rows')}` rows, unsafe rows `{fixed_h12_safe_both_only.get('absolute_unsafe_count')}`.",
        "",
        "## Combined context",
        "",
        f"- Fresh v21+v23+v25+v27 rows: `{combined_fresh['rows']}`, relative H12-bad `{combined_fresh['h12_bad_relative']}`, H12-beneficial `{combined_fresh['h12_beneficial']}`, decision saving `{pct(combined_fresh['decision_saving_vs_H15'])}`, known v27 H15/both-unsafe rows `{combined_fresh['h15_or_both_unsafe_known_from_v27']}`.",
        f"- All opened v19+v21+v23+v25+v27 rows: `{combined_all['rows']}`, relative H12-bad `{combined_all['h12_bad_relative']}`, decision saving `{pct(combined_all['decision_saving_vs_H15'])}`, physical delta `{combined_all['physical_delta_H12_minus_H15']:.6g}`.",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Limits",
        "",
        "This audit does not change v27's frozen result or delete hard rows. It clarifies that relative fixed-H12-vs-H15 pass is not equivalent to absolute success/safety because source242 failed under both horizons. Future comparisons must report both semantics.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v27_postdiagnostic",
        "created_utc": result["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "new v27 postdiagnostic analysis, docs/state/registry/response-log updates, and backup-proof materialization",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", "research_artifacts/aws_backup_proofs/backup_proof_20260930T023554_from_supervisor_context_after_v27_source_preflight.json"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
    })
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v27 fixed-H12-primary postdiagnostic

UTC: {result['created_utc']}. Analysis-only audit of v27 completed; no simulations/control steps/training, no validation64-bank access, no sealed-test access. It preserves v27's relative result (fixed H12 decision saving {pct(fixed_h12['decision_relative_saving_vs_H15'])}, relative bad={fixed_h12['catastrophic_false_positive_count']}, H15-referenced pass5={fixed_h12['h15_referenced_pass5_zero_cat_physical']}) but clarifies failure accounting: absolute unsafe rows for fixed H12={fixed_h12['absolute_unsafe_count']}, fixed H15={fixed_h15['absolute_unsafe_count']}, both-fail rows={len(both_unsafe)} from source_candidate_index {sorted(set(si(r.get('source_candidate_index'), -1) for r in both_unsafe))}; success-sensitive fixed-H12 pass5={fixed_h12['success_sensitive_pass5_zero_cat_physical']}. Combined fresh v21+v23+v25+v27 relative rows={combined_fresh['rows']} with H12_bad={combined_fresh['h12_bad_relative']} and decision saving={pct(combined_fresh['decision_saving_vs_H15'])}, but this remains stress-pool development evidence. Decision: {decision} Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. Backup required: `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
## Postdiagnostic clarification after v27 fixed-H12-primary confirmation

Updated by GPT-5.5 executor at `{result['created_utc']}`. This is an analysis-only clarification of v27; no validation64/sealed-test access and no new simulation/training.

| linked recommendation(s) | disposition after postdiagnostic | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; sharpened | v27 fixed H12 has H15-referenced pass5=True (saving {pct(fixed_h12['decision_relative_saving_vs_H15'])}, bad=0) but success-sensitive pass5=False because fixed H12 has {fixed_h12['absolute_unsafe_count']} absolute unsafe rows; fixed H15 also has {fixed_h15['absolute_unsafe_count']} unsafe rows and both-fail rows come from source_candidate_index {sorted(set(si(r.get('source_candidate_index'), -1) for r in both_unsafe))}. | Future comparator tables must separate relative H12-vs-H15 cost/speed from absolute success/safety. Do not describe v27 as deployable success on all selected stress rows. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | Combined fresh v21+v23+v25+v27 has {combined_fresh['rows']} relative rows, H12_bad={combined_fresh['h12_bad_relative']}, decision saving={pct(combined_fresh['decision_saving_vs_H15'])}, but includes v27 hard both-fail rows and remains stress-pool development evidence. | No population or validation claim; use only to choose the next development intervention. |
| `A11_training_failure_modes_need_separation` | accepted; training still deferred but not indefinitely | v27 selector is too conservative (H counts {selector['chosen_counts']}, save {pct(selector['decision_relative_saving_vs_H15'])}) and fixed H12 dominates relative to H15 on feasible fresh rows; the current bottleneck is comparison/scenario opportunity and failure accounting rather than missing a richer selector on these rows. | After verified backup, freeze a success-aware scenario/comparison audit/protocol that separates feasible states, both-fail hard cases, and v19 H12-negative cases. Pivot to terminal-risk/value refit only if that protocol shows adaptive opportunity not captured by fixed H12. |
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | new; accepted | Source242 contributes two v27 branch states where both H12 and H15 run to 150 steps with physical costs >20k; relative physical delta is 0, so a purely H15-referenced gate can pass while absolute success fails. | Add success-sensitive pass/failure accounting to all future H12-primary comparison summaries and reviewer handoff. |
| `A12_registry_backup_schema_contract` | accepted; active | This postdiagnostic wrote `{rel(BACKUP_REQUEST)}` and updated docs/state/registry/response log. | Require verified external backup covering v27 and this postdiagnostic before any further unique science. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([result["created_utc"], NAME, result["classification"], "deterministic_existing_v27_rows", "existing_v27_development_outputs_only_no_validation64_no_test", 0, 0, 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v27 postdiagnostic\n\nUTC: {result['created_utc']}\n\nDecision: {decision}\n\nKey audit: v27 fixed H12 relative pass5={fixed_h12['h15_referenced_pass5_zero_cat_physical']} with {pct(fixed_h12['decision_relative_saving_vs_H15'])} decision saving and bad={fixed_h12['catastrophic_false_positive_count']}, but success-sensitive pass5={fixed_h12['success_sensitive_pass5_zero_cat_physical']} because both H12/H15 are unsafe for {len(both_unsafe)} source242 states. Selector success-sensitive pass5={selector['success_sensitive_pass5_zero_cat_physical']}, H counts={selector['chosen_counts']}, saving={pct(selector['decision_relative_saving_vs_H15'])}.\n\nNext after verified external backup: freeze a success-aware scenario/comparison audit/protocol rather than validating the current adaptive selector or running another unchanged label-density sweep.\n\nBackup request: {rel(BACKUP_REQUEST)}\n", encoding="utf-8")
    completed = {
        "status": "complete",
        "passed": True,
        "created_utc": result["created_utc"],
        "classification": result["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "headline": {
            "v27_fixed_H12_relative_pass5": fixed_h12["h15_referenced_pass5_zero_cat_physical"],
            "v27_fixed_H12_success_sensitive_pass5": fixed_h12["success_sensitive_pass5_zero_cat_physical"],
            "v27_fixed_H12_decision_saving": fixed_h12["decision_relative_saving_vs_H15"],
            "v27_both_fail_count": len(both_unsafe),
            "v27_both_fail_source_indices": sorted(set(si(r.get("source_candidate_index"), -1) for r in both_unsafe)),
            "combined_fresh_rows": combined_fresh["rows"],
            "combined_fresh_h12_bad_relative": combined_fresh["h12_bad_relative"],
            "combined_fresh_decision_saving": combined_fresh["decision_saving_vs_H15"],
            "decision": decision,
        },
        "backup_request": rel(BACKUP_REQUEST),
        "hashes": {},
    }
    files = [Path(__file__).resolve(), RAW, DONE, SUMMARY, V26B_PROTOCOL, RUN_DIR / "raw.json", RUN_DIR / "summary.md", STATE, BACKUP_REQUEST, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
    completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
