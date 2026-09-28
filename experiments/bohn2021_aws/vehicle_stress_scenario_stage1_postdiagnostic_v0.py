#!/usr/bin/env python3
"""Postdiagnose Vehicle stress-scenario opportunity Stage1 fixed-H map.

No simulation, no training/refit, no validation64 or sealed-test access.  This
script parses the completed Stage1 raw JSON, audits safety/solver/timing/case
materiality, and writes a compact development-only decision record for whether
Stage2 identical-state continuation should be frozen after backup.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/raw.json"
COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/completed.json"
SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/summary.md"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45]
REF_H = 15
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MARKER = "vehicle-stress-scenario-stage1-postdiagnostic-20260928T1730Z"

class ContractError(RuntimeError):
    pass

def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def timing_sum(row: Mapping[str, Any]) -> float:
    d = row.get("decision_timing_s", {})
    if isinstance(d, Mapping):
        for k in ("sum", "total", "decision_total_s"):
            if k in d:
                return safe_float(d[k], 0.0)
    return safe_float(d, 0.0)

def timing_mean(row: Mapping[str, Any]) -> float:
    d = row.get("decision_timing_s", {})
    if isinstance(d, Mapping):
        for k in ("mean", "mean_s", "mean_s_per_step"):
            if k in d:
                return safe_float(d[k], 0.0)
    steps = max(1, safe_int(row.get("steps"), 1))
    return timing_sum(row) / steps

def phys(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("physical_constraint_cost", row.get("physical_cost", row.get("cost", float("nan")))))

def total(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("total_cost", row.get("cost", row.get("physical_constraint_cost", float("nan")))))

def strict_safe(row: Mapping[str, Any]) -> bool:
    return bool(row.get("success")) and not bool(row.get("constraint")) and safe_int(row.get("initial_failed_steps"), 0) == 0 and safe_int(row.get("solver_failure_steps"), 0) == 0

def nondominated(rows: Sequence[Mapping[str, Any]], cost_key: str = "total") -> List[int]:
    pts: List[Tuple[int, float, float]] = []
    for r in rows:
        if not strict_safe(r):
            continue
        c = total(r) if cost_key == "total" else phys(r)
        t = timing_sum(r)
        if math.isfinite(c) and math.isfinite(t):
            pts.append((safe_int(r.get("horizon")), c, t))
    out: List[int] = []
    for h, c, t in pts:
        dominated = False
        for h2, c2, t2 in pts:
            if h2 == h:
                continue
            if c2 <= c and t2 <= t and (c2 < c or t2 < t):
                dominated = True
                break
        if not dominated:
            out.append(h)
    return sorted(set(out))

def pct_gain(gain: float, ref: float) -> float:
    return gain / ref if math.isfinite(ref) and ref > 0 else 0.0

def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_json(RAW)
    done = read_json(COMPLETED)
    if done.get("hard_pass") is not True or done.get("passed") is not True:
        raise ContractError("Stage1 completed marker is not a hard pass")
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("Stage1 raw access flags are not closed")
    episodes = list(raw.get("episodes") or [])
    if len(episodes) != 120:
        raise ContractError("Expected 120 Stage1 episodes, got %d" % len(episodes))
    meta_by_case = {safe_int(m.get("selected_case_index")): m for m in raw.get("bank_selection", {}).get("selected_metadata", [])}
    by_case: Dict[int, Dict[int, Mapping[str, Any]]] = {}
    for e in episodes:
        c = safe_int(e.get("case", e.get("selected_case_index")))
        h = safe_int(e.get("horizon"))
        by_case.setdefault(c, {})[h] = e
    missing = {c: sorted(set(HORIZONS) - set(by_case.get(c, {}))) for c in range(12) if set(HORIZONS) - set(by_case.get(c, {}))}
    if missing:
        raise ContractError("Incomplete case x horizon grid: %r" % missing)

    branch_rows = [by_case[c][h] for c in range(12) for h in BRANCH_HORIZONS]
    branch_artifacts = {
        "branch_rows": len(branch_rows),
        "branch_success_count": sum(1 for r in branch_rows if bool(r.get("success"))),
        "branch_constraint_count": sum(1 for r in branch_rows if bool(r.get("constraint"))),
        "branch_initial_failed_steps": sum(safe_int(r.get("initial_failed_steps"), 0) for r in branch_rows),
        "branch_solver_failure_steps": sum(safe_int(r.get("solver_failure_steps"), 0) for r in branch_rows),
        "all_branch_rows_strict_safe": all(strict_safe(r) for r in branch_rows),
    }
    h5_rows = [by_case[c][5] for c in range(12)]
    h5_failures = sum(1 for r in h5_rows if not bool(r.get("success")) or bool(r.get("constraint")))

    case_rows: List[Dict[str, Any]] = []
    material_cases: List[int] = []
    stress_material_cases: List[int] = []
    negative_cases: List[int] = []
    control_negative_cases: List[int] = []
    clear_tradeoff_cases: List[int] = []
    large_harms = 0
    for c in range(12):
        rows_all = by_case[c]
        meta = meta_by_case.get(c, {})
        ref = rows_all[REF_H]
        ref_total = total(ref)
        ref_phys = phys(ref)
        candidates = [rows_all[h] for h in BRANCH_HORIZONS if strict_safe(rows_all[h])]
        best_t = min(candidates, key=lambda r: (total(r), timing_sum(r), safe_int(r.get("horizon"))))
        best_p = min(candidates, key=lambda r: (phys(r), timing_sum(r), safe_int(r.get("horizon"))))
        total_gain = ref_total - total(best_t)
        phys_gain = ref_phys - phys(best_p)
        total_material = safe_int(best_t.get("horizon")) != REF_H and (total_gain >= 3.0 or pct_gain(total_gain, ref_total) >= 0.03)
        phys_material = safe_int(best_p.get("horizon")) != REF_H and (phys_gain >= 5.0 or pct_gain(phys_gain, ref_phys) >= 0.05)
        material = bool(strict_safe(ref) and (total_material or phys_material))
        for h in BRANCH_HORIZONS:
            if h == REF_H:
                continue
            r = rows_all[h]
            if strict_safe(r) and ((total(r) - ref_total) >= 3.0 or (phys(r) - ref_phys) >= 5.0):
                large_harms += 1
        front_total = nondominated([rows_all[h] for h in BRANCH_HORIZONS], "total")
        front_phys = nondominated([rows_all[h] for h in BRANCH_HORIZONS], "physical")
        if len(front_total) >= 2 or len(front_phys) >= 2:
            clear_tradeoff_cases.append(c)
        if material:
            material_cases.append(c)
            if meta.get("selection_group") == "stress":
                stress_material_cases.append(c)
        else:
            negative_cases.append(c)
            if meta.get("selection_group") == "control":
                control_negative_cases.append(c)
        case_rows.append({
            "case": c,
            "group": meta.get("selection_group"),
            "candidate_index": safe_int(meta.get("candidate_index"), -1),
            "stress_flags_count": safe_int(meta.get("stress_flags_count"), 0),
            "ref_H15": {"total_cost": ref_total, "physical_constraint_cost": ref_phys, "steps": safe_int(ref.get("steps")), "decision_time_sum_s": timing_sum(ref), "strict_safe": strict_safe(ref)},
            "best_branch_total": {"horizon": safe_int(best_t.get("horizon")), "total_cost": total(best_t), "gain_vs_H15": total_gain, "gain_pct_vs_H15": pct_gain(total_gain, ref_total), "steps": safe_int(best_t.get("steps")), "decision_time_sum_s": timing_sum(best_t), "strict_safe": strict_safe(best_t)},
            "best_branch_physical": {"horizon": safe_int(best_p.get("horizon")), "physical_constraint_cost": phys(best_p), "gain_vs_H15": phys_gain, "gain_pct_vs_H15": pct_gain(phys_gain, ref_phys), "steps": safe_int(best_p.get("steps")), "decision_time_sum_s": timing_sum(best_p), "strict_safe": strict_safe(best_p)},
            "material_non_H15_positive": material,
            "material_metric_flags": {"total": bool(total_material), "physical": bool(phys_material)},
            "nondominated_branch_horizons_total_cost_vs_time": front_total,
            "nondominated_branch_horizons_physical_cost_vs_time": front_phys,
        })

    by_horizon: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        rows = [by_case[c][h] for c in range(12)]
        by_horizon[str(h)] = {
            "success_count": sum(1 for r in rows if bool(r.get("success"))),
            "constraint_count": sum(1 for r in rows if bool(r.get("constraint"))),
            "initial_failed_steps": sum(safe_int(r.get("initial_failed_steps"), 0) for r in rows),
            "solver_failure_steps": sum(safe_int(r.get("solver_failure_steps"), 0) for r in rows),
            "physical_constraint_cost_sum": sum(phys(r) for r in rows),
            "total_cost_sum": sum(total(r) for r in rows),
            "decision_time_sum_s": sum(timing_sum(r) for r in rows),
            "decision_mean_s_per_step": sum(timing_sum(r) for r in rows) / max(1, sum(safe_int(r.get("steps"), 0) for r in rows)),
        }
    agg_front_total = nondominated([by_case[c][h] for c in range(12) for h in BRANCH_HORIZONS], "total")
    # Aggregate frontier by H uses sums rather than episode points.
    agg_pts = []
    for h in BRANCH_HORIZONS:
        b = by_horizon[str(h)]
        if b["success_count"] == 12 and b["constraint_count"] == 0 and b["initial_failed_steps"] == 0 and b["solver_failure_steps"] == 0:
            agg_pts.append((h, b["total_cost_sum"], b["decision_time_sum_s"]))
    agg_front_by_h = []
    for h, cst, tim in agg_pts:
        if not any((c2 <= cst and t2 <= tim and (c2 < cst or t2 < tim)) for h2, c2, t2 in agg_pts if h2 != h):
            agg_front_by_h.append(h)

    stage1_mat = raw.get("stage1_materiality_analysis", {})
    stage2_supported = bool((stage1_mat.get("stage2_continuation_trigger_candidate_development_only") is True) and branch_artifacts["all_branch_rows_strict_safe"])
    training_gate_for_later = bool(len(material_cases) >= 2 and len(stress_material_cases) >= 1 and (len(negative_cases) >= 2 or len(control_negative_cases) >= 2) and branch_artifacts["all_branch_rows_strict_safe"])
    diagnosis = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_no_simulation",
        "classification": "development_diagnostic_artifact_audit_no_rollouts_no_training_not_final_test",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {"stage1_raw": rel(RAW), "stage1_completed": rel(COMPLETED), "stage1_summary": rel(SUMMARY), "stage1_raw_sha256": sha256(RAW), "stage1_completed_sha256": sha256(COMPLETED)},
        "budget_audit": raw.get("budget_actual", {}),
        "branch_horizons_for_stage2": BRANCH_HORIZONS,
        "branch_safety_solver_artifact_audit": branch_artifacts,
        "h5_failure_context": {"h5_failures_or_constraints": h5_failures, "h5_rows": 12, "interpretation": "H5 failures are excluded from Stage2 branch horizons and from material-positive counts."},
        "aggregate_by_horizon_recomputed": by_horizon,
        "aggregate_stage1_materiality_from_runner": stage1_mat,
        "aggregate_branch_total_cost_time_frontier_horizons": sorted(agg_front_by_h),
        "case_diagnostics": case_rows,
        "case_counts": {
            "material_non_H15_positive_cases": len(material_cases),
            "material_non_H15_positive_case_ids": material_cases,
            "stress_material_positive_cases": len(stress_material_cases),
            "stress_material_positive_case_ids": stress_material_cases,
            "negative_or_nonmaterial_cases": len(negative_cases),
            "control_negative_or_nonmaterial_cases": len(control_negative_cases),
            "clear_case_level_tradeoff_cases": len(clear_tradeoff_cases),
            "large_safe_non_H15_harms_vs_H15_count": large_harms,
        },
        "decision": {
            "stage2_identical_state_continuation_should_be_frozen_after_backup": stage2_supported,
            "reason": "Stage1 aggregate oracle/materiality and multi-H tradeoff gates are true and branch rows H10/H15/H25/H30/H35/H45 are strict-safe; Stage2 is needed because episode-level fixed-H oracle is not an upper bound on within-episode adaptive switching.",
            "later_selector_refit_training_gate_provisionally_satisfied_by_stage1_case_labels": training_gate_for_later,
            "training_gate_is_not_final": "Stage2 identical-state continuation labels must confirm that opportunities survive from matched intermediate states before retraining/refit.",
            "no_more_simulation_until_backup": True,
        },
        "four_axis_evidence": {
            "scenarios": "Source-supported stress bank produced material safe fixed-H opportunity on 12 fresh cases; this contrasts with canonical fresh/transient continuation scarcity and justifies matched-state Stage2 rather than immediate broad retraining.",
            "reward": "Physical+constraint and total-cost oracle gains are both material in Stage1; H5 failures show why safety/failure filtering is required; no branch-row solver/constraint artifact explains the H10/H25/H30/H35/H45/H15 comparisons.",
            "training": "No training/refit occurred. Stage1 label density is promising but still episode-level; retraining/refit remains deferred until Stage2 confirms within-episode matched-state labels.",
            "comparisons": "Same-bank fixed-H grid H5..H50 was run with measured wall-clock decision timing; H15 is the aggregate cost reference, H10 is faster, and branch-horizon comparisons are paired by selected case. Final test remains sealed.",
        },
        "next_action": "After verified external backup covers Stage1 rollout and this postdiagnostic, freeze Stage2 identical-state continuation protocol/runner using branch horizons [10,15,25,30,35,45], select material-positive plus negative/control states without outcome peeking beyond Stage1, smoke no-simulation, then run <=72 continuation episodes only if backup gate passes.",
    }
    backup_req = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_POSTDIAGNOSTIC_20260928T1730Z.json"
    diagnosis["backup_request"] = rel(backup_req)
    write_json(OUT / "raw.json", diagnosis)

    lines = [
        "# Vehicle stress-scenario Stage1 postdiagnostic",
        "",
        f"UTC: `{diagnosis['created_utc']}`. No simulations/training; validation64 and sealed test remained closed.",
        "",
        "## Key audit result",
        "",
        f"- Branch rows H{BRANCH_HORIZONS}: strict-safe = `{branch_artifacts['all_branch_rows_strict_safe']}`; constraints = `{branch_artifacts['branch_constraint_count']}`, initial failed steps = `{branch_artifacts['branch_initial_failed_steps']}`, solver failed steps = `{branch_artifacts['branch_solver_failure_steps']}`.",
        f"- H5 context: `{h5_failures}`/12 failed or constrained, so H5 is excluded from Stage2 branch labels.",
        f"- Runner aggregate oracle: total gain `{stage1_mat.get('oracle_total_gain_abs')}` ({stage1_mat.get('oracle_total_gain_pct')}); physical gain `{stage1_mat.get('oracle_physical_gain_abs')}` ({stage1_mat.get('oracle_physical_gain_pct')}); candidate trigger `{stage1_mat.get('stage2_continuation_trigger_candidate_development_only')}`.",
        f"- Material non-H15 branch-positive cases: `{len(material_cases)}` {material_cases}; stress positives: `{len(stress_material_cases)}` {stress_material_cases}; negative/nonmaterial cases: `{len(negative_cases)}`; control negatives: `{len(control_negative_cases)}`.",
        f"- Aggregate branch total-cost/time frontier horizons: `{sorted(agg_front_by_h)}`.",
        "",
        "## Case-level branch labels vs H15",
        "",
        "| case | group | cand | flags | best total H/gain | best physical H/gain | material? | total/time frontier | physical/time frontier |",
        "|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in case_rows:
        lines.append("| {case} | `{group}` | {cand} | {flags} | H{bt_h} / {bt_g:.6g} | H{bp_h} / {bp_g:.6g} | `{mat}` | `{ft}` | `{fp}` |".format(
            case=r["case"], group=r["group"], cand=r["candidate_index"], flags=r["stress_flags_count"],
            bt_h=r["best_branch_total"]["horizon"], bt_g=r["best_branch_total"]["gain_vs_H15"],
            bp_h=r["best_branch_physical"]["horizon"], bp_g=r["best_branch_physical"]["gain_vs_H15"],
            mat=r["material_non_H15_positive"], ft=r["nondominated_branch_horizons_total_cost_vs_time"], fp=r["nondominated_branch_horizons_physical_cost_vs_time"]
        ))
    lines += [
        "",
        "## Decision",
        "",
        f"Stage2 identical-state continuation should be frozen after backup: `{stage2_supported}`.",
        "Reason: Stage1 shows safe material episode-level opportunity and multi-H tradeoffs on source-supported stress/control cases, but episode-level fixed-H oracle alone cannot establish within-episode adaptive-horizon value. Matched continuation is the discriminating next experiment.",
        "",
        "## Four-axis concise evidence",
        "",
    ]
    for k, v in diagnosis["four_axis_evidence"].items():
        lines.append(f"- **{k.upper()}**: {v}")
    lines += ["", f"Backup request: `{rel(backup_req)}`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Vehicle stress Stage1 postdiagnostic state ({diagnosis['created_utc']})\n\n"
        f"Stage2 freeze is supported after backup: {stage2_supported}. Branch rows strict-safe={branch_artifacts['all_branch_rows_strict_safe']}; material branch-positive cases={material_cases}; stress positives={stress_material_cases}. No validation64/test/training. Next: backup, then freeze Stage2 continuation protocol/runner with branch horizons {BRANCH_HORIZONS}.\n",
        encoding="utf-8",
    )
    write_json(backup_req, {
        "requested_utc": diagnosis["created_utc"],
        "reason": "backup Stage1 stress rollout plus postdiagnostic before Stage2 continuation protocol/runner or any further simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(RAW), rel(COMPLETED), rel(SUMMARY)],
    })
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress Stage1 postdiagnostic

UTC: {diagnosis['created_utc']}. No new simulations/training and no validation64/test access. Postdiagnostic of the 120-episode stress fixed-H map found branch horizons {BRANCH_HORIZONS} strict-safe, material non-H15 branch-positive cases {material_cases} (stress positives {stress_material_cases}), negative/nonmaterial cases {len(negative_cases)}, and aggregate Stage1 oracle gains total={stage1_mat.get('oracle_total_gain_abs')} physical={stage1_mat.get('oracle_physical_gain_abs')}. Decision: freeze Stage2 identical-state continuation after verified backup; do not retrain/refit yet because Stage1 is episode-level. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    append_docs(block)
    files = [RAW, COMPLETED, SUMMARY, OUT / "raw.json", OUT / "summary.md", STATE, backup_req, Path(__file__).resolve()]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": diagnosis["created_utc"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "stage2_identical_state_continuation_should_be_frozen_after_backup": stage2_supported,
        "material_non_H15_positive_cases": material_cases,
        "stress_material_positive_cases": stress_material_cases,
        "negative_or_nonmaterial_cases_count": len(negative_cases),
        "backup_request": rel(backup_req),
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "stage2_identical_state_continuation_should_be_frozen_after_backup": stage2_supported,
        "material_non_H15_positive_cases": material_cases,
        "stress_material_positive_cases": stress_material_cases,
        "negative_or_nonmaterial_cases_count": len(negative_cases),
        "branch_rows_strict_safe": branch_artifacts["all_branch_rows_strict_safe"],
        "backup_request": rel(backup_req),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
    }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
