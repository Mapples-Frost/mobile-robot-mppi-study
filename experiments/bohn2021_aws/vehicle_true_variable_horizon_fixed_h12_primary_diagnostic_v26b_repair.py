#!/usr/bin/env python3
"""v26b repair for fixed-H12-primary offline audit.

Development-only metadata/results audit. No MPC simulation, no validation64 bank
open, no sealed test, no training/refit.

Why this exists: v26 correctly froze a metadata-only source-independent
case-selection protocol, but its generic row parser omitted two older raw-result
schemas: v19 stores H12/H15 evidence in analysis.aggregate + candidate_rows, and
v21 stores it in analysis.aggregate/state_rows. Consequently the v26 summary
under-counted v19 and v21 as zero rows. This script creates a new immutable
repair artifact and a repaired protocol amendment; it never overwrites v26.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair"
STAMP = "20260930T0235Z"
MARKER = f"vehicle-fixed-h12-primary-diagnostic-v26b-repair-{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_corrected_fixed_H12_primary_confirmation_amendment_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0235_after_v26b_repair.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V26B_REPAIR_{STAMP}.json"

V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V21_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json"
V23_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/raw.json"
V25_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/raw.json"
V24_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/summary.md"
V26_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26_preoutcome_fixed_H12_primary_confirmation_20260930T0225Z.json"
V26_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26_20260930T0225Z/completed.json"
V26_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26_20260930T0225Z/summary.md"
POST_V26_SOURCE_BACKUP = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T021413_from_supervisor_context_after_v26_source.json"
POST_V25_BACKUP = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T020708_from_supervisor_context_after_v25.json"

TARGET_CASES = 12
BRANCH_STATES_PER_CASE = 2
TRUE_HORIZONS = [12, 15]
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS)
CONTROL_STEP_CAP = TOTAL_EPISODES * 150

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

def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x

def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default

def pct(x: Optional[float]) -> str:
    return "NA" if x is None else f"{100.0 * x:.2f}%"

def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> Optional[float]:
    if n <= 0:
        return None
    ph = k / float(n)
    den = 1.0 + z * z / n
    centre = ph + z * z / (2.0 * n)
    rad = z * math.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n))
    return min(1.0, (centre + rad) / den)

def agg_from_sums(name: str, rows: int, ben: int, bad: int, h12_dec: float, h15_dec: float,
                  h12_sol: float, h15_sol: float, h12_phys: float, h15_phys: float,
                  tol: float, provenance: str, pass_override: Optional[bool] = None) -> Dict[str, Any]:
    dec_save = (1.0 - h12_dec / h15_dec) if h15_dec > 0 else None
    sol_save = (1.0 - h12_sol / h15_sol) if h15_sol > 0 else None
    phys_delta = h12_phys - h15_phys
    pass5 = bool(pass_override) if pass_override is not None else (bad == 0 and phys_delta <= tol and dec_save is not None and dec_save >= 0.05)
    return {
        "dataset": name,
        "rows": int(rows),
        "h12_beneficial": int(ben),
        "h12_bad": int(bad),
        "h12_decision_sum_s": h12_dec,
        "h15_decision_sum_s": h15_dec,
        "decision_saving_vs_H15": dec_save,
        "h12_solver_sum_s": h12_sol,
        "h15_solver_sum_s": h15_sol,
        "solver_saving_vs_H15": sol_save,
        "h12_physical_sum": h12_phys,
        "h15_physical_sum": h15_phys,
        "physical_delta_H12_minus_H15": phys_delta,
        "physical_tolerance_sum": tol,
        "physical_gate": phys_delta <= tol,
        "pass5": pass5,
        "provenance": provenance,
    }

def v19_state_level() -> Dict[str, Any]:
    raw = read_json(V19_RAW)
    ana = raw.get("analysis") or {}
    ag = ana.get("aggregate") or {}
    candidates = ana.get("candidate_rows") or []
    rows = len(candidates)
    pair_count = sf(ag.get("pair_count"), rows)
    factor = pair_count / rows if rows else 1.0
    if rows <= 0 or factor <= 0:
        raise ContractError("v19 candidate_rows/pair_count missing")
    return agg_from_sums(
        "v19_opened_boundary_state_level",
        rows=rows,
        ben=round(sf(ag.get("beneficial_H12_rows")) / factor),
        bad=round(sf(ag.get("catastrophic_H12_rows")) / factor),
        h12_dec=sf(ag.get("fixed_H12_decision_sum_s")) / factor,
        h15_dec=sf(ag.get("fixed_H15_decision_sum_s")) / factor,
        h12_sol=sf(ag.get("fixed_H12_solver_sum_s")) / factor,
        h15_sol=sf(ag.get("fixed_H15_solver_sum_s")) / factor,
        h12_phys=sf(ag.get("fixed_H12_physical_sum")) / factor,
        h15_phys=sf(ag.get("fixed_H15_physical_sum")) / factor,
        tol=sf(ag.get("physical_tolerance_sum_vs_H15")) / factor,
        provenance="v19 analysis.aggregate divided by uniform repeat factor pair_count/candidate_rows; preserves v24 state-level interpretation",
        pass_override=bool(ag.get("fixed_H12_tradeoff_pass5", False)),
    )

def standard_state_level(path: Path, name: str) -> Dict[str, Any]:
    raw = read_json(path)
    ana = raw.get("analysis") or {}
    lab = ana.get("label_analysis") if isinstance(ana.get("label_analysis"), Mapping) else None
    ag = (lab or ana).get("aggregate") or {}
    rows = int(ag.get("states") or len((lab or ana).get("state_rows") or ana.get("evaluation_rows") or []))
    if rows <= 0:
        raise ContractError(f"{name} rows missing")
    return agg_from_sums(
        name,
        rows=rows,
        ben=int(ag.get("beneficial_H12_count", 0)),
        bad=int(ag.get("catastrophic_H12_count", 0)),
        h12_dec=sf(ag.get("fixed_H12_decision_sum_s")),
        h15_dec=sf(ag.get("fixed_H15_decision_sum_s")),
        h12_sol=sf(ag.get("fixed_H12_solver_sum_s")),
        h15_sol=sf(ag.get("fixed_H15_solver_sum_s")),
        h12_phys=sf(ag.get("fixed_H12_physical_sum")),
        h15_phys=sf(ag.get("fixed_H15_physical_sum")),
        tol=sf(ag.get("physical_tolerance_sum")),
        provenance=f"{name} analysis aggregate/state_rows schema",
        pass_override=bool(ag.get("fixed_H12_tradeoff_pass5", False)),
    )

def combine(name: str, parts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return agg_from_sums(
        name,
        rows=sum(int(p["rows"]) for p in parts),
        ben=sum(int(p["h12_beneficial"]) for p in parts),
        bad=sum(int(p["h12_bad"]) for p in parts),
        h12_dec=math.fsum(sf(p["h12_decision_sum_s"]) for p in parts),
        h15_dec=math.fsum(sf(p["h15_decision_sum_s"]) for p in parts),
        h12_sol=math.fsum(sf(p["h12_solver_sum_s"]) for p in parts),
        h15_sol=math.fsum(sf(p["h15_solver_sum_s"]) for p in parts),
        h12_phys=math.fsum(sf(p["h12_physical_sum"]) for p in parts),
        h15_phys=math.fsum(sf(p["h15_physical_sum"]) for p in parts),
        tol=math.fsum(sf(p["physical_tolerance_sum"]) for p in parts),
        provenance="sum of corrected state-level aggregates: " + ", ".join(str(p["dataset"]) for p in parts),
    )

def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")

def append_registry(created: str) -> None:
    p = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    if MARKER in old:
        return
    with p.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([created, NAME, "development_offline_audit_repair_no_sim", "metadata_only", "existing_development_results_no_validation64_no_test", "0", "0", "0", "0", "0", "False", rel(RUN_DIR / "completed.json"), MARKER])

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-audit", action="store_true", required=True)
    ap.add_argument("--i-accept-v26b-correction", action="store_true", required=True)
    args = ap.parse_args(argv)
    del args
    for p in [V19_RAW, V21_RAW, V23_RAW, V25_RAW, V24_SUMMARY, V26_PROTOCOL, V26_COMPLETED, V26_SUMMARY]:
        if not p.exists():
            raise ContractError("missing required input: " + rel(p))
    created = now_utc()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    a19 = v19_state_level()
    a21 = standard_state_level(V21_RAW, "v21_source_independent")
    a23 = standard_state_level(V23_RAW, "v23_broader_source_independent")
    a25 = standard_state_level(V25_RAW, "v25_source72_risk_family")
    fresh = combine("fresh_v21_v23_v25", [a21, a23, a25])
    all_opened = combine("all_opened_v19_v21_v23_v25", [a19, a21, a23, a25])
    old_v26 = read_json(V26_PROTOCOL)
    selected = (((old_v26.get("case_selection_diagnostics") or {}).get("selected_source_candidate_indices")) or [c.get("source_candidate_index") for c in old_v26.get("selected_cases", [])])
    risk_bounds = {
        "fresh_h12_bad_count": fresh["h12_bad"],
        "fresh_rows": fresh["rows"],
        "wilson_95_upper_bad_rate": wilson_upper(int(fresh["h12_bad"]), int(fresh["rows"])),
        "zero_failure_rule_of_three_upper": 3.0 / float(fresh["rows"]),
        "all_opened_h12_bad_count": all_opened["h12_bad"],
        "all_opened_rows": all_opened["rows"],
    }
    repaired_protocol = {
        "protocol_id": f"{NAME}_corrected_fixed_H12_primary_confirmation_amendment_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_v26_offline_audit_repair_no_sim_no_validation_no_test",
        "repair_reason": "v26 omitted v19 and v21 due schema mismatch; this amendment corrects before_evidence/risk_bounds and preserves the already frozen metadata-only selected cases.",
        "original_v26_protocol": {"path": rel(V26_PROTOCOL), "sha256": sha256(V26_PROTOCOL)},
        "original_v26_completed": {"path": rel(V26_COMPLETED), "sha256": sha256(V26_COMPLETED)},
        "corrected_before_evidence": {"v19": a19, "v21": a21, "v23": a23, "v25": a25, "fresh_v21_v23_v25": fresh, "all_opened_v19_v21_v23_v25": all_opened},
        "risk_bounds": risk_bounds,
        "selected_source_candidate_indices_preserved_from_v26": selected,
        "case_selection_hash_preserved": sha256(V26_PROTOCOL),
        "budget_declared_for_future_confirmation": {"total_episodes_exact": TOTAL_EPISODES, "control_step_upper_bound": CONTROL_STEP_CAP, "new_gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rules": old_v26.get("decision_rules", {}),
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), V19_RAW, V21_RAW, V23_RAW, V25_RAW, V24_SUMMARY, V26_PROTOCOL, V26_COMPLETED, V26_SUMMARY, POST_V26_SOURCE_BACKUP, POST_V25_BACKUP] if p.exists()},
    }
    write_json(PROTOCOL, repaired_protocol)
    raw = {"created_utc": created.isoformat(), "classification": repaired_protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulation_episodes": 0, "new_control_steps": 0, "new_gradient_steps": 0, "corrected_aggregates": repaired_protocol["corrected_before_evidence"], "risk_bounds": risk_bounds, "selected_source_candidate_indices_preserved_from_v26": selected, "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}}
    write_json(RUN_DIR / "raw.json", raw)
    rows = [a19, a21, a23, a25, fresh, all_opened]
    lines = [
        "# Vehicle fixed-H12-primary diagnostic v26b repair",
        "",
        f"UTC: `{created.isoformat()}`. Offline correction only; no simulation, no validation64, no sealed test, no training/refit.",
        "",
        "## Repair finding",
        "",
        "v26's generic raw parser omitted v19 (`analysis.aggregate` + `candidate_rows`) and v21 (`analysis.aggregate` + `state_rows`), so its table incorrectly showed zero rows for those datasets. The v26 selected source indices remain metadata-only and are preserved, but the before-evidence/risk-bounds protocol text must be amended before any fixed-H12-primary simulation.",
        "",
        "| dataset | rows | H12 beneficial | H12 bad | fixed-H12 decision saving | solver saving | phys Δ H12-H15 | physical gate | pass5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in rows:
        lines.append(f"| `{a['dataset']}` | {a['rows']} | {a['h12_beneficial']} | {a['h12_bad']} | {pct(a['decision_saving_vs_H15'])} | {pct(a['solver_saving_vs_H15'])} | {a['physical_delta_H12_minus_H15']:.6g} | `{a['physical_gate']}` | `{a['pass5']}` |")
    lines += [
        "",
        f"Fresh v21+v23+v25 corrected zero-bad count is `{fresh['h12_bad']}/{fresh['rows']}`; Wilson 95% upper bad-rate bound `{pct(risk_bounds['wilson_95_upper_bad_rate'])}` and rule-of-three `{pct(risk_bounds['zero_failure_rule_of_three_upper'])}`. This still does not prove a population failure rate of zero.",
        f"All opened state-level rows including v19 have `{all_opened['h12_bad']}/{all_opened['rows']}` H12 bad rows and fixed H12 fails physical/pass5, preserving v19 as adaptive-opportunity counterevidence.",
        "",
        f"Repaired protocol amendment: `{rel(PROTOCOL)}` SHA256 `{sha256(PROTOCOL)}`. Preserved selected source indices: `{selected}`. Future confirmation budget remains `{TOTAL_EPISODES}` development episodes, cap `{CONTROL_STEP_CAP}` control steps, validation64/test `0`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(BACKUP_REQUEST, {"request": "backup_after_v26b_repair", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "v26b source/repair artifacts/docs/response-log amendment and prior v26 outputs must be externally recoverable before simulations", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(PROTOCOL), rel(STATE), rel(BACKUP_REQUEST), rel(POST_V26_SOURCE_BACKUP), rel(V26_PROTOCOL), rel(V26_COMPLETED), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "new_simulation_episodes": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle v26b fixed-H12 audit repair

UTC: {created.isoformat()}. Offline correction only; no simulation, no validation64, no sealed test. v26's parser omitted v19/v21 schemas, so v26's before-evidence table/risk bound was incomplete. Corrected fresh v21+v23+v25: n={fresh['rows']}, H12_bad={fresh['h12_bad']}, fixed_H12 decision saving={fresh['decision_saving_vs_H15']}, pass5={fresh['pass5']}. Corrected all opened v19+v21+v23+v25: n={all_opened['rows']}, H12_bad={all_opened['h12_bad']}, fixed_H12 pass5={all_opened['pass5']}; v19 counterevidence remains. Repaired protocol amendment `{rel(PROTOCOL)}` preserves v26 selected source indices {selected}. Backup request `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
## Correction after v26b fixed-H12 audit repair

Updated by GPT-5.5 executor at `{created.isoformat()}`. This correction supersedes the v26 row-count/risk-bound entries only; it does not overwrite v26 and does not open validation64 or sealed test.

| linked recommendation(s) | disposition after v26b repair | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; v26 evidence corrected | v26 parser bug found: v19 and v21 were counted as zero. Corrected fresh v21+v23+v25 fixed H12: rows={fresh['rows']}, bad={fresh['h12_bad']}, decision saving={pct(fresh['decision_saving_vs_H15'])}, pass5={fresh['pass5']}. Corrected all opened including v19: rows={all_opened['rows']}, bad={all_opened['h12_bad']}, pass5={all_opened['pass5']}. | Do not run fixed-H12-primary simulation from the unamended v26 evidence. Use repaired protocol `{rel(PROTOCOL)}`; fixed H12 remains primary on fresh rows, while v19 remains adaptive-opportunity counterevidence. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; quantified with corrected denominator | Corrected fresh zero-bad count {fresh['h12_bad']}/{fresh['rows']}; Wilson upper {pct(risk_bounds['wilson_95_upper_bad_rate'])}. | Still development/stress-pool only; no population or final-test claim. |
| `A11_training_failure_modes_need_separation` | accepted; next gate unchanged but now evidence-correct | Fresh corrected rows still favor fixed H12, while all-opened rows fail fixed H12 because of v19. | After backup, either run the repaired fixed-H12-primary confirmation or, if fixed H12 fails there, pivot to terminal-risk/value refit or bounded training. |
| `A12_registry_backup_schema_contract` | accepted; active | v26b wrote repair artifacts and backup request `{rel(BACKUP_REQUEST)}`. | Require verified backup before more unique science. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v26b repair\n\nUTC: {created.isoformat()}\n\nNo simulations/training/validation/test. v26 parser issue repaired in `{rel(PROTOCOL)}`. Corrected fresh rows: n={fresh['rows']}, H12_bad={fresh['h12_bad']}, fixed_H12_save={pct(fresh['decision_saving_vs_H15'])}, pass5={fresh['pass5']}. Corrected all opened: n={all_opened['rows']}, H12_bad={all_opened['h12_bad']}, pass5={all_opened['pass5']}. Next action after verified backup: run repaired fixed-H12-primary confirmation or implement its simulation runner, not the unamended v26 evidence.\n", encoding="utf-8")
    append_registry(created.isoformat())
    completed = {"status": "complete", "passed": True, "created_utc": created.isoformat(), "classification": repaired_protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulation_episodes": 0, "new_control_steps": 0, "new_gradient_steps": 0, "summary": rel(RUN_DIR / "summary.md"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST)}
    files = [Path(__file__).resolve(), RUN_DIR / "raw.json", RUN_DIR / "summary.md", PROTOCOL, STATE, BACKUP_REQUEST, ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
    completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "protocol": rel(PROTOCOL), "fresh_rows": fresh["rows"], "fresh_h12_bad": fresh["h12_bad"], "fresh_fixed_h12_decision_saving": fresh["decision_saving_vs_H15"], "all_opened_rows": all_opened["rows"], "all_opened_h12_bad": all_opened["h12_bad"], "selected_source_candidate_indices": selected, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
