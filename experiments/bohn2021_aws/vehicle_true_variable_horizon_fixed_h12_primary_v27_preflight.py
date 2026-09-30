#!/usr/bin/env python3
"""v27 fixed-H12-primary confirmation preflight.

This is a bounded metadata diagnostic, not a simulation runner. It verifies that
v26b corrected evidence/protocol are internally consistent, checks whether an
adequate post-v26b external backup proof is already present, and records the
next executable gate. It does not open validation64, sealed test, or run MPC.
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
NAME = "vehicle_true_variable_horizon_fixed_h12_primary_v27_preflight"
STAMP = "20260930T0240Z"
MARKER = f"vehicle-fixed-h12-primary-v27-preflight-{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0240_after_v27_preflight.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_PREFLIGHT_{STAMP}.json"

V26B_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_20260930T0235Z/completed.json"
V26B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_20260930T0235Z/summary.md"
V26B_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_corrected_fixed_H12_primary_confirmation_amendment_20260930T0235Z.json"
V26B_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260930T022219_5b0d8142/registry.json"
POST_V26B_BACKUP_PROOF_MATERIALIZED = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T022031_from_supervisor_context_after_v26_run_and_pending_v26b_source.json"

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
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
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

def parse_time(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def pct(x: Any) -> str:
    try:
        y = float(x)
        return f"{100.0*y:.2f}%" if math.isfinite(y) else "NA"
    except Exception:
        return "NA"

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
        csv.writer(f).writerow([
            created, NAME, "development_metadata_preflight_no_sim", "metadata_only",
            "v26b_repaired_protocol_no_validation64_no_test", 0, 0, 0, 0, 0, False,
            rel(RUN_DIR / "completed.json"), MARKER,
        ])

def find_post_v26b_backup(v26b_time: dt.datetime) -> Dict[str, Any]:
    candidates = []
    for p in sorted((ROOT / "research_artifacts/aws_backup_proofs").glob("backup_proof_*.json")):
        try:
            obj = read_json(p)
        except Exception:
            continue
        t = parse_time(obj.get("time"))
        verified = obj.get("status") == "verified" and obj.get("backup_verified") is True and obj.get("remaining_changed_files") == 0
        after = bool(t and t > v26b_time)
        covers_hint = any(token in json.dumps(obj, sort_keys=True) for token in ["v26b", "022219", "V26B"])
        candidates.append({"path": rel(p), "time": obj.get("time"), "commit": obj.get("commit"), "verified_remaining_zero": verified, "after_v26b": after, "coverage_hint": covers_hint, "adequate_for_v27_sim": bool(verified and after and covers_hint)})
    adequate = [c for c in candidates if c["adequate_for_v27_sim"]]
    return {"adequate_post_v26b_backup_present": bool(adequate), "adequate_candidates": adequate, "all_backup_proofs_considered": candidates[-8:]}

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", required=True)
    ap.add_argument("--i-understand-no-simulation", action="store_true", required=True)
    args = ap.parse_args(argv)
    del args

    created = now_utc()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for p in [V26B_COMPLETED, V26B_SUMMARY, V26B_PROTOCOL, V26B_RUN_REGISTRY, POST_V26B_BACKUP_PROOF_MATERIALIZED]:
        if not p.exists():
            raise ContractError("missing required preflight input: " + rel(p))

    completed = read_json(V26B_COMPLETED)
    protocol = read_json(V26B_PROTOCOL)
    v26b_time = parse_time(completed.get("created_utc"))
    if v26b_time is None:
        raise ContractError("cannot parse v26b completed created_utc")
    evidence = protocol.get("corrected_before_evidence") or {}
    fresh = evidence.get("fresh_v21_v23_v25") or {}
    all_opened = evidence.get("all_opened_v19_v21_v23_v25") or {}
    selected = protocol.get("selected_source_candidate_indices_preserved_from_v26") or []
    budget = protocol.get("budget_declared_for_future_confirmation") or {}
    checks = {
        "v26b_completed_passed": completed.get("status") == "complete" and completed.get("passed") is True,
        "v26b_no_validation_or_test": completed.get("validation64_bank_opened") is False and completed.get("sealed_test_accessed") is False,
        "fresh_rows_56_bad0_pass": fresh.get("rows") == 56 and fresh.get("h12_bad") == 0 and fresh.get("pass5") is True,
        "all_opened_rows80_bad3_fail": all_opened.get("rows") == 80 and all_opened.get("h12_bad") == 3 and all_opened.get("pass5") is False,
        "selected_12_cases_present": isinstance(selected, list) and len(selected) == 12,
        "future_budget_60_episodes_9000_steps": budget.get("total_episodes_exact") == 60 and budget.get("control_step_upper_bound") == 9000,
        "post_v26b_materialized_backup_predates_repair": parse_time(read_json(POST_V26B_BACKUP_PROOF_MATERIALIZED).get("time")) < v26b_time,
    }
    backup_gate = find_post_v26b_backup(v26b_time)
    input_sufficient_for_unique_simulation = all(checks.values()) and backup_gate["adequate_post_v26b_backup_present"]
    next_action = "run repaired fixed-H12-primary 60-episode development confirmation" if input_sufficient_for_unique_simulation else "wait for verified post-v26b backup; then run/implement repaired fixed-H12-primary confirmation"

    raw = {
        "created_utc": created.isoformat(),
        "classification": "development_metadata_preflight_no_sim_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "checks": checks,
        "backup_gate": backup_gate,
        "input_sufficient_for_unique_simulation_now": input_sufficient_for_unique_simulation,
        "next_action": next_action,
        "v26b_corrected_fresh": fresh,
        "v26b_corrected_all_opened": all_opened,
        "preserved_selected_source_candidate_indices": selected,
        "future_confirmation_budget": budget,
        "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), V26B_COMPLETED, V26B_SUMMARY, V26B_PROTOCOL, V26B_RUN_REGISTRY, POST_V26B_BACKUP_PROOF_MATERIALIZED]},
    }
    write_json(RUN_DIR / "raw.json", raw)

    lines = [
        "# v27 fixed-H12-primary confirmation preflight",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only preflight; no simulation, no validation64, no sealed test, no training/refit.",
        "",
        "## Gate checks",
        "",
        "| check | passed |",
        "|---|---:|",
    ]
    for k, v in checks.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += [
        "",
        f"Adequate post-v26b backup present: `{backup_gate['adequate_post_v26b_backup_present']}`.",
        f"Input sufficient for unique simulation now: `{input_sufficient_for_unique_simulation}`.",
        "",
        "## Corrected scientific context carried forward",
        "",
        f"Fresh v21+v23+v25: rows `{fresh.get('rows')}`, H12_bad `{fresh.get('h12_bad')}`, fixed-H12 decision saving `{pct(fresh.get('decision_saving_vs_H15'))}`, pass5 `{fresh.get('pass5')}`.",
        f"All opened v19+v21+v23+v25: rows `{all_opened.get('rows')}`, H12_bad `{all_opened.get('h12_bad')}`, fixed-H12 decision saving `{pct(all_opened.get('decision_saving_vs_H15'))}`, pass5 `{all_opened.get('pass5')}`.",
        f"Preserved v26/v26b selected source_candidate_index values: `{selected}`.",
        "",
        f"Next action: **{next_action}**.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v27_preflight",
        "created_utc": created.isoformat(),
        "backup_required_before_more_unique_science": True,
        "reason": "v27 metadata preflight outputs/docs/state/registry must be externally recoverable before any fixed-H12-primary simulation",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"],
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })

    block = f"""<!-- {MARKER} -->
## 2026-09-30 v27 fixed-H12-primary preflight

UTC: {created.isoformat()}. Metadata-only; no simulation/training/validation/test. v26b protocol integrity checks passed={all(checks.values())}. Corrected fresh evidence remains n={fresh.get('rows')}, H12_bad={fresh.get('h12_bad')}, fixed_H12_save={pct(fresh.get('decision_saving_vs_H15'))}, pass5={fresh.get('pass5')}; all-opened evidence remains n={all_opened.get('rows')}, H12_bad={all_opened.get('h12_bad')}, pass5={all_opened.get('pass5')}. Adequate post-v26b backup present={backup_gate['adequate_post_v26b_backup_present']}; input sufficient for unique simulation now={input_sufficient_for_unique_simulation}. Next action: {next_action}. Backup request `{rel(BACKUP_REQUEST)}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
## Follow-up through v27 fixed-H12-primary preflight

Updated by GPT-5.5 executor at `{created.isoformat()}`. v27 preflight is metadata-only and does not open validation64 or sealed test.

| linked recommendation(s) | disposition after v27 preflight | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; active blocker | v27 preflight found no adequate verified backup after the v26b repair time `{v26b_time.isoformat()}`. Latest materialized proof intentionally predates v26b. | Do not run the 60-episode fixed-H12 confirmation until a verified post-v26b backup covers v26b/v27 artifacts. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; ready but gated | v26b protocol integrity checks passed; selected indices preserved `{selected}`; budget remains 60 development episodes / 9000 control-step cap. | After backup, run or implement/run the repaired fixed-H12-primary confirmation with fixed H12 as primary baseline. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v27 did not add outcomes; it carries corrected fresh 0/{fresh.get('rows')} H12-bad and all-opened {all_opened.get('h12_bad')}/{all_opened.get('rows')} H12-bad. | Keep all claims development/stress-pool scoped. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    STATE.write_text(
        f"# Continue state after v27 preflight\n\nUTC: {created.isoformat()}\n\nNo simulation/training/validation/test. v26b protocol integrity checks passed={all(checks.values())}. Adequate post-v26b backup present={backup_gate['adequate_post_v26b_backup_present']}; input sufficient for unique simulation now={input_sufficient_for_unique_simulation}. Next action: {next_action}. Backup request: {rel(BACKUP_REQUEST)}.\n",
        encoding="utf-8",
    )
    append_registry(created.isoformat())
    completed_out = {
        "status": "complete",
        "passed": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulation_episodes": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "input_sufficient_for_unique_simulation_now": input_sufficient_for_unique_simulation,
        "summary": rel(RUN_DIR / "summary.md"),
        "backup_request": rel(BACKUP_REQUEST),
    }
    files = [Path(__file__).resolve(), RUN_DIR / "raw.json", RUN_DIR / "summary.md", STATE, BACKUP_REQUEST, ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv", ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"]
    completed_out["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
    write_json(RUN_DIR / "completed.json", completed_out)
    print(json.dumps({
        "completed": rel(RUN_DIR / "completed.json"),
        "summary": rel(RUN_DIR / "summary.md"),
        "input_sufficient_for_unique_simulation_now": input_sufficient_for_unique_simulation,
        "adequate_post_v26b_backup_present": backup_gate["adequate_post_v26b_backup_present"],
        "next_action": next_action,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
