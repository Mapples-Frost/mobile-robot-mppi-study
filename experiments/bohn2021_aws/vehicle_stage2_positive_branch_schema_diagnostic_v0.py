#!/usr/bin/env python3
"""Stage2 positive-branch schema diagnostic v0.

Analysis-only diagnostic over existing stress Stage2 raw artifacts.  It repairs a
known blind spot in the previous source audit: the Stage2 postdiagnostic summary
reported a relaxed positive state (case 5, branch step 18, positive H [10,25,30])
but the generic parser did not recover horizons from raw.  This script inspects
raw JSON schemas and extracts positive-branch/state-availability information for
planning the next terminal-value instrumentation smoke.

No simulations, no training/refit, no validation64-bank access and no sealed-test
access.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "stage2_continuation_raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json",
    "stage2_continuation_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/completed.json",
    "stage2_postdiagnostic_raw": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z/raw.json",
    "stage2_postdiagnostic_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z/completed.json",
    "terminal_objective_audit_completed": ROOT / "research_artifacts/aws_diagnostics/vehicle_terminal_objective_source_audit_v0_20260928T1830Z/completed.json",
}
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stage2_positive_branch_schema_diagnostic_v0_20260928T1840Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_stage2_positive_branch_schema_diagnostic_v0_20260928T1840Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-stage2-positive-branch-schema-diagnostic-v0-20260928T1840Z"


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def sha256(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
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


def num(x: Any) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def integer(x: Any) -> Optional[int]:
    y = num(x)
    if y is None:
        return None
    if abs(y - round(y)) <= 1e-9:
        return int(round(y))
    return None


def walk(obj: Any, path: str = "$", depth: int = 0) -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if depth > 32:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            safe = str(k).replace("/", "_")[:80]
            yield from walk(v, path + "/" + safe, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5000]):
            yield from walk(v, path + f"[{i}]", depth + 1)


def is_num_list(v: Any, min_len: int = 1, max_len: int = 64) -> bool:
    if not isinstance(v, list) or not (min_len <= len(v) <= max_len):
        return False
    return all(num(x) is not None for x in v)


def compact_value(v: Any, max_chars: int = 160) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if is_num_list(v):
        return {"len": len(v), "head": v[:8]}
    s = repr(v)
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def dict_case_step(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    case = None
    step = None
    for k, v in d.items():
        lk = str(k).lower()
        if case is None and lk in {"case", "case_id", "case_index", "source_case"}:
            case = integer(v)
        if step is None and lk in {"branch_step", "step", "prefix_step", "decision_step", "state_step"}:
            step = integer(v)
    return case, step


def dict_horizon(d: Dict[str, Any]) -> Optional[int]:
    for k, v in d.items():
        lk = str(k).lower()
        if lk in {"h", "horizon", "branch_h", "branch_horizon", "candidate_h", "candidate_horizon", "positive_h"}:
            z = integer(v)
            if z is not None and 0 < z <= 100:
                return z
    return None


def summarize_matching_dict(path: str, d: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(str(k) for k in d.keys())
    vector_fields = []
    cost_fields = {}
    terminal_fields = {}
    solver_fields = {}
    for k, v in d.items():
        lk = str(k).lower()
        if is_num_list(v, 2, 80) and any(tok in lk for tok in ("state", "x0", "obs", "pose", "trajectory", "tvp")):
            vector_fields.append({"key": str(k), "len": len(v), "head": v[:8]})
        if any(tok in lk for tok in ("cost", "gain", "reward", "physical", "total")) and not isinstance(v, (dict, list)):
            cost_fields[str(k)] = compact_value(v)
        if any(tok in lk for tok in ("terminal", "mterm", "vf", "value")) and not isinstance(v, (dict, list)):
            terminal_fields[str(k)] = compact_value(v)
        if any(tok in lk for tok in ("solver", "success", "failure", "status")) and not isinstance(v, (dict, list)):
            solver_fields[str(k)] = compact_value(v)
    return {
        "path": path,
        "keys": keys[:80],
        "case_step": dict_case_step(d),
        "horizon": dict_horizon(d),
        "vector_fields": vector_fields[:20],
        "cost_fields": cost_fields,
        "terminal_like_scalar_fields": terminal_fields,
        "solver_like_scalar_fields": solver_fields,
    }


def scan_positive_info(name: str, obj: Any) -> Dict[str, Any]:
    positive_list_hits: List[Dict[str, Any]] = []
    gain_rows: List[Dict[str, Any]] = []
    matching_case5_step18: List[Dict[str, Any]] = []
    schema_hits: List[Dict[str, Any]] = []
    horizons = set()
    for path, node in walk(obj):
        if isinstance(node, dict):
            case, step = dict_case_step(node)
            h = dict_horizon(node)
            if case == 5 and step == 18 and len(matching_case5_step18) < 60:
                matching_case5_step18.append(summarize_matching_dict(path, node))
            # key paths mentioning positive horizons or best horizon.
            for k, v in node.items():
                lk = str(k).lower()
                if "positive" in lk and ("h" in lk or "horizon" in lk):
                    vals = []
                    if isinstance(v, list):
                        vals = [integer(x) for x in v]
                    else:
                        vals = [integer(v)]
                    vals = [x for x in vals if x is not None and 0 < x <= 100]
                    if vals:
                        positive_list_hits.append({"path": path + "/" + str(k), "values": vals, "case": case, "branch_step": step})
                        horizons.update(vals)
                if ("best" in lk or "positive" in lk) and ("h" in lk or "horizon" in lk):
                    z = integer(v)
                    if z is not None and 0 < z <= 100 and z != 15:
                        schema_hits.append({"path": path + "/" + str(k), "value": z, "case": case, "branch_step": step})
            # numeric gain rows by schema-agnostic key names.
            total_gain = None
            physical_gain = None
            for k, v in node.items():
                lk = str(k).lower()
                val = num(v)
                if val is None:
                    continue
                if "gain" in lk and "total" in lk:
                    total_gain = val if total_gain is None else max(total_gain, val)
                if "gain" in lk and ("physical" in lk or "phys" in lk):
                    physical_gain = val if physical_gain is None else max(physical_gain, val)
            if h is not None and h != 15 and ((total_gain is not None and total_gain >= 3.0) or (physical_gain is not None and physical_gain >= 3.0)):
                row = summarize_matching_dict(path, node)
                row.update({"source": name, "total_gain_detected": total_gain, "physical_gain_detected": physical_gain})
                gain_rows.append(row)
                horizons.add(h)
    return {
        "source": name,
        "positive_list_hits": positive_list_hits[:80],
        "gain_rows": gain_rows[:120],
        "schema_best_h_hits": schema_hits[:80],
        "matching_case5_step18_dicts": matching_case5_step18[:60],
        "positive_horizons_union": sorted(horizons),
        "case5_step18_saved_vector_field_count": sum(len(x.get("vector_fields", [])) for x in matching_case5_step18),
    }


def append_once(path: Path, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_once(ROOT / name, block)


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    loaded: Dict[str, Any] = {}
    inventory: Dict[str, Any] = {}
    for name, path in INPUTS.items():
        inventory[name] = {"path": rel(path), "exists": path.exists(), "sha256": sha256(path)}
        if path.exists() and path.suffix == ".json":
            obj = read_json(path)
            loaded[name] = obj
            if isinstance(obj, dict):
                inventory[name]["top_level_keys"] = sorted(str(k) for k in obj.keys())[:80]
            elif isinstance(obj, list):
                inventory[name]["top_level_type"] = "list"
                inventory[name]["top_level_len"] = len(obj)
    scans = {name: scan_positive_info(name, obj) for name, obj in loaded.items() if "stage2" in name}
    union_h = sorted(set(h for s in scans.values() for h in s.get("positive_horizons_union", [])))
    case5_vec_count = sum(s.get("case5_step18_saved_vector_field_count", 0) for s in scans.values())
    direct_state_available = case5_vec_count > 0
    # If no vector is stored in raw, deterministic replay from case/branch metadata remains the likely path.
    next_action = (
        "After verified backup, freeze terminal-value instrumentation using saved case5/step18 state vectors plus neutral/harm controls."
        if direct_state_available else
        "After verified backup, freeze a tiny deterministic replay-to-branch terminal-value smoke for case5 step18 plus neutral/harm controls; raw artifacts do not expose enough direct state-vector fields for state injection."
    )
    findings = [
        f"Parsed positive-horizon union from Stage2 raw/schema scan: {union_h if union_h else 'none detected by raw scan'}.",
        f"Case5 step18 raw matching dictionaries found: {sum(len(s.get('matching_case5_step18_dicts', [])) for s in scans.values())}; saved vector-like fields: {case5_vec_count}.",
        "This analysis-only diagnostic does not change the frozen Stage2 gate: frozen positive_state_count remains 0 and refit remains unjustified.",
        "If raw does not contain direct state vectors, terminal-value measurement should reproduce the deterministic prefix rather than inventing state injection code.",
    ]
    decision = {
        "formal_scientific_evidence": False,
        "retrain_or_selector_refit_now": False,
        "terminal_instrumentation_inputs_sufficient_for_direct_state_injection": direct_state_available,
        "next_action_after_backup": next_action,
        "backup_required_before_more_simulations_or_source_revision": True,
    }
    raw = {"created_utc": created, "method": "vehicle_stage2_positive_branch_schema_diagnostic_v0_analysis_only", "inputs": inventory, "scans": scans, "findings": findings, "decision": decision, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0}
    write_json(OUT / "raw.json", raw)
    summary = ["# Vehicle Stage2 positive-branch schema diagnostic v0", "", f"UTC: `{created}`. Analysis-only; no simulations/training/refit, no validation64, no sealed test.", "", "## Findings"]
    summary.extend("- " + f for f in findings)
    summary += ["", "## Decision", f"- Retrain/refit now: `{decision['retrain_or_selector_refit_now']}`.", f"- Direct saved-state vector available for terminal instrumentation: `{direct_state_available}`.", f"- Next after backup: {next_action}"]
    for name, scan in scans.items():
        summary += ["", f"## Scan: {name}", f"- Positive horizons union: `{scan.get('positive_horizons_union')}`", f"- Positive-list hits: `{len(scan.get('positive_list_hits', []))}`; gain rows: `{len(scan.get('gain_rows', []))}`; case5/step18 dicts: `{len(scan.get('matching_case5_step18_dicts', []))}`; vector fields: `{scan.get('case5_step18_saved_vector_field_count')}`"]
        for row in scan.get("positive_list_hits", [])[:5]:
            summary.append(f"  - list hit `{row['path']}` values={row['values']} case={row.get('case')} step={row.get('branch_step')}")
        for row in scan.get("gain_rows", [])[:5]:
            summary.append(f"  - gain row `{row['path']}` H={row.get('horizon')} case_step={row.get('case_step')} total_gain={row.get('total_gain_detected')} physical_gain={row.get('physical_gain_detected')}")
    req = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_POSITIVE_BRANCH_SCHEMA_DIAGNOSTIC_V0_20260928T1840Z.json"
    write_json(req, {"requested_utc": created, "reason": "backup Stage2 positive-branch schema diagnostic before any terminal instrumentation simulation/source revision", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(req)]})
    raw["backup_request"] = rel(req)
    write_json(OUT / "raw.json", raw)
    summary += ["", f"Backup request: `{rel(req)}`."]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("\n".join(summary) + "\n", encoding="utf-8")
    block = f"""<!-- {MARKER} -->
## 2026-09-28 Stage2 positive-branch schema diagnostic v0

UTC: {created}. Analysis-only; no rollouts, no training/refit, no validation64/test access. Findings: parsed Stage2 positive horizon union `{union_h if union_h else []}`; case5 step18 matching raw dictionaries `{sum(len(s.get('matching_case5_step18_dicts', [])) for s in scans.values())}`; direct saved vector fields `{case5_vec_count}`. Frozen Stage2 gate remains failed and retrain/refit remains false. Next after verified backup: {next_action} Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`. Backup request: `{rel(req)}`.
"""
    append_docs(block)
    completed = {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "parsed_positive_horizons_union": union_h, "case5_step18_matching_dict_count": sum(len(s.get('matching_case5_step18_dicts', [])) for s in scans.values()), "case5_step18_saved_vector_field_count": case5_vec_count, "direct_state_available": direct_state_available, "retrain_or_selector_refit_now": False, "next_action_after_backup": next_action, "backup_request": rel(req), "hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), OUT / 'raw.json', OUT / 'summary.md', STATE, req] + list(INPUTS.values()) if p.exists()}}
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "parsed_positive_horizons_union": union_h, "case5_step18_saved_vector_field_count": case5_vec_count, "direct_state_available": direct_state_available, "retrain_or_selector_refit_now": False, "backup_request": rel(req), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
