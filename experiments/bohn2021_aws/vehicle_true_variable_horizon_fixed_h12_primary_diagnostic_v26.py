#!/usr/bin/env python3
"""v26 fixed-H12-primary diagnostic/protocol freeze.

Development-only, no simulation, no validation64, no sealed test.

Purpose after v25: quantify the current fixed-H12-vs-adaptive evidence from
opened H12/H15 branch datasets, materialize the supervisor-provided post-v25
backup proof, and freeze the next bounded fixed-H12-primary confirmation design
without reading validation/test outcomes or running MPC.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26"
STAMP = "20260930T0225Z"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
SOURCE = Path(__file__).resolve()
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_fixed_H12_primary_confirmation_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0225_after_fixed_h12_primary_diagnostic_v26.md"
BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260930T020708_from_supervisor_context_after_v25.json"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_FIXED_H12_PRIMARY_DIAGNOSTIC_V26_{STAMP}.json"
MARKER = f"vehicle-fixed-h12-primary-diagnostic-v26-{STAMP}"

V19_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json"
V21_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json"
V23_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/raw.json"
V25_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/raw.json"
V24_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/summary.md"
V25_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/completed.json"
V25_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_preoutcome_frozen_20260930T0210Z.json"

TARGET_CASES = 12
BRANCH_STATES_PER_CASE = 2
TRUE_HORIZONS = [12, 15]
MAX_STEPS = 150
TOTAL_EPISODES = TARGET_CASES + TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS)
CONTROL_STEP_CAP = TOTAL_EPISODES * MAX_STEPS

SUPERVISOR_BACKUP = {
    "time": "2026-09-30T02:07:08.464651+00:00",
    "status": "verified",
    "backup_verified": True,
    "source": "user/supervisor context supplied at start of v26 bounded iteration",
    "purpose": "verified external backup after v25 H12/H15 risk-family acquisition run and docs before v26 diagnostic/protocol-freeze work",
    "remaining_changed_files": 0,
    "commit": "5051487a986750931cd2da0d90d7e3195e48d5ef",
    "changed_files": 353,
    "packages_this_run": [{
        "name": "20260930T020704_092ec6f5.tar.gz",
        "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260930T020704_092ec6f5.tar.gz",
        "id": 599790937,
        "sha256": "8ee39c30ae7fb66e2dc728d3d929a4346e7da17b800fa71a90f92ce5c802a3df",
        "bytes": 17286762,
        "verification": "github_server_sha256",
    }],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 147869,
    "notes": ["Local materialization of supervisor/user-supplied backup proof; this materialization and v26 outputs require the next backup before simulation."],
}

class ContractError(RuntimeError):
    pass

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)

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

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

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

def sf(v: Any, default: float = 0.0) -> float:
    try:
        y = float(v)
        return y if math.isfinite(y) else default
    except Exception:
        return default

def si(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def pct(x: Any) -> str:
    return "NA" if x is None else f"{100.0 * sf(x):.2f}%"

def quantile(vals: Sequence[float], q: float) -> float:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    p = q * (len(xs) - 1)
    lo = int(math.floor(p)); hi = int(math.ceil(p))
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)

def materialize_backup() -> None:
    if BACKUP_PROOF.exists():
        obj = read_json(BACKUP_PROOF)
        if obj.get("status") != "verified" or obj.get("backup_verified") is not True:
            raise ContractError("existing post-v25 backup proof is not verified")
        return
    write_json(BACKUP_PROOF, SUPERVISOR_BACKUP)

def normalize_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    h12 = r.get("h12") or {}
    h15 = r.get("h15") or {}
    return {
        "candidate_id": str(r.get("candidate_id") or r.get("base_state_id") or r.get("source_key") or "row"),
        "source_candidate_index": si(r.get("source_candidate_index"), -1),
        "role": r.get("role") or r.get("risk_probe_role") or r.get("slot_name") or "unknown",
        "h12_decision": sf(r.get("h12_decision_sum_s", h12.get("decision_sum_s"))),
        "h15_decision": sf(r.get("h15_decision_sum_s", h15.get("decision_sum_s"))),
        "h12_solver": sf(r.get("h12_solver_sum_s", h12.get("solver_sum_s"))),
        "h15_solver": sf(r.get("h15_solver_sum_s", h15.get("solver_sum_s"))),
        "h12_physical": sf(r.get("h12_physical", h12.get("physical"))),
        "h15_physical": sf(r.get("h15_physical", h15.get("physical"))),
        "phys_delta": sf(r.get("phys_delta_h12_minus_h15", r.get("physical_delta_h12_minus_h15"))),
        "tolerance": sf(r.get("row_physical_tolerance", r.get("row_tolerance_vs_H15", 2.0)), 2.0),
        "h12_bad": bool(r.get("h12_catastrophic_vs_h15", False)),
        "h12_beneficial": bool(r.get("h12_beneficial_vs_h15", False)),
    }

def rows_from_raw(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = read_json(path)
    ana = raw.get("analysis") or {}
    rows = ana.get("evaluation_rows")
    if not rows:
        rows = ((ana.get("label_analysis") or {}).get("state_rows") or [])
    return [normalize_row(r) for r in rows]

def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    h12_dec = math.fsum(sf(r.get("h12_decision")) for r in rows)
    h15_dec = math.fsum(sf(r.get("h15_decision")) for r in rows)
    h12_sol = math.fsum(sf(r.get("h12_solver")) for r in rows)
    h15_sol = math.fsum(sf(r.get("h15_solver")) for r in rows)
    h12_phys = math.fsum(sf(r.get("h12_physical")) for r in rows)
    h15_phys = math.fsum(sf(r.get("h15_physical")) for r in rows)
    tol = math.fsum(sf(r.get("tolerance"), 2.0) for r in rows)
    bad = sum(1 for r in rows if r.get("h12_bad"))
    ben = sum(1 for r in rows if r.get("h12_beneficial"))
    dec_save = (1.0 - h12_dec / h15_dec) if h15_dec > 0 else None
    sol_save = (1.0 - h12_sol / h15_sol) if h15_sol > 0 else None
    phys_delta = h12_phys - h15_phys
    return {
        "rows": n, "h12_beneficial": ben, "h12_bad": bad,
        "h12_decision_sum_s": h12_dec, "h15_decision_sum_s": h15_dec,
        "decision_saving_vs_H15": dec_save,
        "h12_solver_sum_s": h12_sol, "h15_solver_sum_s": h15_sol,
        "solver_saving_vs_H15": sol_save,
        "h12_physical_sum": h12_phys, "h15_physical_sum": h15_phys,
        "physical_delta_H12_minus_H15": phys_delta, "physical_tolerance_sum": tol,
        "physical_gate": phys_delta <= tol,
        "pass5": (bad == 0 and phys_delta <= tol and dec_save is not None and dec_save >= 0.05),
    }

def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> Optional[float]:
    if n <= 0:
        return None
    ph = k / float(n)
    den = 1.0 + z * z / n
    centre = ph + z * z / (2 * n)
    rad = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return min(1.0, (centre + rad) / den)

def collect_indices(obj: Any) -> List[int]:
    out: List[int] = []
    if isinstance(obj, Mapping):
        if "source_candidate_index" in obj:
            x = si(obj.get("source_candidate_index"), -1)
            if x >= 0:
                out.append(x)
        for v in obj.values():
            out.extend(collect_indices(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_indices(v))
    return out

def find_bank_path() -> Path:
    proto = read_json(V25_PROTOCOL)
    for key in (proto.get("input_hashes") or {}).keys():
        p = ROOT / key
        if not p.exists() or p.suffix.lower() != ".json":
            continue
        try:
            obj = read_json(p)
        except Exception:
            continue
        if isinstance(obj, Mapping) and isinstance(obj.get("candidate_cases"), list) and isinstance((obj.get("selection") or {}).get("all_candidate_metadata"), list):
            return p
    raise ContractError("could not find stress-bank candidate JSON from v25 protocol input hashes")

def dist(a: Mapping[str, Any], b: Mapping[str, Any], scales: Mapping[str, float]) -> float:
    keys = ["theta_r", "abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]
    return math.sqrt(math.fsum(((sf(a.get(k)) - sf(b.get(k))) / max(scales.get(k, 1.0), 1e-9)) ** 2 for k in keys) / len(keys))

def select_v26_cases() -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    bank_path = find_bank_path()
    bank = read_json(bank_path)
    metas = list((bank.get("selection") or {}).get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) < TARGET_CASES:
        raise ContractError("invalid stress bank dimensions")
    prior_paths = [V19_RAW, V21_RAW, V23_RAW, V25_RAW, V25_PROTOCOL,
                   ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_preoutcome_frozen_20260930T0130Z.json",
                   ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_preoutcome_frozen_20260930T0145Z.json"]
    excluded: set = set()
    for p in prior_paths:
        if p.exists():
            try:
                excluded.update(collect_indices(read_json(p)))
            except Exception:
                pass
    vals = {k: [sf(r.get(k)) for r in metas] for k in ["theta_r", "abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]}
    q33 = quantile(vals["stress_v1_score"], 0.33); q66 = quantile(vals["stress_v1_score"], 0.66)
    clear25 = quantile(vals["min_reference_obstacle_clearance"], 0.25)
    head66 = quantile(vals["abs_theta_r"], 0.66); traj_med = quantile(vals["traj_steps"], 0.50)
    scales = {k: max(max(vs) - min(vs), 1e-9) for k, vs in vals.items() if vs}
    by_idx = {si(r.get("candidate_index"), -1): r for r in metas}
    eligible = []
    for r0 in metas:
        r = dict(r0); idx = si(r.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases) or idx in excluded:
            continue
        stress = sf(r.get("stress_v1_score")); clear = sf(r.get("min_reference_obstacle_clearance")); head = sf(r.get("abs_theta_r")); traj = sf(r.get("traj_steps"))
        tags = []
        if stress <= q33: tags.append("lower_stress")
        if clear <= clear25: tags.append("low_clearance")
        if head >= head66 and traj >= traj_med: tags.append("high_heading_long")
        if head >= head66 and traj < traj_med: tags.append("high_heading_short")
        if not tags: tags.append("global")
        r["v26_tags"] = tags
        eligible.append(r)
    if len(eligible) < TARGET_CASES:
        raise ContractError("too few unused source candidates for v26")
    roles = [
        ("lower_stress_case72_E", "lower_stress", "near", 72),
        ("lower_stress_case72_F", "lower_stress", "near", 72),
        ("lower_stress_diverse", "lower_stress", "farthest", 72),
        ("low_clearance_case190_C", "low_clearance", "near", 190),
        ("low_clearance_case134_C", "low_clearance", "near", 134),
        ("low_clearance_extreme", "low_clearance", "extreme_clearance", None),
        ("high_heading_long_case74_C", "high_heading_long", "near", 74),
        ("high_heading_long_case108_C", "high_heading_long", "near", 108),
        ("high_heading_long_highstress", "high_heading_long", "high_stress", None),
        ("high_heading_short_case242_C", "high_heading_short", "near", 242),
        ("high_heading_short_highstress", "high_heading_short", "high_stress", None),
        ("global_diverse_remaining", "global", "diverse", None),
    ]
    used: set = set(); selected: List[Dict[str, Any]] = []; role_diag: List[Dict[str, Any]] = []
    selected_refs: List[Mapping[str, Any]] = []
    def pool(tag: str) -> List[Mapping[str, Any]]:
        p = [r for r in eligible if si(r.get("candidate_index"), -1) not in used and tag in (r.get("v26_tags") or [])]
        return p or [r for r in eligible if si(r.get("candidate_index"), -1) not in used]
    for role, tag, mode, anchor_idx in roles:
        p = pool(tag)
        anchor = by_idx.get(anchor_idx) if anchor_idx is not None else None
        if mode == "near" and anchor is not None:
            ordered = sorted(p, key=lambda r: (dist(r, anchor, scales), si(r.get("candidate_index"), 9999)))
        elif mode == "farthest" and anchor is not None:
            ordered = sorted(p, key=lambda r: (-dist(r, anchor, scales), si(r.get("candidate_index"), 9999)))
        elif mode == "extreme_clearance":
            ordered = sorted(p, key=lambda r: (sf(r.get("min_reference_obstacle_clearance")), -sf(r.get("stress_v1_score")), si(r.get("candidate_index"), 9999)))
        elif mode == "high_stress":
            ordered = sorted(p, key=lambda r: (-sf(r.get("stress_v1_score")), -sf(r.get("abs_theta_r")), si(r.get("candidate_index"), 9999)))
        else:
            refs = selected_refs or [by_idx.get(72, metas[0]), by_idx.get(190, metas[0]), by_idx.get(242, metas[0])]
            ordered = sorted(p, key=lambda r: (-min(dist(r, rr, scales) for rr in refs), si(r.get("candidate_index"), 9999)))
        ch = dict(ordered[0]); idx = si(ch.get("candidate_index"), -1); used.add(idx); selected_refs.append(ch)
        case_snapshot = cases[idx]
        selected.append({
            "fresh_case_index": len(selected), "source_candidate_index": idx,
            "role": role, "selection_tag": tag, "selection_mode": mode,
            "anchor_candidate_index": anchor_idx,
            "anchor_distance": dist(ch, anchor, scales) if anchor is not None else None,
            "theta_r": sf(ch.get("theta_r")), "abs_theta_r": sf(ch.get("abs_theta_r")),
            "traj_steps": sf(ch.get("traj_steps")), "min_reference_obstacle_clearance": sf(ch.get("min_reference_obstacle_clearance")),
            "stress_v1_score": sf(ch.get("stress_v1_score")), "v26_tags": ch.get("v26_tags"),
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(case_snapshot), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "case_snapshot_from_candidate_pool": case_snapshot,
            "selection_rule": "v26 metadata-only fixed-H12-primary source-independent confirmation; selected after v25 backup and before any v26 H12/H15 outcome",
        })
        role_diag.append({"role": role, "tag": tag, "mode": mode, "pool_size": len(p), "selected_source_candidate_index": idx})
    diag = {"bank_path": rel(bank_path), "excluded_source_candidate_indices_count": len(excluded), "excluded_source_candidate_indices": sorted(excluded), "eligible_unused_count": len(eligible), "thresholds": {"stress_q33": q33, "stress_q66": q66, "clearance_q25": clear25, "abs_heading_q66": head66, "traj_median": traj_med}, "role_diagnostics": role_diag, "selected_source_candidate_indices": [c["source_candidate_index"] for c in selected]}
    return bank_path, selected, diag

def policy_extract(raw_path: Path) -> Dict[str, Any]:
    if not raw_path.exists():
        return {}
    pol = ((read_json(raw_path).get("analysis") or {}).get("policies") or {})
    out = {}
    for name in ["fixed_H12", "selector_preoutcome_actual_overhead", "oracle_H12_H15"]:
        p = pol.get(name)
        if isinstance(p, Mapping):
            out[name] = {"decision_saving_vs_H15": p.get("decision_relative_saving_vs_H15"), "solver_saving_vs_H15": p.get("solver_relative_saving_vs_H15"), "bad": p.get("catastrophic_false_positive_count"), "pass5": p.get("pass5_zero_cat_physical"), "chosen_counts": p.get("chosen_counts")}
    return out

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
        csv.writer(f).writerow([created, NAME, "development_fixed_H12_primary_offline_audit_and_protocol_freeze_no_sim", "metadata_only", "development_no_validation64_no_test", "0", "0", "0", "0", "0", "False", rel(RUN_DIR / "completed.json"), MARKER])

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-and-freeze", action="store_true", required=True)
    ap.add_argument("--i-accept-v26-diagnostic", action="store_true", required=True)
    args = ap.parse_args(argv)
    del args
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    materialize_backup()
    created = now_utc()
    for p in [V21_RAW, V23_RAW, V25_RAW, V25_DONE, V25_PROTOCOL, BACKUP_PROOF]:
        if not p.exists():
            raise ContractError("missing required v26 input: " + rel(p))
    datasets = {
        "v19_opened_boundary": rows_from_raw(V19_RAW),
        "v21_source_independent": rows_from_raw(V21_RAW),
        "v23_broader_source_independent": rows_from_raw(V23_RAW),
        "v25_source72_risk_family": rows_from_raw(V25_RAW),
    }
    fresh_names = ["v21_source_independent", "v23_broader_source_independent", "v25_source72_risk_family"]
    fresh_rows = [r for name in fresh_names for r in datasets[name]]
    combined_names = ["v19_opened_boundary"] + fresh_names
    combined_rows = [r for name in combined_names for r in datasets[name]]
    aggregates = {name: aggregate(rows) for name, rows in datasets.items()}
    aggregates["fresh_v21_v23_v25"] = aggregate(fresh_rows)
    aggregates["all_opened_h12_h15_rows"] = aggregate(combined_rows)
    fresh_bad = aggregates["fresh_v21_v23_v25"]["h12_bad"]; fresh_n = aggregates["fresh_v21_v23_v25"]["rows"]
    risk_bounds = {"fresh_h12_bad_count": fresh_bad, "fresh_rows": fresh_n, "wilson_95_upper_bad_rate": wilson_upper(fresh_bad, fresh_n), "zero_failure_rule_of_three_upper": (3.0 / fresh_n if fresh_n else None)}
    bank_path, selected_cases, case_diag = select_v26_cases()
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_fixed_H12_primary_confirmation_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_fixed_H12_primary_source_independent_confirmation_protocol_no_sim_no_validation_no_test",
        "primary_question": "Does fixed true H12 remain safe and materially faster than H15 on a new source-independent stress-bank batch, and is there any residual adaptive H12/H15 value beyond fixed H12?",
        "before_evidence": aggregates,
        "risk_bounds": risk_bounds,
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "budget_declared": {"stage_A_h15_trace_episodes": TARGET_CASES, "stage_B_branch_episodes": TARGET_CASES * BRANCH_STATES_PER_CASE * len(TRUE_HORIZONS), "total_episodes_exact": TOTAL_EPISODES, "control_step_upper_bound": CONTROL_STEP_CAP, "new_gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "stage_A_rule": "Run H15 trace only, then choose two branch states per case from predeclared H15-trace risk/opportunity windows before any v26 H12 branch outcome.",
        "stage_B_rule": "Blocked/randomized true-H12 and true-H15 branch rollouts with shared H15 terminal; fixed H12 is primary comparator, selector/oracle H12/H15 are secondary diagnostics charged with measured overhead.",
        "decision_rules": {"fixed_H12_passes": "treat fixed H12 as stronger simple baseline; do not validate adaptive selector on this distribution without scenario redesign/adaptive-opportunity evidence", "fixed_H12_fails_selector_or_oracle_passes": "adaptive mechanism has renewed value; inspect failure morphology and consider terminal-risk/value refit before validation", "fixed_H12_and_selector_fail": "do not validate; pivot to scenario/reward/terminal/training diagnosis"},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "backup_proof_used": {"path": rel(BACKUP_PROOF), "sha256": sha256(BACKUP_PROOF), "commit": read_json(BACKUP_PROOF).get("commit")},
        "input_hashes": {rel(p): sha256(p) for p in [SOURCE, V19_RAW, V21_RAW, V23_RAW, V25_RAW, V25_DONE, V25_PROTOCOL, V24_SUMMARY, bank_path, BACKUP_PROOF] if p.exists()},
    }
    write_json(PROTOCOL, protocol)
    policies = {"v23": policy_extract(V23_RAW), "v25": policy_extract(V25_RAW)}
    raw = {"created_utc": created.isoformat(), "classification": protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulation_episodes": 0, "new_control_steps": 0, "new_gradient_steps": 0, "aggregates": aggregates, "risk_bounds": risk_bounds, "selector_policy_extracts": policies, "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}, "selected_source_candidate_indices": case_diag["selected_source_candidate_indices"], "backup_proof_materialized": rel(BACKUP_PROOF), "backup_request": rel(BACKUP_REQUEST)}
    write_json(RUN_DIR / "raw.json", raw)
    lines = [
        "# Vehicle fixed-H12-primary diagnostic v26",
        "",
        f"UTC: `{created.isoformat()}`. Offline audit + protocol freeze only; no simulation, no validation64, no sealed test, no training/refit.",
        "",
        "## Fixed-H12 evidence from opened development H12/H15 branch rows",
        "",
        "| dataset | rows | H12 beneficial | H12 bad | fixed-H12 decision saving | fixed-H12 solver saving | phys Δ H12-H15 | pass5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["v19_opened_boundary", "v21_source_independent", "v23_broader_source_independent", "v25_source72_risk_family", "fresh_v21_v23_v25", "all_opened_h12_h15_rows"]:
        a = aggregates[name]
        lines.append(f"| `{name}` | {a['rows']} | {a['h12_beneficial']} | {a['h12_bad']} | {pct(a['decision_saving_vs_H15'])} | {pct(a['solver_saving_vs_H15'])} | {a['physical_delta_H12_minus_H15']:.6g} | `{a['pass5']}` |")
    lines += [
        "",
        f"Fresh v21+v23+v25 rows have `{fresh_bad}/{fresh_n}` H12-bad labels; Wilson 95% upper bad-rate bound is `{pct(risk_bounds['wilson_95_upper_bad_rate'])}` (rule-of-three `{pct(risk_bounds['zero_failure_rule_of_three_upper'])}`), so zero observed failures is not a population guarantee.",
        "",
        "## Selector comparison note",
        "",
        f"v23/v25 preoutcome selectors remained safe but selected H15 on many rows where H12 was beneficial. Therefore current evidence supports fixed-H12-primary comparison before any adaptive validation. Policy extracts: `{json.dumps(clean(policies), sort_keys=True)}`.",
        "",
        "## Frozen next diagnostic protocol",
        "",
        f"Protocol: `{rel(PROTOCOL)}` SHA256 `{sha256(PROTOCOL)}`.",
        f"Selected unused source_candidate_index values: `{case_diag['selected_source_candidate_indices']}`.",
        f"Planned future budget after backup: `{TOTAL_EPISODES}` development episodes, cap `{CONTROL_STEP_CAP}` control steps, validation64/test episodes `0`.",
        "",
        "Decision: run the bounded v26 fixed-H12-primary confirmation only after backing up this protocol/source. If fixed H12 remains safe and faster, pivot to scenario-opportunity redesign or a credible negative adaptive-opportunity conclusion for this stress distribution; if fixed H12 failures recur, inspect telemetry and then run terminal-risk/value refit or bounded training rather than another static sweep.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(BACKUP_REQUEST, {"request": "backup_after_v26_fixed_h12_primary_diagnostic_protocol_freeze", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v26 source/offline audit/protocol/docs/state/response log plus materialized post-v25 backup proof", "must_cover": [rel(SOURCE), rel(RUN_DIR), rel(PROTOCOL), rel(STATE), rel(BACKUP_PROOF), rel(BACKUP_REQUEST), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"], "new_simulation_episodes": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle fixed-H12-primary diagnostic v26

UTC: {created.isoformat()}. Offline audit and protocol freeze only; no simulation, no validation64, no sealed test, no training/refit. Fresh v21+v23+v25 H12/H15 rows: n={fresh_n}, H12-bad={fresh_bad}, fixed-H12 decision saving={aggregates['fresh_v21_v23_v25']['decision_saving_vs_H15']}, pass5={aggregates['fresh_v21_v23_v25']['pass5']}. v19 remains opened counterevidence with H12-bad={aggregates['v19_opened_boundary']['h12_bad']}. Froze v26 fixed-H12-primary source-independent confirmation protocol `{rel(PROTOCOL)}` selecting source_candidate_index values {case_diag['selected_source_candidate_indices']} for a future {TOTAL_EPISODES}-episode development run after backup. Backup request `{rel(BACKUP_REQUEST)}`.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / name, MARKER, block)
    response = f"""
## Follow-up through v26 fixed-H12-primary diagnostic/protocol freeze

Updated by GPT-5.5 executor at `{created.isoformat()}`. v26 is an offline audit/protocol freeze only; validation64 and sealed test remain closed.

| linked recommendation(s) | disposition after v26 diagnostic | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; acted | Fresh v21+v23+v25 aggregate fixed H12: rows={fresh_n}, bad={fresh_bad}, decision saving={pct(aggregates['fresh_v21_v23_v25']['decision_saving_vs_H15'])}, pass5={aggregates['fresh_v21_v23_v25']['pass5']}; v19 opened boundary still has bad={aggregates['v19_opened_boundary']['h12_bad']}. | Froze fixed-H12-primary v26 confirmation protocol `{rel(PROTOCOL)}`; fixed H12 is primary, selector/oracle are secondary. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; quantified | Fresh zero-bad count is {fresh_bad}/{fresh_n}; Wilson 95% upper bad-rate bound {pct(risk_bounds['wilson_95_upper_bad_rate'])}. | Do not treat zero fresh bad as population proof; v26 remains development/stress-pool confirmation, not validation/test. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred with falsifiable gate | Current fresh evidence favors fixed H12, while adaptive value only appears in opened v19. | If v26 fixed H12 fails or selector/oracle uniquely helps, pivot to terminal-risk/value refit or bounded training; if fixed H12 passes, prioritize scenario-opportunity redesign/negative adaptive-opportunity conclusion. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; still open | v26 ran no deployment timing; v23/v25 selector extracts remain development only. | Future v26 run must charge measured selector overhead and full whole-decision timing; no final speed claim. |
| `A12_registry_backup_schema_contract` | accepted; active | Materialized post-v25 verified backup proof `{rel(BACKUP_PROOF)}` and wrote v26 backup request `{rel(BACKUP_REQUEST)}`. | Require verified backup covering v26 source/protocol/results/docs before simulations. |
"""
    append_if_missing(ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", MARKER, response)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v26 fixed-H12-primary diagnostic\n\nUTC: {created.isoformat()}\n\nNo simulations, no validation64, no sealed test. Fresh v21+v23+v25 rows: n={fresh_n}, H12_bad={fresh_bad}, fixed_H12_save={pct(aggregates['fresh_v21_v23_v25']['decision_saving_vs_H15'])}, pass5={aggregates['fresh_v21_v23_v25']['pass5']}. v26 protocol frozen at `{rel(PROTOCOL)}` selecting {case_diag['selected_source_candidate_indices']}. Next action after verified backup: implement/run the bounded v26 fixed-H12-primary development confirmation, or if compute/backup blocks arise preserve this as the scenario/comparison diagnosis.\n", encoding="utf-8")
    append_registry(created.isoformat())
    completed = {"status": "complete", "passed": True, "created_utc": created.isoformat(), "classification": protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulation_episodes": 0, "new_control_steps": 0, "new_gradient_steps": 0, "summary": rel(RUN_DIR / "summary.md"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST), "hashes": {}}
    files = [SOURCE, RUN_DIR / "raw.json", RUN_DIR / "summary.md", PROTOCOL, STATE, BACKUP_PROOF, BACKUP_REQUEST, ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md", ROOT / "STATUS.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
    completed["hashes"] = {rel(p): sha256(p) for p in files if p.exists()}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "protocol": rel(PROTOCOL), "fresh_rows": fresh_n, "fresh_h12_bad": fresh_bad, "fresh_fixed_h12_decision_saving": aggregates["fresh_v21_v23_v25"]["decision_saving_vs_H15"], "selected_source_candidate_indices": case_diag["selected_source_candidate_indices"], "backup_request": rel(BACKUP_REQUEST), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
