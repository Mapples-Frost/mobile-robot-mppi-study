#!/usr/bin/env python3
"""v15 boundary-centre true-H continuation smoke.

Development-only IMPROVED diagnostic following v14/v15-preflight evidence.
It executes only the two v14 nested false-positive centre states, each with
true n_horizon=10 and true n_horizon=15 under the shared-H15 terminal profile.

Purpose: before spending the full 96-episode boundary-acquisition budget, verify
that the saved H15-prefix branch states can be replayed through the true-variable
H runner and that the two false-positive centre labels remain interpretable
under the current continuation code/timing instrumentation.

No validation64 or sealed-test access. No selector refit/search. No gradient
training. Budget: 4 development MPC continuation episodes, <=600 control steps.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_case5_smoke_v0_runner as case_runner  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_v15_boundary_smoke_v0"
STAMP = "20260929T2245Z"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

AUDIT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/audit.json"
AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/completed.json"
PREFLIGHT = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/preflight.json"
PREFLIGHT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/completed.json"
V8C_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/raw.json"
V8C_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json"
V11_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/raw.json"
V11_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/completed.json"

RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2245_after_v15_boundary_smoke.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V15_BOUNDARY_SMOKE_V0_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-v15-boundary-smoke-v0-{STAMP}"

TRUE_HORIZONS = [10, 15]
MAX_STEPS = 150
EPISODES_EXACT = 4
CONTROL_STEP_CAP = EPISODES_EXACT * MAX_STEPS
PRIMARY_TERMINAL_PROFILE = "shared_h15_terminal"


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
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


def completed_dev_only(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError("prerequisite not complete: " + rel(path))
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {label}")
    return obj


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def recursive_case_snapshots(obj: Any, source_idx: int, out: List[Mapping[str, Any]]) -> None:
    if isinstance(obj, Mapping):
        if si(obj.get("source_candidate_index"), -999999) == source_idx and isinstance(obj.get("case_snapshot_from_candidate_pool"), Mapping):
            out.append(obj)
        for v in obj.values():
            if isinstance(v, (Mapping, list)):
                recursive_case_snapshots(v, source_idx, out)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (Mapping, list)):
                recursive_case_snapshots(v, source_idx, out)


def find_case_snapshot(bank_id: str, source_idx: int, raw_by_bank: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    raw = raw_by_bank[bank_id]
    matches: List[Mapping[str, Any]] = []
    recursive_case_snapshots(raw, source_idx, matches)
    if not matches:
        raise ContractError(f"no case_snapshot_from_candidate_pool for {bank_id} source_candidate_index={source_idx}")
    # Prefer top-level selected_cases rows if present; otherwise deterministic shortest canonical object.
    selected = [m for m in matches if "fresh_case_index" in m and "case_snapshot_sha256" in m]
    chosen = selected[0] if selected else sorted(matches, key=lambda m: len(json.dumps(clean(m), sort_keys=True)))[0]
    case = chosen.get("case_snapshot_from_candidate_pool")
    if not isinstance(case, Mapping) or not isinstance(case.get("tvp"), Mapping):
        raise ContractError(f"case snapshot lacks tvp for {bank_id} source_candidate_index={source_idx}")
    return case


def trace_row_at(trace_dir_rel: str, step: int) -> Mapping[str, Any]:
    trace_path = ROOT / trace_dir_rel / "trace.json"
    if not trace_path.exists():
        raise ContractError("missing trace.json: " + rel(trace_path))
    trace = read_json(trace_path)
    if not isinstance(trace, list):
        raise ContractError("trace.json is not a list: " + rel(trace_path))
    exact = [r for r in trace if isinstance(r, Mapping) and si(r.get("step"), -1) == int(step)]
    if not exact:
        if 0 <= int(step) < len(trace) and isinstance(trace[int(step)], Mapping):
            exact = [trace[int(step)]]
        else:
            raise ContractError(f"trace {rel(trace_path)} lacks step {step}")
    return exact[0]


def state_core(state: Any) -> Dict[str, float]:
    if not isinstance(state, Mapping):
        raise ContractError("branch state is not a mapping")
    out: Dict[str, float] = {}
    for k in ("x", "y", "theta"):
        val = state.get(k)
        if isinstance(val, list):
            val = val[0] if val else 0.0
        out[k] = float(val)
    return out


def safe_id(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:110]


def metric_sum(e: Mapping[str, Any], name: str) -> float:
    return sf((e.get(name) or {}).get("sum"), 0.0)


def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 999) == 0 and si(e.get("initial_failed_steps"), 999) == 0 and si(e.get("final_failed_steps"), 999) == 0


def verify_inputs() -> Dict[str, Any]:
    done = {
        "audit": completed_dev_only(AUDIT_DONE, "v14 audit done"),
        "preflight": completed_dev_only(PREFLIGHT_DONE, "v15 preflight done"),
        "v8c": completed_dev_only(V8C_DONE, "v8c done"),
        "v11": completed_dev_only(V11_DONE, "v11 done"),
    }
    audit = read_json(AUDIT)
    preflight = read_json(PREFLIGHT)
    for label, obj in (("audit", audit), ("preflight", preflight)):
        for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
            if obj.get(flag) is True:
                raise ContractError(f"forbidden {flag}=True in {label}")
    h = preflight.get("headline") or {}
    if h.get("blocked_candidate_count") not in (0, 0.0) or h.get("trace_step_coverage_count") != h.get("unique_trace_count"):
        raise ContractError("v15 preflight did not establish full trace coverage")
    candidates = list(((audit.get("v15_candidate_plan") or {}).get("candidate_branch_states") or []))
    centers = [c for c in candidates if c.get("role") == "false_positive_center" and si(c.get("offset_from_center"), 999) == 0]
    if len(centers) != 2:
        raise ContractError(f"expected exactly two false-positive centers, got {len(centers)}")
    raw_by_bank = {"fresh_v8c": read_json(V8C_RAW), "fresh_v11": read_json(V11_RAW)}
    return {"done": done, "audit": audit, "preflight": preflight, "centers": centers, "raw_by_bank": raw_by_bank}


def prepare_centers(centers: Sequence[Mapping[str, Any]], raw_by_bank: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for i, c in enumerate(sorted(centers, key=lambda x: str(x.get("source_key")))):
        bank_id = str(c.get("bank_id"))
        if bank_id not in raw_by_bank:
            raise ContractError("unsupported bank for smoke: " + bank_id)
        source_idx = si(c.get("source_key", "").split("source")[-1], -1)
        if source_idx < 0:
            # The audit candidate itself does not carry source_candidate_index, but source_key plus trace path are enough.
            # Recover it from the stage trace summary path naming convention when needed.
            trace = str(c.get("h15_trace_episode_path"))
            import re
            m = re.search(r"source(\d+)", trace)
            if not m:
                raise ContractError("cannot infer source_candidate_index from " + trace)
            source_idx = int(m.group(1))
        row = trace_row_at(str(c.get("h15_trace_episode_path")), si(c.get("candidate_branch_step"), -1))
        branch_state = row.get("previous_state") or row.get("state")
        obs = row.get("observation") or row.get("next_observation") or []
        case = find_case_snapshot(bank_id, source_idx, raw_by_bank)
        key = str(c.get("source_key"))
        prepared.append({
            "center_index": i,
            "center_key": key,
            "bank_id": bank_id,
            "source_candidate_index": source_idx,
            "base_state_id": str(c.get("base_state_id")),
            "source_group": c.get("source_group"),
            "source_window": c.get("source_window"),
            "candidate_branch_step": si(c.get("candidate_branch_step"), -1),
            "center_branch_step": si(c.get("center_branch_step"), -1),
            "h15_trace_episode_path": str(c.get("h15_trace_episode_path")),
            "trace_row_step": si(row.get("step"), -1),
            "branch_previous_state": state_core(branch_state),
            "initial_observation_from_h15_trace": copy.deepcopy(obs),
            "case_snapshot_sha256": canonical_sha(case),
            "case_snapshot_from_candidate_pool": case,
            "source_label_positive": bool(c.get("source_label_positive")),
            "source_catastrophic": bool(c.get("source_catastrophic")),
            "source_phys_delta": sf(c.get("source_phys_delta"), 0.0),
            "source_decision_gain_s": sf(c.get("source_decision_gain_s"), 0.0),
        })
    return prepared


def build_protocol(created: dt.datetime, prepared: Sequence[Mapping[str, Any]], backup_commit: str, input_hashes: Mapping[str, str]) -> Dict[str, Any]:
    schedule: List[Dict[str, Any]] = []
    exe = 0
    for i, center in enumerate(prepared):
        # Block within centre while alternating order across centres to reduce monotonic timing drift.
        order = [10, 15] if i % 2 == 0 else [15, 10]
        for h in order:
            schedule.append({
                "execution_index": exe,
                "center_key": center["center_key"],
                "state_id": safe_id(f"v15fp{i}_{center['bank_id']}_{center['base_state_id']}_step{int(center['candidate_branch_step']):03d}"),
                "case": int(center["center_index"]),
                "source_candidate_index": int(center["source_candidate_index"]),
                "branch_step": int(center["candidate_branch_step"]),
                "true_mpc_n_horizon": int(h),
                "commanded_horizon": int(h),
                "terminal_mode": PRIMARY_TERMINAL_PROFILE,
                "initialization": "v15_boundary_smoke_direct_branch_state_from_saved_H15_prefix_trace",
            })
            exe += 1
    if len(schedule) != EPISODES_EXACT:
        raise ContractError("unexpected v15 smoke schedule length")
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_H_boundary_center_smoke_preoutcome_not_validation_not_test",
        "hypothesis": "The two v14 nested false-positive centre states are reproducible risk-boundary continuations under the current true-variable-H runner. A four-episode smoke should confirm replay fidelity and whether H10 remains catastrophic versus H15 before the full 96-episode boundary acquisition is run.",
        "before_evidence": [
            "v14 nested calibrated risk/value selector selected two H10 false positives on opened development rows: fresh_v8c/fresh_case02_slot1_mid_late_control and fresh_v11/fresh_case05_slot1_mid_late_control.",
            "v15 preflight v0b found 24 candidate branch states across 8 existing H15-prefix traces with complete parsed step coverage and recommended a 4-episode centre smoke first.",
        ],
        "centers": [{k: v for k, v in c.items() if k != "case_snapshot_from_candidate_pool"} for c in prepared],
        "schedule": schedule,
        "terminal_profile": PRIMARY_TERMINAL_PROFILE,
        "terminal_source_horizon": 15,
        "budget_declared": {"development_mpc_simulation_episodes": EPISODES_EXACT, "development_control_step_upper_bound": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rules": {
            "smoke_success": "both centers replay with H15 safe, H10/H15 pairs present, reset distance within inherited runner tolerance, and interpretable physical/timing rows",
            "risk_reproduced": "both centers remain catastrophic H10 versus H15 using threshold max(2, 0.25*abs(H15 physical)); next run full boundary acquisition around centers/lookalikes",
            "risk_not_reproduced": "do not expand immediately; inspect replay mismatch, timing/noise and source-run differences before any label/refit claims",
        },
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_source_write_from_supervisor_context": backup_commit,
        "input_hashes": input_hashes,
    }
    write_json(PROTOCOL, proto)
    return proto


def summarize_pair(h10: Mapping[str, Any], h15: Mapping[str, Any], center: Mapping[str, Any]) -> Dict[str, Any]:
    h10_phys = sf(h10.get("physical_constraint_cost"), 0.0)
    h15_phys = sf(h15.get("physical_constraint_cost"), 0.0)
    h10_dec = metric_sum(h10, "decision_timing_s")
    h15_dec = metric_sum(h15, "decision_timing_s")
    h10_solver = metric_sum(h10, "solver_attempt_timing_s")
    h15_solver = metric_sum(h15, "solver_attempt_timing_s")
    phys_delta = h10_phys - h15_phys
    decision_delta = h10_dec - h15_dec
    solver_delta = h10_solver - h15_solver
    row_tol = max(2.0, 0.05 * abs(h15_phys))
    catastrophic_threshold = max(2.0, 0.25 * abs(h15_phys))
    h15_safe = is_safe(h15)
    h10_safe = is_safe(h10)
    catastrophic = bool(h15_safe and ((not h10_safe) or phys_delta > catastrophic_threshold))
    beneficial = bool(h15_safe and h10_safe and phys_delta <= row_tol and decision_delta < 0.0)
    return {
        "center_key": center["center_key"],
        "bank_id": center["bank_id"],
        "base_state_id": center["base_state_id"],
        "source_group": center.get("source_group"),
        "source_window": center.get("source_window"),
        "branch_step": center["candidate_branch_step"],
        "h10_safe": h10_safe,
        "h15_safe": h15_safe,
        "h10_success": bool(h10.get("success")),
        "h15_success": bool(h15.get("success")),
        "h10_steps": si(h10.get("steps"), 0),
        "h15_steps": si(h15.get("steps"), 0),
        "h10_opt_x_sizes": h10.get("opt_x_sizes_observed"),
        "h15_opt_x_sizes": h15.get("opt_x_sizes_observed"),
        "h10_physical": h10_phys,
        "h15_physical": h15_phys,
        "physical_delta_h10_minus_h15": phys_delta,
        "row_tolerance_vs_H15": row_tol,
        "catastrophic_threshold": catastrophic_threshold,
        "h10_catastrophic_vs_h15": catastrophic,
        "h10_beneficial_vs_h15": beneficial,
        "h10_decision_sum_s": h10_dec,
        "h15_decision_sum_s": h15_dec,
        "decision_delta_h10_minus_h15": decision_delta,
        "decision_gain_h10_vs_h15_s": -decision_delta,
        "h10_solver_sum_s": h10_solver,
        "h15_solver_sum_s": h15_solver,
        "solver_delta_h10_minus_h15": solver_delta,
        "solver_gain_h10_vs_h15_s": -solver_delta,
        "source_expected_catastrophic": bool(center.get("source_catastrophic")),
        "source_expected_positive": bool(center.get("source_label_positive")),
        "source_phys_delta": center.get("source_phys_delta"),
        "source_decision_gain_s": center.get("source_decision_gain_s"),
        "h10_path": h10.get("path"),
        "h15_path": h15.get("path"),
        "max_branch_reset_distance": max(sf((h10.get("branch_reset") or {}).get("branch_state_distance_after_reset"), 999.0), sf((h15.get("branch_reset") or {}).get("branch_state_distance_after_reset"), 999.0)),
    }


def analyze(episodes: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_center_h: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for e in episodes:
        by_center_h[(str(e.get("center_key")), si(e.get("true_mpc_n_horizon"), -1))] = e
    rows: List[Dict[str, Any]] = []
    for c in prepared:
        h10 = by_center_h.get((str(c["center_key"]), 10))
        h15 = by_center_h.get((str(c["center_key"]), 15))
        if h10 is None or h15 is None:
            rows.append({"center_key": c["center_key"], "missing_pair": True, "h10_catastrophic_vs_h15": None})
        else:
            rows.append(summarize_pair(h10, h15, c))
    fixed15_dec = math.fsum(sf(r.get("h15_decision_sum_s"), 0.0) for r in rows)
    fixed10_dec = math.fsum(sf(r.get("h10_decision_sum_s"), 0.0) for r in rows)
    fixed15_phys = math.fsum(sf(r.get("h15_physical"), 0.0) for r in rows)
    fixed10_phys = math.fsum(sf(r.get("h10_physical"), 0.0) for r in rows)
    cats = [r for r in rows if r.get("h10_catastrophic_vs_h15") is True]
    beneficial = [r for r in rows if r.get("h10_beneficial_vs_h15") is True]
    all_pairs = all(not r.get("missing_pair") for r in rows)
    all_h15_safe = all(r.get("h15_safe") is True for r in rows if not r.get("missing_pair"))
    if all_pairs and all_h15_safe and len(cats) == len(rows):
        decision = "v15 centre smoke reproduced both false-positive centre H10 catastrophes with valid H15 pairs; next freeze the full boundary acquisition around centres/lookalikes, then refit only if added labels improve zero-catastrophe selection."
    elif all_pairs and all_h15_safe and len(cats) == 0:
        decision = "v15 centre smoke did not reproduce the source catastrophic labels; inspect replay/noise/source-run mismatch before expanding boundary acquisition or refitting."
    elif all_pairs and all_h15_safe:
        decision = "v15 centre smoke is mixed; expand only the ambiguous centre with a repeat/offset mini-block before full acquisition."
    else:
        decision = "v15 centre smoke failed replay/safety completeness; inspect episode traces before any expanded acquisition."
    return {
        "pair_rows": rows,
        "aggregate": {
            "pair_count": len(rows),
            "all_pairs_present": all_pairs,
            "all_h15_safe": all_h15_safe,
            "catastrophic_h10_rows": len(cats),
            "beneficial_h10_rows": len(beneficial),
            "fixed_H15_physical_sum": fixed15_phys,
            "fixed_H10_physical_sum": fixed10_phys,
            "fixed_H15_decision_sum_s": fixed15_dec,
            "fixed_H10_decision_sum_s": fixed10_dec,
            "fixed_H10_decision_relative_saving_vs_H15": None if fixed15_dec <= 0 else (fixed15_dec - fixed10_dec) / fixed15_dec,
        },
        "decision": decision,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle true-variable-H v15 boundary-centre smoke v0",
        "",
        f"UTC `{raw['created_utc']}`. Development-only 4-episode smoke on the two v14 nested false-positive centres; no validation64, no sealed test, no selector refit/search, no gradient training.",
        "",
        "## Budget/access",
        "",
        f"- Episodes/control steps: `{raw['budget_actual']['development_mpc_simulation_episodes']}` / `{raw['budget_declared']['development_mpc_simulation_episodes']}` episodes; `{raw['budget_actual']['development_control_steps']}` / `{raw['budget_declared']['development_control_step_upper_bound']}` control steps.",
        f"- validation64_bank_opened: `{raw['validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Aggregate",
        "",
        f"- `{a['aggregate']}`",
        f"- Decision: {a['decision']}",
        "",
        "## Pair rows",
        "",
        "| center | H10 cat | H10 ben | H10 phys | H15 phys | physΔ | H10 dec | H15 dec | dec gain | H10 safe | H15 safe | opt H10/H15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in a["pair_rows"]:
        if r.get("missing_pair"):
            lines.append(f"| `{r.get('center_key')}` | MISSING |  |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append("| `%s` | `%s` | `%s` | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | `%s` | `%s` | `%s/%s` |" % (
            r.get("center_key"), r.get("h10_catastrophic_vs_h15"), r.get("h10_beneficial_vs_h15"),
            sf(r.get("h10_physical")), sf(r.get("h15_physical")), sf(r.get("physical_delta_h10_minus_h15")),
            sf(r.get("h10_decision_sum_s")), sf(r.get("h15_decision_sum_s")), sf(r.get("decision_gain_h10_vs_h15_s")),
            r.get("h10_safe"), r.get("h15_safe"), r.get("h10_opt_x_sizes"), r.get("h15_opt_x_sizes")))
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is a replay-fidelity and risk-boundary smoke on already-opened development states, not validation evidence. One repeat per horizon is not a timing conclusion. It decides whether the full v15 boundary acquisition is worth running and whether the source false-positive labels remain reproducible under the current true-variable-H continuation path.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`; completed: `{rel(RUN_DIR / 'completed.json')}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    ag = raw["analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v15 boundary-centre smoke v0

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only four-episode smoke on two v14 nested false-positive centres; no validation64/sealed-test access, no selector refit/search, no gradient training. Budget actual: {raw['budget_actual']['development_mpc_simulation_episodes']} episodes, {raw['budget_actual']['development_control_steps']} control steps. Aggregate: {ag}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_v15_boundary_center_smoke,opened_dev_false_positive_centers,{raw['budget_actual']['development_mpc_simulation_episodes']},0,0,0,0,False,{rel(RUN_DIR / 'completed.json')},{MARKER}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-v15-boundary-smoke", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v15_boundary_smoke:
        raise ContractError("requires --run and explicit development v15 boundary smoke acknowledgement")
    if (RUN_DIR / "completed.json").exists():
        done = read_json(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    prepared = prepare_centers(inputs["centers"], inputs["raw_by_bank"])
    input_paths = [SOURCE, AUDIT, AUDIT_DONE, PREFLIGHT, PREFLIGHT_DONE, V8C_RAW, V8C_DONE, V11_RAW, V11_DONE]
    protocol = build_protocol(now(), prepared, args.backup_verified_commit, {rel(p): sha256(p) for p in input_paths if p.exists()})
    # Legacy runtime/terminal loading after pre-outcome protocol is durably written.
    case_runner.SMOKE_DIR = RUN_DIR
    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    if 15 not in terminals:
        raise ContractError("terminal grid missing H15")
    write_json(RUN_DIR / "run_started.json", {"started_utc": now().isoformat(), "pid": os.getpid(), "method": NAME, "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "validation64_bank_opened": False, "sealed_test_accessed": False, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    center_by_key = {str(c["center_key"]): c for c in prepared}
    episodes: List[Dict[str, Any]] = []
    for item in protocol["schedule"]:
        center = center_by_key[str(item["center_key"])]
        run_item = dict(item)
        run_item["branch_previous_state"] = copy.deepcopy(center["branch_previous_state"])
        case = center["case_snapshot_from_candidate_pool"]
        summary = case_runner.run_true_h_episode(run_item, case, terminals[15])
        summary["stage"] = "v15_boundary_center_smoke_trueH10_H15"
        summary["center_key"] = item["center_key"]
        summary["bank_id"] = center["bank_id"]
        summary["base_state_id"] = center["base_state_id"]
        summary["source_group"] = center.get("source_group")
        summary["source_window"] = center.get("source_window")
        summary["source_catastrophic"] = center.get("source_catastrophic")
        summary["source_label_positive"] = center.get("source_label_positive")
        summary["source_phys_delta"] = center.get("source_phys_delta")
        summary["source_decision_gain_s"] = center.get("source_decision_gain_s")
        summary["terminal_source_horizon"] = 15
        summary["terminal_receipt_effective"] = terminal_receipts.get("15") or terminal_receipts.get(15)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": EPISODES_EXACT, "control_steps_done": int(sum(si(e.get("steps"), 0) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("center_key", "true_mpc_n_horizon", "steps", "success", "termination", "physical_constraint_cost")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(si(e.get("steps"), 0) for e in episodes))
    if len(episodes) != EPISODES_EXACT:
        raise ContractError("episode count mismatch")
    if control_steps > CONTROL_STEP_CAP:
        raise ContractError("control-step budget exceeded")
    analysis = analyze(episodes, prepared)
    created = now()
    write_json(BACKUP_REQUEST, {"requested_utc": created.isoformat(), "reason": "backup v15 boundary-centre smoke source/protocol/raw/docs before any full boundary acquisition", "backup_required_before_more_simulations": True, "development_mpc_simulation_episodes": len(episodes), "development_control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(RUN_DIR), rel(STATE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"]})
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_H_boundary_center_smoke_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "runtime_preflight": preflight,
        "terminal_sources": {str(k): v for k, v in terminal_receipts.items()},
        "centers_preoutcome": [{k: v for k, v in c.items() if k != "case_snapshot_from_candidate_pool"} for c in prepared],
        "episodes": episodes,
        "analysis": analysis,
        "budget_declared": {"development_mpc_simulation_episodes": EPISODES_EXACT, "development_control_step_upper_bound": CONTROL_STEP_CAP, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"development_mpc_simulation_episodes": len(episodes), "development_control_steps": control_steps, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "platform": {"python": sys.version, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "backup_request": rel(BACKUP_REQUEST),
        "interpretation_limits": ["opened development false-positive centers", "one repeat per horizon", "not validation/test", "no selector refit", "timing is logged but not sufficient for a final speed claim"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"""# Continue state after v15 boundary-centre smoke v0

UTC: {created.isoformat()}
Elapsed since first supervisor event: {(created - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Completed `{NAME}` development-only smoke: {len(episodes)} episodes, {control_steps} control steps. No validation64, no sealed test, no selector refit/search, no training.

Aggregate: {analysis['aggregate']}
Decision: {analysis['decision']}
Summary: `{rel(RUN_DIR / 'summary.md')}`
Raw: `{rel(RUN_DIR / 'raw.json')}`
Completed: `{rel(RUN_DIR / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action: externally back up source/protocol/raw/docs. If both centre catastrophes reproduced, freeze the full v15 boundary acquisition around centers/lookalikes; if not, inspect replay mismatch and do not expand blindly.
""", encoding="utf-8")
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, AUDIT, PREFLIGHT, V8C_RAW, V11_RAW]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_actual": raw["budget_actual"], "headline": analysis["aggregate"], "decision": analysis["decision"], "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST), "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
    write_json(RUN_DIR / "completed.json", completed)
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "headline": analysis["aggregate"], "decision": analysis["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now().isoformat(), "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "budget_declared": {"development_mpc_simulation_episodes": EPISODES_EXACT, "development_control_step_upper_bound": CONTROL_STEP_CAP}, "next_recovery_hint": "Preserve partial outputs. If source repair is needed, version it and request backup before rerun; do not overwrite episode directories."})
        raise
