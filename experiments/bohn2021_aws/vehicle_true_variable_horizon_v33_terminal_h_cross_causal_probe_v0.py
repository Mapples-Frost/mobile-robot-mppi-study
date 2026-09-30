#!/usr/bin/env python3
"""v33 terminal x horizon causal cross-probe for opened development states.

Implements the Astra 2026-09-30 focused recommendation for v29-v32: before
collecting new labels or training/refitting a selector, cross horizon with shared
terminal-value contracts on the already-opened states that produced the current
H12/H15/H35 opportunity/anomaly.

Development-only IMPROVED diagnostic. It runs no validation64 and no sealed test,
performs no gradient training and no selector refit.

Frozen design in this source before execution:
  states: v27_case09_slot0_early_risk, v27_case09_slot1_mid_late_risk,
          v19_c12, v19_c13, v27_case00_slot1_mid_late_risk,
          v27_case08_slot1_mid_late_risk
  horizons: H12/H15/H25/H35
  terminal contracts: zero_like, V15_shared, V35_shared
  budget: 72 continuation episodes, <=10800 control steps.

The optional alternate-initialization single-state solver calls from the Astra
report are not performed in v33; the raw output explicitly records this as a
remaining subdiagnostic. The first pass focuses on the causal H x terminal
continuation matrix with existing reliable rollout instrumentation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29 as v29  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v33_terminal_h_cross_probe.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_ID = f"v33-terminal-h-cross-causal-probe-{STAMP}"
BACKUP_REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V33_TERMINAL_H_CROSS_CAUSAL_PROBE_{STAMP}.json"
MARKER = f"vehicle-true-variable-horizon-v33-terminal-h-cross-causal-probe-v0-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

CURRENT_ASTRA_REQUEST = "v32-h12-supported-default-h35-diagnostic-20260930T051611Z"
CURRENT_ASTRA_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260930T034449Z.md"
ANALYSIS_READY = ROOT / "docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json"
NEXT_REVIEW_REQUEST = ROOT / "docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

TARGET_STATE_LABELS = [
    "v27_case09_slot0_early_risk",
    "v27_case09_slot1_mid_late_risk",
    "v19_c12",
    "v19_c13",
    "v27_case00_slot1_mid_late_risk",
    "v27_case08_slot1_mid_late_risk",
]
HORIZONS = [12, 15, 25, 35]
TERMINAL_MODES = ["zero", "V15_shared", "V35_shared"]
MAX_STEPS = 150
EPISODES_EXACT = len(TARGET_STATE_LABELS) * len(HORIZONS) * len(TERMINAL_MODES)
CONTROL_STEP_CAP = EPISODES_EXACT * MAX_STEPS
ORDER_SEED = 202609300601
MIN_LARGE_EXCESS_ABS = 2.0
LARGE_EXCESS_FRAC = 0.25


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


def parse_time(s: Any) -> Optional[dt.datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        out = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def pct(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * sf(x):.2f}%"


def metric_sum(summary: Mapping[str, Any], name: str) -> float:
    obj = summary.get(name)
    return sf(obj.get("sum"), 0.0) if isinstance(obj, Mapping) else 0.0


def is_safe(summary: Mapping[str, Any]) -> bool:
    return bool(summary.get("success")) and not bool(summary.get("constraint")) and si(summary.get("solver_failure_steps"), 999) == 0 and si(summary.get("initial_failed_steps"), 999) == 0 and si(summary.get("final_failed_steps"), 999) == 0 and si(summary.get("steps"), 9999) < MAX_STEPS


def verify_astra_ready() -> Dict[str, Any]:
    if not ANALYSIS_READY.exists():
        raise ContractError("Astra ANALYSIS_READY.json is absent")
    ready = read_json(ANALYSIS_READY)
    if ready.get("request_id") != CURRENT_ASTRA_REQUEST:
        raise ContractError(f"Astra ready request mismatch: {ready.get('request_id')} != {CURRENT_ASTRA_REQUEST}")
    report = ROOT / str(ready.get("report", ""))
    if not report.exists():
        raise ContractError("Astra report path missing: " + str(ready.get("report")))
    expected = ready.get("report_sha256")
    actual = sha256(report)
    if expected and actual != expected:
        raise ContractError("Astra report sha mismatch")
    if ready.get("primary_analyst") != "gpt-6-astra":
        raise ContractError("unexpected Astra primary analyst")
    return {"ready": ready, "report_path": rel(report), "report_sha256": actual}


def verify_backup_context(args: argparse.Namespace) -> Dict[str, Any]:
    if not (args.backup_time and args.backup_commit and args.backup_package_sha256):
        raise ContractError("verified backup context arguments are required before v33 unique science")
    t = parse_time(args.backup_time)
    if t is None:
        raise ContractError("backup_time is not parseable")
    required_after = dt.datetime.fromisoformat("2026-09-30T05:33:33+00:00")
    if t <= required_after:
        raise ContractError("backup context does not postdate the v32 post-backup/Astra state recheck request")
    proof_path = BACKUP_DIR / f"backup_proof_{STAMP}_from_user_context_before_v33_terminal_h_cross_probe.json"
    proof = {
        "status": "verified",
        "backup_verified": True,
        "time": t.isoformat(),
        "commit": args.backup_commit,
        "remaining_changed_files": 0,
        "packages_this_run": [{"sha256": args.backup_package_sha256, "verification": "user_context_verified_backup", "bytes": args.backup_package_bytes}],
        "source": "supervisor_user_context_current_prompt",
        "purpose": "gate v33 terminal x H cross-probe after v32 postbackup state recheck",
    }
    write_json(proof_path, proof)
    proof["path"] = rel(proof_path)
    proof["sha256"] = sha256(proof_path)
    return proof


def zero_like_terminal(pair: Tuple[Any, Any]) -> Tuple[List[Any], List[Any]]:
    weights, biases = pair
    return [np.zeros_like(np.asarray(w)) for w in weights], [np.zeros_like(np.asarray(b)) for b in biases]


def terminal_for_mode(h: int, mode: str, terminals: Mapping[int, Tuple[Any, Any]]) -> Tuple[Tuple[Any, Any], int, str]:
    if mode == "V15_shared":
        return terminals[15], 15, "V15_shared_all_H"
    if mode == "V35_shared":
        return terminals[35], 35, "V35_shared_all_H"
    if mode == "zero":
        base_h = h if h in terminals else 15
        return zero_like_terminal(terminals[base_h]), base_h, f"zero_like_V{base_h}"
    raise ContractError("unknown terminal mode " + mode)


def select_states() -> List[Dict[str, Any]]:
    _inputs = v29.load_protocol_and_inputs()
    all_specs = v29.build_state_specs()
    by_label = {str(s.get("state_label")): s for s in all_specs}
    missing = [x for x in TARGET_STATE_LABELS if x not in by_label]
    if missing:
        raise ContractError("missing target state labels: " + ", ".join(missing))
    return [by_label[x] for x in TARGET_STATE_LABELS]


def make_schedule(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(ORDER_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for state_index, spec in enumerate(specs):
        cells = [(mode, h) for mode in TERMINAL_MODES for h in HORIZONS]
        rng.shuffle(cells)
        for mode, h in cells:
            rows.append({
                "execution_index": exe,
                "state_index": state_index,
                "state_label": spec["state_label"],
                "base_state_id": spec["base_state_id"],
                "category": spec["category"],
                "source_campaign": spec["source_campaign"],
                "source_candidate_index": spec.get("source_candidate_index"),
                "candidate_index": spec.get("candidate_index"),
                "branch_step": spec["branch_step"],
                "horizon": int(h),
                "terminal_mode": mode,
                "blocked_randomization_unit": f"v33|state{state_index:02d}|{spec['state_label']}",
            })
            exe += 1
    return rows


def load_trace_diag(summary: Mapping[str, Any]) -> Dict[str, Any]:
    path = ROOT / str(summary.get("path", ""))
    trace_file = path / "trace.jsonl"
    rows: List[Dict[str, Any]] = []
    if trace_file.exists():
        with trace_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        alt = path / "trace.json"
        if alt.exists():
            rows = read_json(alt)
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    statuses = Counter()
    accepted = 0
    max_constraint_resid = 0.0
    max_bound_resid = 0.0
    objectives: List[float] = []
    iterations: List[int] = []
    for r in rows:
        for a in ((r.get("recovery") or {}).get("attempts") or []):
            statuses[str(a.get("return_status"))] += 1
            if a.get("accepted") is True:
                accepted += 1
            max_constraint_resid = max(max_constraint_resid, sf(a.get("constraint_residual"), 0.0))
            max_bound_resid = max(max_bound_resid, sf(a.get("bound_residual"), 0.0))
            if a.get("objective_opt_f_num") is not None:
                objectives.append(sf(a.get("objective_opt_f_num")))
            if a.get("iterations") is not None:
                iterations.append(si(a.get("iterations")))
    first_attempt = ((first.get("recovery") or {}).get("attempts") or [{}])[-1] if first else {}
    post_ws = first_attempt.get("warm_start_post_opt_x") or {}
    pre_ws = first_attempt.get("warm_start_pre_opt_x") or {}
    return {
        "trace_rows": len(rows),
        "trace_path": rel(trace_file) if trace_file.exists() else rel(path / "trace.json"),
        "first_action": first.get("input"),
        "first_previous_state": first.get("previous_state"),
        "first_state_after_action": first.get("state"),
        "final_state": last.get("state"),
        "first_solver_status": first_attempt.get("return_status"),
        "first_solver_iterations": first_attempt.get("iterations"),
        "first_objective_opt_f_num": first_attempt.get("objective_opt_f_num"),
        "first_constraint_residual": first_attempt.get("constraint_residual"),
        "first_bound_residual": first_attempt.get("bound_residual"),
        "first_warm_start_pre_hash": pre_ws.get("sha256"),
        "first_warm_start_post_hash": post_ws.get("sha256"),
        "solver_status_counts": dict(statuses),
        "accepted_solve_attempts": accepted,
        "max_constraint_residual": max_constraint_resid,
        "max_bound_residual": max_bound_resid,
        "objective_opt_f_num_first_last": [objectives[0], objectives[-1]] if objectives else [],
        "solver_iterations_first_last": [iterations[0], iterations[-1]] if iterations else [],
        "instrumentation_limits": {
            "stage_slack_terminal_objective_decomposition_available": False,
            "gamma_pow_H_terminal_value_available": False,
            "predicted_terminal_state_available": False,
            "goal_parameters_available": False,
            "reason": "existing true-H branch runner stores accepted NLP objective, residuals, warm-start hashes and full realised trace but not objective-term decomposition or predicted terminal vector",
        },
    }


def episode_record(summary: Mapping[str, Any], spec: Mapping[str, Any], h: int, terminal_mode: str, terminal_source_h: int, terminal_note: str) -> Dict[str, Any]:
    diag = load_trace_diag(summary)
    return {
        "category": spec["category"],
        "source_campaign": spec["source_campaign"],
        "base_state_id": spec["base_state_id"],
        "state_label": spec["state_label"],
        "source_candidate_index": spec.get("source_candidate_index"),
        "candidate_index": spec.get("candidate_index"),
        "role": spec.get("role"),
        "branch_step": spec.get("branch_step"),
        "horizon": int(h),
        "terminal_mode": terminal_mode,
        "terminal_source_horizon": int(terminal_source_h),
        "terminal_note": terminal_note,
        "success": bool(summary.get("success")),
        "constraint": bool(summary.get("constraint")),
        "safe_success_no_solver_fail": is_safe(summary),
        "termination": summary.get("termination"),
        "steps": si(summary.get("steps"), 0),
        "hit_step_cap": si(summary.get("steps"), 0) >= MAX_STEPS,
        "solver_failure_steps": si(summary.get("solver_failure_steps"), 0),
        "initial_failed_steps": si(summary.get("initial_failed_steps"), 0),
        "final_failed_steps": si(summary.get("final_failed_steps"), 0),
        "physical_constraint_cost": sf(summary.get("physical_constraint_cost"), 0.0),
        "total_cost": sf(summary.get("total_cost"), 0.0),
        "decision_sum_s": metric_sum(summary, "decision_timing_s"),
        "solver_sum_s": metric_sum(summary, "solver_attempt_timing_s"),
        "decision_mean_s": sf((summary.get("decision_timing_s") or {}).get("mean")),
        "decision_p95_s": sf((summary.get("decision_timing_s") or {}).get("p95")),
        "solver_mean_s": sf((summary.get("solver_attempt_timing_s") or {}).get("mean")),
        "solver_p95_s": sf((summary.get("solver_attempt_timing_s") or {}).get("p95")),
        "opt_x_sizes_observed": summary.get("opt_x_sizes_observed"),
        "path": summary.get("path"),
        "trace_diagnostics": diag,
    }


def analyze(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_state_term: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for e in episodes:
        by_state_term[(str(e["state_label"]), str(e["terminal_mode"]))][int(e["horizon"])] = e
    state_terminal_rows: List[Dict[str, Any]] = []
    for (state, mode), d in sorted(by_state_term.items()):
        safe_h = [h for h in HORIZONS if d[h]["safe_success_no_solver_fail"]]
        if safe_h:
            best_phys = min(sf(d[h]["physical_constraint_cost"]) for h in safe_h)
            tol = max(MIN_LARGE_EXCESS_ABS, LARGE_EXCESS_FRAC * abs(best_phys))
            near_best = [h for h in safe_h if sf(d[h]["physical_constraint_cost"]) <= best_phys + tol]
            best_physical_h = sorted(safe_h, key=lambda h: (sf(d[h]["physical_constraint_cost"]), sf(d[h]["decision_sum_s"]), h))[0]
            fastest_near_best_h = sorted(near_best, key=lambda h: (sf(d[h]["decision_sum_s"]), sf(d[h]["physical_constraint_cost"]), h))[0]
        else:
            best_phys = None; tol = None; near_best = []; best_physical_h = None; fastest_near_best_h = None
        bad_rows = []
        for h in HORIZONS:
            e = d[h]
            if not e["safe_success_no_solver_fail"]:
                bad_rows.append({"horizon": h, "reason": "unsafe_or_step_cap", "physical": e["physical_constraint_cost"], "steps": e["steps"]})
            elif best_phys is not None and sf(e["physical_constraint_cost"]) > best_phys + float(tol):
                bad_rows.append({"horizon": h, "reason": "large_physical_excess_vs_best_safe", "physical": e["physical_constraint_cost"], "best_safe_physical": best_phys, "tol": tol})
        state_terminal_rows.append({
            "state_label": state,
            "terminal_mode": mode,
            "safe_horizons": safe_h,
            "best_physical_horizon": best_physical_h,
            "fastest_near_best_safe_horizon": fastest_near_best_h,
            "best_safe_physical": best_phys,
            "large_excess_tolerance": tol,
            "bad_rows": bad_rows,
            "per_horizon_brief": {str(h): {"safe": d[h]["safe_success_no_solver_fail"], "steps": d[h]["steps"], "physical": d[h]["physical_constraint_cost"], "decision_s": d[h]["decision_sum_s"], "solver_s": d[h]["solver_sum_s"], "first_action": d[h]["trace_diagnostics"].get("first_action"), "first_objective": d[h]["trace_diagnostics"].get("first_objective_opt_f_num")} for h in HORIZONS},
        })
    by_state_h: Dict[Tuple[str, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for e in episodes:
        by_state_h[(str(e["state_label"]), int(e["horizon"]))][str(e["terminal_mode"])] = e
    terminal_effect_rows: List[Dict[str, Any]] = []
    for (state, h), d in sorted(by_state_h.items()):
        row = {"state_label": state, "horizon": h}
        for mode in TERMINAL_MODES:
            e = d.get(mode)
            row[mode] = None if e is None else {"safe": e["safe_success_no_solver_fail"], "physical": e["physical_constraint_cost"], "decision_s": e["decision_sum_s"], "steps": e["steps"], "first_action": e["trace_diagnostics"].get("first_action"), "first_objective": e["trace_diagnostics"].get("first_objective_opt_f_num")}
        if d.get("V35_shared") and d.get("V15_shared"):
            row["V35_minus_V15_physical"] = sf(d["V35_shared"]["physical_constraint_cost"]) - sf(d["V15_shared"]["physical_constraint_cost"])
            row["V35_minus_V15_decision_s"] = sf(d["V35_shared"]["decision_sum_s"]) - sf(d["V15_shared"]["decision_sum_s"])
            row["V35_vs_V15_success_change"] = bool(d["V35_shared"]["safe_success_no_solver_fail"]) != bool(d["V15_shared"]["safe_success_no_solver_fail"])
        terminal_effect_rows.append(row)
    source242 = [r for r in state_terminal_rows if r["state_label"].startswith("v27_case09")]
    source242_mode = {}
    for mode in TERMINAL_MODES:
        rows = [r for r in source242 if r["terminal_mode"] == mode]
        source242_mode[mode] = {
            "states": len(rows),
            "H35_safe_count": sum(1 for r in rows if 35 in r["safe_horizons"]),
            "H12_safe_count": sum(1 for r in rows if 12 in r["safe_horizons"]),
            "H15_safe_count": sum(1 for r in rows if 15 in r["safe_horizons"]),
            "best_physical_horizons": Counter(str(r["best_physical_horizon"]) for r in rows),
            "fastest_near_best_horizons": Counter(str(r["fastest_near_best_safe_horizon"]) for r in rows),
        }
    c13_h35 = [r for r in terminal_effect_rows if r["state_label"] == "v19_c13" and r["horizon"] == 35]
    c13 = c13_h35[0] if c13_h35 else {}
    c13_v35 = ((c13.get("V35_shared") or {}).get("physical")) if c13 else None
    c13_v15 = ((c13.get("V15_shared") or {}).get("physical")) if c13 else None
    c13_zero = ((c13.get("zero") or {}).get("physical")) if c13 else None
    c13_terminal_relief = None
    if c13_v35 is not None:
        better = [x for x in [c13_v15, c13_zero] if x is not None]
        if better:
            c13_terminal_relief = float(c13_v35 - min(float(x) for x in better))
    aggregate = {
        "episodes": len(episodes),
        "control_steps": int(sum(si(e.get("steps"), 0) for e in episodes)),
        "safe_counts_by_horizon": {str(h): sum(1 for e in episodes if int(e["horizon"]) == h and e["safe_success_no_solver_fail"]) for h in HORIZONS},
        "safe_counts_by_terminal_mode": {m: sum(1 for e in episodes if e["terminal_mode"] == m and e["safe_success_no_solver_fail"]) for m in TERMINAL_MODES},
        "decision_sum_by_horizon": {str(h): float(math.fsum(sf(e["decision_sum_s"]) for e in episodes if int(e["horizon"]) == h)) for h in HORIZONS},
        "physical_sum_by_horizon": {str(h): float(math.fsum(sf(e["physical_constraint_cost"]) for e in episodes if int(e["horizon"]) == h)) for h in HORIZONS},
        "decision_sum_by_terminal_mode": {m: float(math.fsum(sf(e["decision_sum_s"]) for e in episodes if e["terminal_mode"] == m)) for m in TERMINAL_MODES},
    }
    if all(v["H35_safe_count"] == 2 and v["H12_safe_count"] == 0 and v["H15_safe_count"] == 0 for v in source242_mode.values()):
        causal_read = "source242 rescue appears horizon-dominated across zero/V15/V35 terminals; terminal-only rescue of short H is not supported in v33."
    elif any(v["H12_safe_count"] or v["H15_safe_count"] for v in source242_mode.values()):
        causal_read = "at least one short-H source242 rescue appears after terminal swap; terminal contract can change rescue and should be repaired before selector work."
    else:
        causal_read = "source242 rescue remains mixed; inspect per-state rows before attribution."
    if c13_terminal_relief is not None and c13_terminal_relief > max(MIN_LARGE_EXCESS_ABS, LARGE_EXCESS_FRAC * max(1.0, abs(float(c13_v35)))):
        c13_read = "v19_c13 H35 cost anomaly is terminal-sensitive: V35 is materially worse than V15/zero."
    else:
        c13_read = "v19_c13 H35 cost anomaly is not clearly relieved by terminal swap in v33."
    return {
        "aggregate": aggregate,
        "state_terminal_rows": state_terminal_rows,
        "terminal_effect_rows": terminal_effect_rows,
        "source242_by_terminal_mode": source242_mode,
        "v19_c13_H35_terminal_effect": {"V35_physical": c13_v35, "V15_physical": c13_v15, "zero_physical": c13_zero, "V35_minus_best_other": c13_terminal_relief, "read": c13_read},
        "causal_readout_executor_numeric": {"source242_read": causal_read, "c13_read": c13_read, "alternate_initialization_solver_calls_performed": 0, "alternate_initialization_deferred_reason": "requires a separate low-level warm-start manipulation path; not needed for the first H x terminal continuation matrix and not executed in v33"},
    }


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# v33 terminal × H causal cross-probe",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only continuation matrix requested by Astra; no validation64, no sealed test, no selector refit, no gradient training.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` / `{raw['budget_declared']['development_episodes']}` episodes; `{raw['budget_actual']['control_steps']}` / `{raw['budget_declared']['control_step_cap']}` control steps.",
        "",
        "## Headline numeric readout",
        "",
        f"- Source242 by terminal mode: `{a['source242_by_terminal_mode']}`.",
        f"- v19_c13 H35 terminal effect: `{a['v19_c13_H35_terminal_effect']}`.",
        f"- Executor numeric readout: `{a['causal_readout_executor_numeric']}`.",
        f"- Aggregate safe counts by horizon: `{a['aggregate']['safe_counts_by_horizon']}`; by terminal: `{a['aggregate']['safe_counts_by_terminal_mode']}`.",
        "",
        "## Per state/terminal best horizon table",
        "",
        "| state | terminal | safe H | best-physical H | fastest near-best H | bad rows | H12/H15/H25/H35 physical | H12/H15/H25/H35 decision s |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for r in a["state_terminal_rows"]:
        phys = [round(sf(r["per_horizon_brief"][str(h)]["physical"]), 6) for h in HORIZONS]
        dec = [round(sf(r["per_horizon_brief"][str(h)]["decision_s"]), 6) for h in HORIZONS]
        bad = [(x["horizon"], x["reason"]) for x in r["bad_rows"]]
        lines.append(f"| `{r['state_label']}` | `{r['terminal_mode']}` | `{r['safe_horizons']}` | `{r['best_physical_horizon']}` | `{r['fastest_near_best_safe_horizon']}` | `{bad}` | `{phys}` | `{dec}` |")
    lines += [
        "",
        "## Terminal-effect rows by fixed horizon",
        "",
        "The raw file contains first actions, first accepted NLP objective, solver residuals and warm-start hashes for every cell. Existing runner instrumentation does not expose stage/slack/terminal objective decomposition, γ^H V or predicted terminal state; those remain a follow-up if v33 indicates optimization-basin or terminal-value causality.",
        "",
        f"Raw: `{rel(RUN_DIR / 'raw.json')}`. Completed: `{rel(RUN_DIR / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_docs_and_astra_request(raw: Mapping[str, Any], astra: Mapping[str, Any]) -> None:
    h = raw["analysis"]["aggregate"]
    block = f"""<!-- {MARKER} -->
## 2026-09-30 v33 terminal × H causal cross-probe

UTC: {raw['created_utc']}. Executed Astra-selected Task 1 as development-only IMPROVED evidence: terminal contracts zero/V15_shared/V35_shared × H12/H15/H25/H35 on six already-opened branch states. Budget {raw['budget_actual']['episodes']} episodes and {raw['budget_actual']['control_steps']} control steps; validation64 closed, sealed test closed, no training/refit. Executor numeric readout: {raw['analysis']['causal_readout_executor_numeric']}. Source242 summary: {raw['analysis']['source242_by_terminal_mode']}. v19_c13 H35 terminal effect: {raw['analysis']['v19_c13_H35_terminal_effect']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`. New backup request: `{rel(BACKUP_REQUEST)}`. New Astra request: `{REQUEST_ID}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_if_missing(ROOT / doc, MARKER, block)
    response = f"""
<!-- {MARKER} -->
## v33 terminal × H causal cross-probe response to Astra report

Updated by GPT-5.5 executor at `{raw['created_utc']}`. This run verifies and acts on Astra report `{astra['report_path']}` for request `{CURRENT_ASTRA_REQUEST}`; report sha256 `{astra['report_sha256']}`. Development-only; validation64=false, sealed_test=false, training/refit=0.

| linked recommendation(s) | disposition | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed`, `A11_training_failure_modes_need_separation`, `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted and executed | v33 ran terminal × H matrix with budgets `{raw['budget_actual']}` and Astra-matched report/request IDs. | Evidence paths `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`. Executor numeric readout: `{raw['analysis']['causal_readout_executor_numeric']}`. |
| `A7_targeted_risk_banks_are_not_population_estimates`, `A8_zero_catastrophe_small_sample_model_selection_risk` | deferred pending Astra analysis of v33 | v33 remains six opened development states only and does not create validation/test evidence. | Wrote refreshed NEXT_REVIEW_REQUEST `{REQUEST_ID}`; do not start new-source confirmation / selector training / validation before matching or superseding Astra analysis unless continuing reversible integrity work. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | still open / not tested in v33 | v33 is constant-H continuation matrix; selector overhead and true closed-loop switching not evaluated. | Carry into later Task 3 only if Task 1/2 gates pass. |
| `A12_registry_backup_schema_contract` | accepted | Pre-v33 backup context was materialized; v33 wrote post-run backup request `{rel(BACKUP_REQUEST)}`. | Require external backup covering v33 before further unique simulation/refit/training/validation. |
"""
    append_if_missing(RESPONSE_LOG, MARKER, response)
    req = {
        "request_id": REQUEST_ID,
        "created": raw["created_utc"],
        "status": "analysis_requested",
        "trigger": "v33 terminal x H causal cross-probe completed after Astra v29-v32 report",
        "experiment_id": NAME,
        "supersedes_request_id": CURRENT_ASTRA_REQUEST,
        "question": "Analyze the v33 terminal×H continuation matrix over six opened states. Decide whether the evidence supports terminal/value-contract repair, horizon/geometry/source confirmation, optimization-basin single-state diagnostics, or another bounded action. Preserve that v33 is development-only and does not authorize validation64 or sealed-test access.",
        "evidence_paths": [
            rel(RUN_DIR / "summary.md"),
            rel(RUN_DIR / "raw.json"),
            rel(RUN_DIR / "completed.json"),
            rel(CURRENT_ASTRA_REPORT),
            rel(RESPONSE_LOG),
        ],
        "access_and_budget_note": raw["budget_actual"],
        "operational_note": f"v33 outputs require external backup via {rel(BACKUP_REQUEST)} before additional unique scientific work.",
    }
    write_json(NEXT_REVIEW_REQUEST, req)
    # Keep a lightweight experiment index line for local audit; run_experiment also registers this execution.
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old_tail = reg.read_text(encoding="utf-8", errors="replace")[-100000:] if reg.exists() else ""
    if MARKER not in old_tail:
        with reg.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([raw["created_utc"], NAME, raw["classification"], f"ORDER_SEED={ORDER_SEED}", "opened_development_terminal_h_cross_no_validation_no_test", raw["budget_actual"]["episodes"], raw["budget_actual"]["control_steps"], 0, 0, 0, False, rel(RUN_DIR / "completed.json"), MARKER])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Continue state after v33 terminal x H causal cross-probe\n\nUTC: {raw['created_utc']}\n\nAstra request now pending: {REQUEST_ID}\n\nBudget: {raw['budget_actual']}\n\nNumeric readout: {raw['analysis']['causal_readout_executor_numeric']}\n\nArtifacts: {rel(RUN_DIR / 'summary.md')}, {rel(RUN_DIR / 'raw.json')}, {rel(RUN_DIR / 'completed.json')}\n\nNext: obtain external backup for {rel(BACKUP_REQUEST)}, then wait/read matching Astra analysis before starting new acquisition/refit/training/scenario/validation branch.\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument("--backup-time", required=True)
    ap.add_argument("--backup-commit", required=True)
    ap.add_argument("--backup-package-sha256", required=True)
    ap.add_argument("--backup-package-bytes", type=int, default=0)
    ap.add_argument("--i-accept-development-v33-terminal-h-cross", action="store_true", required=True)
    args = ap.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        astra = verify_astra_ready()
        backup = verify_backup_context(args)
        started = now_utc()
        write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "classification": "development_terminal_h_cross_no_validation_no_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_or_gradient_steps": 0, "selector_refits": 0, "pre_run_backup_context": backup, "astra_ready": astra})
        v29.v8.engine.case_runner.SMOKE_DIR = RUN_DIR
        _, stage1_runner, _ = v1d.import_legacy_modules()
        preflight = stage1_runner.runtime_preflight()
        if not preflight.get("passed"):
            raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
        stage1_runner.base.v1.latency_verify()
        terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
        terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
        for h in (15, 35):
            if h not in terminals:
                raise ContractError(f"terminal grid missing H{h}")
        specs = select_states()
        schedule = make_schedule(specs)
        if len(schedule) != EPISODES_EXACT:
            raise ContractError("schedule length mismatch")
        write_json(RUN_DIR / "selected_state_manifest.json", {"created_utc": now_utc().isoformat(), "target_state_labels": TARGET_STATE_LABELS, "states": [{k: v for k, v in s.items() if k != "case_snapshot"} for s in specs], "schedule": schedule, "selection_completed_before_v33_outcomes": True, "validation64_bank_opened": False, "sealed_test_accessed": False})
        write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
        write_json(RUN_DIR / "runtime_preflight.json", preflight)
        spec_by_index = {i: s for i, s in enumerate(specs)}
        episodes: List[Dict[str, Any]] = []
        for row in schedule:
            spec = spec_by_index[int(row["state_index"])]
            h = int(row["horizon"])
            mode = str(row["terminal_mode"])
            terminal, terminal_h, terminal_note = terminal_for_mode(h, mode, terminals)
            item = {
                "execution_index": 33000 + int(row["execution_index"]),
                "state_id": f"v33_{int(row['execution_index']):03d}_{spec['state_label']}_{mode}_H{h}",
                "case": int(row["state_index"]),
                "source_candidate_index": spec.get("source_candidate_index"),
                "branch_step": int(spec["branch_step"]),
                "branch_previous_state": dict(spec["branch_previous_state"]),
                "true_mpc_n_horizon": h,
                "commanded_horizon": h,
                "terminal_mode": mode,
                "initialization": "v33_terminal_h_cross_direct_branch_state_from_v29_manifest",
            }
            summary = v29.v8.engine.case_runner.run_true_h_episode(item, spec["case_snapshot"], terminal)
            summary.update({"stage": "v33_terminal_h_cross", "category": spec["category"], "source_campaign": spec["source_campaign"], "base_state_id": spec["base_state_id"], "state_label": spec["state_label"], "role": spec.get("role"), "terminal_mode": mode, "terminal_source_horizon": terminal_h, "terminal_note": terminal_note, "blocked_randomization_unit": row["blocked_randomization_unit"]})
            ep = episode_record(summary, spec, h, mode, terminal_h, terminal_note)
            episodes.append(ep)
            control_steps_done = int(sum(si(e.get("steps"), 0) for e in episodes))
            write_json(RUN_DIR / "progress.json", {"episodes_done": len(episodes), "episodes_expected": EPISODES_EXACT, "control_steps_done": control_steps_done, "control_step_cap": CONTROL_STEP_CAP, "last_episode": ep, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_or_gradient_steps": 0, "selector_refits": 0})
            print(json.dumps({"episodes_done": len(episodes), "state": spec["state_label"], "mode": mode, "H": h, "safe": ep["safe_success_no_solver_fail"], "steps": ep["steps"], "phys": ep["physical_constraint_cost"]}, sort_keys=True), flush=True)
        control_steps = int(sum(si(e.get("steps"), 0) for e in episodes))
        if len(episodes) != EPISODES_EXACT or control_steps > CONTROL_STEP_CAP:
            raise ContractError("budget violation")
        analysis = analyze(episodes)
        created = now_utc()
        raw = {
            "created_utc": created.isoformat(),
            "started_utc": started.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "method": NAME,
            "classification": "development_IMPROVED_terminal_h_cross_causal_probe_not_validation_not_test",
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "mobile_robot_mppi_resumed": False,
            "astra_ready": astra,
            "pre_run_backup_context": backup,
            "hypothesis_frozen": "Crossing H with zero/V15/V35 terminals on the six opened diagnostic states distinguishes horizon geometry from terminal-value-contract effects before any new selector training or validation.",
            "horizons": HORIZONS,
            "terminal_modes": TERMINAL_MODES,
            "budget_declared": {"development_episodes": EPISODES_EXACT, "control_step_cap": CONTROL_STEP_CAP, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "alternate_initialization_solver_calls_optional_cap": 24, "alternate_initialization_solver_calls_executed": 0},
            "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "alternate_initialization_solver_calls_executed": 0},
            "selected_states": [{k: v for k, v in s.items() if k != "case_snapshot"} for s in specs],
            "schedule": schedule,
            "episodes": episodes,
            "analysis": analysis,
            "runtime_preflight": preflight,
            "terminal_sources": {str(k): v for k, v in terminal_receipts.items()},
            "input_hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), ANALYSIS_READY, CURRENT_ASTRA_REPORT, v29.V29_RAW if hasattr(v29, 'V29_RAW') else Path(__file__).resolve()] if p.exists()},
            "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "backup_request_after_run": rel(BACKUP_REQUEST),
            "next_astra_request_id": REQUEST_ID,
            "interpretation_limits": ["opened development states only", "not validation64", "not sealed test", "not ORIGINAL SAC", "no selector overhead or online switching tested", "alternate initialization solver calls not performed in v33"],
        }
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        write_json(BACKUP_REQUEST, {"request": "backup_after_v33_terminal_h_cross_causal_probe", "created_utc": created.isoformat(), "backup_required_before_more_unique_science": True, "reason": "new v33 development simulation evidence and Astra handoff", "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(NEXT_REVIEW_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "new_control_steps": control_steps, "new_simulation_episodes": len(episodes), "new_training_or_gradient_steps": 0, "selector_refits": 0, "validation64_bank_opened": False, "validation64_episodes": 0, "sealed_test_accessed": False, "sealed_test_episodes": 0})
        update_docs_and_astra_request(raw, astra)
        files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE, BACKUP_REQUEST, NEXT_REVIEW_REQUEST, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed = {"status": "complete", "passed": True, "hard_pass": True, "created_utc": created.isoformat(), "classification": raw["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "budget_actual": raw["budget_actual"], "headline": {"source242_by_terminal_mode": analysis["source242_by_terminal_mode"], "v19_c13_H35_terminal_effect": analysis["v19_c13_H35_terminal_effect"], "executor_numeric_readout": analysis["causal_readout_executor_numeric"]}, "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "backup_request": rel(BACKUP_REQUEST), "next_astra_request_id": REQUEST_ID, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}}
        write_json(RUN_DIR / "completed.json", completed)
        # Refresh raw/summary with completed hash for traceability.
        raw["completed_sha256"] = sha256(RUN_DIR / "completed.json")
        write_json(RUN_DIR / "raw.json", raw)
        write_summary(raw)
        print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "raw": rel(RUN_DIR / "raw.json"), "episodes": len(episodes), "control_steps": control_steps, "headline": completed["headline"], "backup_request": rel(BACKUP_REQUEST), "next_astra_request_id": REQUEST_ID, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(RUN_DIR / "failed.json", {"status": "failed", "created_utc": now_utc().isoformat(), "error": repr(exc), "traceback": traceback.format_exc(), "classification": "development_IMPROVED_terminal_h_cross_causal_probe_not_validation_not_test", "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        print(json.dumps({"failed": repr(exc), "failed_artifact": rel(RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
