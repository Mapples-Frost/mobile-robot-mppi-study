#!/usr/bin/env python3
"""Run the frozen source-independent true-variable-H risk-anchor acquisition v0.

This is an IMPROVED development-only diagnostic for the Bøhn 2021 adaptive MPC
prediction-horizon reproduction/improvement effort.  The preceding oracle bank
found a real measured control/compute opportunity but the offline deployable
selector feasibility gate failed because conservative H15/H25 labels were too
sparse/concentrated and several labels changed across terminal profiles.  This
runner therefore executes the already frozen, source-independent acquisition
protocol:

  research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_frozen_20260929T0810Z.json

The acquisition uses only development stress-v1 Stage1 H15-trace states outside
the earlier oracle-bank cases, then evaluates true n_horizon in {10, 15, 25}
under matched-terminal and shared-H15-terminal profiles with blocked timing.
It opens no validation64 bank and no sealed final test, performs no training,
no gradient updates, no selector/value refit and no candidate-pool reset.

Modes:
  * --dry-run: no simulation.  Verify the frozen protocol/input artifacts and
    write a backup request covering this new runner before any rollout.
  * --run-acquisition: after a verified external backup postdating the freeze,
    dry-run and this source, execute the 120 frozen direct-branch episodes.
"""
from __future__ import annotations

import argparse
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

import vehicle_true_variable_horizon_case5_smoke_v0_runner as v0  # noqa:E402
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_anchor_acquisition_v0"
STAMP = "20260929T0825Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

FREEZE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_frozen_20260929T0810Z.json"
FREEZE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_20260929T0810Z/completed.json"
FREEZE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_20260929T0810Z/raw.json"
SELECTOR_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0_20260929T0755Z/completed.json"
SELECTOR_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0_20260929T0755Z/raw.json"
STAGE1_BANK = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
STAGE1_BANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/completed.json"
STAGE1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
ORACLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"

DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_dryrun_{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/{NAME}_dryrun_{STAMP}.md"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BEFORE_RUN = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_RISK_ANCHOR_ACQUISITION_V0_RUN_{STAMP}.json"

TRUE_HORIZONS = [10, 15, 25]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
REPEATS = 2
TARGETS = 10
EPISODES_EXACT = TARGETS * len(TRUE_HORIZONS) * len(TERMINAL_PROFILES) * REPEATS
MAX_BRANCH_STEPS = 150
CONTROL_STEP_UPPER = EPISODES_EXACT * MAX_BRANCH_STEPS
MIN_DECISION_SAVING = 0.15
MARKER_DRYRUN = f"vehicle-true-variable-H-risk-anchor-acquisition-v0-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-true-variable-H-risk-anchor-acquisition-v0-run-{STAMP}"


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v0.parse_time(value)


def file_mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def sf(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: " + rel(path))
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "historical_validation64_bank_opened"):
        if key in obj and obj.get(key) is not False:
            raise ContractError(f"{key} flag must be false in {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError("hash mismatch from %s for %s" % (rel(path), name))
    return obj


def median(xs: Iterable[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}

    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * q
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)

    return {"n": len(vals), "min": vals[0], "median": pct(0.5), "mean": float(math.fsum(vals) / len(vals)), "p95": pct(0.95), "max": vals[-1], "sum": float(math.fsum(vals))}


def metric_sum(e: Mapping[str, Any], field: str) -> float:
    return sf((e.get(field) or {}).get("sum"), 0.0)


def is_safe_episode(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 999) == 0 and si(e.get("initial_failed_steps"), 999) == 0 and si(e.get("final_failed_steps"), 999) == 0


def is_risk_anchor_role(role: Any) -> bool:
    text = str(role or "")
    return "risk_anchor" in text and "control" not in text


def verify_protocol_and_inputs() -> Dict[str, Any]:
    for p in (FREEZE_PROTOCOL, FREEZE_DONE, FREEZE_RAW, SELECTOR_DONE, SELECTOR_RAW, STAGE1_BANK, STAGE1_BANK_DONE, STAGE1_DONE, ORACLE_DONE, ORACLE_RAW):
        if not p.exists():
            raise ContractError("required input missing: " + rel(p))
    freeze_done = completed_ok(FREEZE_DONE, check_hashes=False)
    completed_ok(SELECTOR_DONE, check_hashes=False)
    completed_ok(STAGE1_BANK_DONE, check_hashes=False)
    completed_ok(STAGE1_DONE, check_hashes=False)
    completed_ok(ORACLE_DONE, check_hashes=False)
    protocol = read_json(FREEZE_PROTOCOL)
    if protocol.get("protocol_id") != "vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_frozen_20260929T0810Z":
        raise ContractError("unexpected risk-anchor acquisition protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("protocol access flags invalid")
    if access.get("requires_verified_external_backup_before_any_future_run") is not True:
        raise ContractError("protocol must require backup before acquisition run")
    targets = (protocol.get("target_selection") or {}).get("targets") or []
    schedule = (protocol.get("arms") or {}).get("schedule") or []
    budget = protocol.get("budget_declared") or {}
    if len(targets) != TARGETS:
        raise ContractError("unexpected target count: %d" % len(targets))
    if len(schedule) != EPISODES_EXACT:
        raise ContractError("unexpected schedule length: %d" % len(schedule))
    if int(budget.get("planned_development_branch_episodes_exact", -1)) != EPISODES_EXACT:
        raise ContractError("declared episode budget mismatch")
    if int(budget.get("development_control_step_upper_bound", -1)) != CONTROL_STEP_UPPER:
        raise ContractError("declared control-step cap mismatch")
    if [int(x) for x in (protocol.get("arms") or {}).get("true_horizons", [])] != TRUE_HORIZONS:
        raise ContractError("horizon grid changed")
    if [str(x) for x in (protocol.get("arms") or {}).get("terminal_profiles", [])] != TERMINAL_PROFILES:
        raise ContractError("terminal profile grid changed")
    target_by_state = {str(t.get("state_id")): t for t in targets}
    if len(target_by_state) != TARGETS:
        raise ContractError("duplicate state_id in targets")
    for t in targets:
        if t.get("source_independent_from_oracle_bank_cases") is not True:
            raise ContractError("target is not source-independent from oracle bank: " + str(t.get("state_id")))
        if "branch_previous_state" not in t:
            raise ContractError("target lacks branch_previous_state: " + str(t.get("state_id")))
    for item in schedule:
        sid = str(item.get("state_id"))
        if sid not in target_by_state:
            raise ContractError("schedule references unknown state: " + sid)
        if int(item.get("true_mpc_n_horizon", -1)) not in TRUE_HORIZONS:
            raise ContractError("invalid horizon in schedule")
        if str(item.get("terminal_profile")) not in TERMINAL_PROFILES:
            raise ContractError("invalid terminal profile in schedule")
    stage1_bank = read_json(STAGE1_BANK)
    selected_cases = stage1_bank.get("selected_cases_full") or stage1_bank.get("selected_cases") or []
    if not isinstance(selected_cases, list) or len(selected_cases) < max(int(t["case"]) for t in targets) + 1:
        raise ContractError("stage1 selected cases missing/too short")
    return {"protocol": protocol, "freeze_done": freeze_done, "target_by_state": target_by_state, "selected_cases": selected_cases}


def terminal_for(profile: str, h: int, terminals: Mapping[int, Tuple[Any, Any]], receipts: Mapping[str, Any]) -> Tuple[Tuple[Any, Any], Any, int]:
    if profile == "shared_h15_terminal":
        src_h = 15
    elif profile == "matched_terminal":
        src_h = h
    else:
        raise ContractError("unknown terminal profile: " + profile)
    if src_h not in terminals:
        raise ContractError("terminal grid missing H%d" % src_h)
    return terminals[src_h], (receipts.get(str(src_h)) or receipts.get(src_h)), src_h


def make_run_item(schedule_item: Mapping[str, Any], target_by_state: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    sid = str(schedule_item["state_id"])
    target = dict(target_by_state[sid])
    merged = dict(target)
    merged.update(dict(schedule_item))
    h = int(schedule_item["true_mpc_n_horizon"])
    profile = str(schedule_item["terminal_profile"])
    merged["commanded_horizon"] = h
    merged["branch_horizon"] = h
    merged["terminal_mode"] = profile
    merged["initialization"] = "direct_branch_state_with_shifted_stress_v1_case_tvp_cold_mpc_guess_risk_anchor_acquisition_v0"
    return merged


def finalize_episode_summary(summary: Dict[str, Any], ep_dir: Path) -> None:
    v0.write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    v0.write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})


def group_key(e: Mapping[str, Any]) -> Tuple[str, str]:
    return str(e.get("state_id")), str(e.get("terminal_profile") or e.get("terminal_mode"))


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"groups": float(len(rows))}
    for h in TRUE_HORIZONS:
        out[f"fixed_H{h}"] = {
            "physical_sum": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("physical"), 0.0) for r in rows)),
            "decision_sum_s": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("decision_sum_s"), 0.0) for r in rows)),
            "solver_sum_s": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("solver_sum_s"), 0.0) for r in rows)),
        }
    o_phys: List[float] = []
    o_dec: List[float] = []
    o_solver: List[float] = []
    used = 0
    for r in rows:
        label = r.get("oracle_label")
        if label is None:
            continue
        m = r["median_by_h"][str(label)]
        o_phys.append(sf(m.get("physical"), 0.0)); o_dec.append(sf(m.get("decision_sum_s"), 0.0)); o_solver.append(sf(m.get("solver_sum_s"), 0.0)); used += 1
    out["near_best_fastest_oracle"] = {"groups_used": float(used), "physical_sum": float(math.fsum(o_phys)), "decision_sum_s": float(math.fsum(o_dec)), "solver_sum_s": float(math.fsum(o_solver))}
    return out


def comparison(ag: Mapping[str, Any], base_h: int) -> Dict[str, Any]:
    o = ag["near_best_fastest_oracle"]
    b = ag[f"fixed_H{base_h}"]
    return {
        "physical_delta_oracle_minus_base": float(o["physical_sum"] - b["physical_sum"]),
        "decision_relative_saving": None if b["decision_sum_s"] <= 0 else float((b["decision_sum_s"] - o["decision_sum_s"]) / b["decision_sum_s"]),
        "solver_relative_saving": None if b["solver_sum_s"] <= 0 else float((b["solver_sum_s"] - o["solver_sum_s"]) / b["solver_sum_s"]),
    }


def fixed_absorption(ag: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    h25_phys = sf(ag["fixed_H25"].get("physical_sum"), 0.0)
    tol = max(2.0 * n, 0.05 * abs(h25_phys)) if n else 0.0
    out: Dict[str, Any] = {"physical_tolerance_vs_H25": tol}
    for h in (10, 15):
        phys_delta = sf(ag[f"fixed_H{h}"]["physical_sum"], 0.0) - h25_phys
        dec = sf(ag[f"fixed_H{h}"]["decision_sum_s"], 0.0)
        dec25 = sf(ag["fixed_H25"]["decision_sum_s"], 0.0)
        save = None if dec25 <= 0 else (dec25 - dec) / dec25
        out[f"fixed_H{h}_absorbs_vs_H25"] = bool(phys_delta <= tol and save is not None and save >= MIN_DECISION_SAVING)
        out[f"fixed_H{h}_physical_delta_vs_H25"] = float(phys_delta)
        out[f"fixed_H{h}_decision_saving_vs_H25"] = save
    out["any_fixed_short_absorbs"] = bool(out.get("fixed_H10_absorbs_vs_H25") or out.get("fixed_H15_absorbs_vs_H25"))
    return out


def analyze(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    targets = {str(t.get("state_id")): t for t in (protocol.get("target_selection") or {}).get("targets", [])}
    rows: List[Dict[str, Any]] = []
    label_counts: Dict[str, int] = {}
    pair_savings_vs_h25: List[float] = []
    risk_pair_savings_vs_h25: List[float] = []
    unsafe_groups: List[Dict[str, Any]] = []
    groups = sorted(set(group_key(e) for e in episodes))
    for sid, profile in groups:
        target = dict(targets.get(sid, {}))
        role = str(target.get("target_role"))
        is_risk = is_risk_anchor_role(role)
        med_by_h: Dict[int, Dict[str, Any]] = {}
        for h in TRUE_HORIZONS:
            reps = [e for e in episodes if group_key(e) == (sid, profile) and si(e.get("true_mpc_n_horizon"), -1) == h]
            safe_all = bool(reps) and all(is_safe_episode(e) for e in reps)
            med_by_h[h] = {
                "n": len(reps),
                "safe_all": safe_all,
                "physical": median([sf(e.get("physical_constraint_cost"), 0.0) for e in reps]),
                "total_cost": median([sf(e.get("total_cost"), 0.0) for e in reps]),
                "decision_sum_s": median([metric_sum(e, "decision_timing_s") for e in reps]),
                "solver_sum_s": median([metric_sum(e, "solver_attempt_timing_s") for e in reps]),
                "steps": median([si(e.get("steps"), 0) for e in reps]),
                "opt_x_size": sorted(set(si(x) for e in reps for x in (e.get("opt_x_sizes_observed") or []))),
                "success_all": bool(reps) and all(bool(e.get("success")) for e in reps),
                "constraint_any": any(bool(e.get("constraint")) for e in reps),
                "solver_failure_steps_sum": int(sum(si(e.get("solver_failure_steps"), 0) for e in reps)),
            }
        safe_hs = [h for h, m in med_by_h.items() if m["safe_all"] and m["physical"] is not None]
        if not safe_hs:
            label = None
            best_physical = None
            tol = None
            near_best_hs: List[int] = []
            unsafe_groups.append({"state_id": sid, "terminal_profile": profile, "reason": "no safe horizon", "median_by_h": med_by_h})
        else:
            best_physical = min(float(med_by_h[h]["physical"]) for h in safe_hs)
            tol = max(2.0, 0.05 * abs(best_physical))
            near_best_hs = [h for h in safe_hs if float(med_by_h[h]["physical"]) <= best_physical + tol]
            label = sorted(near_best_hs, key=lambda h: (float(med_by_h[h]["decision_sum_s"]), float(med_by_h[h]["solver_sum_s"]), h))[0]
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1
        row = {
            "state_id": sid,
            "case": int(target.get("case", -1)),
            "target_index": int(target.get("target_index", -1)),
            "target_role": role,
            "is_risk_anchor_primary": is_risk,
            "selection_group": target.get("selection_group"),
            "window": target.get("window"),
            "terminal_profile": profile,
            "median_by_h": {str(h): med_by_h[h] for h in TRUE_HORIZONS},
            "best_physical": best_physical,
            "near_best_tolerance": tol,
            "near_best_horizons": near_best_hs,
            "oracle_label": label,
        }
        if label is not None and med_by_h[25]["decision_sum_s"] not in (None, 0):
            saving = (float(med_by_h[25]["decision_sum_s"]) - float(med_by_h[label]["decision_sum_s"])) / float(med_by_h[25]["decision_sum_s"])
            row["oracle_vs_H25_decision_relative_saving"] = saving
            pair_savings_vs_h25.append(saving)
            if is_risk:
                risk_pair_savings_vs_h25.append(saving)
        rows.append(row)

    risk_rows = [r for r in rows if r.get("is_risk_anchor_primary")]
    control_rows = [r for r in rows if not r.get("is_risk_anchor_primary")]
    ag_all = aggregate(rows)
    ag_risk = aggregate(risk_rows)
    ag_control = aggregate(control_rows)
    comps = {
        "all": {f"oracle_vs_fixed_H{h}": comparison(ag_all, h) for h in TRUE_HORIZONS},
        "risk_anchor_primary": {f"oracle_vs_fixed_H{h}": comparison(ag_risk, h) for h in TRUE_HORIZONS},
        "controls": {f"oracle_vs_fixed_H{h}": comparison(ag_control, h) for h in TRUE_HORIZONS},
    }

    labels_by_state: Dict[str, Dict[str, Optional[int]]] = {}
    for r in rows:
        labels_by_state.setdefault(str(r["state_id"]), {})[str(r["terminal_profile"])] = None if r.get("oracle_label") is None else int(r["oracle_label"])
    disagreements: List[Dict[str, Any]] = []
    stable_labels: Dict[str, int] = {}
    for sid, profile_labels in sorted(labels_by_state.items()):
        values = {k: v for k, v in profile_labels.items() if v is not None}
        if len(values) == len(TERMINAL_PROFILES) and len(set(values.values())) == 1:
            stable_labels[sid] = int(next(iter(values.values())))
        else:
            target = targets.get(sid, {})
            disagreements.append({"state_id": sid, "case": int(target.get("case", -1)), "target_role": target.get("target_role"), "labels_by_profile": values})

    risk_h15_h25_cases = sorted({int(r["case"]) for r in risk_rows if r.get("oracle_label") in (15, 25)})
    stable_risk_h15_h25_cases = sorted({int((targets.get(sid) or {}).get("case", -1)) for sid, label in stable_labels.items() if label in (15, 25) and is_risk_anchor_role((targets.get(sid) or {}).get("target_role"))})
    all_risk_safe_h10 = bool(risk_rows and all(r.get("oracle_label") == 10 for r in risk_rows) and not unsafe_groups)
    risk_h25_unique_states = sorted({str(r["state_id"]) for r in risk_rows if r.get("oracle_label") == 25})
    risk_h15_unique_states = sorted({str(r["state_id"]) for r in risk_rows if r.get("oracle_label") == 15})
    risk_label_counts: Dict[str, int] = {}
    for r in risk_rows:
        if r.get("oracle_label") is not None:
            risk_label_counts[str(r["oracle_label"])] = risk_label_counts.get(str(r["oracle_label"]), 0) + 1
    profile_disagreement_rate = len(disagreements) / float(len(labels_by_state) or 1)
    risk_disagreements = [d for d in disagreements if is_risk_anchor_role((targets.get(str(d["state_id"])) or {}).get("target_role"))]
    terminal_consistency_not_dominated = bool(profile_disagreement_rate <= 0.5)

    h25_phys = sf(ag_risk["fixed_H25"].get("physical_sum"), 0.0)
    risk_tol = max(2.0 * len(risk_rows), 0.05 * abs(h25_phys)) if risk_rows else 0.0
    phys_gate = bool(risk_rows and comps["risk_anchor_primary"]["oracle_vs_fixed_H25"]["physical_delta_oracle_minus_base"] <= risk_tol)
    dec_save = comps["risk_anchor_primary"]["oracle_vs_fixed_H25"].get("decision_relative_saving")
    aggregate_decision_gate = bool(dec_save is not None and float(dec_save) >= MIN_DECISION_SAVING)
    median_decision_gate = bool(risk_pair_savings_vs_h25 and float(finite_summary(risk_pair_savings_vs_h25)["median"]) >= MIN_DECISION_SAVING)
    safety_gate = bool(not unsafe_groups and all(r.get("oracle_label") is not None for r in risk_rows))
    fresh_anchor_gate = bool(len(risk_h15_h25_cases) >= 2 or all_risk_safe_h10)
    stable_anchor_gate = bool(len(stable_risk_h15_h25_cases) >= 2 or all_risk_safe_h10)
    absorption = fixed_absorption(ag_risk, risk_rows)
    pass_to_offline_cv = bool(fresh_anchor_gate and stable_anchor_gate and terminal_consistency_not_dominated and phys_gate and aggregate_decision_gate and median_decision_gate and safety_gate and not absorption["any_fixed_short_absorbs"] and not all_risk_safe_h10)
    if pass_to_offline_cv:
        next_decision = "freeze no-simulation offline deployable selector/value-model CV using acquisition+oracle-bank development labels before any closed-loop selector rollout"
    elif all_risk_safe_h10:
        next_decision = "document all fresh risk-anchor states safely H10; do not train selector, reassess scenario opportunity and fixed-short-H tradeoff"
    elif not stable_anchor_gate or not terminal_consistency_not_dominated:
        next_decision = "block selector refit; prioritize terminal-value/objective/modeling repair because labels are terminal-profile dependent or too concentrated"
    elif absorption["any_fixed_short_absorbs"]:
        next_decision = "block adaptive selector; fixed H10/H15 appears to absorb the control/compute tradeoff on these development states"
    else:
        next_decision = "block selector refit; inspect raw risk-anchor failures and choose terminal/modeling/scenario intervention rather than scaling sparse labels"
    return {
        "state_profile_rows": rows,
        "labels_by_state": labels_by_state,
        "stable_labels_by_state": stable_labels,
        "profile_label_disagreements": disagreements,
        "profile_label_disagreement_rate": profile_disagreement_rate,
        "risk_profile_label_disagreements": risk_disagreements,
        "oracle_label_counts_all": label_counts,
        "oracle_label_counts_risk_anchor_primary": risk_label_counts,
        "risk_h25_unique_states": risk_h25_unique_states,
        "risk_h15_unique_states": risk_h15_unique_states,
        "risk_h15_h25_cases": risk_h15_h25_cases,
        "stable_risk_h15_h25_cases": stable_risk_h15_h25_cases,
        "all_risk_safe_h10": all_risk_safe_h10,
        "aggregate_all_groups": ag_all,
        "aggregate_risk_anchor_primary_groups": ag_risk,
        "aggregate_control_groups": ag_control,
        "comparisons": comps,
        "risk_oracle_vs_H25_decision_saving_summary": finite_summary(risk_pair_savings_vs_h25),
        "all_oracle_vs_H25_decision_saving_summary": finite_summary(pair_savings_vs_h25),
        "risk_physical_tolerance_vs_H25": risk_tol,
        "unsafe_groups": unsafe_groups,
        "fixed_short_absorption_risk_groups": absorption,
        "gates": {
            "fresh_H15_H25_anchor_cases_or_all_H10_gate": fresh_anchor_gate,
            "stable_H15_H25_anchor_cases_or_all_H10_gate": stable_anchor_gate,
            "terminal_consistency_not_dominated_gate": terminal_consistency_not_dominated,
            "physical_gate_vs_H25_risk_anchor_primary": phys_gate,
            "aggregate_decision_saving_gate_vs_H25_risk_anchor_primary": aggregate_decision_gate,
            "median_decision_saving_gate_vs_H25_risk_anchor_primary": median_decision_gate,
            "safety_gate": safety_gate,
            "fixed_short_not_absorbed_gate": not absorption["any_fixed_short_absorbs"],
            "pass_to_offline_deployable_selector_cv": pass_to_offline_cv,
            "train_or_refit_now": False,
        },
        "decision": next_decision,
    }


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle true-variable-H risk-anchor acquisition v0 dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no training/refit, no validation64, no sealed test.",
        "",
        "## Verified frozen protocol",
        "",
        f"- Protocol: `{raw['protocol']['path']}` sha256 `{raw['protocol']['sha256']}`.",
        f"- Frozen targets: `{raw['target_count']}`; schedule episodes: `{raw['planned_episodes_exact']}`; control-step cap: `{raw['planned_control_step_upper_bound']}`.",
        f"- Target cases: `{raw['target_cases']}`; roles: `{raw['target_roles']}`.",
        "- The acquisition remains development-only and source-independent from the oracle-bank cases; validation64 and sealed test remain closed.",
        "",
        "## Next action after backup",
        "",
        "Run the frozen 120-episode true-H branch acquisition only after an external backup proof postdates this dry-run/source/protocol. The run will report measured whole-decision and solver timing, fixed H10/H15/H25 absorption, terminal-profile consistency and whether any selector/value-model CV is justified.",
        "",
        f"Backup request before rollout: `{raw['backup_request_before_run']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    g = a["gates"]
    lines = [
        "# Vehicle true-variable-H risk-anchor acquisition v0 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only source-independent true-H risk-anchor acquisition; no validation64, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['planned_development_branch_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['development_control_step_upper_bound']}`.",
        "",
        "## Gates",
        "",
        f"- Risk-anchor labels: `{a['oracle_label_counts_risk_anchor_primary']}`; all labels: `{a['oracle_label_counts_all']}`.",
        f"- Risk H15/H25 cases: `{a['risk_h15_h25_cases']}`; stable risk H15/H25 cases: `{a['stable_risk_h15_h25_cases']}`; all risk safe-H10: `{a['all_risk_safe_h10']}`.",
        f"- Terminal-profile disagreements: `{len(a['profile_label_disagreements'])}` / `{len(a['labels_by_state'])}` states; risk disagreements: `{len(a['risk_profile_label_disagreements'])}`.",
        f"- Oracle-vs-H25 risk physical gate: `{g['physical_gate_vs_H25_risk_anchor_primary']}`; comparison `{a['comparisons']['risk_anchor_primary']['oracle_vs_fixed_H25']}`; tolerance `{a['risk_physical_tolerance_vs_H25']}`.",
        f"- Aggregate decision-time saving gate: `{g['aggregate_decision_saving_gate_vs_H25_risk_anchor_primary']}`; median saving gate: `{g['median_decision_saving_gate_vs_H25_risk_anchor_primary']}`; summary `{a['risk_oracle_vs_H25_decision_saving_summary']}`.",
        f"- Fixed-short absorption: `{a['fixed_short_absorption_risk_groups']}`.",
        f"- Pass to offline deployable selector CV: `{g['pass_to_offline_deployable_selector_cv']}`; train/refit now: `{g['train_or_refit_now']}`.",
        "",
        "## State/profile labels",
        "",
        "| state | case | role | profile | label | H10 phys | H15 phys | H25 phys | H10 dec | H15 dec | H25 dec |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in a["state_profile_rows"]:
        med = r["median_by_h"]
        def fmt(h: int, k: str) -> str:
            val = med[str(h)].get(k)
            return "NA" if val is None else "%.6g" % float(val)
        lines.append("| `%s` | %d | `%s` | `%s` | `%s` | %s | %s | %s | %s | %s | %s |" % (
            r["state_id"], int(r.get("case", -1)), r.get("target_role"), r.get("terminal_profile"), r.get("oracle_label"),
            fmt(10, "physical"), fmt(15, "physical"), fmt(25, "physical"), fmt(10, "decision_sum_s"), fmt(15, "decision_sum_s"), fmt(25, "decision_sum_s"),
        ))
    lines += ["", "## Decision", "", str(a["decision"]), "", "This is development evidence only. It is not validation/test success and not ORIGINAL reproduction; true variable-H work is IMPROVED. Timing claims are measured whole-decision/solver timing, not H proxies.", "", f"Backup request after run: `{raw['backup_request_after_run']}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run() -> int:
    done_path = DRYRUN_DIR / "completed.json"
    if done_path.exists():
        done = completed_ok(done_path, check_hashes=True)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError("partial dry-run output exists; inspect first: " + rel(DRYRUN_DIR))
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    info = verify_protocol_and_inputs()
    protocol = info["protocol"]
    created = now_utc()
    target_roles: Dict[str, int] = {}
    target_cases = []
    for t in (protocol.get("target_selection") or {}).get("targets") or []:
        target_cases.append(int(t["case"]))
        role = str(t.get("target_role"))
        target_roles[role] = target_roles.get(role, 0) + 1
    write_json(REQUEST_BEFORE_RUN, {
        "requested_utc": created.isoformat(),
        "reason": "backup risk-anchor acquisition runner/dry-run/source plus frozen protocol before 120 development true-H branch simulations",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "planned_development_branch_episodes_exact": EPISODES_EXACT,
        "development_control_step_upper_bound": CONTROL_STEP_UPPER,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(FREEZE_PROTOCOL), rel(FREEZE_DONE), rel(FREEZE_RAW), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(REQUEST_BEFORE_RUN)],
    })
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": f"{NAME}_dryrun",
        "classification": "development_metadata_no_simulation_risk_anchor_acquisition_runner_readiness",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "candidate_pool_resets": 0,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "protocol": {"path": rel(FREEZE_PROTOCOL), "sha256": sha256(FREEZE_PROTOCOL)},
        "target_count": TARGETS,
        "target_cases": sorted(set(target_cases)),
        "target_roles": target_roles,
        "planned_episodes_exact": EPISODES_EXACT,
        "planned_control_step_upper_bound": CONTROL_STEP_UPPER,
        "source_hashes": {rel(p): sha256(p) for p in [SOURCE, FREEZE_PROTOCOL, FREEZE_DONE, FREEZE_RAW, SELECTOR_DONE, SELECTOR_RAW, STAGE1_BANK_DONE, STAGE1_DONE, ORACLE_DONE, ORACLE_RAW] if p.exists()},
        "backup_request_before_run": rel(REQUEST_BEFORE_RUN),
        "next_action": "await verified external backup after this dry-run/source, then run --run-acquisition with legacy interpreter; do not run validation64/test",
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text((DRYRUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle true-variable-H risk-anchor acquisition v0 dry-run

UTC: {created.isoformat()}. Wrote and dry-ran `{rel(SOURCE)}` with no simulations, no candidate resets, no training/refit, no validation64-bank access and no sealed-test access. It verifies the frozen source-independent risk-anchor protocol `{rel(FREEZE_PROTOCOL)}` ({EPISODES_EXACT} planned true-H branch episodes, cap {CONTROL_STEP_UPPER} control steps) after the selector-feasibility failure. Further simulation is blocked until verified external backup covers this runner/dry-run/freeze artifacts and `{rel(REQUEST_BEFORE_RUN)}`.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, FREEZE_PROTOCOL, FREEZE_DONE, FREEZE_RAW, STATE_DRYRUN, REQUEST_BEFORE_RUN]
    write_json(done_path, {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "classification": "metadata_no_simulation_runner_readiness", "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_simulations": 0, "new_control_steps": 0, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "planned_development_branch_episodes_exact": EPISODES_EXACT, "development_control_step_upper_bound": CONTROL_STEP_UPPER, "backup_required_before_run": True, "backup_request": rel(REQUEST_BEFORE_RUN), "headline": {"runner_ready_after_backup": True, "target_count": TARGETS, "planned_episodes_exact": EPISODES_EXACT, "train_or_refit_now": False}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(done_path), "summary": rel(DRYRUN_DIR / "summary.md"), "planned_episodes_exact": EPISODES_EXACT, "planned_control_step_upper_bound": CONTROL_STEP_UPPER, "new_simulations": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(REQUEST_BEFORE_RUN)}, sort_keys=True), flush=True)
    return 0


def run_acquisition(backup_proof: Path) -> int:
    info = verify_protocol_and_inputs()
    protocol = info["protocol"]
    dry_done = completed_ok(DRYRUN_DIR / "completed.json", check_hashes=True)
    min_time_candidates = [file_mtime_utc(SOURCE), file_mtime_utc(FREEZE_PROTOCOL), parse_time(info["freeze_done"].get("created_utc")), parse_time(dry_done.get("created_utc"))]
    min_time = max(t for t in min_time_candidates if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, NAME)
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial acquisition output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v0.SMOKE_DIR = RUN_DIR

    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: %r" % (preflight,))
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    for h in TRUE_HORIZONS:
        if h not in terminals:
            raise ContractError("terminal grid missing H%d" % h)
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    started_dt = now_utc()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started_dt.isoformat(), "pid": os.getpid(), "method": NAME, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0})

    selected_cases = info["selected_cases"]
    episodes: List[Dict[str, Any]] = []
    schedule = (protocol.get("arms") or {}).get("schedule") or []
    for item0 in schedule:
        item = make_run_item(item0, info["target_by_state"])
        case_index = int(item["case"])
        h = int(item["true_mpc_n_horizon"])
        profile = str(item["terminal_profile"])
        terminal_tuple, receipt, source_h = terminal_for(profile, h, terminals, terminal_receipts)
        summary = v0.run_true_h_episode(item, selected_cases[case_index], terminal_tuple)
        summary["repeat"] = int(item["repeat"])
        summary["terminal_profile"] = profile
        summary["target_role"] = item.get("target_role")
        summary["selection_group"] = item.get("selection_group")
        summary["terminal_source_horizon"] = int(source_h)
        summary["terminal_receipt_effective"] = receipt
        summary["is_risk_anchor_primary"] = is_risk_anchor_role(item.get("target_role"))
        ep_dir = ROOT / str(summary["path"])
        finalize_episode_summary(summary, ep_dir)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule), "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "repeat", "terminal_profile", "true_mpc_n_horizon", "steps", "success", "termination", "opt_x_sizes_observed")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    declared = protocol.get("budget_declared") or {}
    if len(episodes) != int(declared.get("planned_development_branch_episodes_exact", -1)):
        raise ContractError("episode budget violation")
    if control_steps > int(declared.get("development_control_step_upper_bound", -1)):
        raise ContractError("control-step budget violation")
    analysis = analyze(episodes, protocol)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_RISK_ANCHOR_ACQUISITION_V0_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup risk-anchor acquisition outputs before offline selector CV, terminal-value repair or further simulations", "backup_required_before_more_simulations": True, "backup_required_before_training_or_refit": True, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(RUN_DIR), rel(STATE_RUN), rel(SOURCE), rel(FREEZE_PROTOCOL), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_H_risk_anchor_acquisition_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_proof": backup,
        "protocol": {"json": rel(FREEZE_PROTOCOL), "sha256": sha256(FREEZE_PROTOCOL)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "episodes": episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development-only", "source-independent from oracle-bank cases but selected from opened stress-v1 development traces", "not validation/model selection", "not final test", "not ORIGINAL SAC", "does not infer speed from H alone; measured solver/decision timing reported"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_run_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle true-variable-H risk-anchor acquisition v0 run

UTC: {created.isoformat()}. Development-only source-independent risk-anchor acquisition completed: {len(episodes)} branch episodes, {control_steps} control steps, validation64 closed, sealed test closed, no training/refit. Gates: {analysis['gates']}. Risk labels: {analysis['oracle_label_counts_risk_anchor_primary']}. Decision: {analysis['decision']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""", MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, FREEZE_PROTOCOL, STATE_RUN, req, backup_proof, DRYRUN_DIR / "completed.json", FREEZE_DONE, SELECTOR_DONE, STAGE1_BANK_DONE, STAGE1_DONE]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": analysis["gates"], "risk_label_counts": analysis["oracle_label_counts_risk_anchor_primary"], "stable_risk_h15_h25_cases": analysis["stable_risk_h15_h25_cases"], "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": analysis["gates"], "risk_label_counts": analysis["oracle_label_counts_risk_anchor_primary"], "stable_risk_h15_h25_cases": analysis["stable_risk_h15_h25_cases"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-acquisition", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-risk-anchor-acquisition-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_risk_anchor_acquisition_v0:
        raise ContractError("explicit risk-anchor acquisition acknowledgement required")
    if bool(args.dry_run) == bool(args.run_acquisition):
        raise ContractError("exactly one of --dry-run or --run-acquisition required")
    if args.dry_run:
        return dry_run()
    if args.backup_proof is None:
        raise ContractError("--run-acquisition requires --backup-proof")
    return run_acquisition(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else RUN_DIR if "--run-acquisition" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/{NAME}_failure_unknown_{STAMP}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "candidate_pool_resets": 0, "new_simulations": 0 if "--dry-run" in sys.argv else None, "new_control_steps": 0 if "--dry-run" in sys.argv else None, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial output. If dry-run failed, repair source/protocol checks only; if acquisition failed, audit completed episode count and hashes before any rerun."})
        raise
