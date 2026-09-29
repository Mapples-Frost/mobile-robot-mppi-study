#!/usr/bin/env python3
"""Run the frozen vehicle true-variable-H oracle-label bank v0.

This is an IMPROVED development-only diagnostic runner.  It executes the
protocol frozen by vehicle_true_variable_horizon_oracle_bank_freeze_v0.py:
16 source-supported branch states x 2 terminal profiles x H=[10,15,25] x
2 blocked repeats = 192 direct-branch true-MPC episodes.

Access contract: no validation64 bank, no sealed test, no gradient training,
no selector/value refit, no candidate-pool reset.  The runner requires a
verified external backup proof postdating this source and the frozen protocol
before --run-bank performs any simulation.
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

NAME = "vehicle_true_variable_horizon_oracle_bank_v0"
STAMP = "20260929T0725Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_frozen_20260929T0725Z.json"
FREEZE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_freeze_v0_20260929T0725Z/completed.json"
V1D_BANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json"
BROADER_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z/completed.json"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BEFORE = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_ORACLE_BANK_V0_RUN_20260929T0725Z.json"
MARKER_RUN = f"vehicle-true-variable-horizon-oracle-bank-v0-run-{STAMP}"

TRUE_HORIZONS = [10, 15, 25]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
MAX_BRANCH_STEPS = 150
MIN_DECISION_SAVING = 0.15


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
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


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
    if obj.get("validation64_bank_opened") is not False:
        raise ContractError("validation64 flag must be false in " + rel(path))
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError("sealed-test flag must be false in " + rel(path))
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


def verify_protocol() -> Dict[str, Any]:
    freeze_done = completed_ok(FREEZE_DONE, check_hashes=False)
    completed_ok(V1D_BANK_DONE, check_hashes=False)
    completed_ok(BROADER_DONE, check_hashes=False)
    if not PROTOCOL_JSON.exists():
        raise ContractError("missing frozen protocol: " + rel(PROTOCOL_JSON))
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_true_variable_horizon_oracle_bank_v0_frozen_20260929T0725Z":
        raise ContractError("unexpected oracle-bank protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("protocol access flags invalid")
    arms = protocol.get("arms") or {}
    schedule = arms.get("schedule") or []
    targets = (protocol.get("target_selection") or {}).get("targets") or []
    budget = protocol.get("budget_declared") or {}
    if [int(x) for x in arms.get("true_horizons", [])] != TRUE_HORIZONS:
        raise ContractError("horizon grid changed")
    if [str(x) for x in arms.get("terminal_profiles", [])] != TERMINAL_PROFILES:
        raise ContractError("terminal profile grid changed")
    if len(targets) != 16 or len(schedule) != 192:
        raise ContractError("unexpected target/schedule size")
    if int(budget.get("development_branch_episodes_exact", -1)) != len(schedule):
        raise ContractError("declared episode budget mismatch")
    target_by_state = {str(t.get("state_id")): t for t in targets}
    for item in schedule:
        sid = str(item.get("state_id"))
        if sid not in target_by_state:
            raise ContractError("schedule references missing state: " + sid)
        if int(item.get("true_mpc_n_horizon", -1)) not in TRUE_HORIZONS:
            raise ContractError("invalid horizon in schedule")
        if str(item.get("terminal_profile")) not in TERMINAL_PROFILES:
            raise ContractError("invalid terminal profile in schedule")
    return {"protocol": protocol, "freeze_done": freeze_done, "target_by_state": target_by_state}


def load_case_banks() -> Dict[str, List[Mapping[str, Any]]]:
    # Stress-v1d states came from the v1c/v1d development bank.  Broader anchors
    # came from the stage1/v1e case bank used by the true-H broader block.
    v1d_bank = v1d.load_v1c_bank(v1d.V1C_BANK_ACTUAL)
    stage1_bank = read_json(Path(v0.STAGE1_BANK))
    out = {
        "stress_v1d_trace_selected_primary": v1d_bank.get("selected_cases_full") or v1d_bank.get("selected_cases") or [],
        "true_h_broader_block_anchor_previously_used": stage1_bank.get("selected_cases_full") or stage1_bank.get("selected_cases") or [],
    }
    for name, cases in out.items():
        if not isinstance(cases, list) or not cases:
            raise ContractError("empty case bank for " + name)
    return out


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
    merged["initialization"] = "direct_branch_state_with_shifted_case_tvp_cold_mpc_guess_oracle_bank_v0"
    if "branch_previous_state" not in merged:
        raise ContractError("target lacks branch_previous_state: " + sid)
    return merged


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


def finalize_episode_summary(summary: Dict[str, Any], ep_dir: Path) -> None:
    v0.write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    v0.write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})


def group_key(e: Mapping[str, Any]) -> Tuple[str, str]:
    return str(e.get("state_id")), str(e.get("terminal_profile") or e.get("terminal_mode"))


def analyze(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    targets = {str(t.get("state_id")): t for t in (protocol.get("target_selection") or {}).get("targets", [])}
    rows: List[Dict[str, Any]] = []
    label_counts: Dict[str, int] = {}
    oracle_labels: Dict[str, Optional[int]] = {}
    pair_savings_vs_h25: List[float] = []
    noncontrol_pair_savings_vs_h25: List[float] = []
    unsafe_groups: List[Dict[str, Any]] = []
    groups = sorted(set(group_key(e) for e in episodes))
    for sid, profile in groups:
        target = targets.get(sid, {})
        target_noncontrol = bool(target.get("noncontrol_for_primary_gate"))
        source_stratum = str(target.get("source_stratum"))
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
            "terminal_profile": profile,
            "source_stratum": source_stratum,
            "target_role": target.get("target_role"),
            "noncontrol_for_primary_gate": target_noncontrol,
            "previously_used_in_true_h_broader_block": bool(target.get("previously_used_in_true_h_broader_block")),
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
            if target_noncontrol:
                noncontrol_pair_savings_vs_h25.append(saving)
        rows.append(row)
        oracle_labels[f"{sid}|{profile}"] = label

    def aggregate(rows_in: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"groups": float(len(rows_in))}
        for h in TRUE_HORIZONS:
            out[f"fixed_H{h}"] = {
                "physical_sum": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("physical"), 0.0) for r in rows_in)),
                "decision_sum_s": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("decision_sum_s"), 0.0) for r in rows_in)),
                "solver_sum_s": float(math.fsum(sf((r["median_by_h"][str(h)] or {}).get("solver_sum_s"), 0.0) for r in rows_in)),
            }
        oracle_phys = []
        oracle_dec = []
        oracle_solver = []
        used = 0
        for r in rows_in:
            label = r.get("oracle_label")
            if label is None:
                continue
            m = r["median_by_h"][str(label)]
            oracle_phys.append(sf(m.get("physical"), 0.0)); oracle_dec.append(sf(m.get("decision_sum_s"), 0.0)); oracle_solver.append(sf(m.get("solver_sum_s"), 0.0)); used += 1
        out["near_best_fastest_oracle"] = {"groups_used": float(used), "physical_sum": float(math.fsum(oracle_phys)), "decision_sum_s": float(math.fsum(oracle_dec)), "solver_sum_s": float(math.fsum(oracle_solver))}
        return out

    all_rows = rows
    primary_noncontrol_rows = [r for r in rows if r.get("noncontrol_for_primary_gate")]
    ag_all = aggregate(all_rows)
    ag_non = aggregate(primary_noncontrol_rows)

    def comparison(ag: Mapping[str, Any], base_h: int) -> Dict[str, Any]:
        o = ag["near_best_fastest_oracle"]; b = ag[f"fixed_H{base_h}"]
        return {
            "physical_delta_oracle_minus_base": float(o["physical_sum"] - b["physical_sum"]),
            "decision_relative_saving": None if b["decision_sum_s"] <= 0 else float((b["decision_sum_s"] - o["decision_sum_s"]) / b["decision_sum_s"]),
            "solver_relative_saving": None if b["solver_sum_s"] <= 0 else float((b["solver_sum_s"] - o["solver_sum_s"]) / b["solver_sum_s"]),
        }

    comparisons = {
        "all": {f"oracle_vs_fixed_H{h}": comparison(ag_all, h) for h in TRUE_HORIZONS},
        "primary_noncontrol": {f"oracle_vs_fixed_H{h}": comparison(ag_non, h) for h in TRUE_HORIZONS},
    }
    n_non = len(primary_noncontrol_rows)
    fixed25_phys = ag_non["fixed_H25"]["physical_sum"]
    oracle_vs_h25_non = comparisons["primary_noncontrol"]["oracle_vs_fixed_H25"]
    physical_tol_non = max(2.0 * n_non, 0.05 * abs(float(fixed25_phys))) if n_non else 0.0
    physical_gate = bool(n_non and oracle_vs_h25_non["physical_delta_oracle_minus_base"] <= physical_tol_non)
    aggregate_decision_gate = bool(oracle_vs_h25_non["decision_relative_saving"] is not None and float(oracle_vs_h25_non["decision_relative_saving"]) >= MIN_DECISION_SAVING)
    median_decision_gate = bool(noncontrol_pair_savings_vs_h25 and float(finite_summary(noncontrol_pair_savings_vs_h25)["median"]) >= MIN_DECISION_SAVING)
    safety_gate = not unsafe_groups and all((r.get("oracle_label") is not None) for r in primary_noncontrol_rows)
    selected_non_h25 = sum(1 for r in primary_noncontrol_rows if r.get("oracle_label") is not None and int(r.get("oracle_label")) != 25)
    nondegenerate_gate = bool(len(label_counts) >= 2 and n_non and (selected_non_h25 / float(n_non)) >= 0.20)
    pass_to_selector = bool(physical_gate and aggregate_decision_gate and median_decision_gate and safety_gate and nondegenerate_gate)
    if pass_to_selector:
        decision = "freeze bounded IMPROVED selector/value-modeling/refit using oracle-bank labels and measured timing, with >=3 independent seeds and fair fixed-H baselines; keep validation/test independent"
    else:
        decision = "do not train selector from this bank yet; choose terminal-value/modeling/objective or scenario-opportunity intervention according to failed gate(s), not another sparse-label sweep"
    return {
        "state_profile_rows": rows,
        "oracle_labels": oracle_labels,
        "oracle_label_counts": label_counts,
        "aggregate_all_groups": ag_all,
        "aggregate_primary_noncontrol_groups": ag_non,
        "comparisons": comparisons,
        "primary_noncontrol_count": n_non,
        "primary_noncontrol_oracle_vs_H25_physical_tolerance": physical_tol_non,
        "oracle_vs_H25_decision_saving_summary_all": finite_summary(pair_savings_vs_h25),
        "oracle_vs_H25_decision_saving_summary_primary_noncontrol": finite_summary(noncontrol_pair_savings_vs_h25),
        "unsafe_groups": unsafe_groups,
        "gates": {
            "physical_gate_vs_H25_primary_noncontrol": physical_gate,
            "aggregate_decision_saving_gate_vs_H25_primary_noncontrol": aggregate_decision_gate,
            "median_decision_saving_gate_vs_H25_primary_noncontrol": median_decision_gate,
            "safety_gate": safety_gate,
            "label_non_degeneracy_gate": nondegenerate_gate,
            "pass_to_selector_or_value_modeling_design": pass_to_selector,
            "train_or_refit_now": False,
        },
        "decision": decision,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    g = a["gates"]
    lines = [
        "# Vehicle true variable-H oracle bank v0 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only true per-H oracle-label bank; no validation64, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['development_branch_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['development_control_step_upper_bound']}`.",
        "",
        "## Gates",
        "",
        f"- Oracle labels: `{a['oracle_label_counts']}`.",
        f"- Primary non-control groups: `{a['primary_noncontrol_count']}`.",
        f"- Oracle-vs-H25 primary non-control physical gate: `{g['physical_gate_vs_H25_primary_noncontrol']}`; comparison `{a['comparisons']['primary_noncontrol']['oracle_vs_fixed_H25']}`; tolerance `{a['primary_noncontrol_oracle_vs_H25_physical_tolerance']}`.",
        f"- Aggregate decision-time saving gate: `{g['aggregate_decision_saving_gate_vs_H25_primary_noncontrol']}`.",
        f"- Median decision-time saving gate: `{g['median_decision_saving_gate_vs_H25_primary_noncontrol']}`; summary `{a['oracle_vs_H25_decision_saving_summary_primary_noncontrol']}`.",
        f"- Safety gate: `{g['safety_gate']}`.",
        f"- Label non-degeneracy gate: `{g['label_non_degeneracy_gate']}`.",
        f"- Pass to selector/value-modeling design: `{g['pass_to_selector_or_value_modeling_design']}`.",
        "",
        "## State/profile labels",
        "",
        "| state | stratum | profile | label | H10 phys | H15 phys | H25 phys | H10 dec | H15 dec | H25 dec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in a["state_profile_rows"]:
        med = r["median_by_h"]
        def fmt(h: int, k: str) -> str:
            val = med[str(h)].get(k)
            return "NA" if val is None else "%.6g" % float(val)
        lines.append("| `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s | %s | %s |" % (
            r["state_id"], r.get("source_stratum"), r["terminal_profile"], r.get("oracle_label"),
            fmt(10, "physical"), fmt(15, "physical"), fmt(25, "physical"),
            fmt(10, "decision_sum_s"), fmt(15, "decision_sum_s"), fmt(25, "decision_sum_s"),
        ))
    lines += ["", "## Decision", "", str(a["decision"]), "", "Development evidence only: this does not claim validation success, final-test support, or ORIGINAL reproduction. Timing claims are based on measured whole-decision/solver timing, not H alone.", "", f"Backup request after run: `{raw['backup_request_after_run']}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run_bank(backup_proof: Path) -> int:
    info = verify_protocol()
    protocol = info["protocol"]
    min_time = max(t for t in [file_mtime_utc(SOURCE), file_mtime_utc(PROTOCOL_JSON), parse_time(info["freeze_done"].get("created_utc"))] if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, NAME)
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v0.SMOKE_DIR = RUN_DIR

    case_banks = load_case_banks()
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

    started_dt = now_utc()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started_dt.isoformat(), "pid": os.getpid(), "method": NAME, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "candidate_pool_resets": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    schedule = protocol.get("arms", {}).get("schedule") or []
    for item0 in schedule:
        item = make_run_item(item0, info["target_by_state"])
        stratum = str(item.get("source_stratum"))
        cases = case_banks.get(stratum)
        if not cases:
            raise ContractError("no case bank for stratum " + stratum)
        case_index = int(item["case"])
        if case_index < 0 or case_index >= len(cases):
            raise ContractError("case index out of range for %s: %d" % (stratum, case_index))
        h = int(item["true_mpc_n_horizon"])
        profile = str(item["terminal_profile"])
        terminal_tuple, receipt, source_h = terminal_for(profile, h, terminals, terminal_receipts)
        summary = v0.run_true_h_episode(item, cases[case_index], terminal_tuple)
        summary["repeat"] = int(item["repeat"])
        summary["terminal_profile"] = profile
        summary["source_stratum"] = stratum
        summary["target_role"] = item.get("target_role")
        summary["noncontrol_for_primary_gate"] = bool(item.get("noncontrol_for_primary_gate"))
        summary["previously_used_in_true_h_broader_block"] = bool(item.get("previously_used_in_true_h_broader_block"))
        summary["terminal_source_horizon"] = int(source_h)
        summary["terminal_receipt_effective"] = receipt
        ep_dir = ROOT / str(summary["path"])
        finalize_episode_summary(summary, ep_dir)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule), "control_steps_done": int(sum(si(e.get("steps")) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "repeat", "terminal_profile", "true_mpc_n_horizon", "steps", "success", "termination", "opt_x_sizes_observed")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    control_steps = int(sum(si(e.get("steps")) for e in episodes))
    declared = protocol.get("budget_declared") or {}
    if len(episodes) != int(declared.get("development_branch_episodes_exact", -1)):
        raise ContractError("episode budget violation")
    if control_steps > int(declared.get("development_control_step_upper_bound", -1)):
        raise ContractError("control-step budget violation")
    analysis = analyze(episodes, protocol)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_ORACLE_BANK_V0_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup true-variable-H oracle-label bank outputs before selector/value-modeling or further diagnostics", "backup_required_before_more_simulations": True, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(RUN_DIR), rel(STATE_RUN), rel(PROTOCOL_JSON), rel(SOURCE), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started_dt.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_horizon_oracle_label_bank_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "sha256": sha256(PROTOCOL_JSON)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "episodes": episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development-only", "not validation/model selection", "not final test", "not ORIGINAL SAC", "does not infer speed from H alone; measured solver/decision timing reported", "broader-block anchors are previously used development calibration states"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle true variable-H oracle bank v0 run

UTC: {created.isoformat()}. Development-only oracle-label bank completed: {len(episodes)} branch episodes, {control_steps} control steps, validation64 closed, sealed test closed, no training/refit. Gates: {analysis['gates']}. Label counts: {analysis['oracle_label_counts']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""", MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL_JSON, STATE_RUN, req, backup_proof, FREEZE_DONE, V1D_BANK_DONE, BROADER_DONE]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": analysis["gates"], "oracle_label_counts": analysis["oracle_label_counts"], "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": analysis["gates"], "oracle_label_counts": analysis["oracle_label_counts"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def write_backup_request_status() -> int:
    info = verify_protocol()
    created = now_utc()
    min_time = max(t for t in [file_mtime_utc(SOURCE), file_mtime_utc(PROTOCOL_JSON), parse_time(info["freeze_done"].get("created_utc"))] if t is not None)
    out = {
        "requested_utc": created.isoformat(),
        "existing_freeze_request": rel(REQUEST_BEFORE),
        "reason": "backup oracle-bank runner source plus frozen protocol/freezer outputs before 192 development true-H branch simulations",
        "backup_required_before_more_simulations": True,
        "required_backup_time_after_utc": min_time.isoformat(),
        "development_branch_episodes_exact": 192,
        "development_control_step_upper_bound": 28800,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(PROTOCOL_JSON), rel(FREEZE_DONE), rel(REQUEST_BEFORE)],
    }
    path = BACKUP_DIR / "REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_ORACLE_BANK_V0_RUNNER_AND_RUN_20260929T0730Z.json"
    write_json(path, out)
    print(json.dumps({"backup_request": rel(path), "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-backup-request", action="store_true")
    ap.add_argument("--run-bank", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-true-variable-horizon-oracle-bank-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_true_variable_horizon_oracle_bank_v0:
        raise ContractError("explicit oracle-bank acknowledgement required")
    if bool(args.write_backup_request) == bool(args.run_bank):
        raise ContractError("exactly one of --write-backup-request or --run-bank required")
    if args.write_backup_request:
        return write_backup_request_status()
    if args.backup_proof is None:
        raise ContractError("--run-bank requires --backup-proof")
    return run_bank(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial outputs; audit completed episode count and hashes before rerun. If failure occurred before episodes, repair wrapper and re-backup before simulation."})
        raise
