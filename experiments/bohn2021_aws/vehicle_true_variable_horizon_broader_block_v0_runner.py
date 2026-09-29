#!/usr/bin/env python3
"""Broader true variable-H vehicle development block runner v0.

This IMPROVED development runner follows the v0b case5 true-controller smoke and
its no-simulation freeze.  v0b showed that constructing true per-H do-mpc
controllers reduces optimizer dimension and measured solve/decision time, but it
did not retain the earlier fixed-size common-prefix physical gains.  Therefore
this runner does not train or refit a selector.  It executes only the frozen
small randomized development block to test whether a real measured
control-vs-compute tradeoff survives beyond two cold-start states.

Access contract: development artifacts only; no validation64 bank, no sealed
final test, no new training/refit/candidate-pool resets.  A verified external
backup postdating this runner and the frozen protocol is required before
--run-block will execute simulations.
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

NAME = "vehicle_true_variable_horizon_broader_block_v0"
STAMP = "20260929T0650Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_broader_block_v0_frozen_20260929T0650Z.json"
FREEZE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_freeze_v0_20260929T0650Z/completed.json"
FREEZE_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_freeze_v0_20260929T0650Z/summary.md"
V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/completed.json"
V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/raw.json"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_RUN = BACKUP_DIR / "REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_BROADER_BLOCK_V0_RUNNER_AND_RUN_20260929T0655Z.json"
MARKER_RUN = f"vehicle-true-variable-horizon-broader-block-v0-run-{STAMP}"

TRUE_HORIZONS = [10, 15, 25]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
MAX_BRANCH_STEPS = 150
MIN_REL_SAVING = 0.15


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


def parse_time(x: Any) -> Optional[dt.datetime]:
    return v0.parse_time(x)


def file_mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


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


def verify_protocol() -> Dict[str, Any]:
    freeze_done = completed_ok(FREEZE_DONE, check_hashes=False)
    v0b_done = completed_ok(V0B_DONE, check_hashes=False)
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_true_variable_horizon_broader_block_v0_frozen_20260929T0650Z":
        raise ContractError("unexpected broader block protocol id")
    if protocol.get("access_rules", {}).get("validation64_bank_opened") is not False or protocol.get("access_rules", {}).get("sealed_test_accessed") is not False:
        raise ContractError("protocol access flags invalid")
    schedule = protocol.get("arms", {}).get("schedule") or []
    targets = protocol.get("target_selection", {}).get("targets") or []
    if [int(x) for x in protocol.get("arms", {}).get("true_horizons", [])] != TRUE_HORIZONS:
        raise ContractError("true horizon grid changed")
    if [str(x) for x in protocol.get("arms", {}).get("terminal_profiles", [])] != TERMINAL_PROFILES:
        raise ContractError("terminal profiles changed")
    budget = protocol.get("budget_declared") or {}
    if int(budget.get("development_branch_episodes_exact", -1)) != len(schedule):
        raise ContractError("episode budget does not match schedule")
    if len(schedule) != 48 or len(targets) != 4:
        raise ContractError("unexpected broader block size")
    if (v0b_done.get("headline") or {}).get("pass_to_broader_variable_horizon_block") is not True:
        raise ContractError("v0b implementation gate did not pass")
    by_state = {str(t.get("state_id")): t for t in targets}
    for item in schedule:
        sid = str(item.get("state_id"))
        if sid not in by_state:
            raise ContractError("schedule item missing target: " + sid)
        h = int(item.get("true_mpc_n_horizon", -1))
        if h not in TRUE_HORIZONS:
            raise ContractError("schedule horizon not in grid")
        prof = str(item.get("terminal_profile"))
        if prof not in TERMINAL_PROFILES:
            raise ContractError("schedule terminal profile not in grid")
    return {"protocol": protocol, "freeze_done": freeze_done, "v0b_done": v0b_done, "target_by_state": by_state}


def write_backup_request_only() -> None:
    info = verify_protocol()
    created = now_utc()
    min_required = max(t for t in [file_mtime_utc(SOURCE), file_mtime_utc(PROTOCOL_JSON), parse_time(info["freeze_done"].get("created_utc")), parse_time(info["v0b_done"].get("created_utc"))] if t is not None)
    write_json(REQUEST_BACKUP_BEFORE_RUN, {
        "requested_utc": created.isoformat(),
        "reason": "backup broader true-variable-H runner/source/freeze artifacts before running 48 development branch simulations",
        "backup_required_before_more_simulations": True,
        "required_backup_time_after_utc": min_required.isoformat(),
        "development_branch_episodes_exact": 48,
        "development_control_step_upper_bound": 7200,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(PROTOCOL_JSON), rel(FREEZE_DONE), rel(FREEZE_SUMMARY), rel(V0B_DONE), rel(V0B_RAW), rel(REQUEST_BACKUP_BEFORE_RUN)],
    })


def make_run_item(schedule_item: Mapping[str, Any], target_by_state: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    sid = str(schedule_item["state_id"])
    merged = dict(target_by_state[sid])
    merged.update(dict(schedule_item))
    h = int(schedule_item["true_mpc_n_horizon"])
    merged["commanded_horizon"] = h
    merged["branch_horizon"] = h
    merged["terminal_mode"] = str(schedule_item["terminal_profile"])
    merged["initialization"] = "direct_branch_state_with_shifted_case_tvp_cold_mpc_guess_broader_v0"
    if "branch_previous_state" not in merged:
        raise ContractError("merged item lacks branch_previous_state: " + sid)
    return merged


def terminal_for(profile: str, h: int, terminals: Mapping[int, Tuple[Any, Any]], receipts: Mapping[str, Any]) -> Tuple[Tuple[Any, Any], Any, int]:
    if profile == "shared_h15_terminal":
        src_h = 15
    elif profile == "matched_terminal":
        src_h = h
    else:
        raise ContractError("unknown terminal profile: " + profile)
    if src_h not in terminals:
        raise ContractError("terminal grid lacks H%d for profile %s" % (src_h, profile))
    return terminals[src_h], (receipts.get(str(src_h)) or receipts.get(src_h)), src_h


def finalize_episode_summary(summary: Dict[str, Any], ep_dir: Path) -> None:
    v0.write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    v0.write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})


def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps"), 999) == 0 and si(e.get("initial_failed_steps"), 999) == 0 and si(e.get("final_failed_steps"), 999) == 0


def no_safety_regression(short: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(short.get("success")):
        return False
    if bool(short.get("constraint")) and not bool(ref.get("constraint")):
        return False
    for k in ("solver_failure_steps", "initial_failed_steps", "final_failed_steps"):
        if si(short.get(k), 0) > si(ref.get(k), 0):
            return False
    return True


def metric_sum(e: Mapping[str, Any], field: str) -> float:
    return sf((e.get(field) or {}).get("sum"), 0.0)


def pair_row(a: Mapping[str, Any], b: Mapping[str, Any], short_h: int, ref_h: int) -> Dict[str, Any]:
    a_solver = metric_sum(a, "solver_attempt_timing_s")
    b_solver = metric_sum(b, "solver_attempt_timing_s")
    a_dec = metric_sum(a, "decision_timing_s")
    b_dec = metric_sum(b, "decision_timing_s")
    a_phys = sf(a.get("physical_constraint_cost"), 0.0)
    b_phys = sf(b.get("physical_constraint_cost"), 0.0)
    return {
        "state_id": a.get("state_id"),
        "terminal_profile": a.get("terminal_profile") or a.get("terminal_mode"),
        "repeat": si(a.get("repeat")),
        "short_h": short_h,
        "ref_h": ref_h,
        "solver_relative_saving": (b_solver - a_solver) / b_solver if b_solver > 0 else None,
        "decision_relative_saving": (b_dec - a_dec) / b_dec if b_dec > 0 else None,
        "solver_sum_short_s": a_solver,
        "solver_sum_ref_s": b_solver,
        "decision_sum_short_s": a_dec,
        "decision_sum_ref_s": b_dec,
        "physical_gain_ref_minus_short": b_phys - a_phys,
        "total_gain_ref_minus_short": sf(b.get("total_cost"), 0.0) - sf(a.get("total_cost"), 0.0),
        "short_success": bool(a.get("success")),
        "ref_success": bool(b.get("success")),
        "short_constraint": bool(a.get("constraint")),
        "ref_constraint": bool(b.get("constraint")),
        "short_solver_failure_steps": si(a.get("solver_failure_steps")),
        "ref_solver_failure_steps": si(b.get("solver_failure_steps")),
        "safety_ok_vs_ref": no_safety_regression(a, b),
        "short_opt_x_sizes": sorted(set(si(x) for x in (a.get("opt_x_sizes_observed") or []))),
        "ref_opt_x_sizes": sorted(set(si(x) for x in (b.get("opt_x_sizes_observed") or []))),
        "short_steps": si(a.get("steps")),
        "ref_steps": si(b.get("steps")),
    }


def analyze(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    by = {(str(e.get("state_id")), str(e.get("terminal_profile") or e.get("terminal_mode")), si(e.get("repeat")), si(e.get("true_mpc_n_horizon"))): e for e in episodes}
    targets = {str(t.get("state_id")): t for t in protocol.get("target_selection", {}).get("targets", [])}
    schedule = protocol.get("arms", {}).get("schedule") or []
    groups = sorted(set((str(s["state_id"]), str(s["terminal_profile"]), si(s["repeat"])) for s in schedule))
    block_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for sid, profile, rep in groups:
        hs = {h: by.get((sid, profile, rep, h)) for h in TRUE_HORIZONS}
        missing = [h for h, e in hs.items() if e is None]
        dims = {str(h): sorted(set(si(x) for x in ((hs[h] or {}).get("opt_x_sizes_observed") or []))) for h in TRUE_HORIZONS}
        dim_mono = False
        if not missing and all(dims[str(h)] for h in TRUE_HORIZONS):
            dim_mono = max(dims["10"]) < min(dims["15"]) < min(dims["25"])
        safe_block = False
        if not missing:
            safe_block = bool(no_safety_regression(hs[10], hs[15]) and no_safety_regression(hs[10], hs[25]) and no_safety_regression(hs[15], hs[25]))  # type: ignore[arg-type]
            pair_rows.append(pair_row(hs[10], hs[15], 10, 15))  # type: ignore[arg-type]
            pair_rows.append(pair_row(hs[10], hs[25], 10, 25))  # type: ignore[arg-type]
            pair_rows.append(pair_row(hs[15], hs[25], 15, 25))  # type: ignore[arg-type]
        block_rows.append({"state_id": sid, "terminal_profile": profile, "repeat": rep, "missing_horizons": missing, "opt_x_sizes_by_h": dims, "dimension_monotonic_H10_H15_H25": dim_mono, "safety_block_ok": safe_block})

    h10_h15 = [r for r in pair_rows if r["short_h"] == 10 and r["ref_h"] == 15]
    med_solver_10_15 = median([r["solver_relative_saving"] for r in h10_h15 if r.get("solver_relative_saving") is not None])
    med_dec_10_15 = median([r["decision_relative_saving"] for r in h10_h15 if r.get("decision_relative_saving") is not None])
    implementation_pass = all(r["dimension_monotonic_H10_H15_H25"] for r in block_rows) and len(block_rows) == 16
    safety_pass = all(r["safety_block_ok"] for r in block_rows) and len(block_rows) == 16
    compute_pass = bool(med_solver_10_15 is not None and med_dec_10_15 is not None and med_solver_10_15 >= MIN_REL_SAVING and med_dec_10_15 >= MIN_REL_SAVING)

    # Physical tradeoff is evaluated on median repeated costs per state/profile,
    # excluding states whose frozen role explicitly marks them as controls.
    state_profile_rows: List[Dict[str, Any]] = []
    for sid in sorted(set(str(e.get("state_id")) for e in episodes)):
        role = str((targets.get(sid) or {}).get("role") or "unknown")
        noncontrol = "control" not in role.lower() and "negative" not in role.lower()
        for profile in TERMINAL_PROFILES:
            costs = {h: [sf(e.get("physical_constraint_cost"), 0.0) for e in episodes if str(e.get("state_id")) == sid and str(e.get("terminal_profile")) == profile and si(e.get("true_mpc_n_horizon")) == h] for h in TRUE_HORIZONS}
            med = {h: median(costs[h]) for h in TRUE_HORIZONS}
            if med[10] is None or med[15] is None:
                no_worse = False
                tol = None
                gain = None
            else:
                tol = max(2.0, 0.05 * abs(float(med[15])))
                gain = float(med[15]) - float(med[10])
                no_worse = gain >= -tol
            state_profile_rows.append({"state_id": sid, "role": role, "terminal_profile": profile, "noncontrol": noncontrol, "median_physical_cost_by_h": {str(h): med[h] for h in TRUE_HORIZONS}, "h10_vs_h15_physical_gain": gain, "no_worse_than_H15_with_tol": no_worse, "no_worse_tolerance": tol})
    noncontrol_rows = [r for r in state_profile_rows if r["noncontrol"]]
    no_worse_count = sum(1 for r in noncontrol_rows if r["no_worse_than_H15_with_tol"])
    physical_pass = bool(noncontrol_rows and no_worse_count >= int(math.ceil(0.5 * len(noncontrol_rows))))
    strict_improvement_rows = [r for r in state_profile_rows if r.get("h10_vs_h15_physical_gain") is not None and float(r["h10_vs_h15_physical_gain"]) > 0.0]

    pass_to_learning = bool(implementation_pass and safety_pass and compute_pass and physical_pass)
    if pass_to_learning:
        next_action = "freeze selector/value-modeling experiment using true controller cache across >=3 seeds with fair fixed-H baselines; still development-only until independent validation"
    elif compute_pass and safety_pass and not physical_pass:
        next_action = "pivot to terminal-value/modeling/objective or scenario-opportunity design; true-H is faster but physical tradeoff is not acceptable"
    else:
        next_action = "do not train/refit selector; diagnose failed implementation/safety/compute gate and pivot away from sparse-label sweeps"
    return {
        "block_rows": block_rows,
        "pair_rows": pair_rows,
        "state_profile_physical_rows": state_profile_rows,
        "h10_h15_solver_relative_saving_summary": finite_summary([r["solver_relative_saving"] for r in h10_h15 if r.get("solver_relative_saving") is not None]),
        "h10_h15_decision_relative_saving_summary": finite_summary([r["decision_relative_saving"] for r in h10_h15 if r.get("decision_relative_saving") is not None]),
        "median_solver_relative_saving_H10_vs_H15": med_solver_10_15,
        "median_decision_relative_saving_H10_vs_H15": med_dec_10_15,
        "implementation_pass": implementation_pass,
        "safety_pass": safety_pass,
        "compute_pass": compute_pass,
        "physical_pass": physical_pass,
        "noncontrol_physical_no_worse_count": no_worse_count,
        "noncontrol_physical_block_count": len(noncontrol_rows),
        "strict_physical_improvement_rows": strict_improvement_rows,
        "pass_to_selector_or_value_learning_design": pass_to_learning,
        "train_or_refit_now": False,
        "next_action": next_action,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle true variable-H broader block v0 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only true variable-H branch block; no validation64, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['development_branch_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['development_control_step_upper_bound']}`.",
        "",
        "## Gates",
        "",
        f"- Implementation dimension monotonic H10<H15<H25: `{a['implementation_pass']}`.",
        f"- Safety/no solver regression in paired blocks: `{a['safety_pass']}`.",
        f"- Compute gate (median H10-vs-H15 solver and decision saving >=15%): `{a['compute_pass']}`; solver `{a['median_solver_relative_saving_H10_vs_H15']}`, decision `{a['median_decision_relative_saving_H10_vs_H15']}`.",
        f"- Physical tradeoff gate on non-control state/profile blocks: `{a['physical_pass']}` ({a['noncontrol_physical_no_worse_count']}/{a['noncontrol_physical_block_count']} no-worse within tolerance).",
        f"- Pass to selector/value-learning design now: `{a['pass_to_selector_or_value_learning_design']}`.",
        "",
        "## Physical state/profile medians",
        "",
        "| state | role | profile | median physical H10 | H15 | H25 | H10 gain vs H15 | no-worse |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in a["state_profile_physical_rows"]:
        med = r["median_physical_cost_by_h"]
        lines.append("| `%s` | `%s` | `%s` | %s | %s | %s | %s | `%s` |" % (
            r["state_id"], r["role"], r["terminal_profile"],
            "NA" if med.get("10") is None else "%.6g" % float(med["10"]),
            "NA" if med.get("15") is None else "%.6g" % float(med["15"]),
            "NA" if med.get("25") is None else "%.6g" % float(med["25"]),
            "NA" if r.get("h10_vs_h15_physical_gain") is None else "%.6g" % float(r["h10_vs_h15_physical_gain"]),
            bool(r["no_worse_than_H15_with_tol"]),
        ))
    lines += [
        "",
        "## Decision",
        "",
        str(a["next_action"]),
        "",
        "This remains development evidence only. It does not claim validation success, general acceleration, or ORIGINAL reproduction.",
        "",
        f"Backup request after run: `{raw['backup_request_after_run']}`.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run_block(backup_proof: Path) -> int:
    info = verify_protocol()
    protocol = info["protocol"]
    min_time = max(t for t in [file_mtime_utc(SOURCE), file_mtime_utc(PROTOCOL_JSON), parse_time(info["freeze_done"].get("created_utc")), parse_time(info["v0b_done"].get("created_utc"))] if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, NAME)
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError("partial run output exists; inspect before rerun: " + rel(RUN_DIR))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    v0.SMOKE_DIR = RUN_DIR

    bank = read_json(Path(v0.STAGE1_BANK))
    selected_cases = bank.get("selected_cases_full") or bank.get("selected_cases")
    if not isinstance(selected_cases, list):
        raise ContractError("stage1 selected cases unavailable")
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

    started = now_utc()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started.isoformat(), "pid": os.getpid(), "method": NAME, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})

    episodes: List[Dict[str, Any]] = []
    schedule = protocol.get("arms", {}).get("schedule") or []
    for item0 in schedule:
        item = make_run_item(item0, info["target_by_state"])
        h = int(item["true_mpc_n_horizon"])
        profile = str(item["terminal_profile"])
        terminal_tuple, receipt, source_h = terminal_for(profile, h, terminals, terminal_receipts)
        case = selected_cases[int(item["case"])]
        summary = v0.run_true_h_episode(item, case, terminal_tuple)
        summary["repeat"] = int(item["repeat"])
        summary["terminal_profile"] = profile
        summary["target_role"] = item.get("role")
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
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_BROADER_BLOCK_V0_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup broader true-variable-H development block outputs before any further simulation/training/refit", "backup_required_before_more_simulations": True, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "artifacts": [rel(RUN_DIR), rel(STATE_RUN), rel(PROTOCOL_JSON), rel(SOURCE), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_true_variable_horizon_broader_block_not_validation_not_final_test",
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
        "interpretation_limits": ["development-only", "not validation/model selection", "not final test", "not ORIGINAL SAC", "does not infer speed from H alone; measured solver/decision timing reported"],
    }
    write_json(RUN_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle true variable-H broader block v0 run

UTC: {created.isoformat()}. Development-only broader true variable-H block completed: {len(episodes)} branch episodes, {control_steps} control steps, validation64 closed, sealed test closed, no training/refit. Gates: implementation={analysis['implementation_pass']}, safety={analysis['safety_pass']}, compute={analysis['compute_pass']}, physical={analysis['physical_pass']}, pass_to_selector_or_value_learning_design={analysis['pass_to_selector_or_value_learning_design']}. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""", MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL_JSON, STATE_RUN, req, backup_proof, FREEZE_DONE, V0B_DONE]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"implementation_pass": analysis["implementation_pass"], "safety_pass": analysis["safety_pass"], "compute_pass": analysis["compute_pass"], "physical_pass": analysis["physical_pass"], "pass_to_selector_or_value_learning_design": analysis["pass_to_selector_or_value_learning_design"], "train_or_refit_now": False}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "headline": analysis, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-backup-request", action="store_true")
    ap.add_argument("--run-block", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-true-variable-horizon-broader-block-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_true_variable_horizon_broader_block_v0:
        raise ContractError("explicit broader-block acknowledgement required")
    if bool(args.write_backup_request) == bool(args.run_block):
        raise ContractError("exactly one of --write-backup-request or --run-block required")
    if args.write_backup_request:
        write_backup_request_only()
        print(json.dumps({"backup_request": rel(REQUEST_BACKUP_BEFORE_RUN), "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if args.backup_proof is None:
        raise ContractError("--run-block requires --backup-proof")
    return run_block(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RUN_DIR / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial outputs; audit completed episode count and hashes before any rerun. If failure occurred before episodes, repair wrapper and re-backup before simulation."})
        raise
