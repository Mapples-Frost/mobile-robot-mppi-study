#!/usr/bin/env python3
"""Vehicle v1d compute-tradeoff repeated timing confirmation runner v0.

Frozen protocol:
  research_artifacts/aws_protocols/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_frozen_20260929T0410Z.json

This is a development-only diagnostic.  It does not train/refit a selector and
it does not access validation64 or sealed test.  The dry-run mode builds and
audits the blocked repeated-timing schedule from existing v1d artifacts.  The
confirmation mode, for a later cycle after external backup, replays the same
v1d branch states with repeated randomized blocks to test whether the observed
H10-dominated local compute labels survive measured whole-decision and
solver-attempt timing noise without control/safety loss.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402

STAMP = "20260929T0410Z"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_frozen_20260929T0410Z.json"
V1D_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"
V1D_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json"
SOLVER_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/raw.json"
SOLVER_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/completed.json"
H10_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_20260929T0350Z/completed.json"
H10_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_20260929T0350Z/raw.json"
DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_dryrun_{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_run_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_dryrun_{STAMP}.md"
STATE_RUN = ROOT / f"research_artifacts/aws_state/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_RUN = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_V1D_COMPUTE_TRADEOFF_REPEATED_TIMING_CONFIRMATION_V0_RUN_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-v1d-compute-tradeoff-repeated-timing-confirmation-v0-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-v1d-compute-tradeoff-repeated-timing-confirmation-v0-run-{STAMP}"
FIRST_SUPERVISOR_EVENT = dt.datetime(2026, 9, 26, 10, 55, 29, 419331, tzinfo=dt.timezone.utc)

PREFIX_H = 15
MAX_STEPS = 150


class ContractError(RuntimeError):
    pass


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
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


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


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v1d.parse_time(value)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def median(xs: Iterable[float], default: float = 0.0) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return default
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def summary_stats(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "sum": 0.0}
    return {"count": len(vals), "min": vals[0], "median": median(vals), "mean": sum(vals) / len(vals), "max": vals[-1], "sum": sum(vals)}


def completed_ok(path: Path, allow_existing_validation_aggregate: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is True or obj.get("sealed_test_bank_opened") is True or obj.get("test_accessed") is True:
        raise ContractError(f"sealed-test flag true in {rel(path)}")
    if not allow_existing_validation_aggregate and obj.get("validation64_bank_reopened") is True:
        raise ContractError(f"validation64 bank reopened in {rel(path)}")
    return obj


def query_tokens() -> Dict[str, Any]:
    for db in (ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT.parent / "research.sqlite"):
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db)); cur = con.cursor(); total = 0; found = False; by_table = {}
            for (table,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "total_tokens" in cols:
                    val = int(cur.execute(f"SELECT COALESCE(SUM(total_tokens),0) FROM {table}").fetchone()[0] or 0)
                    total += val; found = True; by_table[table] = val
            con.close()
            if found:
                return {"available": True, "path": rel(db), "total_tokens": total, "by_table": by_table}
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": repr(exc)}
    return {"available": False, "reason": "research.sqlite not found in repository-visible candidate paths"}


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def load_inputs() -> Dict[str, Any]:
    for p in (PROTOCOL_JSON, V1D_RAW, V1D_DONE, SOLVER_RAW, SOLVER_DONE, H10_DONE, H10_RAW):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_frozen_20260929T0410Z":
        raise ContractError("unexpected compute-tradeoff protocol id")
    access = protocol.get("access_rules") or {}
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if access.get(key) is not False:
            raise ContractError(f"protocol access flag must be false: {key}")
    design = protocol.get("rollout_design") or {}
    if int(design.get("prefix_horizon", -1)) != PREFIX_H or int(design.get("max_steps_per_episode", -1)) != MAX_STEPS:
        raise ContractError("protocol horizon/max-step mismatch")
    if int(design.get("new_training_episodes", -1)) != 0 or int(design.get("new_gradient_steps", -1)) != 0 or int(design.get("new_refit_steps", -1)) != 0:
        raise ContractError("protocol unexpectedly permits training/refit")
    v1d_done = completed_ok(V1D_DONE)
    solver_done = completed_ok(SOLVER_DONE)
    h10_done = completed_ok(H10_DONE, allow_existing_validation_aggregate=True)
    v1d_raw = read_json(V1D_RAW)
    solver_raw = read_json(SOLVER_RAW)
    h10_raw = read_json(H10_RAW)
    if v1d_raw.get("sealed_test_accessed") is not False or solver_raw.get("sealed_test_accessed") is not False or h10_raw.get("sealed_test_accessed") is not False:
        raise ContractError("input raw has unexpected sealed-test access")
    if solver_raw.get("decision", {}).get("train_or_refit_now") is not False:
        raise ContractError("solver timing diagnostic no longer blocks training/refit")
    if h10_raw.get("decision", {}).get("train_or_refit_now") is not False:
        raise ContractError("H10 absorption diagnostic no longer blocks training/refit")
    return {"protocol": protocol, "v1d_done": v1d_done, "solver_done": solver_done, "h10_done": h10_done, "v1d_raw": v1d_raw, "solver_raw": solver_raw, "h10_raw": h10_raw}


def build_schedule(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = inputs["protocol"]
    v1d_raw = inputs["v1d_raw"]
    solver_raw = inputs["solver_raw"]
    selected = {str(t["state_id"]): t for t in (v1d_raw.get("target_selection", {}).get("selected_targets") or [])}
    solver_rows = {str(r["state_id"]): r for r in (solver_raw.get("per_state") or [])}
    state_spec = protocol.get("selected_states") or {}
    labelled = [str(x) for x in state_spec.get("h10_labelled_compute_states") or []]
    controls = [str(x) for x in state_spec.get("negative_control_states_no_both_relaxed_label") or []]
    all_states = labelled + controls
    if len(labelled) != 6 or len(controls) != 4 or len(set(all_states)) != 10:
        raise ContractError("unexpected protocol selected-state counts")
    missing = [s for s in all_states if s not in selected or s not in solver_rows]
    if missing:
        raise ContractError(f"selected states absent from v1d/solver inputs: {missing}")
    design = protocol["rollout_design"]
    terminal_modes = [str(x) for x in design["post_branch_terminal_modes"]]
    base_h = [int(x) for x in design["base_branch_horizons"]]
    extra_by_state = {str(k): [int(x) for x in v] for k, v in (design.get("extra_branch_horizons_by_state") or {}).items()}
    blocks = int(design["repeat_blocks"])
    seed = int(design["order_seed"])
    rows: List[Dict[str, Any]] = []
    for block in range(blocks):
        block_rows: List[Dict[str, Any]] = []
        for sid in all_states:
            target = selected[sid]
            solver = solver_rows[sid]
            role = "h10_labelled_compute" if sid in labelled else "negative_control_no_both_relaxed"
            horizons = sorted(set(base_h + extra_by_state.get(sid, [])))
            for mode in terminal_modes:
                for h in horizons:
                    block_rows.append({
                        "state_id": sid,
                        "case": int(target["full_case_index"]),
                        "source_candidate_index": int(target.get("source_candidate_index", -1)),
                        "selection_group": str(target.get("selection_group")),
                        "target_index": int(target.get("target_index", -1)),
                        "target_role": str(target.get("target_role")),
                        "compute_tradeoff_role": role,
                        "branch_step": int(target["branch_step"]),
                        "horizon": int(h),
                        "terminal_mode": mode,
                        "prefix_horizon": PREFIX_H,
                        "repeat_block": int(block),
                        "solver_diagnostic_both_relaxed_horizons": solver.get("both_relaxed_horizons") or [],
                        "solver_diagnostic_both_strict_horizons": solver.get("both_strict_horizons") or [],
                        "expected_best_both_relaxed_horizon": None if not solver.get("best_both_relaxed") else int(solver["best_both_relaxed"].get("horizon")),
                        "run_kind": "compute_tradeoff_repeated_timing_confirmation_v0",
                    })
        rng = random.Random(seed + block)
        rng.shuffle(block_rows)
        for j, row in enumerate(block_rows):
            row["execution_index"] = int(4100 + block * 1000 + j)
            rows.append(row)
    declared = int(design["episodes_exact"])
    if len(rows) != declared:
        raise ContractError(f"schedule length {len(rows)} != declared {declared}")
    return {"episodes": rows, "state_ids": all_states, "h10_labelled": labelled, "negative_controls": controls, "declared_episodes": declared, "control_step_upper_bound": int(design["control_step_upper_bound"]), "repeat_blocks": blocks, "order_seed": seed}


def write_summary_dryrun(raw: Mapping[str, Any]) -> None:
    h = raw["schedule_audit"]
    lines = [
        "# Vehicle v1d compute-tradeoff repeated timing confirmation dry-run v0",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations/training/refit; no validation64 or sealed-test access.",
        "",
        "## Headline",
        "",
        f"- Frozen schedule episodes: `{h['episodes']}`; state count `{h['state_count']}`; H10-labelled states `{h['h10_labelled_state_count']}`; negative controls `{h['negative_control_state_count']}`.",
        f"- Repeat blocks: `{h['repeat_blocks']}`; terminal modes `{h['terminal_modes']}`; horizon counts `{h['horizon_counts']}`.",
        f"- Control-step cap for later confirmation: `{h['control_step_upper_bound']}`.",
        f"- Existing H10 absorption context: `{raw['h10_absorption_context']['classification']}`; fixed H10 success `{raw['h10_absorption_context']['H10_success']}` vs H25 `{raw['h10_absorption_context']['H25_success']}`, physical ratio `{raw['h10_absorption_context']['H10_physical_ratio_vs_H25']}`.",
        "- Train/refit now: `False`. Actual repeated timing remains backup-gated.",
        "",
        "## Next action after verified backup",
        "",
        "Run the same source with `--run-confirmation --backup-proof <post-dryrun-proof>` under the legacy interpreter. Passing the predeclared timing gate would only authorize a compact IMPROVED compute-safe selector/refit design; it would not be a reproduction or speed claim.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dryrun() -> int:
    if (DRYRUN_DIR / "completed.json").exists():
        completed_ok(DRYRUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(DRYRUN_DIR / "completed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if DRYRUN_DIR.exists() and any(DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    inputs = load_inputs()
    schedule = build_schedule(inputs)
    now = dt.datetime.now(dt.timezone.utc)
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    terminal_modes = sorted(set(r["terminal_mode"] for r in schedule["episodes"]))
    horizon_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    for r in schedule["episodes"]:
        horizon_counts[str(r["horizon"])] = horizon_counts.get(str(r["horizon"]), 0) + 1
        role_counts[str(r["compute_tradeoff_role"])] = role_counts.get(str(r["compute_tradeoff_role"]), 0) + 1
    h10_head = inputs["h10_done"].get("headline") or {}
    raw = {
        "created_utc": now.isoformat(),
        "method": "vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_dryrun_no_simulation",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "elapsed_since_first_supervisor_event_seconds": (now - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "server_api_total_tokens_best_effort": query_tokens(),
        "inputs": {
            "protocol": rel(PROTOCOL_JSON), "protocol_sha256": sha256(PROTOCOL_JSON),
            "v1d_raw": rel(V1D_RAW), "v1d_raw_sha256": sha256(V1D_RAW),
            "solver_raw": rel(SOLVER_RAW), "solver_raw_sha256": sha256(SOLVER_RAW),
            "h10_absorption_completed": rel(H10_DONE), "h10_absorption_completed_sha256": sha256(H10_DONE),
        },
        "schedule_audit": {
            "episodes": len(schedule["episodes"]),
            "state_count": len(schedule["state_ids"]),
            "h10_labelled_state_count": len(schedule["h10_labelled"]),
            "negative_control_state_count": len(schedule["negative_controls"]),
            "repeat_blocks": schedule["repeat_blocks"],
            "terminal_modes": terminal_modes,
            "horizon_counts": horizon_counts,
            "role_counts": role_counts,
            "control_step_upper_bound": schedule["control_step_upper_bound"],
            "order_seed": schedule["order_seed"],
        },
        "h10_absorption_context": {
            "classification": h10_head.get("classification"),
            "H10_success": f"{h10_head.get('validation_H10_success_count')}/{h10_head.get('validation_H10_episodes')}",
            "H25_success": f"{h10_head.get('validation_H25_success_count')}/{h10_head.get('validation_H25_episodes')}",
            "H10_failures": h10_head.get("validation_H10_failures"),
            "H10_physical_ratio_vs_H25": h10_head.get("validation_H10_physical_cost_ratio_vs_H25"),
            "H10_decision_rel_gain_vs_H25": h10_head.get("validation_H10_decision_rel_gain_vs_H25"),
            "global_H10_noninferior_success_and_physical_3pct": h10_head.get("global_H10_noninferior_success_and_physical_3pct"),
        },
        "schedule": schedule,
        "decision": {
            "train_or_refit_now": False,
            "run_confirmation_now": False,
            "reason": "dry-run only; repeated timing simulation requires verified external backup covering protocol/source/dry-run and recent H10 absorption artifacts",
            "next_action": "after backup, run --run-confirmation under legacy interpreter; analyse strict/relaxed H10 timing gate before any selector/refit",
        },
        "backup_request_before_run": rel(REQUEST_BACKUP_BEFORE_RUN),
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_json(DRYRUN_DIR / "schedule.json", schedule)
    write_summary_dryrun(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text(f"# Vehicle v1d compute-tradeoff repeated timing dry-run\n\nUTC: {now.isoformat()}. Built {len(schedule['episodes'])}-episode repeated timing schedule over 10 v1d states; no simulations/training/validation/test. Backup required before run.\n", encoding="utf-8")
    write_json(REQUEST_BACKUP_BEFORE_RUN, {
        "requested_utc": now.isoformat(),
        "reason": "backup protocol/source/dry-run/H10 absorption artifacts before v1d compute-tradeoff repeated timing rollouts",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_episodes": len(schedule["episodes"]),
        "planned_control_step_upper_bound": schedule["control_step_upper_bound"],
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(PROTOCOL_JSON), rel(Path(__file__).resolve()), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(REQUEST_BACKUP_BEFORE_RUN), rel(H10_DONE), rel(SOLVER_DONE), rel(V1D_DONE)],
    })
    block = f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle v1d compute-tradeoff repeated timing confirmation dry-run

UTC: {now.isoformat()}. No-simulation dry-run built a frozen {len(schedule['episodes'])}-episode repeated-timing confirmation schedule over 10 v1d development states: 6 H10-labelled compute states plus 4 no-label negative controls, 3 repeat blocks, terminal modes {terminal_modes}, horizon counts {horizon_counts}. No validation64-bank or sealed-test access and no training/refit. Existing H10 absorption context remains negative for constant fixed-H10 as a global control/safety baseline. Actual repeated timing rollouts are blocked until external backup covers `{rel(REQUEST_BACKUP_BEFORE_RUN)}`, the protocol/source/dry-run artifacts, and the recent v1d/H10 diagnostics.
"""
    append_docs(block, MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_DRYRUN, REQUEST_BACKUP_BEFORE_RUN, PROTOCOL_JSON, Path(__file__).resolve(), V1D_DONE, SOLVER_DONE, H10_DONE]
    write_json(DRYRUN_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": now.isoformat(),
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "planned_episodes": len(schedule["episodes"]),
        "planned_control_step_upper_bound": schedule["control_step_upper_bound"],
        "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN),
        "headline": {
            "schedule_episodes": len(schedule["episodes"]),
            "h10_labelled_state_count": len(schedule["h10_labelled"]),
            "negative_control_state_count": len(schedule["negative_controls"]),
            "train_or_refit_now": False,
            "confirmation_blocked_until_backup": True,
            "next_action": "after backup, run repeated timing confirmation under legacy interpreter",
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(DRYRUN_DIR / "completed.json"), "summary": rel(DRYRUN_DIR / "summary.md"), "planned_episodes": len(schedule["episodes"]), "planned_control_step_upper_bound": schedule["control_step_upper_bound"], "h10_labelled_states": len(schedule["h10_labelled"]), "negative_controls": len(schedule["negative_controls"]), "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "train_or_refit_now": False}, sort_keys=True), flush=True)
    return 0


def aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "repeat_count": len(rows),
        "success_count": sum(1 for r in rows if bool(r.get("success"))),
        "constraint_count": sum(1 for r in rows if bool(r.get("constraint"))),
        "initial_failed_steps_sum": sum(int(r.get("initial_failed_steps", 0)) for r in rows),
        "final_failed_steps_sum": sum(int(r.get("final_failed_steps", 0)) for r in rows),
        "solver_failure_steps_sum": sum(int(r.get("solver_failure_steps", 0)) for r in rows),
        "steps_median": median(int(r.get("steps", 0)) for r in rows),
        "physical_median": median(fnum(r.get("continuation_physical_constraint_cost_from_branch")) for r in rows),
        "total_median": median(fnum(r.get("continuation_total_cost_from_branch")) for r in rows),
        "decision_sum_median_s": median(fnum((r.get("decision_timing_s") or {}).get("sum")) for r in rows),
        "solver_attempt_sum_median_s": median(fnum((r.get("solver_attempt_timing_s") or {}).get("sum")) for r in rows),
        "decision_sum_stats_s": summary_stats(fnum((r.get("decision_timing_s") or {}).get("sum")) for r in rows),
        "solver_attempt_sum_stats_s": summary_stats(fnum((r.get("solver_attempt_timing_s") or {}).get("sum")) for r in rows),
    }


def no_safety_regression_agg(cand: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    return bool(
        int(cand["success_count"]) >= int(ref["success_count"])
        and int(cand["constraint_count"]) <= int(ref["constraint_count"])
        and int(cand["initial_failed_steps_sum"]) <= int(ref["initial_failed_steps_sum"])
        and int(cand["final_failed_steps_sum"]) <= int(ref["final_failed_steps_sum"])
        and int(cand["solver_failure_steps_sum"]) <= int(ref["solver_failure_steps_sum"])
    )


def analyze_repeats(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], schedule: Mapping[str, Any]) -> Dict[str, Any]:
    by: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    for e in episodes:
        by.setdefault((str(e["state_id"]), str(e["terminal_mode"]), int(e["branch_horizon"])), []).append(e)
    terminal_modes = [str(x) for x in protocol["rollout_design"]["post_branch_terminal_modes"]]
    h10_states = set(schedule["h10_labelled"])
    rows = []
    strict_count_labelled = 0
    strict_count_control = 0
    relaxed_count_labelled = 0
    relaxed_count_control = 0
    for sid in schedule["state_ids"]:
        per_mode: Dict[str, Any] = {}
        strict_modes = []
        relaxed_modes = []
        for mode in terminal_modes:
            arms = {h: aggregate_arm(by.get((sid, mode, h), [])) for h in sorted({k[2] for k in by if k[0] == sid and k[1] == mode})}
            if 10 not in arms or 15 not in arms or 25 not in arms:
                per_mode[mode] = {"missing_required_arm": True, "arms": arms}
                strict_modes.append(False); relaxed_modes.append(False); continue
            h10, h15, h25 = arms[10], arms[15], arms[25]
            dec_gain15 = h15["decision_sum_median_s"] - h10["decision_sum_median_s"]
            sol_gain15 = h15["solver_attempt_sum_median_s"] - h10["solver_attempt_sum_median_s"]
            dec_gain25 = h25["decision_sum_median_s"] - h10["decision_sum_median_s"]
            sol_gain25 = h25["solver_attempt_sum_median_s"] - h10["solver_attempt_sum_median_s"]
            dec_rel15 = dec_gain15 / h15["decision_sum_median_s"] if h15["decision_sum_median_s"] > 0 else 0.0
            sol_rel15 = sol_gain15 / h15["solver_attempt_sum_median_s"] if h15["solver_attempt_sum_median_s"] > 0 else 0.0
            phys_loss15 = h10["physical_median"] - h15["physical_median"]
            phys_loss25 = h10["physical_median"] - h25["physical_median"]
            safe15 = no_safety_regression_agg(h10, h15)
            safe25 = no_safety_regression_agg(h10, h25)
            strict = bool(dec_rel15 >= 0.05 and sol_rel15 >= 0.05 and dec_gain25 > 0.0 and sol_gain25 > 0.0 and safe15 and safe25 and phys_loss15 <= 0.10 and phys_loss25 <= 0.10)
            relaxed = bool(dec_gain15 > 0.0 and sol_gain15 > 0.0 and safe15 and phys_loss15 <= 0.25)
            strict_modes.append(strict); relaxed_modes.append(relaxed)
            per_mode[mode] = {"arms": arms, "H10_vs_H15": {"decision_gain_s": dec_gain15, "solver_gain_s": sol_gain15, "decision_rel_gain": dec_rel15, "solver_rel_gain": sol_rel15, "physical_loss": phys_loss15, "safe": safe15}, "H10_vs_H25": {"decision_gain_s": dec_gain25, "solver_gain_s": sol_gain25, "physical_loss": phys_loss25, "safe": safe25}, "strict_mode": strict, "relaxed_mode": relaxed}
        strict_state = all(strict_modes) and len(strict_modes) == len(terminal_modes)
        relaxed_state = all(relaxed_modes) and len(relaxed_modes) == len(terminal_modes)
        if sid in h10_states:
            strict_count_labelled += int(strict_state); relaxed_count_labelled += int(relaxed_state)
        else:
            strict_count_control += int(strict_state); relaxed_count_control += int(relaxed_state)
        rows.append({"state_id": sid, "role": "h10_labelled_compute" if sid in h10_states else "negative_control_no_both_relaxed", "strict_confirmed_H10_compute_state": strict_state, "relaxed_confirmed_H10_compute_state": relaxed_state, "per_mode": per_mode})
    pass_gate = bool(strict_count_labelled >= 4 and strict_count_control <= 1)
    return {"state_rows": rows, "strict_confirmed_h10_labelled_count": strict_count_labelled, "relaxed_confirmed_h10_labelled_count": relaxed_count_labelled, "strict_confirmed_negative_control_count": strict_count_control, "relaxed_confirmed_negative_control_count": relaxed_count_control, "pass_to_compact_compute_safe_selector_refit_design": pass_gate, "train_or_refit_now": False, "decision": "freeze compact selector/refit design only after backup" if pass_gate else "do not train/refit; timing opportunity not dense/specific enough"}


def run_confirmation(backup_proof: Path) -> int:
    if not (DRYRUN_DIR / "completed.json").exists():
        raise ContractError("confirmation requires completed dry-run")
    dry_done = completed_ok(DRYRUN_DIR / "completed.json")
    if (RUN_DIR / "completed.json").exists():
        completed_ok(RUN_DIR / "completed.json")
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise ContractError(f"partial run output exists; inspect first: {rel(RUN_DIR)}")
    inputs = load_inputs()
    schedule = build_schedule(inputs)
    min_time = max(t for t in [source_mtime_utc(), parse_time(dry_done.get("created_utc")), parse_time(inputs["h10_done"].get("created_utc"))] if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, "vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_run")
    base_smoke, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError(f"legacy runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    v1d_inputs = v1d.verify_protocol_and_inputs()
    bank = v1d.load_v1c_bank(v1d_inputs["v1c_bank_path"])
    selected_cases_full = bank["selected_cases_full"]
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_run", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(RUN_DIR / "schedule.json", schedule)
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    base_smoke.OUT = RUN_DIR
    base_smoke.PREFIX_H = PREFIX_H
    base_smoke.MAX_STEPS = MAX_STEPS
    base_smoke.MATERIAL_GAIN = 3.0
    episodes: List[Dict[str, Any]] = []
    for item in schedule["episodes"]:
        summary = base_smoke.run_one(item, selected_cases_full[int(item["case"])], terminals, terminal_receipts)
        summary = v1d.update_h15_common_receipt(summary, terminal_receipts)
        summary.update({"phase": "compute_tradeoff_repeated_timing_confirmation", "repeat_block": int(item["repeat_block"]), "source_candidate_index": int(item["source_candidate_index"]), "selection_group": item.get("selection_group"), "target_index": int(item["target_index"]), "target_role": item.get("target_role"), "compute_tradeoff_role": item.get("compute_tradeoff_role")})
        summary = v1d.annotate_physical_prefix(summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule["episodes"]), "control_steps_done": int(sum(int(e.get("steps", 0)) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_horizon", "terminal_mode", "repeat_block", "steps", "success")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    if len(episodes) != int(inputs["protocol"]["rollout_design"]["episodes_exact"]) or control_steps > int(inputs["protocol"]["rollout_design"]["control_step_upper_bound"]):
        raise ContractError("confirmation budget violation")
    analysis = analyze_repeats(episodes, inputs["protocol"], schedule)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1D_COMPUTE_TRADEOFF_REPEATED_TIMING_CONFIRMATION_V0_RUN_%s.json" % created.replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created, "reason": "backup repeated timing confirmation outputs before selector/refit or further simulation", "backup_required_before_more_simulations": True, "episodes": len(episodes), "control_steps": control_steps, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(RUN_DIR), rel(STATE_RUN), rel(req), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON)]})
    raw = {"created_utc": created, "started_utc": started, "method": "vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_run", "classification": "development_IMPROVED_compute_tradeoff_repeated_timing_confirmation_not_validation_not_final_test", "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "backup_proof": backup, "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)}, "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0}, "schedule": schedule, "episodes": episodes, "analysis": analysis, "backup_request_after_run": rel(req), "interpretation_limits": inputs["protocol"].get("interpretation_limits")}
    write_json(RUN_DIR / "raw.json", raw)
    lines = ["# Vehicle v1d compute-tradeoff repeated timing confirmation run v0", "", f"UTC: `{created}`. Development-only repeated timing; no validation64/sealed test/training/refit.", "", f"Episodes `{len(episodes)}`, control steps `{control_steps}`.", "", "## Headline", "", f"- Strict confirmed H10-labelled states: `{analysis['strict_confirmed_h10_labelled_count']}/6`.", f"- Relaxed confirmed H10-labelled states: `{analysis['relaxed_confirmed_h10_labelled_count']}/6`.", f"- Strict confirmed negative controls: `{analysis['strict_confirmed_negative_control_count']}/4`.", f"- Pass to compact compute-safe selector/refit design: `{analysis['pass_to_compact_compute_safe_selector_refit_design']}`.", f"- Decision: `{analysis['decision']}`.", "", f"Backup request: `{rel(req)}`."]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text(f"# Vehicle v1d compute-tradeoff repeated timing run\n\nUTC: {created}. strict_labelled={analysis['strict_confirmed_h10_labelled_count']}/6, strict_controls={analysis['strict_confirmed_negative_control_count']}/4, pass_gate={analysis['pass_to_compact_compute_safe_selector_refit_design']}. No training/refit/validation/test.\n", encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle v1d compute-tradeoff repeated timing confirmation run

UTC: {created}. Development-only repeated timing completed with {len(episodes)} episodes/{control_steps} control steps. Strict confirmed H10-labelled states={analysis['strict_confirmed_h10_labelled_count']}/6, relaxed={analysis['relaxed_confirmed_h10_labelled_count']}/6, strict negative controls={analysis['strict_confirmed_negative_control_count']}/4, pass-to-selector-refit-design={analysis['pass_to_compact_compute_safe_selector_refit_design']}. No validation64/sealed-test access and no training/refit. This is not a speed or superiority claim; backup required before any next simulation/refit.
""", MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_RUN, req, Path(__file__).resolve(), PROTOCOL_JSON, backup_proof]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"strict_confirmed_h10_labelled_count": analysis["strict_confirmed_h10_labelled_count"], "strict_confirmed_negative_control_count": analysis["strict_confirmed_negative_control_count"], "pass_to_compact_compute_safe_selector_refit_design": analysis["pass_to_compact_compute_safe_selector_refit_design"], "train_or_refit_now": False, "next_action": analysis["decision"]}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "strict_confirmed_h10_labelled_count": analysis["strict_confirmed_h10_labelled_count"], "strict_confirmed_negative_control_count": analysis["strict_confirmed_negative_control_count"], "pass_to_compact_compute_safe_selector_refit_design": analysis["pass_to_compact_compute_safe_selector_refit_design"], "validation64_bank_opened": False, "sealed_test_accessed": False, "train_or_refit_now": False}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-confirmation", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-compute-tradeoff-timing-diagnostic", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_compute_tradeoff_timing_diagnostic:
        raise ContractError("explicit acceptance flag required")
    if bool(args.dry_run) == bool(args.run_confirmation):
        raise ContractError("exactly one of --dry-run or --run-confirmation required")
    if args.dry_run:
        return run_dryrun()
    if args.backup_proof is None:
        raise ContractError("--run-confirmation requires --backup-proof")
    return run_confirmation(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else RUN_DIR if "--run-confirmation" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/vehicle_v1d_compute_tradeoff_repeated_timing_confirmation_v0_failure_unknown_{STAMP}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve failure. If dry-run failed, repair only schedule/schema; if run failed, inspect partial outputs before any rerun."})
        raise
