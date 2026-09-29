#!/usr/bin/env python3
"""Repeated stability/timing diagnostic for v1e case-5 positive states.

Hypothesis: the two robust positives found by the v1e common-prefix smoke may be
real local state-dependent H10 opportunities, but they may also be one-run solver,
terminal, or timing artifacts.  This runner freezes and executes a small paired
repeat block: the two case-5 positive H15-prefix states, H10 versus H15, both
terminal modes, three repeats, randomized/block-balanced order.

Development-only IMPROVED diagnostic. No selector/refit/training, no validation64,
no sealed test. It does not claim speed unless repeated measured timing supports it.
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
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_runner as v1e_v0b  # noqa:E402

NAME = "vehicle_stress_v1e_case5_positive_stability_timing_v0"
STAMP = "20260929T0605Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PARENT_SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z"
PARENT_SMOKE_RAW = PARENT_SMOKE_DIR / "raw.json"
PARENT_SMOKE_DONE = PARENT_SMOKE_DIR / "completed.json"
FAILED_POSTDIAG_V0_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0_20260929T0550Z/raw.json"
PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_dryrun_{STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/{NAME}_dryrun_{STAMP}.md"
STATE_RUN = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_RUN = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_CASE5_POSITIVE_STABILITY_TIMING_V0_RUN_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-v1e-case5-positive-stability-timing-v0-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-stress-v1e-case5-positive-stability-timing-v0-run-{STAMP}"

EXPECTED_POSITIVE_STATE_IDS = [
    "v1e_t04_case05_cand148_b053_middle",
    "v1e_t05_case05_cand148_b054_late",
]
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
HORIZONS = [10, 15]
REPEATS = 3
MAX_STEPS = v1e_v0b.MAX_STEPS
EPISODES_EXACT = len(EXPECTED_POSITIVE_STATE_IDS) * len(TERMINAL_MODES) * len(HORIZONS) * REPEATS
CONTROL_STEP_UPPER = EPISODES_EXACT * MAX_STEPS
STATE_DISTANCE_TOL = 1e-5
MATERIAL_GAIN = 3.0
TIMING_RELATIVE_SAVING_GATE = 0.05


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


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(SOURCE.stat().st_mtime, dt.timezone.utc)


def file_mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v1d.parse_time(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError(f"sealed-test flag not false in {rel(path)}")
    if obj.get("validation64_bank_opened") is not False:
        raise ContractError(f"validation64 flag not false in {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if p.exists() and sha256(p) != expected:
                raise ContractError(f"hash mismatch from {rel(path)}: {name}")
    return obj


def finite_summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "mean": sum(xs) / len(xs), "median": pct(0.5), "p95": pct(0.95), "min": xs[0], "max": xs[-1]}


def verify_parent_smoke() -> Dict[str, Any]:
    done = completed_ok(PARENT_SMOKE_DONE, check_hashes=False)
    raw = read_json(PARENT_SMOKE_RAW)
    if raw.get("validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("parent smoke unexpectedly opened validation64 or sealed test")
    analysis = raw.get("analysis") or {}
    rows = analysis.get("state_rows") or []
    positives = [r for r in rows if bool(r.get("robust_positive_state"))]
    ids = sorted(str(r.get("state_id")) for r in positives)
    if ids != sorted(EXPECTED_POSITIVE_STATE_IDS):
        raise ContractError(f"unexpected parent positive states: {ids}")
    if sorted({int(r.get("case")) for r in positives}) != [5]:
        raise ContractError("parent positives are not confined to expected case 5")
    for r in positives:
        if 10 not in [int(x) for x in (r.get("robust_positive_horizons") or [])]:
            raise ContractError(f"positive state lacks robust H10: {r.get('state_id')}")
    if bool(analysis.get("smoke_pass_to_selector_or_value_refit_design")):
        raise ContractError("parent v1e smoke unexpectedly passed selector/refit gate; this artifact check is no longer the next action")
    return {"done": done, "raw": raw, "analysis": analysis, "positives": positives}


def build_schedule(parent_raw: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_schedule = parent_raw.get("schedule") or []
    raw_targets = parent_raw.get("targets") or []
    target_by_id = {str(t.get("state_id")): t for t in raw_targets}
    target_by_idx = {int(t.get("target_index")): t for t in raw_targets}
    selected_targets = []
    for sid in EXPECTED_POSITIVE_STATE_IDS:
        t = target_by_id.get(sid)
        if not t:
            raise ContractError(f"target not found in parent smoke raw: {sid}")
        selected_targets.append(t)
    item_by_key = {}
    for item in base_schedule:
        item_by_key[(int(item["target_index"]), str(item["terminal_mode"]), int(item["horizon"]))] = item
    schedule: List[Dict[str, Any]] = []
    exec_index = 0
    for rep in range(REPEATS):
        blocks: List[Dict[str, Any]] = []
        for t in selected_targets:
            for mode in TERMINAL_MODES:
                order = [15, 10] if (rep + int(t["target_index"]) + len(mode)) % 2 == 0 else [10, 15]
                for h in order:
                    key = (int(t["target_index"]), mode, h)
                    if key not in item_by_key:
                        raise ContractError(f"missing parent schedule item {key}")
                    row = dict(item_by_key[key])
                    row.update({
                        "execution_index": exec_index,
                        "diagnostic_repeat": rep,
                        "diagnostic_pair": "H10_vs_H15_case5_positive",
                        "case_role": target_by_idx[int(row["target_index"])].get("case_role"),
                        "role": target_by_idx[int(row["target_index"])].get("case_role"),
                    })
                    blocks.append(row)
                    exec_index += 1
        # Keep pairing local but randomize the four state/mode blocks deterministically to reduce drift confounding.
        rng = random.Random(202609290605 + rep)
        grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for row in blocks:
            grouped.setdefault((int(row["target_index"]), str(row["terminal_mode"])), []).append(row)
        keys = list(grouped.keys())
        rng.shuffle(keys)
        for key in keys:
            schedule.extend(grouped[key])
    if len(schedule) != EPISODES_EXACT:
        raise ContractError(f"unexpected diagnostic schedule length {len(schedule)}")
    return selected_targets, schedule


def write_protocol(created: dt.datetime, parent: Mapping[str, Any], targets: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]) -> None:
    protocol = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_repeated_positive_state_stability_timing_diagnostic_not_validation_not_final_test",
        "hypothesis": "The two v1e case-5 robust positives are genuine repeated H10-vs-H15 state-continuation opportunities with retained safety and measured decision-time reduction, rather than single-run terminal/solver/timing artifacts.",
        "falsifying_outcomes": [
            "H10 fails to retain material physical gain >=3 versus paired H15 in both terminal modes for both states in at least 2/3 repeats",
            "H10 introduces success/constraint/solver regression versus paired H15",
            "H10 median measured decision time is not lower than H15, or overall relative saving is below 5%",
        ],
        "source_parent_smoke": rel(PARENT_SMOKE_RAW),
        "parent_smoke_completed": rel(PARENT_SMOKE_DONE),
        "positive_state_ids": list(EXPECTED_POSITIVE_STATE_IDS),
        "targets": list(targets),
        "terminal_modes": TERMINAL_MODES,
        "horizons": HORIZONS,
        "repeats": REPEATS,
        "schedule": list(schedule),
        "budgets": {"episodes_exact": EPISODES_EXACT, "control_step_upper_bound": CONTROL_STEP_UPPER, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "gates": {"material_gain_per_pair": MATERIAL_GAIN, "state_distance_tol": STATE_DISTANCE_TOL, "per_state_mode_material_repeats_required": "2/3", "overall_relative_decision_time_saving_required": TIMING_RELATIVE_SAVING_GATE},
        "access_rules": {"validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False},
        "source_hashes": {rel(SOURCE): sha256(SOURCE), rel(PARENT_SMOKE_RAW): sha256(PARENT_SMOKE_RAW), rel(PARENT_SMOKE_DONE): sha256(PARENT_SMOKE_DONE)},
        "locked_decisions": ["development only", "do not train/refit from v1e labels unless a separately frozen gate later supports it", "do not claim speed from H alone; only repeated measured timing is descriptive development evidence"],
    }
    write_json(PROTOCOL_JSON, protocol)


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-v1e case5 positive stability/timing v0 dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Frozen diagnostic",
        "",
        f"- Positive states: `{raw['positive_state_ids']}`.",
        f"- Episodes: `{raw['planned_episodes_exact']}`; control-step cap: `{raw['planned_control_step_upper_bound']}`.",
        f"- Design: repeats `{REPEATS}`, terminal modes `{TERMINAL_MODES}`, horizons `{HORIZONS}`.",
        f"- Protocol: `{raw['protocol_json']}`.",
        "",
        "## Decision",
        "",
        "Run the repeated paired diagnostic only after verified external backup covers this dry-run/protocol/source and the parent smoke outputs. Passing this diagnostic would show stable local case-5 opportunity, not validation success and not a general selector-ready label bank.",
        "",
        f"Backup request before run: `{raw['backup_request_before_run']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    done_path = DRYRUN_DIR / "completed.json"
    if done_path.exists():
        done = completed_ok(done_path, check_hashes=True)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    parent = verify_parent_smoke()
    targets, schedule = build_schedule(parent["raw"])
    write_protocol(created, parent, targets, schedule)
    write_json(REQUEST_BACKUP_BEFORE_RUN, {
        "requested_utc": created.isoformat(),
        "reason": "backup frozen case5 positive repeated stability/timing diagnostic before any additional simulation",
        "backup_required_before_more_simulations": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "planned_episodes_exact": EPISODES_EXACT,
        "planned_control_step_upper_bound": CONTROL_STEP_UPPER,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(SOURCE), rel(PROTOCOL_JSON), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(PARENT_SMOKE_DIR), rel(REQUEST_BACKUP_BEFORE_RUN)],
    })
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": "development_no_simulation_case5_positive_stability_timing_readiness",
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "parent_smoke_completed": rel(PARENT_SMOKE_DONE),
        "parent_smoke_raw": rel(PARENT_SMOKE_RAW),
        "failed_postdiag_v0_raw_preserved": rel(FAILED_POSTDIAG_V0_RAW) if FAILED_POSTDIAG_V0_RAW.exists() else None,
        "protocol_json": rel(PROTOCOL_JSON),
        "positive_state_ids": EXPECTED_POSITIVE_STATE_IDS,
        "planned_episodes_exact": EPISODES_EXACT,
        "planned_control_step_upper_bound": CONTROL_STEP_UPPER,
        "target_count": len(targets),
        "schedule_count": len(schedule),
        "backup_request_before_run": rel(REQUEST_BACKUP_BEFORE_RUN),
        "next_action": "after verified backup, run --run-repeats with legacy interpreter; then decide whether case5 opportunity is stable/timed or an artifact",
        "source_hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL_JSON): sha256(PROTOCOL_JSON), rel(PARENT_SMOKE_RAW): sha256(PARENT_SMOKE_RAW), rel(PARENT_SMOKE_DONE): sha256(PARENT_SMOKE_DONE)},
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text((DRYRUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle stress-v1e case5 positive stability/timing v0 dry-run

UTC: {created.isoformat()}. No-simulation readiness for a repeated paired diagnostic of the two v1e case-5 robust-positive states. Frozen design: {EPISODES_EXACT} development episodes, cap {CONTROL_STEP_UPPER} control steps, H10 vs H15, terminal modes {TERMINAL_MODES}, repeats {REPEATS}; no validation64 or sealed-test access; no training/refit. Run is blocked until verified backup covers `{rel(REQUEST_BACKUP_BEFORE_RUN)}`.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL_JSON, STATE_DRYRUN, REQUEST_BACKUP_BEFORE_RUN, PARENT_SMOKE_RAW, PARENT_SMOKE_DONE]
    write_json(done_path, {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "planned_episodes_exact": EPISODES_EXACT,
        "planned_control_step_upper_bound": CONTROL_STEP_UPPER,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN),
        "headline": {"positive_state_ids": EXPECTED_POSITIVE_STATE_IDS, "planned_episodes_exact": EPISODES_EXACT, "run_blocked_until_verified_backup": True, "train_or_refit_now": False},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(done_path), "summary": rel(DRYRUN_DIR / "summary.md"), "protocol": rel(PROTOCOL_JSON), "planned_episodes_exact": EPISODES_EXACT, "planned_control_step_upper_bound": CONTROL_STEP_UPPER, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN)}, sort_keys=True), flush=True)
    return 0


def decision_time(e: Mapping[str, Any]) -> float:
    return safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0)


def solver_time(e: Mapping[str, Any]) -> float:
    return safe_float((e.get("solver_attempt_timing_s") or {}).get("sum"), 0.0)


def analyze_repeats(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Tuple[int, str, int, int], Mapping[str, Any]] = {}
    for e in episodes:
        by_key[(int(e["target_index"]), str(e["terminal_mode"]), int(e["diagnostic_repeat"]), int(e["branch_horizon"]))] = e
    pair_rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for target_index in sorted({int(e["target_index"]) for e in episodes}):
        for mode in TERMINAL_MODES:
            for rep in range(REPEATS):
                h10 = by_key.get((target_index, mode, rep, 10))
                h15 = by_key.get((target_index, mode, rep, 15))
                if h10 is None or h15 is None:
                    pair = {"target_index": target_index, "terminal_mode": mode, "repeat": rep, "missing_pair": True, "material_positive": False}
                else:
                    dist = v1d.state_distance(v1d.state_tuple(h10.get("branch_previous_state")), v1d.state_tuple(h15.get("branch_previous_state")))
                    prefix_match = bool(h10.get("physical_prefix_equivalence_sha256") == h15.get("physical_prefix_equivalence_sha256"))
                    safety_ok = v1d.no_regression(h10, h15)
                    phys_gain = safe_float(h15.get("continuation_physical_constraint_cost_from_branch")) - safe_float(h10.get("continuation_physical_constraint_cost_from_branch"))
                    total_gain = safe_float(h15.get("continuation_total_cost_from_branch")) - safe_float(h10.get("continuation_total_cost_from_branch"))
                    time_saving = decision_time(h15) - decision_time(h10)
                    solver_saving = solver_time(h15) - solver_time(h10)
                    material = bool(h10.get("branch_reached") and h15.get("branch_reached") and dist <= STATE_DISTANCE_TOL and prefix_match and safety_ok and phys_gain >= MATERIAL_GAIN)
                    pair = {
                        "target_index": target_index,
                        "state_id": h10.get("state_id"),
                        "case": h10.get("case"),
                        "case_role": h10.get("case_role"),
                        "terminal_mode": mode,
                        "repeat": rep,
                        "material_positive": material,
                        "h10_success": bool(h10.get("success")),
                        "h15_success": bool(h15.get("success")),
                        "h10_constraint": bool(h10.get("constraint")),
                        "h15_constraint": bool(h15.get("constraint")),
                        "h10_solver_failure_steps": int(h10.get("solver_failure_steps", 0)),
                        "h15_solver_failure_steps": int(h15.get("solver_failure_steps", 0)),
                        "state_distance": dist,
                        "physical_prefix_matches": prefix_match,
                        "safety_ok_vs_H15": safety_ok,
                        "physical_gain_H10_vs_H15": phys_gain,
                        "total_gain_H10_vs_H15": total_gain,
                        "decision_time_saving_H10_vs_H15_s": time_saving,
                        "solver_time_saving_H10_vs_H15_s": solver_saving,
                        "h10_decision_sum_s": decision_time(h10),
                        "h15_decision_sum_s": decision_time(h15),
                        "h10_solver_sum_s": solver_time(h10),
                        "h15_solver_sum_s": solver_time(h15),
                        "h10_steps": int(h10.get("steps", 0)),
                        "h15_steps": int(h15.get("steps", 0)),
                        "h10_path": h10.get("path"),
                        "h15_path": h15.get("path"),
                    }
                pair_rows.append(pair)
                groups.setdefault((target_index, mode), []).append(pair)
    group_rows: List[Dict[str, Any]] = []
    for (target_index, mode), pairs in sorted(groups.items()):
        material_count = sum(1 for p in pairs if p.get("material_positive"))
        timing_savings = [safe_float(p.get("decision_time_saving_H10_vs_H15_s")) for p in pairs if not p.get("missing_pair")]
        h10_times = [safe_float(p.get("h10_decision_sum_s")) for p in pairs if not p.get("missing_pair")]
        h15_times = [safe_float(p.get("h15_decision_sum_s")) for p in pairs if not p.get("missing_pair")]
        med_h10 = finite_summary(h10_times).get("median") if h10_times else None
        med_h15 = finite_summary(h15_times).get("median") if h15_times else None
        rel_save = None
        if med_h10 is not None and med_h15 and med_h15 > 0:
            rel_save = (float(med_h15) - float(med_h10)) / float(med_h15)
        group_rows.append({
            "target_index": target_index,
            "terminal_mode": mode,
            "pair_count": len(pairs),
            "material_positive_repeats": material_count,
            "material_repeat_gate_2_of_3": material_count >= 2,
            "all_repeats_material": material_count == REPEATS,
            "decision_time_saving_s": finite_summary(timing_savings),
            "h10_decision_sum_s": finite_summary(h10_times),
            "h15_decision_sum_s": finite_summary(h15_times),
            "median_relative_decision_saving": rel_save,
            "timing_gate_positive": bool(rel_save is not None and rel_save >= TIMING_RELATIVE_SAVING_GATE),
        })
    material_pair_count = sum(1 for p in pair_rows if p.get("material_positive"))
    all_group_stable = all(g["material_repeat_gate_2_of_3"] for g in group_rows) and len(group_rows) == len(EXPECTED_POSITIVE_STATE_IDS) * len(TERMINAL_MODES)
    all_group_timed = all(g["timing_gate_positive"] for g in group_rows) and len(group_rows) == len(EXPECTED_POSITIVE_STATE_IDS) * len(TERMINAL_MODES)
    h10_all = [safe_float(p.get("h10_decision_sum_s")) for p in pair_rows if not p.get("missing_pair")]
    h15_all = [safe_float(p.get("h15_decision_sum_s")) for p in pair_rows if not p.get("missing_pair")]
    med_h10_all = finite_summary(h10_all).get("median") if h10_all else None
    med_h15_all = finite_summary(h15_all).get("median") if h15_all else None
    overall_rel = None
    if med_h10_all is not None and med_h15_all and med_h15_all > 0:
        overall_rel = (float(med_h15_all) - float(med_h10_all)) / float(med_h15_all)
    pass_artifact_check = bool(all_group_stable and overall_rel is not None and overall_rel >= TIMING_RELATIVE_SAVING_GATE)
    return {
        "pair_count": len(pair_rows),
        "material_pair_count": material_pair_count,
        "group_rows": group_rows,
        "pair_rows": pair_rows,
        "all_group_stable_2_of_3": all_group_stable,
        "all_group_timing_positive_5pct": all_group_timed,
        "overall_h10_decision_sum_s": finite_summary(h10_all),
        "overall_h15_decision_sum_s": finite_summary(h15_all),
        "overall_median_relative_decision_saving": overall_rel,
        "pass_to_next_design_consideration": pass_artifact_check,
        "decision": {
            "train_or_refit_now": False,
            "if_pass": "treat case5 as a stable local opportunity; next freeze a broader source-supported scenario/state-selection or value-modeling diagnostic, not immediate final validation",
            "if_fail": "classify v1e positives as timing/control artifacts or too narrow; pivot to terminal/modeling or scenario-opportunity diagnosis",
        },
    }


def write_run_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle stress-v1e case5 positive stability/timing v0 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only repeated paired diagnostic; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- Material H10-vs-H15 pairs: `{a['material_pair_count']}/{a['pair_count']}`.",
        f"- All state/mode groups stable at >=2/3 repeats: `{a['all_group_stable_2_of_3']}`.",
        f"- Overall median relative H10 decision-time saving: `{a['overall_median_relative_decision_saving']}`.",
        f"- Pass to next design consideration: `{a['pass_to_next_design_consideration']}`.",
        "",
        "## Group summary",
        "",
        "| target | terminal | material repeats | median rel decision saving | timing gate |",
        "|---:|---|---:|---:|---|",
    ]
    for g in a["group_rows"]:
        lines.append(f"| {g['target_index']} | `{g['terminal_mode']}` | {g['material_positive_repeats']}/{g['pair_count']} | {g['median_relative_decision_saving']} | `{g['timing_gate_positive']}` |")
    lines += [
        "",
        "## Decision",
        "",
        f"- Train/refit now: `{a['decision']['train_or_refit_now']}`.",
        f"- Backup request: `{raw['backup_request_after_run']}`.",
        "- This remains development evidence only. Do not claim validation success or general speedup from this local case-5 diagnostic.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_repeats(backup_proof: Path) -> int:
    dry_path = DRYRUN_DIR / "completed.json"
    completed_ok(dry_path, check_hashes=True)
    if (RUN_DIR / "completed.json").exists():
        done = completed_ok(RUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(RUN_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if RUN_DIR.exists() and any(p.name != "run.lock" for p in RUN_DIR.iterdir()):
        raise ContractError(f"partial run output exists; inspect first: {rel(RUN_DIR)}")
    parent = verify_parent_smoke()
    protocol = read_json(PROTOCOL_JSON)
    schedule = protocol["schedule"]
    min_time = max(t for t in [source_mtime_utc(), file_mtime_utc(PROTOCOL_JSON), parse_time(completed_ok(dry_path).get("created_utc")), parse_time(parent["done"].get("created_utc"))] if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, NAME)
    inputs = v1e_v0b.verify_inputs(load_bank=True)
    bank = read_json(v1e_v0b.v0.STAGE1_BANK)
    selected_cases = bank.get("selected_cases_full") or bank.get("selected_cases")
    base_smoke, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError(f"legacy runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = now_utc().isoformat()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": f"{NAME}_run", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    write_json(RUN_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    base_smoke.OUT = RUN_DIR
    base_smoke.PREFIX_H = v1e_v0b.PREFIX_H
    base_smoke.MAX_STEPS = MAX_STEPS
    base_smoke.MATERIAL_GAIN = MATERIAL_GAIN
    target_by_idx = {int(t["target_index"]): t for t in inputs["targets"]}
    branch_episodes: List[Dict[str, Any]] = []
    for item in schedule:
        case_id = int(item["case"])
        target = target_by_idx[int(item["target_index"])]
        run_item = dict(item)
        run_item["role"] = str(run_item.get("role") or run_item.get("case_role") or target.get("case_role") or "case5_positive")
        summary = base_smoke.run_one(run_item, selected_cases[case_id], terminals, terminal_receipts)
        summary = v1d.update_h15_common_receipt(summary, terminal_receipts)
        summary.update({
            "phase": "v1e_case5_positive_stability_timing_repeat_v0",
            "diagnostic_repeat": int(item["diagnostic_repeat"]),
            "diagnostic_pair": item.get("diagnostic_pair"),
            "source_candidate_index": int(item["source_candidate_index"]),
            "selection_group": item.get("selection_group"),
            "target_index": int(item["target_index"]),
            "case_role": target.get("case_role"),
            "protocol_terminal_label_family": "stress_v1e_case5_repeated_H10_vs_H15",
        })
        summary = v1d.annotate_physical_prefix(summary)
        branch_episodes.append(summary)
        control_steps_done = int(sum(int(e.get("steps", 0)) for e in branch_episodes))
        progress = {"pid": os.getpid(), "episodes_done": len(branch_episodes), "episodes_expected": EPISODES_EXACT, "control_steps_done": control_steps_done, "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination", "branch_reached")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in branch_episodes))
    if len(branch_episodes) != EPISODES_EXACT or control_steps > CONTROL_STEP_UPPER:
        raise ContractError("case5 stability/timing diagnostic budget violation")
    analysis = analyze_repeats(branch_episodes)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_CASE5_POSITIVE_STABILITY_TIMING_V0_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup repeated case5 positive stability/timing diagnostic outputs before further simulation/training/refit", "backup_required_before_more_simulations": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(RUN_DIR), rel(STATE_RUN), rel(SOURCE), rel(PROTOCOL_JSON), rel(req)]})
    raw = {"created_utc": created.isoformat(), "started_utc": started, "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "method": f"{NAME}_run", "classification": "development_IMPROVED_repeated_positive_state_stability_timing_not_validation_not_final_test", "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_proof": backup, "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)}, "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}}, "runtime_preflight": preflight, "budget_declared": {"episodes_exact": EPISODES_EXACT, "control_step_upper_bound": CONTROL_STEP_UPPER, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "budget_actual": {"episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(branch_episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in branch_episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0}, "branch_episodes": branch_episodes, "analysis": analysis, "backup_request_after_run": rel(req), "interpretation_limits": ["development diagnostic only", "not validation/model selection", "not final test", "not training/refit", "not a general speed claim beyond this local repeated block"]}
    write_json(RUN_DIR / "raw.json", raw)
    write_run_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text((RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_RUN} -->
## 2026-09-29 vehicle stress-v1e case5 positive stability/timing v0 run

UTC: {created.isoformat()}. Development-only repeated H10-vs-H15 diagnostic completed on the two v1e case-5 positive states: {len(branch_episodes)} episodes, {control_steps} control steps. Material pairs={analysis['material_pair_count']}/{analysis['pair_count']}, all_group_stable_2_of_3={analysis['all_group_stable_2_of_3']}, overall median relative H10 decision-time saving={analysis['overall_median_relative_decision_saving']}, pass_to_next_design_consideration={analysis['pass_to_next_design_consideration']}. No validation64-bank or sealed-test access; no training/refit. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
""", MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, STATE_RUN, req, PROTOCOL_JSON, PARENT_SMOKE_DONE, PARENT_SMOKE_RAW, dry_path, backup_proof]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(branch_episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"material_pair_count": analysis["material_pair_count"], "pair_count": analysis["pair_count"], "all_group_stable_2_of_3": analysis["all_group_stable_2_of_3"], "overall_median_relative_decision_saving": analysis["overall_median_relative_decision_saving"], "pass_to_next_design_consideration": analysis["pass_to_next_design_consideration"], "train_or_refit_now": False}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(branch_episodes), "control_steps": control_steps, "material_pair_count": analysis["material_pair_count"], "pair_count": analysis["pair_count"], "all_group_stable_2_of_3": analysis["all_group_stable_2_of_3"], "overall_median_relative_decision_saving": analysis["overall_median_relative_decision_saving"], "pass_to_next_design_consideration": analysis["pass_to_next_design_consideration"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-repeats", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-v1e-case5-stability-timing-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_v1e_case5_stability_timing_v0:
        raise ContractError("explicit --i-accept-development-v1e-case5-stability-timing-v0 required")
    if bool(args.dry_run) == bool(args.run_repeats):
        raise ContractError("exactly one of --dry-run or --run-repeats is required")
    if args.dry_run:
        return run_dry_run()
    if args.backup_proof is None:
        raise ContractError("--run-repeats requires --backup-proof")
    return run_repeats(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else RUN_DIR if "--run-repeats" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/{NAME}_failure_unknown_{STAMP}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0 if "--dry-run" in sys.argv else None, "new_control_steps": 0 if "--dry-run" in sys.argv else None, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial output. If dry-run failed, repair source only; if run failed, audit partial repeated pairs before rerun."})
        raise
