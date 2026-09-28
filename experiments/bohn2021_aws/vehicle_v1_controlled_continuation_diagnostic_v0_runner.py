#!/usr/bin/env python3
"""Vehicle V1 controlled-continuation diagnostic runner.

Development-only diagnostic frozen by
research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.*.

The runner replays a common fixed-H15 prefix on selected V1 development cases,
then branches from the same prefix state to a small target-specific horizon set.
It is intended to distinguish genuine state-dependent horizon opportunity from
fixed-H episode-level hindsight labels.  It performs no training/refit, opens no
historical validation64 bank, and never accesses the sealed final test.

A verified external backup proof after the target-mining diagnostic/protocol is
required before any rollout is allowed.
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
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_fixed_h_opportunity_probe_v1_runner as v1probe  # noqa:E402

base = v1probe.base
v1 = base.v1

TASK = "vehicle"
PREFIX_H = 15
MAX_STEPS = 150
ORDER_SEED = 2609287301
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0_20260928"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_controlled_continuation_diagnostic_v0_20260928.md"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.json"
TARGET_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z/completed.json"
TARGET_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z/raw.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-v1-controlled-continuation-diagnostic-v0-20260928"
IMPROVEMENT_ABS_THRESHOLD = 3.0
REQUIRED_CONFIRMED_STATES = 2


class ContractError(RuntimeError):
    """Raised for protocol, budget or access violations."""


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=serial, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(xs.size),
        "sum": float(xs.sum()),
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
        "p95": float(np.percentile(xs, 95)),
        "max": float(xs.max()),
    }


def verify_completed_marker(path: Path, check_hashes: bool = True) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("Completed marker did not pass: %s" % rel(path))
    if check_hashes:
        for name, expected in done.get("hashes", {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError("Completed marker references missing file: %s" % name)
            actual = sha256(p)
            if actual != expected:
                raise ContractError("Hash mismatch for %s" % name)
    return done


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("Backup proof path does not exist: %s" % rel(path))
    proof = read_json(path)
    verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
    if not verified:
        raise ContractError("Backup proof is not verified: %s" % rel(path))
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("Backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("Backup proof lacks commit")
    has_asset = bool(
        proof.get("asset_sha256")
        or proof.get("release_asset_sha256")
        or proof.get("package_sha256")
        or proof.get("packages_this_run")
    )
    if not has_asset:
        raise ContractError("Backup proof lacks a verified release/package SHA")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError("Backup proof lacks parseable time; cannot prove it covers the frozen continuation protocol")
        if proof_time < min_time:
            raise ContractError("Backup proof predates the frozen continuation target/protocol artifacts")
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "time": proof_time.isoformat() if proof_time else None,
        "raw_status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
        "packages_this_run": proof.get("packages_this_run"),
    }


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed_marker(completed)
        raise SystemExit("controlled-continuation diagnostic already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("Partial controlled-continuation output exists; inspect/recover first: " + ", ".join(rel(p) for p in leftovers[:20]))


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for path in (PROTOCOL_MD, PROTOCOL_JSON, TARGET_COMPLETED, TARGET_RAW, V1_RAW, V1_COMPLETED):
        if not path.exists():
            raise ContractError("Required input missing: %s" % rel(path))
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928":
        raise ContractError("Unexpected controlled-continuation protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("historical_validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("Frozen protocol access flags are invalid")
    if access.get("requires_verified_backup_before_rollout") is not True:
        raise ContractError("Frozen protocol must require verified backup before rollout")
    design = protocol.get("design") or {}
    if int(design.get("max_episodes", -1)) <= 0 or int(design.get("control_step_upper_bound", -1)) <= 0:
        raise ContractError("Frozen protocol budget missing")
    if int(design.get("new_training_episodes", -1)) != 0 or int(design.get("new_gradient_steps", -1)) != 0:
        raise ContractError("Frozen protocol unexpectedly allows training")
    target_done = verify_completed_marker(TARGET_COMPLETED)
    v1_done = verify_completed_marker(V1_COMPLETED, check_hashes=False)
    target_raw = read_json(TARGET_RAW)
    v1_raw = read_json(V1_RAW)
    for label, obj in (("target_completed", target_done), ("target_raw", target_raw), ("v1_completed", v1_done), ("v1_raw", v1_raw)):
        if obj.get("historical_validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise ContractError(label + " access flags are invalid")
    source = protocol.get("source_bank") or {}
    if source.get("v1_raw_sha256") and sha256(V1_RAW) != source["v1_raw_sha256"]:
        raise ContractError("V1 raw hash mismatch versus frozen continuation protocol")
    if source.get("v1_completed_sha256") and sha256(V1_COMPLETED) != source["v1_completed_sha256"]:
        raise ContractError("V1 completed hash mismatch versus frozen continuation protocol")
    # Verify that the frozen protocol target list is exactly the target-miner output.
    if canonical_sha(protocol.get("targets")) != canonical_sha(target_raw.get("selected_targets")):
        raise ContractError("Frozen protocol targets differ from target diagnostic output")
    return protocol, target_done, target_raw, v1_raw


def terminal_receipt_for_horizon(receipts: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    rec = receipts.get(str(h))
    if not rec:
        raise ContractError("Missing terminal receipt for H%d" % h)
    return rec


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def step_physical(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("performance")) + safe_float(row.get("constraint"))


def step_total(row: Mapping[str, Any]) -> float:
    return step_physical(row) + safe_float(row.get("compute"))


def tail_sum(trace: Sequence[Mapping[str, Any]], start: int, which: str) -> float:
    if start >= len(trace):
        return 0.0
    if which == "physical":
        return float(math.fsum(step_physical(r) for r in trace[start:]))
    if which == "total":
        return float(math.fsum(step_total(r) for r in trace[start:]))
    raise ValueError(which)


def state_tuple(row: Mapping[str, Any], key: str = "previous_state") -> Tuple[float, float, float]:
    state = row.get(key) or {}
    return (safe_float(state.get("x"), float("nan")), safe_float(state.get("y"), float("nan")), safe_float(state.get("theta"), float("nan")))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def clean_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(list(trace[:branch_step]))
    for row in cleaned:
        row.pop("timing", None)
        row.pop("decision_plus_terminal_switch_s", None)
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def summarize_episode(
    trace: List[Dict[str, Any]],
    reset: Dict[str, Any],
    construction_s: float,
    episode_wall_s: float,
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    terminal_switches: Sequence[Mapping[str, Any]],
    terminal_receipts: Mapping[str, Any],
    counts: Mapping[str, Any],
    logging: Any,
) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    branch_step = int(item["branch_step"])
    branch_h = int(item["branch_horizon"])
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"].get("selection_s", 0.0)) for r in trace]
    logging_times = [float(r["timing"].get("logging_s", 0.0)) for r in trace]
    decision_plus_switch = [float(r.get("decision_plus_terminal_switch_s", r["timing"]["decision_s"])) for r in trace]
    solver_times: List[float] = []
    horizon_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}
    for row in trace:
        hh = str(row["horizon"])
        horizon_counts[hh] = horizon_counts.get(hh, 0) + 1
        phase = str((row.get("decision") or {}).get("phase", "unknown"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    branch_reached = len(trace) > branch_step
    branch_state = state_tuple(trace[branch_step], "previous_state") if branch_reached else (float("nan"), float("nan"), float("nan"))
    out = dict(metric)
    out.update({
        "execution_index": int(item["execution_index"]),
        "case": int(item["case"]),
        "source_candidate_index": int(target.get("source_candidate_index", -1)),
        "stratum": target.get("stratum"),
        "branch_step": branch_step,
        "branch_horizon": branch_h,
        "prefix_horizon": PREFIX_H,
        "arm_id": "case%02d_branch%03d_H%02d" % (int(item["case"]), branch_step, branch_h),
        "family": "controlled_continuation_H15_prefix_then_fixed_branch",
        "episode_failure": not bool(metric.get("success")),
        "branch_reached": bool(branch_reached),
        "branch_previous_state": {"x": branch_state[0], "y": branch_state[1], "theta": branch_state[2]},
        "prefix_clean_sha256": canonical_sha(clean_prefix(trace, branch_step)),
        "continuation_physical_constraint_cost_from_branch": tail_sum(trace, branch_step, "physical"),
        "continuation_total_cost_from_branch": tail_sum(trace, branch_step, "total"),
        "prefix_physical_constraint_cost_before_branch": tail_sum(trace, 0, "physical") - tail_sum(trace, branch_step, "physical"),
        "prefix_total_cost_before_branch": tail_sum(trace, 0, "total") - tail_sum(trace, branch_step, "total"),
        "terminal_receipt_for_prefix_H15": terminal_receipt_for_horizon(terminal_receipts, PREFIX_H),
        "terminal_receipt_for_branch_H": terminal_receipt_for_horizon(terminal_receipts, branch_h),
        "terminal_switches": list(terminal_switches),
        "terminal_switch_timing_s": values_summary(float(s.get("switch_s", 0.0)) for s in terminal_switches),
        "construction_s": float(construction_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision_times),
        "decision_gross_timing_s": values_summary(decision_gross),
        "decision_plus_terminal_switch_timing_s": values_summary(decision_plus_switch),
        "controller_timing_s_logging_deducted": values_summary(controller_times),
        "selection_timing_s": values_summary(selection_times),
        "logging_timing_s": values_summary(logging_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "horizon_counts": horizon_counts,
        "phase_counts": phase_counts,
        "unique_horizons": sorted(int(x) for x in horizon_counts),
        "steps_metered": int(counts.get("step_calls", 0)),
        "resets_metered": int(counts.get("reset_calls", 0)),
        "logging_operations": int(getattr(logging, "operations", 0)),
        "logging_total_s": float(getattr(logging, "seconds", 0.0)),
    })
    return out


def run_episode(
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    case: Mapping[str, Any],
    terminals: Mapping[int, Tuple[Any, Any]],
    terminal_receipts: Mapping[str, Any],
) -> Dict[str, Any]:
    branch_step = int(item["branch_step"])
    branch_h = int(item["branch_horizon"])
    ep_dir = OUT_DIR / "episodes" / ("exec%03d_case%02d_b%03d_H%02d" % (int(item["execution_index"]), int(item["case"]), branch_step, branch_h))
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, ep_dir)
    env.set_value_function_weights_and_biases(*terminals[PREFIX_H])
    current_terminal_h = PREFIX_H
    construction_s = time.perf_counter() - construct_start
    controller = env.control_system.controller
    original = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    terminal_switches: List[Dict[str, Any]] = []
    with v1.LoggingTimer(ep_dir) as logging:
        recovery = v1.recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            if not (0.0 <= logged < gross):
                raise ContractError("Invalid logging timing bounds")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=int(item["case"]), step=-1, horizon=PREFIX_H)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_gross_s = time.perf_counter() - reset_start
        if obs is None or len(measured) != 1:
            raise ContractError("Unexpected reset/controller warmup state")
        reset_record = {"reset_gross_s": float(reset_gross_s)}
        reset_record.update(measured.pop())
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                selected_h = PREFIX_H if t < branch_step else branch_h
                switch_s = 0.0
                switched_terminal = False
                if selected_h != current_terminal_h:
                    st = time.perf_counter()
                    env.set_value_function_weights_and_biases(*terminals[selected_h])
                    switch_s = float(time.perf_counter() - st)
                    terminal_switches.append({"step": int(t), "from_horizon": int(current_terminal_h), "to_horizon": int(selected_h), "switch_s": switch_s})
                    current_terminal_h = int(selected_h)
                    switched_terminal = True
                recovery["step"] = int(t)
                recovery["horizon"] = int(selected_h)
                _, terminated, row = v1.observed_step(env, TASK, int(selected_h), dict(case), int(t))
                if len(measured) != 1:
                    raise ContractError("Expected exactly one measured controller call")
                timing = measured.pop()
                timing.update({
                    "selection_s": 0.0,
                    "decision_s": float(timing["controller_s"]),
                    "decision_gross_s": float(timing["controller_gross_s"]),
                    "terminal_switch_s_before_controller": switch_s,
                })
                phase = "prefix" if t < branch_step else "branch"
                row.update({
                    "decision": {
                        "kind": "controlled_continuation_H15_prefix_then_fixed_branch",
                        "phase": phase,
                        "prefix_horizon": PREFIX_H,
                        "branch_step": branch_step,
                        "branch_horizon": branch_h,
                        "selected_horizon": int(selected_h),
                        "terminal_horizon": int(current_terminal_h),
                        "terminal_switched_before_step": bool(switched_terminal),
                    },
                    "recovery": recovery["events"][-1],
                    "timing": timing,
                    "decision_plus_terminal_switch_s": float(timing["decision_s"] + switch_s),
                })
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                if terminated:
                    break
        if not trace or not trace[-1].get("termination"):
            raise ContractError("Episode did not terminate within max steps")
        if len(trace) <= branch_step:
            raise ContractError("Episode terminated before requested branch step case=%d branch_step=%d H=%d" % (int(item["case"]), branch_step, branch_h))
        v1.audit_trace(TASK, dict(case), trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(
            trace,
            reset_record,
            construction_s,
            time.perf_counter() - episode_start,
            item,
            target,
            terminal_switches,
            terminal_receipts,
            counts,
            logging,
        )
        summary.update({"path": rel(ep_dir), "solver_counts": recovery["counts"]})
        write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


def build_schedule(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target_index, target in enumerate(protocol.get("targets") or []):
        case_id = int(target["case"])
        for branch_step in target.get("branch_steps") or []:
            for branch_h in target.get("branch_horizons") or []:
                rows.append({
                    "target_index": int(target_index),
                    "case": case_id,
                    "branch_step": int(branch_step),
                    "branch_horizon": int(branch_h),
                })
    expected = int((protocol.get("design") or {}).get("max_episodes"))
    if len(rows) != expected:
        raise ContractError("Schedule length %d does not match frozen max_episodes %d" % (len(rows), expected))
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    for execution_index, base_index in enumerate(rng.permutation(len(rows)).tolist()):
        item = dict(rows[int(base_index)])
        item["schedule_base_index"] = int(base_index)
        item["execution_index"] = int(execution_index)
        schedule.append(item)
    return schedule


def no_regression(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if bool(reference.get("success")) and not bool(candidate.get("success")):
        return False
    if bool(candidate.get("constraint")) and not bool(reference.get("constraint")):
        return False
    if int(candidate.get("initial_failed_steps", 0)) > int(reference.get("initial_failed_steps", 0)):
        return False
    if int(candidate.get("solver_failure_steps", 0)) > int(reference.get("solver_failure_steps", 0)):
        return False
    return True


def analyze_results(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for e in episodes:
        groups.setdefault((int(e["case"]), int(e["branch_step"])), []).append(e)
    state_rows: List[Dict[str, Any]] = []
    confirmed_states = 0
    missing_reference: List[Dict[str, int]] = []
    for key in sorted(groups):
        case_id, branch_step = key
        rows = sorted(groups[key], key=lambda r: int(r["branch_horizon"]))
        ref_candidates = [r for r in rows if int(r["branch_horizon"]) == PREFIX_H]
        if not ref_candidates:
            missing_reference.append({"case": case_id, "branch_step": branch_step})
            continue
        ref = ref_candidates[0]
        ref_state = state_tuple(ref, "branch_previous_state")
        comparisons: List[Dict[str, Any]] = []
        best_phys = None
        best_total = None
        confirmed_here = False
        for row in rows:
            h = int(row["branch_horizon"])
            cand_state = state_tuple(row, "branch_previous_state")
            state_dist = state_distance(cand_state, ref_state)
            phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - row["continuation_physical_constraint_cost_from_branch"])
            total_gain = float(ref["continuation_total_cost_from_branch"] - row["continuation_total_cost_from_branch"])
            safe = no_regression(row, ref)
            material = bool(h != PREFIX_H and safe and (phys_gain >= IMPROVEMENT_ABS_THRESHOLD or total_gain >= IMPROVEMENT_ABS_THRESHOLD))
            if material:
                confirmed_here = True
            comp = {
                "horizon": h,
                "success": bool(row.get("success")),
                "constraint": bool(row.get("constraint")),
                "steps": int(row.get("steps", 0)),
                "continuation_physical": float(row["continuation_physical_constraint_cost_from_branch"]),
                "continuation_total": float(row["continuation_total_cost_from_branch"]),
                "gain_vs_H15_physical": phys_gain,
                "gain_vs_H15_total": total_gain,
                "decision_sum_s": float(row["decision_timing_s"]["sum"]),
                "decision_plus_terminal_switch_sum_s": float(row["decision_plus_terminal_switch_timing_s"]["sum"]),
                "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
                "state_distance_vs_H15_branch_state": state_dist,
                "prefix_clean_sha256_matches_H15": bool(row.get("prefix_clean_sha256") == ref.get("prefix_clean_sha256")),
                "no_success_constraint_solver_regression_vs_H15": safe,
                "material_improvement_vs_H15": material,
                "path": row.get("path"),
            }
            comparisons.append(comp)
            if best_phys is None or (comp["continuation_physical"], comp["decision_sum_s"], h) < (best_phys["continuation_physical"], best_phys["decision_sum_s"], best_phys["horizon"]):
                best_phys = comp
            if best_total is None or (comp["continuation_total"], comp["decision_sum_s"], h) < (best_total["continuation_total"], best_total["decision_sum_s"], best_total["horizon"]):
                best_total = comp
        if confirmed_here:
            confirmed_states += 1
        state_rows.append({
            "case": case_id,
            "branch_step": branch_step,
            "reference_H15": {
                "success": bool(ref.get("success")),
                "constraint": bool(ref.get("constraint")),
                "continuation_physical": float(ref["continuation_physical_constraint_cost_from_branch"]),
                "continuation_total": float(ref["continuation_total_cost_from_branch"]),
                "steps": int(ref.get("steps", 0)),
                "initial_failed_steps": int(ref.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(ref.get("solver_failure_steps", 0)),
            },
            "best_physical": best_phys,
            "best_total": best_total,
            "confirmed_material_noninitial_state": bool(confirmed_here),
            "comparisons": comparisons,
        })
    return {
        "state_count": len(state_rows),
        "missing_H15_reference": missing_reference,
        "confirmed_material_state_count": int(confirmed_states),
        "selector_smoke_gate_pass": bool(confirmed_states >= REQUIRED_CONFIRMED_STATES and not missing_reference),
        "acceptance_rule": "At least two noninitial branch states show a safe branch horizon improving continuation physical or total cost by >=3 absolute units versus H15 without solver/constraint regression.",
        "state_rows": state_rows,
    }


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = str(raw["created_utc"]).replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_CONTROLLED_CONTINUATION_DIAGNOSTIC_V0_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup controlled-continuation diagnostic before selector/value-refit or scenario-redesign work",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_MD), rel(PROTOCOL_JSON)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    lines = [
        "# Vehicle V1 controlled-continuation diagnostic v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development diagnostic only: fixed H15 common prefixes followed by target-specific fixed-H branches; no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / frozen max `{raw['budget_declared']['max_episodes']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / frozen upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Gate result for next selector/value-refit smoke",
        "",
        f"- Confirmed material noninitial branch states: `{analysis['confirmed_material_state_count']}` / required `{REQUIRED_CONFIRMED_STATES}`.",
        f"- selector_smoke_gate_pass: `{analysis['selector_smoke_gate_pass']}`.",
        f"- Rule: {analysis['acceptance_rule']}",
        "",
        "## Per-state continuation comparisons",
        "",
        "| case | branch step | H15 cont phys | H15 cont total | best phys H | phys gain vs H15 | best total H | total gain vs H15 | confirmed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["state_rows"]:
        bp = row.get("best_physical") or {}
        bt = row.get("best_total") or {}
        ref = row["reference_H15"]
        lines.append("| %d | %d | %.6g | %.6g | %s | %.6g | %s | %.6g | `%s` |" % (
            int(row["case"]),
            int(row["branch_step"]),
            float(ref["continuation_physical"]),
            float(ref["continuation_total"]),
            str(bp.get("horizon")),
            float(bp.get("gain_vs_H15_physical", 0.0)),
            str(bt.get("horizon")),
            float(bt.get("gain_vs_H15_total", 0.0)),
            str(bool(row["confirmed_material_noninitial_state"])),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if analysis["selector_smoke_gate_pass"]:
        lines.append("Identical-prefix continuation confirmed enough noninitial state-dependent horizon opportunity to justify freezing one small IMPROVED selector/value-refit smoke, with H15/H10/H30 fixed-H references retained.")
    else:
        lines.append("Identical-prefix continuation did not confirm enough exploitable state-dependent opportunity under the frozen rule; next work should diagnose weak canonical opportunity/scenario design or terminal/objective mismatch before broad retraining.")
    lines += ["", f"Backup request before further simulations: `{raw['backup_request']}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 controlled-continuation diagnostic v0\n\n"
        f"UTC: {raw['created_utc']}. Development-only identical-prefix continuation diagnostic completed: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, no validation64/test access. "
        f"Confirmed material states={analysis['confirmed_material_state_count']}; selector-smoke gate={analysis['selector_smoke_gate_pass']}. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(v1probe.__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(base.smoke_base.__file__).resolve(),
        Path(base.v1.__file__).resolve(),
        PROTOCOL_MD,
        PROTOCOL_JSON,
        TARGET_COMPLETED,
        TARGET_RAW,
        V1_RAW,
        V1_COMPLETED,
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after the continuation target diagnostic/protocol")
    ap.add_argument("--i-accept-controlled-continuation-development-rollout", action="store_true", help="explicit acknowledgement: development diagnostic only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_controlled_continuation_development_rollout:
        raise ContractError("Explicit --i-accept-controlled-continuation-development-rollout is required")
    assert_fresh_output_dir()
    protocol, target_done, target_raw, v1_raw = verify_inputs()
    min_backup_time = parse_time(target_raw.get("created_utc")) or parse_time(protocol.get("created_utc"))
    backup = verify_backup_proof(args.backup_proof, min_backup_time)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_v1_controlled_continuation_diagnostic_v0",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    preflight = v1probe.base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(preflight.get("diagnosis", "runtime preflight failed") + " " + str(preflight.get("exception", "")))
    v1.latency_verify()
    v1_completed = verify_completed_marker(V1_COMPLETED, check_hashes=False)
    bank_path = ROOT / str(v1_raw["bank"]["path"])
    bank_done = bank_path.parent / "completed.json"
    verify_completed_marker(bank_done)
    bank = read_json(bank_path)
    selected_cases = bank["selected_cases"]
    if len(selected_cases) < max(int(t["case"]) for t in protocol["targets"]) + 1:
        raise ContractError("Frozen target references unavailable V1 selected case")
    terminals, terminal_receipts = v1probe.load_terminal_grid(v1_raw["protocol_full"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    schedule = build_schedule(protocol)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "targets": protocol["targets"]})
    target_by_index = {i: t for i, t in enumerate(protocol["targets"])}
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        target = target_by_index[int(item["target_index"])]
        case = selected_cases[int(item["case"])]
        summary = run_episode(item, target, case, terminals, terminal_receipts)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "case", "branch_step", "branch_horizon", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    budget = protocol["design"]
    if len(episodes) != int(budget["max_episodes"]) or control_steps > int(budget["control_step_upper_bound"]):
        raise ContractError("Controlled-continuation budget violation")
    analysis = analyze_results(episodes)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_v1_controlled_continuation_diagnostic_v0_identical_H15_prefix_branch_fixed_H",
        "classification": "development_IMPROVED_diagnostic_not_model_selection_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"path": rel(PROTOCOL_MD), "sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "protocol_full": protocol,
        "target_diagnostic": {"completed": rel(TARGET_COMPLETED), "completed_sha256": sha256(TARGET_COMPLETED), "raw": rel(TARGET_RAW), "raw_sha256": sha256(TARGET_RAW), "headline": target_done.get("headline")},
        "v1_source": {"raw": rel(V1_RAW), "raw_sha256": sha256(V1_RAW), "completed": rel(V1_COMPLETED), "completed_sha256": sha256(V1_COMPLETED), "completed_headline": v1_completed.get("headline")},
        "bank": {"path": rel(bank_path), "sha256": sha256(bank_path)},
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"max_episodes": int(budget["max_episodes"]), "control_step_upper_bound": int(budget["control_step_upper_bound"]), "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": [
            "development diagnostic only",
            "common H15 prefix replayed independently for each branch arm",
            "not adaptive model selection",
            "not new training/refit",
            "not ORIGINAL SAC",
            "not final test",
            "actual timing from AWS run but terminal-switch overhead is reported separately from solver time",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 controlled-continuation diagnostic v0 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} episodes / {control_steps} control steps. No validation64/test access. "
        f"Confirmed material states={analysis['confirmed_material_state_count']}; selector-smoke gate={analysis['selector_smoke_gate_pass']}. Backup required before next simulation.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON, TARGET_COMPLETED, TARGET_RAW]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": raw["backup_request"],
        "headline": {
            "confirmed_material_state_count": analysis["confirmed_material_state_count"],
            "selector_smoke_gate_pass": analysis["selector_smoke_gate_pass"],
            "state_count": analysis["state_count"],
            "next_action": "freeze selector/value-refit smoke if gate passed; otherwise diagnose scenario/terminal/objective before broad retraining",
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "confirmed_material_state_count": analysis["confirmed_material_state_count"],
        "selector_smoke_gate_pass": analysis["selector_smoke_gate_pass"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": raw["backup_request"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve partial directory, audit failure, and create a versioned one-variable repair. Use legacy interpreter and a verified backup proof after the continuation target diagnostic/protocol.",
        })
        raise
