#!/usr/bin/env python3
"""Vehicle gated-horizon risk-reselection v1 engineering smoke.

Development-only smoke for the finite risk-first re-selection result.  It runs
risk-reselected policies, current stored gated policies, and same-seed fixed H25
on a tiny fresh engineering bank.  It does not train, does not mutate historical
controllers, does not open validation64, and does not access sealed tests.
"""

from __future__ import annotations

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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import vehicle_safe_shortening_v1_smoke as v1  # noqa:E402
from run import snapshot  # noqa:E402

ROOT = v1.ROOT
TASK = "vehicle"
SEEDS = (0, 1, 2)
BASE_H = 25
CASES_EXPECTED = 2
REPEATS = 2
MAX_STEPS = 150
BANK_RNG = 2609284100
ORDER_SEED = 2609284200
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_gated_horizon_risk_reselection_v1_smoke_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_risk_reselection_v1_smoke_protocol_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_risk_reselection_v1_smoke_protocol_20260928.json"
RISK_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000/completed.json"
NOMINATED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000/nominated_policies.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-gated-horizon-risk-reselection-v1-smoke-20260928"


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
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


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


def verify_completed_marker(path: Path) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise RuntimeError("Completed marker did not pass: %s" % rel(path))
    for name, expected in done.get("hashes", {}).items():
        actual = sha256(ROOT / name)
        if actual != expected:
            raise RuntimeError("Hash mismatch for %s" % name)
    return done


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    if (OUT_DIR / "completed.json").exists():
        verify_completed_marker(OUT_DIR / "completed.json")
        raise SystemExit("risk-reselection smoke already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise RuntimeError("Partial risk-reselection smoke output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10]))


def clean_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(trace)
    for row in cleaned:
        row.pop("timing", None)
        for attempt in row.get("recovery", {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def runtime_preflight() -> Dict[str, Any]:
    try:
        import tensorflow as tf  # noqa:F401
    except BaseException as exc:
        return {
            "passed": False,
            "exception": repr(exc),
            "python": sys.version,
            "executable": sys.executable,
            "diagnosis": "legacy Python/TF1 interpreter is required",
        }
    return {
        "passed": True,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "python": sys.version,
        "executable": sys.executable,
    }


def generate_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        verify_completed_marker(BANK_COMPLETED)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "risk_reselection_v1_smoke" or len(bank.get("cases", [])) != CASES_EXPECTED:
            raise RuntimeError("Existing risk-reselection smoke bank metadata mismatch")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": CASES_EXPECTED}
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, gen_dir)
    recovery = v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(BANK_RNG)
    np.random.seed(BANK_RNG)
    cases: List[Dict[str, Any]] = []
    for cid in range(CASES_EXPECTED):
        recovery.update(enabled=False, events=[], case=cid, step=-1)
        env.reset()
        cases.append(snapshot(env))
    if counts["step_calls"] != 0 or counts["reset_calls"] != CASES_EXPECTED:
        raise RuntimeError("Unexpected bank generation counts: %r" % counts)
    bank = {
        "task": TASK,
        "split": "risk_reselection_v1_smoke",
        "rng": BANK_RNG,
        "cases": cases,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "fresh engineering smoke only for IMPROVED vehicle gated-horizon risk re-selection v1",
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "task": TASK,
        "split": "risk_reselection_v1_smoke",
        "rng": BANK_RNG,
        "cases": CASES_EXPECTED,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "new_scored_control_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": CASES_EXPECTED, "reset_calls": counts["reset_calls"]}


def load_risk_nominations() -> Dict[int, Dict[str, Any]]:
    completed = read_json(RISK_COMPLETED)
    expected = completed.get("nominated_policies_sha256")
    if expected and sha256(NOMINATED) != expected:
        raise RuntimeError("Nominated-policy hash mismatch")
    data = read_json(NOMINATED)
    out: Dict[int, Dict[str, Any]] = {}
    expected_ids = {0: "h20_p1_g15", 1: "h20_p2_g15", 2: "h15_p1_g15"}
    for seed in SEEDS:
        row = data[str(seed)]["nominated_policy"]
        if not bool(row.get("primary_adaptive_eligible")):
            raise RuntimeError("Seed %d nomination is not primary eligible" % seed)
        cid = str(row["candidate_id"])
        if cid != expected_ids[seed]:
            raise RuntimeError("Unexpected nominated id for seed %d: %s" % (seed, cid))
        policy = {
            "id": cid,
            "task": TASK,
            "short_h": int(row["short_h"]),
            "profile": int(row["profile"]),
            "guard": int(row["guard"]),
            "training_seed": int(seed),
        }
        out[seed] = policy
    return out


def build_arms() -> List[Dict[str, Any]]:
    nominations = load_risk_nominations()
    arms: List[Dict[str, Any]] = []
    for seed in SEEDS:
        risk_policy = nominations[seed]
        current_policy = v1.load_old_policy(seed)
        arms.append({
            "arm_id": "risk_reselected_v1_vehicle_s%d" % seed,
            "role": "risk_reselected_adaptive_candidate",
            "family": "IMPROVED_risk_reselected_gated_policy_common_safe_wrapper",
            "seed": int(seed),
            "policy": risk_policy,
            "policy_path": rel(NOMINATED),
            "policy_sha256": canonical_sha(risk_policy),
        })
        arms.append({
            "arm_id": "current_gated_vehicle_s%d" % seed,
            "role": "current_stored_gated_policy_comparator",
            "family": "IMPROVED_current_historical_gated_policy_common_safe_wrapper",
            "seed": int(seed),
            "policy": current_policy,
            "policy_path": rel(v1.old_policy_path(seed)),
            "policy_sha256": sha256(v1.old_policy_path(seed)),
        })
        arms.append({
            "arm_id": "fixed_H25_vehicle_s%d" % seed,
            "role": "same_seed_fixed_H25_comparator",
            "family": "fixed_H25_primary_same_seed",
            "seed": int(seed),
            "policy": {"id": "fixed", "task": TASK, "h": BASE_H},
            "policy_path": None,
            "policy_sha256": canonical_sha({"id": "fixed", "task": TASK, "h": BASE_H}),
        })
    if len(arms) != 9:
        raise RuntimeError("Unexpected arm count")
    return arms


def randomized_schedule(arms: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    index = 0
    for repeat in range(REPEATS):
        for arm_index in [int(x) for x in rng.permutation(len(arms)).tolist()]:
            for case_id in [int(x) for x in rng.permutation(CASES_EXPECTED).tolist()]:
                schedule.append({
                    "episode_index": int(index),
                    "repeat": int(repeat),
                    "arm_index": int(arm_index),
                    "arm_id": str(arms[arm_index]["arm_id"]),
                    "seed": int(arms[arm_index]["seed"]),
                    "case": int(case_id),
                })
                index += 1
    expected = len(arms) * CASES_EXPECTED * REPEATS
    if len(schedule) != expected:
        raise RuntimeError("Unexpected schedule length")
    return schedule


def common_safe_decide(arm: Mapping[str, Any], ctx: Dict[str, Any], previous_initial_failure: bool,
                       previous_final_failure: bool) -> Tuple[int, Dict[str, Any]]:
    if arm["role"] == "same_seed_fixed_H25_comparator":
        return BASE_H, {"kind": "fixed", "raw_horizon": BASE_H, "selected_horizon": BASE_H, "fallback": False, "reason": "same_seed_H25"}
    if previous_initial_failure or previous_final_failure:
        return BASE_H, {
            "kind": "common_safe_gated_wrapper",
            "fallback": True,
            "reason": "previous_solver_failure",
            "raw_horizon": BASE_H,
            "selected_horizon": BASE_H,
            "policy_role": arm["role"],
        }
    raw_h, gate = v1.gated_decide(arm["policy"], ctx)
    raw_h = int(raw_h)
    if raw_h not in range(5, 51, 5):
        raise RuntimeError("unsafe non-grid horizon requested: %r" % raw_h)
    selected = min(raw_h, BASE_H)
    return int(selected), {
        "kind": "common_safe_gated_wrapper",
        "fallback": False,
        "raw_horizon": raw_h,
        "selected_horizon": int(selected),
        "clamped_to_base": raw_h > BASE_H,
        "policy_role": arm["role"],
        "gate": gate,
    }


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                      terminal_load_s: float, episode_wall_s: float, item: Mapping[str, Any],
                      arm: Mapping[str, Any]) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"]["selection_s"]) for r in trace]
    logging_times = [float(r["timing"].get("logging_s", 0.0)) for r in trace]
    solver_times: List[float] = []
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    clamps = 0
    fallbacks = 0
    for row in trace:
        h = str(row["horizon"])
        horizons[h] = horizons.get(h, 0) + 1
        decision_meta = row.get("decision", {})
        raw = decision_meta.get("raw_horizon", row["horizon"])
        raw_horizons[str(raw)] = raw_horizons.get(str(raw), 0) + 1
        clamps += int(bool(decision_meta.get("clamped_to_base", False)))
        fallbacks += int(bool(decision_meta.get("fallback", False)))
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    out = dict(metric)
    out.update({
        "episode_index": int(item["episode_index"]),
        "arm_id": arm["arm_id"],
        "family": arm["family"],
        "role": arm["role"],
        "seed": int(arm["seed"]),
        "case": int(item["case"]),
        "repeat": int(item["repeat"]),
        "episode_failure": not bool(metric.get("success")),
        "construction_s": float(construction_s),
        "terminal_load_s_reference": float(terminal_load_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision_times),
        "decision_gross_timing_s": values_summary(decision_gross),
        "controller_timing_s_logging_deducted": values_summary(controller_times),
        "selection_timing_s": values_summary(selection_times),
        "logging_timing_s": values_summary(logging_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(x) for x in horizons),
        "adapted_below_H25": any(int(x) < BASE_H for x in horizons),
        "unsafe_above_H25_dispatch": any(int(x) > BASE_H for x in horizons),
        "clamped_steps": int(clamps),
        "solver_failure_fallback_steps": int(fallbacks),
    })
    return out


def run_episode(item: Mapping[str, Any], arm: Mapping[str, Any], case: Mapping[str, Any],
                terminal: Tuple[Any, Any], terminal_load_s: float) -> Dict[str, Any]:
    ep_dir = OUT_DIR / "episodes" / (
        "ep%03d_r%d_%s_case%02d" % (int(item["episode_index"]), int(item["repeat"]), str(arm["arm_id"]), int(item["case"]))
    )
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = v1.make_env(TASK, int(arm["seed"]), aligned=True, scaled_obs=True)
    counts = v1.meter(env, ep_dir)
    env.set_value_function_weights_and_biases(*terminal)
    construction_s = time.perf_counter() - construct_start
    controller = env.control_system.controller
    original = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    with v1.LoggingTimer(ep_dir) as logging:
        recovery = v1.recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            if not (0.0 <= logged < gross):
                raise RuntimeError("Invalid logging timing bounds")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=int(item["case"]), step=-1)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_gross_s = time.perf_counter() - reset_start
        if obs is None or len(measured) != 1:
            raise RuntimeError("Unexpected reset/controller warmup state")
        reset_record = {"reset_gross_s": float(reset_gross_s)}
        reset_record.update(measured.pop())
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        previous_initial = False
        previous_final = False
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                recovery["step"] = int(t)
                ctx = v1.context(env, TASK)
                ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                select_start = time.perf_counter()
                selected_h, decision = common_safe_decide(arm, ctx, previous_initial, previous_final)
                selection_s = time.perf_counter() - select_start
                _, terminated, row = v1.observed_step(env, TASK, selected_h, dict(case), t)
                if len(measured) != 1:
                    raise RuntimeError("Expected exactly one measured controller call")
                timing = measured.pop()
                timing.update({
                    "selection_s": float(selection_s),
                    "decision_s": float(selection_s + timing["controller_s"]),
                    "decision_gross_s": float(selection_s + timing["controller_gross_s"]),
                })
                row.update({"policy_context": ctx, "decision": decision, "recovery": recovery["events"][-1], "timing": timing})
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                previous_initial = not bool(row["recovery"]["attempts"][0]["success"])
                previous_final = not bool(row["solver_success"])
                if terminated:
                    break
        if not trace or not trace[-1].get("termination"):
            raise RuntimeError("Episode did not terminate within max steps")
        v1.audit_trace(TASK, dict(case), trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(trace, reset_record, construction_s, terminal_load_s, time.perf_counter() - episode_start, item, arm)
        summary.update({
            "path": rel(ep_dir),
            "steps_metered": counts["step_calls"],
            "resets_metered": counts["reset_calls"],
            "solver_counts": recovery["counts"],
            "logging_operations": logging.operations,
            "logging_total_s": float(logging.seconds),
        })
        write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


def aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "episodes": len(episodes),
        "steps": int(sum(e.get("steps", 0) for e in episodes)),
        "success_count": int(sum(1 for e in episodes if e.get("success"))),
        "episode_failure_count": int(sum(1 for e in episodes if e.get("episode_failure"))),
        "constraint_count": int(sum(1 for e in episodes if e.get("constraint"))),
        "initial_failed_steps": int(sum(e.get("initial_failed_steps", 0) for e in episodes)),
        "solver_failure_steps": int(sum(e.get("solver_failure_steps", 0) for e in episodes)),
        "retries": int(sum(e.get("retries", 0) for e in episodes)),
        "recovered_steps": int(sum(e.get("recovered_steps", 0) for e in episodes)),
        "deadline_exceed_steps": int(sum(e.get("deadline_exceed_steps", 0) for e in episodes)),
        "switches": int(sum(e.get("switches", 0) for e in episodes)),
        "clamped_steps": int(sum(e.get("clamped_steps", 0) for e in episodes)),
        "solver_failure_fallback_steps": int(sum(e.get("solver_failure_fallback_steps", 0) for e in episodes)),
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "decision_total_s": float(math.fsum(float(e["decision_timing_s"]["sum"]) for e in episodes)),
        "decision_gross_total_s": float(math.fsum(float(e["decision_gross_timing_s"]["sum"]) for e in episodes)),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
    }
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[h] = horizons.get(h, 0) + int(n)
        for h, n in (e.get("raw_horizon_counts_before_clamp") or {}).items():
            raw_horizons[h] = raw_horizons.get(h, 0) + int(n)
    out.update({
        "total_cost_mean_episode": out["total_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "physical_constraint_cost_mean_episode": out["physical_constraint_cost_sum"] / out["episodes"] if out["episodes"] else None,
        "decision_mean_s_per_step": out["decision_total_s"] / out["steps"] if out["steps"] else None,
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(x) for x in horizons),
        "adapted_below_H25": any(int(x) < BASE_H for x in horizons),
        "unsafe_above_H25_dispatch": any(int(x) > BASE_H for x in horizons),
    })
    return out


def arm_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "physical_constraint_delta": float(a["physical_constraint_cost_sum"] - b["physical_constraint_cost_sum"]),
        "total_cost_delta": float(a["total_cost_sum"] - b["total_cost_sum"]),
        "decision_time_ratio": float(a["decision_total_s"] / b["decision_total_s"]) if b.get("decision_total_s") else None,
        "success_counts": {"a": a["success_count"], "b": b["success_count"]},
        "episode_failure_counts": {"a": a["episode_failure_count"], "b": b["episode_failure_count"]},
        "constraint_counts": {"a": a["constraint_count"], "b": b["constraint_count"]},
        "initial_failed_steps": {"a": a["initial_failed_steps"], "b": b["initial_failed_steps"]},
        "solver_failure_steps": {"a": a["solver_failure_steps"], "b": b["solver_failure_steps"]},
    }


def compare(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        risk = aggregates["risk_reselected_v1_vehicle_s%d" % seed]
        current = aggregates["current_gated_vehicle_s%d" % seed]
        fixed = aggregates["fixed_H25_vehicle_s%d" % seed]
        out[str(seed)] = {
            "risk_unique_horizons": risk["unique_horizons"],
            "current_unique_horizons": current["unique_horizons"],
            "fixed_unique_horizons": fixed["unique_horizons"],
            "risk_adapted_below_H25": risk["adapted_below_H25"],
            "current_adapted_below_H25": current["adapted_below_H25"],
            "risk_unsafe_above_H25_dispatch": risk["unsafe_above_H25_dispatch"],
            "current_unsafe_above_H25_dispatch": current["unsafe_above_H25_dispatch"],
            "risk_minus_fixed": arm_delta(risk, fixed),
            "risk_minus_current": arm_delta(risk, current),
            "current_minus_fixed": arm_delta(current, fixed),
        }
    return out


def smoke_pass_checks(raw: Mapping[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["runtime_preflight_passed"] = bool(raw["runtime_preflight"].get("passed"))
    checks["episode_count_ok"] = raw["budget_actual"]["episodes"] == raw["budget_declared"]["episodes_exact"]
    checks["control_step_bound_ok"] = raw["budget_actual"]["control_steps"] <= raw["budget_declared"]["control_step_upper_bound"]
    checks["replay_passed"] = bool(raw["replay"].get("passed"))
    aggregates = raw["aggregates"]
    checks["no_adaptive_above_H25_dispatch"] = all(
        not a.get("unsafe_above_H25_dispatch") for a in aggregates.values() if a.get("role") != "same_seed_fixed_H25_comparator"
    )
    safety_ok = True
    safety_failures: List[Dict[str, Any]] = []
    for seed in SEEDS:
        risk = aggregates["risk_reselected_v1_vehicle_s%d" % seed]
        fixed = aggregates["fixed_H25_vehicle_s%d" % seed]
        comparisons = [
            ("success_count", risk["success_count"] < fixed["success_count"]),
            ("constraint_count", risk["constraint_count"] > fixed["constraint_count"]),
            ("initial_failed_steps", risk["initial_failed_steps"] > fixed["initial_failed_steps"]),
            ("solver_failure_steps", risk["solver_failure_steps"] > fixed["solver_failure_steps"]),
        ]
        bad = [name for name, failed in comparisons if failed]
        if bad:
            safety_ok = False
            safety_failures.append({"seed": seed, "failed_checks": bad})
    checks["risk_not_worse_than_fixed_smoke_safety_counts"] = safety_ok
    checks["risk_safety_failures"] = safety_failures
    checks["risk_reselected_below_H25_seed_count"] = sum(
        1 for seed in SEEDS if aggregates["risk_reselected_v1_vehicle_s%d" % seed].get("adapted_below_H25")
    )
    checks["diagnostic_expectation_two_or_more_risk_seeds_adapted"] = checks["risk_reselected_below_H25_seed_count"] >= 2
    hard_keys = [
        "runtime_preflight_passed", "episode_count_ok", "control_step_bound_ok", "replay_passed",
        "no_adaptive_above_H25_dispatch", "risk_not_worse_than_fixed_smoke_safety_counts",
    ]
    checks["hard_pass"] = all(bool(checks[k]) for k in hard_keys)
    return checks


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle gated-horizon risk-reselection v1 smoke",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Engineering smoke only on a fresh risk-reselection smoke bank. No validation64 access and no sealed-test access.",
        "",
        f"Protocol: `{raw['protocol']['path']}` sha256 `{raw['protocol']['sha256']}`.",
        f"Risk re-selection input: `{raw['risk_reselection']['completed_path']}` sha256 `{raw['risk_reselection']['completed_sha256']}`.",
        "",
        "## Budget/access",
        "",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- New training episodes / gradient steps: `{raw['budget_actual']['new_training_episodes']}` / `{raw['budget_actual']['new_gradient_steps']}`.",
        f"- Bank generation resets: `{raw['budget_actual']['bank_generation_resets']}`.",
        f"- Replay passed: `{raw['replay']['passed']}` over `{raw['replay']['pairs_checked']}` repeat pairs.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Hard smoke checks",
        "",
    ]
    for key, value in raw["smoke_pass_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Arm aggregates",
        "",
        "| arm | seed | role | episodes | steps | success | failures | constraints | init-fail | final-fail | physical+constraint | total | decision mean s/step | horizons | raw horizons | fallbacks |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for arm_id in sorted(raw["aggregates"]):
        a = raw["aggregates"][arm_id]
        lines.append(
            f"| `{arm_id}` | {a['seed']} | {a['role']} | {a['episodes']} | {a['steps']} | {a['success_count']} | "
            f"{a['episode_failure_count']} | {a['constraint_count']} | {a['initial_failed_steps']} | {a['solver_failure_steps']} | "
            f"{a['physical_constraint_cost_sum']:.6g} | {a['total_cost_sum']:.6g} | {a['decision_mean_s_per_step']:.6g} | "
            f"{a['horizon_counts']} | {a['raw_horizon_counts_before_clamp']} | {a['solver_failure_fallback_steps']} |"
        )
    lines += ["", "## Same-seed smoke deltas", ""]
    for seed, cmp in raw["paired_comparisons"].items():
        rf = cmp["risk_minus_fixed"]
        rc = cmp["risk_minus_current"]
        cf = cmp["current_minus_fixed"]
        lines.append(
            f"- seed {seed}: risk horizons={cmp['risk_unique_horizons']}, current horizons={cmp['current_unique_horizons']}; "
            f"risk-fixed physical Δ={rf['physical_constraint_delta']:.6g}, total Δ={rf['total_cost_delta']:.6g}, decision ratio={rf['decision_time_ratio']:.6g}; "
            f"risk-current physical Δ={rc['physical_constraint_delta']:.6g}, total Δ={rc['total_cost_delta']:.6g}, decision ratio={rc['decision_time_ratio']:.6g}; "
            f"current-fixed physical Δ={cf['physical_constraint_delta']:.6g}, total Δ={cf['total_cost_delta']:.6g}, decision ratio={cf['decision_time_ratio']:.6g}."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is an implementation smoke only. It may justify scheduling a fresh development-validation comparison after backup, but it is not independent validation/model-selection evidence and cannot support a reproduction or improved-method success claim. H distribution is reported as behavior, not as timing proof; measured decision/solver timings are retained in raw traces.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(raw: Dict[str, Any]) -> None:
    stamp = raw["created_utc"].replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_SMOKE_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup risk-reselection smoke protocol/source/artifacts before any larger fresh development validation",
        "artifacts": [rel(OUT_DIR), rel(PROTOCOL), rel(PROTOCOL_JSON), rel(Path(__file__).resolve()), rel(RISK_COMPLETED), rel(NOMINATED)],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    raw["backup_request"] = rel(path)


def append_docs(raw: Mapping[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle gated-horizon risk-reselection v1 smoke\n\n"
        f"UTC: {raw['created_utc']}. IMPROVED risk-first gated-horizon re-selection smoke completed on a fresh engineering bank: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, hard_pass={raw['smoke_pass_checks']['hard_pass']}, "
        f"risk_reselected_below_H25_seed_count={raw['smoke_pass_checks']['risk_reselected_below_H25_seed_count']}. "
        "No validation64 bank or sealed test was opened. This is not model-selection/final evidence and does not modify the frozen v2 shard campaign. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(), Path(v1.__file__).resolve(), PROTOCOL, PROTOCOL_JSON, RISK_COMPLETED, NOMINATED,
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/relative_policy_features.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def main() -> int:
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED_vehicle_gated_horizon_risk_reselection_v1_smoke",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    preflight = runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise RuntimeError(preflight["diagnosis"] + " " + preflight.get("exception", ""))
    v1.latency_verify()
    for p in (PROTOCOL, PROTOCOL_JSON, RISK_COMPLETED, NOMINATED):
        if not p.exists():
            raise RuntimeError("Required frozen input missing: %s" % rel(p))
    bank_info = generate_bank_if_needed()
    bank_data = read_json(BANK_PATH)
    arms = build_arms()
    schedule = randomized_schedule(arms)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "arms": arms})
    terminal_hashes: Dict[str, Any] = {}
    terminals: Dict[int, Tuple[Any, Any]] = {}
    terminal_load_s: Dict[int, float] = {}
    for seed in SEEDS:
        start = time.perf_counter()
        terminals[seed] = v1.load_terminal(seed, terminal_hashes)
        terminal_load_s[seed] = time.perf_counter() - start
    write_json(OUT_DIR / "terminal_sources.json", terminal_hashes)
    arm_by_id = {a["arm_id"]: a for a in arms}
    episodes: List[Dict[str, Any]] = []
    trace_index: Dict[Tuple[str, int, int], Path] = {}
    replay = {"passed": True, "pairs_checked": 0, "mismatches": []}
    for item in schedule:
        arm = arm_by_id[item["arm_id"]]
        summary = run_episode(item, arm, bank_data["cases"][item["case"]], terminals[int(arm["seed"])], terminal_load_s[int(arm["seed"])])
        episodes.append(summary)
        trace_path = ROOT / summary["path"] / "trace.json"
        trace_index[(item["arm_id"], item["case"], item["repeat"])] = trace_path
        if item["repeat"] == 1:
            base = trace_index[(item["arm_id"], item["case"], 0)]
            replay["pairs_checked"] += 1
            if clean_trace(read_json(base)) != clean_trace(read_json(trace_path)):
                replay["passed"] = False
                replay["mismatches"].append({"arm_id": item["arm_id"], "case": item["case"], "repeat0": rel(base), "repeat1": rel(trace_path)})
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(e["steps"] for e in episodes)),
            "last_episode": {k: summary[k] for k in ("episode_index", "arm_id", "case", "repeat", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    if not replay["passed"]:
        raise RuntimeError("Repeat replay mismatch: %r" % replay["mismatches"][:3])
    control_steps = int(sum(e["steps"] for e in episodes))
    if len(episodes) != len(arms) * CASES_EXPECTED * REPEATS or control_steps > len(arms) * CASES_EXPECTED * REPEATS * MAX_STEPS:
        raise RuntimeError("Smoke budget exceeded or incomplete")
    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({
            "seed": int(arm["seed"]), "family": arm["family"], "role": arm["role"],
            "policy": arm["policy"], "policy_path": arm["policy_path"], "policy_sha256": arm["policy_sha256"],
        })
        aggregates[arm["arm_id"]] = agg
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_vehicle_gated_horizon_risk_reselection_v1_smoke_not_original_SAC",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": "fresh_engineering_smoke_bank_only_risk_reselection_v1",
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "risk_reselection": {"completed_path": rel(RISK_COMPLETED), "completed_sha256": sha256(RISK_COMPLETED), "nominated_path": rel(NOMINATED), "nominated_sha256": sha256(NOMINATED)},
        "bank": bank_info,
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "method_change": {
            "from": "current stored gated policies selected by mean raw-cost finite search",
            "to": "risk-first metadata re-selected gated policies executed through the same common safe wrapper",
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "v2_h10_transition_hold_used": False,
        },
        "budget_declared": {"episodes_exact": len(arms) * CASES_EXPECTED * REPEATS, "control_step_upper_bound": len(arms) * CASES_EXPECTED * REPEATS * MAX_STEPS, "bank_generation_resets": CASES_EXPECTED, "new_training_episodes": 0, "new_gradient_steps": 0, "development_validation_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "resets": int(sum(e["resets_metered"] for e in episodes)), "environment_constructions": len(episodes), "bank_generation_resets": CASES_EXPECTED if bank_info.get("created_now") else 0, "new_training_episodes": 0, "new_gradient_steps": 0, "development_validation_episodes": 0, "sealed_test_episodes": 0},
        "arms": arms,
        "randomized_schedule": schedule,
        "terminal_sources": terminal_hashes,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired_comparisons": compare(aggregates),
        "replay": replay,
        "interpretation_limits": ["engineering smoke only", "not validation/model selection", "not ORIGINAL SAC", "finite candidate re-selection not neural training", "H distribution is not timing evidence", "sealed final test remains closed"],
    }
    raw["smoke_pass_checks"] = smoke_pass_checks(raw)
    write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], PROTOCOL, PROTOCOL_JSON, Path(__file__).resolve(), Path(v1.__file__).resolve(), RISK_COMPLETED, NOMINATED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": bool(raw["smoke_pass_checks"]["hard_pass"]),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
        "backup_request": raw["backup_request"],
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "headline": {"control_steps": control_steps, "smoke_pass_checks": raw["smoke_pass_checks"], "paired_comparisons": raw["paired_comparisons"]},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "hard_pass": raw["smoke_pass_checks"]["hard_pass"],
        "smoke_pass_checks": raw["smoke_pass_checks"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "paired_comparisons": raw["paired_comparisons"],
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
            "next_recovery_hint": "Inspect failure, preserve this directory, and create a versioned one-variable recovery if necessary. Use legacy interpreter for TF1 runtime.",
        })
        raise
