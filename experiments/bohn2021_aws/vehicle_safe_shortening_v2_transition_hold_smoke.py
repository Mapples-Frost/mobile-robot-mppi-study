#!/usr/bin/env python3
"""Vehicle safe-shortening v2 transition-hold engineering smoke.

IMPROVED development-only amendment after v1 devval rejection and case9
transition diagnostics.  This script does not train, does not mutate historical
checkpoints, does not open validation64 or sealed final-test content, and uses a
fresh engineering smoke bank generated only for v2 implementation readiness.
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
CASES_EXPECTED = 2
REPEATS = 2
MAX_STEPS = 150
BANK_RNG = 2609281100
ORDER_SEED = 2609281200
H10_DWELL_STEPS = 3
BASE_H = 25
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_safe_shortening_v2_transition_hold_smoke_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_safe_shortening_v2_transition_hold_protocol_20260928.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_TRANSITION_HOLD_SMOKE_20260928T000000Z.json"
MARKER = "vehicle-safe-shortening-v2-transition-hold-smoke-20260928"


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
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
        raise SystemExit("v2 transition-hold smoke already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise RuntimeError("Partial v2 smoke output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10]))


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(data.size),
        "sum": float(data.sum()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
    }


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
        if bank.get("task") != TASK or bank.get("split") != "v2_transition_hold_smoke" or len(bank.get("cases", [])) != CASES_EXPECTED:
            raise RuntimeError("Existing v2 smoke bank metadata mismatch")
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
    if counts["step_calls"] != 0:
        raise RuntimeError("Bank generation unexpectedly stepped environment: %r" % counts)
    bank = {
        "task": TASK,
        "split": "v2_transition_hold_smoke",
        "rng": BANK_RNG,
        "cases": cases,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "fresh engineering smoke only for IMPROVED vehicle safe-shortening v2 transition-hold wrapper",
        "validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "task": TASK,
        "split": "v2_transition_hold_smoke",
        "rng": BANK_RNG,
        "cases": CASES_EXPECTED,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "new_scored_control_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": CASES_EXPECTED, "reset_calls": counts["reset_calls"]}


def load_terminal(seed: int, terminal_hashes: Dict[str, Any]) -> Tuple[Any, Any]:
    terminal = v1.load_terminal(seed, terminal_hashes)
    return terminal


def build_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = []
    for seed in SEEDS:
        p = v1.old_policy_path(seed)
        policy = v1.load_old_policy(seed)
        arms.append({
            "arm_id": "safe_shortening_v2_hold3_vehicle_s%d" % seed,
            "role": "adaptive_candidate_v2_transition_hold",
            "family": "IMPROVED_safe_shortening_v2_h10_min_dwell3_reused_gated_policy",
            "seed": seed,
            "policy": policy,
            "policy_path": rel(p),
            "policy_sha256": sha256(p),
            "terminal_h": BASE_H,
        })
        arms.append({
            "arm_id": "fixed_H25_vehicle_s%d" % seed,
            "role": "same_seed_fixed_H25_comparator",
            "family": "fixed_H25_primary_same_seed",
            "seed": seed,
            "policy": {"id": "fixed", "task": TASK},
            "policy_path": None,
            "policy_sha256": canonical_sha({"id": "fixed", "task": TASK, "h": BASE_H}),
            "terminal_h": BASE_H,
        })
    return arms


def randomized_schedule(arms: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    index = 0
    for repeat in range(REPEATS):
        arm_order = [int(x) for x in rng.permutation(len(arms)).tolist()]
        for arm_index in arm_order:
            case_order = [int(x) for x in rng.permutation(CASES_EXPECTED).tolist()]
            for case_id in case_order:
                schedule.append({
                    "episode_index": index,
                    "repeat": int(repeat),
                    "arm_id": arms[arm_index]["arm_id"],
                    "arm_index": arm_index,
                    "seed": int(arms[arm_index]["seed"]),
                    "case": int(case_id),
                })
                index += 1
    expected = len(arms) * CASES_EXPECTED * REPEATS
    if len(schedule) != expected:
        raise RuntimeError("Unexpected schedule length")
    return schedule


def v2_decide(arm: Mapping[str, Any], ctx: Dict[str, Any], dwell_state: Dict[str, int],
              previous_initial_failure: bool, previous_final_failure: bool) -> Tuple[int, Dict[str, Any]]:
    if arm["role"] == "same_seed_fixed_H25_comparator":
        return BASE_H, {"kind": "fixed", "reason": "same_seed_H25", "raw_horizon": BASE_H, "selected_horizon": BASE_H}

    if previous_initial_failure or previous_final_failure:
        before = int(dwell_state.get("h10_remaining_after_previous", 0))
        dwell_state["h10_remaining_after_previous"] = 0
        return BASE_H, {
            "kind": "safe_shortening_v2_transition_hold",
            "fallback": True,
            "reason": "previous_solver_failure_cancels_dwell",
            "raw_horizon": BASE_H,
            "selected_horizon": BASE_H,
            "clamped_to_base": False,
            "v2_h10_dwell_remaining_before": before,
            "v2_h10_dwell_remaining_after": 0,
        }

    raw_h, gate = v1.gated_decide(arm["policy"], ctx)
    raw_h = int(raw_h)
    if raw_h not in range(5, 51, 5):
        raise RuntimeError("unsafe non-grid horizon requested: %r" % raw_h)
    clamped_raw = min(raw_h, BASE_H)
    before = int(dwell_state.get("h10_remaining_after_previous", 0))
    dwell_started = False
    dwell_forced = False
    if before > 0:
        selected = 10
        after = before - 1
        dwell_forced = raw_h != 10
    elif clamped_raw == 10:
        selected = 10
        after = H10_DWELL_STEPS - 1
        dwell_started = True
    else:
        selected = clamped_raw
        after = 0
    dwell_state["h10_remaining_after_previous"] = int(after)
    return int(selected), {
        "kind": "safe_shortening_v2_transition_hold",
        "fallback": False,
        "raw_horizon": raw_h,
        "selected_horizon": int(selected),
        "clamped_to_base": raw_h > BASE_H,
        "gate": gate,
        "v2_h10_dwell_steps": H10_DWELL_STEPS,
        "v2_h10_dwell_remaining_before": before,
        "v2_h10_dwell_remaining_after": int(after),
        "v2_h10_dwell_started": bool(dwell_started),
        "v2_h10_dwell_forced": bool(dwell_forced),
    }


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                      terminal_load_s: float, episode_wall_s: float, case_id: int, repeat: int,
                      arm: Mapping[str, Any]) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross_times = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"]["selection_s"]) for r in trace]
    logging_times = [float(r["timing"].get("logging_s", 0.0)) for r in trace]
    solver_times: List[float] = []
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    clamps = 0
    fallbacks = 0
    dwell_starts = 0
    dwell_forced = 0
    dwell_executed_h10 = 0
    for row in trace:
        horizons[str(row["horizon"])] = horizons.get(str(row["horizon"]), 0) + 1
        decision_meta = row.get("decision", {})
        raw = decision_meta.get("raw_horizon", row["horizon"])
        raw_horizons[str(raw)] = raw_horizons.get(str(raw), 0) + 1
        clamps += int(bool(decision_meta.get("clamped_to_base", False)))
        fallbacks += int(bool(decision_meta.get("fallback", False)))
        dwell_starts += int(bool(decision_meta.get("v2_h10_dwell_started", False)))
        dwell_forced += int(bool(decision_meta.get("v2_h10_dwell_forced", False)))
        dwell_executed_h10 += int(row["horizon"] == 10 and decision_meta.get("kind") == "safe_shortening_v2_transition_hold")
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    out = dict(metric)
    out.update({
        "arm_id": arm["arm_id"],
        "family": arm["family"],
        "role": arm["role"],
        "seed": int(arm["seed"]),
        "case": int(case_id),
        "repeat": int(repeat),
        "episode_failure": not bool(metric.get("success")),
        "construction_s": float(construction_s),
        "terminal_load_s_amortized_reference": float(terminal_load_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision_times),
        "decision_gross_timing_s": values_summary(decision_gross_times),
        "controller_timing_s_logging_deducted": values_summary(controller_times),
        "selection_timing_s": values_summary(selection_times),
        "logging_timing_s": values_summary(logging_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "horizon_counts": horizons,
        "raw_horizon_counts_before_clamp": raw_horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "clamped_steps": int(clamps),
        "solver_failure_fallback_steps": int(fallbacks),
        "v2_h10_dwell_starts": int(dwell_starts),
        "v2_h10_dwell_forced_steps": int(dwell_forced),
        "v2_h10_dwell_executed_h10_steps": int(dwell_executed_h10),
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
    dwell_state = {"h10_remaining_after_previous": 0}
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
                recovery["step"] = t
                ctx = v1.context(env, TASK)
                ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                select_start = time.perf_counter()
                selected_h, decision = v2_decide(arm, ctx, dwell_state, previous_initial, previous_final)
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
        summary = summarize_episode(trace, reset_record, construction_s, terminal_load_s, time.perf_counter() - episode_start, int(item["case"]), int(item["repeat"]), arm)
        summary.update({
            "episode_index": int(item["episode_index"]),
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
        "v2_h10_dwell_starts": int(sum(e.get("v2_h10_dwell_starts", 0) for e in episodes)),
        "v2_h10_dwell_forced_steps": int(sum(e.get("v2_h10_dwell_forced_steps", 0) for e in episodes)),
        "v2_h10_dwell_executed_h10_steps": int(sum(e.get("v2_h10_dwell_executed_h10_steps", 0) for e in episodes)),
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
        "unique_horizons": sorted(int(h) for h in horizons),
        "adapted_below_H25": any(int(h) < BASE_H for h in horizons),
        "unsafe_above_H25_dispatch": any(int(h) > BASE_H for h in horizons),
    })
    return out


def compare(aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        a = aggregates["safe_shortening_v2_hold3_vehicle_s%d" % seed]
        b = aggregates["fixed_H25_vehicle_s%d" % seed]
        out[str(seed)] = {
            "adaptive_unique_horizons": a["unique_horizons"],
            "fixed_unique_horizons": b["unique_horizons"],
            "adapted_below_H25": a["adapted_below_H25"],
            "unsafe_above_H25_dispatch": a["unsafe_above_H25_dispatch"],
            "success_counts": {"adaptive": a["success_count"], "fixed": b["success_count"]},
            "episode_failure_counts": {"adaptive": a["episode_failure_count"], "fixed": b["episode_failure_count"]},
            "physical_constraint_delta_adaptive_minus_fixed": a["physical_constraint_cost_sum"] - b["physical_constraint_cost_sum"],
            "total_cost_delta_adaptive_minus_fixed": a["total_cost_sum"] - b["total_cost_sum"],
            "decision_time_ratio_adaptive_over_fixed": a["decision_total_s"] / b["decision_total_s"] if b["decision_total_s"] else None,
            "v2_h10_dwell_starts": a["v2_h10_dwell_starts"],
            "v2_h10_dwell_forced_steps": a["v2_h10_dwell_forced_steps"],
        }
    return out


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(v1.__file__).resolve(),
        PROTOCOL,
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/relative_policy_features.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def write_summary(raw: Dict[str, Any]) -> None:
    lines = [
        "# Vehicle safe-shortening v2 transition-hold smoke",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Engineering smoke only on a fresh v2 smoke bank; no validation64 access and no sealed-test access.",
        "",
        f"Protocol: `{raw['protocol']['path']}` sha256 `{raw['protocol']['sha256']}`.",
        "",
        "## Budget/access",
        "",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- New training episodes / gradient steps: `{raw['budget_actual']['new_training_episodes']}` / `{raw['budget_actual']['new_gradient_steps']}`.",
        f"- Replay passed: `{raw['replay']['passed']}` over `{raw['replay']['pairs_checked']}` repeat pairs.",
        f"- validation64_bank_opened: `{raw['validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Arm aggregates",
        "",
        "| arm | seed | episodes | steps | success | failures | physical+constraint | total | decision mean s/step | horizons | raw horizons | dwell starts | dwell forced | solver failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for arm_id in sorted(raw["aggregates"]):
        a = raw["aggregates"][arm_id]
        lines.append(
            f"| `{arm_id}` | {a['seed']} | {a['episodes']} | {a['steps']} | {a['success_count']} | {a['episode_failure_count']} | "
            f"{a['physical_constraint_cost_sum']:.6g} | {a['total_cost_sum']:.6g} | {a['decision_mean_s_per_step']:.6g} | "
            f"{a['horizon_counts']} | {a['raw_horizon_counts_before_clamp']} | {a['v2_h10_dwell_starts']} | {a['v2_h10_dwell_forced_steps']} | {a['solver_failure_steps']} |"
        )
    lines += ["", "## Same-seed smoke deltas", ""]
    for seed, c in raw["paired_comparisons"].items():
        lines.append(
            f"- seed {seed}: horizons={c['adaptive_unique_horizons']}; adapted_below_H25={c['adapted_below_H25']}; "
            f"unsafe_above_H25_dispatch={c['unsafe_above_H25_dispatch']}; success adaptive/fixed={c['success_counts']['adaptive']}/{c['success_counts']['fixed']}; "
            f"physical delta={c['physical_constraint_delta_adaptive_minus_fixed']:.6g}; total delta={c['total_cost_delta_adaptive_minus_fixed']:.6g}; "
            f"decision ratio={c['decision_time_ratio_adaptive_over_fixed']:.6g}; dwell starts={c['v2_h10_dwell_starts']}; dwell forced steps={c['v2_h10_dwell_forced_steps']}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This smoke checks only implementation readiness of the IMPROVED v2 H10 transition-hold wrapper.  It is not validation/model-selection evidence.  Passing smoke allows a fresh non-test development-validation campaign to be frozen; it does not authorize sealed final-test access.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(raw: Dict[str, Any]) -> None:
    request_path = BACKUP_REQUEST.with_name(
        "REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_TRANSITION_HOLD_SMOKE_%s.json" % raw["created_utc"].replace(":", "").replace("-", "").split(".")[0].replace("T", "T")
    )
    write_json(request_path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup v2 transition-hold protocol/source/smoke artifacts before fresh development validation",
        "artifacts": [rel(OUT_DIR), rel(PROTOCOL), rel(Path(__file__).resolve())],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    raw["backup_request"] = rel(request_path)


def append_docs(raw: Dict[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle safe-shortening v2 transition-hold smoke\n\n"
        f"UTC: {raw['created_utc']}. IMPROVED v2 transition-hold protocol frozen and engineering smoke completed on a fresh v2 smoke bank: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, replay passed={raw['replay']['passed']}. "
        "No validation64 bank or sealed test was opened. This is not model-selection/final evidence. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def main() -> int:
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED_vehicle_safe_shortening_v2_transition_hold_smoke",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    preflight = runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise RuntimeError(preflight["diagnosis"] + " " + preflight.get("exception", ""))
    v1.latency_verify()
    if not PROTOCOL.exists():
        raise RuntimeError("v2 protocol must be frozen before smoke")
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
        terminals[seed] = load_terminal(seed, terminal_hashes)
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
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    if not replay["passed"]:
        raise RuntimeError("Repeat replay mismatch: %r" % replay["mismatches"][:3])
    control_steps = int(sum(e["steps"] for e in episodes))
    if len(episodes) != 24 or control_steps > 3600:
        raise RuntimeError("Smoke budget exceeded or incomplete")
    aggregates: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        agg = aggregate([e for e in episodes if e["arm_id"] == arm["arm_id"]])
        agg.update({"seed": int(arm["seed"]), "family": arm["family"], "role": arm["role"], "policy": arm["policy"], "policy_path": arm["policy_path"], "policy_sha256": arm["policy_sha256"]})
        aggregates[arm["arm_id"]] = agg
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_vehicle_safe_shortening_v2_h10_min_dwell3_reused_gated_policies_not_original_SAC",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": "fresh_engineering_smoke_bank_only_v2_transition_hold",
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "bank": bank_info,
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "method_change": {
            "from": "safe-shortening v1 reused gated policies with clamp/fallback",
            "to": "v2 adds H10 minimum dwell of 3 consecutive executed steps after a raw H10 request; fallback cancels dwell",
            "h10_dwell_steps": H10_DWELL_STEPS,
            "new_gradient_steps": 0,
        },
        "budget_declared": {"episodes_exact": 24, "control_step_upper_bound": 3600, "new_training_episodes": 0, "new_gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "resets": int(sum(e["resets_metered"] for e in episodes)), "environment_constructions": len(episodes), "new_training_episodes": 0, "new_gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "arms": arms,
        "randomized_schedule": schedule,
        "terminal_sources": terminal_hashes,
        "episodes": episodes,
        "aggregates": aggregates,
        "paired_comparisons": compare(aggregates),
        "replay": replay,
        "interpretation_limits": ["engineering smoke only", "not validation/model selection", "reuses historical gated policies with zero new gradient training", "H distribution is not timing evidence", "sealed final test remains closed"],
    }
    write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], PROTOCOL, Path(__file__).resolve(), Path(v1.__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
        "backup_request": raw["backup_request"],
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "headline": {"control_steps": control_steps, "paired_comparisons": raw["paired_comparisons"]},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "replay_passed": replay["passed"],
        "validation64_bank_opened": False,
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
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "next_recovery_hint": "Inspect failure, preserve this directory, and create a versioned one-variable recovery if necessary. Use legacy interpreter for TF1 runtime.",
        })
        raise
