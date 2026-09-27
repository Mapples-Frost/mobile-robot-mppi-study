#!/usr/bin/env python3
"""Fresh vehicle safe-shortening v1 development-validation paired shard runner.

This is an IMPROVED-method development-validation runner, not ORIGINAL Bøhn SAC
and not a final-test runner.  It follows the frozen safe-shortening v1 protocol:
reuse the audited safe-shortening policies for seeds 0/1/2, compare against a
strong fixed-H grid, measure actual controller/solver timing on AWS, and keep the
sealed final test closed.

The first execution generates a new vehicle-only development-validation bank
from RNG 2609273000 (reset snapshots only; no scored rollouts), freezes a paired
case-block schedule, and runs one bounded shard.  Shards are complete case
blocks: each shard contains all arms for four fresh development-validation cases,
so same-case paired comparisons are immediately auditable while the full block
remains frozen for continuation.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Reuse the already audited v1 smoke helpers without overwriting them.  This
# imports the frozen safe-shortening policy wrapper, timing utilities, simulator
# construction and solver-recovery instrumentation.
import vehicle_safe_shortening_v1_smoke as v1  # noqa:E402
from run import snapshot  # noqa:E402

ROOT = v1.ROOT
ART = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17"
TASK = "vehicle"
SEEDS = (0, 1, 2)
H_GRID = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
MAX_STEPS = 150
DEVVAL_CASES = 64
CASES_PER_SHARD = 4
BANK_RNG = 2609273000
SCHEDULE_SEED = 2609273100
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_safe_shortening_v1_protocol_20260927.md"
OUT_ROOT = ROOT / "research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1"
BANK_DIR = OUT_ROOT / "bank"
BANK_PATH = BANK_DIR / "vehicle_safe_shortening_v1_devval64_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
GATE_PATH = OUT_ROOT / "vehicle_safe_shortening_v1_devval64_gate.json"
GATE_COMPLETED = OUT_ROOT / "gate_completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-safe-shortening-v1-devval64-shard-20260927-v1"


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
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


def safe_name(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in "_.=-" else "_")
    return "".join(out)[:120]


def terminal_h25_source(seed: int) -> Path:
    return v1.model_dir(TASK, seed)


def independent_seed0_source(h: int) -> Path:
    if h in (5, 10, 15):
        return ART / ("results/paper_defaults/vehicle_fixed_h%d" % h)
    return ART / ("results/paper_exact_grid_2026-09-23/vehicle_fixed_h%d" % h)


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(v1.__file__).resolve(),
        PROTOCOL,
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_protocol.py",
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


def verify_runtime() -> Dict[str, Any]:
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
        data = read_json(BANK_PATH)
        if data.get("task") != TASK or data.get("split") != "safe_shortening_v1_devval64" or len(data.get("cases", [])) != DEVVAL_CASES:
            raise RuntimeError("Existing devval bank has unexpected metadata")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": len(data["cases"])}
    if BANK_DIR.exists():
        leftovers = [p for p in BANK_DIR.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise RuntimeError("Partial devval bank exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10]))
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, gen_dir)
    recovery = v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(BANK_RNG)
    np.random.seed(BANK_RNG)
    cases: List[Dict[str, Any]] = []
    for cid in range(DEVVAL_CASES):
        recovery.update(enabled=False, events=[], case=cid, step=-1)
        env.reset()
        cases.append(snapshot(env))
    if counts["reset_calls"] != DEVVAL_CASES or counts["step_calls"] != 0:
        raise RuntimeError("Unexpected bank generation counts: %r" % counts)
    bank = {
        "task": TASK,
        "split": "safe_shortening_v1_devval64",
        "rng": BANK_RNG,
        "cases": cases,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sealed_test": False,
        "validation64_historical_bank_opened": False,
        "test_bank_opened": False,
        "purpose": "fresh development validation for IMPROVED safe-shortening v1; not final test",
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "created_now": True,
        "task": TASK,
        "split": "safe_shortening_v1_devval64",
        "rng": BANK_RNG,
        "cases": DEVVAL_CASES,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "new_scored_control_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases": DEVVAL_CASES, "reset_calls": counts["reset_calls"]}


def build_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = []
    for seed in SEEDS:
        p = v1.old_policy_path(seed)
        policy = v1.load_old_policy(seed)
        arms.append({
            "arm_id": "safe_shortening_v1_vehicle_s%d" % seed,
            "role": "adaptive_candidate",
            "family": "IMPROVED_safe_shortening_v1_reused_gated_policy",
            "seed": seed,
            "controller_h": "safe_shortening_v1_policy",
            "terminal_h": 25,
            "terminal_source": rel(terminal_h25_source(seed)),
            "policy": policy,
            "policy_path": rel(p),
            "policy_sha256": sha256(p),
        })
    for seed in SEEDS:
        for h in H_GRID:
            arms.append({
                "arm_id": "matched_terminal_fixed_H%d_vehicle_s%d" % (h, seed),
                "role": "matched_terminal_fixed_grid",
                "family": "matched_terminal_fixed_H_grid",
                "seed": seed,
                "controller_h": h,
                "terminal_h": 25,
                "terminal_source": rel(terminal_h25_source(seed)),
                "policy": {"kind": "constant", "task": TASK, "h": h},
                "policy_path": None,
                "policy_sha256": canonical_sha({"kind": "constant", "task": TASK, "h": h}),
            })
    for h in H_GRID:
        source = independent_seed0_source(h)
        arms.append({
            "arm_id": "independent_terminal_seed0_fixed_H%d" % h,
            "role": "independent_terminal_seed0_grid",
            "family": "independent_terminal_seed0_H_grid",
            "seed": 0,
            "controller_h": h,
            "terminal_h": h,
            "terminal_source": rel(source),
            "policy": {"kind": "constant", "task": TASK, "h": h},
            "policy_path": None,
            "policy_sha256": canonical_sha({"kind": "constant", "task": TASK, "h": h}),
        })
    if len(arms) != 43:
        raise RuntimeError("Unexpected arm count: %d" % len(arms))
    return arms


def make_schedule(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rng = np.random.RandomState(SCHEDULE_SEED)
    case_order = [int(x) for x in rng.permutation(DEVVAL_CASES).tolist()]
    rows: List[Dict[str, Any]] = []
    for block_index, case_index in enumerate(case_order):
        arm_order = [int(x) for x in rng.permutation(len(arms)).tolist()]
        for local_arm_order, arm_index in enumerate(arm_order):
            arm = arms[arm_index]
            rows.append({
                "execution_index": len(rows),
                "case_block_index": block_index,
                "case_index": int(case_index),
                "local_arm_order": local_arm_order,
                "arm_index": arm_index,
                "arm_id": arm["arm_id"],
                "family": arm["family"],
                "seed": int(arm["seed"]),
                "controller_h": arm["controller_h"],
                "terminal_source": arm["terminal_source"],
            })
    episodes_per_shard = CASES_PER_SHARD * len(arms)
    shards = []
    for start in range(0, len(rows), episodes_per_shard):
        chunk = rows[start:start + episodes_per_shard]
        shards.append({
            "shard_index": len(shards),
            "start_execution_index": chunk[0]["execution_index"],
            "end_execution_index_inclusive": chunk[-1]["execution_index"],
            "cases": sorted(set(int(r["case_index"]) for r in chunk)),
            "case_block_indices": sorted(set(int(r["case_block_index"]) for r in chunk)),
            "episodes": len(chunk),
            "status": "planned_not_run",
        })
    return {
        "schedule_seed": SCHEDULE_SEED,
        "case_blocked_for_pairing": True,
        "cases_per_shard": CASES_PER_SHARD,
        "arms": list(arms),
        "rows": rows,
        "shards": shards,
        "episode_count": len(rows),
        "shard_count": len(shards),
        "episodes_per_full_shard": episodes_per_shard,
    }


def freeze_gate_if_needed() -> Dict[str, Any]:
    bank_info = generate_bank_if_needed()
    if GATE_COMPLETED.exists():
        verify_completed_marker(GATE_COMPLETED)
        return read_json(GATE_PATH)
    if GATE_PATH.exists():
        raise RuntimeError("Partial gate exists without completed marker: %s" % rel(GATE_PATH))
    arms = build_arms()
    schedule = make_schedule(arms)
    gate = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_devval64_paired_block_not_original_SAC",
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "source_hashes": source_hashes(),
        "bank": bank_info,
        "split": {
            "id": "vehicle_safe_shortening_v1_devval64",
            "fresh_bank_rng": BANK_RNG,
            "historical_validation64_bank_opened": False,
            "sealed_test_bank_opened": False,
        },
        "schedule": schedule,
        "budgets_declared": {
            "bank_generation_resets": DEVVAL_CASES,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "development_validation_episodes_planned": schedule["episode_count"],
            "development_validation_control_step_upper_bound": schedule["episode_count"] * MAX_STEPS,
            "per_shard_cases": CASES_PER_SHARD,
            "per_full_shard_episodes": schedule["episodes_per_full_shard"],
            "per_full_shard_control_step_upper_bound": schedule["episodes_per_full_shard"] * MAX_STEPS,
            "sealed_test_episodes": 0,
        },
        "selection_rules": {
            "fixed_H_grid": list(H_GRID),
            "primary_same_seed_comparator": "matched_terminal_fixed_H25_vehicle_s{seed}",
            "paired_unit": "fresh development-validation case index; steps/repeats are not independent",
            "no_controller_change_mid_campaign": True,
            "not_final_test_gate": True,
        },
        "access_flags": {
            "historical_validation64_bank_opened": False,
            "sealed_test_bank_opened": False,
            "final_test_authorization_requested": False,
        },
    }
    write_json(GATE_PATH, gate)
    files = [GATE_PATH, BANK_PATH, BANK_COMPLETED, PROTOCOL, Path(__file__).resolve(), Path(v1.__file__).resolve()]
    write_json(GATE_COMPLETED, {
        "passed": True,
        "development_validation_gate_frozen": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
    })
    return gate


def load_terminal(source_rel: str) -> Tuple[Tuple[Any, Any], Dict[str, Any]]:
    _, SAC, _ = v1.imports()
    folder = ROOT / source_rel
    manifest = read_json(folder / "manifest.json")
    done = read_json(folder / "completed.json")
    if manifest.get("task") != TASK or done.get("status") != "complete" or done.get("steps") != 15000:
        raise RuntimeError("Terminal source is not complete vehicle 15k model: %s" % source_rel)
    model = SAC.load(str(folder / "model.zip"))
    try:
        if v1.weights_hash(model) != done["final_hash"]:
            raise RuntimeError("Loaded terminal weights hash mismatch: %s" % source_rel)
        terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally:
        model.sess.close()
    receipt = {
        "source": source_rel,
        "fixed_horizon": manifest.get("fixed_horizon"),
        "seed": manifest.get("seed"),
        "model_zip_sha256": sha256(folder / "model.zip"),
        "manifest_sha256": sha256(folder / "manifest.json"),
        "completed_sha256": sha256(folder / "completed.json"),
        "weights_hash": done["final_hash"],
    }
    return (terminal[0], terminal[1]), receipt


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                      terminal_load_s: float, episode_wall_s: float, row: Mapping[str, Any],
                      arm: Mapping[str, Any]) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"]["selection_s"]) for r in trace]
    logging_times = [float(r["timing"]["logging_s"]) for r in trace]
    solver_times: List[float] = []
    horizons: Dict[str, int] = {}
    raw_horizons: Dict[str, int] = {}
    clamps = 0
    fallbacks = 0
    for step_row in trace:
        h = str(step_row["horizon"])
        horizons[h] = horizons.get(h, 0) + 1
        decision_meta = step_row.get("decision", {})
        raw = decision_meta.get("raw_horizon", step_row["horizon"])
        raw_horizons[str(raw)] = raw_horizons.get(str(raw), 0) + 1
        clamps += int(bool(decision_meta.get("clamped_to_base", False)))
        fallbacks += int(bool(decision_meta.get("fallback", False)))
        for attempt in step_row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    out = dict(metric)
    out.update({
        "execution_index": int(row["execution_index"]),
        "case_block_index": int(row["case_block_index"]),
        "arm_index": int(row["arm_index"]),
        "arm_id": arm["arm_id"],
        "family": arm["family"],
        "role": arm["role"],
        "seed": int(arm["seed"]),
        "case": int(row["case_index"]),
        "controller_h": arm["controller_h"],
        "terminal_h": arm["terminal_h"],
        "terminal_source": arm["terminal_source"],
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
        "unique_horizons": sorted(int(h) for h in horizons),
        "clamped_steps": int(clamps),
        "solver_failure_fallback_steps": int(fallbacks),
    })
    return out


def decide_for_arm(arm: Mapping[str, Any], env: Any, previous_initial: bool, previous_final: bool) -> Tuple[int, Dict[str, Any], Dict[str, Any], float]:
    ctx = v1.context(env, TASK)
    ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
    start = time.perf_counter()
    if arm["role"] == "adaptive_candidate":
        h, decision = v1.safe_decide(dict(arm), ctx, previous_initial, previous_final)
    else:
        h = int(arm["controller_h"])
        if h not in H_GRID:
            raise RuntimeError("fixed comparator requested non-grid H: %r" % h)
        decision = {"kind": "fixed_grid", "horizon": h, "raw_horizon": h, "fallback": False, "clamped_to_base": False}
    selection_s = time.perf_counter() - start
    return int(h), decision, ctx, selection_s


def run_episode(shard_dir: Path, row: Mapping[str, Any], arm: Mapping[str, Any], case: Mapping[str, Any],
                terminal: Tuple[Any, Any], terminal_load_s: float) -> Dict[str, Any]:
    ep_dir = shard_dir / "episodes" / (
        "exec%04d_case%02d_%s" % (int(row["execution_index"]), int(row["case_index"]), safe_name(str(arm["arm_id"])))
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
        recovery.update(enabled=False, events=[], case=int(row["case_index"]), step=-1)
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
                selected_h, decision, ctx, selection_s = decide_for_arm(arm, env, previous_initial, previous_final)
                _, terminated, step_row = v1.observed_step(env, TASK, selected_h, dict(case), t)
                if len(measured) != 1:
                    raise RuntimeError("Expected exactly one measured controller call")
                timing = measured.pop()
                timing.update({
                    "selection_s": float(selection_s),
                    "decision_s": float(selection_s + timing["controller_s"]),
                    "decision_gross_s": float(selection_s + timing["controller_gross_s"]),
                })
                step_row.update({
                    "policy_context": ctx,
                    "decision": decision,
                    "recovery": recovery["events"][-1],
                    "timing": timing,
                    "devval_execution_index": int(row["execution_index"]),
                    "devval_case_index": int(row["case_index"]),
                    "arm_id": arm["arm_id"],
                })
                stream.write(json.dumps(step_row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(step_row)
                previous_initial = not bool(step_row["recovery"]["attempts"][0]["success"])
                previous_final = not bool(step_row["solver_success"])
                if terminated:
                    break
        if not trace or not trace[-1].get("termination"):
            raise RuntimeError("Episode did not terminate within max steps")
        v1.audit_trace(TASK, dict(case), trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(trace, reset_record, construction_s, terminal_load_s,
                                    time.perf_counter() - episode_start, row, arm)
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


def aggregate(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hcounts: Dict[str, int] = {}
    raw_hcounts: Dict[str, int] = {}
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            hcounts[str(h)] = hcounts.get(str(h), 0) + int(n)
        for h, n in (e.get("raw_horizon_counts_before_clamp") or {}).items():
            raw_hcounts[str(h)] = raw_hcounts.get(str(h), 0) + int(n)
    steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    decision_total = float(math.fsum(float((e.get("decision_timing_s") or {}).get("sum", 0.0)) for e in episodes))
    solver_count = int(sum(int((e.get("solver_attempt_timing_s") or {}).get("count", 0)) for e in episodes))
    solver_total = float(math.fsum(float((e.get("solver_attempt_timing_s") or {}).get("sum", 0.0)) for e in episodes))
    return {
        "episodes": len(episodes),
        "cases": sorted(set(int(e.get("case")) for e in episodes)),
        "steps": steps,
        "success_count": int(sum(1 for e in episodes if e.get("success"))),
        "episode_failure_count": int(sum(1 for e in episodes if e.get("episode_failure"))),
        "constraint_count": int(sum(1 for e in episodes if e.get("constraint"))),
        "initial_failed_steps": int(sum(int(e.get("initial_failed_steps", 0)) for e in episodes)),
        "solver_failure_steps": int(sum(int(e.get("solver_failure_steps", 0)) for e in episodes)),
        "retries": int(sum(int(e.get("retries", 0)) for e in episodes)),
        "switches": int(sum(int(e.get("switches", 0)) for e in episodes)),
        "clamped_steps": int(sum(int(e.get("clamped_steps", 0)) for e in episodes)),
        "solver_failure_fallback_steps": int(sum(int(e.get("solver_failure_fallback_steps", 0)) for e in episodes)),
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "h_penalty_sum": float(math.fsum(float(e.get("h_penalty", 0.0)) for e in episodes)),
        "decision_total_s": decision_total,
        "decision_mean_s_per_step": (decision_total / steps if steps else None),
        "solver_total_s": solver_total,
        "solver_mean_s_per_attempt": (solver_total / solver_count if solver_count else None),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_gross_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
        "episode_wall_total_s": float(math.fsum(float(e.get("episode_wall_s_including_construction_reset_tracewrites", 0.0)) for e in episodes)),
        "horizon_counts": hcounts,
        "raw_horizon_counts_before_clamp": raw_hcounts,
        "unique_horizons": sorted(int(h) for h in hcounts),
        "used_h_above_25": any(int(h) > 25 for h in hcounts),
        "used_h_below_25": any(int(h) < 25 for h in hcounts),
    }


def paired_primary_deltas(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_arm_case: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for e in episodes:
        by_arm_case[(str(e["arm_id"]), int(e["case"]))] = e
    out: Dict[str, Any] = {}
    for seed in SEEDS:
        a_id = "safe_shortening_v1_vehicle_s%d" % seed
        f_id = "matched_terminal_fixed_H25_vehicle_s%d" % seed
        cases = sorted(c for (arm, c) in by_arm_case if arm == a_id and (f_id, c) in by_arm_case)
        rows = []
        for case in cases:
            a = by_arm_case[(a_id, case)]
            f = by_arm_case[(f_id, case)]
            rows.append({
                "case": case,
                "adaptive_success": bool(a.get("success")),
                "fixed_success": bool(f.get("success")),
                "adaptive_horizons": a.get("horizon_counts"),
                "fixed_horizons": f.get("horizon_counts"),
                "delta_physical_constraint_cost": float(a.get("physical_constraint_cost", 0.0)) - float(f.get("physical_constraint_cost", 0.0)),
                "delta_total_cost": float(a.get("total_cost", 0.0)) - float(f.get("total_cost", 0.0)),
                "decision_ratio": ((a.get("decision_timing_s") or {}).get("sum", 0.0) / (f.get("decision_timing_s") or {}).get("sum", 1.0)),
            })
        out[str(seed)] = {
            "paired_cases": cases,
            "case_deltas": rows,
            "adaptive_success_count": int(sum(1 for r in rows if r["adaptive_success"])),
            "fixed_success_count": int(sum(1 for r in rows if r["fixed_success"])),
            "physical_delta_sum": float(math.fsum(r["delta_physical_constraint_cost"] for r in rows)),
            "total_delta_sum": float(math.fsum(r["delta_total_cost"] for r in rows)),
            "decision_ratio_mean_unweighted_cases": (float(np.mean([r["decision_ratio"] for r in rows])) if rows else None),
        }
    return out


def write_summary(raw: Mapping[str, Any], shard_dir: Path) -> None:
    lines = [
        "# Vehicle safe-shortening v1 fresh development-validation shard",
        "",
        "Created UTC: `%s`." % raw["created_utc"],
        "",
        "This is fresh development-validation evidence for an IMPROVED adaptive-horizon method; it is not final-test evidence and not ORIGINAL SAC reproduction.",
        "",
        "## Access and budget",
        "",
        "- Historical validation64 bank opened: `%s`." % raw["access_flags"]["historical_validation64_bank_opened"],
        "- Sealed final test opened: `%s`." % raw["access_flags"]["sealed_test_bank_opened"],
        "- Bank generation reset calls counted: `%s`." % raw["budget_actual"]["bank_generation_reset_calls_counted_total"],
        "- Shard episodes/control steps: `%s` / `%s`." % (raw["budget_actual"]["episodes"], raw["budget_actual"]["control_steps"]),
        "- New gradient steps: `0`.",
        "- Shard cases: `%s`." % raw["shard"]["cases"],
        "",
        "## Primary same-seed H25 paired deltas in this shard",
        "",
    ]
    for seed, item in sorted(raw["paired_primary_H25_deltas"].items()):
        lines.append("- seed %s: paired_cases=%s, success adaptive/fixed=%s/%s, physical_delta_sum=%.6g, total_delta_sum=%.6g, decision_ratio_mean=%s" % (
            seed, item["paired_cases"], item["adaptive_success_count"], item["fixed_success_count"],
            item["physical_delta_sum"], item["total_delta_sum"], item["decision_ratio_mean_unweighted_cases"]))
    lines += ["", "## Adaptive arm aggregates", ""]
    for arm_id in sorted(k for k in raw["aggregates_by_arm"] if k.startswith("safe_shortening")):
        a = raw["aggregates_by_arm"][arm_id]
        lines.append("- `%s`: episodes=%d, success=%d, failures=%d, phys=%.6g, decision_mean_s_per_step=%s, horizons=%s, H>25=%s" % (
            arm_id, a["episodes"], a["success_count"], a["episode_failure_count"], a["physical_constraint_cost_sum"],
            a["decision_mean_s_per_step"], a["horizon_counts"], a["used_h_above_25"]))
    lines += ["", "Interpretation: this shard is a bounded paired case-block from a frozen fresh development-validation campaign. Do not select, retune, or request final test from this shard alone; continue remaining shards and aggregate all fixed-H grid comparators."]
    (shard_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    token = "<!-- %s-shard%02d -->" % (MARKER, int(raw["shard_index"]))
    text = (
        token + "\n"
        "## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard %02d\n\n"
        "UTC: %s. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: %d episodes, %d control steps, cases %s, no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `%s`, `%s`, `%s`.\n"
    ) % (
        int(raw["shard_index"]), raw["created_utc"], raw["budget_actual"]["episodes"], raw["budget_actual"]["control_steps"],
        raw["shard"]["cases"], raw["artifacts"]["summary_md"], raw["artifacts"]["raw_json"], raw["artifacts"]["completed_json"]
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if token not in old:
                path.write_text(old.rstrip() + "\n\n" + text, encoding="utf-8")


def write_backup_request(raw: Mapping[str, Any]) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD%02d_%s.json" % (int(raw["shard_index"]), stamp))
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup fresh safe-shortening v1 development-validation shard before continuing additional shards",
        "artifacts": [
            raw["artifacts"]["summary_md"],
            raw["artifacts"]["raw_json"],
            raw["artifacts"]["completed_json"],
            rel(GATE_PATH),
            rel(GATE_COMPLETED),
            rel(BANK_PATH),
            rel(BANK_COMPLETED),
            rel(Path(__file__).resolve()),
            rel(PROTOCOL),
        ],
        "development_validation_episodes": raw["budget_actual"]["episodes"],
        "development_validation_control_steps": raw["budget_actual"]["control_steps"],
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    })
    return path


def run_shard(shard_index: int) -> Dict[str, Any]:
    if shard_index < 0:
        raise RuntimeError("Invalid shard index")
    runtime = verify_runtime()
    if not runtime.get("passed"):
        raise RuntimeError(runtime.get("diagnosis") + ": " + runtime.get("exception", ""))
    v1.latency_verify()
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen v1 protocol missing")
    gate = freeze_gate_if_needed()
    schedule = gate["schedule"]
    if shard_index >= len(schedule["shards"]):
        raise RuntimeError("Shard index out of range")
    shard = schedule["shards"][shard_index]
    shard_dir = OUT_ROOT / ("shard%02d" % shard_index)
    if shard_dir.exists():
        completed = shard_dir / "completed.json"
        if completed.exists():
            verify_completed_marker(completed)
            raise SystemExit("Shard already completed and verified: %s" % rel(shard_dir))
        leftovers = [p for p in shard_dir.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise RuntimeError("Partial shard exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10]))
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(shard_dir / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_devval64_shard_not_original_SAC",
        "shard_index": shard_index,
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "runtime_preflight": runtime,
    })
    rows = [r for r in schedule["rows"] if shard["start_execution_index"] <= int(r["execution_index"]) <= shard["end_execution_index_inclusive"]]
    rows.sort(key=lambda r: int(r["execution_index"]))
    if len(rows) != int(shard["episodes"]):
        raise RuntimeError("Shard row count mismatch")
    write_json(shard_dir / "schedule.json", {"shard": shard, "rows": rows, "arms": schedule["arms"]})
    bank_data = read_json(BANK_PATH)
    arms_by_id = {a["arm_id"]: a for a in schedule["arms"]}
    terminal_cache: Dict[str, Tuple[Tuple[Any, Any], float, Dict[str, Any]]] = {}
    terminal_receipts: Dict[str, Any] = {}
    episodes: List[Dict[str, Any]] = []
    for row in rows:
        arm = arms_by_id[row["arm_id"]]
        source = arm["terminal_source"]
        if source not in terminal_cache:
            load_start = time.perf_counter()
            terminal, receipt = load_terminal(source)
            terminal_cache[source] = (terminal, time.perf_counter() - load_start, receipt)
            terminal_receipts[source] = receipt
            write_json(shard_dir / "terminal_sources_progress.json", terminal_receipts)
        terminal, terminal_load_s, _ = terminal_cache[source]
        case = bank_data["cases"][int(row["case_index"])]
        summary = run_episode(shard_dir, row, arm, case, terminal, terminal_load_s)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "shard_index": shard_index,
            "episodes_done": len(episodes),
            "episodes_expected": len(rows),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "arm_id", "case", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_bank_opened": False,
        }
        write_json(shard_dir / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e["steps"]) for e in episodes))
    if control_steps > len(rows) * MAX_STEPS:
        raise RuntimeError("Control step upper bound exceeded")
    aggregates_by_arm = {arm_id: aggregate([e for e in episodes if e["arm_id"] == arm_id]) for arm_id in sorted({e["arm_id"] for e in episodes})}
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_vehicle_safe_shortening_v1_devval64_paired_shard_not_original_SAC",
        "shard_index": shard_index,
        "shard": shard,
        "runtime_preflight": runtime,
        "protocol": gate["protocol"],
        "gate": {"path": rel(GATE_PATH), "sha256": sha256(GATE_PATH)},
        "bank": {"path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "cases_total": len(bank_data["cases"]), "rng": BANK_RNG},
        "access_flags": {
            "historical_validation64_bank_opened": False,
            "sealed_test_bank_opened": False,
            "final_test_authorization_requested": False,
        },
        "budget_declared": {
            "episodes_exact": len(rows),
            "control_step_upper_bound": len(rows) * MAX_STEPS,
            "development_validation_cases_in_shard": len(shard["cases"]),
            "bank_generation_reset_calls_total": DEVVAL_CASES,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "sealed_test_episodes": 0,
        },
        "budget_actual": {
            "episodes": len(episodes),
            "control_steps": control_steps,
            "environment_constructions": len(episodes),
            "resets": int(sum(int(e["resets_metered"]) for e in episodes)),
            "bank_generation_reset_calls_counted_total": int((read_json(BANK_COMPLETED)).get("reset_calls", DEVVAL_CASES)),
            "new_gradient_steps": 0,
            "sealed_test_episodes": 0,
        },
        "source_hashes": source_hashes(),
        "terminal_sources": terminal_receipts,
        "rows": rows,
        "episodes": episodes,
        "aggregates_by_arm": aggregates_by_arm,
        "paired_primary_H25_deltas": paired_primary_deltas(episodes),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "interpretation_limits": [
            "development-validation shard only; do not claim success or request final test from this shard alone",
            "IMPROVED safe-shortening wrapper using reused historical gated policies; not ORIGINAL SAC",
            "fixed-H grid and all seeds must be aggregated over the complete frozen devval64 campaign before model-selection decisions",
            "H distribution alone is not acceleration evidence; actual measured timing is recorded but shard-level timing is partial",
        ],
    }
    raw_path = shard_dir / "raw.json"
    summary_path = shard_dir / "summary.md"
    completed_path = shard_dir / "completed.json"
    write_json(raw_path, raw)
    raw["artifacts"] = {"raw_json": rel(raw_path), "summary_md": rel(summary_path), "completed_json": rel(completed_path)}
    write_json(raw_path, raw)
    write_summary(raw, shard_dir)
    append_docs(raw)
    backup_request = write_backup_request(raw)
    files = [p for p in shard_dir.rglob("*") if p.is_file() and p.name != "completed.json"] + [GATE_PATH, GATE_COMPLETED, BANK_PATH, BANK_COMPLETED, backup_request, PROTOCOL, Path(__file__).resolve(), Path(v1.__file__).resolve()]
    write_json(completed_path, {
        "passed": True,
        "method": raw["method"],
        "shard_index": shard_index,
        "development_validation_evidence": True,
        "formal_final_test_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "cases": shard["cases"],
        "backup_request": rel(backup_request),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
    })
    print(json.dumps({
        "completed": rel(completed_path),
        "summary": rel(summary_path),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "cases": shard["cases"],
        "sealed_test_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "adaptive_primary_deltas": raw["paired_primary_H25_deltas"],
    }, sort_keys=True), flush=True)
    return raw


def main() -> int:
    shard_index = 0
    if len(sys.argv) > 1:
        if len(sys.argv) == 3 and sys.argv[1] == "--shard":
            shard_index = int(sys.argv[2])
        else:
            raise RuntimeError("usage: vehicle_safe_shortening_v1_devval_shard_runner.py [--shard N]")
    try:
        run_shard(shard_index)
        return 0
    except SystemExit:
        raise
    except BaseException as exc:
        shard_dir = OUT_ROOT / ("shard%02d" % shard_index)
        shard_dir.mkdir(parents=True, exist_ok=True)
        progress_path = shard_dir / "progress.json"
        write_json(shard_dir / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "shard_index": shard_index,
            "progress": read_json(progress_path) if progress_path.exists() else None,
            "historical_validation64_bank_opened": False,
            "sealed_test_bank_opened": False,
            "new_gradient_steps": 0,
            "next_action": "Preserve partial shard and diagnose one-variable cause before rerun/recovery; do not open sealed test.",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
