#!/usr/bin/env python3
"""Vehicle fixed-H opportunity probe V0 runner.

Fresh development diagnostic under
research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.*

The runner intentionally evaluates only fixed-H arms on a fresh diagnostic bank:
8 selected source-supported vehicle cases x H={5,10,...,50}.  It does not train,
does not open the historical validation64 bank, and does not open the sealed test.
A verified external backup proof is required before execution because the frozen
preflight explicitly made the rollout backup-blocked.
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
REPRO = ROOT / "experiments/bohn2021_reproduction"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

# Reuse already-audited AWS smoke machinery for environment construction,
# solver-recovery tracing, timing and trace auditing.  This runner supplies its
# own bank/protocol/analysis and does not reuse any adaptive policy selection.
import vehicle_gated_horizon_risk_reselection_v1_smoke as smoke_base  # noqa:E402
from run import snapshot, weights_hash  # noqa:E402
from runtime import imports  # noqa:E402

v1 = smoke_base.v1

TASK = "vehicle"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MAX_STEPS = 150
CANDIDATE_RESETS = 24
SELECTED_CASES = 8
BANK_RNG = 2609286001
ORDER_SEED = 2609286101
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_fixed_h_opportunity_probe_v0_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
FROZEN_MD = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.md"
FROZEN_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.json"
PREFLIGHT_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z/completed.json"
PREFLIGHT_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z/raw.json"
CAPABILITY_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json"
CAPABILITY_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/raw.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v0_runner_state_20260928.md"
MARKER = "vehicle-fixed-h-opportunity-probe-v0-runner-20260928"


class ContractError(RuntimeError):
    """Raised for protocol/access/budget violations."""


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
        raise ContractError("Completed marker did not pass: %s" % rel(path))
    for name, expected in done.get("hashes", {}).items():
        actual = sha256(ROOT / name)
        if actual != expected:
            raise ContractError("Hash mismatch for %s" % name)
    return done


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed_marker(completed)
        raise SystemExit("fixed-H opportunity probe V0 already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("Partial fixed-H opportunity probe output exists; inspect/recover before rerun: " + ", ".join(rel(p) for p in leftovers[:20]))


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def verify_backup_proof(path: Path) -> Dict[str, Any]:
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
    has_asset = bool(proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("packages_this_run"))
    if not has_asset:
        raise ContractError("Backup proof lacks a verified release/package SHA")
    pre = read_json(PREFLIGHT_COMPLETED)
    pre_time = parse_time(pre.get("created_utc"))
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if pre_time is not None and proof_time is not None and proof_time < pre_time:
        raise ContractError("Backup proof predates the fixed-H opportunity preflight")
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "time": proof_time.isoformat() if proof_time else None,
        "time_checked_against_preflight": proof_time is not None and pre_time is not None,
        "raw_status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
    }


def verify_protocol_inputs() -> Dict[str, Any]:
    for path in (FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, PREFLIGHT_RAW, CAPABILITY_COMPLETED, CAPABILITY_RAW):
        if not path.exists():
            raise ContractError("Required frozen input missing: %s" % rel(path))
    pre = verify_completed_marker(PREFLIGHT_COMPLETED)
    cap = verify_completed_marker(CAPABILITY_COMPLETED)
    if pre.get("hard_pass") is not True or pre.get("historical_validation64_bank_opened") is not False or pre.get("sealed_test_accessed") is not False:
        raise ContractError("Preflight access/readiness flags are invalid")
    if cap.get("hard_pass") is not True or cap.get("historical_validation64_bank_opened") is not False or cap.get("sealed_test_accessed") is not False:
        raise ContractError("Capability diagnostic access/readiness flags are invalid")
    protocol = read_json(FROZEN_JSON)
    if protocol.get("protocol_id") != "vehicle_fixed_h_opportunity_probe_v0_frozen_20260928":
        raise ContractError("Unexpected frozen protocol id")
    if protocol.get("split") != "vehicle_fixed_h_opportunity_probe_v0_fresh_no_validation64_no_test":
        raise ContractError("Unexpected frozen split")
    if list(protocol["rollout_design"]["horizons"]) != HORIZONS:
        raise ContractError("Frozen horizon grid changed")
    if int(protocol["rollout_design"]["episodes_exact"]) != SELECTED_CASES * len(HORIZONS):
        raise ContractError("Frozen episode count mismatch")
    term = protocol.get("terminal_grid_readiness") or {}
    if term.get("available_all_required") is not True:
        raise ContractError("Frozen terminal readiness is not true")
    return {
        "protocol": protocol,
        "hashes": {rel(p): sha256(p) for p in (FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, PREFLIGHT_RAW, CAPABILITY_COMPLETED, CAPABILITY_RAW)},
    }


def runtime_preflight() -> Dict[str, Any]:
    try:
        import tensorflow as tf  # noqa:F401
    except BaseException as exc:
        return {
            "passed": False,
            "exception": repr(exc),
            "python": sys.version,
            "executable": sys.executable,
            "diagnosis": "legacy Python/TF1 interpreter is required for vehicle fixed-H opportunity probe",
        }
    return {
        "passed": True,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "python": sys.version,
        "executable": sys.executable,
    }


def true_series(case: Mapping[str, Any], key: str) -> List[float]:
    out: List[float] = []
    for row in (case.get("tvp") or {}).get(key, []) or []:
        vals = row.get("true") if isinstance(row, Mapping) else None
        if vals:
            out.append(float(vals[0]))
    return out


def first_true(case: Mapping[str, Any], key: str) -> Optional[float]:
    vals = true_series(case, key)
    return vals[0] if vals else None


def case_metadata(case: Mapping[str, Any], candidate_index: int) -> Dict[str, Any]:
    ref = case.get("reference") or {}
    theta_r = float(ref.get("theta_r", float("nan")))
    traj_steps = int(ref.get("traj_steps", 0) or 0)
    tx = true_series(case, "trajectory_x")
    ty = true_series(case, "trajectory_y")
    n_path = min(len(tx), len(ty), max(traj_steps, 1))
    if n_path <= 0:
        path_length = float("nan")
        goal_distance = float("nan")
    else:
        path_length = float(math.hypot(tx[n_path - 1] - tx[0], ty[n_path - 1] - ty[0]))
        goal_distance = float(math.hypot(tx[n_path - 1], ty[n_path - 1]))
    min_clearance = float("inf")
    object_summaries: List[Dict[str, Any]] = []
    for i in range(3):
        ox = true_series(case, "obj_%d_x" % i)
        oy = true_series(case, "obj_%d_y" % i)
        rr = true_series(case, "obj_%d_r" % i)
        n = min(n_path, len(ox), len(oy), len(rr))
        if n <= 0:
            object_summaries.append({"object": i, "available": False})
            continue
        clearances = [math.hypot(tx[j] - ox[j], ty[j] - oy[j]) - rr[j] for j in range(n)]
        obj_min = float(min(clearances))
        min_clearance = min(min_clearance, obj_min)
        object_summaries.append({
            "object": i,
            "available": True,
            "x0": float(ox[0]),
            "y0": float(oy[0]),
            "r0": float(rr[0]),
            "min_clearance_to_reference": obj_min,
        })
    if not math.isfinite(min_clearance):
        min_clearance = float("nan")
    ns = ref.get("ns") or []
    return {
        "candidate_index": int(candidate_index),
        "theta_r": theta_r,
        "abs_theta_r": abs(theta_r) if math.isfinite(theta_r) else float("nan"),
        "traj_steps": traj_steps,
        "path_length": path_length,
        "goal_distance": goal_distance,
        "min_reference_obstacle_clearance": min_clearance,
        "object_noise_seed": copy.deepcopy(ns),
        "object_summaries": object_summaries,
    }


def normalized_feature(meta: Mapping[str, Any], medians: Mapping[str, float], scales: Mapping[str, float]) -> Tuple[float, float, float]:
    vals = [float(meta.get("abs_theta_r", 0.0)), float(meta.get("traj_steps", 0.0)), float(meta.get("min_reference_obstacle_clearance", 0.0))]
    keys = ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"]
    out = []
    for key, val in zip(keys, vals):
        if not math.isfinite(val):
            val = medians[key]
        out.append((val - medians[key]) / scales[key])
    return float(out[0]), float(out[1]), float(out[2])


def select_case_indices(metas: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"]
    finite: Dict[str, List[float]] = {k: [] for k in keys}
    for m in metas:
        for k in keys:
            val = float(m.get(k, float("nan")))
            if math.isfinite(val):
                finite[k].append(val)
    medians = {k: float(np.median(finite[k])) if finite[k] else 0.0 for k in keys}
    scales = {k: float(max(np.std(finite[k]) if finite[k] else 1.0, 1e-9)) for k in keys}
    enriched: List[Dict[str, Any]] = []
    for m in metas:
        row = dict(m)
        abs_theta = float(row.get("abs_theta_r", medians["abs_theta_r"]))
        traj_steps = float(row.get("traj_steps", medians["traj_steps"]))
        clearance = float(row.get("min_reference_obstacle_clearance", medians["min_reference_obstacle_clearance"]))
        row["theta_bin"] = "high_abs_theta" if abs_theta >= medians["abs_theta_r"] else "low_abs_theta"
        row["length_bin"] = "long_traj" if traj_steps >= medians["traj_steps"] else "short_traj"
        row["clearance_bin"] = "low_clearance" if clearance <= medians["min_reference_obstacle_clearance"] else "high_clearance"
        row["stratum"] = "%s__%s__%s" % (row["theta_bin"], row["length_bin"], row["clearance_bin"])
        z = normalized_feature(row, medians, scales)
        row["normalized_features"] = list(z)
        row["extremeness_score"] = float(sum(abs(x) for x in z))
        enriched.append(row)
    selected: List[int] = []
    selection_reasons: Dict[int, str] = {}
    strata = [
        (theta, length, clearance)
        for theta in ("low_abs_theta", "high_abs_theta")
        for length in ("short_traj", "long_traj")
        for clearance in ("low_clearance", "high_clearance")
    ]
    for theta, length, clearance in strata:
        label = "%s__%s__%s" % (theta, length, clearance)
        candidates = [m for m in enriched if m["stratum"] == label and int(m["candidate_index"]) not in selected]
        if not candidates:
            continue
        choice = max(candidates, key=lambda m: (float(m["extremeness_score"]), -int(m["candidate_index"])))
        idx = int(choice["candidate_index"])
        selected.append(idx)
        selection_reasons[idx] = "one_per_available_stratum:%s" % label
    while len(selected) < SELECTED_CASES:
        remaining = [m for m in enriched if int(m["candidate_index"]) not in selected]
        if not remaining:
            break
        selected_features = [enriched[i]["normalized_features"] for i in selected]
        def diversity_score(m: Mapping[str, Any]) -> Tuple[float, float, int]:
            z = np.asarray(m["normalized_features"], dtype=float)
            if not selected_features:
                mind = float("inf")
            else:
                mind = float(min(np.linalg.norm(z - np.asarray(s, dtype=float)) for s in selected_features))
            return mind, float(m["extremeness_score"]), -int(m["candidate_index"])
        choice = max(remaining, key=diversity_score)
        idx = int(choice["candidate_index"])
        selected.append(idx)
        selection_reasons[idx] = "diversity_fill_after_missing_strata"
    if len(selected) != SELECTED_CASES:
        raise ContractError("Unable to select required number of cases")
    selected_sorted = sorted(selected)
    selected_meta = []
    for idx in selected_sorted:
        row = dict(enriched[idx])
        row["selection_reason"] = selection_reasons[idx]
        selected_meta.append(row)
    return {
        "medians": medians,
        "scales": scales,
        "all_candidate_metadata": enriched,
        "selected_indices": selected_sorted,
        "selected_metadata": selected_meta,
        "selection_rule": "one case per available theta/length/clearance stratum, then farthest-diversity fill if needed; deterministic, no outcome data",
    }


def generate_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        verify_completed_marker(BANK_COMPLETED)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "vehicle_fixed_h_opportunity_probe_v0_fresh_no_validation64_no_test":
            raise ContractError("Existing fixed-H opportunity bank metadata mismatch")
        if len(bank.get("candidate_cases", [])) != CANDIDATE_RESETS or len(bank.get("selected_cases", [])) != SELECTED_CASES:
            raise ContractError("Existing fixed-H opportunity bank dimensions mismatch")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_cases": SELECTED_CASES}
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, gen_dir)
    recovery = v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(BANK_RNG)
    np.random.seed(BANK_RNG)
    cases: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for cid in range(CANDIDATE_RESETS):
        recovery.update(enabled=False, events=[], case=cid, step=-1)
        env.reset()
        case = snapshot(env)
        cases.append(case)
        metas.append(case_metadata(case, cid))
    if counts["step_calls"] != 0 or counts["reset_calls"] != CANDIDATE_RESETS:
        raise ContractError("Unexpected bank generation counts: %r" % counts)
    selection = select_case_indices(metas)
    selected_cases = [cases[i] for i in selection["selected_indices"]]
    bank = {
        "task": TASK,
        "split": "vehicle_fixed_h_opportunity_probe_v0_fresh_no_validation64_no_test",
        "rng": BANK_RNG,
        "candidate_case_count": CANDIDATE_RESETS,
        "selected_case_count": SELECTED_CASES,
        "candidate_cases": cases,
        "selected_cases": selected_cases,
        "selection": selection,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "fresh development-only fixed-H opportunity probe; no validation64 or sealed test access",
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "task": TASK,
        "split": bank["split"],
        "rng": BANK_RNG,
        "candidate_resets": CANDIDATE_RESETS,
        "selected_cases": SELECTED_CASES,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_cases": SELECTED_CASES, "reset_calls": counts["reset_calls"]}


def load_terminal_grid(protocol: Mapping[str, Any]) -> Tuple[Dict[int, Tuple[Any, Any]], Dict[str, Any]]:
    _, SAC, _ = imports()
    sources = (protocol.get("terminal_grid_readiness") or {}).get("terminal_sources") or {}
    terminals: Dict[int, Tuple[Any, Any]] = {}
    receipts: Dict[str, Any] = {}
    for h in HORIZONS:
        source = sources.get(str(h))
        if not source:
            raise ContractError("Missing terminal source for H%d" % h)
        folder = ROOT / source["path"]
        if not folder.exists():
            raise ContractError("Terminal folder missing for H%d: %s" % (h, rel(folder)))
        model_zip = folder / "model.zip"
        manifest_path = folder / "manifest.json"
        completed_path = folder / "completed.json"
        for p in (model_zip, manifest_path, completed_path):
            if not p.exists():
                raise ContractError("Terminal artifact missing: %s" % rel(p))
        if sha256(model_zip) != source["model_zip_sha256"]:
            raise ContractError("Terminal model zip hash mismatch for H%d" % h)
        manifest = read_json(manifest_path)
        done = read_json(completed_path)
        if manifest.get("task") != TASK or int(manifest.get("fixed_horizon")) != h:
            raise ContractError("Terminal manifest mismatch for H%d" % h)
        if done.get("status") != "complete" or int(done.get("steps")) != 15000:
            raise ContractError("Terminal completion mismatch for H%d" % h)
        start = time.perf_counter()
        model = SAC.load(str(model_zip))
        try:
            wh = weights_hash(model)
            if wh != done["final_hash"]:
                raise ContractError("Loaded terminal weights hash mismatch for H%d" % h)
            terminals[h] = model.policy_tf.get_mpc_vfn_weights_and_biases()
        finally:
            model.sess.close()
        receipts[str(h)] = {
            "folder": rel(folder),
            "model_zip_sha256": sha256(model_zip),
            "manifest_sha256": sha256(manifest_path),
            "completed_sha256": sha256(completed_path),
            "weights_hash": done["final_hash"],
            "load_s": float(time.perf_counter() - start),
            "manifest_summary": {"task": manifest.get("task"), "seed": manifest.get("seed"), "fixed_horizon": manifest.get("fixed_horizon"), "steps": manifest.get("steps")},
        }
    return terminals, receipts


def clean_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(trace)
    for row in cleaned:
        row.pop("timing", None)
        for attempt in row.get("recovery", {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float,
                      episode_wall_s: float, case_id: int, selected_meta: Mapping[str, Any], h: int,
                      terminal_receipt: Mapping[str, Any], execution_index: int) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    logging_times = [float(r["timing"].get("logging_s", 0.0)) for r in trace]
    solver_times: List[float] = []
    for row in trace:
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    horizons: Dict[str, int] = {}
    for row in trace:
        hh = str(row["horizon"])
        horizons[hh] = horizons.get(hh, 0) + 1
    out = dict(metric)
    out.update({
        "execution_index": int(execution_index),
        "case": int(case_id),
        "selected_case_index": int(case_id),
        "source_candidate_index": int(selected_meta["candidate_index"]),
        "stratum": selected_meta.get("stratum"),
        "case_metadata": selected_meta,
        "horizon": int(h),
        "arm_id": "fixed_H%02d_seed0_terminal" % h,
        "family": "fixed_H_independent_terminal_seed0_grid",
        "terminal_source": terminal_receipt,
        "episode_failure": not bool(metric.get("success")),
        "construction_s": float(construction_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision_times),
        "decision_gross_timing_s": values_summary(decision_gross),
        "controller_timing_s_logging_deducted": values_summary(controller_times),
        "selection_timing_s": values_summary([0.0 for _ in trace]),
        "logging_timing_s": values_summary(logging_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "horizon_counts": horizons,
        "unique_horizons": sorted(int(x) for x in horizons),
    })
    return out


def run_episode(item: Mapping[str, Any], case: Mapping[str, Any], selected_meta: Mapping[str, Any],
                terminal: Tuple[Any, Any], terminal_receipt: Mapping[str, Any]) -> Dict[str, Any]:
    h = int(item["horizon"])
    case_id = int(item["selected_case_index"])
    ep_dir = OUT_DIR / "episodes" / ("exec%03d_case%02d_H%02d" % (int(item["execution_index"]), case_id, h))
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
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
                raise ContractError("Invalid logging timing bounds")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=case_id, step=-1, horizon=h)
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
                recovery["step"] = int(t)
                recovery["horizon"] = h
                _, terminated, row = v1.observed_step(env, TASK, h, dict(case), t)
                if len(measured) != 1:
                    raise ContractError("Expected exactly one measured controller call")
                timing = measured.pop()
                timing.update({
                    "selection_s": 0.0,
                    "decision_s": float(timing["controller_s"]),
                    "decision_gross_s": float(timing["controller_gross_s"]),
                })
                row.update({
                    "decision": {"kind": "fixed_H", "horizon": h, "terminal": "independent_seed0_H%d" % h},
                    "recovery": recovery["events"][-1],
                    "timing": timing,
                })
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                if terminated:
                    break
        if not trace or not trace[-1].get("termination"):
            raise ContractError("Episode did not terminate within max steps")
        v1.audit_trace(TASK, dict(case), trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(trace, reset_record, construction_s, time.perf_counter() - episode_start,
                                    case_id, selected_meta, h, terminal_receipt, int(item["execution_index"]))
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


def randomized_schedule(selected_count: int) -> List[Dict[str, Any]]:
    base: List[Dict[str, Any]] = []
    idx = 0
    for case_id in range(selected_count):
        for h in HORIZONS:
            base.append({"selected_case_index": int(case_id), "horizon": int(h), "rollout_index": int(idx)})
            idx += 1
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    for exec_idx, base_idx in enumerate(rng.permutation(len(base)).tolist()):
        row = dict(base[int(base_idx)])
        row["execution_index"] = int(exec_idx)
        schedule.append(row)
    return schedule


def aggregate(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    out: Dict[str, Any] = {
        "episodes": len(episodes),
        "steps": steps,
        "success_count": int(sum(1 for e in episodes if e.get("success"))),
        "episode_failure_count": int(sum(1 for e in episodes if e.get("episode_failure"))),
        "constraint_count": int(sum(1 for e in episodes if e.get("constraint"))),
        "initial_failed_steps": int(sum(int(e.get("initial_failed_steps", 0)) for e in episodes)),
        "solver_failure_steps": int(sum(int(e.get("solver_failure_steps", 0)) for e in episodes)),
        "retries": int(sum(int(e.get("retries", 0)) for e in episodes)),
        "recovered_steps": int(sum(int(e.get("recovered_steps", 0)) for e in episodes)),
        "deadline_exceed_steps": int(sum(int(e.get("deadline_exceed_steps", 0)) for e in episodes)),
        "switches": int(sum(int(e.get("switches", 0)) for e in episodes)),
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "performance_cost_sum": float(math.fsum(float(e.get("performance_cost", 0.0)) for e in episodes)),
        "constraint_cost_sum": float(math.fsum(float(e.get("constraint_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "h_penalty_sum": float(math.fsum(float(e.get("h_penalty", 0.0)) for e in episodes)),
        "decision_total_s": float(math.fsum(float(e["decision_timing_s"]["sum"]) for e in episodes)),
        "decision_gross_total_s": float(math.fsum(float(e["decision_gross_timing_s"]["sum"]) for e in episodes)),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
    }
    out.update({
        "decision_mean_s_per_step": out["decision_total_s"] / steps if steps else None,
        "physical_constraint_cost_mean_episode": out["physical_constraint_cost_sum"] / len(episodes) if episodes else None,
        "total_cost_mean_episode": out["total_cost_sum"] / len(episodes) if episodes else None,
        "safe_all_success_no_constraint_no_solver_fail": bool(out["success_count"] == len(episodes) and out["constraint_count"] == 0 and out["initial_failed_steps"] == 0 and out["solver_failure_steps"] == 0),
    })
    return out


def eligible_episode(e: Mapping[str, Any], strict_solver: bool = True) -> bool:
    if not bool(e.get("success")) or bool(e.get("constraint")):
        return False
    if strict_solver and (int(e.get("initial_failed_steps", 0)) > 0 or int(e.get("solver_failure_steps", 0)) > 0):
        return False
    return True


def pareto_set(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    eligible = [r for r in rows if eligible_episode(r, strict_solver=True)]
    if not eligible:
        eligible = [r for r in rows if eligible_episode(r, strict_solver=False)]
    if not eligible:
        eligible = list(rows)
    nondom: List[int] = []
    for a in eligible:
        ah = int(a["horizon"])
        ac = float(a["physical_constraint_cost"])
        at = float(a["decision_timing_s"]["sum"])
        dominated = False
        for b in eligible:
            if int(b["horizon"]) == ah:
                continue
            bc = float(b["physical_constraint_cost"])
            bt = float(b["decision_timing_s"]["sum"])
            if (bc <= ac and bt <= at) and (bc < ac or bt < at):
                dominated = True
                break
        if not dominated:
            nondom.append(ah)
    return sorted(set(nondom))


def analyze_opportunity(episodes: Sequence[Mapping[str, Any]], selected_meta: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_h: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        by_h[str(h)] = aggregate([e for e in episodes if int(e["horizon"]) == h])
    by_case: Dict[str, Any] = {}
    cheapest_horizons: List[int] = []
    fastest_horizons: List[int] = []
    nondominated_sets: List[Tuple[int, ...]] = []
    for case_id in range(len(selected_meta)):
        rows = [e for e in episodes if int(e["case"]) == case_id]
        eligible = [r for r in rows if eligible_episode(r, strict_solver=True)]
        eligibility = "success_no_constraint_no_solver_fail"
        if not eligible:
            eligible = [r for r in rows if eligible_episode(r, strict_solver=False)]
            eligibility = "success_no_constraint"
        if not eligible:
            eligible = list(rows)
            eligibility = "all_rows_failures_retained"
        best_phys = min(eligible, key=lambda r: (float(r["physical_constraint_cost"]), float(r["decision_timing_s"]["sum"]), int(r["horizon"])))
        fastest = min(eligible, key=lambda r: (float(r["decision_timing_s"]["sum"]), float(r["physical_constraint_cost"]), int(r["horizon"])))
        best_total = min(eligible, key=lambda r: (float(r["total_cost"]), float(r["decision_timing_s"]["sum"]), int(r["horizon"])))
        nondom = pareto_set(rows)
        cheapest_horizons.append(int(best_phys["horizon"]))
        fastest_horizons.append(int(fastest["horizon"]))
        nondominated_sets.append(tuple(nondom))
        by_case[str(case_id)] = {
            "source_candidate_index": int(selected_meta[case_id]["candidate_index"]),
            "stratum": selected_meta[case_id].get("stratum"),
            "theta_r": selected_meta[case_id].get("theta_r"),
            "traj_steps": selected_meta[case_id].get("traj_steps"),
            "min_reference_obstacle_clearance": selected_meta[case_id].get("min_reference_obstacle_clearance"),
            "eligibility_filter_used": eligibility,
            "best_physical_horizon": int(best_phys["horizon"]),
            "best_total_horizon": int(best_total["horizon"]),
            "fastest_horizon": int(fastest["horizon"]),
            "nondominated_horizons_cost_vs_decision_total_s": nondom,
            "per_h_summary": {
                str(int(r["horizon"])): {
                    "success": bool(r.get("success")),
                    "termination": r.get("termination"),
                    "steps": int(r.get("steps", 0)),
                    "physical_constraint_cost": float(r.get("physical_constraint_cost", 0.0)),
                    "total_cost": float(r.get("total_cost", 0.0)),
                    "decision_total_s": float(r["decision_timing_s"]["sum"]),
                    "decision_mean_s_per_step": float(r["decision_timing_s"]["mean"]),
                    "initial_failed_steps": int(r.get("initial_failed_steps", 0)),
                    "solver_failure_steps": int(r.get("solver_failure_steps", 0)),
                    "constraint": bool(r.get("constraint")),
                }
                for r in sorted(rows, key=lambda x: int(x["horizon"]))
            },
        }
    safe_horizons = [h for h in HORIZONS if by_h[str(h)]["success_count"] == len(selected_meta) and by_h[str(h)]["constraint_count"] == 0]
    overall_candidates = [h for h in safe_horizons] or HORIZONS
    strongest_total_h = min(overall_candidates, key=lambda h: (by_h[str(h)]["total_cost_sum"], by_h[str(h)]["decision_total_s"], h))
    strongest_physical_h = min(overall_candidates, key=lambda h: (by_h[str(h)]["physical_constraint_cost_sum"], by_h[str(h)]["decision_total_s"], h))
    fastest_overall_h = min(overall_candidates, key=lambda h: (by_h[str(h)]["decision_total_s"], by_h[str(h)]["physical_constraint_cost_sum"], h))
    distinct_best = sorted(set(cheapest_horizons))
    distinct_fastest = sorted(set(fastest_horizons))
    distinct_nondom_sets = sorted(set(nondominated_sets))
    protocol_opportunity = bool(len(distinct_best) >= 2 or len(distinct_fastest) >= 2 or len(distinct_nondom_sets) >= 2)
    weak_single_h = bool(len(distinct_best) == 1 and len(distinct_fastest) == 1 and len(distinct_nondom_sets) == 1)
    return {
        "by_horizon": by_h,
        "by_case": by_case,
        "safe_horizons_success_no_constraint_all_cases": safe_horizons,
        "overall_strongest_total_horizon": int(strongest_total_h),
        "overall_strongest_physical_horizon": int(strongest_physical_h),
        "overall_fastest_safe_horizon": int(fastest_overall_h),
        "distinct_case_best_physical_horizons": distinct_best,
        "distinct_case_fastest_horizons": distinct_fastest,
        "distinct_case_nondominated_sets": [list(x) for x in distinct_nondom_sets],
        "protocol_opportunity_flag_development_only": protocol_opportunity,
        "weak_single_fixed_H_pattern_flag_development_only": weak_single_h,
        "interpretation": "Development-only fixed-H Pareto opportunity map. If one H dominates, document weak adaptive opportunity rather than forcing switching; if strata differ, design a versioned adaptive selector/training diagnostic before any final-test use.",
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    opp = raw["opportunity_analysis"]
    lines = [
        "# Vehicle fixed-H opportunity probe V0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development diagnostic only. Fresh source-supported vehicle cases; fixed-H grid only; no adaptive policy, no training, no historical validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Candidate-bank resets: `{raw['budget_actual']['fresh_bank_candidate_resets']}` / declared `{raw['budget_declared']['fresh_bank_candidate_resets']}`.",
        f"- Rollout episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['rollout_episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Selected source-supported cases",
        "",
        "| selected case | source candidate | stratum | theta_r | traj_steps | min reference obstacle clearance |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for i, meta in enumerate(raw["bank_selection"]["selected_metadata"]):
        lines.append("| %d | %d | `%s` | %.6g | %d | %.6g |" % (
            i, int(meta["candidate_index"]), meta.get("stratum"), float(meta.get("theta_r", float("nan"))), int(meta.get("traj_steps", 0)), float(meta.get("min_reference_obstacle_clearance", float("nan")))
        ))
    lines += [
        "",
        "## Aggregate fixed-H grid",
        "",
        "| H | episodes | success | constraints | init-fail steps | final-fail steps | physical+constraint | total | decision mean s/step | decision total s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        a = opp["by_horizon"][str(h)]
        lines.append("| %d | %d | %d | %d | %d | %d | %.6g | %.6g | %.6g | %.6g |" % (
            h, a["episodes"], a["success_count"], a["constraint_count"], a["initial_failed_steps"], a["solver_failure_steps"],
            a["physical_constraint_cost_sum"], a["total_cost_sum"], a["decision_mean_s_per_step"], a["decision_total_s"]
        ))
    lines += [
        "",
        "## Case/stratum opportunity map",
        "",
        "| case | stratum | best physical H | best total H | fastest H | nondominated H set |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for case_id in sorted(opp["by_case"], key=lambda x: int(x)):
        row = opp["by_case"][case_id]
        lines.append("| %s | `%s` | %d | %d | %d | `%s` |" % (
            case_id, row["stratum"], row["best_physical_horizon"], row["best_total_horizon"], row["fastest_horizon"], row["nondominated_horizons_cost_vs_decision_total_s"]
        ))
    lines += [
        "",
        "## Development interpretation",
        "",
        f"- Safe horizons succeeding without constraints on all selected cases: `{opp['safe_horizons_success_no_constraint_all_cases']}`.",
        f"- Overall strongest total-cost H: `{opp['overall_strongest_total_horizon']}`; strongest physical-cost H: `{opp['overall_strongest_physical_horizon']}`; fastest safe H: `{opp['overall_fastest_safe_horizon']}`.",
        f"- Distinct per-case best physical H: `{opp['distinct_case_best_physical_horizons']}`; distinct fastest H: `{opp['distinct_case_fastest_horizons']}`.",
        f"- protocol_opportunity_flag_development_only: `{opp['protocol_opportunity_flag_development_only']}`; weak_single_fixed_H_pattern_flag_development_only: `{opp['weak_single_fixed_H_pattern_flag_development_only']}`.",
        "",
        "This probe is not validation/model-selection evidence and cannot support a final success claim. It is intended to decide whether scenario opportunity exists before another long adaptive validation campaign.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = raw["created_utc"].replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V0_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup fresh fixed-H opportunity probe V0 before any adaptive selector/training diagnostic or further validation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(FROZEN_MD), rel(FROZEN_JSON)],
    })
    return rel(path)


def append_docs(raw: Mapping[str, Any]) -> None:
    opp = raw["opportunity_analysis"]
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle fixed-H opportunity probe V0\n\n"
        f"UTC: {raw['created_utc']}. Fresh development-only fixed-H opportunity probe completed: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, "
        f"candidate resets={raw['budget_actual']['fresh_bank_candidate_resets']}. "
        f"No validation64 or sealed-test access. Development opportunity flag={opp['protocol_opportunity_flag_development_only']}; "
        f"weak single-H pattern flag={opp['weak_single_fixed_H_pattern_flag_development_only']}; "
        f"safe horizons={opp['safe_horizons_success_no_constraint_all_cases']}; strongest total H={opp['overall_strongest_total_horizon']}. "
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
        Path(__file__).resolve(), Path(smoke_base.__file__).resolve(), Path(v1.__file__).resolve(),
        FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, PREFLIGHT_RAW, CAPABILITY_COMPLETED, CAPABILITY_RAW,
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
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after the V0 preflight and runner source are backed up")
    ap.add_argument("--i-accept-fresh-development-rollout", action="store_true", help="explicit acknowledgement: development diagnostic only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_fresh_development_rollout:
        raise ContractError("Explicit --i-accept-fresh-development-rollout is required")
    assert_fresh_output_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_fixed_h_opportunity_probe_v0_fixed_H_grid_development_diagnostic",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    protocol_inputs = verify_protocol_inputs()
    backup = verify_backup_proof(args.backup_proof)
    preflight = runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(preflight["diagnosis"] + " " + preflight.get("exception", ""))
    v1.latency_verify()
    bank_info = generate_bank_if_needed()
    bank = read_json(BANK_PATH)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    terminals, terminal_receipts = load_terminal_grid(protocol_inputs["protocol"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    schedule = randomized_schedule(len(selected_cases))
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "horizons": HORIZONS})
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        case_id = int(item["selected_case_index"])
        h = int(item["horizon"])
        summary = run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals[h], terminal_receipts[str(h)])
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "case", "horizon", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e["steps"]) for e in episodes))
    declared = protocol_inputs["protocol"]["budget"]
    if len(episodes) != declared["rollout_episodes_exact"] or control_steps > declared["control_step_upper_bound"]:
        raise ContractError("Fixed-H opportunity probe budget violation")
    analysis = analyze_opportunity(episodes, selected_meta)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_fixed_h_opportunity_probe_v0_fixed_H_grid_development_diagnostic_not_model_selection_not_final_test",
        "classification": "diagnostic_development_fixed_H_opportunity_probe",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": protocol_inputs["protocol"]["split"],
        "backup_proof": backup,
        "protocol": {"path": rel(FROZEN_MD), "sha256": sha256(FROZEN_MD), "json_path": rel(FROZEN_JSON), "json_sha256": sha256(FROZEN_JSON)},
        "protocol_inputs": protocol_inputs["hashes"],
        "bank": bank_info,
        "bank_selection": bank["selection"],
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": declared,
        "budget_actual": {
            "fresh_bank_candidate_resets": CANDIDATE_RESETS if bank_info.get("created_now") else 0,
            "episodes": len(episodes),
            "control_steps": control_steps,
            "environment_constructions": len(episodes),
            "episode_resets": int(sum(int(e["resets_metered"]) for e in episodes)),
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "opportunity_analysis": analysis,
        "interpretation_limits": [
            "development diagnostic only", "fixed-H grid only", "not adaptive model selection", "not ORIGINAL SAC", "not final test", "actual timing from this AWS run only", "scenario bank generated from source-supported straight-line vehicle factors",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    state_text = (
        f"# Vehicle fixed-H opportunity probe V0 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} fixed-H episodes / {control_steps} control steps on fresh development bank. "
        f"No validation64 or sealed-test access. Opportunity flag={analysis['protocol_opportunity_flag_development_only']}; "
        f"weak single-H pattern={analysis['weak_single_fixed_H_pattern_flag_development_only']}. "
        "Backup required before further simulations or method-selection work.\n"
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state_text, encoding="utf-8")
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), FROZEN_MD, FROZEN_JSON, PREFLIGHT_COMPLETED, CAPABILITY_COMPLETED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "backup_request": raw["backup_request"],
        "headline": {
            "opportunity_flag_development_only": analysis["protocol_opportunity_flag_development_only"],
            "weak_single_fixed_H_pattern_flag_development_only": analysis["weak_single_fixed_H_pattern_flag_development_only"],
            "overall_strongest_total_horizon": analysis["overall_strongest_total_horizon"],
            "safe_horizons": analysis["safe_horizons_success_no_constraint_all_cases"],
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "opportunity_flag_development_only": analysis["protocol_opportunity_flag_development_only"],
        "weak_single_fixed_H_pattern_flag_development_only": analysis["weak_single_fixed_H_pattern_flag_development_only"],
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
            "next_recovery_hint": "Preserve this partial directory, audit failure, and create a versioned one-variable repair. Use legacy interpreter and a verified backup proof.",
        })
        raise
