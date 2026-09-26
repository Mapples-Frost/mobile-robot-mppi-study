#!/usr/bin/env python3
"""Frozen vehicle validation64 shard runner for the 2026-09-26 gate.

This runner is intentionally separate from the metadata-only gate freezer.  It
supports:

* ``--dry-run``: verify the frozen gate, source/model/policy hashes, schedule,
  budgets, and output refusal behavior without opening vehicle_validation_bank
  content and without any simulations.
* ``--shard N``: after a verified external backup proof is supplied, open the
  frozen vehicle validation bank and execute exactly one contiguous randomized
  validation shard from the gate.  The sealed test bank is never opened here.

The method remains an IMPROVED latency-tree evaluation, not the ORIGINAL SAC
method from Bøhn et al. 2021.
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
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "experiments/bohn2021_reproduction"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from latency_tree_protocol import OUT as LAT_OUT, REG, bank_name, verify as verify_latency_registration  # noqa:E402
from latency_tree_policy import choose, constant, features, policy_key  # noqa:E402
from conservative_canonical_reset import make_env  # noqa:E402
from branch_calibration_run import meter, observed_step  # noqa:E402
from branch_calibration_audit import audit_trace  # noqa:E402
from gated_horizon_timing import LoggingTimer  # noqa:E402
from gated_horizon_search import case_metrics  # noqa:E402
from relative_policy_features import context  # noqa:E402
from runtime import imports  # noqa:E402
from run import weights_hash  # noqa:E402
import conservative_solver_recovery as recovery_module  # noqa:E402

TASK = "vehicle"
MAX_STEPS = 150
VALIDATION_CASES = 64
SHARD_MAX_EPISODES = 224
EXPECTED_GATE_SHA256 = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
GATE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926"
GATE_JSON = GATE_DIR / "vehicle_validation_gate_20260926.json"
GATE_COMPLETED = GATE_DIR / "completed.json"
DRY_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926"
FORMAL_ROOT = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
MARKER_DRY = "vehicle-validation64-shard-runner-dryrun-20260926"
MARKER_SHARD = "vehicle-validation64-shard-complete-20260926"


class ContractError(RuntimeError):
    """Raised for a preregistered contract violation."""


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def root_path(name: str | Path) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return ROOT / path


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


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)[:120]


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


def verify_existing_completed(completed_path: Path) -> Dict[str, Any]:
    done = read_json(completed_path)
    if done.get("passed") is not True:
        raise ContractError(f"Existing completed marker did not pass: {rel(completed_path)}")
    for name, expected in done.get("hashes", {}).items():
        actual = sha256(ROOT / name)
        if actual != expected:
            raise ContractError(f"Existing artifact hash mismatch: {name}")
    return done


def assert_fresh_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    completed = out_dir / "completed.json"
    if completed.exists():
        verify_existing_completed(completed)
        raise SystemExit(f"{rel(out_dir)} already completed and verified; refusing to rerun")
    leftovers = [p for p in out_dir.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError(
            "Partial output exists; preserve and audit before recovery: " + ", ".join(rel(p) for p in leftovers[:20])
        )


def load_gate() -> Dict[str, Any]:
    if not GATE_JSON.exists() or not GATE_COMPLETED.exists():
        raise ContractError("Frozen vehicle validation gate is missing")
    gate_sha = sha256(GATE_JSON)
    if gate_sha != EXPECTED_GATE_SHA256:
        raise ContractError(f"Frozen gate sha mismatch: {gate_sha} != {EXPECTED_GATE_SHA256}")
    completed = read_json(GATE_COMPLETED)
    if completed.get("passed") is not True:
        raise ContractError("Gate completed marker did not pass")
    if completed.get("validation_accessed") is not False or completed.get("test_accessed") is not False:
        raise ContractError("Gate completed marker has unexpected access flags")
    if completed.get("hashes", {}).get(rel(GATE_JSON)) != gate_sha:
        raise ContractError("Gate completed marker does not record the expected gate hash")
    for name, expected in completed.get("hashes", {}).items():
        actual = sha256(ROOT / name)
        if actual != expected:
            raise ContractError(f"Gate artifact hash mismatch: {name}")
    gate = read_json(GATE_JSON)
    required_false = [
        "validation_accessed",
        "validation64_bank_content_opened",
        "test_accessed",
        "sealed_test_bank_content_opened",
        "formal_scientific_evidence_created",
        "final_test_authorization_requested",
    ]
    for name in required_false:
        if gate.get(name) is not False:
            raise ContractError(f"Gate access/authorization flag is not false: {name}")
    if gate.get("new_simulations") != 0 or gate.get("new_gradient_steps") != 0:
        raise ContractError("Gate unexpectedly records simulations or gradient steps")
    schedule = gate.get("validation_schedule") or {}
    if schedule.get("episode_count") != 2688 or schedule.get("unique_rollout_arm_count") != 42:
        raise ContractError("Unexpected gate schedule dimensions")
    if schedule.get("shard_max_episodes") != SHARD_MAX_EPISODES or schedule.get("expected_shards") != 12:
        raise ContractError("Unexpected gate shard plan")
    return gate


def verify_gate_sources(gate: Mapping[str, Any]) -> Dict[str, str]:
    """Verify all source hashes frozen by the metadata gate.

    These hashes intentionally exclude vehicle_validation_bank content.  The
    actual validation bank hash is computed only during a formal shard run.
    """
    verified: Dict[str, str] = {}
    for name, expected in sorted((gate.get("source_hashes") or {}).items()):
        path = ROOT / name
        if not path.exists():
            raise ContractError(f"Frozen source/input missing: {name}")
        actual = sha256(path)
        if actual != expected:
            raise ContractError(f"Frozen source/input changed: {name}")
        verified[name] = actual
    # Also verify the original latency-tree registration without touching the
    # validation/test bank content.
    verify_latency_registration()
    verified[rel(REG)] = sha256(REG)
    return verified


def iter_model_records(value: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(value, dict):
        if "path" in value and "model.zip" in value and "manifest.json" in value and "completed.json" in value:
            yield value
        for child in value.values():
            yield from iter_model_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_model_records(child)


def collect_model_records(gate: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for rec in iter_model_records(gate.get("fixed_comparators")):
        source = rec.get("path")
        if not source:
            continue
        old = records.get(source)
        if old is not None:
            # Same source may appear under multiple logical comparator aliases.
            for key in ("model.zip", "manifest.json", "completed.json"):
                if old[key].get("sha256") != rec[key].get("sha256"):
                    raise ContractError(f"Ambiguous model hash for {source}")
            continue
        records[source] = rec
    return records


def verify_model_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": record.get("path"), "available_complete": bool(record.get("available_complete", True))}
    for name in ("manifest.json", "completed.json", "model.zip"):
        item = record.get(name) or {}
        path = ROOT / item.get("path", "")
        if not path.exists():
            raise ContractError(f"Required model artifact missing: {item.get('path')}")
        expected = item.get("sha256")
        actual = sha256(path)
        if expected and actual != expected:
            raise ContractError(f"Model artifact hash mismatch: {item.get('path')}")
        out[name] = {"path": rel(path), "sha256": actual, "size_bytes": path.stat().st_size}
    manifest = read_json(ROOT / record["manifest.json"]["path"])
    done = read_json(ROOT / record["completed.json"]["path"])
    if manifest.get("task") != TASK or done.get("status") != "complete" or done.get("steps") != 15000:
        raise ContractError(f"Unexpected model manifest/completion: {record.get('path')}")
    out["manifest_summary"] = {
        "task": manifest.get("task"),
        "seed": manifest.get("seed"),
        "fixed_horizon": manifest.get("fixed_horizon"),
        "steps": manifest.get("steps"),
    }
    out["completed_summary"] = {
        "status": done.get("status"),
        "steps": done.get("steps"),
        "final_hash": done.get("final_hash"),
    }
    return out


def verify_policies(gate: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cand in gate.get("learned_candidates") or []:
        arm_id = cand["arm_id"]
        path = ROOT / cand["policy_path"]
        policy = read_json(path)
        if sha256(path) != cand["policy_file_sha256"]:
            raise ContractError(f"Learned policy file hash mismatch: {arm_id}")
        if canonical_hash(policy) != cand["policy_canonical_sha256"]:
            raise ContractError(f"Learned policy canonical hash mismatch: {arm_id}")
        if policy.get("task") != TASK:
            raise ContractError(f"Learned policy task mismatch: {arm_id}")
        completed = ROOT / cand["completed_json"]["path"]
        fit = ROOT / cand["fit_json"]["path"]
        selection = ROOT / cand["selection_registration_json"]["path"]
        for item_path, item_key in ((completed, "completed_json"), (fit, "fit_json"), (selection, "selection_registration_json")):
            expected = cand[item_key].get("sha256")
            actual = sha256(item_path)
            if expected and actual != expected:
                raise ContractError(f"Learned policy metadata hash mismatch: {arm_id} {item_key}")
        fit_json = read_json(fit)
        if fit_json.get("validation_access") is not False or fit_json.get("test_access") is not False:
            raise ContractError(f"Learned policy training fit has invalid access flags: {arm_id}")
        out[arm_id] = {
            "policy_path": rel(path),
            "policy_file_sha256": cand["policy_file_sha256"],
            "policy_canonical_sha256": cand["policy_canonical_sha256"],
            "classification": cand.get("classification"),
            "policy": policy,
        }
    if len(out) != 3:
        raise ContractError(f"Expected three learned policies, found {len(out)}")
    return out


def required_terminal_sources(gate: Mapping[str, Any], model_records: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    required = sorted({arm["terminal_source"] for arm in gate["unique_rollout_arms"]})
    out: Dict[str, Dict[str, Any]] = {}
    for source in required:
        rec = model_records.get(source)
        if rec is None:
            raise ContractError(f"No frozen model record for terminal source {source}")
        out[source] = verify_model_record(rec)
    return out


def verify_bank_metadata_without_opening(gate: Mapping[str, Any]) -> Dict[str, Any]:
    """Stat validation/test banks without hashing or reading content."""
    result: Dict[str, Any] = {}
    for split_key, suffix in (("validation64", "vehicle_validation_bank.json"), ("sealed_test128", "vehicle_test_bank.json")):
        split = gate["splits"][split_key]
        bank_file = split["bank_file"]
        path = ROOT / bank_file["path"]
        if not path.exists():
            raise ContractError(f"Bank file missing: {bank_file['path']}")
        st = path.stat()
        if st.st_size != bank_file.get("size_bytes") or st.st_mtime_ns != bank_file.get("mtime_ns"):
            raise ContractError(f"Bank stat changed for {split_key}; do not open until audited: {bank_file['path']}")
        if not bank_file["path"].endswith(suffix):
            raise ContractError(f"Unexpected bank suffix for {split_key}: {bank_file['path']}")
        result[split_key] = {
            "path": bank_file["path"],
            "size_bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "content_opened": False,
            "sha256_computed_now": False,
            "recorded_sha256_from_gate_metadata": split.get("recorded_sha256_from_banks_completed"),
        }
    return result


def verification_bundle(gate: Mapping[str, Any]) -> Dict[str, Any]:
    source_verified = verify_gate_sources(gate)
    policies = verify_policies(gate)
    model_records = collect_model_records(gate)
    terminal_sources = required_terminal_sources(gate, model_records)
    bank_stats = verify_bank_metadata_without_opening(gate)
    return {
        "gate": {"path": rel(GATE_JSON), "sha256": sha256(GATE_JSON), "completed": rel(GATE_COMPLETED)},
        "runner": {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "source_hashes_verified_count": len(source_verified),
        "learned_policies_verified": {k: {kk: vv for kk, vv in v.items() if kk != "policy"} for k, v in policies.items()},
        "terminal_sources_verified": terminal_sources,
        "bank_metadata_no_content_opened": bank_stats,
    }


def read_backup_proof(path: Path, runner_sha: str, gate_sha: str) -> Dict[str, Any]:
    proof = read_json(path)
    if proof.get("backup_verified") is not True:
        raise ContractError("Backup proof does not record backup_verified=true")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("Backup proof does not record remaining_changed_files=0")
    if proof.get("runner_sha256") != runner_sha:
        raise ContractError("Backup proof runner sha does not match current runner")
    if proof.get("gate_sha256") != gate_sha:
        raise ContractError("Backup proof gate sha does not match current gate")
    if not proof.get("asset_sha256") and not proof.get("packages_this_run"):
        raise ContractError("Backup proof lacks a verified asset/package SHA record")
    if not proof.get("commit"):
        raise ContractError("Backup proof lacks commit")
    return proof


def load_policy_for_arm(arm: Mapping[str, Any], policies: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    if arm["controller_h"] == "policy_selected_each_step":
        policy = copy.deepcopy(policies[arm["arm_id"]]["policy"])
        if policy.get("task") != TASK:
            raise ContractError(f"Learned arm task mismatch: {arm['arm_id']}")
        return policy
    h = int(arm["controller_h"])
    policy = constant(TASK, h)
    return policy


def load_terminal(source_rel: str, terminal_verification: Mapping[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    folder = ROOT / source_rel
    _, SAC, _ = imports()
    manifest = read_json(folder / "manifest.json")
    done = read_json(folder / "completed.json")
    if manifest.get("task") != TASK or done.get("status") != "complete" or done.get("steps") != 15000:
        raise ContractError(f"Terminal source is not a complete vehicle 15k model: {source_rel}")
    model_zip_sha = sha256(folder / "model.zip")
    expected_zip_sha = terminal_verification[source_rel]["model.zip"]["sha256"]
    if model_zip_sha != expected_zip_sha:
        raise ContractError(f"Terminal model zip changed before load: {source_rel}")
    model = SAC.load(str(folder / "model.zip"))
    try:
        wh = weights_hash(model)
        if wh != done["final_hash"]:
            raise ContractError(f"Loaded terminal weights hash mismatch: {source_rel}")
        terminal = model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally:
        model.sess.close()
    receipt = {
        "source": source_rel,
        "model_zip_sha256": model_zip_sha,
        "manifest_sha256": sha256(folder / "manifest.json"),
        "completed_sha256": sha256(folder / "completed.json"),
        "weights_hash": done["final_hash"],
        "manifest_summary": {
            "task": manifest.get("task"),
            "seed": manifest.get("seed"),
            "fixed_horizon": manifest.get("fixed_horizon"),
            "steps": manifest.get("steps"),
        },
    }
    return terminal[0], terminal[1], receipt


def load_validation_bank(gate: Mapping[str, Any]) -> Dict[str, Any]:
    bank = bank_name(TASK, "validation")
    expected_path = ROOT / gate["splits"]["validation64"]["bank_file"]["path"]
    if bank.resolve() != expected_path.resolve():
        raise ContractError(f"Validation bank path mismatch: {rel(bank)} vs {rel(expected_path)}")
    digest = sha256(bank)
    expected = gate["splits"]["validation64"]["recorded_sha256_from_banks_completed"]
    if digest != expected:
        raise ContractError("Validation bank content hash does not match frozen banks/completed metadata")
    data = read_json(bank)
    if data.get("task") != TASK or data.get("split") != "validation" or len(data.get("cases", [])) != VALIDATION_CASES:
        raise ContractError("Validation bank has unexpected task/split/case count")
    return {"path": rel(bank), "sha256": digest, "cases": data["cases"], "case_count": len(data["cases"])}


def summarize_episode(
    task: str,
    trace: List[Dict[str, Any]],
    reset: Dict[str, Any],
    construction_s: float,
    terminal_load_s_reference: float,
    episode_wall_s: float,
    schedule_row: Mapping[str, Any],
    arm: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    metric = case_metrics(task, trace)
    decision = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller = [float(r["timing"]["controller_s"]) for r in trace]
    selection = [float(r["timing"]["selection_s"]) for r in trace]
    logging = [float(r["timing"]["logging_s"]) for r in trace]
    solver_times: List[float] = []
    for row in trace:
        for attempt in row.get("recovery", {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    horizons: Dict[str, int] = {}
    for row in trace:
        h = str(row.get("horizon"))
        horizons[h] = horizons.get(h, 0) + 1
    out = dict(metric)
    out.update({
        "execution_index": int(schedule_row["execution_index"]),
        "rollout_index": int(schedule_row["rollout_index"]),
        "arm_index": int(schedule_row["arm_index"]),
        "rollout_key": schedule_row["rollout_key"],
        "primary_arm_id": schedule_row["primary_arm_id"],
        "logical_aliases": list(schedule_row["logical_aliases"]),
        "seed": int(arm["seed"]),
        "case": int(schedule_row["case_index"]),
        "validation_case_index": int(schedule_row["case_index"]),
        "family": arm["family"],
        "controller_h": arm["controller_h"],
        "terminal_source": arm["terminal_source"],
        "policy_key": policy_key(policy),
        "episode_failure": not bool(metric.get("success")),
        "construction_s": float(construction_s),
        "terminal_load_s_reference": float(terminal_load_s_reference),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision),
        "decision_gross_timing_s": values_summary(decision_gross),
        "controller_timing_s_logging_deducted": values_summary(controller),
        "selection_timing_s": values_summary(selection),
        "logging_timing_s": values_summary(logging),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision) > 0.1)),
        "horizon_counts": horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
    })
    return out


def run_episode(
    out_dir: Path,
    schedule_row: Mapping[str, Any],
    arm: Mapping[str, Any],
    policy: Mapping[str, Any],
    case: Mapping[str, Any],
    terminal: Tuple[Any, Any],
    terminal_load_s_reference: float,
) -> Dict[str, Any]:
    ep_dir = out_dir / "episodes" / (
        "exec%04d_%s_case%02d" % (
            int(schedule_row["execution_index"]),
            safe_name(str(schedule_row["rollout_key"])),
            int(schedule_row["case_index"]),
        )
    )
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = make_env(TASK, int(arm["seed"]), aligned=True, scaled_obs=True)
    counts = meter(env, ep_dir)
    env.set_value_function_weights_and_biases(*terminal)
    construction_s = time.perf_counter() - construct_start

    controller = env.control_system.controller
    original = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    reset_record: Dict[str, Any]
    with LoggingTimer(ep_dir) as logging:
        recovery = recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            if not (0.0 <= logged < gross):
                raise ContractError(f"Invalid logging subtraction bounds: logged={logged}, gross={gross}")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=int(schedule_row["case_index"]), step=-1)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_gross_s = time.perf_counter() - reset_start
        if obs is None:
            raise ContractError("Environment reset returned None")
        if len(measured) != 1:
            raise ContractError("Expected exactly one controller call during reset warmup")
        reset_record = {"reset_gross_s": float(reset_gross_s), **measured.pop()}
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        previous_initial = False
        previous_final = False
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                recovery["step"] = t
                select_start = time.perf_counter()
                if policy.get("kind") in ("tree", "combined"):
                    ctx = context(env, TASK)
                    ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                    selected_h, decision = choose(policy, ctx)
                else:
                    selected_h, decision = choose(policy, {"state": env.control_system.current_state})
                    ctx = context(env, TASK)
                    ctx.update(previous_initial_failure=previous_initial, previous_final_failure=previous_final)
                selection_s = time.perf_counter() - select_start
                _, terminated, row = observed_step(env, TASK, selected_h, dict(case), t)
                if len(measured) != 1:
                    raise ContractError("Expected exactly one measured controller call for scored step")
                timing = measured.pop()
                timing.update({
                    "selection_s": float(selection_s),
                    "decision_s": float(selection_s + timing["controller_s"]),
                    "decision_gross_s": float(selection_s + timing["controller_gross_s"]),
                })
                row.update({
                    "policy_context": ctx,
                    "tree_features": features(TASK, ctx),
                    "decision": decision,
                    "recovery": recovery["events"][-1],
                    "timing": timing,
                    "validation_execution_index": int(schedule_row["execution_index"]),
                    "validation_case_index": int(schedule_row["case_index"]),
                    "rollout_key": schedule_row["rollout_key"],
                })
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                previous_initial = not bool(row["recovery"]["attempts"][0]["success"])
                previous_final = not bool(row["solver_success"])
                if terminated:
                    break
        if not trace or not trace[-1].get("termination"):
            raise ContractError("Episode did not terminate within max steps")
        audit_trace(TASK, dict(case), trace)
        if logging.operations != 1 + 3 * recovery["counts"]["solve_completed"]:
            raise ContractError(f"Unexpected logging operation count: {logging.operations}, {recovery['counts']}")
        write_json(ep_dir / "trace.json", trace)
        episode_wall_s = time.perf_counter() - episode_start
        summary = summarize_episode(TASK, trace, reset_record, construction_s, terminal_load_s_reference, episode_wall_s,
                                    schedule_row, arm, policy)
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


def aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {"episodes": 0, "steps": 0}
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
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "performance_cost_sum": float(math.fsum(float(e.get("performance_cost", 0.0)) for e in episodes)),
        "constraint_cost_sum": float(math.fsum(float(e.get("constraint_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "h_penalty_sum": float(math.fsum(float(e.get("h_penalty", 0.0)) for e in episodes)),
        "decision_total_s": float(math.fsum(float(e["decision_timing_s"]["sum"]) for e in episodes)),
        "decision_gross_total_s": float(math.fsum(float(e["decision_gross_timing_s"]["sum"]) for e in episodes)),
        "logging_total_s": float(math.fsum(float(e["logging_timing_s"]["sum"]) for e in episodes)),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
    }
    horizons: Dict[str, int] = {}
    per_episode_decisions: List[float] = []
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[h] = horizons.get(h, 0) + int(n)
        if e["decision_timing_s"].get("mean") is not None:
            per_episode_decisions.append(float(e["decision_timing_s"]["mean"]))
    out.update({
        "total_cost_mean_episode": out["total_cost_sum"] / out["episodes"],
        "physical_constraint_cost_mean_episode": out["physical_constraint_cost_sum"] / out["episodes"],
        "decision_mean_s_per_step": out["decision_total_s"] / out["steps"] if out["steps"] else None,
        "horizon_counts": horizons,
        "unique_horizons": sorted(int(h) for h in horizons),
        "episode_mean_decision_s_distribution": values_summary(per_episode_decisions),
    })
    return out


def write_markdown(path: Path, raw: Mapping[str, Any], dry_run: bool) -> None:
    lines = [
        "# Vehicle validation64 shard runner " + ("dry-run" if dry_run else f"shard {raw.get('shard_index')}"),
        "",
        f"Created UTC: {raw['created_utc']}",
        "",
        f"Validation accessed: {raw['validation_accessed']}",
        f"Validation bank content opened: {raw['validation64_bank_content_opened']}",
        f"Sealed test accessed: {raw['test_accessed']}",
        f"Formal scientific evidence created: {raw['formal_scientific_evidence_created']}",
        "",
    ]
    if dry_run:
        lines.extend([
            "Dry-run verified the frozen gate, source/model/policy hashes, bank metadata by stat only, schedule dimensions, and backup-proof requirement. It did not open validation or test bank content and ran no simulations.",
            "",
            f"- planned shards: {raw['gate_schedule']['expected_shards']}",
            f"- shard max episodes: {raw['gate_schedule']['shard_max_episodes']}",
            f"- formal validation blocked until backup proof: {raw['formal_validation_blocked_until_backup_proof']}",
        ])
    else:
        budget = raw["budget_actual"]
        lines.extend([
            f"Shard index: {raw['shard_index']}",
            f"Episodes completed: {budget['episodes']} / declared {raw['budget_declared']['episodes_exact']}",
            f"Control steps: {budget['control_steps']} / upper bound {raw['budget_declared']['control_step_upper_bound']}",
            "",
            "## Aggregate by rollout key",
            "",
            "| rollout key | episodes | steps | success | constraints | init fail steps | final fail steps | total cost | physical+constraint cost | decision mean s/step | horizons |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for key in sorted(raw["aggregates_by_rollout_key"]):
            agg = raw["aggregates_by_rollout_key"][key]
            lines.append(
                f"| `{key}` | {agg['episodes']} | {agg['steps']} | {agg['success_count']} | {agg['constraint_count']} | "
                f"{agg['initial_failed_steps']} | {agg['solver_failure_steps']} | {agg['total_cost_sum']:.6g} | "
                f"{agg['physical_constraint_cost_sum']:.6g} | {agg['decision_mean_s_per_step']:.6g} | {agg['horizon_counts']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any], dry_run: bool) -> None:
    if dry_run:
        marker = MARKER_DRY
        text = (
            f"\n<!-- {marker} -->\n"
            "## 2026-09-26 vehicle validation64 shard runner dry-run\n\n"
            f"UTC: {raw['created_utc']}. New runner `{rel(Path(__file__).resolve())}` dry-run completed with "
            "validation_accessed=false, test_accessed=false, simulations=0. It verified the frozen gate, source/model/policy hashes, "
            "bank stat metadata without opening validation/test content, and formal-run backup-proof requirements. "
            f"Artifacts: `{raw['artifacts']['dry_run_json']}`, `{raw['artifacts']['summary_md']}`. "
            "Runner and dry-run outputs now require external backup before any validation64 content access.\n"
        )
        names = ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md")
    else:
        marker = f"{MARKER_SHARD}-shard{int(raw['shard_index']):02d}"
        text = (
            f"\n<!-- {marker} -->\n"
            f"## 2026-09-26 vehicle validation64 shard {int(raw['shard_index']):02d}\n\n"
            f"UTC: {raw['created_utc']}. Formal vehicle validation shard completed with validation_accessed=true, "
            f"test_accessed=false, episodes={raw['budget_actual']['episodes']}, control_steps={raw['budget_actual']['control_steps']}. "
            f"Artifacts: `{raw['artifacts']['raw_json']}`, `{raw['artifacts']['summary_md']}`. "
            "This is shard-level validation evidence only; final sealed test remains unauthorized.\n"
        )
        names = ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md")
    for name in names:
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if marker not in old:
            path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def dry_run() -> int:
    assert_fresh_output_dir(DRY_DIR)
    DRY_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(DRY_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "purpose": "metadata-only runner preflight",
    })
    gate = load_gate()
    verification = verification_bundle(gate)
    schedule = gate["validation_schedule"]
    shards = schedule["shards"]
    if len(shards) != 12 or shards[0]["episodes"] != SHARD_MAX_EPISODES:
        raise ContractError("Unexpected shard inventory")
    for row in schedule["episodes"][:20]:
        if "case_index" not in row or "rollout_key" not in row or "execution_index" not in row:
            raise ContractError("Malformed validation schedule row")
        if not (0 <= int(row["case_index"]) < VALIDATION_CASES):
            raise ContractError("Schedule case index out of range")
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_latency_tree_vehicle_validation64_shard_runner_dry_run_not_original_SAC",
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "gate_verification": verification,
        "gate_schedule": {
            "episode_count": schedule["episode_count"],
            "logical_arm_count": schedule["logical_arm_count"],
            "unique_rollout_arm_count": schedule["unique_rollout_arm_count"],
            "expected_shards": schedule["expected_shards"],
            "shard_max_episodes": schedule["shard_max_episodes"],
            "first_shard": shards[0],
        },
        "budget_declared": {
            "episodes_exact": 0,
            "control_step_upper_bound": 0,
            "validation_bank_cases_opened": 0,
            "test_bank_cases_opened": 0,
            "new_gradient_steps": 0,
        },
        "budget_actual": {
            "episodes": 0,
            "control_steps": 0,
            "validation_bank_cases_opened": 0,
            "test_bank_cases_opened": 0,
            "new_gradient_steps": 0,
        },
        "formal_validation_blocked_until_backup_proof": True,
        "backup_proof_contract": {
            "required_for_shard_run": True,
            "required_fields": ["backup_verified", "remaining_changed_files", "commit", "asset_sha256 or packages_this_run", "runner_sha256", "gate_sha256"],
            "current_runner_sha256": verification["runner"]["sha256"],
            "current_gate_sha256": verification["gate"]["sha256"],
        },
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")},
        },
        "interpretation_limits": [
            "Dry-run only; no validation outcomes and no simulations.",
            "Runner is a post-gate implementation and must be externally backed up before validation bank content access.",
            "Sealed test remains unauthorized and is not supported by this runner.",
        ],
    }
    raw_path = DRY_DIR / "dry_run.json"
    summary_path = DRY_DIR / "summary.md"
    write_json(raw_path, raw)
    raw["artifacts"] = {"dry_run_json": rel(raw_path), "summary_md": rel(summary_path), "completed_json": rel(DRY_DIR / "completed.json")}
    write_json(raw_path, raw)
    write_markdown(summary_path, raw, dry_run=True)
    append_docs(raw, dry_run=True)
    files = [p for p in DRY_DIR.rglob("*") if p.is_file() and p.name != "completed.json"]
    write_json(DRY_DIR / "completed.json", {
        "passed": True,
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    print(json.dumps({
        "dry_run": rel(raw_path),
        "summary": rel(summary_path),
        "completed": rel(DRY_DIR / "completed.json"),
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "formal_validation_blocked_until_backup_proof": True,
        "runner_sha256": verification["runner"]["sha256"],
        "gate_sha256": verification["gate"]["sha256"],
    }, sort_keys=True), flush=True)
    return 0


def shard_rows(gate: Mapping[str, Any], shard_index: int) -> List[Dict[str, Any]]:
    shards = gate["validation_schedule"]["shards"]
    if not (0 <= shard_index < len(shards)):
        raise ContractError(f"Shard index out of range: {shard_index}")
    shard = shards[shard_index]
    rows = [row for row in gate["validation_schedule"]["episodes"]
            if shard["start_execution_index"] <= int(row["execution_index"]) <= shard["end_execution_index_inclusive"]]
    rows.sort(key=lambda row: int(row["execution_index"]))
    if len(rows) != int(shard["episodes"]):
        raise ContractError(f"Shard row count mismatch for shard {shard_index}")
    if rows[0]["execution_index"] != shard["start_execution_index"] or rows[-1]["execution_index"] != shard["end_execution_index_inclusive"]:
        raise ContractError("Shard row execution bounds mismatch")
    return rows


def formal_shard(shard_index: int, backup_proof_path: Path, accept_validation_access: bool) -> int:
    if not accept_validation_access:
        raise ContractError("Formal shard requires --i-accept-validation-access")
    out_dir = FORMAL_ROOT / ("shard%02d" % shard_index)
    assert_fresh_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out_dir / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "shard_index": shard_index,
        "validation_accessed": True,
        "validation64_bank_content_opened": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "method": "IMPROVED_latency_tree_vehicle_validation64_formal_shard_not_original_SAC",
    })
    gate = load_gate()
    verification = verification_bundle(gate)
    backup_proof = read_backup_proof(root_path(backup_proof_path), verification["runner"]["sha256"], verification["gate"]["sha256"])
    rows = shard_rows(gate, shard_index)
    if shard_index == 0 and len(rows) != SHARD_MAX_EPISODES:
        raise ContractError("Shard0 must contain exactly 224 episodes")
    bank = load_validation_bank(gate)
    policies = verify_policies(gate)
    model_records = collect_model_records(gate)
    terminal_verification = required_terminal_sources(gate, model_records)
    arms_by_key = {arm["rollout_key"]: arm for arm in gate["unique_rollout_arms"]}
    write_json(out_dir / "schedule.json", {
        "shard_index": shard_index,
        "rows": rows,
        "unique_rollout_arm_count_in_shard": len({r["rollout_key"] for r in rows}),
        "validation_accessed": True,
        "test_accessed": False,
    })

    terminal_cache: Dict[str, Tuple[Tuple[Any, Any], float, Dict[str, Any]]] = {}
    terminal_receipts: Dict[str, Any] = {}
    episodes: List[Dict[str, Any]] = []
    for row in rows:
        arm = arms_by_key[row["rollout_key"]]
        source = arm["terminal_source"]
        if source not in terminal_cache:
            load_start = time.perf_counter()
            w, b, receipt = load_terminal(source, terminal_verification)
            terminal_cache[source] = ((w, b), time.perf_counter() - load_start, receipt)
            terminal_receipts[source] = receipt
            write_json(out_dir / "terminal_sources_progress.json", terminal_receipts)
        terminal, terminal_load_s, _ = terminal_cache[source]
        policy = load_policy_for_arm(arm, policies)
        case = bank["cases"][int(row["case_index"])]
        summary = run_episode(out_dir, row, arm, policy, case, terminal, terminal_load_s)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "shard_index": shard_index,
            "episodes_done": len(episodes),
            "episodes_expected": len(rows),
            "control_steps_done": int(sum(e["steps"] for e in episodes)),
            "last_episode": {
                k: summary[k]
                for k in ("execution_index", "rollout_key", "case", "steps", "success", "termination")
            },
            "validation_accessed": True,
            "test_accessed": False,
        }
        write_json(out_dir / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    control_steps = int(sum(e["steps"] for e in episodes))
    upper = len(rows) * MAX_STEPS
    if control_steps > upper:
        raise ContractError("Control-step upper bound exceeded")
    aggregates_by_rollout: Dict[str, Any] = {}
    for key in sorted({e["rollout_key"] for e in episodes}):
        aggregates_by_rollout[key] = aggregate_episodes([e for e in episodes if e["rollout_key"] == key])
    aggregates_by_family: Dict[str, Any] = {}
    for family in sorted({e["family"] for e in episodes}):
        aggregates_by_family[family] = aggregate_episodes([e for e in episodes if e["family"] == family])
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "IMPROVED_latency_tree_vehicle_validation64_formal_shard_not_original_SAC",
        "shard_index": shard_index,
        "validation_accessed": True,
        "validation64_bank_content_opened": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "final_test_authorization_requested": False,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": True,
        "gate_verification": verification,
        "backup_proof": backup_proof,
        "validation_bank": {"path": bank["path"], "sha256": bank["sha256"], "case_count": bank["case_count"]},
        "budget_declared": {
            "episodes_exact": len(rows),
            "control_step_upper_bound": upper,
            "validation_cases_available": VALIDATION_CASES,
            "test_episodes": 0,
            "new_gradient_steps": 0,
        },
        "budget_actual": {
            "episodes": len(episodes),
            "control_steps": control_steps,
            "environment_constructions": len(episodes),
            "resets": int(sum(e["resets_metered"] for e in episodes)),
            "validation_bank_cases_opened": VALIDATION_CASES,
            "test_bank_cases_opened": 0,
            "new_gradient_steps": 0,
        },
        "terminal_sources": terminal_receipts,
        "schedule_rows": rows,
        "episodes": episodes,
        "aggregates_by_rollout_key": aggregates_by_rollout,
        "aggregates_by_family": aggregates_by_family,
        "overall_aggregate": aggregate_episodes(episodes),
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")},
        },
        "interpretation_limits": [
            "Shard-level validation evidence only; model selection and final claims require all preregistered validation shards/audits as applicable.",
            "Vehicle-only IMPROVED latency-tree evidence, not ORIGINAL SAC reproduction and not whole two-task evidence.",
            "Sealed final test remains unauthorized and was not opened.",
            "Timing observations are actual same-host measurements for this shard, but formal timing-route claims require the preregistered replay block after fixed-H nominations.",
        ],
    }
    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    write_json(raw_path, raw)
    raw["artifacts"] = {"raw_json": rel(raw_path), "summary_md": rel(summary_path), "completed_json": rel(out_dir / "completed.json")}
    write_json(raw_path, raw)
    write_markdown(summary_path, raw, dry_run=False)
    append_docs(raw, dry_run=False)
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"]
    write_json(out_dir / "completed.json", {
        "passed": True,
        "validation_accessed": True,
        "validation64_bank_content_opened": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "formal_scientific_evidence_created": True,
        "new_gradient_steps": 0,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    print(json.dumps({
        "completed": rel(out_dir / "completed.json"),
        "raw": rel(raw_path),
        "summary": rel(summary_path),
        "shard_index": shard_index,
        "validation_accessed": True,
        "test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
    }, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Verify runner/gate preconditions without opening validation/test content")
    mode.add_argument("--shard", type=int, help="Run exactly one frozen validation shard; requires backup proof")
    parser.add_argument("--backup-proof", type=Path, help="Repository-relative JSON proof of external backup after runner/gate hashes")
    parser.add_argument("--i-accept-validation-access", action="store_true", help="Required for --shard; records intentional validation access")
    args = parser.parse_args()
    if args.dry_run:
        if args.backup_proof or args.i_accept_validation_access:
            raise ContractError("Dry-run must not pass validation access/proof flags")
        return dry_run()
    if args.backup_proof is None:
        raise ContractError("Formal shard requires --backup-proof")
    return formal_shard(int(args.shard), args.backup_proof, bool(args.i_accept_validation_access))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc), "validation_accessed": False, "test_accessed": False}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
