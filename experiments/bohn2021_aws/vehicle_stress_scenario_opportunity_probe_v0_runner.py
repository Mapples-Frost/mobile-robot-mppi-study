#!/usr/bin/env python3
"""Vehicle stress-scenario opportunity probe v0 Stage1 runner.

Frozen protocol:
  research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.*

This is a development-only IMPROVED scenario-opportunity diagnostic.  It does
not train/refit, does not open the historical validation64 bank, and does not
open the sealed final test.  The dry-run path performs import/argparse/protocol
and backup checks only; the rollout path performs the frozen Stage1 fixed-H
stress map: 12 metadata-selected cases x H={5,10,...,50} = 120 episodes.
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

import vehicle_fixed_h_opportunity_probe_v0_runner as base  # noqa:E402

TASK = "vehicle"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MAX_STEPS = 150
CANDIDATE_RESETS = 128
SELECTED_CASES = 12
STRESS_CASES = 8
CONTROL_CASES = 4
BANK_RNG = 2609288801
ORDER_SEED = 2609288901
STAMP = "20260928"
DRYRUN_STAMP = "20260928T1645Z"

PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
PROTOCOL_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v0_20260928T1625Z/completed.json"
PROTOCOL_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v0_20260928T1625Z/raw.json"

OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_{STAMP}"
DRYRUN_OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_runner_dryrun_{DRYRUN_STAMP}"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_stress_scenario_opportunity_probe_v0_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_opportunity_probe_v0_stage1_{STAMP}.md"
DRYRUN_STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_opportunity_probe_v0_runner_dryrun_{DRYRUN_STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
DRYRUN_BACKUP_REQ = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_RUNNER_DRYRUN_{DRYRUN_STAMP}.json"
MARKER_DRYRUN = "vehicle-stress-scenario-opportunity-stage1-runner-dryrun-20260928T1645Z"
MARKER_STAGE1 = "vehicle-stress-scenario-opportunity-stage1-runner-20260928"

ContractError = base.ContractError


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


def verify_completed_marker(path: Path) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("Completed marker did not pass: %s" % rel(path))
    for name, expected in done.get("hashes", {}).items():
        p = ROOT / name
        if not p.exists() or sha256(p) != expected:
            raise ContractError("Hash mismatch for completed marker %s entry %s" % (rel(path), name))
    return done


def patch_base_globals() -> None:
    """Point audited V0 rollout helpers at the stress-v0 Stage1 artifacts."""
    base.TASK = TASK
    base.HORIZONS = list(HORIZONS)
    base.MAX_STEPS = MAX_STEPS
    base.CANDIDATE_RESETS = CANDIDATE_RESETS
    base.SELECTED_CASES = SELECTED_CASES
    base.BANK_RNG = BANK_RNG
    base.ORDER_SEED = ORDER_SEED
    base.OUT_DIR = OUT_DIR
    base.BANK_DIR = BANK_DIR
    base.BANK_PATH = BANK_PATH
    base.BANK_COMPLETED = BANK_COMPLETED
    base.FROZEN_MD = PROTOCOL_MD
    base.FROZEN_JSON = PROTOCOL_JSON
    base.PREFLIGHT_COMPLETED = PROTOCOL_COMPLETED
    base.PREFLIGHT_RAW = PROTOCOL_RAW
    base.STATE_PATH = STATE_PATH
    base.MARKER = MARKER_STAGE1


patch_base_globals()


def finite_values(metas: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for m in metas:
        try:
            v = float(m.get(key, float("nan")))
        except Exception:
            v = float("nan")
        if math.isfinite(v):
            vals.append(v)
    return vals


def percentile(vals: Sequence[float], q: float, default: float = 0.0) -> float:
    if not vals:
        return float(default)
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def median_and_scale(vals: Sequence[float], default: float = 0.0) -> Tuple[float, float]:
    if not vals:
        return float(default), 1.0
    arr = np.asarray(vals, dtype=float)
    return float(np.median(arr)), float(max(np.std(arr), 1e-9))


def rank_fraction(vals: Sequence[float], value: float, high_is_stress: bool = True) -> float:
    if not vals or not math.isfinite(value):
        return 0.5
    arr = np.asarray(vals, dtype=float)
    if high_is_stress:
        return float(np.mean(arr <= value))
    return float(np.mean(arr >= value))


def stress_select_case_indices(metas: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Frozen metadata-only stress/control selection.

    The frozen protocol requires 8 source-supported natural-stress cases plus
    4 lower-stress controls, selected before any horizon outcome.  This concrete
    implementation uses only abs(theta_r), traj_steps and natural obstacle
    clearance from reset metadata, then applies deterministic farthest-diversity
    selection in normalized metadata space.
    """
    if len(metas) != CANDIDATE_RESETS:
        raise ContractError("Stress bank selection expected %d candidate metadata rows" % CANDIDATE_RESETS)
    abs_vals = finite_values(metas, "abs_theta_r")
    traj_vals = finite_values(metas, "traj_steps")
    clear_vals = finite_values(metas, "min_reference_obstacle_clearance")
    med_abs, scale_abs = median_and_scale(abs_vals)
    med_traj, scale_traj = median_and_scale(traj_vals)
    med_clear, scale_clear = median_and_scale(clear_vals)
    thresholds = {
        "abs_theta_high_q75": percentile(abs_vals, 75.0, med_abs),
        "traj_steps_low_q25": percentile(traj_vals, 25.0, med_traj),
        "traj_steps_high_q75": percentile(traj_vals, 75.0, med_traj),
        "clearance_low_q25": percentile(clear_vals, 25.0, med_clear),
        "abs_theta_median": med_abs,
        "traj_steps_median": med_traj,
        "clearance_median": med_clear,
        "scales": {"abs_theta_r": scale_abs, "traj_steps": scale_traj, "min_reference_obstacle_clearance": scale_clear},
    }
    enriched: List[Dict[str, Any]] = []
    for m in metas:
        row = copy.deepcopy(dict(m))
        idx = int(row["candidate_index"])
        abs_theta = float(row.get("abs_theta_r", med_abs))
        traj_steps = float(row.get("traj_steps", med_traj))
        clearance = float(row.get("min_reference_obstacle_clearance", med_clear))
        if not math.isfinite(abs_theta):
            abs_theta = med_abs
        if not math.isfinite(traj_steps):
            traj_steps = med_traj
        if not math.isfinite(clearance):
            clearance = med_clear
        high_heading = bool(abs_theta >= thresholds["abs_theta_high_q75"])
        extreme_length = bool(traj_steps <= thresholds["traj_steps_low_q25"] or traj_steps >= thresholds["traj_steps_high_q75"])
        low_clearance = bool(clearance <= thresholds["clearance_low_q25"])
        flags = {
            "high_abs_theta_q75": high_heading,
            "extreme_traj_steps_q25_q75": extreme_length,
            "low_clearance_q25": low_clearance,
        }
        flags_count = int(sum(1 for v in flags.values() if v))
        z = np.asarray([
            (abs_theta - med_abs) / scale_abs,
            (traj_steps - med_traj) / scale_traj,
            (clearance - med_clear) / scale_clear,
        ], dtype=float)
        length_stress = abs(traj_steps - med_traj) / scale_traj
        score = (
            2.0 * flags_count
            + rank_fraction(abs_vals, abs_theta, high_is_stress=True)
            + rank_fraction(clear_vals, clearance, high_is_stress=False)
            + float(min(length_stress, 4.0) / 4.0)
        )
        row.update({
            "candidate_index": idx,
            "stress_flags": flags,
            "stress_flags_count": flags_count,
            "stress_score": float(score),
            "normalized_features": [float(x) for x in z],
            "selection_group": None,
            "selection_reason": None,
        })
        enriched.append(row)

    def choose_diverse(pool: Sequence[Mapping[str, Any]], k: int, group: str, already: Iterable[int]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        already_set = set(int(x) for x in already)
        candidates = [copy.deepcopy(dict(r)) for r in pool if int(r["candidate_index"]) not in already_set]
        if len(candidates) < k:
            raise ContractError("Not enough %s candidates for stress selection" % group)
        def primary(row: Mapping[str, Any]) -> Tuple[float, int, int]:
            if group == "stress":
                return (float(row["stress_score"]), int(row["stress_flags_count"]), -int(row["candidate_index"]))
            return (-float(row["stress_score"]), -int(row["stress_flags_count"]), -int(row["candidate_index"]))
        while len(selected) < k:
            if not selected:
                choice = max(candidates, key=primary)
            else:
                selected_features = [np.asarray(r["normalized_features"], dtype=float) for r in selected]
                def div_key(row: Mapping[str, Any]) -> Tuple[float, float, int, int]:
                    z = np.asarray(row["normalized_features"], dtype=float)
                    min_d = float(min(np.linalg.norm(z - s) for s in selected_features))
                    if group == "stress":
                        return (min_d, float(row["stress_score"]), int(row["stress_flags_count"]), -int(row["candidate_index"]))
                    return (min_d, -float(row["stress_score"]), -int(row["stress_flags_count"]), -int(row["candidate_index"]))
                choice = max(candidates, key=div_key)
            candidates = [r for r in candidates if int(r["candidate_index"]) != int(choice["candidate_index"])]
            choice["selection_group"] = group
            choice["selection_reason"] = (
                "stress:>=2 metadata stress flags with deterministic farthest-diversity"
                if group == "stress"
                else "control:lower-stress metadata stratum with deterministic farthest-diversity"
            )
            selected.append(choice)
        return selected

    stress_pool = [r for r in enriched if int(r["stress_flags_count"]) >= 2]
    if len(stress_pool) < STRESS_CASES:
        top = sorted(enriched, key=lambda r: (-int(r["stress_flags_count"]), -float(r["stress_score"]), int(r["candidate_index"])))
        seen = set(int(r["candidate_index"]) for r in stress_pool)
        for row in top:
            if len(stress_pool) >= STRESS_CASES:
                break
            if int(row["candidate_index"]) not in seen:
                stress_pool.append(row)
                seen.add(int(row["candidate_index"]))
    selected_stress = choose_diverse(stress_pool, STRESS_CASES, "stress", already=[])
    selected_stress_ids = [int(r["candidate_index"]) for r in selected_stress]

    control_pool = [r for r in enriched if int(r["candidate_index"]) not in selected_stress_ids and int(r["stress_flags_count"]) == 0]
    if len(control_pool) < CONTROL_CASES:
        remaining = [r for r in enriched if int(r["candidate_index"]) not in selected_stress_ids]
        for row in sorted(remaining, key=lambda r: (int(r["stress_flags_count"]), float(r["stress_score"]), int(r["candidate_index"]))):
            if len(control_pool) >= CONTROL_CASES:
                break
            if int(row["candidate_index"]) not in [int(x["candidate_index"]) for x in control_pool]:
                control_pool.append(row)
    selected_control = choose_diverse(control_pool, CONTROL_CASES, "control", already=selected_stress_ids)

    selected_meta: List[Dict[str, Any]] = []
    for selected_case_index, row in enumerate(selected_stress + selected_control):
        out = copy.deepcopy(dict(row))
        out["selected_case_index"] = int(selected_case_index)
        selected_meta.append(out)
    selected_indices = [int(r["candidate_index"]) for r in selected_meta]
    if len(selected_indices) != SELECTED_CASES or len(set(selected_indices)) != SELECTED_CASES:
        raise ContractError("Stress selection produced duplicate/wrong number of cases")
    return {
        "thresholds": thresholds,
        "all_candidate_metadata": enriched,
        "selected_indices": selected_indices,
        "selected_metadata": selected_meta,
        "selection_rule": "metadata-only pre-outcome natural-stress selection: 8 cases from high |theta_r| / extreme traj_steps / low natural obstacle-clearance strata using farthest-diversity, plus 4 lower-stress controls; no rollout outcomes used",
        "selection_counts": {"stress": STRESS_CASES, "control": CONTROL_CASES},
        "rng_seed": BANK_RNG,
    }


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed_marker(completed)
        raise SystemExit("stress-scenario Stage1 already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("Partial stress-scenario Stage1 output exists; inspect/recover before rerun: " + ", ".join(rel(p) for p in leftovers[:20]))


def verify_protocol_inputs() -> Dict[str, Any]:
    for p in (PROTOCOL_MD, PROTOCOL_JSON, PROTOCOL_COMPLETED, PROTOCOL_RAW):
        if not p.exists():
            raise ContractError("Required stress protocol input missing: %s" % rel(p))
    done = verify_completed_marker(PROTOCOL_COMPLETED)
    raw = read_json(PROTOCOL_RAW)
    protocol = read_json(PROTOCOL_JSON)
    if done.get("hard_pass") is not True:
        raise ContractError("Stress protocol did not hard-pass")
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise ContractError("Stress protocol access flags invalid")
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("Stress protocol raw access flags invalid")
    if protocol.get("protocol_id") != "vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928":
        raise ContractError("Unexpected stress protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("historical_validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("Frozen stress protocol access rules invalid")
    gen = protocol.get("scenario_generator_v0") or {}
    stage1 = protocol.get("stage1_fixed_H_stress_map") or {}
    if int(gen.get("candidate_pool_resets")) != CANDIDATE_RESETS:
        raise ContractError("Stress protocol candidate reset count changed")
    if int(gen.get("selected_cases")) != SELECTED_CASES:
        raise ContractError("Stress protocol selected case count changed")
    if list(stage1.get("horizons") or []) != HORIZONS:
        raise ContractError("Stress protocol horizon grid changed")
    if int(stage1.get("episodes_exact")) != SELECTED_CASES * len(HORIZONS):
        raise ContractError("Stress protocol episode count changed")
    if int(stage1.get("control_step_upper_bound")) != SELECTED_CASES * len(HORIZONS) * MAX_STEPS:
        raise ContractError("Stress protocol control-step cap changed")
    term = protocol.get("terminal_grid_readiness_reused_from_v1") or {}
    if term.get("available_all_required") is not True:
        raise ContractError("Stress protocol terminal grid is not ready")
    if set(int(h) for h in term.get("required_horizons") or []) != set(HORIZONS):
        raise ContractError("Stress terminal required horizons do not match Stage1 grid")
    return {
        "protocol": protocol,
        "protocol_raw": raw,
        "protocol_completed": done,
        "hashes": {rel(p): sha256(p) for p in (PROTOCOL_MD, PROTOCOL_JSON, PROTOCOL_COMPLETED, PROTOCOL_RAW)},
    }


def verify_backup_proof(path: Path, protocol: Mapping[str, Any], require_dryrun_backup: bool = False) -> Dict[str, Any]:
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
    packages = proof.get("packages_this_run") or []
    has_asset = bool(proof.get("asset_sha256") or proof.get("release_asset_sha256") or packages)
    if not has_asset:
        raise ContractError("Backup proof lacks release/package SHA256")
    for pkg in packages:
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            raise ContractError("Backup package entry lacks sha256/bytes/verification")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    protocol_time = parse_time(protocol.get("created_utc"))
    if protocol_time is not None and proof_time is not None and proof_time < protocol_time:
        raise ContractError("Backup proof predates frozen stress protocol")
    covers = proof.get("covers") or []
    covers_protocol = any(x in covers for x in (rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(PROTOCOL_COMPLETED.parent))) if covers else None
    if covers and not covers_protocol:
        raise ContractError("Backup proof has covers list but does not cover stress protocol artifacts")
    dryrun_completed = DRYRUN_OUT_DIR / "completed.json"
    dryrun_time = None
    if require_dryrun_backup:
        if not dryrun_completed.exists():
            raise ContractError("Stage1 rollout requires completed dry-run before backup gate")
        dry_done = verify_completed_marker(dryrun_completed)
        dryrun_time = parse_time(dry_done.get("created_utc"))
        if dryrun_time is not None and proof_time is not None and proof_time < dryrun_time:
            raise ContractError("Backup proof predates Stage1 runner dry-run/source outputs")
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "time": proof_time.isoformat() if proof_time else None,
        "protocol_created_utc": protocol_time.isoformat() if protocol_time else None,
        "time_postdates_protocol": bool(proof_time is not None and protocol_time is not None and proof_time >= protocol_time),
        "dryrun_created_utc": dryrun_time.isoformat() if dryrun_time else None,
        "time_postdates_dryrun_when_required": None if not require_dryrun_backup else bool(proof_time is not None and dryrun_time is not None and proof_time >= dryrun_time),
        "covers_protocol_artifacts": covers_protocol,
        "raw_status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
        "package_count": len(packages),
    }


def verify_terminal_metadata(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    term = protocol.get("terminal_grid_readiness_reused_from_v1") or {}
    sources = term.get("terminal_sources") or {}
    receipts: Dict[str, Any] = {}
    for h in HORIZONS:
        src = sources.get(str(h))
        if not src:
            raise ContractError("Missing stress Stage1 terminal source for H%d" % h)
        folder = ROOT / src["path"]
        model_zip = folder / "model.zip"
        manifest_path = folder / "manifest.json"
        completed_path = folder / "completed.json"
        for p in (folder, model_zip, manifest_path, completed_path):
            if not p.exists():
                raise ContractError("Terminal artifact missing for H%d: %s" % (h, rel(p)))
        model_hash = sha256(model_zip)
        if model_hash != src.get("model_zip_sha256"):
            raise ContractError("Terminal model hash mismatch for H%d" % h)
        manifest = read_json(manifest_path)
        completed = read_json(completed_path)
        if manifest.get("task") != TASK or int(manifest.get("fixed_horizon")) != h:
            raise ContractError("Terminal manifest mismatch for H%d" % h)
        if completed.get("status") != "complete" or int(completed.get("steps")) != 15000:
            raise ContractError("Terminal completed marker mismatch for H%d" % h)
        receipts[str(h)] = {
            "folder": rel(folder),
            "model_zip_sha256": model_hash,
            "manifest_sha256": sha256(manifest_path),
            "completed_sha256": sha256(completed_path),
            "manifest_summary": {"task": manifest.get("task"), "seed": manifest.get("seed"), "fixed_horizon": manifest.get("fixed_horizon"), "steps": manifest.get("steps")},
            "completed_status": completed.get("status"),
        }
    return receipts


def runtime_preflight() -> Dict[str, Any]:
    return base.runtime_preflight()


def generate_stress_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        verify_completed_marker(BANK_COMPLETED)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "vehicle_stress_scenario_opportunity_probe_v0_stage1_fresh128_select12_no_validation64_no_test":
            raise ContractError("Existing stress bank metadata mismatch")
        if len(bank.get("candidate_cases", [])) != CANDIDATE_RESETS or len(bank.get("selected_cases", [])) != SELECTED_CASES:
            raise ContractError("Existing stress bank dimensions mismatch")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_cases": SELECTED_CASES}
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = base.v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = base.v1.meter(env, gen_dir)
    recovery = base.v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(BANK_RNG)
    np.random.seed(BANK_RNG)
    cases: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for cid in range(CANDIDATE_RESETS):
        recovery.update(enabled=False, events=[], case=cid, step=-1)
        env.reset()
        case = base.snapshot(env)
        cases.append(case)
        metas.append(base.case_metadata(case, cid))
    if counts["step_calls"] != 0 or counts["reset_calls"] != CANDIDATE_RESETS:
        raise ContractError("Unexpected stress bank generation counts: %r" % counts)
    selection = stress_select_case_indices(metas)
    selected_cases = [cases[i] for i in selection["selected_indices"]]
    bank = {
        "task": TASK,
        "split": "vehicle_stress_scenario_opportunity_probe_v0_stage1_fresh128_select12_no_validation64_no_test",
        "rng": BANK_RNG,
        "candidate_case_count": CANDIDATE_RESETS,
        "selected_case_count": SELECTED_CASES,
        "candidate_cases": cases,
        "selected_cases": selected_cases,
        "selection": selection,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "development-only stress fixed-H opportunity map; candidate selection uses metadata only before horizon outcomes",
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
        "stress_cases": STRESS_CASES,
        "control_cases": CONTROL_CASES,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_cases": SELECTED_CASES, "reset_calls": counts["reset_calls"]}


def randomized_schedule(selected_count: int) -> List[Dict[str, Any]]:
    base_schedule: List[Dict[str, Any]] = []
    idx = 0
    for case_id in range(selected_count):
        for h in HORIZONS:
            base_schedule.append({"selected_case_index": int(case_id), "horizon": int(h), "rollout_index": int(idx)})
            idx += 1
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    for exec_idx, base_idx in enumerate(rng.permutation(len(base_schedule)).tolist()):
        row = dict(base_schedule[int(base_idx)])
        row["execution_index"] = int(exec_idx)
        schedule.append(row)
    return schedule


def safe_numeric(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def eligible_rows(rows: Sequence[Mapping[str, Any]], strict_solver: bool = True) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for row in rows:
        if not bool(row.get("success")) or bool(row.get("constraint")):
            continue
        if strict_solver and (int(row.get("initial_failed_steps", 0)) > 0 or int(row.get("solver_failure_steps", 0)) > 0):
            continue
        out.append(row)
    return out


def evaluate_stage1_materiality(episodes: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    by_h = analysis["by_horizon"]
    safe_h = [h for h in HORIZONS if by_h[str(h)]["success_count"] == SELECTED_CASES and by_h[str(h)]["constraint_count"] == 0]
    strict_safe_h = [h for h in safe_h if by_h[str(h)]["initial_failed_steps"] == 0 and by_h[str(h)]["solver_failure_steps"] == 0]
    ref_pool = strict_safe_h or safe_h or HORIZONS
    ref_total_h = min(ref_pool, key=lambda h: (safe_numeric(by_h[str(h)]["total_cost_sum"]), safe_numeric(by_h[str(h)]["decision_total_s"]), h))
    ref_phys_h = min(ref_pool, key=lambda h: (safe_numeric(by_h[str(h)]["physical_constraint_cost_sum"]), safe_numeric(by_h[str(h)]["decision_total_s"]), h))
    oracle_total = 0.0
    oracle_phys = 0.0
    oracle_total_h: Dict[str, int] = {}
    oracle_phys_h: Dict[str, int] = {}
    oracle_any_safety_fallback = False
    for case_id in range(SELECTED_CASES):
        rows = [e for e in episodes if int(e["case"]) == case_id]
        elig = eligible_rows(rows, strict_solver=True)
        if not elig:
            elig = eligible_rows(rows, strict_solver=False)
            oracle_any_safety_fallback = True
        if not elig:
            elig = list(rows)
            oracle_any_safety_fallback = True
        best_total = min(elig, key=lambda r: (safe_numeric(r.get("total_cost")), safe_numeric(r["decision_timing_s"]["sum"]), int(r["horizon"])))
        best_phys = min(elig, key=lambda r: (safe_numeric(r.get("physical_constraint_cost")), safe_numeric(r["decision_timing_s"]["sum"]), int(r["horizon"])))
        oracle_total += safe_numeric(best_total.get("total_cost"))
        oracle_phys += safe_numeric(best_phys.get("physical_constraint_cost"))
        oracle_total_h[str(case_id)] = int(best_total["horizon"])
        oracle_phys_h[str(case_id)] = int(best_phys["horizon"])
    ref_total = safe_numeric(by_h[str(ref_total_h)]["total_cost_sum"])
    ref_phys = safe_numeric(by_h[str(ref_phys_h)]["physical_constraint_cost_sum"])
    total_gain = ref_total - oracle_total
    phys_gain = ref_phys - oracle_phys
    total_gain_pct = total_gain / ref_total if ref_total > 0 else 0.0
    phys_gain_pct = phys_gain / ref_phys if ref_phys > 0 else 0.0
    material_total = bool(total_gain >= 3.0 or total_gain_pct >= 0.03)
    material_phys = bool(phys_gain >= 5.0 or phys_gain_pct >= 0.05)
    safety_clean = bool(strict_safe_h and not oracle_any_safety_fallback)
    distinct_best_phys = list(analysis.get("distinct_case_best_physical_horizons") or [])
    distinct_fast = list(analysis.get("distinct_case_fastest_horizons") or [])
    clear_multi_h_tradeoff = bool(len(distinct_best_phys) >= 2 and len(distinct_fast) >= 2)
    trigger_stage2 = bool((material_total or material_phys or clear_multi_h_tradeoff) and safety_clean)
    return {
        "reference_pool": ref_pool,
        "strict_safe_horizons_all_cases_no_constraint_no_solver_fail": strict_safe_h,
        "safe_horizons_all_cases_no_constraint": safe_h,
        "reference_total_horizon": int(ref_total_h),
        "reference_physical_horizon": int(ref_phys_h),
        "reference_total_cost_sum": ref_total,
        "reference_physical_constraint_cost_sum": ref_phys,
        "oracle_total_cost_sum": float(oracle_total),
        "oracle_physical_constraint_cost_sum": float(oracle_phys),
        "oracle_total_gain_abs": float(total_gain),
        "oracle_total_gain_pct": float(total_gain_pct),
        "oracle_physical_gain_abs": float(phys_gain),
        "oracle_physical_gain_pct": float(phys_gain_pct),
        "oracle_total_h_by_case": oracle_total_h,
        "oracle_physical_h_by_case": oracle_phys_h,
        "oracle_any_safety_or_solver_fallback": oracle_any_safety_fallback,
        "material_total_oracle_gate_development_only": bool(material_total and safety_clean),
        "material_physical_oracle_gate_development_only": bool(material_phys and safety_clean),
        "clear_multi_H_tradeoff_gate_development_only": clear_multi_h_tradeoff,
        "stage2_continuation_trigger_candidate_development_only": trigger_stage2,
        "note": "Stage2 is only a candidate trigger. A post-run inspection must rule out solver/safety/timing artifacts before any selector/refit training.",
    }


def load_terminal_grid_for_stage1(protocol: Mapping[str, Any]) -> Tuple[Dict[int, Tuple[Any, Any]], Dict[str, Any]]:
    terminal_protocol = {"terminal_grid_readiness": protocol["terminal_grid_readiness_reused_from_v1"]}
    return base.load_terminal_grid(terminal_protocol)


def source_hashes(extra: Sequence[Path] = ()) -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(base.smoke_base.__file__).resolve(),
        Path(base.v1.__file__).resolve(),
        PROTOCOL_MD,
        PROTOCOL_JSON,
        PROTOCOL_COMPLETED,
        PROTOCOL_RAW,
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ] + list(extra)
    return {rel(p): sha256(p) for p in paths if p.exists()}


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-scenario opportunity Stage1 runner dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Verified gates",
        "",
        f"- Frozen protocol: `{raw['protocol']['json_path']}` sha256 `{raw['protocol']['json_sha256']}`.",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`, postdates protocol: `{raw['backup_proof']['time_postdates_protocol']}`.",
        f"- Runtime preflight passed: `{raw['runtime_preflight']['passed']}` executable `{raw['runtime_preflight'].get('executable')}`.",
        f"- Terminal metadata horizons verified: `{sorted(raw['terminal_metadata'].keys(), key=int)}`.",
        "",
        "## Frozen Stage1 budget",
        "",
        f"- Candidate-pool resets: `{raw['frozen_budget']['candidate_pool_resets']}`.",
        f"- Selected cases: `{raw['frozen_budget']['selected_cases']}` = 8 stress + 4 controls, metadata-only before outcomes.",
        f"- Horizons: `{raw['frozen_budget']['horizons']}`.",
        f"- Episodes/control-step cap: `{raw['frozen_budget']['episodes_exact']}` / `{raw['frozen_budget']['control_step_upper_bound']}`.",
        "",
        f"Backup request before rollout: `{raw['backup_request']}`.",
    ]
    (DRYRUN_OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if marker not in old:
                path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def run_dry_run(args: argparse.Namespace) -> int:
    if DRYRUN_OUT_DIR.exists() and (DRYRUN_OUT_DIR / "completed.json").exists():
        verify_completed_marker(DRYRUN_OUT_DIR / "completed.json")
        print(json.dumps({"dry_run_already_completed": rel(DRYRUN_OUT_DIR / "completed.json"), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    DRYRUN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(DRYRUN_OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_stress_scenario_opportunity_probe_v0_stage1_runner_dryrun_no_simulation",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
        "candidate_resets": 0,
        "rollout_episodes": 0,
    })
    inputs = verify_protocol_inputs()
    backup = verify_backup_proof(args.backup_proof, inputs["protocol"], require_dryrun_backup=False)
    preflight = runtime_preflight()
    write_json(DRYRUN_OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(preflight.get("diagnosis", "runtime preflight failed") + " " + preflight.get("exception", ""))
    terminal_meta = verify_terminal_metadata(inputs["protocol"])
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_stress_scenario_opportunity_probe_v0_stage1_runner_dryrun_no_simulation",
        "classification": "development_diagnostic_runner_smoke_no_simulation_not_model_selection_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "candidate_pool_resets": 0,
        "backup_proof": backup,
        "protocol": {"md_path": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "protocol_inputs": inputs["hashes"],
        "frozen_budget": {
            "candidate_pool_resets": CANDIDATE_RESETS,
            "selected_cases": SELECTED_CASES,
            "stress_cases": STRESS_CASES,
            "control_cases": CONTROL_CASES,
            "horizons": HORIZONS,
            "episodes_exact": SELECTED_CASES * len(HORIZONS),
            "control_step_upper_bound": SELECTED_CASES * len(HORIZONS) * MAX_STEPS,
            "order_seed": ORDER_SEED,
        },
        "terminal_metadata": terminal_meta,
        "runtime_preflight": preflight,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "source_hashes": source_hashes(extra=[args.backup_proof]),
        "interpretation_limits": ["dry-run only", "no simulations", "no candidate reset generation", "not validation/model selection", "not final test"],
    }
    write_json(DRYRUN_BACKUP_REQ, {
        "requested_utc": raw["created_utc"],
        "reason": "backup stress Stage1 runner source and dry-run outputs before any 120-episode rollout simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(Path(__file__).resolve()), rel(DRYRUN_OUT_DIR), rel(DRYRUN_STATE_PATH), rel(DRYRUN_BACKUP_REQ), rel(PROTOCOL_MD), rel(PROTOCOL_JSON), rel(args.backup_proof)],
    })
    raw["backup_request"] = rel(DRYRUN_BACKUP_REQ)
    write_json(DRYRUN_OUT_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    DRYRUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRYRUN_STATE_PATH.write_text(
        f"# Vehicle stress Stage1 runner dry-run state ({raw['created_utc']})\n\n"
        "No-simulation runner dry-run passed. Source, dry-run outputs and persisted proof now require external backup before the 120-episode Stage1 rollout.\n",
        encoding="utf-8",
    )
    block = f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-28 vehicle stress-scenario Stage1 runner dry-run

UTC: {raw['created_utc']}. Wrote/froze and dry-ran `experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_probe_v0_runner.py` with no simulations, no candidate resets, no training, no validation64-bank access and no sealed-test access. The dry-run verified the frozen stress protocol, post-protocol backup proof, legacy runtime import and H5..H50 terminal metadata. Stage1 rollout remains blocked until external backup covers the runner source, dry-run outputs, docs/state and backup request. Artifacts: `{rel(DRYRUN_OUT_DIR / 'summary.md')}`, `{rel(DRYRUN_OUT_DIR / 'completed.json')}`.
"""
    append_docs(block, MARKER_DRYRUN)
    files = [p for p in DRYRUN_OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [DRYRUN_STATE_PATH, DRYRUN_BACKUP_REQ, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON, args.backup_proof]
    write_json(DRYRUN_OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": raw["created_utc"],
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "candidate_pool_resets": 0,
        "backup_required_before_rollout": True,
        "backup_request": rel(DRYRUN_BACKUP_REQ),
        "next_after_backup": "run Stage1 stress fixed-H map under legacy interpreter with --i-accept-stage1-stress-rollout",
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(DRYRUN_OUT_DIR / "completed.json"),
        "summary": rel(DRYRUN_OUT_DIR / "summary.md"),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(DRYRUN_BACKUP_REQ),
        "next": "await_verified_external_backup_before_stage1_rollout",
    }, sort_keys=True), flush=True)
    return 0


def write_stage1_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = raw["created_utc"].replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_ROLLOUT_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup development-only stress Stage1 fixed-H opportunity map before postdiagnostic/Stage2/retraining decisions",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "candidate_pool_resets": raw["budget_actual"]["fresh_bank_candidate_resets"],
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_MD), rel(PROTOCOL_JSON)],
    })
    return rel(path)


def write_stage1_summary(raw: Mapping[str, Any]) -> None:
    opp = raw["opportunity_analysis"]
    mat = raw["stage1_materiality_analysis"]
    lines = [
        "# Vehicle stress-scenario opportunity probe v0 Stage1",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development-only IMPROVED stress-scenario diagnostic. Fixed-H grid only; no adaptive policy, no training/refit, no historical validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Candidate-bank resets: `{raw['budget_actual']['fresh_bank_candidate_resets']}` / declared `{raw['budget_declared']['fresh_bank_candidate_resets']}`.",
        f"- Rollout episodes: `{raw['budget_actual']['episodes']}` / declared `{raw['budget_declared']['rollout_episodes_exact']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Selected cases",
        "",
        "| selected case | source candidate | group | stress flags | stress score | theta_r | traj_steps | min reference obstacle clearance |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for meta in raw["bank_selection"]["selected_metadata"]:
        flags = int(meta.get("stress_flags_count", 0))
        lines.append("| %d | %d | `%s` | %d | %.6g | %.6g | %d | %.6g |" % (
            int(meta["selected_case_index"]),
            int(meta["candidate_index"]),
            meta.get("selection_group"),
            flags,
            float(meta.get("stress_score", float("nan"))),
            float(meta.get("theta_r", float("nan"))),
            int(meta.get("traj_steps", 0)),
            float(meta.get("min_reference_obstacle_clearance", float("nan"))),
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
            h,
            int(a["episodes"]),
            int(a["success_count"]),
            int(a["constraint_count"]),
            int(a["initial_failed_steps"]),
            int(a["solver_failure_steps"]),
            float(a["physical_constraint_cost_sum"]),
            float(a["total_cost_sum"]),
            float(a["decision_mean_s_per_step"]),
            float(a["decision_total_s"]),
        ))
    lines += [
        "",
        "## Stage1 materiality snapshot",
        "",
        f"- Strict safe horizons: `{mat['strict_safe_horizons_all_cases_no_constraint_no_solver_fail']}`; safe no-constraint horizons: `{mat['safe_horizons_all_cases_no_constraint']}`.",
        f"- Reference total H: `{mat['reference_total_horizon']}`; oracle total gain: `{mat['oracle_total_gain_abs']:.6g}` ({mat['oracle_total_gain_pct']:.3%}).",
        f"- Reference physical H: `{mat['reference_physical_horizon']}`; oracle physical gain: `{mat['oracle_physical_gain_abs']:.6g}` ({mat['oracle_physical_gain_pct']:.3%}).",
        f"- Material total gate: `{mat['material_total_oracle_gate_development_only']}`; material physical gate: `{mat['material_physical_oracle_gate_development_only']}`; clear multi-H tradeoff gate: `{mat['clear_multi_H_tradeoff_gate_development_only']}`.",
        f"- Stage2 continuation trigger candidate: `{mat['stage2_continuation_trigger_candidate_development_only']}` (post-run artifact inspection still required).",
        "",
        "Next: inspect completed/raw evidence. If material opportunity or a clear non-dominated multi-H tradeoff survives solver/safety/timing audit, freeze Stage2 identical-state continuation; otherwise preserve negative stress evidence and diagnose terminal/reward/modeling limits before retraining.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_stage1_docs(raw: Mapping[str, Any]) -> None:
    mat = raw["stage1_materiality_analysis"]
    block = f"""<!-- {MARKER_STAGE1} -->
## 2026-09-28 vehicle stress-scenario opportunity Stage1

UTC: {raw['created_utc']}. Development-only stress fixed-H map completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, candidate resets={raw['budget_actual']['fresh_bank_candidate_resets']}. No training, no validation64-bank access and no sealed-test access. Stage2 candidate trigger={mat['stage2_continuation_trigger_candidate_development_only']}; total oracle gain={mat['oracle_total_gain_abs']:.6g}; physical oracle gain={mat['oracle_physical_gain_abs']:.6g}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
"""
    append_docs(block, MARKER_STAGE1)


def run_stage1(args: argparse.Namespace) -> int:
    if not args.i_accept_stage1_stress_rollout:
        raise ContractError("Full Stage1 rollout requires --i-accept-stage1-stress-rollout")
    assert_fresh_output_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_stress_scenario_opportunity_probe_v0_stage1_fixed_H_grid_development_diagnostic",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    inputs = verify_protocol_inputs()
    backup = verify_backup_proof(args.backup_proof, inputs["protocol"], require_dryrun_backup=True)
    preflight = runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(preflight.get("diagnosis", "runtime preflight failed") + " " + preflight.get("exception", ""))
    base.v1.latency_verify()
    bank_info = generate_stress_bank_if_needed()
    bank = read_json(BANK_PATH)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    terminals, terminal_receipts = load_terminal_grid_for_stage1(inputs["protocol"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    schedule = randomized_schedule(len(selected_cases))
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "horizons": HORIZONS})
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        case_id = int(item["selected_case_index"])
        h = int(item["horizon"])
        summary = base.run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals[h], terminal_receipts[str(h)])
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
    declared = {"fresh_bank_candidate_resets": CANDIDATE_RESETS, "rollout_episodes_exact": SELECTED_CASES * len(HORIZONS), "control_step_upper_bound": SELECTED_CASES * len(HORIZONS) * MAX_STEPS}
    if len(episodes) != declared["rollout_episodes_exact"] or control_steps > declared["control_step_upper_bound"]:
        raise ContractError("Stress Stage1 fixed-H opportunity budget violation")
    analysis = base.analyze_opportunity(episodes, selected_meta)
    materiality = evaluate_stage1_materiality(episodes, analysis, inputs["protocol"])
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_stress_scenario_opportunity_probe_v0_stage1_fixed_H_grid_development_diagnostic_not_model_selection_not_final_test",
        "classification": "development_diagnostic_IMPROVED_stress_fixed_H_opportunity_map",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": "vehicle_stress_scenario_opportunity_probe_v0_stage1_fresh128_select12_no_validation64_no_test",
        "backup_proof": backup,
        "protocol": {"md_path": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "protocol_full": inputs["protocol"],
        "protocol_inputs": inputs["hashes"],
        "bank": bank_info,
        "bank_selection": bank["selection"],
        "source_hashes": source_hashes(extra=[args.backup_proof]),
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
        "stage1_materiality_analysis": materiality,
        "interpretation_limits": [
            "development diagnostic only",
            "fixed-H grid only",
            "not adaptive model selection",
            "not ORIGINAL SAC",
            "not final test",
            "actual timing from this AWS run only",
            "scenario bank generated from source-supported straight-line vehicle natural-stress factors",
        ],
    }
    raw["backup_request"] = write_stage1_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_stage1_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle stress-scenario opportunity Stage1 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} fixed-H stress-map episodes / {control_steps} control steps. No validation64/test/training. "
        f"Stage2 candidate trigger={materiality['stage2_continuation_trigger_candidate_development_only']}. Backup required before postdiagnostic or further simulation.\n",
        encoding="utf-8",
    )
    append_stage1_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "candidate_pool_resets": raw["budget_actual"]["fresh_bank_candidate_resets"],
        "backup_request": raw["backup_request"],
        "headline": {
            "stage2_continuation_trigger_candidate_development_only": materiality["stage2_continuation_trigger_candidate_development_only"],
            "material_total_oracle_gate_development_only": materiality["material_total_oracle_gate_development_only"],
            "material_physical_oracle_gate_development_only": materiality["material_physical_oracle_gate_development_only"],
            "clear_multi_H_tradeoff_gate_development_only": materiality["clear_multi_H_tradeoff_gate_development_only"],
            "reference_total_horizon": materiality["reference_total_horizon"],
            "reference_physical_horizon": materiality["reference_physical_horizon"],
            "oracle_total_gain_abs": materiality["oracle_total_gain_abs"],
            "oracle_physical_gain_abs": materiality["oracle_physical_gain_abs"],
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
        "stage2_continuation_trigger_candidate_development_only": materiality["stage2_continuation_trigger_candidate_development_only"],
        "backup_request": raw["backup_request"],
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof; for rollout it must postdate dry-run/source outputs")
    ap.add_argument("--dry-run", action="store_true", help="no-simulation import/argparse/protocol/backup/runtime/terminal-metadata smoke")
    ap.add_argument("--i-accept-stress-development-diagnostic", action="store_true", help="acknowledge development-only diagnostic, no validation64/test access")
    ap.add_argument("--i-accept-stage1-stress-rollout", action="store_true", help="explicitly allow the 120-episode Stage1 fixed-H stress map after post-dryrun backup")
    args = ap.parse_args(argv)
    if not args.i_accept_stress_development_diagnostic:
        raise ContractError("Explicit --i-accept-stress-development-diagnostic is required")
    if args.dry_run:
        return run_dry_run(args)
    return run_stage1(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_OUT_DIR if "--dry-run" in sys.argv else OUT_DIR
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0 if "--dry-run" in sys.argv else None,
            "new_control_steps": 0 if "--dry-run" in sys.argv else None,
            "next_recovery_hint": "Preserve this partial directory. If dry-run failed, repair only import/protocol/backup logic; if rollout failed, audit partial episodes before any rerun.",
        })
        raise
