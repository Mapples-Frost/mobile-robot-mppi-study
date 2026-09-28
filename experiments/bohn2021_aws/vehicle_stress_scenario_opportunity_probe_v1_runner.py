#!/usr/bin/env python3
"""Vehicle stress-scenario opportunity probe v1 Stage1 runner.

Frozen protocol:
  research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.*

This is a development-only IMPROVED scenario-opportunity diagnostic.  It runs no
training/refit, opens no historical validation64 bank, and opens no sealed final
test.  Dry-run performs protocol/backup/runtime/terminal-source checks only.
Rollout, after a separate post-dry-run backup, performs the frozen Stage1
fixed-H map: 20 metadata-selected cases x H={5,10,...,50} = 200 episodes.
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

# Reuse audited environment, episode, timing and terminal-loading utilities from
# the fixed-H opportunity runner.  This v1 runner supplies its own frozen
# protocol, source-supported metadata selection and materiality gates.
import vehicle_fixed_h_opportunity_probe_v0_runner as base  # noqa:E402

TASK = "vehicle"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MAX_STEPS = 150
CANDIDATE_RESETS = 256
SELECTED_CASES = 20
GROUP_COUNTS = {
    "high_heading_long_or_medium": 10,
    "high_heading_short": 4,
    "low_heading_low_clearance": 2,
    "lower_stress_control": 4,
}
BANK_RNG = 2609289301
ORDER_SEED = 2609289401
STAMP = "20260928T2045Z"
PROTOCOL_STAMP = "20260928T2025Z"

PROTOCOL_MD = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_{PROTOCOL_STAMP}.md"
PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_{PROTOCOL_STAMP}.json"
PROTOCOL_COMPLETED = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_{PROTOCOL_STAMP}/completed.json"
PROTOCOL_RAW = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_{PROTOCOL_STAMP}/raw.json"
# Terminal-source grid is inherited from the already-verified stress-v0 protocol;
# v1 deliberately changes only metadata-selected scenarios, not terminal learning.
TERMINAL_SOURCE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
TERMINAL_SOURCE_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v0_20260928T1625Z/completed.json"

OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_{STAMP}"
DRYRUN_OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_runner_dryrun_{STAMP}"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_stress_scenario_opportunity_probe_v1_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_opportunity_probe_v1_stage1_{STAMP}.md"
DRYRUN_STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_opportunity_probe_v1_runner_dryrun_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
DRYRUN_BACKUP_REQ = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_V1_RUNNER_DRYRUN_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-scenario-opportunity-stage1-v1-runner-dryrun-{STAMP}"
MARKER_STAGE1 = f"vehicle-stress-scenario-opportunity-stage1-v1-runner-{STAMP}"

ContractError = base.ContractError

# Patch globals in the reused fixed-H helper before any helper that writes paths
# or enforces budgets is called.
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
base.STATE_PATH = STATE_PATH
base.MARKER = MARKER_STAGE1


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
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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


def verify_completed_marker(path: Path, allow_hard_pass: bool = True) -> Dict[str, Any]:
    obj = read_json(path)
    if obj.get("passed") is not True and not (allow_hard_pass and obj.get("hard_pass") is True):
        raise ContractError("completed marker did not pass: %s" % rel(path))
    for name, expected in (obj.get("hashes") or {}).items():
        p = ROOT / name
        if not p.exists():
            raise ContractError("completed marker references missing file: %s" % name)
        if sha256(p) != expected:
            raise ContractError("hash mismatch in completed marker %s entry %s" % (rel(path), name))
    return obj


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        try:
            val = float(row.get(key, float("nan")))
        except Exception:
            val = float("nan")
        if math.isfinite(val):
            vals.append(val)
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


def normalized_features(row: Mapping[str, Any], med: Mapping[str, float], scale: Mapping[str, float]) -> List[float]:
    out: List[float] = []
    for key in ("abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"):
        try:
            val = float(row.get(key, med[key]))
        except Exception:
            val = med[key]
        if not math.isfinite(val):
            val = med[key]
        out.append(float((val - med[key]) / scale[key]))
    # sign dimension is included for diversity/balance but not for stress scoring.
    theta = float(row.get("theta_r", 0.0) or 0.0)
    out.append(-1.0 if theta < 0 else 1.0)
    return out


def choose_diverse(pool: Sequence[Mapping[str, Any]], k: int, group: str, already: Iterable[int], prefer_high_score: bool = True) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used = set(int(x) for x in already)
    candidates = [copy.deepcopy(dict(r)) for r in pool if int(r["candidate_index"]) not in used]
    if len(candidates) < k:
        raise ContractError("not enough candidates for group %s: have %d need %d" % (group, len(candidates), k))

    def score(row: Mapping[str, Any]) -> Tuple[float, int, int]:
        primary = float(row.get("stress_v1_score", 0.0))
        if not prefer_high_score:
            primary = -primary
        return (primary, int(row.get("stress_flags_count", 0)), -int(row["candidate_index"]))

    while len(selected) < k:
        if not selected:
            choice = max(candidates, key=score)
        else:
            chosen_z = [np.asarray(r["normalized_features"], dtype=float) for r in selected]

            def div_key(row: Mapping[str, Any]) -> Tuple[float, float, int, int]:
                z = np.asarray(row["normalized_features"], dtype=float)
                min_d = float(min(np.linalg.norm(z - s) for s in chosen_z))
                primary = float(row.get("stress_v1_score", 0.0))
                if not prefer_high_score:
                    primary = -primary
                return (min_d, primary, int(row.get("stress_flags_count", 0)), -int(row["candidate_index"]))

            choice = max(candidates, key=div_key)
        candidates = [r for r in candidates if int(r["candidate_index"]) != int(choice["candidate_index"])]
        choice["selection_group"] = group
        selected.append(choice)
    return selected


def choose_balanced_sign(pool: Sequence[Mapping[str, Any]], k: int, group: str, already: Iterable[int], prefer_high_score: bool = True) -> List[Dict[str, Any]]:
    used = set(int(x) for x in already)
    available = [copy.deepcopy(dict(r)) for r in pool if int(r["candidate_index"]) not in used]
    neg = [r for r in available if float(r.get("theta_r", 0.0)) < 0.0]
    pos = [r for r in available if float(r.get("theta_r", 0.0)) >= 0.0]
    target_neg = k // 2
    target_pos = k - target_neg
    chosen: List[Dict[str, Any]] = []
    if len(neg) >= target_neg and len(pos) >= target_pos:
        chosen.extend(choose_diverse(neg, target_neg, group, already=used, prefer_high_score=prefer_high_score))
        used.update(int(r["candidate_index"]) for r in chosen)
        chosen.extend(choose_diverse(pos, target_pos, group, already=used, prefer_high_score=prefer_high_score))
    else:
        # Fall back deterministically while recording imbalance; the candidate IDs
        # still remain fixed before any horizon outcome.
        chosen.extend(choose_diverse(available, k, group, already=used, prefer_high_score=prefer_high_score))
        for row in chosen:
            row.setdefault("selection_fallbacks", []).append("balanced_sign_underfilled")
    return chosen


def select_case_indices_v1(metas: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(metas) != CANDIDATE_RESETS:
        raise ContractError("v1 selection expected %d candidate rows" % CANDIDATE_RESETS)
    abs_vals = finite_values(metas, "abs_theta_r")
    traj_vals = finite_values(metas, "traj_steps")
    clear_vals = finite_values(metas, "min_reference_obstacle_clearance")
    med = {
        "abs_theta_r": median_and_scale(abs_vals)[0],
        "traj_steps": median_and_scale(traj_vals)[0],
        "min_reference_obstacle_clearance": median_and_scale(clear_vals)[0],
    }
    scale = {
        "abs_theta_r": median_and_scale(abs_vals)[1],
        "traj_steps": median_and_scale(traj_vals)[1],
        "min_reference_obstacle_clearance": median_and_scale(clear_vals)[1],
    }
    thr = {
        "abs_q60": percentile(abs_vals, 60.0, med["abs_theta_r"]),
        "abs_q70": percentile(abs_vals, 70.0, med["abs_theta_r"]),
        "abs_median": med["abs_theta_r"],
        "traj_q40": percentile(traj_vals, 40.0, med["traj_steps"]),
        "traj_median": med["traj_steps"],
        "traj_q60": percentile(traj_vals, 60.0, med["traj_steps"]),
        "clear_q30": percentile(clear_vals, 30.0, med["min_reference_obstacle_clearance"]),
        "clear_q40": percentile(clear_vals, 40.0, med["min_reference_obstacle_clearance"]),
        "clear_median": med["min_reference_obstacle_clearance"],
        "medians": med,
        "scales": scale,
    }
    enriched: List[Dict[str, Any]] = []
    for m in metas:
        row = copy.deepcopy(dict(m))
        abs_theta = float(row.get("abs_theta_r", med["abs_theta_r"]))
        traj = float(row.get("traj_steps", med["traj_steps"]))
        clearance = float(row.get("min_reference_obstacle_clearance", med["min_reference_obstacle_clearance"]))
        if not math.isfinite(abs_theta):
            abs_theta = med["abs_theta_r"]
        if not math.isfinite(traj):
            traj = med["traj_steps"]
        if not math.isfinite(clearance):
            clearance = med["min_reference_obstacle_clearance"]
        flags = {
            "high_abs_theta_q70": bool(abs_theta >= thr["abs_q70"]),
            "long_or_medium_traj_ge_median": bool(traj >= thr["traj_median"]),
            "short_traj_lt_median": bool(traj < thr["traj_median"]),
            "low_clearance_q30": bool(clearance <= thr["clear_q30"]),
            "low_heading_le_median": bool(abs_theta <= thr["abs_median"]),
        }
        # High score means high heading, long transient and/or natural obstacle stress.
        stress_score = (
            2.0 * float(abs_theta >= thr["abs_q70"])
            + 1.2 * float(traj >= thr["traj_median"])
            + 1.0 * float(clearance <= thr["clear_q30"])
            + 0.25 * abs((traj - med["traj_steps"]) / scale["traj_steps"])
        )
        row.update({
            "candidate_index": int(row["candidate_index"]),
            "stress_flags_v1": flags,
            "stress_flags_count": int(sum(1 for key in ("high_abs_theta_q70", "long_or_medium_traj_ge_median", "low_clearance_q30") if flags[key])),
            "stress_v1_score": float(stress_score),
            "normalized_features": normalized_features(row, med, scale),
            "selection_group": None,
            "selection_reason": None,
            "selection_fallbacks": [],
        })
        enriched.append(row)

    def relax_pool(name: str, strict: Sequence[Mapping[str, Any]], relaxed: Sequence[Mapping[str, Any]], k: int, already: Iterable[int]) -> Tuple[List[Mapping[str, Any]], Optional[str]]:
        used = set(int(x) for x in already)
        strict_avail = [r for r in strict if int(r["candidate_index"]) not in used]
        if len(strict_avail) >= k:
            return strict_avail, None
        relaxed_avail = [r for r in relaxed if int(r["candidate_index"]) not in used]
        if len(relaxed_avail) >= k:
            return relaxed_avail, "%s_relaxed_thresholds" % name
        remaining = [r for r in enriched if int(r["candidate_index"]) not in used]
        return remaining, "%s_underfilled_used_global_remaining" % name

    used: List[int] = []
    selected: List[Dict[str, Any]] = []
    fallbacks: List[str] = []

    high_long_strict = [r for r in enriched if float(r["abs_theta_r"]) >= thr["abs_q70"] and float(r["traj_steps"]) >= thr["traj_median"]]
    high_long_relaxed = [r for r in enriched if float(r["abs_theta_r"]) >= thr["abs_q60"] and float(r["traj_steps"]) >= thr["traj_q40"]]
    pool, fb = relax_pool("high_heading_long_or_medium", high_long_strict, high_long_relaxed, GROUP_COUNTS["high_heading_long_or_medium"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_balanced_sign(pool, GROUP_COUNTS["high_heading_long_or_medium"], "high_heading_long_or_medium", used, prefer_high_score=True)
    for r in rows:
        r["selection_reason"] = "high-|theta_r| long/medium transient stratum; deterministic sign-balanced farthest-diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)
    used.extend(int(r["candidate_index"]) for r in rows)

    high_short_strict = [r for r in enriched if float(r["abs_theta_r"]) >= thr["abs_q70"] and float(r["traj_steps"]) < thr["traj_median"]]
    high_short_relaxed = [r for r in enriched if float(r["abs_theta_r"]) >= thr["abs_q60"] and float(r["traj_steps"]) < thr["traj_q60"]]
    pool, fb = relax_pool("high_heading_short", high_short_strict, high_short_relaxed, GROUP_COUNTS["high_heading_short"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_balanced_sign(pool, GROUP_COUNTS["high_heading_short"], "high_heading_short", used, prefer_high_score=True)
    for r in rows:
        r["selection_reason"] = "high-|theta_r| short transient control stratum; deterministic sign-balanced diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)
    used.extend(int(r["candidate_index"]) for r in rows)

    low_clear_strict = [r for r in enriched if float(r["abs_theta_r"]) <= thr["abs_median"] and float(r["min_reference_obstacle_clearance"]) <= thr["clear_q30"]]
    low_clear_relaxed = [r for r in enriched if float(r["abs_theta_r"]) <= thr["abs_q60"] and float(r["min_reference_obstacle_clearance"]) <= thr["clear_q40"]]
    pool, fb = relax_pool("low_heading_low_clearance", low_clear_strict, low_clear_relaxed, GROUP_COUNTS["low_heading_low_clearance"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_diverse(pool, GROUP_COUNTS["low_heading_low_clearance"], "low_heading_low_clearance", used, prefer_high_score=True)
    for r in rows:
        r["selection_reason"] = "low-heading natural-obstacle stress control; deterministic farthest-diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)
    used.extend(int(r["candidate_index"]) for r in rows)

    control_strict = [r for r in enriched if float(r["abs_theta_r"]) <= thr["abs_median"] and float(r["min_reference_obstacle_clearance"]) > thr["clear_q30"]]
    control_relaxed = [r for r in enriched if float(r["abs_theta_r"]) <= thr["abs_q60"]]
    pool, fb = relax_pool("lower_stress_control", control_strict, control_relaxed, GROUP_COUNTS["lower_stress_control"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_diverse(pool, GROUP_COUNTS["lower_stress_control"], "lower_stress_control", used, prefer_high_score=False)
    for r in rows:
        r["selection_reason"] = "lower-stress control stratum; deterministic diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)

    if len(selected) != SELECTED_CASES or len({int(r["candidate_index"]) for r in selected}) != SELECTED_CASES:
        raise ContractError("v1 metadata selection produced duplicate or wrong case count")
    selected_meta: List[Dict[str, Any]] = []
    for selected_case_index, row in enumerate(selected):
        out = copy.deepcopy(row)
        out["selected_case_index"] = int(selected_case_index)
        out["stratum"] = out.get("selection_group")
        selected_meta.append(out)
    return {
        "thresholds": thr,
        "group_counts": GROUP_COUNTS,
        "all_candidate_metadata": enriched,
        "selected_indices": [int(r["candidate_index"]) for r in selected_meta],
        "selected_metadata": selected_meta,
        "selection_rule": "stress-v1 metadata-only pre-outcome selection: 10 high-|theta_r| long/medium, 4 high-|theta_r| short, 2 low-heading low-clearance, 4 lower-stress controls; sign-balanced where applicable; deterministic diversity; no horizon outcomes used",
        "selection_fallbacks": fallbacks,
        "rng_seed": BANK_RNG,
    }


def verify_protocol_inputs() -> Dict[str, Any]:
    for p in (PROTOCOL_MD, PROTOCOL_JSON, PROTOCOL_COMPLETED, PROTOCOL_RAW, TERMINAL_SOURCE_PROTOCOL, TERMINAL_SOURCE_COMPLETED):
        if not p.exists():
            raise ContractError("required input missing: %s" % rel(p))
    done = verify_completed_marker(PROTOCOL_COMPLETED)
    raw = read_json(PROTOCOL_RAW)
    protocol = read_json(PROTOCOL_JSON)
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise ContractError("v1 protocol completed marker has invalid access flags")
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("v1 protocol raw has invalid access flags")
    if protocol.get("protocol_id") != f"vehicle_stress_scenario_opportunity_probe_v1_frozen_{PROTOCOL_STAMP}":
        raise ContractError("unexpected v1 protocol id")
    gen = protocol.get("scenario_generator_v1") or {}
    stage1 = protocol.get("stage1_fixed_H_opportunity_map") or {}
    if int(gen.get("candidate_pool_resets")) != CANDIDATE_RESETS:
        raise ContractError("v1 candidate reset count changed")
    if int(gen.get("selected_cases")) != SELECTED_CASES:
        raise ContractError("v1 selected case count changed")
    if list(stage1.get("horizons") or []) != HORIZONS:
        raise ContractError("v1 horizon grid changed")
    if int(stage1.get("episodes_exact")) != SELECTED_CASES * len(HORIZONS):
        raise ContractError("v1 Stage1 episode count changed")
    if int(stage1.get("control_step_upper_bound")) != SELECTED_CASES * len(HORIZONS) * MAX_STEPS:
        raise ContractError("v1 Stage1 control-step cap changed")
    term_protocol = read_json(TERMINAL_SOURCE_PROTOCOL)
    term_done = verify_completed_marker(TERMINAL_SOURCE_COMPLETED)
    term = term_protocol.get("terminal_grid_readiness_reused_from_v1") or {}
    if term_done.get("hard_pass") is not True:
        raise ContractError("terminal-source protocol completed marker is not hard-pass")
    if term.get("available_all_required") is not True:
        raise ContractError("terminal-source grid is not complete")
    if sorted(int(h) for h in term.get("required_horizons", [])) != HORIZONS:
        raise ContractError("terminal-source horizon grid mismatch")
    return {
        "protocol": protocol,
        "protocol_raw": raw,
        "protocol_completed": done,
        "terminal_grid_readiness": term,
        "hashes": {rel(p): sha256(p) for p in (PROTOCOL_MD, PROTOCOL_JSON, PROTOCOL_COMPLETED, PROTOCOL_RAW, TERMINAL_SOURCE_PROTOCOL, TERMINAL_SOURCE_COMPLETED)},
    }


def verify_backup_proof(path: Path, protocol: Mapping[str, Any], require_dryrun_backup: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof path does not exist: %s" % rel(path))
    proof = read_json(path)
    if not (proof.get("backup_verified") is True or proof.get("status") == "verified"):
        raise ContractError("backup proof is not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof lacks remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    packages = proof.get("packages_this_run") or []
    if not (proof.get("asset_sha256") or proof.get("release_asset_sha256") or packages):
        raise ContractError("backup proof lacks package/release SHA256")
    for pkg in packages:
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            raise ContractError("backup package entry lacks sha256/bytes/verification")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    protocol_time = parse_time(protocol.get("created_utc"))
    if proof_time is not None and protocol_time is not None and proof_time < protocol_time:
        raise ContractError("backup proof predates v1 protocol freeze")
    covers = proof.get("covers") or []
    covers_protocol = None
    if covers:
        covers_protocol = any(x in covers for x in (rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(PROTOCOL_COMPLETED.parent)))
        if not covers_protocol:
            raise ContractError("backup proof covers list does not include v1 protocol artifacts")
    dryrun_time = None
    if require_dryrun_backup:
        dry_completed = DRYRUN_OUT_DIR / "completed.json"
        if not dry_completed.exists():
            raise ContractError("Stage1 rollout requires completed dry-run marker")
        dry_done = verify_completed_marker(dry_completed)
        dryrun_time = parse_time(dry_done.get("created_utc"))
        if proof_time is not None and dryrun_time is not None and proof_time < dryrun_time:
            raise ContractError("backup proof predates v1 runner dry-run")
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


def verify_terminal_metadata(terminal_grid: Mapping[str, Any]) -> Dict[str, Any]:
    sources = terminal_grid.get("terminal_sources") or {}
    receipts: Dict[str, Any] = {}
    for h in HORIZONS:
        src = sources.get(str(h))
        if not src:
            raise ContractError("missing terminal source for H%d" % h)
        folder = ROOT / src["path"]
        model_zip = folder / "model.zip"
        manifest_path = folder / "manifest.json"
        completed_path = folder / "completed.json"
        for p in (folder, model_zip, manifest_path, completed_path):
            if not p.exists():
                raise ContractError("terminal artifact missing for H%d: %s" % (h, rel(p)))
        if sha256(model_zip) != src.get("model_zip_sha256"):
            raise ContractError("terminal model hash mismatch for H%d" % h)
        manifest = read_json(manifest_path)
        completed = read_json(completed_path)
        if manifest.get("task") != TASK or int(manifest.get("fixed_horizon")) != h:
            raise ContractError("terminal manifest mismatch for H%d" % h)
        if completed.get("status") != "complete" or int(completed.get("steps")) != 15000:
            raise ContractError("terminal completion mismatch for H%d" % h)
        receipts[str(h)] = {
            "folder": rel(folder),
            "model_zip_sha256": sha256(model_zip),
            "manifest_sha256": sha256(manifest_path),
            "completed_sha256": sha256(completed_path),
            "manifest_summary": {"task": manifest.get("task"), "seed": manifest.get("seed"), "fixed_horizon": manifest.get("fixed_horizon"), "steps": manifest.get("steps")},
            "completed_status": completed.get("status"),
        }
    return receipts


def runtime_preflight() -> Dict[str, Any]:
    return base.runtime_preflight()


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed_marker(completed)
        raise SystemExit("stress-v1 Stage1 already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("partial stress-v1 Stage1 output exists; inspect before rerun: " + ", ".join(rel(p) for p in leftovers[:20]))


def generate_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        verify_completed_marker(BANK_COMPLETED)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "vehicle_stress_scenario_opportunity_probe_v1_stage1_fresh256_select20_no_validation64_no_test":
            raise ContractError("existing v1 stress bank metadata mismatch")
        if len(bank.get("candidate_cases", [])) != CANDIDATE_RESETS or len(bank.get("selected_cases", [])) != SELECTED_CASES:
            raise ContractError("existing v1 stress bank dimensions mismatch")
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
        raise ContractError("unexpected v1 bank generation counts: %r" % counts)
    selection = select_case_indices_v1(metas)
    selected_cases = [cases[i] for i in selection["selected_indices"]]
    bank = {
        "task": TASK,
        "split": "vehicle_stress_scenario_opportunity_probe_v1_stage1_fresh256_select20_no_validation64_no_test",
        "rng": BANK_RNG,
        "candidate_case_count": CANDIDATE_RESETS,
        "selected_case_count": SELECTED_CASES,
        "candidate_cases": cases,
        "selected_cases": selected_cases,
        "selection": selection,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "development-only stress-v1 fixed-H opportunity map; candidate selection uses source-supported metadata before horizon outcomes",
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
        "group_counts": GROUP_COUNTS,
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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def eligible(row: Mapping[str, Any], strict_solver: bool = True) -> bool:
    if not bool(row.get("success")) or bool(row.get("constraint")):
        return False
    if strict_solver and (int(row.get("initial_failed_steps", 0)) > 0 or int(row.get("solver_failure_steps", 0)) > 0):
        return False
    return True


def evaluate_stage1_materiality(episodes: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any], selected_meta: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_h = analysis["by_horizon"]
    safe_h = [h for h in HORIZONS if by_h[str(h)]["success_count"] == SELECTED_CASES and by_h[str(h)]["constraint_count"] == 0]
    strict_safe_h = [h for h in safe_h if by_h[str(h)]["initial_failed_steps"] == 0 and by_h[str(h)]["solver_failure_steps"] == 0]
    ref_pool = strict_safe_h or safe_h or HORIZONS
    ref_total_h = min(ref_pool, key=lambda h: (safe_float(by_h[str(h)]["total_cost_sum"]), safe_float(by_h[str(h)]["decision_total_s"]), h))
    ref_phys_h = min(ref_pool, key=lambda h: (safe_float(by_h[str(h)]["physical_constraint_cost_sum"]), safe_float(by_h[str(h)]["decision_total_s"]), h))

    oracle_total = 0.0
    oracle_phys = 0.0
    oracle_total_h: Dict[str, int] = {}
    oracle_phys_h: Dict[str, int] = {}
    material_cases: List[Dict[str, Any]] = []
    union_nondom: set = set()
    for case_id in range(SELECTED_CASES):
        rows = [e for e in episodes if int(e["case"]) == case_id]
        strict_rows = [r for r in rows if eligible(r, strict_solver=True)]
        loose_rows = [r for r in rows if eligible(r, strict_solver=False)]
        elig_rows = strict_rows or loose_rows or list(rows)
        best_total = min(elig_rows, key=lambda r: (safe_float(r.get("total_cost")), safe_float(r["decision_timing_s"]["sum"]), int(r["horizon"])))
        best_phys = min(elig_rows, key=lambda r: (safe_float(r.get("physical_constraint_cost")), safe_float(r["decision_timing_s"]["sum"]), int(r["horizon"])))
        oracle_total += safe_float(best_total.get("total_cost"))
        oracle_phys += safe_float(best_phys.get("physical_constraint_cost"))
        oracle_total_h[str(case_id)] = int(best_total["horizon"])
        oracle_phys_h[str(case_id)] = int(best_phys["horizon"])
        case_analysis = (analysis.get("by_case") or {}).get(str(case_id), {})
        union_nondom.update(int(h) for h in case_analysis.get("nondominated_horizons_cost_vs_decision_total_s") or [])

        h15_rows = [r for r in rows if int(r["horizon"]) == 15]
        ref_rows = [r for r in rows if int(r["horizon"]) == ref_phys_h]
        case_ref = h15_rows[0] if h15_rows and eligible(h15_rows[0], strict_solver=False) else (ref_rows[0] if ref_rows else h15_rows[0])
        ref_phys = safe_float(case_ref.get("physical_constraint_cost"), 0.0)
        ref_total = safe_float(case_ref.get("total_cost"), 0.0)
        best_material: Optional[Dict[str, Any]] = None
        for cand in elig_rows:
            h = int(cand["horizon"])
            if h == 15:
                continue
            if h == 5 and (not eligible(cand, strict_solver=False)):
                continue
            phys_gain = ref_phys - safe_float(cand.get("physical_constraint_cost"), 0.0)
            total_gain = ref_total - safe_float(cand.get("total_cost"), 0.0)
            phys_pct = phys_gain / ref_phys if ref_phys > 0 else 0.0
            total_pct = total_gain / ref_total if ref_total > 0 else 0.0
            material = (phys_gain >= 3.0 or phys_pct >= 0.05 or total_gain >= 3.0 or total_pct >= 0.05)
            if material:
                entry = {
                    "case": case_id,
                    "source_candidate_index": int(selected_meta[case_id].get("candidate_index", -1)),
                    "group": selected_meta[case_id].get("selection_group"),
                    "horizon": h,
                    "reference_horizon": int(case_ref["horizon"]),
                    "physical_gain": float(phys_gain),
                    "physical_gain_pct": float(phys_pct),
                    "total_gain": float(total_gain),
                    "total_gain_pct": float(total_pct),
                    "success": bool(cand.get("success")),
                    "constraint": bool(cand.get("constraint")),
                    "initial_failed_steps": int(cand.get("initial_failed_steps", 0)),
                    "solver_failure_steps": int(cand.get("solver_failure_steps", 0)),
                }
                if best_material is None or (entry["physical_gain"], entry["total_gain"]) > (best_material["physical_gain"], best_material["total_gain"]):
                    best_material = entry
        if best_material is not None:
            material_cases.append(best_material)

    ref_total_sum = safe_float(by_h[str(ref_total_h)]["total_cost_sum"])
    ref_phys_sum = safe_float(by_h[str(ref_phys_h)]["physical_constraint_cost_sum"])
    total_gain = ref_total_sum - oracle_total
    phys_gain = ref_phys_sum - oracle_phys
    total_gain_pct = total_gain / ref_total_sum if ref_total_sum > 0 else 0.0
    phys_gain_pct = phys_gain / ref_phys_sum if ref_phys_sum > 0 else 0.0
    aggregate_guard = bool(phys_gain >= 5.0 or total_gain >= 3.0 or len(union_nondom) >= 3)
    material_case_ids = sorted({int(x["case"]) for x in material_cases})
    material_groups = sorted({str(x.get("group")) for x in material_cases})
    case_diversity = bool(len(material_case_ids) >= 3 and len(material_groups) >= 2)
    stage2_candidate = bool(aggregate_guard and case_diversity)
    return {
        "reference_pool": ref_pool,
        "strict_safe_horizons_all_cases_no_constraint_no_solver_fail": strict_safe_h,
        "safe_horizons_all_cases_no_constraint": safe_h,
        "reference_total_horizon": int(ref_total_h),
        "reference_physical_horizon": int(ref_phys_h),
        "reference_total_cost_sum": ref_total_sum,
        "reference_physical_constraint_cost_sum": ref_phys_sum,
        "oracle_total_cost_sum": float(oracle_total),
        "oracle_physical_constraint_cost_sum": float(oracle_phys),
        "oracle_total_gain_abs": float(total_gain),
        "oracle_total_gain_pct": float(total_gain_pct),
        "oracle_physical_gain_abs": float(phys_gain),
        "oracle_physical_gain_pct": float(phys_gain_pct),
        "oracle_total_h_by_case": oracle_total_h,
        "oracle_physical_h_by_case": oracle_phys_h,
        "union_case_nondominated_horizons": sorted(int(h) for h in union_nondom),
        "aggregate_guard_development_only": aggregate_guard,
        "material_case_count": len(material_case_ids),
        "material_case_ids": material_case_ids,
        "material_groups": material_groups,
        "material_cases": material_cases,
        "case_diversity_gate_development_only": case_diversity,
        "stage2_continuation_trigger_candidate_development_only": stage2_candidate,
        "interpretation": "Episode-level fixed-H diversity is only a Stage2 trigger candidate. Training/refit requires separately frozen matched-state continuation labels.",
    }


def load_terminal_grid(terminal_grid: Mapping[str, Any]) -> Tuple[Dict[int, Tuple[Any, Any]], Dict[str, Any]]:
    return base.load_terminal_grid({"terminal_grid_readiness": terminal_grid})


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
        TERMINAL_SOURCE_PROTOCOL,
        TERMINAL_SOURCE_COMPLETED,
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ] + list(extra)
    return {rel(p): sha256(p) for p in paths if p.exists()}


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if marker not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-scenario opportunity Stage1 v1 runner dry-run",
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
        f"- Selected cases: `{raw['frozen_budget']['selected_cases']}` with group counts `{raw['frozen_budget']['group_counts']}`.",
        f"- Horizons: `{raw['frozen_budget']['horizons']}`.",
        f"- Episodes/control-step cap: `{raw['frozen_budget']['episodes_exact']}` / `{raw['frozen_budget']['control_step_upper_bound']}`.",
        "",
        f"Backup request before rollout: `{raw['backup_request']}`.",
    ]
    (DRYRUN_OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "method": "vehicle_stress_scenario_opportunity_probe_v1_stage1_runner_dryrun_no_simulation",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "candidate_resets": 0,
        "rollout_episodes": 0,
        "training_gradient_steps": 0,
    })
    inputs = verify_protocol_inputs()
    backup = verify_backup_proof(args.backup_proof, inputs["protocol"], require_dryrun_backup=False)
    preflight = runtime_preflight()
    write_json(DRYRUN_OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(str(preflight.get("diagnosis", "runtime preflight failed")) + " " + str(preflight.get("exception", "")))
    terminal_meta = verify_terminal_metadata(inputs["terminal_grid_readiness"])
    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_stress_scenario_opportunity_probe_v1_stage1_runner_dryrun_no_simulation",
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
            "group_counts": GROUP_COUNTS,
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
        "reason": "backup stress-v1 Stage1 runner source and dry-run outputs before any 200-episode rollout simulation",
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
        f"# Vehicle stress-v1 Stage1 runner dry-run state ({raw['created_utc']})\n\n"
        "No-simulation runner dry-run passed. Source, dry-run outputs and this proof file require external backup before the 200-episode Stage1 fixed-H rollout.\n",
        encoding="utf-8",
    )
    block = f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-28 vehicle stress-scenario Stage1 v1 runner dry-run

UTC: {raw['created_utc']}. Wrote and dry-ran `experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_probe_v1_runner.py` with no simulations, no candidate resets, no training, no validation64-bank access and no sealed-test access. The dry-run verified the frozen stress-v1 protocol, post-protocol backup proof, legacy runtime import and H5..H50 terminal metadata inherited from the verified fixed-H grid. Stage1 rollout remains blocked until external backup covers the runner source, dry-run outputs, docs/state and backup request. Artifacts: `{rel(DRYRUN_OUT_DIR / 'summary.md')}`, `{rel(DRYRUN_OUT_DIR / 'completed.json')}`.
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
        "next_after_backup": "run stress-v1 Stage1 fixed-H map under legacy interpreter with --i-accept-stage1-stress-v1-rollout",
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
        "next": "await_verified_external_backup_before_stage1_v1_rollout",
    }, sort_keys=True), flush=True)
    return 0


def write_stage1_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = raw["created_utc"].replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_V1_ROLLOUT_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup development-only stress-v1 Stage1 fixed-H opportunity map before postdiagnostic/Stage2/retraining decisions",
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
        "# Vehicle stress-scenario opportunity probe v1 Stage1",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development-only IMPROVED stress-v1 scenario diagnostic. Fixed-H grid only; no adaptive policy, no training/refit, no historical validation64-bank access, no sealed-test access.",
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
        "| selected case | source candidate | group | stress score | theta_r | traj_steps | min reference obstacle clearance | fallbacks |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for meta in raw["bank_selection"]["selected_metadata"]:
        lines.append("| %d | %d | `%s` | %.6g | %.6g | %d | %.6g | `%s` |" % (
            int(meta["selected_case_index"]),
            int(meta["candidate_index"]),
            meta.get("selection_group"),
            float(meta.get("stress_v1_score", float("nan"))),
            float(meta.get("theta_r", float("nan"))),
            int(meta.get("traj_steps", 0)),
            float(meta.get("min_reference_obstacle_clearance", float("nan"))),
            meta.get("selection_fallbacks", []),
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
        "## Stage1 v1 gate snapshot",
        "",
        f"- Strict safe horizons: `{mat['strict_safe_horizons_all_cases_no_constraint_no_solver_fail']}`; safe no-constraint horizons: `{mat['safe_horizons_all_cases_no_constraint']}`.",
        f"- Reference total H: `{mat['reference_total_horizon']}`; oracle total gain: `{mat['oracle_total_gain_abs']:.6g}` ({mat['oracle_total_gain_pct']:.3%}).",
        f"- Reference physical H: `{mat['reference_physical_horizon']}`; oracle physical gain: `{mat['oracle_physical_gain_abs']:.6g}` ({mat['oracle_physical_gain_pct']:.3%}).",
        f"- Material cases: `{mat['material_case_ids']}` across groups `{mat['material_groups']}`.",
        f"- Aggregate guard: `{mat['aggregate_guard_development_only']}`; case-diversity gate: `{mat['case_diversity_gate_development_only']}`.",
        f"- Stage2 continuation trigger candidate: `{mat['stage2_continuation_trigger_candidate_development_only']}`.",
        "",
        "Next: inspect completed/raw evidence. If Stage1 gate passes after solver/safety/timing audit, freeze Stage2 identical-state continuation before any selector/refit. If it fails, preserve negative stress-v1 evidence and do not train on sparse labels.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_stage1_docs(raw: Mapping[str, Any]) -> None:
    mat = raw["stage1_materiality_analysis"]
    block = f"""<!-- {MARKER_STAGE1} -->
## 2026-09-28 vehicle stress-scenario opportunity Stage1 v1

UTC: {raw['created_utc']}. Development-only stress-v1 fixed-H map completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, candidate resets={raw['budget_actual']['fresh_bank_candidate_resets']}. No training, no validation64-bank access and no sealed-test access. Stage2 candidate trigger={mat['stage2_continuation_trigger_candidate_development_only']}; total oracle gain={mat['oracle_total_gain_abs']:.6g}; physical oracle gain={mat['oracle_physical_gain_abs']:.6g}; material cases={mat['material_case_ids']}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
"""
    append_docs(block, MARKER_STAGE1)


def run_stage1(args: argparse.Namespace) -> int:
    if not args.i_accept_stage1_stress_v1_rollout:
        raise ContractError("Full Stage1 v1 rollout requires --i-accept-stage1-stress-v1-rollout")
    assert_fresh_output_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_stress_scenario_opportunity_probe_v1_stage1_fixed_H_grid_development_diagnostic",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    inputs = verify_protocol_inputs()
    backup = verify_backup_proof(args.backup_proof, inputs["protocol"], require_dryrun_backup=True)
    preflight = runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(str(preflight.get("diagnosis", "runtime preflight failed")) + " " + str(preflight.get("exception", "")))
    base.v1.latency_verify()
    bank_info = generate_bank_if_needed()
    bank = read_json(BANK_PATH)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    terminals, terminal_receipts = load_terminal_grid(inputs["terminal_grid_readiness"])
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
        raise ContractError("stress-v1 Stage1 fixed-H opportunity budget violation")
    analysis = base.analyze_opportunity(episodes, selected_meta)
    materiality = evaluate_stage1_materiality(episodes, analysis, selected_meta)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_stress_scenario_opportunity_probe_v1_stage1_fixed_H_grid_development_diagnostic_not_model_selection_not_final_test",
        "classification": "development_diagnostic_IMPROVED_stress_v1_fixed_H_opportunity_map",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "split": "vehicle_stress_scenario_opportunity_probe_v1_stage1_fresh256_select20_no_validation64_no_test",
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
        f"# Vehicle stress-scenario opportunity Stage1 v1 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} fixed-H stress-v1 map episodes / {control_steps} control steps. No validation64/test/training. "
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
            "aggregate_guard_development_only": materiality["aggregate_guard_development_only"],
            "case_diversity_gate_development_only": materiality["case_diversity_gate_development_only"],
            "material_case_ids": materiality["material_case_ids"],
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
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof; rollout proof must postdate dry-run/source outputs")
    ap.add_argument("--dry-run", action="store_true", help="no-simulation import/argparse/protocol/backup/runtime/terminal-metadata smoke")
    ap.add_argument("--i-accept-stress-v1-development-diagnostic", action="store_true", help="acknowledge development-only stress-v1 diagnostic, no validation64/test access")
    ap.add_argument("--i-accept-stage1-stress-v1-rollout", action="store_true", help="explicitly allow the 200-episode Stage1 v1 fixed-H stress map after post-dryrun backup")
    args = ap.parse_args(argv)
    if not args.i_accept_stress_v1_development_diagnostic:
        raise ContractError("Explicit --i-accept-stress-v1-development-diagnostic is required")
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
