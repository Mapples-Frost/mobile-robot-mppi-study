#!/usr/bin/env python3
"""Vehicle stress-v1c terminal-stable label-density probe v0 runner.

Frozen protocol:
  research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json

This runner is development-only IMPROVED evidence for the Bohn 2021 adaptive
MPC prediction-horizon question.  It deliberately runs no adaptive selector and
no training/refit.  Its purpose is to test whether fresh, non-mined vehicle
states have enough state-dependent horizon-choice labels when labels are based
on realised continuation costs under terminal-source-stable modes only.

Modes:
  * --dry-run: no simulation, no candidate resets, no TF import.  Verifies the
    frozen protocol and recent terminal/objective diagnostic, then writes a
    backup request before any rollout.
  * --run-smoke: after a verified external backup, generate the frozen fresh
    candidate pool and run the 6-case smoke schedule (168 episodes max).
  * --run-full: after smoke and a fresh backup, run the 20-case full schedule
    (560 episodes max).  This is available for the next bounded iteration if
    the smoke gate passes.

Access contract: no historical validation64-bank read, no current validation64
bank read, no sealed-test read/open, no training episodes, no gradient steps.
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
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

TASK = "vehicle"
STAMP = "20260929T0040Z"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json"
V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/completed.json"
V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/raw.json"
V0B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/summary.md"
V0B_BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_OBJECTIVE_ALIGNMENT_V0B_SCHEMA_REPAIR_20260929T0035Z.json"

DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_dryrun_{STAMP}"
SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_smoke_{STAMP}"
FULL_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_full_{STAMP}"
BANK_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_bank_{STAMP}"
BANK_PATH = BANK_DIR / "vehicle_stress_v1c_terminal_stable_label_density_probe_v0_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_dryrun_{STAMP}.md"
STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_smoke_{STAMP}.md"
STATE_FULL = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_full_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_SMOKE = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0_SMOKE_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0-dryrun-{STAMP}"
MARKER_SMOKE = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0-smoke-{STAMP}"
MARKER_FULL = f"vehicle-stress-v1c-terminal-stable-label-density-probe-v0-full-{STAMP}"

PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45, 50]
TERMINAL_MODES = ["zero_terminal", "h15_common_terminal"]
MAX_STEPS = 150
CANDIDATE_RESETS = 256
SELECTED_FULL = 20
SELECTED_SMOKE = 6
CANDIDATE_POOL_SEED = 2609291101
SELECTION_SEED = 2609291102
ORDER_SEED = 2609291103
MATERIAL_GAIN = 3.0
STATE_DISTANCE_TOL = 1e-5

GROUP_COUNTS_FULL = {
    "fresh_high_heading_long_or_medium": 12,
    "fresh_high_heading_short": 3,
    "fresh_low_heading_low_clearance_control": 2,
    "fresh_lower_stress_control": 3,
}
GROUP_COUNTS_SMOKE = {
    "fresh_high_heading_long_or_medium": 4,
    "fresh_high_heading_short": 1,
    "fresh_lower_stress_control": 1,
}

PREVIOUS_BANKS = [
    ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/bank/vehicle_v1_fresh_continuation_label_probe_v0_bank.json",
    ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json",
]


class ContractError(RuntimeError):
    """Protocol/access/budget violation."""


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return clean_jsonable(value.tolist())
    if hasattr(value, "item"):
        return clean_jsonable(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
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
        json.dumps(clean_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
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


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


def completed_passed(path: Path, check_hashes: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError(f"sealed-test flag not false in {rel(path)}")
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError(f"historical validation64 flag not false in {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError(f"completed marker references missing file: {name}")
            if sha256(p) != expected:
                raise ContractError(f"completed marker hash mismatch for {name}")
    return obj


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime], purpose: str) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError(f"backup proof path does not exist: {rel(path)}")
    proof = read_json(path)
    if not (proof.get("backup_verified") is True or proof.get("status") == "verified"):
        raise ContractError("backup proof is not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    packages = proof.get("packages_this_run") or []
    if not (proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256") or packages):
        raise ContractError("backup proof lacks release/package SHA")
    for pkg in packages:
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            raise ContractError("backup package entry lacks sha256/bytes/verification")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError(f"backup proof lacks parseable time for {purpose}")
        if proof_time < min_time:
            raise ContractError(f"backup proof predates required artifacts for {purpose}: {proof_time.isoformat()} < {min_time.isoformat()}")
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "time": None if proof_time is None else proof_time.isoformat(),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "packages_this_run": packages,
        "purpose": purpose,
    }


def verify_protocol_and_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (PROTOCOL_JSON, V0B_DONE, V0B_RAW, V0B_SUMMARY, V0B_BACKUP_REQUEST):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z":
        raise ContractError("unexpected v1c protocol id")
    if protocol.get("classification") != "development_IMPROVED_fresh_nonmined_terminal_stable_label_density_probe_not_validation_not_final_test":
        raise ContractError("unexpected v1c protocol classification")
    access = protocol.get("access_rules") or {}
    for key in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if access.get(key) is not False:
            raise ContractError(f"protocol access flag {key} must be false")
    if access.get("requires_verified_external_backup_after_v0b_before_any_simulation") is not True:
        raise ContractError("protocol must require verified backup before simulation")
    gen = protocol.get("scenario_generator_v1c") or {}
    if int(gen.get("fresh_candidate_pool_resets", -1)) != CANDIDATE_RESETS:
        raise ContractError("candidate pool reset budget changed")
    if int(gen.get("selected_case_count_full", -1)) != SELECTED_FULL or int(gen.get("selected_case_count_smoke", -1)) != SELECTED_SMOKE:
        raise ContractError("selected case budgets changed")
    if int(gen.get("candidate_pool_seed", -1)) != CANDIDATE_POOL_SEED or int(gen.get("selection_seed", -1)) != SELECTION_SEED or int(gen.get("order_seed", -1)) != ORDER_SEED:
        raise ContractError("v1c seeds changed")
    if dict(gen.get("selection_groups_full") or {}) != GROUP_COUNTS_FULL:
        raise ContractError("full selection groups changed")
    if dict(gen.get("selection_groups_smoke") or {}) != GROUP_COUNTS_SMOKE:
        raise ContractError("smoke selection groups changed")
    design = protocol.get("rollout_design") or {}
    if int(design.get("prefix_horizon", -1)) != PREFIX_H:
        raise ContractError("prefix horizon changed")
    if [int(x) for x in design.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("branch horizons changed")
    if [str(x) for x in design.get("primary_training_label_terminal_modes", [])] != TERMINAL_MODES:
        raise ContractError("primary terminal modes changed")
    if int(design.get("smoke_episodes_exact_upper_bound", -1)) != 168 or int(design.get("full_episodes_exact_upper_bound", -1)) != 560:
        raise ContractError("v1c episode upper bounds changed")
    if int(design.get("new_training_episodes", -1)) != 0 or int(design.get("new_gradient_steps", -1)) != 0 or int(design.get("new_refit_steps", -1)) != 0:
        raise ContractError("v1c protocol unexpectedly permits training/refit")
    label = protocol.get("primary_label_rule") or {}
    if float(label.get("material_gain_threshold_physical", -1)) != MATERIAL_GAIN:
        raise ContractError("material threshold changed")
    if float(label.get("branch_state_distance_tolerance", -1)) != STATE_DISTANCE_TOL:
        raise ContractError("state-distance tolerance changed")
    v0b_done = completed_passed(V0B_DONE, check_hashes=False)
    v0b_raw = read_json(V0B_RAW)
    headline = v0b_done.get("headline") or {}
    if v0b_done.get("hard_pass") is not True:
        raise ContractError("v0b corrected objective diagnostic was not hard-pass")
    if headline.get("train_or_refit_now") is not False:
        raise ContractError("v0b must block training/refit before fresh terminal-stable labels")
    if int(headline.get("terminal_flips", -1)) < 1:
        raise ContractError("v0b headline unexpectedly lacks terminal-flip evidence")
    if int(headline.get("min_objective_selected_non_success_rows", -1)) < 1:
        raise ContractError("v0b headline unexpectedly lacks unsafe objective-selected rows")
    return protocol, v0b_done, v0b_raw


def finite_summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    n = len(xs)
    def pct(q: float) -> float:
        if n == 1:
            return xs[0]
        idx = (n - 1) * q / 100.0
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"count": n, "min": xs[0], "median": pct(50), "mean": float(math.fsum(xs) / n), "p95": pct(95), "max": xs[-1], "sum": float(math.fsum(xs))}


def percentile(vals: Sequence[float], q: float, default: float = 0.0) -> float:
    xs = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if not xs:
        return default
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * q / 100.0
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def median(vals: Sequence[float], default: float = 0.0) -> float:
    return percentile(vals, 50.0, default)


def std(vals: Sequence[float], default: float = 1.0) -> float:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if not xs:
        return default
    m = math.fsum(xs) / len(xs)
    return max(math.sqrt(math.fsum((x - m) ** 2 for x in xs) / len(xs)), 1e-9)


def case_hash(case: Mapping[str, Any]) -> str:
    return canonical_sha(case)


def previous_selected_case_hashes() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for bank_path in PREVIOUS_BANKS:
        if not bank_path.exists():
            continue
        try:
            bank = read_json(bank_path)
        except Exception:
            continue
        for i, case in enumerate(bank.get("selected_cases") or []):
            try:
                out[case_hash(case)] = f"{rel(bank_path)}#selected_cases[{i}]"
            except Exception:
                pass
    return out


def normalized_features(row: Mapping[str, Any], med: Mapping[str, float], scale: Mapping[str, float]) -> List[float]:
    vals: List[float] = []
    for key in ("abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"):
        try:
            val = float(row.get(key, med[key]))
        except Exception:
            val = med[key]
        if not math.isfinite(val):
            val = med[key]
        vals.append(float((val - med[key]) / scale[key]))
    theta = float(row.get("theta_r", 0.0) or 0.0)
    vals.append(-1.0 if theta < 0.0 else 1.0)
    return vals


def choose_diverse(pool: Sequence[Mapping[str, Any]], k: int, group: str, used: Iterable[int], rng_priority: Mapping[int, float], prefer_high_score: bool = True) -> List[Dict[str, Any]]:
    used_set = set(int(x) for x in used)
    candidates = [copy.deepcopy(dict(r)) for r in pool if int(r["candidate_index"]) not in used_set]
    if len(candidates) < k:
        raise ContractError(f"not enough candidates for {group}: have {len(candidates)} need {k}")
    selected: List[Dict[str, Any]] = []

    def base_score(row: Mapping[str, Any]) -> Tuple[float, float, int]:
        score = float(row.get("stress_v1c_score", 0.0))
        if not prefer_high_score:
            score = -score
        return (score, -float(rng_priority[int(row["candidate_index"])]), -int(row["candidate_index"]))

    while len(selected) < k:
        if not selected:
            choice = max(candidates, key=base_score)
        else:
            chosen_z = [list(map(float, r["normalized_features"])) for r in selected]

            def diversity_score(row: Mapping[str, Any]) -> Tuple[float, float, float, int]:
                z = list(map(float, row["normalized_features"]))
                mind = min(math.sqrt(sum((a - b) ** 2 for a, b in zip(z, s))) for s in chosen_z)
                score = float(row.get("stress_v1c_score", 0.0))
                if not prefer_high_score:
                    score = -score
                return (mind, score, -float(rng_priority[int(row["candidate_index"])]), -int(row["candidate_index"]))
            choice = max(candidates, key=diversity_score)
        choice["selection_group"] = group
        selected.append(choice)
        candidates = [r for r in candidates if int(r["candidate_index"]) != int(choice["candidate_index"])]
    return selected


def choose_balanced_sign(pool: Sequence[Mapping[str, Any]], k: int, group: str, used: Iterable[int], rng_priority: Mapping[int, float], prefer_high_score: bool = True) -> List[Dict[str, Any]]:
    used_set = set(int(x) for x in used)
    avail = [copy.deepcopy(dict(r)) for r in pool if int(r["candidate_index"]) not in used_set]
    neg = [r for r in avail if float(r.get("theta_r", 0.0) or 0.0) < 0.0]
    pos = [r for r in avail if float(r.get("theta_r", 0.0) or 0.0) >= 0.0]
    n_neg = k // 2
    n_pos = k - n_neg
    chosen: List[Dict[str, Any]] = []
    if len(neg) >= n_neg and len(pos) >= n_pos:
        chosen.extend(choose_diverse(neg, n_neg, group, used_set, rng_priority, prefer_high_score=prefer_high_score))
        used_set.update(int(r["candidate_index"]) for r in chosen)
        chosen.extend(choose_diverse(pos, n_pos, group, used_set, rng_priority, prefer_high_score=prefer_high_score))
    else:
        chosen.extend(choose_diverse(avail, k, group, used_set, rng_priority, prefer_high_score=prefer_high_score))
        for r in chosen:
            r.setdefault("selection_fallbacks", []).append("balanced_sign_underfilled")
    return chosen


def select_cases_v1c(metas: Sequence[Mapping[str, Any]], previous_hashes: Mapping[str, str]) -> Dict[str, Any]:
    if len(metas) != CANDIDATE_RESETS:
        raise ContractError("unexpected candidate metadata count")
    abs_vals = [float(m.get("abs_theta_r", float("nan"))) for m in metas]
    traj_vals = [float(m.get("traj_steps", float("nan"))) for m in metas]
    clear_vals = [float(m.get("min_reference_obstacle_clearance", float("nan"))) for m in metas]
    med = {
        "abs_theta_r": median(abs_vals),
        "traj_steps": median(traj_vals),
        "min_reference_obstacle_clearance": median(clear_vals),
    }
    scale = {
        "abs_theta_r": std(abs_vals),
        "traj_steps": std(traj_vals),
        "min_reference_obstacle_clearance": std(clear_vals),
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
    rng = random.Random(SELECTION_SEED)
    rng_priority = {int(m["candidate_index"]): rng.random() for m in metas}
    enriched: List[Dict[str, Any]] = []
    excluded_exact_previous: List[Dict[str, Any]] = []
    for m in metas:
        row = copy.deepcopy(dict(m))
        row["candidate_index"] = int(row["candidate_index"])
        abs_theta = float(row.get("abs_theta_r", med["abs_theta_r"]))
        traj = float(row.get("traj_steps", med["traj_steps"]))
        clearance = float(row.get("min_reference_obstacle_clearance", med["min_reference_obstacle_clearance"]))
        if not math.isfinite(abs_theta):
            abs_theta = med["abs_theta_r"]
        if not math.isfinite(traj):
            traj = med["traj_steps"]
        if not math.isfinite(clearance):
            clearance = med["min_reference_obstacle_clearance"]
        row["normalized_features"] = normalized_features(row, med, scale)
        # Metadata-only stress score: large heading and longer transient are the
        # intended opportunity subfamily; lower clearance is retained only as a
        # natural-stress control signal because the vehicle environment already
        # exposes it in metadata.
        row["stress_v1c_score"] = float(
            2.0 * (abs_theta >= thr["abs_q70"])
            + 1.3 * (traj >= thr["traj_median"])
            + 0.6 * (clearance <= thr["clear_q30"])
            + 0.20 * abs((traj - med["traj_steps"]) / scale["traj_steps"])
            + 0.20 * abs((abs_theta - med["abs_theta_r"]) / scale["abs_theta_r"])
        )
        row["selection_random_priority"] = float(rng_priority[row["candidate_index"]])
        row["selection_fallbacks"] = []
        row["previous_exact_case_match"] = previous_hashes.get(str(row.get("case_snapshot_sha256", "")))
        enriched.append(row)
    # The row-level metadata produced by case_metadata does not include the full
    # reset snapshot hash, so exact previous-case exclusion is applied in
    # generate_bank_if_needed after hashes are attached.  If a match is found the
    # candidate is removed from all pools before outcome-free selection.
    eligible_rows = [r for r in enriched if not r.get("previous_exact_case_match")]
    if len(eligible_rows) < SELECTED_FULL:
        raise ContractError("too few fresh non-previous candidates after exact-case exclusion")

    def relax_pool(name: str, strict: Sequence[Mapping[str, Any]], relaxed: Sequence[Mapping[str, Any]], k: int, used: Iterable[int]) -> Tuple[List[Mapping[str, Any]], Optional[str]]:
        used_set = set(int(x) for x in used)
        strict_avail = [r for r in strict if int(r["candidate_index"]) not in used_set and not r.get("previous_exact_case_match")]
        if len(strict_avail) >= k:
            return strict_avail, None
        relaxed_avail = [r for r in relaxed if int(r["candidate_index"]) not in used_set and not r.get("previous_exact_case_match")]
        if len(relaxed_avail) >= k:
            return relaxed_avail, f"{name}_relaxed_thresholds"
        remaining = [r for r in eligible_rows if int(r["candidate_index"]) not in used_set]
        return remaining, f"{name}_underfilled_used_global_remaining"

    selected: List[Dict[str, Any]] = []
    used: List[int] = []
    fallbacks: List[str] = []

    high_long_strict = [r for r in eligible_rows if float(r["abs_theta_r"]) >= thr["abs_q70"] and float(r["traj_steps"]) >= thr["traj_median"]]
    high_long_relaxed = [r for r in eligible_rows if float(r["abs_theta_r"]) >= thr["abs_q60"] and float(r["traj_steps"]) >= thr["traj_q40"]]
    pool, fb = relax_pool("fresh_high_heading_long_or_medium", high_long_strict, high_long_relaxed, GROUP_COUNTS_FULL["fresh_high_heading_long_or_medium"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_balanced_sign(pool, GROUP_COUNTS_FULL["fresh_high_heading_long_or_medium"], "fresh_high_heading_long_or_medium", used, rng_priority, prefer_high_score=True)
    for r in rows:
        r["selection_reason"] = "fresh high-|theta_r| long/medium transient stratum; metadata-only sign-balanced diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)
    used.extend(int(r["candidate_index"]) for r in rows)

    high_short_strict = [r for r in eligible_rows if float(r["abs_theta_r"]) >= thr["abs_q70"] and float(r["traj_steps"]) < thr["traj_median"]]
    high_short_relaxed = [r for r in eligible_rows if float(r["abs_theta_r"]) >= thr["abs_q60"] and float(r["traj_steps"]) < thr["traj_q60"]]
    pool, fb = relax_pool("fresh_high_heading_short", high_short_strict, high_short_relaxed, GROUP_COUNTS_FULL["fresh_high_heading_short"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_balanced_sign(pool, GROUP_COUNTS_FULL["fresh_high_heading_short"], "fresh_high_heading_short", used, rng_priority, prefer_high_score=True)
    for r in rows:
        r["selection_reason"] = "fresh high-|theta_r| short transient control stratum; metadata-only diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)
    used.extend(int(r["candidate_index"]) for r in rows)

    low_clear_strict = [r for r in eligible_rows if float(r["abs_theta_r"]) <= thr["abs_median"] and float(r["min_reference_obstacle_clearance"]) <= thr["clear_q30"]]
    low_clear_relaxed = [r for r in eligible_rows if float(r["abs_theta_r"]) <= thr["abs_q60"] and float(r["min_reference_obstacle_clearance"]) <= thr["clear_q40"]]
    pool, fb = relax_pool("fresh_low_heading_low_clearance_control", low_clear_strict, low_clear_relaxed, GROUP_COUNTS_FULL["fresh_low_heading_low_clearance_control"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_diverse(pool, GROUP_COUNTS_FULL["fresh_low_heading_low_clearance_control"], "fresh_low_heading_low_clearance_control", used, rng_priority, prefer_high_score=True)
    for r in rows:
        r["selection_reason"] = "fresh low-heading low-clearance control stratum; metadata-only diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)
    used.extend(int(r["candidate_index"]) for r in rows)

    lower_stress_strict = [r for r in eligible_rows if float(r["abs_theta_r"]) <= thr["abs_median"] and float(r["min_reference_obstacle_clearance"]) > thr["clear_q30"]]
    lower_stress_relaxed = [r for r in eligible_rows if float(r["abs_theta_r"]) <= thr["abs_q60"]]
    pool, fb = relax_pool("fresh_lower_stress_control", lower_stress_strict, lower_stress_relaxed, GROUP_COUNTS_FULL["fresh_lower_stress_control"], used)
    if fb:
        fallbacks.append(fb)
    rows = choose_diverse(pool, GROUP_COUNTS_FULL["fresh_lower_stress_control"], "fresh_lower_stress_control", used, rng_priority, prefer_high_score=False)
    for r in rows:
        r["selection_reason"] = "fresh lower-stress control stratum; metadata-only diversity before H scoring"
        if fb:
            r.setdefault("selection_fallbacks", []).append(fb)
    selected.extend(rows)

    if len(selected) != SELECTED_FULL or len({int(r["candidate_index"]) for r in selected}) != SELECTED_FULL:
        raise ContractError("selection produced wrong number or duplicate candidates")
    selected_meta: List[Dict[str, Any]] = []
    for i, r in enumerate(selected):
        out = copy.deepcopy(dict(r))
        out["selected_case_index_full"] = int(i)
        out["stratum"] = out.get("selection_group")
        selected_meta.append(out)
    smoke_indices: List[int] = []
    for group, k in GROUP_COUNTS_SMOKE.items():
        group_indices = [i for i, r in enumerate(selected_meta) if r.get("selection_group") == group]
        if len(group_indices) < k:
            raise ContractError(f"not enough selected full cases for smoke group {group}")
        smoke_indices.extend(group_indices[:k])
    if len(smoke_indices) != SELECTED_SMOKE or len(set(smoke_indices)) != SELECTED_SMOKE:
        raise ContractError("smoke subset selection failed")
    for order, idx in enumerate(smoke_indices):
        selected_meta[idx]["selected_case_index_smoke"] = int(order)
        selected_meta[idx]["included_in_smoke"] = True
    for i, row in enumerate(selected_meta):
        row.setdefault("included_in_smoke", False)
    return {
        "thresholds": thr,
        "group_counts_full": GROUP_COUNTS_FULL,
        "group_counts_smoke": GROUP_COUNTS_SMOKE,
        "all_candidate_metadata": enriched,
        "excluded_exact_previous_cases": excluded_exact_previous,
        "selected_indices_full": [int(r["candidate_index"]) for r in selected_meta],
        "selected_full_metadata": selected_meta,
        "selected_full_case_indices": list(range(SELECTED_FULL)),
        "selected_smoke_case_indices_full_order": smoke_indices,
        "selected_smoke_candidate_indices": [int(selected_meta[i]["candidate_index"]) for i in smoke_indices],
        "selection_rule": "fresh v1c metadata-only selection: 12 high-heading long/medium, 3 high-heading short, 2 low-heading low-clearance controls, 3 lower-stress controls; smoke subset first 4/1/1 from the preselected groups; no horizon outcomes used",
        "selection_fallbacks": fallbacks,
        "selection_seed": SELECTION_SEED,
    }


def branch_steps_for_meta(meta: Mapping[str, Any]) -> List[int]:
    traj_steps = int(meta.get("traj_steps", 0) or 0)
    if traj_steps <= 0:
        raise ContractError("metadata lacks positive traj_steps")
    vals = [int(math.floor(0.15 * traj_steps)), int(math.floor(0.35 * traj_steps))]
    out: List[int] = []
    for v in vals:
        clipped = max(8, min(90, int(v)))
        if clipped not in out:
            out.append(clipped)
    return out


def build_schedule(selection: Mapping[str, Any], run_kind: str) -> List[Dict[str, Any]]:
    if run_kind == "smoke":
        case_indices = [int(x) for x in selection["selected_smoke_case_indices_full_order"]]
    elif run_kind == "full":
        case_indices = [int(x) for x in selection["selected_full_case_indices"]]
    else:
        raise ContractError(f"unknown run_kind {run_kind}")
    selected_meta = selection["selected_full_metadata"]
    base_rows: List[Dict[str, Any]] = []
    for full_case_index in case_indices:
        meta = selected_meta[full_case_index]
        for branch_step in branch_steps_for_meta(meta):
            for mode in TERMINAL_MODES:
                for h in BRANCH_HORIZONS:
                    base_rows.append({
                        "state_id": f"case{full_case_index:02d}_cand{int(meta['candidate_index']):03d}_b{branch_step:03d}",
                        "case": int(full_case_index),
                        "source_candidate_index": int(meta["candidate_index"]),
                        "selection_group": meta.get("selection_group"),
                        "branch_step": int(branch_step),
                        "horizon": int(h),
                        "terminal_mode": str(mode),
                        "prefix_horizon": PREFIX_H,
                        "run_kind": run_kind,
                    })
    rng = random.Random(ORDER_SEED + (0 if run_kind == "smoke" else 100000))
    order = list(range(len(base_rows)))
    rng.shuffle(order)
    return [dict(base_rows[i], schedule_base_index=int(i), execution_index=int(j)) for j, i in enumerate(order)]


def import_legacy_modules() -> Tuple[Any, Any, Any]:
    import vehicle_stage2_terminal_objective_smoke_v0 as base_smoke  # type: ignore
    import vehicle_stress_scenario_opportunity_probe_v1_runner as stage1_runner  # type: ignore
    import vehicle_fixed_h_opportunity_probe_v0_runner as fixed_base  # type: ignore
    return base_smoke, stage1_runner, fixed_base


def generate_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        completed_passed(BANK_COMPLETED, check_hashes=True)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "vehicle_stress_v1c_terminal_stable_label_density_probe_v0_fresh256_select20_no_validation64_no_test":
            raise ContractError("existing v1c bank split mismatch")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_full": SELECTED_FULL, "selected_smoke": SELECTED_SMOKE}
    _, _, fixed_base = import_legacy_modules()
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = fixed_base.v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = fixed_base.v1.meter(env, gen_dir)
    recovery = fixed_base.v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(CANDIDATE_POOL_SEED)
    try:
        import numpy as np  # type: ignore
        np.random.seed(CANDIDATE_POOL_SEED)
    except Exception:
        pass
    previous_hashes = previous_selected_case_hashes()
    cases: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    excluded_matches: List[Dict[str, Any]] = []
    for cid in range(CANDIDATE_RESETS):
        recovery.update(enabled=False, events=[], case=cid, step=-1)
        env.reset()
        case = fixed_base.snapshot(env)
        h = case_hash(case)
        meta = fixed_base.case_metadata(case, cid)
        meta["case_snapshot_sha256"] = h
        if h in previous_hashes:
            meta["previous_exact_case_match"] = previous_hashes[h]
            excluded_matches.append({"candidate_index": cid, "previous": previous_hashes[h], "case_snapshot_sha256": h})
        else:
            meta["previous_exact_case_match"] = None
        cases.append(case)
        metas.append(meta)
    if int(counts.get("step_calls", -1)) != 0 or int(counts.get("reset_calls", -1)) != CANDIDATE_RESETS:
        raise ContractError(f"unexpected bank generation counts: {counts}")
    selection = select_cases_v1c(metas, previous_hashes)
    selection["excluded_exact_previous_cases"] = excluded_matches
    selected_cases_full = [cases[int(i)] for i in selection["selected_indices_full"]]
    bank = {
        "task": TASK,
        "split": "vehicle_stress_v1c_terminal_stable_label_density_probe_v0_fresh256_select20_no_validation64_no_test",
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "selection_seed": SELECTION_SEED,
        "candidate_case_count": CANDIDATE_RESETS,
        "selected_full_case_count": SELECTED_FULL,
        "selected_smoke_case_count": SELECTED_SMOKE,
        "candidate_cases": cases,
        "selected_cases_full": selected_cases_full,
        "selection": selection,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "sealed_test_accessed": False,
        "purpose": "development-only fresh non-mined terminal-stable label-density probe; selection uses reset metadata only before horizon outcomes",
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "hard_pass": True,
        "task": TASK,
        "split": bank["split"],
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "selection_seed": SELECTION_SEED,
        "candidate_resets": CANDIDATE_RESETS,
        "selected_full": SELECTED_FULL,
        "selected_smoke": SELECTED_SMOKE,
        "excluded_exact_previous_cases": len(excluded_matches),
        "reset_calls": counts.get("reset_calls"),
        "step_calls": counts.get("step_calls"),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_full": SELECTED_FULL, "selected_smoke": SELECTED_SMOKE, "reset_calls": counts.get("reset_calls"), "excluded_exact_previous_cases": len(excluded_matches)}


def state_tuple(obj: Any) -> Tuple[float, float, float]:
    if not isinstance(obj, Mapping):
        return (float("nan"), float("nan"), float("nan"))
    return (float(obj.get("x", float("nan"))), float(obj.get("y", float("nan"))), float(obj.get("theta", float("nan"))))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def no_regression(candidate: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(candidate.get("success")):
        return False
    if bool(candidate.get("constraint")) and not bool(ref.get("constraint")):
        return False
    for key in ("initial_failed_steps", "final_failed_steps", "solver_failure_steps"):
        if int(candidate.get(key, 0)) > int(ref.get(key, 0)):
            return False
    return True


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def analyze_episodes(episodes: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], run_kind: str) -> Dict[str, Any]:
    by_state_mode: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    for e in episodes:
        key = (str(e["state_id"]), str(e["terminal_mode"]))
        by_state_mode.setdefault(key, {})[int(e["branch_horizon"])] = e
    state_ids = sorted({str(e["state_id"]) for e in episodes})
    artifact_flags = {
        "missing_H15_reference": 0,
        "missing_horizon_or_terminal_mode": 0,
        "branch_not_reached": 0,
        "state_distance_gt_tol": 0,
        "prefix_hash_mismatch": 0,
        "safety_solver_regression": 0,
        "positive_with_common_terminal_non_success": 0,
    }
    state_rows: List[Dict[str, Any]] = []
    robust_positive_case_ids: List[int] = []
    robust_positive_horizon_counts: Dict[str, int] = {}
    control_positive_states = 0
    control_state_count = 0
    timing_by_h_mode: Dict[str, List[float]] = {}
    for sid in state_ids:
        rows_for_state = [e for e in episodes if str(e["state_id"]) == sid]
        example = rows_for_state[0]
        case_id = int(example["case"])
        group = str(example.get("selection_group"))
        is_control = group in ("fresh_low_heading_low_clearance_control", "fresh_lower_stress_control")
        if is_control:
            control_state_count += 1
        mode_results: Dict[str, Any] = {}
        robust_sets: List[set] = []
        all_comparisons: List[Dict[str, Any]] = []
        for mode in TERMINAL_MODES:
            by_h = by_state_mode.get((sid, mode), {})
            ref = by_h.get(PREFIX_H)
            if ref is None:
                artifact_flags["missing_H15_reference"] += 1
                mode_results[mode] = {"missing_H15_reference": True, "material_horizons": []}
                robust_sets.append(set())
                continue
            ref_state = state_tuple(ref.get("branch_previous_state"))
            ref_prefix_hash = ref.get("prefix_clean_dynamics_sha256") or ref.get("prefix_clean_sha256")
            material_horizons: List[int] = []
            comparisons: List[Dict[str, Any]] = []
            if not bool(ref.get("branch_reached")):
                artifact_flags["branch_not_reached"] += 1
            for h in BRANCH_HORIZONS:
                e = by_h.get(h)
                if e is None:
                    artifact_flags["missing_horizon_or_terminal_mode"] += 1
                    continue
                timing_by_h_mode.setdefault(f"H{h}_{mode}", []).append(safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0))
                if not bool(e.get("branch_reached")):
                    artifact_flags["branch_not_reached"] += 1
                cand_state = state_tuple(e.get("branch_previous_state"))
                dist = state_distance(cand_state, ref_state)
                prefix_hash = e.get("prefix_clean_dynamics_sha256") or e.get("prefix_clean_sha256")
                prefix_match = bool(prefix_hash == ref_prefix_hash)
                if h != PREFIX_H and not prefix_match:
                    artifact_flags["prefix_hash_mismatch"] += 1
                if h != PREFIX_H and dist > STATE_DISTANCE_TOL:
                    artifact_flags["state_distance_gt_tol"] += 1
                phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - e["continuation_physical_constraint_cost_from_branch"])
                total_gain = float(ref["continuation_total_cost_from_branch"] - e["continuation_total_cost_from_branch"])
                safety_ok = no_regression(e, ref)
                if h != PREFIX_H and not safety_ok:
                    artifact_flags["safety_solver_regression"] += 1
                material = bool(h != PREFIX_H and bool(e.get("branch_reached")) and bool(ref.get("branch_reached")) and prefix_match and dist <= STATE_DISTANCE_TOL and safety_ok and phys_gain >= MATERIAL_GAIN)
                if material:
                    material_horizons.append(h)
                bdiag = e.get("branch_step_mpc_diag") or {}
                comp = {
                    "state_id": sid,
                    "case": case_id,
                    "selection_group": group,
                    "terminal_mode": mode,
                    "horizon": h,
                    "branch_reached": bool(e.get("branch_reached")),
                    "success": bool(e.get("success")),
                    "constraint": bool(e.get("constraint")),
                    "steps": int(e.get("steps", 0)),
                    "initial_failed_steps": int(e.get("initial_failed_steps", 0)),
                    "final_failed_steps": int(e.get("final_failed_steps", 0)),
                    "solver_failure_steps": int(e.get("solver_failure_steps", 0)),
                    "continuation_physical": safe_float(e.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                    "continuation_total": safe_float(e.get("continuation_total_cost_from_branch"), 0.0),
                    "synthetic_compute_from_branch": safe_float(e.get("continuation_total_cost_from_branch"), 0.0) - safe_float(e.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                    "gain_vs_H15_physical": phys_gain,
                    "gain_vs_H15_total": total_gain,
                    "state_distance_vs_H15": dist,
                    "prefix_hash_matches_H15": prefix_match,
                    "no_success_constraint_solver_regression_vs_H15": safety_ok,
                    "material_positive_terminal_mode": material,
                    "branch_mpc_value_fn": bdiag.get("info_mpc_value_fn"),
                    "branch_objective_opt_f_num": bdiag.get("objective_opt_f_num"),
                    "decision_sum_s": safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0),
                    "solver_attempt_sum_s": safe_float((e.get("solver_attempt_timing_s") or {}).get("sum"), 0.0),
                    "path": e.get("path"),
                }
                comparisons.append(comp)
                all_comparisons.append(comp)
            mode_results[mode] = {
                "reference_H15": {
                    "success": bool(ref.get("success")),
                    "constraint": bool(ref.get("constraint")),
                    "continuation_physical": safe_float(ref.get("continuation_physical_constraint_cost_from_branch"), 0.0),
                    "continuation_total": safe_float(ref.get("continuation_total_cost_from_branch"), 0.0),
                    "steps": int(ref.get("steps", 0)),
                    "path": ref.get("path"),
                },
                "material_horizons": material_horizons,
                "comparisons": comparisons,
            }
            robust_sets.append(set(material_horizons))
        robust_h = sorted(set.intersection(*robust_sets) if robust_sets else set())
        robust_positive = bool(robust_h)
        if robust_positive:
            robust_positive_case_ids.append(case_id)
            if is_control:
                control_positive_states += 1
            for h in robust_h:
                robust_positive_horizon_counts[str(h)] = robust_positive_horizon_counts.get(str(h), 0) + 1
            # Safety check required by the frozen gate.
            for comp in all_comparisons:
                if comp["horizon"] in robust_h and comp["terminal_mode"] == "h15_common_terminal" and not comp["success"]:
                    artifact_flags["positive_with_common_terminal_non_success"] += 1
        best_by_phys = min(all_comparisons, key=lambda r: (r["continuation_physical"], r["decision_sum_s"], r["horizon"])) if all_comparisons else None
        state_rows.append({
            "state_id": sid,
            "case": case_id,
            "source_candidate_index": int(example.get("source_candidate_index", -1)),
            "selection_group": group,
            "branch_step": int(example.get("branch_step", -1)),
            "is_control_state": bool(is_control),
            "robust_positive_state": robust_positive,
            "robust_positive_horizons": robust_h,
            "label": "robust_positive_non_H15" if robust_positive else "negative_or_neutral",
            "best_by_physical_any_mode": best_by_phys,
            "mode_results": mode_results,
        })
    positive_state_count = sum(1 for r in state_rows if r["robust_positive_state"])
    negative_state_count = sum(1 for r in state_rows if not r["robust_positive_state"])
    distinct_positive_cases = sorted(set(robust_positive_case_ids))
    blocking_artifacts = int(artifact_flags["missing_H15_reference"] + artifact_flags["missing_horizon_or_terminal_mode"] + artifact_flags["branch_not_reached"] + artifact_flags["state_distance_gt_tol"] + artifact_flags["prefix_hash_mismatch"] + artifact_flags["positive_with_common_terminal_non_success"])
    control_false_positive_rate = (float(control_positive_states) / float(control_state_count)) if control_state_count else 0.0
    smoke_gate = bool(
        positive_state_count >= 2
        and len(distinct_positive_cases) >= 2
        and negative_state_count >= 2
        and blocking_artifacts == 0
    )
    positive_groups = sorted(set(str(r["selection_group"]) for r in state_rows if r["robust_positive_state"]))
    full_gate = bool(
        positive_state_count >= 8
        and len(distinct_positive_cases) >= 4
        and (len(positive_groups) >= 2 or positive_groups == ["fresh_high_heading_long_or_medium"] or positive_groups == ["fresh_high_heading_short"])
        and negative_state_count >= 6
        and control_false_positive_rate <= 0.25
        and blocking_artifacts == 0
    )
    return {
        "run_kind": run_kind,
        "state_count": len(state_rows),
        "robust_positive_state_count": int(positive_state_count),
        "negative_or_neutral_state_count": int(negative_state_count),
        "robust_positive_cases": distinct_positive_cases,
        "robust_positive_horizon_counts": robust_positive_horizon_counts,
        "positive_groups": positive_groups,
        "control_state_count": int(control_state_count),
        "control_positive_state_count": int(control_positive_states),
        "control_false_positive_rate": float(control_false_positive_rate),
        "artifact_flags": artifact_flags,
        "blocking_artifact_count": blocking_artifacts,
        "smoke_pass_to_full": smoke_gate,
        "full_label_density_gate_for_selector_or_refit": full_gate,
        "timing_decision_sum_by_horizon_mode_s": {k: finite_summary(v) for k, v in sorted(timing_by_h_mode.items())},
        "state_rows": state_rows,
        "decision": {
            "train_or_refit_now": bool(run_kind == "full" and full_gate),
            "if_smoke": "run full only if smoke_pass_to_full is true after backup; otherwise do not train and pivot to scenario/terminal modeling diagnosis",
            "if_full_pass": "freeze compact IMPROVED selector/value-refit smoke using these labels as development data, then fresh confirmation with fair fixed-H/timing baselines",
            "if_full_fail": "do not train/refit on sparse labels; preserve evidence and prioritize scenario-opportunity or terminal-value modeling diagnostics",
        },
    }


def run_rollout(kind: str, backup_proof: Path) -> int:
    out_dir = SMOKE_DIR if kind == "smoke" else FULL_DIR
    state_path = STATE_SMOKE if kind == "smoke" else STATE_FULL
    marker = MARKER_SMOKE if kind == "smoke" else MARKER_FULL
    if not (DRYRUN_DIR / "completed.json").exists():
        raise ContractError("rollout requires completed dry-run marker")
    dry_done = completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
    if (out_dir / "completed.json").exists():
        completed_passed(out_dir / "completed.json", check_hashes=True)
        raise SystemExit(f"v1c {kind} already completed; refusing rerun")
    if out_dir.exists() and any(p.name != "run.lock" for p in out_dir.iterdir()):
        raise ContractError(f"partial v1c {kind} output exists; inspect before rerun: {rel(out_dir)}")
    protocol, v0b_done, _ = verify_protocol_and_inputs()
    min_time = max(parse_time(dry_done.get("created_utc")) or dt.datetime.now(dt.timezone.utc), source_mtime_utc(), parse_time(v0b_done.get("created_utc")) or dt.datetime.now(dt.timezone.utc))
    backup = verify_backup_proof(backup_proof, min_time, f"vehicle_stress_v1c_terminal_stable_label_density_{kind}")
    base_smoke, stage1_runner, _ = import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError(f"legacy runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    bank_info = generate_bank_if_needed()
    bank = read_json(BANK_PATH)
    selection = bank["selection"]
    selected_cases_full = bank["selected_cases_full"]
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out_dir / "run_started.json", {"started_utc": started, "pid": os.getpid(), "run_kind": kind, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0})
    write_json(out_dir / "runtime_preflight.json", preflight)
    write_json(out_dir / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    schedule = build_schedule(selection, kind)
    expected_upper = 168 if kind == "smoke" else 560
    if len(schedule) > expected_upper:
        raise ContractError(f"{kind} schedule length {len(schedule)} exceeds frozen upper bound {expected_upper}")
    write_json(out_dir / "schedule.json", {"episodes": schedule, "order_seed": ORDER_SEED, "branch_horizons": BRANCH_HORIZONS, "terminal_modes": TERMINAL_MODES, "prefix_horizon": PREFIX_H, "max_steps": MAX_STEPS})
    # Patch the reusable instrumented runner for this diagnostic.  The only
    # semantic extension is the name h15_common_terminal, which intentionally
    # shares H15 terminal weights for every branch horizon.
    base_smoke.OUT = out_dir
    base_smoke.PREFIX_H = PREFIX_H
    base_smoke.MAX_STEPS = MAX_STEPS
    base_smoke.MATERIAL_GAIN = MATERIAL_GAIN
    original_terminal_weights_for = base_smoke.terminal_weights_for

    def terminal_weights_for_v1c(mode: str, h: int, terminals_obj: Mapping[int, Tuple[Any, Any]]) -> Tuple[Any, Any, str]:
        if mode == "h15_common_terminal":
            return terminals_obj[PREFIX_H][0], terminals_obj[PREFIX_H][1], "H15_common"
        return original_terminal_weights_for(mode, h, terminals_obj)

    base_smoke.terminal_weights_for = terminal_weights_for_v1c
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        summary = base_smoke.run_one(item, selected_cases_full[int(item["case"])], terminals, terminal_receipts)
        # Add metadata not known to the reusable runner and rewrite its per-episode summary.
        summary.update({
            "state_id": item["state_id"],
            "source_candidate_index": int(item["source_candidate_index"]),
            "selection_group": item.get("selection_group"),
            "run_kind": kind,
            "terminal_mode": item["terminal_mode"],
            "protocol_terminal_label_family": "terminal_stable_zero_or_H15_common",
        })
        write_json(ROOT / summary["path"] / "summary.json", summary)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "run_kind": kind,
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e.get("steps", 0)) for e in episodes)),
            "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination", "branch_reached")},
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
        }
        write_json(out_dir / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    if control_steps > len(schedule) * MAX_STEPS:
        raise ContractError("control-step budget violation")
    analysis = analyze_episodes(episodes, selection, kind)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0_%s_%s.json" % (kind.upper(), created.replace("-", "").replace(":", "").replace("+00:00", "+0000")))
    write_json(req, {
        "requested_utc": created,
        "reason": f"backup vehicle stress-v1c terminal-stable label-density {kind} outputs before further simulation/training/refit",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "candidate_pool_resets": bank_info.get("candidate_resets") if bank_info.get("created_now") else 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(out_dir), rel(state_path), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(req)],
    })
    raw = {
        "created_utc": created,
        "started_utc": started,
        "method": "vehicle_stress_v1c_terminal_stable_label_density_probe_v0",
        "run_kind": kind,
        "classification": "development_IMPROVED_fresh_nonmined_terminal_stable_label_density_probe_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "v0b_input": {"completed": rel(V0B_DONE), "completed_sha256": sha256(V0B_DONE), "raw": rel(V0B_RAW), "raw_sha256": sha256(V0B_RAW)},
        "dry_run": {"completed": rel(DRYRUN_DIR / "completed.json"), "completed_sha256": sha256(DRYRUN_DIR / "completed.json")},
        "bank": bank_info,
        "bank_selection": selection,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"episodes_upper_bound": expected_upper, "control_step_upper_bound": expected_upper * MAX_STEPS, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": CANDIDATE_RESETS if bank_info.get("created_now") else 0, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": {str(k): v for k, v in terminal_receipts.items()},
        "episodes": episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development diagnostic only", "fresh non-mined metadata-selected bank", "terminal-stable labels only", "not adaptive selector validation", "not training/refit", "not ORIGINAL SAC", "not final test"],
    }
    write_json(out_dir / "raw.json", raw)
    write_run_summary(out_dir, raw)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        f"# Vehicle stress-v1c terminal-stable label-density probe v0 {kind}\n\n"
        f"UTC: {created}. Completed {len(episodes)} episodes / {control_steps} control steps. "
        f"Robust positives={analysis['robust_positive_state_count']} across cases={analysis['robust_positive_cases']}; "
        f"negative/neutral={analysis['negative_or_neutral_state_count']}; smoke_gate={analysis['smoke_pass_to_full']}; full_gate={analysis['full_label_density_gate_for_selector_or_refit']}. "
        "No validation64/test/training. Backup required before further simulation/refit.\n",
        encoding="utf-8",
    )
    append_docs(f"""<!-- {marker} -->
## 2026-09-29 vehicle stress-v1c terminal-stable label-density probe v0 {kind}

UTC: {created}. Development-only terminal-stable label-density {kind} completed: {len(episodes)} episodes, {control_steps} control steps, candidate resets={CANDIDATE_RESETS if bank_info.get('created_now') else 0}. Robust-positive states={analysis['robust_positive_state_count']} across cases={analysis['robust_positive_cases']}; negative/neutral states={analysis['negative_or_neutral_state_count']}; smoke gate={analysis['smoke_pass_to_full']}; full label-density gate={analysis['full_label_density_gate_for_selector_or_refit']}; blocking artifacts={analysis['blocking_artifact_count']}. No validation64-bank or sealed-test access, no training/refit. Artifacts: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`.
""", marker)
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"] + [state_path, req, Path(__file__).resolve(), PROTOCOL_JSON, backup_proof, V0B_DONE, V0B_RAW]
    write_json(out_dir / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "candidate_pool_resets": CANDIDATE_RESETS if bank_info.get("created_now") else 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": rel(req),
        "headline": {
            "robust_positive_state_count": analysis["robust_positive_state_count"],
            "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"],
            "robust_positive_cases": analysis["robust_positive_cases"],
            "smoke_pass_to_full": analysis["smoke_pass_to_full"],
            "full_label_density_gate_for_selector_or_refit": analysis["full_label_density_gate_for_selector_or_refit"],
            "blocking_artifact_count": analysis["blocking_artifact_count"],
            "next_action": "if smoke gate passes, run full after backup; if full gate passes, freeze compact selector/refit; otherwise do not train/refit and pivot to scenario/terminal modeling diagnosis",
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(out_dir / "completed.json"),
        "summary": rel(out_dir / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "candidate_pool_resets": CANDIDATE_RESETS if bank_info.get("created_now") else 0,
        "robust_positive_state_count": analysis["robust_positive_state_count"],
        "negative_or_neutral_state_count": analysis["negative_or_neutral_state_count"],
        "robust_positive_cases": analysis["robust_positive_cases"],
        "smoke_pass_to_full": analysis["smoke_pass_to_full"],
        "full_label_density_gate_for_selector_or_refit": analysis["full_label_density_gate_for_selector_or_refit"],
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(req),
    }, sort_keys=True), flush=True)
    return 0


def write_run_summary(out_dir: Path, raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        f"# Vehicle stress-v1c terminal-stable label-density probe v0 {raw['run_kind']}",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / upper `{raw['budget_declared']['episodes_upper_bound']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- Robust-positive states: `{a['robust_positive_state_count']}` across cases `{a['robust_positive_cases']}`.",
        f"- Negative/neutral states: `{a['negative_or_neutral_state_count']}`.",
        f"- Robust positive horizons: `{a['robust_positive_horizon_counts']}`.",
        f"- Positive groups: `{a['positive_groups']}`.",
        f"- Control false-positive rate: `{a['control_false_positive_rate']}` ({a['control_positive_state_count']}/{a['control_state_count']}).",
        f"- Blocking artifacts: `{a['blocking_artifact_count']}`; flags `{a['artifact_flags']}`.",
        f"- Smoke pass-to-full: `{a['smoke_pass_to_full']}`; full label-density gate: `{a['full_label_density_gate_for_selector_or_refit']}`.",
        "",
        "## Per-state labels",
        "",
        "| state | case | group | branch step | label | robust H | best phys mode/H/cost |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in a["state_rows"]:
        best = row.get("best_by_physical_any_mode") or {}
        lines.append("| `%s` | %d | `%s` | %d | `%s` | `%s` | `%s`/H%s/%.6g |" % (
            row["state_id"], int(row["case"]), row.get("selection_group"), int(row["branch_step"]), row.get("label"), row.get("robust_positive_horizons"), best.get("terminal_mode"), str(best.get("horizon")), float(best.get("continuation_physical", 0.0)),
        ))
    lines += [
        "",
        "## Four-axis evidence update",
        "",
        "### SCENARIOS",
        "",
        f"- verified: fresh non-mined metadata-selected states were probed; robust positives={a['robust_positive_state_count']} across cases={a['robust_positive_cases']}; controls={a['control_state_count']}.",
        "- hypothesis: if positives are sparse, the opportunity is concentrated in mined/high-transient states rather than broadly available; if dense, selector/refit can be justified next.",
        "- missing: independent validation and fixed-H/timing comparison remain unavailable from this development diagnostic.",
        "- discriminator: use the frozen smoke/full gates rather than training on sparse labels.",
        "",
        "### REWARD",
        "",
        "- verified: labels use realised physical continuation cost under zero_terminal and H15-common terminal only; synthetic total and actual timing are reported separately and not used as primary labels.",
        "- hypothesis: terminal-source artifacts are reduced relative to per-H/h25 labels; remaining positives are more credible selector targets.",
        "- missing: terminal-value refit or learned terminal alternative if labels remain sparse.",
        "- discriminator: compare zero vs H15-common material consistency in raw state rows.",
        "",
        "### TRAINING",
        "",
        f"- verified: train/refit now is `{a['decision']['train_or_refit_now']}`; new_training_episodes=0, new_gradient_steps=0.",
        "- hypothesis: selector learning should only proceed if the full label-density gate passes.",
        "- missing: compact IMPROVED selector/value-refit smoke and 3 independent seeds remain future work.",
        "- discriminator: freeze that refit only after enough robust labels exist.",
        "",
        "### COMPARISONS",
        "",
        "- verified: no adaptive-vs-fixed superiority claim; actual timing recorded per horizon/mode but not yet randomized fair baseline validation.",
        "- hypothesis: strong fixed-H/Pareto baselines may absorb gains after terminal artifacts are removed.",
        "- missing: fair fixed-H retuning and fresh validation timing after any selector/refit.",
        "- discriminator: later revised protocol with fixed-H grid and measured timing if label gate passes.",
        "",
        "## Decision",
        "",
        f"- Train/refit now: `{a['decision']['train_or_refit_now']}`.",
        f"- Backup request: `{raw['backup_request_after_run']}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-v1c terminal-stable label-density probe v0 dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no candidate resets, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Verified inputs",
        "",
        f"- Protocol: `{raw['protocol']['json']}` sha256 `{raw['protocol']['json_sha256']}`.",
        f"- Corrected objective diagnostic v0b: `{raw['v0b_input']['completed']}` sha256 `{raw['v0b_input']['completed_sha256']}`.",
        f"- v0b headline: `{raw['v0b_headline']}`.",
        "",
        "## Frozen next simulation budget (not yet run)",
        "",
        f"- Candidate-pool resets: `{CANDIDATE_RESETS}`; full selected cases `{SELECTED_FULL}`; smoke selected cases `{SELECTED_SMOKE}`.",
        f"- Branch horizons: `{BRANCH_HORIZONS}`; terminal modes: `{TERMINAL_MODES}`; prefix H `{PREFIX_H}`.",
        f"- Smoke upper bound: `{raw['planned_smoke']['episodes_upper_bound']}` episodes / `{raw['planned_smoke']['control_step_upper_bound']}` control steps.",
        f"- Full upper bound: `{raw['planned_full']['episodes_upper_bound']}` episodes / `{raw['planned_full']['control_step_upper_bound']}` control steps.",
        "",
        "## Four-axis decision",
        "",
        "- SCENARIOS: next rollout is fresh non-mined high-heading/long-transient opportunity density with controls.",
        "- REWARD: labels are realised physical-continuation primary under zero/H15-common terminal only; per-H/h25 excluded as targets.",
        "- TRAINING: no selector/refit until full label-density gate passes.",
        "- COMPARISONS: no superiority claim; fixed-H/timing baselines remain later after a method/scenario protocol is frozen.",
        "",
        f"Backup request before smoke: `{raw['backup_request_before_smoke']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    if (DRYRUN_DIR / "completed.json").exists():
        completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(DRYRUN_DIR / "completed.json"), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    protocol, v0b_done, _ = verify_protocol_and_inputs()
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DRYRUN_DIR / "run_started.json", {"started_utc": created, "pid": os.getpid(), "method": "vehicle_stress_v1c_terminal_stable_label_density_probe_v0_dryrun", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0})
    # Dry-run deliberately avoids legacy/TF imports and any environment reset.
    smoke_schedule_blueprint = {
        "selected_cases": SELECTED_SMOKE,
        "branch_state_steps_per_case": 2,
        "branch_horizons": BRANCH_HORIZONS,
        "terminal_modes": TERMINAL_MODES,
        "episodes_upper_bound": SELECTED_SMOKE * 2 * len(BRANCH_HORIZONS) * len(TERMINAL_MODES),
        "control_step_upper_bound": SELECTED_SMOKE * 2 * len(BRANCH_HORIZONS) * len(TERMINAL_MODES) * MAX_STEPS,
    }
    full_schedule_blueprint = {
        "selected_cases": SELECTED_FULL,
        "branch_state_steps_per_case": 2,
        "branch_horizons": BRANCH_HORIZONS,
        "terminal_modes": TERMINAL_MODES,
        "episodes_upper_bound": SELECTED_FULL * 2 * len(BRANCH_HORIZONS) * len(TERMINAL_MODES),
        "control_step_upper_bound": SELECTED_FULL * 2 * len(BRANCH_HORIZONS) * len(TERMINAL_MODES) * MAX_STEPS,
    }
    if smoke_schedule_blueprint["episodes_upper_bound"] != 168 or full_schedule_blueprint["episodes_upper_bound"] != 560:
        raise ContractError("dry-run schedule blueprint mismatch")
    write_json(REQUEST_BACKUP_BEFORE_SMOKE, {
        "requested_utc": created,
        "reason": "backup v1c terminal-stable label-density runner dry-run, frozen protocol and v0b corrected objective diagnostic before any fresh smoke simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_smoke_episodes_upper_bound": smoke_schedule_blueprint["episodes_upper_bound"],
        "planned_smoke_control_step_upper_bound": smoke_schedule_blueprint["control_step_upper_bound"],
        "planned_full_episodes_upper_bound": full_schedule_blueprint["episodes_upper_bound"],
        "planned_full_control_step_upper_bound": full_schedule_blueprint["control_step_upper_bound"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(V0B_DONE), rel(V0B_RAW), rel(V0B_SUMMARY), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(REQUEST_BACKUP_BEFORE_SMOKE)],
    })
    raw = {
        "created_utc": created,
        "method": "vehicle_stress_v1c_terminal_stable_label_density_probe_v0_dryrun_no_simulation",
        "classification": "development_protocol_readiness_no_simulation_terminal_stable_label_density_probe",
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
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "full": protocol},
        "v0b_input": {"completed": rel(V0B_DONE), "completed_sha256": sha256(V0B_DONE), "raw": rel(V0B_RAW), "raw_sha256": sha256(V0B_RAW), "summary": rel(V0B_SUMMARY), "summary_sha256": sha256(V0B_SUMMARY)},
        "v0b_backup_request": {"path": rel(V0B_BACKUP_REQUEST), "sha256": sha256(V0B_BACKUP_REQUEST)},
        "v0b_headline": v0b_done.get("headline"),
        "planned_smoke": smoke_schedule_blueprint,
        "planned_full": full_schedule_blueprint,
        "source_hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()), rel(PROTOCOL_JSON): sha256(PROTOCOL_JSON), rel(V0B_DONE): sha256(V0B_DONE), rel(V0B_RAW): sha256(V0B_RAW), rel(V0B_SUMMARY): sha256(V0B_SUMMARY)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "backup_request_before_smoke": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "next": "after verified external backup covering this dry-run/source/protocol/v0b, run --run-smoke under legacy interpreter with explicit development acceptance",
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text(
        f"# Vehicle stress-v1c terminal-stable label-density probe v0 dry-run\n\n"
        f"UTC: {created}. No simulations, no candidate resets, no training/refit, no validation64/test access. "
        f"Verified frozen protocol and v0b corrected objective diagnostic. Smoke remains blocked until external backup covers `{rel(REQUEST_BACKUP_BEFORE_SMOKE)}` plus source/protocol/dry-run outputs.\n",
        encoding="utf-8",
    )
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle stress-v1c terminal-stable label-density probe v0 dry-run

UTC: {created}. No-simulation dry-run completed for the fresh non-mined terminal-stable label-density probe. Verified frozen protocol `{rel(PROTOCOL_JSON)}` and corrected objective diagnostic v0b `{rel(V0B_DONE)}`; no candidate resets, rollouts, training/refit, validation64-bank access or sealed-test access. Smoke plan remains blocked until external backup covers the new runner, dry-run outputs, v0b artifacts and request `{rel(REQUEST_BACKUP_BEFORE_SMOKE)}`. Planned smoke upper bound is 168 episodes/25200 control steps; full upper bound is 560 episodes/84000 control steps.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_DRYRUN, REQUEST_BACKUP_BEFORE_SMOKE, Path(__file__).resolve(), PROTOCOL_JSON, V0B_DONE, V0B_RAW, V0B_SUMMARY]
    write_json(DRYRUN_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
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
        "planned_smoke_episodes_upper_bound": smoke_schedule_blueprint["episodes_upper_bound"],
        "planned_smoke_control_step_upper_bound": smoke_schedule_blueprint["control_step_upper_bound"],
        "backup_required_before_smoke": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "headline": {
            "train_or_refit_now": False,
            "next_action": "await verified external backup, then run v1c smoke under legacy interpreter; if smoke gate fails, do not train/refit",
            "v0b_terminal_flips": (v0b_done.get("headline") or {}).get("terminal_flips"),
            "v0b_min_objective_selected_non_success_rows": (v0b_done.get("headline") or {}).get("min_objective_selected_non_success_rows"),
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(DRYRUN_DIR / "completed.json"),
        "summary": rel(DRYRUN_DIR / "summary.md"),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "planned_smoke_episodes_upper_bound": smoke_schedule_blueprint["episodes_upper_bound"],
        "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE),
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no-simulation protocol/source readiness check")
    ap.add_argument("--run-smoke", action="store_true", help="run the frozen smoke schedule after verified backup")
    ap.add_argument("--run-full", action="store_true", help="run the frozen full schedule after verified backup and smoke pass")
    ap.add_argument("--backup-proof", type=Path, default=None, help="verified external backup proof for simulation modes")
    ap.add_argument("--i-accept-development-terminal-stable-label-density-probe", action="store_true", help="acknowledge development-only diagnostic; no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_development_terminal_stable_label_density_probe:
        raise ContractError("explicit --i-accept-development-terminal-stable-label-density-probe is required")
    modes = [bool(args.dry_run), bool(args.run_smoke), bool(args.run_full)]
    if sum(1 for x in modes if x) != 1:
        raise ContractError("exactly one of --dry-run, --run-smoke, --run-full is required")
    if args.dry_run:
        return run_dry_run()
    if args.backup_proof is None:
        raise ContractError("simulation modes require --backup-proof")
    if args.run_smoke:
        return run_rollout("smoke", args.backup_proof)
    return run_rollout("full", args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else SMOKE_DIR if "--run-smoke" in sys.argv else FULL_DIR if "--run-full" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0_failure_unknown_{STAMP}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_opened": False,
            "new_rollouts": 0 if "--dry-run" in sys.argv else None,
            "new_control_steps": 0 if "--dry-run" in sys.argv else None,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve this partial directory. If dry-run failed, repair protocol/source logic only; if simulation failed, audit partial episodes before any rerun.",
        })
        raise
