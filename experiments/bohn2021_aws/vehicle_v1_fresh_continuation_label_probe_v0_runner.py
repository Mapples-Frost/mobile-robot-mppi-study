#!/usr/bin/env python3
"""Vehicle V1 fresh continuation-label probe v0 runner.

Development-only controlled-continuation label mining frozen by
research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.*.

The runner generates a fresh source-supported vehicle development bank using
reset metadata only, then compares H10/H15/H30/H35 continuations from identical
H15-prefix states.  It performs no adaptive online selector evaluation, no
training/refit/gradient updates, opens no historical validation64 bank, and never
accesses the sealed final test.

A verified external backup proof after the broad selector postdiagnostic,
protocol, and this runner source is required before any rollout simulation.
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
BRANCH_HORIZONS = [10, 15, 30, 35]
MAX_STEPS = 150
CANDIDATE_RESETS = 48
SELECTED_CASES = 12
CANDIDATE_POOL_SEED = 2609287601
SELECTION_SEED = 2609287602
ORDER_SEED = 2609287603
IMPROVEMENT_ABS_THRESHOLD = 3.0
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928"
BANK_DIR = OUT_DIR / "bank"
BANK_PATH = BANK_DIR / "vehicle_v1_fresh_continuation_label_probe_v0_bank.json"
BANK_COMPLETED = BANK_DIR / "completed.json"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_probe_v0_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.md"
POSTDIAG_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_broad_postdiagnostic_v0_20260928T1345Z/raw.json"
POSTDIAG_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_broad_postdiagnostic_v0_20260928T1345Z/completed.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-v1-fresh-continuation-label-probe-v0-20260928"


class ContractError(RuntimeError):
    """Raised for protocol, access or budget violations."""


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


def latest_time(*values: Optional[dt.datetime]) -> Optional[dt.datetime]:
    vals = [v.astimezone(dt.timezone.utc) for v in values if v is not None]
    return max(vals) if vals else None


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


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
        raise ContractError("completed marker did not pass: %s" % rel(path))
    if check_hashes:
        for name, expected in (done.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError("completed marker references missing file: %s" % name)
            if sha256(p) != expected:
                raise ContractError("hash mismatch for completed marker input: %s" % name)
    return done


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof path does not exist: %s" % rel(path))
    proof = read_json(path)
    verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
    if not verified:
        raise ContractError("backup proof is not verified: %s" % rel(path))
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    has_asset = bool(
        proof.get("asset_sha256")
        or proof.get("release_asset_sha256")
        or proof.get("package_sha256")
        or proof.get("packages_this_run")
    )
    if not has_asset:
        raise ContractError("backup proof lacks a verified release/package SHA")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError("backup proof lacks parseable time; cannot prove it covers runner/protocol/postdiagnostic")
        if proof_time < min_time:
            raise ContractError("backup proof predates fresh continuation-label probe source/protocol/postdiagnostic")
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "time": proof_time.isoformat() if proof_time else None,
        "raw_status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
        "packages_this_run": proof.get("packages_this_run"),
        "min_required_time_utc": None if min_time is None else min_time.isoformat(),
    }


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed_marker(completed)
        raise SystemExit("fresh continuation-label probe already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("partial fresh continuation-label probe output exists; inspect/recover first: " + ", ".join(rel(p) for p in leftovers[:20]))


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for path in (PROTOCOL_JSON, PROTOCOL_MD, POSTDIAG_RAW, POSTDIAG_DONE, V1_RAW, V1_DONE):
        if not path.exists():
            raise ContractError("required input missing: %s" % rel(path))
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928":
        raise ContractError("unexpected fresh continuation-label protocol id")
    if protocol.get("classification") != "development_IMPROVED_fresh_state_continuation_label_probe_not_validation_not_final_test":
        raise ContractError("unexpected fresh continuation-label classification")
    access = protocol.get("access_rules") or {}
    if access.get("development_only") is not True or access.get("historical_validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("fresh continuation-label access flags are invalid")
    if access.get("requires_verified_external_backup_after_broad_postdiagnostic_before_rollout") is not True:
        raise ContractError("fresh continuation-label protocol must require verified backup before rollout")
    gen = protocol.get("generator_and_split") or {}
    if int(gen.get("fresh_candidate_pool_resets", -1)) != CANDIDATE_RESETS:
        raise ContractError("candidate-pool reset budget changed")
    if int(gen.get("selected_case_count", -1)) != SELECTED_CASES:
        raise ContractError("selected-case count changed")
    if int(gen.get("candidate_pool_seed", -1)) != CANDIDATE_POOL_SEED or int(gen.get("selection_seed", -1)) != SELECTION_SEED:
        raise ContractError("fresh bank seeds changed")
    design = protocol.get("rollout_design") or {}
    if int(design.get("prefix_horizon", -1)) != PREFIX_H or [int(x) for x in design.get("branch_horizons", [])] != BRANCH_HORIZONS:
        raise ContractError("branch horizons/prefix changed")
    if int(design.get("max_rollout_episodes", -1)) != 96 or int(design.get("control_step_upper_bound", -1)) != 14400:
        raise ContractError("fresh continuation-label rollout budget changed")
    if int(design.get("new_training_episodes", -1)) != 0 or int(design.get("new_gradient_steps", -1)) != 0:
        raise ContractError("fresh continuation-label protocol unexpectedly permits training")
    post_done = verify_completed_marker(POSTDIAG_DONE)
    if post_done.get("hard_pass") is not True or post_done.get("historical_validation64_bank_opened") is not False or post_done.get("sealed_test_accessed") is not False:
        raise ContractError("postdiagnostic completion/access flags invalid")
    post_raw = read_json(POSTDIAG_RAW)
    v1_done = verify_completed_marker(V1_DONE, check_hashes=False)
    v1_raw = read_json(V1_RAW)
    for label, obj in (("post_raw", post_raw), ("v1_raw", v1_raw), ("v1_done", v1_done)):
        if obj.get("historical_validation64_bank_opened") not in (False, None):
            raise ContractError(label + " unexpectedly opened historical validation64")
        if obj.get("sealed_test_accessed") not in (False, None):
            raise ContractError(label + " unexpectedly accessed sealed test")
    frozen_next = (post_done.get("headline") or {}).get("next_protocol") or {}
    if frozen_next.get("json") and frozen_next.get("json_sha256") and sha256(PROTOCOL_JSON) != frozen_next["json_sha256"]:
        raise ContractError("postdiagnostic next-protocol JSON hash mismatch")
    if frozen_next.get("md") and frozen_next.get("md_sha256") and sha256(PROTOCOL_MD) != frozen_next["md_sha256"]:
        raise ContractError("postdiagnostic next-protocol MD hash mismatch")
    return protocol, post_raw, v1_raw


def normalized_feature(meta: Mapping[str, Any], medians: Mapping[str, float], scales: Mapping[str, float]) -> Tuple[float, float, float]:
    vals = [float(meta.get("abs_theta_r", 0.0)), float(meta.get("traj_steps", 0.0)), float(meta.get("min_reference_obstacle_clearance", 0.0))]
    keys = ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"]
    out: List[float] = []
    for key, val in zip(keys, vals):
        if not math.isfinite(val):
            val = medians[key]
        out.append(float((val - medians[key]) / scales[key]))
    return out[0], out[1], out[2]


def enrich_and_select_metadata(metas: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance"]
    finite: Dict[str, List[float]] = {k: [] for k in keys}
    for m in metas:
        for k in keys:
            val = float(m.get(k, float("nan")))
            if math.isfinite(val):
                finite[k].append(val)
    medians = {k: float(np.median(finite[k])) if finite[k] else 0.0 for k in keys}
    scales = {k: float(max(np.std(finite[k]) if finite[k] else 1.0, 1e-9)) for k in keys}
    rng = np.random.RandomState(SELECTION_SEED)
    random_priority = {int(m["candidate_index"]): float(rng.uniform()) for m in metas}
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
        row["selection_random_priority"] = random_priority[int(row["candidate_index"])]
        enriched.append(row)
    selected: List[int] = []
    reasons: Dict[int, str] = {}
    strata = [
        "%s__%s__%s" % (theta, length, clearance)
        for theta in ("low_abs_theta", "high_abs_theta")
        for length in ("short_traj", "long_traj")
        for clearance in ("low_clearance", "high_clearance")
    ]
    # First pass: one deterministic seeded representative from each available metadata stratum.
    for stratum in strata:
        candidates = [m for m in enriched if m["stratum"] == stratum and int(m["candidate_index"]) not in selected]
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda m: (float(m["selection_random_priority"]), -float(m["extremeness_score"]), int(m["candidate_index"])))
        choice = candidates[0]
        idx = int(choice["candidate_index"])
        selected.append(idx)
        reasons[idx] = "seeded_one_per_available_stratum:%s" % stratum
    # Fill to the frozen size by diversity in reset metadata; no outcome values exist at this point.
    while len(selected) < SELECTED_CASES:
        remaining = [m for m in enriched if int(m["candidate_index"]) not in selected]
        if not remaining:
            break
        selected_features = [np.asarray(enriched[i]["normalized_features"], dtype=float) for i in selected]
        def score(m: Mapping[str, Any]) -> Tuple[float, float, float, int]:
            z = np.asarray(m["normalized_features"], dtype=float)
            if selected_features:
                mind = float(min(np.linalg.norm(z - s) for s in selected_features))
            else:
                mind = float("inf")
            return (mind, float(m["extremeness_score"]), -float(m["selection_random_priority"]), -int(m["candidate_index"]))
        choice = max(remaining, key=score)
        idx = int(choice["candidate_index"])
        selected.append(idx)
        reasons[idx] = "metadata_diversity_fill_seed_%d" % SELECTION_SEED
    if len(selected) != SELECTED_CASES:
        raise ContractError("unable to select the frozen number of fresh cases")
    selected_meta: List[Dict[str, Any]] = []
    for selected_order, idx in enumerate(selected):
        row = dict(enriched[idx])
        row["selected_order"] = int(selected_order)
        row["selection_reason"] = reasons[idx]
        selected_meta.append(row)
    return {
        "medians": medians,
        "scales": scales,
        "selection_seed": SELECTION_SEED,
        "selection_rule_implemented": "seeded one-per-available theta/length/clearance stratum, then metadata farthest-diversity fill to 12; reset metadata only, no rollout outcomes",
        "all_candidate_metadata": enriched,
        "selected_indices_in_order": selected,
        "selected_metadata": selected_meta,
    }


def generate_bank_if_needed() -> Dict[str, Any]:
    if BANK_COMPLETED.exists():
        verify_completed_marker(BANK_COMPLETED)
        bank = read_json(BANK_PATH)
        if bank.get("task") != TASK or bank.get("split") != "vehicle_v1_fresh_continuation_label_probe_v0_fresh48_select12_no_validation64_no_test":
            raise ContractError("existing fresh continuation-label bank metadata mismatch")
        if len(bank.get("candidate_cases", [])) != CANDIDATE_RESETS or len(bank.get("selected_cases", [])) != SELECTED_CASES:
            raise ContractError("existing fresh continuation-label bank dimensions mismatch")
        return {"created_now": False, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_cases": SELECTED_CASES}
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = BANK_DIR / "generation_logs"
    gen_dir.mkdir(parents=True, exist_ok=False)
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, gen_dir)
    recovery = v1.recovery_module.install(env.control_system.controller.mpc, gen_dir)
    env.seed(CANDIDATE_POOL_SEED)
    np.random.seed(CANDIDATE_POOL_SEED)
    cases: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for cid in range(CANDIDATE_RESETS):
        recovery.update(enabled=False, events=[], case=cid, step=-1, horizon=PREFIX_H)
        env.reset()
        case = base.snapshot(env)
        cases.append(case)
        metas.append(base.case_metadata(case, cid))
    if counts["step_calls"] != 0 or counts["reset_calls"] != CANDIDATE_RESETS:
        raise ContractError("unexpected bank generation counts: %r" % counts)
    selection = enrich_and_select_metadata(metas)
    selected_cases = [cases[int(i)] for i in selection["selected_indices_in_order"]]
    bank = {
        "task": TASK,
        "split": "vehicle_v1_fresh_continuation_label_probe_v0_fresh48_select12_no_validation64_no_test",
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "selection_seed": SELECTION_SEED,
        "candidate_case_count": CANDIDATE_RESETS,
        "selected_case_count": SELECTED_CASES,
        "candidate_cases": cases,
        "selected_cases": selected_cases,
        "selection": selection,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "fresh development-only continuation-label probe; no validation64 or sealed test access",
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
    }
    write_json(BANK_PATH, bank)
    files = [BANK_PATH] + [p for p in gen_dir.rglob("*") if p.is_file()]
    write_json(BANK_COMPLETED, {
        "passed": True,
        "task": TASK,
        "split": bank["split"],
        "candidate_pool_seed": CANDIDATE_POOL_SEED,
        "selection_seed": SELECTION_SEED,
        "candidate_resets": CANDIDATE_RESETS,
        "selected_cases": SELECTED_CASES,
        "reset_calls": counts["reset_calls"],
        "step_calls": counts["step_calls"],
        "historical_validation64_bank_opened": False,
        "sealed_test_bank_opened": False,
        "hashes": {rel(p): sha256(p) for p in sorted(files)},
    })
    return {"created_now": True, "path": rel(BANK_PATH), "sha256": sha256(BANK_PATH), "candidate_resets": CANDIDATE_RESETS, "selected_cases": SELECTED_CASES, "reset_calls": counts["reset_calls"]}


def branch_steps_for_meta(meta: Mapping[str, Any]) -> List[int]:
    traj_steps = int(meta.get("traj_steps", 0) or 0)
    if traj_steps <= 0:
        raise ContractError("selected metadata lacks positive traj_steps")
    steps = [int(math.floor(0.15 * traj_steps)), int(math.floor(0.35 * traj_steps))]
    clipped = [max(8, min(90, int(s))) for s in steps]
    out: List[int] = []
    for s in clipped:
        if s not in out:
            out.append(s)
    return out


def build_schedule(selected_meta: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_id, meta in enumerate(selected_meta):
        for branch_step in branch_steps_for_meta(meta):
            for h in BRANCH_HORIZONS:
                rows.append({"case": int(case_id), "branch_step": int(branch_step), "branch_horizon": int(h)})
    if len(rows) > 96:
        raise ContractError("fresh continuation-label schedule exceeds frozen max episodes")
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    for execution_index, base_index in enumerate(rng.permutation(len(rows)).tolist()):
        item = dict(rows[int(base_index)])
        item["schedule_base_index"] = int(base_index)
        item["execution_index"] = int(execution_index)
        schedule.append(item)
    return schedule


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


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_tuple(row: Mapping[str, Any], key: str = "previous_state") -> Tuple[float, float, float]:
    state = row.get(key) or {}
    return (safe_float(state.get("x"), float("nan")), safe_float(state.get("y"), float("nan")), safe_float(state.get("theta"), float("nan")))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def clean_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(list(trace[:min(branch_step, len(trace))]))
    for row in cleaned:
        row.pop("timing", None)
        row.pop("decision_plus_terminal_switch_s", None)
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def terminal_receipt_for_horizon(receipts: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    rec = receipts.get(str(h))
    if not rec:
        raise ContractError("missing terminal receipt for H%d" % h)
    return rec


def summarize_episode(
    trace: List[Dict[str, Any]],
    reset: Dict[str, Any],
    construction_s: float,
    episode_wall_s: float,
    item: Mapping[str, Any],
    selected_meta: Mapping[str, Any],
    terminal_switches: Sequence[Mapping[str, Any]],
    terminal_receipts: Mapping[str, Any],
    counts: Mapping[str, Any],
    logging: Any,
) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    branch_step = int(item["branch_step"])
    branch_h = int(item["branch_horizon"])
    branch_reached = len(trace) > branch_step
    branch_state = state_tuple(trace[branch_step], "previous_state") if branch_reached else (float("nan"), float("nan"), float("nan"))
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
    out = dict(metric)
    out.update({
        "execution_index": int(item["execution_index"]),
        "case": int(item["case"]),
        "source_candidate_index": int(selected_meta.get("candidate_index", -1)),
        "selected_case_order": int(selected_meta.get("selected_order", item["case"])),
        "stratum": selected_meta.get("stratum"),
        "case_metadata": selected_meta,
        "branch_step": branch_step,
        "branch_horizon": branch_h,
        "prefix_horizon": PREFIX_H,
        "arm_id": "case%02d_branch%03d_H%02d" % (int(item["case"]), branch_step, branch_h),
        "family": "fresh_controlled_continuation_H15_prefix_then_fixed_branch",
        "episode_failure": not bool(metric.get("success")),
        "branch_reached": bool(branch_reached),
        "skipped_noninformative_branch": bool(not branch_reached),
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
    case: Mapping[str, Any],
    selected_meta: Mapping[str, Any],
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
                raise ContractError("invalid logging timing bounds")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=int(item["case"]), step=-1, horizon=PREFIX_H)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_gross_s = time.perf_counter() - reset_start
        if obs is None or len(measured) != 1:
            raise ContractError("unexpected reset/controller warmup state")
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
                    raise ContractError("expected exactly one measured controller call")
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
                        "kind": "fresh_controlled_continuation_H15_prefix_then_fixed_branch",
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
            raise ContractError("episode did not terminate within max steps")
        v1.audit_trace(TASK, dict(case), trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(trace, reset_record, construction_s, time.perf_counter() - episode_start, item, selected_meta, terminal_switches, terminal_receipts, counts, logging)
        summary.update({"path": rel(ep_dir), "solver_counts": recovery["counts"]})
        write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


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


def analyze_labels(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for e in episodes:
        groups.setdefault((int(e["case"]), int(e["branch_step"])), []).append(e)
    state_rows: List[Dict[str, Any]] = []
    skipped_states: List[Dict[str, Any]] = []
    positive_state_count = 0
    negative_neutral_state_count = 0
    positive_cases: List[int] = []
    positive_horizon_counts: Dict[str, int] = {}
    missing_reference: List[Dict[str, int]] = []
    for key in sorted(groups):
        case_id, branch_step = key
        rows = sorted(groups[key], key=lambda r: int(r["branch_horizon"]))
        by_h = {int(r["branch_horizon"]): r for r in rows}
        if PREFIX_H not in by_h:
            missing_reference.append({"case": case_id, "branch_step": branch_step})
            continue
        ref = by_h[PREFIX_H]
        if not bool(ref.get("branch_reached")):
            skipped_states.append({"case": case_id, "branch_step": branch_step, "reason": "H15_prefix_terminated_before_branch"})
            state_rows.append({
                "case": case_id,
                "branch_step": branch_step,
                "informative": False,
                "skipped_noninformative": True,
                "reason": "H15_prefix_terminated_before_branch",
                "comparisons": [],
            })
            continue
        ref_state = state_tuple(ref, "branch_previous_state")
        comparisons: List[Dict[str, Any]] = []
        material_horizons: List[int] = []
        best_physical: Optional[Dict[str, Any]] = None
        best_total: Optional[Dict[str, Any]] = None
        for h in BRANCH_HORIZONS:
            row = by_h.get(h)
            if row is None:
                continue
            cand_state = state_tuple(row, "branch_previous_state") if bool(row.get("branch_reached")) else (float("nan"), float("nan"), float("nan"))
            state_dist = state_distance(cand_state, ref_state)
            phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - row["continuation_physical_constraint_cost_from_branch"]) if bool(row.get("branch_reached")) else float("nan")
            total_gain = float(ref["continuation_total_cost_from_branch"] - row["continuation_total_cost_from_branch"]) if bool(row.get("branch_reached")) else float("nan")
            safe = bool(row.get("branch_reached") and no_regression(row, ref))
            material = bool(h != PREFIX_H and safe and (phys_gain >= IMPROVEMENT_ABS_THRESHOLD or total_gain >= IMPROVEMENT_ABS_THRESHOLD))
            if material:
                material_horizons.append(h)
                positive_horizon_counts[str(h)] = positive_horizon_counts.get(str(h), 0) + 1
            comp = {
                "horizon": h,
                "branch_reached": bool(row.get("branch_reached")),
                "success": bool(row.get("success")),
                "constraint": bool(row.get("constraint")),
                "steps": int(row.get("steps", 0)),
                "continuation_physical": float(row.get("continuation_physical_constraint_cost_from_branch", 0.0)),
                "continuation_total": float(row.get("continuation_total_cost_from_branch", 0.0)),
                "gain_vs_H15_physical": phys_gain,
                "gain_vs_H15_total": total_gain,
                "decision_sum_s": float(row["decision_timing_s"]["sum"]),
                "decision_plus_terminal_switch_sum_s": float(row["decision_plus_terminal_switch_timing_s"]["sum"]),
                "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
                "state_distance_vs_H15_branch_state": state_dist,
                "prefix_clean_sha256_matches_H15": bool(row.get("prefix_clean_sha256") == ref.get("prefix_clean_sha256")),
                "no_success_constraint_solver_regression_vs_H15": safe,
                "material_positive_vs_H15": material,
                "path": row.get("path"),
            }
            comparisons.append(comp)
            if bool(row.get("branch_reached")):
                if best_physical is None or (comp["continuation_physical"], comp["decision_sum_s"], h) < (best_physical["continuation_physical"], best_physical["decision_sum_s"], best_physical["horizon"]):
                    best_physical = comp
                if best_total is None or (comp["continuation_total"], comp["decision_sum_s"], h) < (best_total["continuation_total"], best_total["decision_sum_s"], best_total["horizon"]):
                    best_total = comp
        is_positive = bool(material_horizons)
        if is_positive:
            positive_state_count += 1
            positive_cases.append(case_id)
        else:
            negative_neutral_state_count += 1
        state_rows.append({
            "case": case_id,
            "branch_step": branch_step,
            "informative": True,
            "skipped_noninformative": False,
            "reference_H15": {
                "success": bool(ref.get("success")),
                "constraint": bool(ref.get("constraint")),
                "continuation_physical": float(ref["continuation_physical_constraint_cost_from_branch"]),
                "continuation_total": float(ref["continuation_total_cost_from_branch"]),
                "steps": int(ref.get("steps", 0)),
                "initial_failed_steps": int(ref.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(ref.get("solver_failure_steps", 0)),
                "branch_previous_state": ref.get("branch_previous_state"),
            },
            "best_physical": best_physical,
            "best_total": best_total,
            "material_positive_state": is_positive,
            "material_positive_horizons": material_horizons,
            "label": "positive_non_H15" if is_positive else "negative_or_neutral",
            "comparisons": comparisons,
        })
    distinct_positive_cases = sorted(set(positive_cases))
    informative_state_count = int(sum(1 for r in state_rows if r.get("informative")))
    positives_span_three_cases = len(distinct_positive_cases) >= 3
    gate_pass = bool(
        positive_state_count >= 4
        and positives_span_three_cases
        and negative_neutral_state_count >= 4
        and not missing_reference
    )
    return {
        "state_count_total": len(state_rows),
        "informative_state_count": informative_state_count,
        "skipped_noninformative_state_count": len(skipped_states),
        "skipped_noninformative_states": skipped_states,
        "missing_H15_reference": missing_reference,
        "positive_state_count": int(positive_state_count),
        "negative_neutral_state_count": int(negative_neutral_state_count),
        "positive_cases": distinct_positive_cases,
        "positive_horizon_counts": positive_horizon_counts,
        "positives_span_at_least_three_cases": positives_span_three_cases,
        "fresh_refit_training_gate_pass": gate_pass,
        "gate_rule": "at least 4 material-positive states spanning at least 3 selected cases, plus at least 4 negative/neutral states, with no evidence positives are one exact case-specific cluster",
        "material_positive_rule": "non-H15 branch horizon succeeds, has no constraint/solver/initial-failure regression versus H15 from the identical prefix state, and improves continuation physical or total cost by >=3 absolute units",
        "state_rows": state_rows,
    }


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = str(raw["created_utc"]).replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_FRESH_CONTINUATION_LABEL_PROBE_V0_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup fresh continuation-label probe before any refit/training or validation work",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "fresh_candidate_resets": raw["budget_actual"]["fresh_bank_candidate_resets"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_MD), rel(PROTOCOL_JSON)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    lines = [
        "# Vehicle V1 fresh continuation-label probe v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development-only fresh identical-prefix continuation-label probe. No online adaptive selector, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Fresh candidate resets: `{raw['budget_actual']['fresh_bank_candidate_resets']}` / declared `48`.",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / frozen max `{raw['budget_declared']['max_rollout_episodes']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / frozen upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Selected fresh cases",
        "",
        "| case | source candidate | stratum | theta_r | traj_steps | branch steps | clearance | selection reason |",
        "|---:|---:|---|---:|---:|---|---:|---|",
    ]
    for i, meta in enumerate(raw["bank_selection"]["selected_metadata"]):
        lines.append("| %d | %d | `%s` | %.6g | %d | `%s` | %.6g | `%s` |" % (
            i,
            int(meta["candidate_index"]),
            meta.get("stratum"),
            float(meta.get("theta_r", float("nan"))),
            int(meta.get("traj_steps", 0)),
            branch_steps_for_meta(meta),
            float(meta.get("min_reference_obstacle_clearance", float("nan"))),
            meta.get("selection_reason"),
        ))
    lines += [
        "",
        "## Label gate result",
        "",
        f"- informative states: `{analysis['informative_state_count']}`; skipped noninformative states: `{analysis['skipped_noninformative_state_count']}`.",
        f"- positive states: `{analysis['positive_state_count']}` across cases `{analysis['positive_cases']}`; negative/neutral states: `{analysis['negative_neutral_state_count']}`.",
        f"- positive horizon counts: `{analysis['positive_horizon_counts']}`.",
        f"- fresh_refit_training_gate_pass: `{analysis['fresh_refit_training_gate_pass']}`.",
        f"- Gate rule: {analysis['gate_rule']}",
        "",
        "## Per-state best continuation labels",
        "",
        "| case | branch step | label | H15 cont phys | H15 cont total | best phys H | phys gain | best total H | total gain | material horizons |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in analysis["state_rows"]:
        if not row.get("informative"):
            lines.append("| %d | %d | `skipped_noninformative` | NA | NA | NA | NA | NA | NA | `[]` |" % (int(row["case"]), int(row["branch_step"])))
            continue
        ref = row["reference_H15"]
        bp = row.get("best_physical") or {}
        bt = row.get("best_total") or {}
        lines.append("| %d | %d | `%s` | %.6g | %.6g | %s | %.6g | %s | %.6g | `%s` |" % (
            int(row["case"]),
            int(row["branch_step"]),
            row.get("label"),
            float(ref["continuation_physical"]),
            float(ref["continuation_total"]),
            str(bp.get("horizon")),
            float(bp.get("gain_vs_H15_physical", 0.0)),
            str(bt.get("horizon")),
            float(bt.get("gain_vs_H15_total", 0.0)),
            row.get("material_positive_horizons"),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if analysis["fresh_refit_training_gate_pass"]:
        lines.append("The fresh probe found enough reusable state-dependent continuation labels to justify freezing a compact supervised/refit or learned horizon-value intervention before broader validation.")
    else:
        lines.append("The fresh probe did not meet the frozen label-density gate; next work should prioritize scenario-opportunity/objective/terminal-value diagnosis before validating another sparse prototype selector.")
    lines += ["", f"Backup request before further simulations: `{raw['backup_request']}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 fresh continuation-label probe v0\n\n"
        f"UTC: {raw['created_utc']}. Development-only fresh continuation-label probe completed: "
        f"{raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, "
        f"fresh candidate resets={raw['budget_actual']['fresh_bank_candidate_resets']}. No validation64/test access. "
        f"Positive states={analysis['positive_state_count']} across cases={analysis['positive_cases']}; "
        f"negative/neutral states={analysis['negative_neutral_state_count']}; refit/training gate={analysis['fresh_refit_training_gate_pass']}. "
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
        PROTOCOL_JSON,
        PROTOCOL_MD,
        POSTDIAG_RAW,
        POSTDIAG_DONE,
        V1_RAW,
        V1_DONE,
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
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after postdiagnostic, protocol and this runner source")
    ap.add_argument("--i-accept-fresh-continuation-label-development-rollout", action="store_true", help="explicit acknowledgement: development label probe only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_fresh_continuation_label_development_rollout:
        raise ContractError("explicit --i-accept-fresh-continuation-label-development-rollout is required")
    assert_fresh_output_dir()
    protocol, post_raw, v1_raw = verify_inputs()
    min_backup_time = latest_time(parse_time(protocol.get("created_utc")), parse_time(post_raw.get("created_utc")), source_mtime_utc())
    backup = verify_backup_proof(args.backup_proof, min_backup_time)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {
        "started_utc": started,
        "pid": os.getpid(),
        "method": "vehicle_v1_fresh_continuation_label_probe_v0",
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "training_gradient_steps": 0,
    })
    preflight = base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(str(preflight.get("diagnosis", "runtime preflight failed")) + " " + str(preflight.get("exception", "")))
    v1.latency_verify()
    bank_info = generate_bank_if_needed()
    bank = read_json(BANK_PATH)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    terminals, terminal_receipts = base.load_terminal_grid(v1_raw["protocol_full"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    schedule = build_schedule(selected_meta)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "branch_horizons": BRANCH_HORIZONS, "prefix_horizon": PREFIX_H})
    episodes: List[Dict[str, Any]] = []
    for item in schedule:
        case_id = int(item["case"])
        summary = run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals, terminal_receipts)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "case", "branch_step", "branch_horizon", "steps", "success", "termination", "branch_reached")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    budget = protocol["rollout_design"]
    if len(episodes) > int(budget["max_rollout_episodes"]) or control_steps > int(budget["control_step_upper_bound"]):
        raise ContractError("fresh continuation-label budget violation")
    analysis = analyze_labels(episodes)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_v1_fresh_continuation_label_probe_v0_identical_H15_prefix_branch_fixed_H",
        "classification": "development_IMPROVED_fresh_state_continuation_label_probe_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"path": rel(PROTOCOL_MD), "sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "protocol_full": protocol,
        "postdiagnostic": {"raw": rel(POSTDIAG_RAW), "raw_sha256": sha256(POSTDIAG_RAW), "completed": rel(POSTDIAG_DONE), "completed_sha256": sha256(POSTDIAG_DONE)},
        "v1_source": {"raw": rel(V1_RAW), "raw_sha256": sha256(V1_RAW), "completed": rel(V1_DONE), "completed_sha256": sha256(V1_DONE)},
        "bank": bank_info,
        "bank_selection": bank["selection"],
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"max_rollout_episodes": int(budget["max_rollout_episodes"]), "control_step_upper_bound": int(budget["control_step_upper_bound"]), "new_training_episodes": 0, "new_gradient_steps": 0, "fresh_candidate_resets": CANDIDATE_RESETS},
        "budget_actual": {"fresh_bank_candidate_resets": CANDIDATE_RESETS if bank_info.get("created_now") else 0, "episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": [
            "development diagnostic only",
            "fresh source-supported bank selected by reset metadata only",
            "common H15 prefix replayed independently for each branch arm",
            "not online adaptive selector validation",
            "not new training/refit",
            "not ORIGINAL SAC",
            "not final test",
            "actual timing from AWS run but terminal-switch overhead is reported separately",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 fresh continuation-label probe v0 state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} episodes / {control_steps} control steps and {raw['budget_actual']['fresh_bank_candidate_resets']} fresh candidate resets. "
        f"No validation64/test access. Positive states={analysis['positive_state_count']} across cases={analysis['positive_cases']}; "
        f"negative/neutral states={analysis['negative_neutral_state_count']}; refit/training gate={analysis['fresh_refit_training_gate_pass']}. Backup required before next simulation.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON, POSTDIAG_DONE, POSTDIAG_RAW]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "fresh_candidate_resets": raw["budget_actual"]["fresh_bank_candidate_resets"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": raw["backup_request"],
        "headline": {
            "positive_state_count": analysis["positive_state_count"],
            "negative_neutral_state_count": analysis["negative_neutral_state_count"],
            "positive_cases": analysis["positive_cases"],
            "fresh_refit_training_gate_pass": analysis["fresh_refit_training_gate_pass"],
            "next_action": "freeze compact refit/training if gate passed; otherwise scenario/objective/terminal diagnosis",
        },
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "episodes": len(episodes),
        "control_steps": control_steps,
        "fresh_candidate_resets": raw["budget_actual"]["fresh_bank_candidate_resets"],
        "positive_state_count": analysis["positive_state_count"],
        "negative_neutral_state_count": analysis["negative_neutral_state_count"],
        "positive_cases": analysis["positive_cases"],
        "fresh_refit_training_gate_pass": analysis["fresh_refit_training_gate_pass"],
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
            "next_recovery_hint": "Preserve partial directory, audit failure, and create a versioned one-variable repair. Use legacy interpreter and a verified backup proof after this runner source is externally backed up.",
        })
        raise
