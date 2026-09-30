#!/usr/bin/env python3
"""v34p / A13c-3 repaired non-converged objective-contract probe.

This is a thin operational-repair wrapper around v34n.  It preserves the active
Opus A13c-3 scientific design, formulas, cell list and <=6 low-level solver
attempt budget, while importing the v34o T-A60 strict previous_input loader and
fail-loud _u0 configuration repair.

Run only after v34o/T-A61 has passed and an external backup covering v34o/v34p
source plus artifacts has been verified.  No plant rollouts, env.step,
validation64, sealed-test access, training, or selector refit are allowed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0 as v34n
import vehicle_true_variable_horizon_v34o_loader_gate_v0 as v34o

ROOT = v34n.ROOT
NAME = "vehicle_true_variable_horizon_v34p_nonconverged_objective_contract_probe_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T113021Z_20dfa3.md"
OPUS_REPORT_SHA = "251993db94a04f38d22747754de141a97fd39ea7008ebd48b901ec382747d959"
OPUS_REQUEST = "execution-result:20260930T112938_862dec3e"
V34M_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34m_residual_attribution_v0_20260930T111306Z/completed.json"
V34N_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34n_nonconverged_objective_contract_probe_v0_20260930T112938Z/failed.json"


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    hh = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hh.update(chunk)
    return hh.hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def latest_v34o_completed() -> Path:
    pattern = str(ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34o_loader_gate_v0_*/completed.json")
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise ContractError("v34o/T-A61 completed.json not found; run loader gate first")
    return paths[-1]


def patched_verify_gates(args: argparse.Namespace) -> Dict[str, Any]:
    for p in [v34n.PLAN_READY, OPUS_REPORT, V34M_DONE, V34N_FAILED]:
        if not p.exists():
            raise ContractError("missing prerequisite " + rel(p))
    ready = read_json(v34n.PLAN_READY)
    if ready.get("request_id") != OPUS_REQUEST or ready.get("report_sha256") != OPUS_REPORT_SHA:
        raise ContractError(f"active PLAN_READY mismatch: request={ready.get('request_id')} sha={ready.get('report_sha256')}")
    if sha256(OPUS_REPORT) != OPUS_REPORT_SHA:
        raise ContractError("active Opus report sha mismatch")
    latest_text = v34n.OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if v34n.OPUS_LATEST.exists() else ""
    if rel(OPUS_REPORT) not in latest_text:
        raise ContractError("Opus LATEST.md does not point to active report")
    v34m = read_json(V34M_DONE)
    if v34m.get("hard_pass") is not True:
        raise ContractError("v34m predecessor did not hard_pass")
    v34n_fail = read_json(V34N_FAILED)
    if int(v34n_fail.get("new_solver_calls_recorded", 0)) != 0:
        raise ContractError("v34n predecessor unexpectedly spent solver calls")
    v34o_done_path = latest_v34o_completed()
    v34o_done = read_json(v34o_done_path)
    if v34o_done.get("hard_pass") is not True or v34o_done.get("passed") is not True:
        raise ContractError("latest v34o/T-A61 loader gate did not pass")
    b0 = v34o_done.get("budget_actual") or {}
    for key in ["solver_calls", "plant_steps", "env_step_calls_after_construction", "env_reset_calls_after_construction", "new_training_or_gradient_steps", "selector_refits", "validation64_episodes", "sealed_test_episodes"]:
        if int(b0.get(key, 0)) != 0:
            raise ContractError(f"v34o loader gate budget was not zero for {key}: {b0.get(key)}")
    bt = parse_time(args.backup_time)
    if bt is None:
        raise ContractError("backup_time not parseable")
    t_v34o = parse_time(v34o_done.get("created_utc"))
    if t_v34o is not None and bt <= t_v34o:
        raise ContractError("backup context must postdate v34o loader gate before v34p solver calls")
    proof = {
        "status": "verified_from_supervisor_context_not_revalidated_by_script",
        "time": bt.isoformat(),
        "commit": args.backup_commit,
        "remaining_changed_files": 0,
        "packages_this_run": [{"sha256": args.backup_package_sha256, "verification": "user_context_verified_backup", "bytes": int(args.backup_package_bytes)}],
        "source": "supervisor_user_context_current_prompt",
        "purpose": "gate v34p/A13c-3 <=6 low-level solver objective-contract probe after v34o loader gate",
    }
    proof_path = v34n.BACKUP_REQUEST.parent / f"backup_proof_{v34n.STAMP}_from_user_context_before_v34p_a13c3_probe.json"
    write_json(proof_path, proof)
    return {
        "active_lead_plan_ready": rel(v34n.PLAN_READY),
        "active_lead_report": rel(OPUS_REPORT),
        "active_lead_report_sha256": OPUS_REPORT_SHA,
        "active_lead_request": OPUS_REQUEST,
        "v34m_completed": rel(V34M_DONE),
        "v34n_failed_preserved": rel(V34N_FAILED),
        "v34o_loader_gate_completed": rel(v34o_done_path),
        "v34o_headline": v34o_done.get("headline"),
        "backup_proof": {**proof, "path": rel(proof_path), "sha256": sha256(proof_path)},
        "authorized_budget": {"low_level_solver_attempt_cap": v34n.SOLVE_CAP, "scheduled_cells": len(v34n.CELLS), "plant_steps": 0, "env_step_calls_after_construction": 0, "validation64_episodes": 0, "sealed_test_episodes": 0, "training_or_refit": 0},
    }


def patch_module_identity_and_contracts() -> None:
    # Unique artifact identity for this repaired execution.
    v34n.__file__ = __file__
    v34n.NAME = NAME
    v34n.STAMP = STAMP
    v34n.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    v34n.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34p_nonconverged_objective_contract_probe.md"
    v34n.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34P_NONCONVERGED_OBJECTIVE_CONTRACT_PROBE_{STAMP}.json"
    v34n.REQUEST_ID = f"v34p-a13c3-nonconverged-objective-contract-{STAMP}"
    v34n.MARKER = f"vehicle-v34p-a13c3-nonconverged-objective-contract-{STAMP}"

    # Current active Opus plan supersedes the failed v34n predecessor plan.
    v34n.OPUS_REPORT = OPUS_REPORT
    v34n.OPUS_REPORT_SHA = OPUS_REPORT_SHA
    v34n.OPUS_REQUEST = OPUS_REQUEST
    v34n.V34M_DONE = V34M_DONE

    # Reset per-process counters defensively in case this wrapper is imported in
    # an interactive process.
    v34n._CAPTURES.clear()
    v34n._PATCHED_MODULES.clear()
    v34n._CURRENT_ARM_ID = None
    v34n._TOTAL_SOLVER_CALLS = 0

    # T-A60 operational repair from v34o.
    v34n.base.load_contexts = v34o.load_contexts_strict
    v34n.base.configure_context_no_reset = v34o.configure_context_no_reset_strict
    v34n.verify_gates = patched_verify_gates  # type: ignore[assignment]


def run(argv: Optional[Sequence[str]] = None) -> int:
    patch_module_identity_and_contracts()
    return v34n.run(argv)


if __name__ == "__main__":
    raise SystemExit(run())
