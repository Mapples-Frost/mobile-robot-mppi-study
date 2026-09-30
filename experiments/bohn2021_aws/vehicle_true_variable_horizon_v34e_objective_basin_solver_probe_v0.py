#!/usr/bin/env python3
"""v34e wrapper: backup-gate repair for the fixed 24-call objective-vs-basin probe.

This is a narrow operational repair after v34d failed before any solver arm with

    ContractError('backup commit arg does not match v0e backup proof')

The failure was a gate implementation error: the v34d wrapper required the
*current* pre-run backup arguments to equal the earlier v34c/v0e proof.  The
scientific contract only requires (1) the v34c/v0e zero-solve contract preflight
hard-passed, (2) that preflight has a verified backup proof, and (3) the current
solver-probe launch is itself preceded by a verified backup context.  A later
backup that covers the thin wrapper and prior artifacts is strictly stronger and
must not be rejected for having a newer commit/package hash.

The scientific design is unchanged from Astra Task2 and v34d:
  - 2 opened development contexts x 2 horizons x 3 terminal modes x 2
    initializations = exactly 24 low-level solver attempts if the run reaches the
    arm loop;
  - 0 plant steps, no env.reset/env.step after construction, no selector
    search/refit/training, no validation64 and no sealed/final test access;
  - v34c/v0e direct-context, numeric TVP, typed initialization/readback and vf4
    primitives remain in use through the v34d parent wrapper.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0 as d

base = d.base
ROOT = d.ROOT
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34e_objective_basin_solver_probe_v0"
REQUEST_ID = f"v34e-objective-basin-solver-probe-{STAMP}"
MARKER = f"vehicle-v34e-objective-basin-solver-probe-{STAMP}"


def _parse_time(value: Any) -> Optional[dt.datetime]:
    try:
        return base.parse_time(value)
    except Exception:
        pass
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def verify_gates_with_post_v0e_backup(args: Any) -> Dict[str, Any]:
    """Accept a verified current backup that postdates the v34c/v0e preflight.

    ``d.ORIGINAL_VERIFY`` is the original v34 gate: it verifies the current
    Astra report, Task1/D33 prerequisites, and records the supervisor-provided
    current backup context.  v34d then over-constrained this by requiring the
    current backup commit/package to equal the older v0e proof.  Here we instead
    check the older v0e proof independently and require the current backup time
    to be no earlier than v34c/v0e completion/proof time.
    """
    gates = d.ORIGINAL_VERIFY(args)

    v0e = base.read_json(d.V0E_DONE)
    if v0e.get("hard_pass") is not True:
        raise base.ContractError("v34c/v0e contract preflight did not hard-pass")
    b = v0e.get("budget_actual") or {}
    if (
        int(b.get("lower_level_solver_calls", 0)) != 0
        or int(b.get("plant_steps", 0)) != 0
        or int(b.get("training_or_refit", 0)) != 0
    ):
        raise base.ContractError("v34c/v0e contract preflight budget was not zero-solve/zero-plant/zero-training")

    v0e_proof = base.read_json(d.V0E_BACKUP_PROOF)
    if v0e_proof.get("status") != "verified" or v0e_proof.get("backup_verified") is not True:
        raise base.ContractError("missing verified supervisor backup proof after v34c/v0e preflight")

    current_backup_time = _parse_time(args.backup_time)
    v0e_done_time = _parse_time(v0e.get("created_utc") or v0e.get("completed") or v0e.get("completed_utc"))
    v0e_proof_time = _parse_time(v0e_proof.get("time"))
    if current_backup_time is None:
        raise base.ContractError("current backup_time not parseable in v34e gate")
    if v0e_done_time is not None and current_backup_time < v0e_done_time:
        raise base.ContractError("current backup context predates v34c/v0e completed evidence")
    if v0e_proof_time is not None and current_backup_time < v0e_proof_time:
        raise base.ContractError("current backup context predates the verified v34c/v0e backup proof")
    if not str(args.backup_commit or "").strip():
        raise base.ContractError("current backup_commit is required")
    if len(str(args.backup_package_sha256 or "")) < 32:
        raise base.ContractError("current backup_package_sha256 is missing/too short")

    gates["v34c_v0e_contract_preflight_gate"] = {
        "completed": base.rel(d.V0E_DONE),
        "hard_pass": True,
        "budget_actual": b,
        "v0e_backup_proof": base.rel(d.V0E_BACKUP_PROOF),
        "v0e_backup_proof_status": v0e_proof.get("status"),
        "v0e_backup_proof_time": v0e_proof.get("time"),
        "current_backup_time": args.backup_time,
        "current_backup_commit": args.backup_commit,
        "current_backup_package_sha256": args.backup_package_sha256,
        "current_backup_package_bytes": int(getattr(args, "backup_package_bytes", 0) or 0),
        "gate_repair": "v34e accepts a later verified current backup instead of requiring equality with the older v0e proof",
    }
    gates["v34e_operational_repair"] = {
        "prior_failed_run": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0_20260930T094857Z/failed.json",
        "prior_failure": "backup commit/package mismatch between newer current backup and older v34c/v0e proof; no solver arm reached",
        "scientific_design_changed": False,
        "solver_budget_cap_unchanged": base.MAX_SOLVES,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    return gates


def patch_runtime() -> None:
    d.patch_runtime()
    base.NAME = NAME
    base.STAMP = STAMP
    base.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    base.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34e_objective_basin_solver_probe.md"
    base.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34E_OBJECTIVE_BASIN_SOLVER_PROBE_{STAMP}.json"
    base.REQUEST_ID = REQUEST_ID
    base.MARKER = MARKER
    base.CURRENT_ASTRA_REQUEST = d.CURRENT_REQUEST
    base.ASTRA_REPORT = d.CURRENT_REPORT
    base.CURRENT_ASTRA_SHA = d.CURRENT_SHA
    base.verify_astra_and_gates = verify_gates_with_post_v0e_backup


def augment_provenance(exit_code: int) -> None:
    try:
        wrapper = Path(__file__).resolve()
        parent = Path(d.__file__).resolve()
        provenance = {
            "status": "completed" if exit_code == 0 else "failed",
            "created_utc": base.now_utc().isoformat(),
            "wrapper_script": base.rel(wrapper),
            "wrapper_script_sha256": base.sha256(wrapper),
            "parent_wrapper_script": base.rel(parent),
            "parent_wrapper_script_sha256": base.sha256(parent),
            "base_script": base.rel(Path(base.__file__).resolve()),
            "base_script_sha256": base.sha256(Path(base.__file__).resolve()),
            "contract_preflight_wrapper": base.rel(Path(d.contract.__file__).resolve()),
            "contract_preflight_wrapper_sha256": base.sha256(Path(d.contract.__file__).resolve()),
            "repair_scope": "backup-gate repair only; accept a later verified current backup while preserving v34d/v34c contract primitives and fixed 24-call design",
            "scientific_design_changed": False,
            "solver_budget_cap": base.MAX_SOLVES,
            "plant_steps_added": 0,
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        prov_path = base.RUN_DIR / "v34e_wrapper_provenance.json"
        base.write_json(prov_path, provenance)
        for name in ("raw.json", "completed.json", "failed.json"):
            p = base.RUN_DIR / name
            if not p.exists():
                continue
            obj = base.read_json(p)
            obj["v34e_wrapper_repair_provenance"] = {**provenance, "path": base.rel(prov_path)}
            if name == "completed.json":
                hashes = dict(obj.get("hashes") or {})
                for hp in (prov_path, wrapper, parent, Path(base.__file__).resolve(), Path(d.contract.__file__).resolve(), Path(d.contract_d.__file__).resolve()):
                    if hp.exists():
                        hashes[base.rel(hp)] = base.sha256(hp)
                obj["hashes"] = hashes
            base.write_json(p, obj)
        if base.BACKUP_REQUEST.exists():
            req = base.read_json(base.BACKUP_REQUEST)
            must = list(req.get("must_cover") or [])
            for add in [base.rel(wrapper), base.rel(parent), base.rel(prov_path)]:
                if add not in must:
                    must.append(add)
            req["must_cover"] = must
            req["v34e_wrapper_repair_provenance"] = {"path": base.rel(prov_path), "sha256": base.sha256(prov_path)}
            base.write_json(base.BACKUP_REQUEST, req)
    except Exception as exc:
        print({"v34e_provenance_augmentation_failed": repr(exc)}, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    patch_runtime()
    code = base.run(argv)
    augment_provenance(code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
