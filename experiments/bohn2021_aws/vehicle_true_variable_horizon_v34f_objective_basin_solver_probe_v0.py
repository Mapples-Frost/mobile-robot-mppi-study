#!/usr/bin/env python3
"""v34f wrapper: Opus-approved backup-gate repair for objective-vs-basin probe.

This is a narrow operational repair after v34d failed before any solver arm
because it required the *current* backup commit/package to equal one older
v34c/v0e proof.  The repaired gate follows the active Opus scientific-lead plan:

  * verify the active Opus plan/report, Task1, v33 evidence, and v34c/v0e
    zero-solve preflight;
  * scan research_artifacts/aws_backup_proofs/backup_proof_*.json;
  * require a verified proof with remaining_changed_files==0, proof.time at or
    after the v34c/v0e completed evidence, and commit/package sha256/bytes that
    match the command-line backup arguments;
  * fail closed with candidate diagnostics if no such proof exists.

The scientific solver design is intentionally unchanged from v34d/v34:
2 opened development contexts x 2 horizons x 3 terminal modes x 2 deterministic
initializations = 24 low-level solver attempts for the full probe, with 0 plant
steps, no env.reset/env.step after construction, no selector search/refit, no
training, no validation64, and no sealed/final test access.

This wrapper does not mutate completed.json after final hashing.  It writes a
provenance sidecar before delegating to the parent run path so the sidecar is
included in the parent completed-hash inventory.
"""
from __future__ import annotations

import datetime as dt
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0 as d

base = d.base
ROOT = d.ROOT
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NAME = "vehicle_true_variable_horizon_v34f_objective_basin_solver_probe_v0"
REQUEST_ID = f"v34f-objective-basin-solver-probe-{STAMP}"
MARKER = f"vehicle-v34f-objective-basin-solver-probe-{STAMP}"

OPUS_PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_LATEST = ROOT / "docs/bohn2021_takeover/opus_lead/LATEST.md"
CURRENT_OPUS_REQUEST = "execution-result:20260930T094857_bf0b0af1"
CURRENT_OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T094740Z_3d9a1e.md"
CURRENT_OPUS_SHA = "efb65f58271438a4eb3331cae954373b554185f375525f6162e7628ceac5f840"
BACKUP_PROOF_GLOB = ROOT / "research_artifacts/aws_backup_proofs" / "backup_proof_*.json"


def _parse_time(value: Any) -> Optional[dt.datetime]:
    try:
        return base.parse_time(value)
    except Exception:
        pass
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _proof_packages(proof: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    pkgs = proof.get("packages_this_run")
    if isinstance(pkgs, list):
        return [p for p in pkgs if isinstance(p, Mapping)]
    if proof.get("package_sha256"):
        return [{"sha256": proof.get("package_sha256"), "bytes": proof.get("package_bytes")}]
    return []


def _proof_summary(path: Path, proof: Mapping[str, Any], min_time: Optional[dt.datetime], args: Namespace) -> Dict[str, Any]:
    t = _parse_time(proof.get("time"))
    packages = _proof_packages(proof)
    package_matches = []
    for p in packages:
        sha_match = p.get("sha256") == args.backup_package_sha256
        bytes_value = _safe_int(p.get("bytes"), None)
        bytes_match = bytes_value == int(getattr(args, "backup_package_bytes", 0) or 0)
        package_matches.append({
            "name": p.get("name"),
            "sha256": p.get("sha256"),
            "bytes": bytes_value,
            "sha256_match": bool(sha_match),
            "bytes_match": bool(bytes_match),
        })
    return {
        "path": base.rel(path),
        "status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "time": proof.get("time"),
        "time_parseable": t is not None,
        "time_ge_min": (t is not None and (min_time is None or t >= min_time)),
        "commit": proof.get("commit"),
        "commit_match": proof.get("commit") == args.backup_commit,
        "packages": package_matches,
        "source": proof.get("source"),
        "purpose": proof.get("purpose"),
    }


def scan_verified_backup_proofs(args: Namespace, min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    """Find the single proof that matches current backup CLI arguments."""
    candidates: List[Dict[str, Any]] = []
    matches: List[Tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = []
    for path in sorted(BACKUP_PROOF_GLOB.parent.glob(BACKUP_PROOF_GLOB.name)):
        try:
            proof = base.read_json(path)
        except Exception as exc:
            candidates.append({"path": base.rel(path), "read_error": repr(exc)})
            continue
        if not isinstance(proof, Mapping):
            candidates.append({"path": base.rel(path), "read_error": "not a JSON object"})
            continue
        summary = _proof_summary(path, proof, min_time, args)
        candidates.append(summary)
        t = _parse_time(proof.get("time"))
        if proof.get("status") != "verified" or proof.get("backup_verified") is not True:
            continue
        if _safe_int(proof.get("remaining_changed_files"), -1) != 0:
            continue
        if t is None or (min_time is not None and t < min_time):
            continue
        if proof.get("commit") != args.backup_commit:
            continue
        matched_pkg = None
        for p in _proof_packages(proof):
            if p.get("sha256") != args.backup_package_sha256:
                continue
            if _safe_int(p.get("bytes"), None) != int(getattr(args, "backup_package_bytes", 0) or 0):
                continue
            matched_pkg = p
            break
        if matched_pkg is None:
            continue
        matches.append((path, proof, matched_pkg))
    if not matches:
        recent = candidates[-12:]
        raise base.ContractError(
            "no verified backup proof matched current backup args and v0e time gate; "
            + base.json.dumps({
                "required_commit": args.backup_commit,
                "required_package_sha256": args.backup_package_sha256,
                "required_package_bytes": int(getattr(args, "backup_package_bytes", 0) or 0),
                "min_time": None if min_time is None else min_time.isoformat(),
                "candidate_count": len(candidates),
                "recent_candidate_summaries": recent,
            }, sort_keys=True)
        )
    path, proof, pkg = matches[-1]
    return {
        "matched": True,
        "path": base.rel(path),
        "sha256": base.sha256(path),
        "time": proof.get("time"),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "status": proof.get("status"),
        "backup_verified": proof.get("backup_verified"),
        "package": {"name": pkg.get("name"), "sha256": pkg.get("sha256"), "bytes": _safe_int(pkg.get("bytes"), None), "verification": pkg.get("verification")},
        "matched_candidate_count": len(matches),
        "verified_candidate_count": sum(1 for c in candidates if c.get("status") == "verified" and c.get("backup_verified") is True),
        "candidate_count": len(candidates),
    }


def verify_opus_plan_gate() -> Dict[str, Any]:
    if not OPUS_PLAN_READY.exists():
        raise base.ContractError("missing active Opus PLAN_READY.json")
    ready = base.read_json(OPUS_PLAN_READY)
    if ready.get("request_id") != CURRENT_OPUS_REQUEST:
        raise base.ContractError("Opus PLAN_READY request mismatch: %r != %r" % (ready.get("request_id"), CURRENT_OPUS_REQUEST))
    report_path = ROOT / str(ready.get("report", ""))
    if report_path.resolve() != CURRENT_OPUS_REPORT.resolve():
        raise base.ContractError("Opus PLAN_READY report path mismatch: %s" % ready.get("report"))
    if ready.get("report_sha256") != CURRENT_OPUS_SHA:
        raise base.ContractError("Opus PLAN_READY recorded sha mismatch")
    if not CURRENT_OPUS_REPORT.exists() or base.sha256(CURRENT_OPUS_REPORT) != CURRENT_OPUS_SHA:
        raise base.ContractError("active Opus report missing or sha mismatch")
    latest_text = OPUS_LATEST.read_text(encoding="utf-8", errors="replace") if OPUS_LATEST.exists() else ""
    if base.rel(CURRENT_OPUS_REPORT) not in latest_text:
        raise base.ContractError("Opus LATEST.md does not point to active report")
    return {
        "plan_ready": base.rel(OPUS_PLAN_READY),
        "latest": base.rel(OPUS_LATEST),
        "request_id": ready.get("request_id"),
        "experiment_id": ready.get("experiment_id"),
        "primary_analyst": ready.get("primary_analyst"),
        "report": base.rel(CURRENT_OPUS_REPORT),
        "report_sha256": CURRENT_OPUS_SHA,
        "completed": ready.get("completed"),
        "supersedes_request_ids": ready.get("supersedes_request_ids"),
    }


def verify_task_prerequisites() -> Dict[str, Any]:
    t1 = base.completed_ok(base.TASK1_DONE, "Task1 v33 bookkeeping preflight")
    t1_budget = t1.get("budget_actual") or {}
    if int(t1_budget.get("solver_calls", 0)) != 0 or int(t1_budget.get("plant_steps", 0)) != 0:
        raise base.ContractError("Task1 bookkeeping preflight did not have zero solver/plant budget")
    d33 = base.completed_ok(base.D33_DONE, "v33 terminal-H cross probe")
    v0e = base.read_json(d.V0E_DONE)
    if v0e.get("hard_pass") is not True:
        raise base.ContractError("v34c/v0e contract preflight did not hard-pass")
    b = v0e.get("budget_actual") or {}
    if (
        int(b.get("lower_level_solver_calls", 0)) != 0
        or int(b.get("plant_steps", 0)) != 0
        or int(b.get("training_or_refit", 0)) != 0
        or int(b.get("validation64_episodes", 0)) != 0
        or int(b.get("sealed_test_episodes", 0)) != 0
    ):
        raise base.ContractError("v34c/v0e contract preflight budget violated zero-solve/zero-plant/no-validation/no-test contract")
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "test_accessed"):
        if v0e.get(flag) is True:
            raise base.ContractError("v34c/v0e preflight has forbidden %s=true" % flag)
    return {
        "task1_completed": base.rel(base.TASK1_DONE),
        "task1_headline": t1.get("headline"),
        "v33_completed": base.rel(base.D33_DONE),
        "v33_headline": d33.get("headline"),
        "v34c_v0e_completed": base.rel(d.V0E_DONE),
        "v34c_v0e_created_utc": v0e.get("created_utc"),
        "v34c_v0e_hard_pass": True,
        "v34c_v0e_budget_actual": b,
        "v34c_v0e_headline": v0e.get("headline"),
    }


def verify_opus_and_backup_gates(args: Namespace) -> Dict[str, Any]:
    if not (args.backup_time and args.backup_commit and args.backup_package_sha256):
        raise base.ContractError("verified backup context args are required before v34f solver evidence")
    current_backup_time = _parse_time(args.backup_time)
    if current_backup_time is None:
        raise base.ContractError("backup_time not parseable")
    lead = verify_opus_plan_gate()
    prereq = verify_task_prerequisites()
    v0e_time = _parse_time(prereq.get("v34c_v0e_created_utc"))
    if v0e_time is not None and current_backup_time < v0e_time:
        raise base.ContractError("current backup_time predates v34c/v0e completed evidence")
    proof = scan_verified_backup_proofs(args, v0e_time)
    gates = {
        "active_scientific_lead_gate": lead,
        "task_prerequisites": prereq,
        "current_backup_arg": {
            "time": args.backup_time,
            "commit": args.backup_commit,
            "package_sha256": args.backup_package_sha256,
            "package_bytes": int(getattr(args, "backup_package_bytes", 0) or 0),
        },
        "matched_verified_backup_proof": proof,
        "backup_gate_repair": {
            "stable_id": "A12_registry_backup_schema_contract/backup-proof-scan-match",
            "prior_failed_run": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34d_objective_basin_solver_probe_v0_20260930T094857Z/failed.json",
            "prior_failure": "v34d hard-coded one older v0e proof and rejected a newer valid pre-run backup before any solver arm",
            "repair": "scan backup_proof_*.json and match status/remaining_changed_files/time/commit/package_sha256/package_bytes against current CLI args",
            "fail_closed_without_proof": True,
            "scientific_design_changed": False,
            "solver_budget_cap_unchanged": base.MAX_SOLVES,
        },
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    return gates


def patch_runtime() -> None:
    """Apply v34c/v0e operational repairs and the v34f Opus/backup gate."""
    d.patch_runtime()
    wrapper = Path(__file__).resolve()
    base.NAME = NAME
    base.STAMP = STAMP
    base.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
    base.STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34f_objective_basin_solver_probe.md"
    base.BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34F_OBJECTIVE_BASIN_SOLVER_PROBE_{STAMP}.json"
    base.REQUEST_ID = REQUEST_ID
    base.MARKER = MARKER
    base.ANALYSIS_READY = OPUS_PLAN_READY
    base.ASTRA_REPORT = CURRENT_OPUS_REPORT
    base.CURRENT_ASTRA_REQUEST = CURRENT_OPUS_REQUEST
    base.CURRENT_ASTRA_SHA = CURRENT_OPUS_SHA
    base.verify_astra_and_gates = verify_opus_and_backup_gates
    # base.run uses its module global __file__ in the completed hash and backup
    # request.  Point it to this wrapper so the actually executed source is
    # backed up; the underlying base/d/contract hashes are in the sidecar.
    base.__file__ = str(wrapper)


def write_static_provenance() -> Path:
    base.RUN_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).resolve()
    provenance = {
        "created_utc": base.now_utc().isoformat(),
        "wrapper_script": base.rel(wrapper),
        "wrapper_script_sha256": base.sha256(wrapper),
        "parent_wrapper_script": base.rel(Path(d.__file__).resolve()),
        "parent_wrapper_script_sha256": base.sha256(Path(d.__file__).resolve()),
        "base_script": base.rel(Path(base.__file__).resolve()) if Path(str(base.__file__)).resolve() == wrapper else "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py",
        "actual_base_script": "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py",
        "actual_base_script_sha256": base.sha256(ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_v34_objective_basin_solver_probe_v0.py"),
        "contract_preflight_wrapper": base.rel(Path(d.contract.__file__).resolve()),
        "contract_preflight_wrapper_sha256": base.sha256(Path(d.contract.__file__).resolve()),
        "contract_preflight_parent_wrapper": base.rel(Path(d.contract_d.__file__).resolve()),
        "contract_preflight_parent_wrapper_sha256": base.sha256(Path(d.contract_d.__file__).resolve()),
        "lead_report": base.rel(CURRENT_OPUS_REPORT),
        "lead_report_sha256": CURRENT_OPUS_SHA,
        "repair_scope": "backup proof scan/match gate plus provenance-sidecar discipline; no scientific design change",
        "scientific_design_changed": False,
        "solver_budget_cap": base.MAX_SOLVES,
        "plant_steps_added": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "completed_json_posthash_mutation": False,
    }
    out = base.RUN_DIR / "v34f_wrapper_provenance.json"
    base.write_json(out, provenance)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    patch_runtime()
    write_static_provenance()
    return int(base.run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
