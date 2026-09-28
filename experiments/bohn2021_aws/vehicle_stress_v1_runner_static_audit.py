#!/usr/bin/env python3
"""Static post-dry-run audit for the vehicle stress-v1 Stage1 runner.

No simulations, no candidate resets, no training/refit, no validation64-bank access,
and no sealed-test access.  The purpose is to turn the new runner dry-run into a
machine-checked gate: are the runner, dry-run artifacts, frozen protocol and
backup state sufficient to launch the 200-episode Stage1 fixed-H map?
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T2040Z"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1_runner_static_audit_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1_runner_static_audit_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1_RUNNER_STATIC_AUDIT_{STAMP}.json"
MARKER = f"vehicle-stress-v1-runner-static-audit-{STAMP}"

RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_probe_v1_runner.py"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.md"
DRYRUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_runner_dryrun_20260928T2045Z"
DRYRUN_COMPLETED = DRYRUN_DIR / "completed.json"
DRYRUN_RAW = DRYRUN_DIR / "raw.json"
DRYRUN_SUMMARY = DRYRUN_DIR / "summary.md"
DRYRUN_STATE = ROOT / "research_artifacts/aws_state/vehicle_stress_scenario_opportunity_probe_v1_runner_dryrun_20260928T2045Z.md"
DRYRUN_REQ = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_V1_RUNNER_DRYRUN_20260928T2045Z.json"
PERSISTED_PRIOR_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260928T203153_from_supervisor_context_after_stress_v1_protocol.json"

EXPECTED = {
    "horizons": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "candidate_pool_resets": 256,
    "selected_cases": 20,
    "episodes_exact": 200,
    "control_step_upper_bound": 30000,
    "training_episodes": 0,
    "gradient_steps": 0,
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def proof_ok(path: Path, after: dt.datetime) -> Dict[str, Any]:
    try:
        obj = read_json(path)
    except Exception as exc:
        return {"path": rel(path), "ok": False, "reason": "unreadable:%r" % (exc,)}
    t = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        t = parse_time(obj.get(key))
        if t is not None:
            break
    packages = obj.get("packages_this_run") or []
    has_asset = bool(obj.get("asset_sha256") or obj.get("release_asset_sha256") or packages)
    ok = bool((obj.get("backup_verified") is True or obj.get("status") == "verified") and int(obj.get("remaining_changed_files", -1)) == 0 and obj.get("commit") and has_asset and t is not None and t >= after)
    return {
        "path": rel(path),
        "ok": ok,
        "time": t.isoformat() if t else None,
        "status": obj.get("status"),
        "backup_verified": obj.get("backup_verified"),
        "remaining_changed_files": obj.get("remaining_changed_files"),
        "commit": obj.get("commit"),
        "package_count": len(packages),
        "has_asset_or_package": has_asset,
        "postdates_required_time": bool(t is not None and t >= after),
    }


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required_files = [RUNNER, PROTOCOL_JSON, PROTOCOL_MD, DRYRUN_COMPLETED, DRYRUN_RAW, DRYRUN_SUMMARY, DRYRUN_STATE, DRYRUN_REQ, PERSISTED_PRIOR_PROOF]
    missing = [rel(p) for p in required_files if not p.exists()]
    if missing:
        raise RuntimeError("missing required inputs: " + ", ".join(missing))

    protocol = read_json(PROTOCOL_JSON)
    dry_done = read_json(DRYRUN_COMPLETED)
    dry_raw = read_json(DRYRUN_RAW)
    dry_req = read_json(DRYRUN_REQ)
    runner_sha = sha256(RUNNER)
    recorded_runner_sha = (dry_done.get("hashes") or {}).get(rel(RUNNER))
    dry_created = parse_time(dry_done.get("created_utc"))
    if dry_created is None:
        raise RuntimeError("dry-run completed marker lacks parseable created_utc")

    hash_failures: List[str] = []
    for name, expected in (dry_done.get("hashes") or {}).items():
        p = ROOT / name
        if not p.exists():
            hash_failures.append(name + ":missing")
        else:
            actual = sha256(p)
            if actual != expected:
                hash_failures.append(name + ":" + actual + "!=" + expected)

    stage1 = protocol.get("stage1_fixed_H_opportunity_map") or {}
    gen = protocol.get("scenario_generator_v1") or {}
    raw_budget = dry_raw.get("frozen_budget") or {}
    protocol_consistency = {
        "protocol_id": protocol.get("protocol_id"),
        "horizons_match": list(stage1.get("horizons") or []) == EXPECTED["horizons"] == list(raw_budget.get("horizons") or []),
        "candidate_resets_match": int(gen.get("candidate_pool_resets", -1)) == EXPECTED["candidate_pool_resets"] == int(raw_budget.get("candidate_pool_resets", -2)),
        "selected_cases_match": int(gen.get("selected_cases", -1)) == EXPECTED["selected_cases"] == int(raw_budget.get("selected_cases", -2)),
        "episodes_match": int(stage1.get("episodes_exact", -1)) == EXPECTED["episodes_exact"] == int(raw_budget.get("episodes_exact", -2)),
        "control_cap_match": int(stage1.get("control_step_upper_bound", -1)) == EXPECTED["control_step_upper_bound"] == int(raw_budget.get("control_step_upper_bound", -2)),
        "terminal_horizons_verified": sorted(int(x) for x in (dry_raw.get("terminal_metadata") or {}).keys()) == EXPECTED["horizons"],
        "access_flags_clean": dry_done.get("historical_validation64_bank_opened") is False and dry_done.get("sealed_test_accessed") is False and dry_raw.get("historical_validation64_bank_opened") is False and dry_raw.get("sealed_test_accessed") is False,
        "dryrun_no_rollouts": int(dry_done.get("new_rollouts", -1)) == 0 and int(dry_done.get("new_control_steps", -1)) == 0,
        "dryrun_backup_request_requires_backup": dry_req.get("backup_required_before_more_simulations") is True,
        "runner_sha_matches_completed": runner_sha == recorded_runner_sha,
        "dryrun_completed_hash_audit_pass": len(hash_failures) == 0,
    }
    all_consistent = all(bool(v) for k, v in protocol_consistency.items() if k != "protocol_id")

    proof_reports = [proof_ok(p, dry_created) for p in sorted((ROOT / "research_artifacts/aws_backup_proofs").glob("backup_proof_*.json"))]
    adequate = [p for p in proof_reports if p.get("ok")]
    rollout_blocked = len(adequate) == 0

    raw = {
        "created_utc": created,
        "method": "vehicle_stress_v1_runner_static_audit_after_dryrun_no_simulation",
        "classification": "development_static_gate_diagnostic_no_rollout_not_validation_not_final_test",
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
        "input_hashes": {rel(p): sha256(p) for p in required_files},
        "runner": {"path": rel(RUNNER), "sha256": runner_sha, "sha256_recorded_in_dryrun": recorded_runner_sha, "matches_dryrun_completed": runner_sha == recorded_runner_sha},
        "dryrun": {"completed": rel(DRYRUN_COMPLETED), "created_utc": dry_done.get("created_utc"), "backup_request": rel(DRYRUN_REQ), "hash_failures": hash_failures},
        "protocol_consistency": protocol_consistency,
        "all_static_checks_passed": all_consistent,
        "backup_gate": {
            "required_post_utc": dry_created.isoformat(),
            "adequate_post_dryrun_backup_count": len(adequate),
            "adequate_post_dryrun_backup_paths": [p["path"] for p in adequate],
            "rollout_blocked_until_backup": rollout_blocked,
            "proof_candidates": proof_reports[-20:],
        },
        "decision": "rollout_blocked_pending_post_dryrun_external_backup" if rollout_blocked else "inputs_sufficient_for_stage1_v1_rollout",
        "next_action": "request/verify external backup covering v1 runner source, dry-run outputs, this static audit and docs; then run one 200-episode Stage1 v1 fixed-H map" if rollout_blocked else "run one 200-episode Stage1 v1 fixed-H map under legacy interpreter",
    }
    write_json(OUT_DIR / "raw.json", raw)

    lines = [
        "# Vehicle stress-v1 runner static post-dry-run audit",
        "",
        f"UTC: `{created}`. No simulations, no candidate resets, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Static checks",
        "",
        f"- Runner SHA matches dry-run completed marker: `{protocol_consistency['runner_sha_matches_completed']}` (`{runner_sha}`).",
        f"- Dry-run artifact hash audit pass: `{protocol_consistency['dryrun_completed_hash_audit_pass']}`; failures: `{hash_failures}`.",
        f"- Protocol/budget constants match dry-run: horizons `{protocol_consistency['horizons_match']}`, candidate resets `{protocol_consistency['candidate_resets_match']}`, selected cases `{protocol_consistency['selected_cases_match']}`, episodes `{protocol_consistency['episodes_match']}`, cap `{protocol_consistency['control_cap_match']}`.",
        f"- Terminal horizons verified in dry-run: `{protocol_consistency['terminal_horizons_verified']}`.",
        f"- Access flags clean and dry-run used zero rollouts/control steps: `{protocol_consistency['access_flags_clean'] and protocol_consistency['dryrun_no_rollouts']}`.",
        "",
        "## Backup gate",
        "",
        f"- Required post-dry-run backup time: `{dry_created.isoformat()}`.",
        f"- Adequate post-dry-run backup proofs found: `{len(adequate)}`.",
        f"- Rollout blocked until backup: `{rollout_blocked}`.",
        "",
        "## Decision",
        "",
        f"`{raw['decision']}`. Next action: {raw['next_action']}.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle stress-v1 runner static audit state ({created})\n\n"
        f"Static checks passed={all_consistent}. Adequate post-dry-run backup proofs={len(adequate)}. "
        f"Rollout blocked until backup={rollout_blocked}. No simulations/control steps/training/validation/test.\n",
        encoding="utf-8",
    )

    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup stress-v1 runner static audit outputs and state before any Stage1 v1 rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(RUNNER), rel(DRYRUN_DIR), rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_REQ), rel(DRYRUN_REQ), rel(PERSISTED_PRIOR_PROOF)],
    })
    raw["backup_request"] = rel(BACKUP_REQ)
    write_json(OUT_DIR / "raw.json", raw)

    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-v1 runner static post-dry-run audit

UTC: {created}. No-simulation static audit completed after the stress-v1 runner dry-run. Runner/source hash, dry-run hashes, frozen protocol constants and terminal metadata checks passed={all_consistent}. Adequate post-dry-run external backup proofs found={len(adequate)}; rollout blocked until backup={rollout_blocked}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
"""
    append_docs(block)

    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, BACKUP_REQ, RUNNER, DRYRUN_COMPLETED, DRYRUN_RAW, DRYRUN_SUMMARY, PROTOCOL_JSON, PROTOCOL_MD]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": bool(all_consistent),
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "all_static_checks_passed": bool(all_consistent),
        "adequate_post_dryrun_backup_count": len(adequate),
        "rollout_blocked_until_backup": rollout_blocked,
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "all_static_checks_passed": bool(all_consistent),
        "adequate_post_dryrun_backup_count": len(adequate),
        "rollout_blocked_until_backup": rollout_blocked,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(BACKUP_REQ),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
