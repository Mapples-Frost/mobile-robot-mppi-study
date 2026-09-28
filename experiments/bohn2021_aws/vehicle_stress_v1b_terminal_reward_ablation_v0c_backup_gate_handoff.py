#!/usr/bin/env python3
"""Backup-gate handoff for vehicle stress-v1b terminal/reward ablation v0c.

No simulation, no training/refit, no validation64-bank access and no sealed-test
access.  The purpose is to make the next action unambiguous after the v0c
legacy schema-repair wrapper was written: verify whether any existing external
backup proof postdates the wrapper, preserve the v0/v0b pre-rollout failures,
write a backup request if the smoke remains storage-gated, and persist the exact
next smoke command/budget.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_backup_gate_handoff_20260928T2348Z"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260928T2348Z_after_v0c_backup_gate_handoff.md"
REQUEST = BACKUP_DIR / "REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0C_SMOKE_20260928T2348Z.json"
MARKER = "vehicle-stress-v1b-terminal-reward-ablation-v0c-backup-gate-handoff-20260928T2348Z"

V0 = ROOT / "experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0.py"
V0B = ROOT / "experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0b_legacy_retry.py"
V0C = ROOT / "experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0c_legacy_schema_repair.py"
READINESS_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_presmoke_readiness_v0_20260928T2325Z/summary.md"
READINESS_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_presmoke_readiness_v0_20260928T2325Z/completed.json"
DRYRUN_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_20260928T2320Z.json"
V0_FAILURE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_smoke_20260928T2320Z/failure.json"
V0B_FAILURE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0b_smoke_20260928T2335Z_legacy_retry/failure.json"

FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
LATEST_HANDOFF_TOKENS = 89.84e6


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


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
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_mtime(path: Path) -> Optional[dt.datetime]:
    if not path.exists():
        return None
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def proof_time(proof: Dict[str, Any]) -> Optional[dt.datetime]:
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        t = parse_time(proof.get(key))
        if t is not None:
            return t
    return None


def proof_has_package_sha(proof: Dict[str, Any]) -> bool:
    if proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256"):
        return True
    packages = proof.get("packages_this_run") or []
    return any(bool(pkg.get("sha256")) and bool(pkg.get("bytes")) and bool(pkg.get("verification")) for pkg in packages if isinstance(pkg, dict))


def load_backup_proofs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        try:
            proof = read_json(path)
        except Exception as exc:
            out.append({"path": rel(path), "parse_error": repr(exc), "adequate_basic": False})
            continue
        t = proof_time(proof)
        adequate = bool((proof.get("backup_verified") is True or proof.get("status") == "verified") and int(proof.get("remaining_changed_files", -1)) == 0 and proof.get("commit") and proof_has_package_sha(proof))
        out.append({
            "path": rel(path),
            "sha256": sha256(path),
            "time": None if t is None else t.isoformat(),
            "commit": proof.get("commit"),
            "remaining_changed_files": proof.get("remaining_changed_files"),
            "adequate_basic": adequate,
            "package_count": len(proof.get("packages_this_run") or []),
        })
    return out


def completed_passed(path: Path) -> Dict[str, Any]:
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise RuntimeError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise RuntimeError(f"sealed-test flag not false: {rel(path)}")
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise RuntimeError(f"historical validation64 flag not false: {rel(path)}")
    return obj


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required_paths = [V0, V0B, V0C, READINESS_SUMMARY, READINESS_COMPLETED, DRYRUN_COMPLETED, PROTOCOL, V0_FAILURE, V0B_FAILURE]
    missing = [rel(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError("missing required artifacts: " + ", ".join(missing))
    dry_done = completed_passed(DRYRUN_COMPLETED)
    ready_done = completed_passed(READINESS_COMPLETED)
    v0_fail = read_json(V0_FAILURE)
    v0b_fail = read_json(V0B_FAILURE)

    now = dt.datetime.now(dt.timezone.utc)
    mtimes = {rel(p): source_mtime(p) for p in required_paths}
    wrapper_mtime = mtimes[rel(V0C)]
    dry_created = parse_time(dry_done.get("created_utc"))
    ready_created = parse_time(ready_done.get("created_utc")) or parse_time(ready_done.get("completed_utc"))
    min_for_v0c_smoke = max(t for t in [wrapper_mtime, dry_created, ready_created] if t is not None)
    proofs = load_backup_proofs()
    adequate_after_v0c = []
    for p in proofs:
        t = parse_time(p.get("time"))
        if p.get("adequate_basic") and t is not None and t >= min_for_v0c_smoke:
            adequate_after_v0c.append(p)
    latest_proof = None
    for p in proofs:
        t = parse_time(p.get("time"))
        if t is not None and p.get("adequate_basic") and (latest_proof is None or t > parse_time(latest_proof.get("time"))):
            latest_proof = p

    smoke_blocked = not bool(adequate_after_v0c)
    request = {
        "requested_utc": now.isoformat(),
        "reason": "Verified external backup required before running vehicle stress-v1b terminal/reward ablation v0c smoke. Existing latest proof predates the v0c schema-repair wrapper, so the v0c source-mtime gate would reject it.",
        "backup_required_before_more_simulations": True,
        "must_postdate_utc": now.isoformat(),
        "min_time_for_v0c_smoke_before_this_request_utc": min_for_v0c_smoke.isoformat(),
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "planned_next_command": "legacy python experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0c_legacy_schema_repair.py --run-smoke --input-backup-proof <post-v0c-backup-proof> --i-accept-development-terminal-reward-ablation",
        "planned_next_budget": {"episodes_exact": 27, "control_step_upper_bound": 4050, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "artifacts_to_backup": [rel(p) for p in required_paths] + [rel(OUT), rel(STATE), rel(REQUEST)],
    }
    write_json(REQUEST, request)

    hashes = {rel(p): sha256(p) for p in required_paths if p.exists()}
    hashes[rel(REQUEST)] = sha256(REQUEST)
    raw = {
        "created_utc": now.isoformat(),
        "method": "vehicle_stress_v1b_terminal_reward_ablation_v0c_backup_gate_handoff",
        "classification": "development_no_simulation_backup_gate_and_handoff_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "mtimes_utc": mtimes,
        "min_time_for_v0c_smoke_utc": min_for_v0c_smoke.isoformat(),
        "latest_adequate_backup_proof": latest_proof,
        "adequate_backup_proofs_after_v0c_source": adequate_after_v0c,
        "smoke_blocked_until_post_v0c_backup": smoke_blocked,
        "preserved_failures": {
            "v0_modern_preflight_failure": {"path": rel(V0_FAILURE), "exception": v0_fail.get("exception"), "sealed_test_accessed": v0_fail.get("sealed_test_accessed"), "validation64_bank_opened": v0_fail.get("validation64_bank_opened")},
            "v0b_legacy_schema_failure": {"path": rel(V0B_FAILURE), "exception": v0b_fail.get("exception"), "sealed_test_accessed": v0b_fail.get("sealed_test_accessed"), "validation64_bank_opened": v0b_fail.get("validation64_bank_opened")},
        },
        "readiness_headline": {
            "smoke_inputs_suffice_except_backup": True,
            "smoke_episodes": 27,
            "control_step_upper_bound": 4050,
            "positive_targets_in_smoke": [2, 4],
            "control_targets_in_smoke": [8, 16],
            "risk_tags": ["sparse_development_mined_positive_labels", "physical_vs_total_best_horizon_disagreement", "multi_horizon_material_labels_possible_terminal_or_path_sensitivity"],
        },
        "next_action": "If and only if a verified external backup proof postdates this request/v0c wrapper, run the v0c legacy schema-repair smoke exactly once and then inspect raw/summary before training/refit/scenario changes.",
        "backup_request": rel(REQUEST),
        "hashes": hashes,
    }
    write_json(OUT / "raw.json", raw)

    elapsed = now - FIRST_SUPERVISOR_EVENT
    elapsed_h = elapsed.total_seconds() / 3600.0
    lines = [
        "# Vehicle stress-v1b terminal/reward ablation v0c backup-gate handoff",
        "",
        f"UTC: `{now.isoformat()}`. Development-only no-simulation handoff; no training/refit, no validation64 bank and no sealed test.",
        "",
        f"Elapsed since first supervisor event: `{elapsed}` (~{elapsed_h:.2f} h). Latest handoff server API total_tokens remained approximately `{LATEST_HANDOFF_TOKENS/1e6:.2f}M`; this diagnostic did not refresh sqlite token counters and excludes desktop conversation usage.",
        "",
        "## Backup gate",
        "",
        f"- v0c wrapper mtime/min source gate: `{min_for_v0c_smoke.isoformat()}`.",
        f"- Latest adequate local proof: `{latest_proof}`.",
        f"- Adequate proofs after v0c source: `{len(adequate_after_v0c)}`.",
        f"- Smoke blocked until post-v0c backup: `{smoke_blocked}`.",
        f"- New backup request: `{rel(REQUEST)}`.",
        "",
        "## Preserved failures",
        "",
        f"- v0 modern preflight failure: `{v0_fail.get('exception')}` at `{rel(V0_FAILURE)}`.",
        f"- v0b legacy schema failure: `{v0b_fail.get('exception')}` at `{rel(V0B_FAILURE)}`.",
        "",
        "## Next exact action",
        "",
        "After a verified external backup proof postdating this request, run under the legacy interpreter:",
        "",
        "```text",
        "experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0c_legacy_schema_repair.py --run-smoke --input-backup-proof <post-v0c-backup-proof> --i-accept-development-terminal-reward-ablation",
        "```",
        "",
        "Budget: 27 development episodes / 4050 control-step cap; zero training episodes and zero gradient steps; validation64 and sealed test remain closed.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    block = f"""<!-- {MARKER} -->
## 2026-09-28 v0c terminal/reward ablation backup-gate handoff

UTC: {now.isoformat()}. No-simulation backup-gate audit after writing `experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0c_legacy_schema_repair.py`. Existing latest adequate backup proof is `{None if latest_proof is None else latest_proof.get('path')}` at `{None if latest_proof is None else latest_proof.get('time')}`, which {'does not postdate' if smoke_blocked else 'postdates'} the v0c source gate `{min_for_v0c_smoke.isoformat()}`. Smoke remains {'blocked pending backup' if smoke_blocked else 'eligible with the adequate post-v0c proof'}; no rollouts/training/validation/test occurred. Preserved v0 modern-TF failure and v0b schema KeyError. Next exact action after backup: run the v0c legacy schema-repair 27-episode smoke, then inspect terminal-label flips, physical-vs-total gains, zero-terminal robustness, safety/solver rows and timing before any training/refit/scenario revision. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`, request `{rel(REQUEST)}`.
"""
    append_docs(block)

    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE, REQUEST] + required_paths + [Path(__file__).resolve()]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": now.isoformat(),
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "smoke_blocked_until_post_v0c_backup": smoke_blocked,
        "adequate_backup_proofs_after_v0c_source": len(adequate_after_v0c),
        "min_time_for_v0c_smoke_utc": min_for_v0c_smoke.isoformat(),
        "backup_request": rel(REQUEST),
        "next_action": raw["next_action"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "smoke_blocked_until_post_v0c_backup": smoke_blocked,
        "adequate_backup_proofs_after_v0c_source": len(adequate_after_v0c),
        "backup_request": rel(REQUEST),
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
