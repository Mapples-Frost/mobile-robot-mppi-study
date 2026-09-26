#!/usr/bin/env python3
"""Schema-repair audit for vehicle validation64 shard00 audit v1.

The first shard00 post-run audit completed all substantive hash, budget,
episode, trace and aggregate checks, but exited non-zero because it expected the
formal shard runner registry to expose `git_commit`.  The run_experiment
registry schema for the shard uses `commit_sha` instead.  This script is a
versioned corrective audit: it does not rerun simulations, does not train, does
not reopen the validation bank, and does not open/hash the sealed test bank.
It reads already-created validation-result artifacts, so validation_accessed is
true and sealed test access remains false.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard00_audit_v2_schema_repair_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
V1_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard00_audit_20260926"
V1_RAW = V1_DIR / "raw.json"
V1_SUMMARY = V1_DIR / "summary.md"
V1_COMPLETED = V1_DIR / "completed.json"
V1_REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T144037_523e1646/registry.json"
SHARD_REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T132324_e8a43cbf/registry.json"
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
GATE = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_COMMIT = "523ec69d0986ebde3f10f24fd6dd8ad7df30e1f9"
EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_V1_SCRIPT_SHA = "051b82b7a7edbb7dc32c8944cb0b307fc91355b15e5c0402c18201b777e2ba13"
EXPECTED_EPISODES = 224
EXPECTED_CONTROL_STEPS = 19832
EXPECTED_CONTROL_UPPER = 33600
DOC_MARKER = "vehicle-validation64-shard00-audit-v2-schema-repair-20260926"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, body: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")
    return True


def require(condition: bool, failures: List[str], message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
            return 0
        raise SystemExit("prior v2 audit exists but did not pass; inspect before rerun")

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    failures: List[str] = []
    required = [V1_RAW, V1_SUMMARY, V1_COMPLETED, V1_REGISTRY, SHARD_REGISTRY, SHARD_DIR / "raw.json", SHARD_DIR / "completed.json", SHARD_DIR / "summary.md", RUNNER, GATE]
    missing = [rel(p) for p in required if not p.exists()]
    require(not missing, failures, "missing required files: %s" % missing)
    if missing:
        raw = {"created_utc": now, "passed": False, "failures": failures, "validation_accessed": True, "test_accessed": False}
        write_json(OUT_DIR / "raw.json", raw)
        return 2

    v1 = read_json(V1_RAW)
    v1_done = read_json(V1_COMPLETED)
    v1_registry = read_json(V1_REGISTRY)
    shard_registry = read_json(SHARD_REGISTRY)

    # The expected v1 failure: all substantive checks passed, but a schema field
    # name mismatch made the audit require registry["git_commit"] even though the
    # run_experiment registry records registry["commit_sha"].
    require(v1.get("passed") is False, failures, "v1 raw was expected to be non-passing")
    require(v1.get("failures") == ["registry commit mismatch"], failures, "v1 failures were not the single expected schema mismatch")
    require(v1_done.get("passed") is False, failures, "v1 completed marker was expected to be non-passing")
    require(v1_registry.get("exit_status") == 2, failures, "v1 registry exit status was not 2")
    require(v1_registry.get("script_sha256") == EXPECTED_V1_SCRIPT_SHA, failures, "v1 audit script sha mismatch")

    require((v1.get("completed_hash_audit") or {}).get("passed") is True, failures, "v1 completed hash audit did not pass")
    require((v1.get("episode_completed_audit") or {}).get("all_episode_hash_audits_passed") is True, failures, "v1 episode hash/trace audit did not pass")
    require((v1.get("aggregate_audit") or {}).get("all_passed") is True, failures, "v1 aggregate audit did not pass")
    require((v1.get("schedule_audit") or {}).get("schedule_mismatches") == [], failures, "v1 schedule mismatch list not empty")
    require((v1.get("schedule_audit") or {}).get("raw_vs_episode_summary_mismatches") == [], failures, "v1 raw-vs-summary mismatch list not empty")

    budget = v1.get("budget_audit") or {}
    actual = budget.get("actual") or {}
    declared = budget.get("declared") or {}
    require(actual.get("episodes") == EXPECTED_EPISODES, failures, "episode count mismatch")
    require(actual.get("control_steps") == EXPECTED_CONTROL_STEPS, failures, "control step count mismatch")
    require(declared.get("control_step_upper_bound") == EXPECTED_CONTROL_UPPER, failures, "declared control upper mismatch")
    require(budget.get("control_step_upper_bound_respected") is True, failures, "control upper not respected")
    require(actual.get("test_bank_cases_opened") == 0, failures, "test bank cases opened in v1 budget")

    access = v1.get("access_audit") or {}
    raw_flags = access.get("raw_flags") or {}
    require(raw_flags.get("validation_accessed") is True, failures, "v1 did not record validation access")
    require(raw_flags.get("test_accessed") is False, failures, "v1 recorded test access")
    require(raw_flags.get("sealed_test_bank_content_opened") is False, failures, "v1 recorded sealed test content opened")
    sealed_meta = access.get("sealed_test_metadata_from_gate_only") or {}
    require(sealed_meta.get("content_opened") is False, failures, "sealed test metadata content_opened not false")
    require(sealed_meta.get("sha256_computed_now") is False, failures, "sealed test metadata sha256_computed_now not false")

    commit_from_schema = shard_registry.get("git_commit") or shard_registry.get("commit_sha")
    require(commit_from_schema == EXPECTED_COMMIT, failures, "shard registry commit_sha/git_commit mismatch")
    require(shard_registry.get("git_commit") is None and shard_registry.get("commit_sha") == EXPECTED_COMMIT, failures, "schema mismatch diagnosis not reproduced")
    require(shard_registry.get("exit_status") == 0, failures, "shard registry exit status not zero")
    require(shard_registry.get("script_sha256") == EXPECTED_RUNNER_SHA, failures, "shard runner sha in registry mismatch")
    require(sha256(RUNNER) == EXPECTED_RUNNER_SHA, failures, "current runner sha mismatch")
    require(sha256(GATE) == EXPECTED_GATE_SHA, failures, "current gate sha mismatch")

    snapshot = v1.get("selection_relevant_snapshot") or {}
    learned = snapshot.get("learned_rollouts") or {}
    require((learned.get("learned_s0") or {}).get("horizon_counts") == {"25": 408}, failures, "learned_s0 horizon counts changed")
    require((learned.get("learned_s1") or {}).get("horizon_counts") == {"25": 390}, failures, "learned_s1 horizon counts changed")
    require((learned.get("learned_s2") or {}).get("horizon_counts") == {"25": 287, "35": 60}, failures, "learned_s2 horizon counts changed")

    observed_hashes = {rel(p): sha256(p) for p in required}
    optional = [V1_DIR / "summary.md", V1_DIR / "completed.json", V1_DIR / "raw.json", ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_20260926T144050.json"]
    observed_hashes.update({rel(p): sha256(p) for p in optional if p.exists()})

    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_AUDIT_V2_%s.json" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    raw_out: Dict[str, Any] = {
        "created_utc": now,
        "passed": not failures,
        "failures": failures,
        "purpose": "versioned schema-repair audit for vehicle validation64 shard00 audit v1 false negative",
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created_by_this_audit": False,
        "v1_failure_diagnosis": {
            "v1_passed": v1.get("passed"),
            "v1_failures": v1.get("failures"),
            "shard_registry_git_commit": shard_registry.get("git_commit"),
            "shard_registry_commit_sha": shard_registry.get("commit_sha"),
            "diagnosis": "false negative caused by audit-v1 expecting git_commit while run_experiment registry stores commit_sha",
        },
        "substantive_v1_checks_reused": {
            "completed_hash_audit_passed": (v1.get("completed_hash_audit") or {}).get("passed"),
            "episode_trace_hash_audit_passed": (v1.get("episode_completed_audit") or {}).get("all_episode_hash_audits_passed"),
            "aggregate_audit_passed": (v1.get("aggregate_audit") or {}).get("all_passed"),
            "schedule_mismatches": (v1.get("schedule_audit") or {}).get("schedule_mismatches"),
            "raw_vs_episode_summary_mismatches": (v1.get("schedule_audit") or {}).get("raw_vs_episode_summary_mismatches"),
        },
        "budget_audit": {"episodes": actual.get("episodes"), "control_steps": actual.get("control_steps"), "control_step_upper_bound": declared.get("control_step_upper_bound")},
        "selection_relevant_snapshot": snapshot,
        "observed_hashes": observed_hashes,
        "interpretation_limits": [
            "This corrective audit does not add new simulations or model-selection evidence; it classifies audit-v1's nonzero exit as a registry-schema false negative.",
            "Shard00 remains only 1/12 of the frozen validation64 block; no final model selection or reproduction claim is allowed.",
            "Vehicle latency-tree evidence is IMPROVED, not ORIGINAL SAC; s0/s1 are fixed/nonadaptive and s2 is only a candidate pending full validation.",
            "Sealed final test remains closed and unauthorized.",
            "A verified external backup covering shard00, audit-v1, this audit-v2, docs, registries and backup requests is required before shard01.",
        ],
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
    }
    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    completed_path = OUT_DIR / "completed.json"
    write_json(raw_path, raw_out)

    lines = [
        "# Vehicle validation64 shard00 audit-v2 schema repair",
        "",
        "Created UTC: `%s`." % now,
        "",
        "This is a metadata/result audit of audit-v1. It reads already-created validation outputs, so validation_accessed=true; sealed test accessed=false; simulations=0; training=0.",
        "",
        "- audit-v2 passed: `%s`" % raw_out["passed"],
        "- audit-v1 failures: `%s`" % v1.get("failures"),
        "- schema diagnosis: shard registry has `commit_sha=%s` and `git_commit=%s`; audit-v1 expected only `git_commit`." % (shard_registry.get("commit_sha"), shard_registry.get("git_commit")),
        "- v1 completed hash audit passed: `%s`" % raw_out["substantive_v1_checks_reused"]["completed_hash_audit_passed"],
        "- v1 episode trace/hash audit passed: `%s`" % raw_out["substantive_v1_checks_reused"]["episode_trace_hash_audit_passed"],
        "- v1 aggregate audit passed: `%s`" % raw_out["substantive_v1_checks_reused"]["aggregate_audit_passed"],
        "- episodes/control steps audited: `%s` / `%s`" % (actual.get("episodes"), actual.get("control_steps")),
        "- learned rollouts: s0 `%s`, s1 `%s`, s2 `%s`" % ((learned.get("learned_s0") or {}).get("horizon_counts"), (learned.get("learned_s1") or {}).get("horizon_counts"), (learned.get("learned_s2") or {}).get("horizon_counts")),
        "",
        "No final selection, reproduction claim, or sealed-test access follows from this audit. Backup is required before shard01.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_request = {
        "status": "external_backup_requested_after_validation64_shard00_audit_v2_schema_repair",
        "backup_verified": False,
        "created_utc": now,
        "reason": "Preserve formal shard00, failed audit-v1, corrective audit-v2, docs and registries before any additional formal validation shards.",
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations_by_request": 0,
        "required_before_shard1": True,
        "artifacts_to_cover": sorted(list(observed_hashes) + [rel(raw_path), rel(summary_path), rel(completed_path)]),
        "next_action_after_verified_backup": "Run vehicle_validation64_shard_runner.py --shard 1 only after audit-v2 passed and a repository-local verified backup proof covering shard00/audit-v1/audit-v2 exists; sealed test remains closed.",
    }
    write_json(backup_request_path, backup_request)

    raw_out["backup_request_sha256"] = sha256(backup_request_path)
    write_json(raw_path, raw_out)

    completed = {
        "passed": raw_out["passed"],
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created_by_this_audit": False,
        "episodes_audited": actual.get("episodes"),
        "control_steps_audited": actual.get("control_steps"),
        "v1_false_negative_repaired": raw_out["passed"],
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
        "hashes": {
            rel(raw_path): sha256(raw_path),
            rel(summary_path): sha256(summary_path),
            rel(backup_request_path): sha256(backup_request_path),
        },
    }
    write_json(completed_path, completed)

    doc_body = """
## 2026-09-26 vehicle validation64 shard00 audit-v2 schema repair

UTC: {now}. Corrective audit-v2 classified audit-v1's nonzero exit as a registry schema false negative: the shard run registry records `commit_sha={commit}` while audit-v1 required a `git_commit` field. Substantive v1 checks were complete and passed: completed-hash audit={hash_ok}, episode trace/hash audit={episode_ok}, aggregate replay={aggregate_ok}, schedule mismatches={schedule_mismatches}. validation_accessed=true because already-created shard00 results were read; sealed test accessed=false; simulations=0; training steps=0. Shard00 remains only 1/12 validation evidence, not model selection. External backup covering shard00, audit-v1, audit-v2, docs, registries and backup requests is required before shard01.
""".format(
        now=now,
        commit=shard_registry.get("commit_sha"),
        hash_ok=raw_out["substantive_v1_checks_reused"]["completed_hash_audit_passed"],
        episode_ok=raw_out["substantive_v1_checks_reused"]["episode_trace_hash_audit_passed"],
        aggregate_ok=raw_out["substantive_v1_checks_reused"]["aggregate_audit_passed"],
        schedule_mismatches=raw_out["substantive_v1_checks_reused"]["schedule_mismatches"],
    ).strip()
    docs_updated = []
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists() and append_once(path, DOC_MARKER, doc_body):
            docs_updated.append(name)
    completed["docs_updated"] = docs_updated
    write_json(completed_path, completed)

    print(json.dumps({
        "passed": raw_out["passed"],
        "audit_raw": rel(raw_path),
        "audit_summary": rel(summary_path),
        "audit_completed": rel(completed_path),
        "backup_request": rel(backup_request_path),
        "validation_accessed": True,
        "test_accessed": False,
        "new_simulations": 0,
        "episodes_audited": actual.get("episodes"),
        "control_steps_audited": actual.get("control_steps"),
        "v1_false_negative_repaired": raw_out["passed"],
    }, sort_keys=True))
    return 0 if raw_out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
