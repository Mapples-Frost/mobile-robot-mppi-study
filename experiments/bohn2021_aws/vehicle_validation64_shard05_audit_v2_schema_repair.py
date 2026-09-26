#!/usr/bin/env python3
"""Schema-repair post-run audit for vehicle validation64 shard05.

This v2 audit preserves the failed v1 audit as evidence and fixes two audit-
schema false positives discovered in v1 (20260926T215916_c2ecb9ed):

1. The formal shard raw aggregates record decision_gross_total_s but do not
   record decision_gross_mean_s_per_step. The v1 audit incorrectly treated this
   derived, absent key as a required recorded aggregate. V2 recomputes and
   stores the derived gross mean as auxiliary evidence but only requires fields
   actually recorded by the frozen shard runner output.
2. The run_experiment registry for shard05 records a closed test budget as
   episodes=0, control_steps=0, sealed_test_bank_content_opened=false, and
   test_authorization=false, but does not include a test_accessed key. V1
   required the missing key to be explicitly false. V2 accepts this registry
   schema only when the shard raw/completed/run_started flags all explicitly
   show test_accessed=false and sealed_test_bank_content_opened=false.

No simulations, training, validation-bank reopen, or sealed-test access occur.
The audit reads already-created shard05 validation outputs and the failed v1
metadata. It writes a new v2 audit directory and a new backup request/blocker
before shard06.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard05_audit.py"
SPEC = importlib.util.spec_from_file_location("shard05_audit_v1", str(V1_PATH))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load shard05 v1 audit module")
v1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v1)  # type: ignore[union-attr]

SHARD_INDEX = 5
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard05"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard05_audit_v2_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T204237_d709e6fb/registry.json"
PRE_SHARD_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260926T204130_after_shard04_audit_blocker_state.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard05_audit_v2_schema_repair.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_EXPERIMENT_ID = "20260926T204237_d709e6fb"
EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_VALIDATION_BANK_SHA = "b0ed14f2738a07975a42d5adc12e58ec3970515dea69e6338ed31d69b5fca48f"
EXPECTED_PRE_SHARD_ASSET_SHA = "85569da08f12e962fb4c49dffcf8764a4aed2ca7c4a389df7f0660e6dca83e2e"
EXPECTED_PRE_SHARD_BACKUP_PROOF_SHA = "80bfb564e60be3930d547c48b5dc48fa06cfafb86352bb0e06817e4e3315d1ea"
EXPECTED_EPISODES = 224
EXPECTED_CONTROL_STEPS = 19660
EXPECTED_CONTROL_UPPER = 33600
EXPECTED_EXECUTION_START = 1120
EXPECTED_EXECUTION_END = 1343
DOC_MARKER = "vehicle-validation64-shard05-audit-v2-schema-repair-20260926"

RAW_JSON = SHARD_DIR / "raw.json"
SUMMARY_MD = SHARD_DIR / "summary.md"
COMPLETED_JSON = SHARD_DIR / "completed.json"
SCHEDULE_JSON = SHARD_DIR / "schedule.json"
PROGRESS_JSON = SHARD_DIR / "progress.json"
RUN_STARTED_JSON = SHARD_DIR / "run_started.json"
TERMINAL_PROGRESS_JSON = SHARD_DIR / "terminal_sources_progress.json"
V1_OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard05_audit_20260926"
V1_RAW = V1_OUT_DIR / "raw.json"
V1_SUMMARY = V1_OUT_DIR / "summary.md"
V1_COMPLETED = V1_OUT_DIR / "completed.json"
V1_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T215916_c2ecb9ed/registry.json"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def root_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def close_float(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def verify_completed_marker(path: Path) -> bool:
    done = read_json(path)
    if done.get("passed") is not True:
        return False
    return bool(v1.verify_hash_map(done.get("hashes") or {}).get("passed"))


def aggregate_from_episodes(episodes: List[Mapping[str, Any]]) -> Dict[str, Any]:
    # Use the frozen v1 recomputation so derived diagnostic values match v1;
    # v2 changes only which fields are required to be recorded by shard raw.
    return v1.aggregate_from_episodes(episodes)


def compare_aggregate_v2(name: str, recomputed: Mapping[str, Any], recorded: Mapping[str, Any], failures: List[str]) -> Dict[str, Any]:
    required_keys = [
        "episodes", "steps", "success_count", "episode_failure_count", "constraint_count",
        "initial_failed_steps", "solver_failure_steps", "retries", "recovered_steps",
        "deadline_exceed_steps", "switches", "total_cost_sum", "performance_cost_sum",
        "constraint_cost_sum", "physical_constraint_cost_sum", "h_penalty_sum", "decision_total_s",
        "decision_gross_total_s", "logging_total_s", "construction_total_s", "reset_total_s",
        "decision_mean_s_per_step", "total_cost_mean_episode", "physical_constraint_cost_mean_episode",
    ]
    optional_derived_keys = ["decision_gross_mean_s_per_step"]
    mismatches: List[Dict[str, Any]] = []
    auxiliary: Dict[str, Any] = {}
    for key in required_keys:
        if key not in recorded and key not in recomputed:
            continue
        a = recomputed.get(key)
        b = recorded.get(key)
        ok = close_float(a, b) if isinstance(a, float) or isinstance(b, float) else (a == b)
        if not ok:
            mismatches.append({"key": key, "recomputed": a, "recorded": b})
    if dict(recomputed.get("horizon_counts") or {}) != dict(recorded.get("horizon_counts") or {}):
        mismatches.append({"key": "horizon_counts", "recomputed": recomputed.get("horizon_counts"), "recorded": recorded.get("horizon_counts")})
    for key in optional_derived_keys:
        auxiliary[key] = {"recomputed": recomputed.get(key), "recorded": recorded.get(key), "required": False}
    if mismatches:
        failures.append("aggregate mismatch for %s: %s" % (name, mismatches[:5]))
    return {"name": name, "passed": not mismatches, "mismatches": mismatches, "auxiliary_not_required": auxiliary}


def registry_test_budget_closed_v2(tb: Mapping[str, Any], raw: Mapping[str, Any], completed: Mapping[str, Any], run_started: Mapping[str, Any]) -> Dict[str, Any]:
    episodes_zero = int(tb.get("episodes", tb.get("test_episodes", 0)) or 0) == 0
    control_zero = int(tb.get("control_steps", 0) or 0) == 0
    sealed_false = tb.get("sealed_test_bank_content_opened") is False
    auth_false_or_absent_closed = (tb.get("test_authorization") is False) or (tb.get("final_test_authorization_requested") is False) or (tb.get("test_accessed") is False)
    registry_test_accessed_field = tb.get("test_accessed", None)
    raw_closed = raw.get("test_accessed") is False and raw.get("sealed_test_bank_content_opened") is False
    completed_closed = completed.get("test_accessed") is False and completed.get("sealed_test_bank_content_opened") is False
    run_started_closed = run_started.get("test_accessed") is False and run_started.get("sealed_test_bank_content_opened") is False
    accepted_missing_test_accessed = registry_test_accessed_field is None and episodes_zero and control_zero and sealed_false and auth_false_or_absent_closed and raw_closed and completed_closed and run_started_closed
    explicit_closed = registry_test_accessed_field is False and episodes_zero and control_zero and sealed_false
    return {
        "passed": bool(explicit_closed or accepted_missing_test_accessed),
        "schema": "explicit_test_accessed_false" if explicit_closed else ("closed_budget_missing_test_accessed_key_accepted_by_v2" if accepted_missing_test_accessed else "not_closed"),
        "registry_test_budget": dict(tb),
        "episodes_zero": episodes_zero,
        "control_steps_zero": control_zero,
        "sealed_test_bank_content_opened_false": sealed_false,
        "test_authorization_false_or_absent_with_other_closed_flags": auth_false_or_absent_closed,
        "registry_test_accessed_field": registry_test_accessed_field,
        "raw_closed": raw_closed,
        "completed_closed": completed_closed,
        "run_started_closed": run_started_closed,
    }


def cumulative_completed_through_shard() -> Dict[str, Any]:
    shards: List[int] = []
    episodes = 0
    control_steps = 0
    missing_or_failed: List[str] = []
    for idx in range(SHARD_INDEX + 1):
        done_path = ROOT / ("research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard%02d/completed.json" % idx)
        if not done_path.exists():
            missing_or_failed.append(rel(done_path))
            continue
        done = read_json(done_path)
        if done.get("passed") is not True:
            missing_or_failed.append(rel(done_path))
            continue
        shards.append(idx)
        episodes += int(done.get("episodes", 0))
        control_steps += int(done.get("control_steps", 0))
    return {"shards": shards, "episodes": episodes, "control_steps": control_steps, "missing_or_failed": missing_or_failed}


def main() -> int:
    completed_path = OUT_DIR / "completed.json"
    if completed_path.exists():
        if verify_completed_marker(completed_path):
            print(json.dumps({"already_completed": True, "completed": rel(completed_path)}, sort_keys=True))
            return 0
        raise SystemExit("prior shard05 v2 audit exists but did not verify; inspect before rerun")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    now = now_dt.isoformat()
    stamp = now_dt.strftime("%Y%m%dT%H%M%S")
    failures: List[str] = []

    required_files = [
        RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON,
        RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY,
        PRE_SHARD_BACKUP_PROOF, RUNNER, THIS_SCRIPT, V1_PATH, GATE_JSON,
        V1_RAW, V1_SUMMARY, V1_COMPLETED, V1_RUN_REGISTRY,
    ]
    missing = [rel(p) for p in required_files if not p.exists()]
    require(not missing, failures, "missing required files: %s" % missing)
    if missing:
        write_json(OUT_DIR / "raw.json", {"created_utc": now, "passed": False, "failures": failures, "validation_accessed": True, "test_accessed": False})
        return 2

    raw = read_json(RAW_JSON)
    completed = read_json(COMPLETED_JSON)
    schedule = read_json(SCHEDULE_JSON)
    progress = read_json(PROGRESS_JSON)
    run_started = read_json(RUN_STARTED_JSON)
    registry = read_json(REGISTRY)
    pre_proof = read_json(PRE_SHARD_BACKUP_PROOF)
    v1_raw = read_json(V1_RAW)
    v1_completed = read_json(V1_COMPLETED)
    v1_registry = read_json(V1_RUN_REGISTRY)
    registry_stdout = Path(registry.get("stdout", "")) if registry.get("stdout") else None
    registry_stderr = Path(registry.get("stderr", "")) if registry.get("stderr") else None
    v1_stdout = Path(v1_registry.get("stdout", "")) if v1_registry.get("stdout") else None
    v1_stderr = Path(v1_registry.get("stderr", "")) if v1_registry.get("stderr") else None

    top_hashes = {rel(p): sha256(p) for p in required_files}
    for extra in (registry_stdout, registry_stderr, v1_stdout, v1_stderr):
        if extra is not None and extra.exists():
            top_hashes[rel(extra)] = sha256(extra)

    completed_hash_audit = v1.verify_hash_map(completed.get("hashes") or {})
    require(completed_hash_audit["passed"], failures, "top-level shard completed hash audit failed")
    require(top_hashes[rel(RUNNER)] == EXPECTED_RUNNER_SHA, failures, "runner sha mismatch")
    require(top_hashes[rel(GATE_JSON)] == EXPECTED_GATE_SHA, failures, "gate sha mismatch")
    require(top_hashes[rel(PRE_SHARD_BACKUP_PROOF)] == EXPECTED_PRE_SHARD_BACKUP_PROOF_SHA, failures, "pre-shard backup proof local sha mismatch")

    for key, expected in {
        "validation_accessed": True,
        "validation64_bank_content_opened": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "formal_scientific_evidence_created": True,
    }.items():
        require(raw.get(key) is expected, failures, "raw flag %s expected %s got %s" % (key, expected, raw.get(key)))
        require(completed.get(key) is expected, failures, "completed flag %s expected %s got %s" % (key, expected, completed.get(key)))
    require(raw.get("final_test_authorization_requested") is False, failures, "raw final_test_authorization_requested not false")
    require(run_started.get("test_accessed") is False and run_started.get("sealed_test_bank_content_opened") is False, failures, "run_started test flags not false")
    require(raw.get("new_gradient_steps") == 0 and completed.get("new_gradient_steps") == 0, failures, "new_gradient_steps not zero")

    declared = raw.get("budget_declared") or {}
    actual = raw.get("budget_actual") or {}
    require(raw.get("shard_index") == SHARD_INDEX, failures, "raw shard index mismatch")
    require(declared.get("episodes_exact") == EXPECTED_EPISODES, failures, "declared episode count mismatch")
    require(declared.get("control_step_upper_bound") == EXPECTED_CONTROL_UPPER, failures, "declared control upper mismatch")
    require(actual.get("episodes") == EXPECTED_EPISODES, failures, "actual episode count mismatch")
    require(actual.get("control_steps") == EXPECTED_CONTROL_STEPS, failures, "actual control steps mismatch")
    require(completed.get("episodes") == EXPECTED_EPISODES and completed.get("control_steps") == EXPECTED_CONTROL_STEPS, failures, "completed budget mismatch")
    require(int(actual.get("control_steps", 10**9)) <= EXPECTED_CONTROL_UPPER, failures, "control upper exceeded")
    require(actual.get("test_bank_cases_opened") == 0 and declared.get("test_episodes") == 0, failures, "test budget not zero")

    validation_bank = raw.get("validation_bank") or {}
    require(validation_bank.get("sha256") == EXPECTED_VALIDATION_BANK_SHA, failures, "validation bank sha in raw mismatch")
    require(str(validation_bank.get("path", "")).endswith("vehicle_validation_bank.json"), failures, "validation bank path unexpected")
    sealed_meta = (((raw.get("gate_verification") or {}).get("bank_metadata_no_content_opened") or {}).get("sealed_test128") or {})
    require(sealed_meta.get("content_opened") is False, failures, "sealed test metadata content_opened not false")
    require(sealed_meta.get("sha256_computed_now") is False, failures, "sealed test metadata sha256_computed_now not false")
    require(str(sealed_meta.get("path", "")).endswith("vehicle_test_bank.json"), failures, "sealed test metadata path unexpected")

    require(pre_proof.get("backup_verified") is True, failures, "pre-shard backup proof not verified")
    require(pre_proof.get("remaining_changed_files") == 0, failures, "pre-shard backup proof remaining_changed_files not zero")
    require(pre_proof.get("runner_sha256") == EXPECTED_RUNNER_SHA, failures, "pre-shard backup proof runner sha mismatch")
    require(pre_proof.get("gate_sha256") == EXPECTED_GATE_SHA, failures, "pre-shard backup proof gate sha mismatch")
    require(pre_proof.get("asset_sha256") == EXPECTED_PRE_SHARD_ASSET_SHA, failures, "pre-shard backup proof asset sha mismatch")
    require(bool(pre_proof.get("commit")), failures, "pre-shard backup proof lacks commit")

    require(registry.get("experiment_id") == EXPECTED_EXPERIMENT_ID, failures, "registry experiment id mismatch")
    require(registry.get("exit_status") == 0, failures, "registry exit status not zero")
    require(registry.get("script_sha256") == EXPECTED_RUNNER_SHA, failures, "registry script sha mismatch")
    require(registry.get("interpreter") == "legacy", failures, "registry interpreter is not legacy")
    registry_commit = registry.get("git_commit") or registry.get("commit_sha")
    require(bool(registry_commit), failures, "registry lacks git_commit/commit_sha")
    require(registry_commit == pre_proof.get("commit"), failures, "registry commit does not match pre-shard proof commit")
    vb = registry.get("validation_budget") or {}
    require(vb.get("episodes") == EXPECTED_EPISODES or vb.get("formal_validation_shard_episodes") == EXPECTED_EPISODES, failures, "registry validation budget episode mismatch")
    tb = registry.get("test_budget") or {}
    registry_test_closed = registry_test_budget_closed_v2(tb, raw, completed, run_started)
    require(registry_test_closed["passed"], failures, "registry test budget flags not closed under v2 schema")

    artifact_inventory_checks: List[Dict[str, Any]] = []
    for item in registry.get("artifact_inventory") or []:
        p = item.get("path")
        if not p or "<experiment_id>" in p:
            continue
        actual_path = root_path(p)
        actual_exists = actual_path.exists()
        passed = (item.get("exists") == actual_exists)
        actual_sha: Optional[str] = None
        if actual_exists and actual_path.is_file():
            actual_sha = sha256(actual_path)
            if item.get("sha256") is not None:
                passed = passed and (item.get("sha256") == actual_sha)
        artifact_inventory_checks.append({"path": p, "registry_exists": item.get("exists"), "actual_exists": actual_exists, "registry_sha256": item.get("sha256"), "actual_sha256": actual_sha, "passed": passed})
    bad_inventory = [x for x in artifact_inventory_checks if not x["passed"]]
    require(not bad_inventory, failures, "registry artifact inventory mismatches: %s" % bad_inventory[:3])

    rows = sorted(schedule.get("rows") or [], key=lambda r: int(r["execution_index"]))
    require(len(rows) == EXPECTED_EPISODES, failures, "schedule row count mismatch")
    if rows:
        require(int(rows[0]["execution_index"]) == EXPECTED_EXECUTION_START and int(rows[-1]["execution_index"]) == EXPECTED_EXECUTION_END, failures, "schedule execution range mismatch")
        require([int(r["execution_index"]) for r in rows] == list(range(EXPECTED_EXECUTION_START, EXPECTED_EXECUTION_END + 1)), failures, "schedule execution indices are not contiguous")

    episodes = sorted(raw.get("episodes") or [], key=lambda e: int(e["execution_index"]))
    require(len(episodes) == EXPECTED_EPISODES, failures, "raw episode count mismatch")
    require(sum(int(e.get("steps", 0)) for e in episodes) == EXPECTED_CONTROL_STEPS, failures, "raw episode step sum mismatch")
    require(progress.get("episodes_done") == EXPECTED_EPISODES and progress.get("control_steps_done") == EXPECTED_CONTROL_STEPS, failures, "progress final counts mismatch")
    require(progress.get("test_accessed") is False, failures, "progress test flag not false")

    schedule_mismatches: List[Dict[str, Any]] = []
    for row, ep in zip(rows, episodes):
        comparisons = [(row.get("execution_index"), ep.get("execution_index"), "execution_index"), (row.get("rollout_key"), ep.get("rollout_key"), "rollout_key"), (row.get("case_index"), ep.get("case"), "case_index/case")]
        for a, b, key in comparisons:
            if a != b:
                schedule_mismatches.append({"row_execution_index": row.get("execution_index"), "key": key, "schedule": a, "episode": b})
    require(not schedule_mismatches, failures, "schedule/raw episode mismatches: %s" % schedule_mismatches[:5])

    episode_root = SHARD_DIR / "episodes"
    episode_dirs = sorted([p for p in episode_root.iterdir() if p.is_dir()]) if episode_root.exists() else []
    require(len(episode_dirs) == EXPECTED_EPISODES, failures, "episode directory count mismatch")
    episode_completed_failures: List[str] = []
    trace_len_mismatches: List[Dict[str, Any]] = []
    summary_by_exec: Dict[int, Dict[str, Any]] = {}
    episode_hash_records = 0
    for ep_dir in episode_dirs:
        ep_completed = ep_dir / "completed.json"
        ep_summary = ep_dir / "summary.json"
        trace_json = ep_dir / "trace.json"
        if not ep_completed.exists() or not ep_summary.exists() or not trace_json.exists():
            episode_completed_failures.append(rel(ep_dir))
            continue
        done = read_json(ep_completed)
        if done.get("passed") is not True:
            episode_completed_failures.append(rel(ep_completed))
        audit = v1.verify_hash_map(done.get("hashes") or {})
        episode_hash_records += int(audit.get("hash_records", 0))
        if not audit["passed"]:
            episode_completed_failures.append(rel(ep_completed) + " hash audit failed")
        summary = read_json(ep_summary)
        exec_idx = int(summary.get("execution_index"))
        summary_by_exec[exec_idx] = summary
        trace_len = len(read_json(trace_json))
        if trace_len != int(summary.get("steps", -1)):
            trace_len_mismatches.append({"path": rel(ep_dir), "steps": summary.get("steps"), "trace_len": trace_len})
    require(not episode_completed_failures, failures, "episode completed/hash failures: %s" % episode_completed_failures[:5])
    require(not trace_len_mismatches, failures, "trace length mismatches: %s" % trace_len_mismatches[:5])
    require(set(summary_by_exec) == set(range(EXPECTED_EXECUTION_START, EXPECTED_EXECUTION_END + 1)), failures, "episode summary execution index set mismatch")

    raw_summary_mismatches: List[Dict[str, Any]] = []
    for ep in episodes:
        summary = summary_by_exec.get(int(ep["execution_index"]))
        if not summary:
            continue
        for key in ("rollout_key", "case", "steps", "success", "termination", "horizon_counts"):
            if ep.get(key) != summary.get(key):
                raw_summary_mismatches.append({"execution_index": ep.get("execution_index"), "key": key, "raw": ep.get(key), "summary": summary.get(key)})
        for key in ("total_cost", "performance_cost", "constraint_cost", "physical_constraint_cost", "h_penalty"):
            if not close_float(ep.get(key), summary.get(key)):
                raw_summary_mismatches.append({"execution_index": ep.get("execution_index"), "key": key, "raw": ep.get(key), "summary": summary.get(key)})
    require(not raw_summary_mismatches, failures, "raw vs summary mismatches: %s" % raw_summary_mismatches[:5])

    aggregate_checks: List[Dict[str, Any]] = []
    aggregate_checks.append(compare_aggregate_v2("overall", aggregate_from_episodes(episodes), raw.get("overall_aggregate") or {}, failures))
    for family_name in sorted({e.get("family") for e in episodes}):
        eps = [e for e in episodes if e.get("family") == family_name]
        aggregate_checks.append(compare_aggregate_v2("family:%s" % family_name, aggregate_from_episodes(eps), (raw.get("aggregates_by_family") or {}).get(family_name) or {}, failures))
    for rollout_key in sorted({e.get("rollout_key") for e in episodes}):
        eps = [e for e in episodes if e.get("rollout_key") == rollout_key]
        aggregate_checks.append(compare_aggregate_v2("rollout:%s" % rollout_key, aggregate_from_episodes(eps), (raw.get("aggregates_by_rollout_key") or {}).get(rollout_key) or {}, failures))
    aggregate_replay_passed = all(item["passed"] for item in aggregate_checks)

    rollout_aggs = raw.get("aggregates_by_rollout_key") or {}
    learned_horizon_audit: Dict[str, Any] = {}
    for key in ("learned_s0", "learned_s1", "learned_s2"):
        agg = rollout_aggs.get(key) or {}
        horizons = {str(k): int(v) for k, v in (agg.get("horizon_counts") or {}).items()}
        learned_horizon_audit[key] = {"episodes": agg.get("episodes"), "steps": agg.get("steps"), "success_count": agg.get("success_count"), "episode_failure_count": agg.get("episode_failure_count"), "switches": agg.get("switches"), "horizon_counts": horizons, "unique_horizons": agg.get("unique_horizons"), "adaptive_in_this_shard": len(horizons) >= 2}

    overall = raw.get("overall_aggregate") or {}
    family = raw.get("aggregates_by_family") or {}
    selection_snapshot = {
        "overall": {"episodes": overall.get("episodes"), "steps": overall.get("steps"), "success_count": overall.get("success_count"), "episode_failure_count": overall.get("episode_failure_count"), "constraint_count": overall.get("constraint_count"), "total_cost_sum": overall.get("total_cost_sum"), "physical_constraint_cost_sum": overall.get("physical_constraint_cost_sum"), "decision_mean_s_per_step": overall.get("decision_mean_s_per_step"), "decision_gross_total_s": overall.get("decision_gross_total_s"), "decision_gross_mean_s_per_step_recomputed_auxiliary": aggregate_from_episodes(episodes).get("decision_gross_mean_s_per_step")},
        "families": {name: {"episodes": agg.get("episodes"), "steps": agg.get("steps"), "success_count": agg.get("success_count"), "total_cost_sum": agg.get("total_cost_sum"), "physical_constraint_cost_sum": agg.get("physical_constraint_cost_sum"), "decision_mean_s_per_step": agg.get("decision_mean_s_per_step"), "decision_gross_total_s": agg.get("decision_gross_total_s")} for name, agg in sorted(family.items())},
        "learned_horizon_audit": learned_horizon_audit,
    }

    cumulative = cumulative_completed_through_shard()
    v1_failure_diagnosis = {
        "v1_audit_script": {"path": rel(V1_PATH), "sha256": top_hashes[rel(V1_PATH)]},
        "v1_run_registry": {"path": rel(V1_RUN_REGISTRY), "sha256": top_hashes[rel(V1_RUN_REGISTRY)], "exit_status": v1_registry.get("exit_status"), "experiment_id": v1_registry.get("experiment_id")},
        "v1_completed": {"path": rel(V1_COMPLETED), "sha256": top_hashes[rel(V1_COMPLETED)], "passed": v1_completed.get("passed")},
        "v1_failures_count": len(v1_raw.get("failures") or []),
        "diagnosed_false_positive_classes": [
            "decision_gross_mean_s_per_step was recomputed by v1 but not a recorded raw aggregate key",
            "registry test_budget omitted test_accessed while other registry plus raw/completed/run_started flags prove sealed test closed",
        ],
        "v1_new_simulations": v1_completed.get("new_simulations"),
        "v1_new_control_steps": v1_completed.get("new_control_steps"),
        "v1_new_gradient_steps": v1_completed.get("new_gradient_steps"),
        "v1_test_accessed": v1_completed.get("test_accessed"),
    }

    passed = not failures
    raw_out = OUT_DIR / "raw.json"
    summary_out = OUT_DIR / "summary.md"
    request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_%s.json" % stamp)
    blocker_path = ROOT / "research_artifacts/aws_diagnostics/post_shard05_audit_v2_backup_blocker_check_20260926.md"
    final_addendum_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_FINAL_ADDENDUM_%s.json" % stamp)

    audit_raw: Dict[str, Any] = {
        "created_utc": now,
        "passed": passed,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_validation64_shard05_postrun_audit_v2_schema_repair_not_original_SAC",
        "shard_index": SHARD_INDEX,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "validation64_bank_content_opened_by_original_shard_run": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "formal_scientific_evidence_audited": True,
        "formal_scientific_evidence_created_by_this_audit": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "repair_scope": {
            "version": "v2_schema_repair",
            "not_algorithm_change": True,
            "not_formal_shard_rerun": True,
            "variables_changed_from_v1": [
                "aggregate comparison no longer requires absent derived decision_gross_mean_s_per_step",
                "registry closed-test-budget check accepts missing test_accessed key only with independent closed-test flags and zero test episodes/control steps",
                "output directory and backup request names changed to v2 to preserve failed v1 evidence",
            ],
        },
        "v1_failure_diagnosis": v1_failure_diagnosis,
        "expected": {"experiment_id": EXPECTED_EXPERIMENT_ID, "episodes": EXPECTED_EPISODES, "control_steps": EXPECTED_CONTROL_STEPS, "control_upper": EXPECTED_CONTROL_UPPER, "execution_start": EXPECTED_EXECUTION_START, "execution_end": EXPECTED_EXECUTION_END, "runner_sha256": EXPECTED_RUNNER_SHA, "gate_sha256": EXPECTED_GATE_SHA, "validation_bank_sha256": EXPECTED_VALIDATION_BANK_SHA, "pre_shard_backup_asset_sha256": EXPECTED_PRE_SHARD_ASSET_SHA},
        "top_hashes": top_hashes,
        "completed_hash_audit": completed_hash_audit,
        "episode_dirs": len(episode_dirs),
        "episode_hash_records": episode_hash_records,
        "episode_trace_hash_audit_passed": not episode_completed_failures and not trace_len_mismatches and not raw_summary_mismatches,
        "aggregate_replay": {"passed": aggregate_replay_passed, "checks": aggregate_checks},
        "registry_audit": {"path": rel(REGISTRY), "experiment_id": registry.get("experiment_id"), "exit_status": registry.get("exit_status"), "interpreter": registry.get("interpreter"), "git_commit": registry_commit, "runtime_seconds": registry.get("runtime_seconds"), "peak_process_rss_kb": registry.get("peak_process_rss_kb"), "mean_process_tree_cpu_percent_instance": registry.get("mean_process_tree_cpu_percent_instance"), "cloudwatch_status": (registry.get("cloudwatch") or {}).get("status"), "cloudwatch_values_are_unknown_not_zero": (registry.get("cloudwatch") or {}).get("status") == "unavailable", "artifact_inventory_checks": artifact_inventory_checks, "test_budget_closed_v2": registry_test_closed},
        "pre_shard_backup_proof": {"path": rel(PRE_SHARD_BACKUP_PROOF), "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)], "backup_verified": pre_proof.get("backup_verified"), "remaining_changed_files": pre_proof.get("remaining_changed_files"), "commit": pre_proof.get("commit"), "asset_sha256": pre_proof.get("asset_sha256")},
        "budget_audit": {"declared": declared, "actual": actual, "completed_marker": {"episodes": completed.get("episodes"), "control_steps": completed.get("control_steps")}, "progress": {"episodes_done": progress.get("episodes_done"), "control_steps_done": progress.get("control_steps_done")}},
        "selection_snapshot": selection_snapshot,
        "cumulative_vehicle_validation64_after_audit": cumulative,
        "interpretation_limits": [
            "Post-run audit only; no new simulations or training were run by this script.",
            "Validation outcomes are development/validation evidence only; sealed final test remains unauthorized and closed.",
            "This is IMPROVED latency-tree evidence, not ORIGINAL SAC evidence.",
            "Shard-level validation does not establish final model selection or a reproduction-success claim.",
            "The failed v1 audit is preserved and counted as an engineering audit-script failure with 0 simulations/control steps/gradient steps.",
        ],
    }
    write_json(raw_out, audit_raw)

    summary_text = """
# Vehicle validation64 shard05 post-run audit v2 schema repair

UTC: {now}

Passed: {passed}
Failures: {failures}

Validation accessed: true (existing shard05 outputs only)
Validation bank reopened by audit: false
Sealed final test accessed/hashed: false/false
New simulations/control steps/gradient steps by audit: 0/0/0

V2 repair scope: no algorithm/config/result change; v1 false positives repaired by (1) not requiring absent derived `decision_gross_mean_s_per_step` in recorded raw aggregates, and (2) accepting the shard05 run-registry closed-test schema where test budget has zero episodes/control steps, sealed_test_bank_content_opened=false, test_authorization=false, and raw/completed/run_started all have test_accessed=false. Failed v1 audit remains preserved.

Shard05 audited episodes/control steps: {episodes}/{steps} (upper {upper})
Completed hash records audited: {hash_records}
Episode directories: {episode_dirs}
Episode per-dir hash records audited: {episode_hash_records}
Episode trace/hash audit passed: {trace_passed}
Aggregate replay passed: {aggregate_passed}
Registry closed-test schema: {registry_schema}

Learned candidates in shard05:
- learned_s0: {s0}
- learned_s1: {s1}
- learned_s2: {s2}

Cumulative vehicle validation64 after this audit: {cum_shards}/12 shards, {cum_eps} episodes, {cum_steps} control steps, plus the preserved failed shard02 modern-interpreter attempt with 0 episodes/control steps/gradient steps and the failed shard05 v1 audit with 0 simulations/control steps/gradient steps.

Backup before shard06 is required; v2 request/addendum/blocker artifacts were written by this audit. This remains validation evidence only for the IMPROVED latency-tree method, not ORIGINAL SAC and not final-test evidence.
""".strip().format(
        now=now, passed=passed, failures=failures, episodes=EXPECTED_EPISODES, steps=EXPECTED_CONTROL_STEPS, upper=EXPECTED_CONTROL_UPPER,
        hash_records=completed_hash_audit.get("hash_records"), episode_dirs=len(episode_dirs), episode_hash_records=episode_hash_records,
        trace_passed=audit_raw["episode_trace_hash_audit_passed"], aggregate_passed=aggregate_replay_passed, registry_schema=registry_test_closed["schema"],
        s0=learned_horizon_audit.get("learned_s0"), s1=learned_horizon_audit.get("learned_s1"), s2=learned_horizon_audit.get("learned_s2"),
        cum_shards=len(cumulative.get("shards") or []), cum_eps=cumulative.get("episodes"), cum_steps=cumulative.get("control_steps"),
    ) + "\n"
    summary_out.write_text(summary_text, encoding="utf-8")

    request = {
        "status": "external_backup_requested_after_validation64_shard05_audit_v2_schema_repair",
        "created_utc": now,
        "backup_verified": False,
        "required_before_shard6": True,
        "reason": "Formal vehicle validation64 shard05 and its v2 repaired audit produced/preserved unique validation evidence; external recoverable backup is required before shard06.",
        "next_action_after_verified_backup": "Run vehicle_validation64_shard_runner.py --shard 6 --backup-proof <post-shard05-v2-audit-proof> --i-accept-validation-access; sealed test remains closed.",
        "required_next_backup_proof_fields_before_shard6": {"backup_verified": True, "remaining_changed_files": 0, "commit": "new commit or verified state after shard05 formal outputs, failed v1 audit, v2 audit/request/addendum/blocker state", "release_or_asset_sha256": "required from GitHub release asset/download verification", "runner_sha256": EXPECTED_RUNNER_SHA, "gate_sha256": EXPECTED_GATE_SHA, "must_cover_validation64_shard05_artifacts_failed_v1_and_v2_audit": True},
        "formal_scientific_evidence_to_preserve": True,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations_by_audit": 0,
        "new_control_steps_by_audit": 0,
        "new_gradient_steps_by_audit": 0,
        "shard05_experiment_id": EXPECTED_EXPERIMENT_ID,
        "shard05_artifact_hashes": {rel(p): sha256(p) for p in [RUNNER, PRE_SHARD_BACKUP_PROOF, GATE_JSON, RAW_JSON, SUMMARY_MD, COMPLETED_JSON, PROGRESS_JSON, SCHEDULE_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY]},
        "shard05_run_registry_stdout_stderr_hashes": {rel(p): sha256(p) for p in [x for x in [registry_stdout, registry_stderr] if x is not None and x.exists()]},
        "failed_v1_audit_hashes": {rel(p): sha256(p) for p in [V1_PATH, V1_RAW, V1_SUMMARY, V1_COMPLETED, V1_RUN_REGISTRY] + [x for x in [v1_stdout, v1_stderr] if x is not None and x.exists()]},
        "audit_v2_artifact_hashes": {rel(THIS_SCRIPT): sha256(THIS_SCRIPT), rel(raw_out): sha256(raw_out), rel(summary_out): sha256(summary_out)},
        "audit_completed_path_pending_at_request_time": rel(completed_path),
        "audit_run_registry_stdout_stderr_required_after_run_experiment_finalizes": True,
    }
    write_json(request_path, request)

    blocker_text = """
# Blocker: backup required after vehicle validation64 shard05 audit v2

UTC: {now}

Shard05 formal validation, the failed v1 audit, and the repaired v2 post-run audit have produced/preserved unique validation evidence. Do not run shard06 or any further formal validation/test simulation until a verified external backup proof exists after this v2 audit state.

Required coverage: shard05 formal outputs, failed shard05 v1 audit outputs/run registry/stdout/stderr, v2 audit raw/summary/completed artifacts, v2 audit run registry/stdout/stderr once finalized, documentation/registry updates, `{request}`, `{addendum}`, and this blocker note. Sealed final test remains closed and unauthorized.
""".strip().format(now=now, request=rel(request_path), addendum=rel(final_addendum_path)) + "\n"
    blocker_path.write_text(blocker_text, encoding="utf-8")

    doc_body = """
## 2026-09-26 vehicle validation64 shard05 audit v2 schema repair

UTC: {now}. Repaired post-run audit of formal shard05 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. V2 preserves failed v1 audit `{v1_completed}` and fixes only audit-schema false positives: absent derived gross-decision mean is auxiliary, and shard05 registry's closed-test budget is accepted because episodes/control_steps are zero, sealed_test_bank_content_opened=false, test_authorization=false, and raw/completed/run_started all show test_accessed=false. Shard05 has 224 episodes and 19660 control steps, within the declared 224/33600 budget. Completed hash audit passed={hash_passed}; episode trace/hash audit passed={trace_passed}; aggregate replay checks passed={aggregate_passed}. Learned candidates in shard05: s0 {s0}, s1 {s1}, s2 {s2}. Vehicle validation64 progress is now {cum_shards}/12 completed formal shards with {cum_eps} episodes and {cum_steps} control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and the failed v1 audit with 0 simulations. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard06; v2 request written at `{request}` and final addendum at `{addendum}`. Sealed test remains closed.
""".strip().format(
        now=now, v1_completed=rel(V1_COMPLETED), hash_passed=completed_hash_audit["passed"], trace_passed=audit_raw["episode_trace_hash_audit_passed"], aggregate_passed=aggregate_replay_passed,
        s0=learned_horizon_audit.get("learned_s0"), s1=learned_horizon_audit.get("learned_s1"), s2=learned_horizon_audit.get("learned_s2"),
        cum_shards=len(cumulative.get("shards") or []), cum_eps=cumulative.get("episodes"), cum_steps=cumulative.get("control_steps"), request=rel(request_path), addendum=rel(final_addendum_path),
    )
    docs_updated: List[str] = []
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"]:
        if append_once(ROOT / doc, DOC_MARKER, doc_body):
            docs_updated.append(doc)

    final_addendum = {
        "status": "external_backup_requested_after_validation64_shard05_audit_v2_final_addendum",
        "created_utc": now,
        "backup_verified": False,
        "required_before_shard6": True,
        "reason": "Final addendum after writing shard05 v2 audit raw/summary, documentation updates, backup request, and blocker note; external backup is required before shard06.",
        "next_action_after_verified_backup": "Run shard06 with legacy interpreter and sealed final test closed, then audit shard06 before any further shard.",
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "gate_sha256": EXPECTED_GATE_SHA,
        "pre_shard05_backup_proof": {"path": rel(PRE_SHARD_BACKUP_PROOF), "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)]},
        "validation_accessed": True,
        "validation_bank_reopened_by_this_addendum": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations_by_addendum": 0,
        "new_control_steps_by_addendum": 0,
        "new_gradient_steps_by_addendum": 0,
        "shard05_formal_artifact_hashes": request["shard05_artifact_hashes"],
        "shard05_run_registry_stdout_stderr_hashes": request["shard05_run_registry_stdout_stderr_hashes"],
        "failed_v1_audit_hashes": request["failed_v1_audit_hashes"],
        "audit_v2_artifact_hashes": {rel(THIS_SCRIPT): sha256(THIS_SCRIPT), rel(raw_out): sha256(raw_out), rel(summary_out): sha256(summary_out), rel(request_path): sha256(request_path), rel(blocker_path): sha256(blocker_path)},
        "doc_hashes_after_update": {doc: sha256(ROOT / doc) for doc in docs_updated},
        "audit_completed_path_pending_at_addendum_time": rel(completed_path),
        "audit_run_registry_stdout_stderr_required_after_run_experiment_finalizes": True,
        "must_cover_this_final_addendum_itself": True,
    }
    write_json(final_addendum_path, final_addendum)

    completed_payload = {
        "passed": passed,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "validation64_bank_content_opened_by_original_shard_run": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "formal_scientific_evidence_audited": True,
        "formal_scientific_evidence_created_by_this_audit": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "episodes_audited": EXPECTED_EPISODES,
        "control_steps_audited": EXPECTED_CONTROL_STEPS,
        "v1_failed_audit_preserved": True,
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(request_path),
        "final_addendum_path": rel(final_addendum_path),
        "blocker_path": rel(blocker_path),
        "docs_updated": docs_updated,
        "hashes": {rel(p): sha256(p) for p in [THIS_SCRIPT, raw_out, summary_out, request_path, final_addendum_path, blocker_path, V1_PATH, V1_RAW, V1_SUMMARY, V1_COMPLETED, V1_RUN_REGISTRY] + [ROOT / d for d in docs_updated]},
    }
    write_json(completed_path, completed_payload)

    result = {"completed": rel(completed_path), "completed_sha256": sha256(completed_path), "raw": rel(raw_out), "raw_sha256": sha256(raw_out), "summary": rel(summary_out), "summary_sha256": sha256(summary_out), "backup_request": rel(request_path), "backup_request_sha256": sha256(request_path), "final_addendum": rel(final_addendum_path), "final_addendum_sha256": sha256(final_addendum_path), "blocker": rel(blocker_path), "blocker_sha256": sha256(blocker_path), "passed": passed, "failures": failures, "episodes_audited": EXPECTED_EPISODES, "control_steps_audited": EXPECTED_CONTROL_STEPS, "completed_hash_records": completed_hash_audit.get("hash_records"), "episode_dirs": len(episode_dirs), "episode_hash_records": episode_hash_records, "aggregate_replay_passed": aggregate_replay_passed, "episode_trace_hash_audit_passed": audit_raw["episode_trace_hash_audit_passed"], "registry_test_budget_schema": registry_test_closed["schema"], "cumulative": cumulative, "validation_accessed": True, "validation_bank_reopened_by_this_audit": False, "test_accessed": False, "sealed_test_bank_content_opened": False, "new_simulations": 0, "new_control_steps": 0, "new_gradient_steps": 0}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
