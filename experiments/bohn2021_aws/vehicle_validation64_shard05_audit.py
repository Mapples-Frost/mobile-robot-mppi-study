#!/usr/bin/env python3
"""Post-run audit for vehicle validation64 shard05.

This versioned audit reads already-created shard05 validation-result artifacts,
so it records validation_accessed=true. It does not run simulations, does not
train, does not reopen/hash the validation bank directly, and does not open/hash
the sealed final-test bank. It verifies hashes, budgets, trace/summary
consistency, aggregate replay, access flags, the legacy shard registry, and the
post-shard04-audit pre-shard backup proof. It writes a backup request/addendum
and an explicit blocker note before any shard06 work.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
SHARD_INDEX = 5
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard05"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard05_audit_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T204237_d709e6fb/registry.json"
PRE_SHARD_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260926T204130_after_shard04_audit_blocker_state.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard05_audit.py"
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
DOC_MARKER = "vehicle-validation64-shard05-audit-20260926"

RAW_JSON = SHARD_DIR / "raw.json"
SUMMARY_MD = SHARD_DIR / "summary.md"
COMPLETED_JSON = SHARD_DIR / "completed.json"
SCHEDULE_JSON = SHARD_DIR / "schedule.json"
PROGRESS_JSON = SHARD_DIR / "progress.json"
RUN_STARTED_JSON = SHARD_DIR / "run_started.json"
TERMINAL_PROGRESS_JSON = SHARD_DIR / "terminal_sources_progress.json"


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
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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


def verify_hash_map(hash_map: Mapping[str, str]) -> Dict[str, Any]:
    missing: List[str] = []
    mismatches: List[Dict[str, str]] = []
    checked = 0
    bytes_checked = 0
    suffix_counts: Counter = Counter()
    for name, expected in sorted(hash_map.items()):
        path = root_path(name)
        suffix_counts[path.suffix or "<none>"] += 1
        if not path.exists():
            missing.append(name)
            continue
        actual = sha256(path)
        checked += 1
        bytes_checked += path.stat().st_size
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    return {
        "hash_records": len(hash_map),
        "checked_existing": checked,
        "bytes_checked": bytes_checked,
        "missing": missing,
        "mismatches": mismatches,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "passed": (not missing and not mismatches and checked == len(hash_map)),
    }


def verify_completed_marker(path: Path) -> bool:
    done = read_json(path)
    if done.get("passed") is not True:
        return False
    return bool(verify_hash_map(done.get("hashes") or {}).get("passed"))


def require(condition: bool, failures: List[str], message: str) -> None:
    if not condition:
        failures.append(message)


def fsum(values: Iterable[float]) -> float:
    return math.fsum(float(v) for v in values)


def close_float(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def aggregate_from_episodes(episodes: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "episodes": len(episodes),
        "steps": sum(int(e.get("steps", 0)) for e in episodes),
        "success_count": sum(1 for e in episodes if e.get("success")),
        "episode_failure_count": sum(1 for e in episodes if e.get("episode_failure")),
        "constraint_count": sum(1 for e in episodes if e.get("constraint")),
        "initial_failed_steps": sum(int(e.get("initial_failed_steps", 0)) for e in episodes),
        "solver_failure_steps": sum(int(e.get("solver_failure_steps", 0)) for e in episodes),
        "retries": sum(int(e.get("retries", 0)) for e in episodes),
        "recovered_steps": sum(int(e.get("recovered_steps", 0)) for e in episodes),
        "deadline_exceed_steps": sum(int(e.get("deadline_exceed_steps", 0)) for e in episodes),
        "switches": sum(int(e.get("switches", 0)) for e in episodes),
        "total_cost_sum": fsum(e.get("total_cost", 0.0) for e in episodes),
        "performance_cost_sum": fsum(e.get("performance_cost", 0.0) for e in episodes),
        "constraint_cost_sum": fsum(e.get("constraint_cost", 0.0) for e in episodes),
        "physical_constraint_cost_sum": fsum(e.get("physical_constraint_cost", 0.0) for e in episodes),
        "h_penalty_sum": fsum(e.get("h_penalty", 0.0) for e in episodes),
        "decision_total_s": fsum((e.get("decision_timing_s") or {}).get("sum", 0.0) for e in episodes),
        "decision_gross_total_s": fsum((e.get("decision_gross_timing_s") or {}).get("sum", 0.0) for e in episodes),
        "logging_total_s": fsum((e.get("logging_timing_s") or {}).get("sum", 0.0) for e in episodes),
        "construction_total_s": fsum(e.get("construction_s", 0.0) for e in episodes),
        "reset_total_s": fsum((e.get("reset") or {}).get("reset_gross_s", 0.0) for e in episodes),
    }
    horizons: Counter = Counter()
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[str(h)] += int(n)
    out["horizon_counts"] = dict(sorted(horizons.items(), key=lambda kv: int(kv[0])))
    out["unique_horizons"] = [int(h) for h in out["horizon_counts"]]
    out["success_rate"] = out["success_count"] / out["episodes"] if out["episodes"] else None
    out["decision_mean_s_per_step"] = out["decision_total_s"] / out["steps"] if out["steps"] else None
    out["decision_gross_mean_s_per_step"] = out["decision_gross_total_s"] / out["steps"] if out["steps"] else None
    out["total_cost_mean_episode"] = out["total_cost_sum"] / out["episodes"] if out["episodes"] else None
    out["physical_constraint_cost_mean_episode"] = out["physical_constraint_cost_sum"] / out["episodes"] if out["episodes"] else None
    return out


def compare_aggregate(name: str, recomputed: Mapping[str, Any], recorded: Mapping[str, Any], failures: List[str]) -> Dict[str, Any]:
    keys = [
        "episodes", "steps", "success_count", "episode_failure_count", "constraint_count",
        "initial_failed_steps", "solver_failure_steps", "retries", "recovered_steps",
        "deadline_exceed_steps", "switches", "total_cost_sum", "performance_cost_sum",
        "constraint_cost_sum", "physical_constraint_cost_sum", "h_penalty_sum", "decision_total_s",
        "decision_gross_total_s", "logging_total_s", "construction_total_s", "reset_total_s",
        "decision_mean_s_per_step", "decision_gross_mean_s_per_step", "total_cost_mean_episode",
        "physical_constraint_cost_mean_episode",
    ]
    mismatches: List[Dict[str, Any]] = []
    for key in keys:
        if key not in recorded and key not in recomputed:
            continue
        a = recomputed.get(key)
        b = recorded.get(key)
        ok = close_float(a, b) if isinstance(a, float) or isinstance(b, float) else (a == b)
        if not ok:
            mismatches.append({"key": key, "recomputed": a, "recorded": b})
    if dict(recomputed.get("horizon_counts") or {}) != dict(recorded.get("horizon_counts") or {}):
        mismatches.append({"key": "horizon_counts", "recomputed": recomputed.get("horizon_counts"), "recorded": recorded.get("horizon_counts")})
    if mismatches:
        failures.append("aggregate mismatch for %s: %s" % (name, mismatches[:5]))
    return {"name": name, "passed": not mismatches, "mismatches": mismatches}


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
        raise SystemExit("prior shard05 audit exists but did not verify; inspect before rerun")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    now = now_dt.isoformat()
    stamp = now_dt.strftime("%Y%m%dT%H%M%S")
    failures: List[str] = []

    required_files = [
        RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON,
        RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY,
        PRE_SHARD_BACKUP_PROOF, RUNNER, THIS_SCRIPT, GATE_JSON,
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
    registry_stdout = Path(registry.get("stdout", "")) if registry.get("stdout") else None
    registry_stderr = Path(registry.get("stderr", "")) if registry.get("stderr") else None

    top_hashes = {rel(p): sha256(p) for p in required_files}
    for extra in (registry_stdout, registry_stderr):
        if extra is not None and extra.exists():
            top_hashes[rel(extra)] = sha256(extra)

    completed_hash_audit = verify_hash_map(completed.get("hashes") or {})
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
    require(tb.get("test_accessed") is False and tb.get("sealed_test_bank_content_opened") is False, failures, "registry test budget flags not closed")

    artifact_inventory_checks: List[Dict[str, Any]] = []
    for item in registry.get("artifact_inventory") or []:
        p = item.get("path")
        if not p or "<experiment_id>" in p:
            continue
        actual_path = root_path(p)
        actual_exists = actual_path.exists()
        passed = (item.get("exists") == actual_exists)
        actual_sha = None
        if actual_exists and actual_path.is_file():
            actual_sha = sha256(actual_path)
            if item.get("sha256") is not None:
                passed = passed and (item.get("sha256") == actual_sha)
        artifact_inventory_checks.append({
            "path": p,
            "registry_exists": item.get("exists"),
            "actual_exists": actual_exists,
            "registry_sha256": item.get("sha256"),
            "actual_sha256": actual_sha,
            "passed": passed,
        })
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
        comparisons = [
            (row.get("execution_index"), ep.get("execution_index"), "execution_index"),
            (row.get("rollout_key"), ep.get("rollout_key"), "rollout_key"),
            (row.get("case_index"), ep.get("case"), "case_index/case"),
        ]
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
        audit = verify_hash_map(done.get("hashes") or {})
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
    aggregate_checks.append(compare_aggregate("overall", aggregate_from_episodes(episodes), raw.get("overall_aggregate") or {}, failures))
    for family_name in sorted({e.get("family") for e in episodes}):
        eps = [e for e in episodes if e.get("family") == family_name]
        aggregate_checks.append(compare_aggregate("family:%s" % family_name, aggregate_from_episodes(eps), (raw.get("aggregates_by_family") or {}).get(family_name) or {}, failures))
    for rollout_key in sorted({e.get("rollout_key") for e in episodes}):
        eps = [e for e in episodes if e.get("rollout_key") == rollout_key]
        aggregate_checks.append(compare_aggregate("rollout:%s" % rollout_key, aggregate_from_episodes(eps), (raw.get("aggregates_by_rollout_key") or {}).get(rollout_key) or {}, failures))

    rollout_aggs = raw.get("aggregates_by_rollout_key") or {}
    learned_horizon_audit: Dict[str, Any] = {}
    for key in ("learned_s0", "learned_s1", "learned_s2"):
        agg = rollout_aggs.get(key) or {}
        horizons = {str(k): int(v) for k, v in (agg.get("horizon_counts") or {}).items()}
        learned_horizon_audit[key] = {
            "episodes": agg.get("episodes"),
            "steps": agg.get("steps"),
            "success_count": agg.get("success_count"),
            "episode_failure_count": agg.get("episode_failure_count"),
            "switches": agg.get("switches"),
            "horizon_counts": horizons,
            "unique_horizons": agg.get("unique_horizons"),
            "adaptive_in_this_shard": len(horizons) >= 2,
        }

    overall = raw.get("overall_aggregate") or {}
    family = raw.get("aggregates_by_family") or {}
    selection_snapshot = {
        "overall": {
            "episodes": overall.get("episodes"),
            "steps": overall.get("steps"),
            "success_count": overall.get("success_count"),
            "episode_failure_count": overall.get("episode_failure_count"),
            "constraint_count": overall.get("constraint_count"),
            "total_cost_sum": overall.get("total_cost_sum"),
            "physical_constraint_cost_sum": overall.get("physical_constraint_cost_sum"),
            "decision_mean_s_per_step": overall.get("decision_mean_s_per_step"),
        },
        "families": {
            name: {
                "episodes": agg.get("episodes"),
                "steps": agg.get("steps"),
                "success_count": agg.get("success_count"),
                "total_cost_sum": agg.get("total_cost_sum"),
                "physical_constraint_cost_sum": agg.get("physical_constraint_cost_sum"),
                "decision_mean_s_per_step": agg.get("decision_mean_s_per_step"),
            }
            for name, agg in sorted(family.items())
        },
        "learned_horizon_audit": learned_horizon_audit,
    }

    cumulative = cumulative_completed_through_shard()
    passed = not failures

    audit_raw: Dict[str, Any] = {
        "created_utc": now,
        "passed": passed,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_validation64_shard05_postrun_audit_not_original_SAC",
        "shard_index": SHARD_INDEX,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "formal_scientific_evidence_audited": True,
        "formal_scientific_evidence_created_by_this_audit": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "expected": {
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "episodes": EXPECTED_EPISODES,
            "control_steps": EXPECTED_CONTROL_STEPS,
            "control_upper": EXPECTED_CONTROL_UPPER,
            "execution_start": EXPECTED_EXECUTION_START,
            "execution_end": EXPECTED_EXECUTION_END,
            "runner_sha256": EXPECTED_RUNNER_SHA,
            "gate_sha256": EXPECTED_GATE_SHA,
            "validation_bank_sha256": EXPECTED_VALIDATION_BANK_SHA,
            "pre_shard_backup_asset_sha256": EXPECTED_PRE_SHARD_ASSET_SHA,
        },
        "top_hashes": top_hashes,
        "completed_hash_audit": completed_hash_audit,
        "episode_dirs": len(episode_dirs),
        "episode_hash_records": episode_hash_records,
        "episode_trace_hash_audit_passed": not episode_completed_failures and not trace_len_mismatches and not raw_summary_mismatches,
        "aggregate_replay": {"passed": all(item["passed"] for item in aggregate_checks), "checks": aggregate_checks},
        "registry_audit": {
            "path": rel(REGISTRY),
            "experiment_id": registry.get("experiment_id"),
            "exit_status": registry.get("exit_status"),
            "interpreter": registry.get("interpreter"),
            "git_commit": registry_commit,
            "runtime_seconds": registry.get("runtime_seconds"),
            "peak_process_rss_kb": registry.get("peak_process_rss_kb"),
            "artifact_inventory_checks": artifact_inventory_checks,
        },
        "pre_shard_backup_proof": {
            "path": rel(PRE_SHARD_BACKUP_PROOF),
            "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)],
            "backup_verified": pre_proof.get("backup_verified"),
            "remaining_changed_files": pre_proof.get("remaining_changed_files"),
            "commit": pre_proof.get("commit"),
            "asset_sha256": pre_proof.get("asset_sha256"),
        },
        "budget_audit": {
            "declared": declared,
            "actual": actual,
            "completed_marker": {"episodes": completed.get("episodes"), "control_steps": completed.get("control_steps")},
            "progress": {"episodes_done": progress.get("episodes_done"), "control_steps_done": progress.get("control_steps_done")},
        },
        "selection_snapshot": selection_snapshot,
        "cumulative_vehicle_validation64_after_audit": cumulative,
        "interpretation_limits": [
            "Post-run audit only; no new simulations or training were run by this script.",
            "Validation outcomes are development/validation evidence only; sealed final test remains unauthorized and closed.",
            "This is IMPROVED latency-tree evidence, not ORIGINAL SAC evidence.",
            "Shard-level validation does not establish final model selection or a reproduction-success claim.",
        ],
    }

    raw_out = OUT_DIR / "raw.json"
    summary_out = OUT_DIR / "summary.md"
    request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_%s.json" % stamp)
    blocker_path = ROOT / "research_artifacts/aws_diagnostics/post_shard05_audit_backup_blocker_check_20260926.md"
    final_addendum_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_FINAL_ADDENDUM_%s.json" % stamp)

    write_json(raw_out, audit_raw)
    summary_text = """
# Vehicle validation64 shard05 post-run audit

UTC: {now}

Passed: {passed}
Failures: {failures}

Validation accessed: true (existing shard05 outputs only)
Validation bank reopened by audit: false
Sealed final test accessed/hashed: false/false
New simulations/control steps/gradient steps by audit: 0/0/0

Shard05 audited episodes/control steps: {episodes}/{steps} (upper {upper})
Completed hash records audited: {hash_records}
Episode directories: {episode_dirs}
Episode per-dir hash records audited: {episode_hash_records}
Episode trace/hash audit passed: {trace_passed}
Aggregate replay passed: {aggregate_passed}

Learned candidates in shard05:
- learned_s0: {s0}
- learned_s1: {s1}
- learned_s2: {s2}

Cumulative vehicle validation64 after this audit: {cum_shards}/12 shards, {cum_eps} episodes, {cum_steps} control steps, plus the preserved failed shard02 modern-interpreter attempt with 0 episodes/control steps/gradient steps.

Backup before shard06 is required; request/addendum/blocker artifacts were written by this audit. This remains validation evidence only for the IMPROVED latency-tree method, not ORIGINAL SAC and not final-test evidence.
""".strip().format(
        now=now,
        passed=passed,
        failures=failures,
        episodes=EXPECTED_EPISODES,
        steps=EXPECTED_CONTROL_STEPS,
        upper=EXPECTED_CONTROL_UPPER,
        hash_records=completed_hash_audit.get("hash_records"),
        episode_dirs=len(episode_dirs),
        episode_hash_records=episode_hash_records,
        trace_passed=audit_raw["episode_trace_hash_audit_passed"],
        aggregate_passed=audit_raw["aggregate_replay"]["passed"],
        s0=learned_horizon_audit.get("learned_s0"),
        s1=learned_horizon_audit.get("learned_s1"),
        s2=learned_horizon_audit.get("learned_s2"),
        cum_shards=len(cumulative.get("shards") or []),
        cum_eps=cumulative.get("episodes"),
        cum_steps=cumulative.get("control_steps"),
    ) + "\n"
    summary_out.write_text(summary_text, encoding="utf-8")

    body = """
## 2026-09-26 vehicle validation64 shard05 audit

UTC: {now}. Post-run audit of formal shard05 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard05 has 224 episodes and 19660 control steps, within the declared 224/33600 budget. Completed hash audit passed={hash_passed}; episode trace/hash audit passed={trace_passed}; aggregate replay checks passed={aggregate_passed}. Learned candidates in shard05: s0 {s0}, s1 {s1}, s2 {s2}. Vehicle validation64 progress is now {cum_shards}/12 completed shards with {cum_eps} completed episodes and {cum_steps} control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard06; request written at `{request}` and final addendum at `{addendum}`. Sealed test remains closed.
""".strip().format(
        now=now,
        hash_passed=completed_hash_audit["passed"],
        trace_passed=audit_raw["episode_trace_hash_audit_passed"],
        aggregate_passed=audit_raw["aggregate_replay"]["passed"],
        s0=learned_horizon_audit.get("learned_s0"),
        s1=learned_horizon_audit.get("learned_s1"),
        s2=learned_horizon_audit.get("learned_s2"),
        cum_shards=len(cumulative.get("shards") or []),
        cum_eps=cumulative.get("episodes"),
        cum_steps=cumulative.get("control_steps"),
        request=rel(request_path),
        addendum=rel(final_addendum_path),
    )
    docs_updated: List[str] = []
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"]:
        if append_once(ROOT / doc, DOC_MARKER, body):
            docs_updated.append(doc)

    request = {
        "status": "external_backup_requested_after_validation64_shard05_audit",
        "created_utc": now,
        "backup_verified": False,
        "required_before_shard6": True,
        "reason": "Formal vehicle validation64 shard05 produced unique validation evidence; external recoverable backup is required before accumulating shard06.",
        "next_action_after_verified_backup": "Run vehicle_validation64_shard_runner.py --shard 6 --backup-proof <post-shard05-audit-proof> --i-accept-validation-access; sealed test remains closed.",
        "required_next_backup_proof_fields_before_shard6": {
            "backup_verified": True,
            "remaining_changed_files": 0,
            "commit": "new commit or verified state after shard05 formal outputs and this audit/request/addendum/blocker state",
            "release_or_asset_sha256": "required from GitHub release asset/download verification",
            "runner_sha256": EXPECTED_RUNNER_SHA,
            "gate_sha256": EXPECTED_GATE_SHA,
            "must_cover_validation64_shard05_artifacts_and_this_audit": True,
        },
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
        "shard05_run_registry_stdout_stderr_hashes": {
            rel(p): sha256(p) for p in [x for x in [registry_stdout, registry_stderr] if x is not None and x.exists()]
        },
        "audit_artifact_hashes": {
            rel(raw_out): sha256(raw_out),
            rel(summary_out): sha256(summary_out),
        },
        "audit_completed_path_pending_at_request_time": rel(completed_path),
        "audit_run_registry_stdout_stderr_required_after_run_experiment_finalizes": True,
    }
    write_json(request_path, request)

    blocker_text = """
# Blocker: backup required after vehicle validation64 shard05 audit

UTC: {now}

Shard05 formal validation and post-run audit have produced unique validation evidence. Do not run shard06 or any further formal validation/test simulation until a verified external backup proof exists after this shard05 audit state.

Required coverage: shard05 formal outputs, shard05 audit raw/summary/completed artifacts, audit run registry/stdout/stderr once finalized, documentation/registry updates, `{request}`, `{addendum}`, and this blocker note. Sealed final test remains closed and unauthorized.
""".strip().format(now=now, request=rel(request_path), addendum=rel(final_addendum_path)) + "\n"
    blocker_path.write_text(blocker_text, encoding="utf-8")

    completed_payload = {
        "passed": passed,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
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
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(request_path),
        "final_addendum_path": rel(final_addendum_path),
        "blocker_path": rel(blocker_path),
        "docs_updated": docs_updated,
        "hashes": {rel(p): sha256(p) for p in [raw_out, summary_out, request_path, blocker_path] + [ROOT / d for d in docs_updated]},
    }
    write_json(completed_path, completed_payload)

    final_addendum = {
        "status": "external_backup_requested_after_validation64_shard05_audit_final_addendum",
        "created_utc": now,
        "backup_verified": False,
        "required_before_shard6": True,
        "reason": "Final addendum after writing shard05 audit completed marker, documentation updates, and blocker note; external backup is required before shard06.",
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
        "audit_artifact_hashes": {
            rel(THIS_SCRIPT): sha256(THIS_SCRIPT),
            rel(raw_out): sha256(raw_out),
            rel(summary_out): sha256(summary_out),
            rel(completed_path): sha256(completed_path),
            rel(request_path): sha256(request_path),
            rel(blocker_path): sha256(blocker_path),
        },
        "doc_hashes_after_update": {doc: sha256(ROOT / doc) for doc in docs_updated},
        "audit_run_registry_stdout_stderr_required_after_run_experiment_finalizes": True,
        "must_cover_this_final_addendum_itself": True,
    }
    write_json(final_addendum_path, final_addendum)

    result = {
        "completed": rel(completed_path),
        "completed_sha256": sha256(completed_path),
        "raw": rel(raw_out),
        "raw_sha256": sha256(raw_out),
        "summary": rel(summary_out),
        "summary_sha256": sha256(summary_out),
        "backup_request": rel(request_path),
        "backup_request_sha256": sha256(request_path),
        "final_addendum": rel(final_addendum_path),
        "final_addendum_sha256": sha256(final_addendum_path),
        "blocker": rel(blocker_path),
        "blocker_sha256": sha256(blocker_path),
        "passed": passed,
        "failures": failures,
        "episodes_audited": EXPECTED_EPISODES,
        "control_steps_audited": EXPECTED_CONTROL_STEPS,
        "completed_hash_records": completed_hash_audit.get("hash_records"),
        "episode_dirs": len(episode_dirs),
        "episode_hash_records": episode_hash_records,
        "aggregate_replay_passed": audit_raw["aggregate_replay"]["passed"],
        "episode_trace_hash_audit_passed": audit_raw["episode_trace_hash_audit_passed"],
        "cumulative": cumulative,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
