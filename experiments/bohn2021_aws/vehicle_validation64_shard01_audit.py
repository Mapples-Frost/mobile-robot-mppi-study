#!/usr/bin/env python3
"""Post-run audit for vehicle validation64 shard01.

Versioned shard01 audit after the preregistered formal validation shard.  This
script reads already-created shard01 validation-result artifacts, so it records
validation_accessed=true.  It does not run simulations, does not train, does not
reopen/hash the validation bank directly, and does not open/hash the sealed final
test bank.  It verifies hashes, budgets, trace/summary consistency, aggregate
replay, access flags, and writes a backup request before any further formal
validation shard.
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
SHARD_INDEX = 1
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard01"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard01_audit_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T144830_43192eec/registry.json"
PRE_SHARD_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260926T144632_after_shard00_audit_v2.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_EXPERIMENT_ID = "20260926T144830_43192eec"
EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_VALIDATION_BANK_SHA = "b0ed14f2738a07975a42d5adc12e58ec3970515dea69e6338ed31d69b5fca48f"
EXPECTED_EPISODES = 224
EXPECTED_CONTROL_STEPS = 20036
EXPECTED_CONTROL_UPPER = 33600
EXPECTED_EXECUTION_START = 224
EXPECTED_EXECUTION_END = 447
DOC_MARKER = "vehicle-validation64-shard01-audit-20260926"

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


def root_path(name: str | Path) -> Path:
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


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


def verify_hash_map(hash_map: Mapping[str, str]) -> Dict[str, Any]:
    missing: List[str] = []
    mismatches: List[Dict[str, str]] = []
    checked = 0
    bytes_checked = 0
    suffix_counts: Counter[str] = Counter()
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


def fsum(values: Iterable[float]) -> float:
    return math.fsum(float(v) for v in values)


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
    horizons: Counter[str] = Counter()
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


def close_float(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def compare_aggregate(name: str, recomputed: Mapping[str, Any], recorded: Mapping[str, Any], failures: List[str]) -> Dict[str, Any]:
    keys = [
        "episodes", "steps", "success_count", "episode_failure_count", "constraint_count",
        "initial_failed_steps", "solver_failure_steps", "retries", "recovered_steps",
        "deadline_exceed_steps", "switches", "total_cost_sum", "performance_cost_sum",
        "constraint_cost_sum", "physical_constraint_cost_sum", "h_penalty_sum", "decision_total_s",
        "decision_gross_total_s", "logging_total_s", "construction_total_s", "reset_total_s",
        "decision_mean_s_per_step", "total_cost_mean_episode", "physical_constraint_cost_mean_episode",
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


def verify_preexisting_completed(path: Path) -> bool:
    done = read_json(path)
    if done.get("passed") is not True:
        return False
    for name, expected in (done.get("hashes") or {}).items():
        p = root_path(name)
        if not p.exists() or sha256(p) != expected:
            return False
    return True


def main() -> int:
    if (OUT_DIR / "completed.json").exists():
        if verify_preexisting_completed(OUT_DIR / "completed.json"):
            print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
            return 0
        raise SystemExit("prior shard01 audit exists but did not verify; inspect before rerun")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    failures: List[str] = []

    required_files = [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY, PRE_SHARD_BACKUP_PROOF, RUNNER, GATE_JSON]
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

    top_hashes = {rel(p): sha256(p) for p in required_files}
    completed_hash_audit = verify_hash_map(completed.get("hashes") or {})
    require(completed_hash_audit["passed"], failures, "top-level shard completed hash audit failed")
    require(top_hashes[rel(RUNNER)] == EXPECTED_RUNNER_SHA, failures, "runner sha mismatch")
    require(top_hashes[rel(GATE_JSON)] == EXPECTED_GATE_SHA, failures, "gate sha mismatch")

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
    require(bool(pre_proof.get("asset_sha256") or pre_proof.get("packages_this_run")), failures, "pre-shard backup proof lacks asset/package sha")

    require(registry.get("experiment_id") == EXPECTED_EXPERIMENT_ID, failures, "registry experiment id mismatch")
    require(registry.get("exit_status") == 0, failures, "registry exit status not zero")
    require(registry.get("script_sha256") == EXPECTED_RUNNER_SHA, failures, "registry script sha mismatch")
    registry_commit = registry.get("git_commit") or registry.get("commit_sha")
    require(bool(registry_commit), failures, "registry lacks git_commit/commit_sha")
    require(registry_commit == pre_proof.get("commit"), failures, "registry commit does not match pre-shard proof commit")

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
        pairs = [("execution_index", "execution_index"), ("rollout_key", "rollout_key"), ("case_index", "case")]
        for row_key, ep_key in pairs:
            if row.get(row_key) != ep.get(ep_key):
                schedule_mismatches.append({"row_execution_index": row.get("execution_index"), "key": row_key, "schedule": row.get(row_key), "episode": ep.get(ep_key)})
    require(not schedule_mismatches, failures, "schedule/raw episode mismatches: %s" % schedule_mismatches[:5])

    episode_dirs = sorted([p for p in (SHARD_DIR / "episodes").iterdir() if p.is_dir()]) if (SHARD_DIR / "episodes").exists() else []
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
    for family in sorted({e.get("family") for e in episodes}):
        eps = [e for e in episodes if e.get("family") == family]
        aggregate_checks.append(compare_aggregate("family:%s" % family, aggregate_from_episodes(eps), (raw.get("aggregates_by_family") or {}).get(family) or {}, failures))
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
            "initial_failed_steps": overall.get("initial_failed_steps"),
            "solver_failure_steps": overall.get("solver_failure_steps"),
            "retries": overall.get("retries"),
            "recovered_steps": overall.get("recovered_steps"),
            "deadline_exceed_steps": overall.get("deadline_exceed_steps"),
            "switches": overall.get("switches"),
            "total_cost_sum": overall.get("total_cost_sum"),
            "physical_constraint_cost_sum": overall.get("physical_constraint_cost_sum"),
            "decision_mean_s_per_step": overall.get("decision_mean_s_per_step"),
            "horizon_counts": overall.get("horizon_counts"),
        },
        "families": {
            key: {
                "episodes": val.get("episodes"),
                "steps": val.get("steps"),
                "success_count": val.get("success_count"),
                "episode_failure_count": val.get("episode_failure_count"),
                "total_cost_sum": val.get("total_cost_sum"),
                "physical_constraint_cost_sum": val.get("physical_constraint_cost_sum"),
                "decision_mean_s_per_step": val.get("decision_mean_s_per_step"),
                "horizon_counts": val.get("horizon_counts"),
            }
            for key, val in sorted(family.items())
        },
        "learned_rollouts": learned_horizon_audit,
        "not_model_selection": "Shard01 is only 1/12 of the preregistered validation64 block and must not be used alone for selection or final claims.",
    }

    formal_artifact_paths = [rel(p) for p in [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY, PRE_SHARD_BACKUP_PROOF, RUNNER, GATE_JSON] if p.exists()]
    formal_artifact_hashes = {p: sha256(root_path(p)) for p in formal_artifact_paths}
    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD01_AUDIT_%s.json" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S"))

    raw_audit: Dict[str, Any] = {
        "created_utc": now,
        "purpose": "post-run audit of formal vehicle validation64 shard01; no simulations, no training, no sealed-test access",
        "passed": not failures,
        "failures": failures,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_audit": False,
        "validation64_bank_content_opened_by_original_shard_run": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created_by_this_audit": False,
        "formal_scientific_evidence_audited": True,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "shard_index": SHARD_INDEX,
        "script_under_audit": {"path": rel(RUNNER), "sha256": top_hashes[rel(RUNNER)]},
        "gate": {"path": rel(GATE_JSON), "sha256": top_hashes[rel(GATE_JSON)]},
        "pre_shard_backup_proof": {"path": rel(PRE_SHARD_BACKUP_PROOF), "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)], "asset_sha256": pre_proof.get("asset_sha256"), "commit": pre_proof.get("commit")},
        "completed_hash_audit": completed_hash_audit,
        "episode_completed_audit": {
            "episode_dirs": len(episode_dirs),
            "episode_hash_records": episode_hash_records,
            "episode_completed_failures": episode_completed_failures,
            "trace_len_mismatches": trace_len_mismatches,
            "all_episode_hash_audits_passed": (not episode_completed_failures and not trace_len_mismatches and len(episode_dirs) == EXPECTED_EPISODES),
        },
        "registry_audit": {
            "path": rel(REGISTRY),
            "exit_status": registry.get("exit_status"),
            "runtime_seconds": registry.get("runtime_seconds"),
            "source_tree_sha256": registry.get("source_tree_sha256"),
            "git_commit": registry.get("git_commit"),
            "commit_sha": registry.get("commit_sha"),
            "commit_used": registry_commit,
            "peak_process_rss_kb": registry.get("peak_process_rss_kb"),
            "cloudwatch_status": (registry.get("cloudwatch") or {}).get("status"),
        },
        "budget_audit": {
            "declared": declared,
            "actual": actual,
            "completed_control_steps": completed.get("control_steps"),
            "progress": progress,
            "control_step_upper_bound_respected": int(actual.get("control_steps", 10**9)) <= EXPECTED_CONTROL_UPPER,
        },
        "access_audit": {
            "raw_flags": {k: raw.get(k) for k in ("validation_accessed", "validation64_bank_content_opened", "test_accessed", "sealed_test_bank_content_opened", "final_test_authorization_requested")},
            "completed_flags": {k: completed.get(k) for k in ("validation_accessed", "validation64_bank_content_opened", "test_accessed", "sealed_test_bank_content_opened", "formal_scientific_evidence_created")},
            "sealed_test_metadata_from_gate_only": sealed_meta,
        },
        "schedule_audit": {
            "rows": len(rows),
            "execution_index_range": [int(rows[0]["execution_index"]), int(rows[-1]["execution_index"])] if rows else None,
            "schedule_mismatches": schedule_mismatches,
            "raw_vs_episode_summary_mismatches": raw_summary_mismatches,
        },
        "aggregate_audit": {
            "aggregate_checks": aggregate_checks,
            "all_passed": all(x["passed"] for x in aggregate_checks),
        },
        "selection_relevant_snapshot": selection_snapshot,
        "formal_artifact_hashes": formal_artifact_hashes,
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
        "interpretation_limits": [
            "Shard01 is formal validation evidence but only one of twelve preregistered validation shards; no model-selection or reproduction claim is made here.",
            "This audit read already-created shard01 validation outputs; it did not run simulations, train, reopen the validation bank, or open/hash the sealed test bank.",
            "Vehicle-only IMPROVED latency-tree evidence, not ORIGINAL SAC and not whole two-task evidence.",
            "All timing is t3a.medium same-host shard timing; CloudWatch/CPU-credit values remain unavailable/unknown unless explicitly present.",
            "A verified external backup covering shard01 formal outputs, this audit, docs, registry, and backup request is required before shard02.",
        ],
    }

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    completed_path = OUT_DIR / "completed.json"
    write_json(raw_path, raw_audit)

    lines = [
        "# Vehicle validation64 shard01 post-run audit",
        "",
        "Created UTC: `%s`." % now,
        "",
        "Validation-result access: validation_accessed=true because this audit reads shard01 outputs; sealed test accessed=false. No simulations or training were run.",
        "",
        "## Gate and budget",
        "",
        "- shard run experiment: `%s`" % EXPECTED_EXPERIMENT_ID,
        "- runner SHA256: `%s`" % top_hashes[rel(RUNNER)],
        "- gate SHA256: `%s`" % top_hashes[rel(GATE_JSON)],
        "- pre-shard backup proof SHA256: `%s`" % top_hashes[rel(PRE_SHARD_BACKUP_PROOF)],
        "- completed hash audit passed: `%s` over `%s` records" % (completed_hash_audit["passed"], completed_hash_audit["hash_records"]),
        "- episode dirs: `%s`; episode trace/hash audit passed: `%s`" % (len(episode_dirs), raw_audit["episode_completed_audit"]["all_episode_hash_audits_passed"]),
        "- episodes/control steps: `%s` / `%s` (upper `%s`)" % (actual.get("episodes"), actual.get("control_steps"), declared.get("control_step_upper_bound")),
        "- runtime_seconds from registry: `%s`; peak RSS KB: `%s`; CloudWatch status: `%s`" % (registry.get("runtime_seconds"), registry.get("peak_process_rss_kb"), raw_audit["registry_audit"]["cloudwatch_status"]),
        "",
        "## Aggregate snapshot (not final selection)",
        "",
        "Overall: episodes={episodes}, steps={steps}, success={success_count}, failures={episode_failure_count}, cost_sum={total_cost_sum:.6g}, physical_sum={physical_constraint_cost_sum:.6g}, decision_mean={decision_mean_s_per_step:.6g}, horizons={horizon_counts}".format(**selection_snapshot["overall"]),
        "",
        "| family | episodes | steps | success | failures | total cost | physical cost | decision mean s/step | horizons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, val in sorted(selection_snapshot["families"].items()):
        lines.append("| `%s` | %s | %s | %s | %s | %.6g | %.6g | %.6g | %s |" % (
            key, val["episodes"], val["steps"], val["success_count"], val["episode_failure_count"],
            float(val["total_cost_sum"]), float(val["physical_constraint_cost_sum"]), float(val["decision_mean_s_per_step"]), val["horizon_counts"]
        ))
    lines.extend([
        "",
        "## Learned candidate horizon audit",
        "",
        "| rollout | episodes | steps | success | failures | switches | horizons | adaptive in this shard? |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for key, val in sorted(learned_horizon_audit.items()):
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
            key, val.get("episodes"), val.get("steps"), val.get("success_count"), val.get("episode_failure_count"),
            val.get("switches"), val.get("horizon_counts"), val.get("adaptive_in_this_shard")
        ))
    lines.extend([
        "",
        "## Failures / blockers",
        "",
        "- audit passed: `%s`" % raw_audit["passed"],
        "- failures: `%s`" % failures,
        "- backup required before more formal validation: `true`",
        "- sealed final test remains unauthorized/closed.",
        "",
        "Next permitted action after verified external backup: continue with `vehicle_validation64_shard_runner.py --shard 2` using the same frozen gate/proof pattern, then audit shard02 before any further shard.",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_request = {
        "status": "external_backup_requested_after_validation64_shard01_audit",
        "backup_verified": False,
        "created_utc": now,
        "reason": "Formal vehicle validation64 shard01 produced unique validation evidence; external recoverable backup is required before accumulating shard02.",
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations_by_request": 0,
        "formal_scientific_evidence_to_preserve": True,
        "shard01_experiment_id": EXPECTED_EXPERIMENT_ID,
        "shard01_artifact_hashes": formal_artifact_hashes,
        "audit_artifact_hashes": {rel(raw_path): sha256(raw_path), rel(summary_path): sha256(summary_path)},
        "required_before_shard2": True,
        "required_next_backup_proof_fields_before_shard2": {
            "backup_verified": True,
            "remaining_changed_files": 0,
            "commit": "new commit or verified state after shard01 formal outputs and this audit/request",
            "release_or_asset_sha256": "required from GitHub release asset/download verification",
            "runner_sha256": EXPECTED_RUNNER_SHA,
            "gate_sha256": EXPECTED_GATE_SHA,
            "must_cover_validation64_shard01_artifacts_and_this_audit": True,
        },
        "next_action_after_verified_backup": "Run vehicle_validation64_shard_runner.py --shard 2 --backup-proof <post-shard01-proof> --i-accept-validation-access; sealed test remains closed.",
    }
    write_json(backup_request_path, backup_request)
    raw_audit["backup_request_sha256"] = sha256(backup_request_path)
    raw_audit["backup_request_path"] = rel(backup_request_path)
    write_json(raw_path, raw_audit)

    doc_body = """
## 2026-09-26 vehicle validation64 shard01 audit

UTC: {now}. Post-run audit of formal shard01 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard01 has {episodes} episodes and {steps} control steps, within the declared 224/33600 budget. Completed hash audit passed={hash_passed}; episode trace/hash audit passed={episode_passed}; aggregate replay checks passed={aggregate_passed}. Learned candidates in shard01: s0 fixed H25 ({s0_steps} steps), s1 fixed H25 ({s1_steps} steps), s2 used horizons {s2_horizons} with {s2_switches} switches. This is only 2/12 validation evidence and not final model selection. New formal evidence requires external backup before shard02; request written at `{backup_request}`. Sealed test remains closed.
""".format(
        now=now,
        episodes=actual.get("episodes"),
        steps=actual.get("control_steps"),
        hash_passed=completed_hash_audit["passed"],
        episode_passed=raw_audit["episode_completed_audit"]["all_episode_hash_audits_passed"],
        aggregate_passed=raw_audit["aggregate_audit"]["all_passed"],
        s0_steps=learned_horizon_audit.get("learned_s0", {}).get("steps"),
        s1_steps=learned_horizon_audit.get("learned_s1", {}).get("steps"),
        s2_horizons=learned_horizon_audit.get("learned_s2", {}).get("horizon_counts"),
        s2_switches=learned_horizon_audit.get("learned_s2", {}).get("switches"),
        backup_request=rel(backup_request_path),
    ).strip()
    docs_updated = []
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists() and append_once(path, DOC_MARKER, doc_body):
            docs_updated.append(name)

    completed_payload = {
        "passed": raw_audit["passed"],
        "validation_accessed": True,
        "validation64_bank_content_opened_by_original_shard_run": True,
        "validation_bank_reopened_by_this_audit": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed_by_this_audit": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created_by_this_audit": False,
        "formal_scientific_evidence_audited": True,
        "episodes_audited": actual.get("episodes"),
        "control_steps_audited": actual.get("control_steps"),
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
        "docs_updated": docs_updated,
        "hashes": {rel(raw_path): sha256(raw_path), rel(summary_path): sha256(summary_path), rel(backup_request_path): sha256(backup_request_path)},
    }
    write_json(completed_path, completed_payload)

    print(json.dumps({
        "passed": raw_audit["passed"],
        "audit_raw": rel(raw_path),
        "audit_summary": rel(summary_path),
        "audit_completed": rel(completed_path),
        "backup_request": rel(backup_request_path),
        "validation_accessed": True,
        "test_accessed": False,
        "new_simulations": 0,
        "episodes_audited": actual.get("episodes"),
        "control_steps_audited": actual.get("control_steps"),
        "learned_horizon_audit": learned_horizon_audit,
        "backup_required_before_more_formal_validation": True,
    }, sort_keys=True))
    return 0 if raw_audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
