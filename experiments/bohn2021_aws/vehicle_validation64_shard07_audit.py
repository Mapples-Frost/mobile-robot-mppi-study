#!/usr/bin/env python3
"""Post-run audit for vehicle validation64 shard07.

Reads only already-created shard07 result artifacts. No simulation, no
training, no validation-bank reopen, and no sealed-test open/hash. The purpose
is to verify trace/hash/budget/access consistency after the formal shard07 run
and to write the post-audit backup request/blocker required before shard08.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
SHARD_INDEX = 7
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard07_audit_20260927"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T233804_e600fab4/registry.json"
PRE_SHARD_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260926T233653_after_shard06_audit_gate_recheck.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard07_audit.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_EXPERIMENT_ID = "20260926T233804_e600fab4"
EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_VALIDATION_BANK_SHA = "b0ed14f2738a07975a42d5adc12e58ec3970515dea69e6338ed31d69b5fca48f"
EXPECTED_PRE_SHARD_ASSET_SHA = "a70c3288a927b56f0e2e38d6ba0abb77a22606a4bd7eb664c21ac435d9dc860a"
EXPECTED_PRE_SHARD_BACKUP_PROOF_SHA = "5040edbb9030e3e57fd476718ec27daf7b045a5c1cc03cd84dd6e31ef5f9d43c"
EXPECTED_EPISODES = 224
EXPECTED_CONTROL_STEPS = 19532
EXPECTED_CONTROL_UPPER = 33600
EXPECTED_EXECUTION_START = 1568
EXPECTED_EXECUTION_END = 1791
DOC_MARKER = "vehicle-validation64-shard07-audit-20260927"

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


def fsum(values: Iterable[float]) -> float:
    return math.fsum(float(v) for v in values)


def close_float(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


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
    try:
        done = read_json(path)
    except Exception:
        return False
    return bool(done.get("passed") is True and verify_hash_map(done.get("hashes") or {}).get("passed"))


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
    required_keys = [
        "episodes", "steps", "success_count", "episode_failure_count", "constraint_count",
        "initial_failed_steps", "solver_failure_steps", "retries", "recovered_steps",
        "deadline_exceed_steps", "switches", "total_cost_sum", "performance_cost_sum",
        "constraint_cost_sum", "physical_constraint_cost_sum", "h_penalty_sum", "decision_total_s",
        "decision_gross_total_s", "logging_total_s", "construction_total_s", "reset_total_s",
        "decision_mean_s_per_step", "total_cost_mean_episode", "physical_constraint_cost_mean_episode",
    ]
    mismatches: List[Dict[str, Any]] = []
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
    if mismatches:
        failures.append("aggregate mismatch for %s: %s" % (name, mismatches[:5]))
    return {"name": name, "passed": not mismatches, "mismatches": mismatches}


def registry_test_budget_closed(tb: Mapping[str, Any], raw: Mapping[str, Any], completed: Mapping[str, Any], run_started: Mapping[str, Any]) -> Dict[str, Any]:
    episodes_zero = int(tb.get("episodes", tb.get("test_episodes", 0)) or 0) == 0
    control_zero = int(tb.get("control_steps", 0) or 0) == 0
    sealed_false = tb.get("sealed_test_bank_content_opened") is False
    registry_test_accessed_field = tb.get("test_accessed", None)
    auth_closed = (tb.get("test_authorization") is False) or (tb.get("final_test_authorization") is False) or (tb.get("final_test_authorization_requested") is False) or (registry_test_accessed_field is False)
    raw_closed = raw.get("test_accessed") is False and raw.get("sealed_test_bank_content_opened") is False
    completed_closed = completed.get("test_accessed") is False and completed.get("sealed_test_bank_content_opened") is False
    run_started_closed = run_started.get("test_accessed") is False and run_started.get("sealed_test_bank_content_opened") is False
    explicit_closed = registry_test_accessed_field is False and episodes_zero and control_zero and sealed_false
    accepted_missing = registry_test_accessed_field is None and episodes_zero and control_zero and sealed_false and auth_closed and raw_closed and completed_closed and run_started_closed
    return {
        "passed": bool(explicit_closed or accepted_missing),
        "schema": "explicit_test_accessed_false" if explicit_closed else ("closed_budget_missing_test_accessed_key_accepted" if accepted_missing else "not_closed"),
        "registry_test_budget": dict(tb),
        "raw_closed": raw_closed,
        "completed_closed": completed_closed,
        "run_started_closed": run_started_closed,
    }


def doc_hashes(names: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in names:
        path = ROOT / name
        if path.exists():
            out[name] = sha256(path)
    return out


def write_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    learned = audit["learned_horizon_audit"]
    lines = [
        "# Vehicle validation64 shard07 post-run audit",
        "",
        "Created UTC: `%s`." % audit["created_utc"],
        "",
        "- Passed: `%s`." % audit["passed"],
        "- Validation accessed: `true` only by reading existing shard07 outputs; validation64 bank reopened: `false`.",
        "- Sealed final test accessed/opened/hashed: `false` / `false` / `false`.",
        "- New simulations/control steps/training steps by audit: `0` / `0` / `0`.",
        "- Shard07 formal episodes/control steps audited: `%s` / `%s` (upper bound `%s`)." % (audit["episodes_audited"], audit["control_steps_audited"], audit["control_step_upper_bound"]),
        "- Completed hash audit: `%s`; episode trace/hash audit: `%s`; aggregate replay: `%s`." % (audit["completed_hash_audit_passed"], audit["episode_trace_hash_audit_passed"], audit["aggregate_replay_passed"]),
        "",
        "## Learned candidate horizon audit",
        "",
        "| candidate | episodes | steps | successes | horizon counts | switches | adaptive in this shard |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for key in sorted(learned):
        row = learned[key]
        lines.append("| `%s` | %s | %s | %s | `%s` | %s | `%s` |" % (key, row["episodes"], row["steps"], row["success_count"], row["horizon_counts"], row["switches"], row["adaptive_in_this_shard"]))
    lines.extend([
        "",
        "Shard07 is one shard of the preregistered vehicle validation64 block for the IMPROVED latency-tree controller, not ORIGINAL SAC and not a final reproduction claim.",
        "Shard08 is blocked until a verified external backup proof covers shard07 formal outputs, this audit, docs/registry updates, backup requests/addenda/blocker, and this audit run registry/stdout/stderr.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    completed_path = OUT_DIR / "completed.json"
    if completed_path.exists():
        if verify_completed_marker(completed_path):
            print(json.dumps({"already_completed": True, "completed": rel(completed_path)}, sort_keys=True))
            return 0
        raise SystemExit("prior shard07 audit exists but did not verify; inspect before rerun")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    now = now_dt.isoformat()
    stamp = now_dt.strftime("%Y%m%dT%H%M%S")
    failures: List[str] = []
    required_files = [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY, PRE_SHARD_BACKUP_PROOF, RUNNER, THIS_SCRIPT, GATE_JSON]
    missing = [rel(p) for p in required_files if not p.exists()]
    require(not missing, failures, "missing required files: %s" % missing)
    if missing:
        write_json(OUT_DIR / "raw.json", {"created_utc": now, "passed": False, "failures": failures, "validation_accessed": True, "validation_bank_reopened": False, "test_accessed": False})
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

    for key, expected in {"validation_accessed": True, "validation64_bank_content_opened": True, "test_accessed": False, "sealed_test_bank_content_opened": False, "formal_scientific_evidence_created": True}.items():
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
    sealed_meta = (((raw.get("gate_verification") or {}).get("bank_metadata_no_content_opened") or {}).get("sealed_test128") or {})
    require(sealed_meta.get("content_opened") is False and sealed_meta.get("sha256_computed_now") is False, failures, "sealed test metadata flags not false")
    require(str(sealed_meta.get("path", "")).endswith("vehicle_test_bank.json"), failures, "sealed test metadata path unexpected")

    require(pre_proof.get("backup_verified") is True and pre_proof.get("remaining_changed_files") == 0, failures, "pre-shard backup proof not verified/clean")
    require(pre_proof.get("runner_sha256") == EXPECTED_RUNNER_SHA and pre_proof.get("gate_sha256") == EXPECTED_GATE_SHA, failures, "pre-shard backup proof runner/gate mismatch")
    require(pre_proof.get("asset_sha256") == EXPECTED_PRE_SHARD_ASSET_SHA, failures, "pre-shard backup proof asset sha mismatch")
    require(bool(pre_proof.get("commit")), failures, "pre-shard backup proof lacks commit")

    require(registry.get("experiment_id") == EXPECTED_EXPERIMENT_ID, failures, "registry experiment id mismatch")
    require(registry.get("exit_status") == 0 and registry.get("status") == "complete", failures, "registry status/exit not complete")
    require(registry.get("script_sha256") == EXPECTED_RUNNER_SHA, failures, "registry script sha mismatch")
    require(registry.get("interpreter") == "legacy", failures, "registry interpreter is not legacy")
    registry_commit = registry.get("git_commit") or registry.get("commit_sha")
    require(bool(registry_commit), failures, "registry lacks git_commit/commit_sha")
    require(registry_commit == pre_proof.get("commit"), failures, "registry commit does not match pre-shard proof commit")
    vb = registry.get("validation_budget") or {}
    require(vb.get("episodes") == EXPECTED_EPISODES or vb.get("validation_episodes") == EXPECTED_EPISODES or vb.get("formal_validation_shard_episodes") == EXPECTED_EPISODES, failures, "registry validation budget episode mismatch")
    registry_test_closed = registry_test_budget_closed(registry.get("test_budget") or {}, raw, completed, run_started)
    require(registry_test_closed["passed"], failures, "registry test budget flags not closed")

    artifact_inventory_checks: List[Dict[str, Any]] = []
    for item in registry.get("artifact_inventory") or []:
        p = item.get("path")
        if not p or "*" in p or "<experiment_id>" in p:
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
        for a, b, key in [(row.get("execution_index"), ep.get("execution_index"), "execution_index"), (row.get("rollout_key"), ep.get("rollout_key"), "rollout_key"), (row.get("case_index"), ep.get("case"), "case_index/case")]:
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
        for key in ["rollout_key", "case", "steps", "success", "termination", "horizon_counts", "switches", "total_cost"]:
            a = ep.get(key)
            b = summary.get(key)
            ok = close_float(a, b) if isinstance(a, float) or isinstance(b, float) else (a == b)
            if not ok:
                raw_summary_mismatches.append({"execution_index": ep.get("execution_index"), "key": key, "raw": a, "summary": b})
    require(not raw_summary_mismatches, failures, "raw/episode summary mismatches: %s" % raw_summary_mismatches[:5])

    aggregate_checks: List[Dict[str, Any]] = []
    aggregate_checks.append(compare_aggregate("overall", aggregate_from_episodes(episodes), raw.get("overall_aggregate") or {}, failures))
    for key, recorded in sorted((raw.get("aggregates_by_rollout_key") or {}).items()):
        aggregate_checks.append(compare_aggregate("rollout:" + key, aggregate_from_episodes([e for e in episodes if e.get("rollout_key") == key]), recorded, failures))
    for key, recorded in sorted((raw.get("aggregates_by_family") or {}).items()):
        aggregate_checks.append(compare_aggregate("family:" + key, aggregate_from_episodes([e for e in episodes if e.get("family") == key]), recorded, failures))
    aggregate_replay_passed = all(x["passed"] for x in aggregate_checks)

    learned_horizon_audit: Dict[str, Any] = {}
    for key in ["learned_s0", "learned_s1", "learned_s2"]:
        eps = [e for e in episodes if e.get("rollout_key") == key]
        agg = aggregate_from_episodes(eps)
        learned_horizon_audit[key] = {
            "episodes": agg["episodes"],
            "steps": agg["steps"],
            "success_count": agg["success_count"],
            "episode_failure_count": agg["episode_failure_count"],
            "switches": agg["switches"],
            "horizon_counts": agg["horizon_counts"],
            "unique_horizons": agg["unique_horizons"],
            "adaptive_in_this_shard": bool(agg["switches"] or len(agg["unique_horizons"]) > 1),
        }

    current_done_shards = sorted(int(p.name[-2:]) for p in (SHARD_DIR.parent).glob("shard[0-9][0-9]") if (p / "completed.json").exists())
    cumulative_episodes = 0
    cumulative_steps = 0
    for idx in current_done_shards:
        done = read_json(SHARD_DIR.parent / ("shard%02d" % idx) / "completed.json")
        cumulative_episodes += int(done.get("episodes", 0))
        cumulative_steps += int(done.get("control_steps", 0))

    passed = not failures
    audit_raw: Dict[str, Any] = {
        "created_utc": now,
        "passed": passed,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_validation64_shard07_postrun_audit_not_original_SAC",
        "validation_accessed": True,
        "validation_access_type": "existing shard07 result artifacts read/hashed only; validation64 bank not reopened",
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "shard_index": SHARD_INDEX,
        "formal_experiment_id": EXPECTED_EXPERIMENT_ID,
        "episodes_audited": EXPECTED_EPISODES,
        "control_steps_audited": EXPECTED_CONTROL_STEPS,
        "control_step_upper_bound": EXPECTED_CONTROL_UPPER,
        "completed_hash_audit": completed_hash_audit,
        "completed_hash_audit_passed": completed_hash_audit["passed"],
        "episode_dirs": len(episode_dirs),
        "episode_hash_records": episode_hash_records,
        "episode_trace_hash_audit_passed": (not episode_completed_failures and not trace_len_mismatches and not raw_summary_mismatches),
        "aggregate_replay_passed": aggregate_replay_passed,
        "aggregate_checks_count": len(aggregate_checks),
        "schedule_mismatches": schedule_mismatches,
        "registry_test_budget_closed": registry_test_closed,
        "registry_artifact_inventory_checks": artifact_inventory_checks,
        "top_hashes": top_hashes,
        "pre_shard_backup_proof": {"path": rel(PRE_SHARD_BACKUP_PROOF), "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)], "asset_sha256": pre_proof.get("asset_sha256"), "commit": pre_proof.get("commit")},
        "learned_horizon_audit": learned_horizon_audit,
        "cumulative_vehicle_validation64_status": {"formal_shards_completed": current_done_shards, "audited_shards_passed_expected_after_this_audit": list(range(0, SHARD_INDEX + 1)) if passed else list(range(0, SHARD_INDEX)), "formal_episodes_completed": cumulative_episodes, "formal_control_steps_completed": cumulative_steps, "planned_total_shards": 12, "planned_total_validation_episodes": 2688},
        "interpretation_limits": ["Shard-level validation audit only; no final-test evidence.", "IMPROVED latency-tree method, not ORIGINAL SAC.", "No reproduction-success or model-selection conclusion from a single shard."],
    }

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    blocker_path = ROOT / "research_artifacts/aws_diagnostics/post_shard07_audit_backup_blocker_check_20260927.md"
    request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_%s.json" % stamp)
    addendum_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_FINAL_ADDENDUM_%s.json" % stamp)
    completed_out = OUT_DIR / "completed.json"

    write_json(raw_path, audit_raw)
    write_markdown(summary_path, audit_raw)

    doc_body = """
## 2026-09-27 vehicle validation64 shard07 audit

UTC: {now}. Post-run audit of formal shard07 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard07 has {episodes} episodes and {steps} control steps, within the declared 224/33600 budget. Completed hash audit passed={hash_pass}; episode trace/hash audit passed={trace_pass}; aggregate replay checks passed={agg_pass}. Learned candidates in shard07: s0 {s0}, s1 {s1}, s2 {s2}. Vehicle validation64 progress is now 8/12 completed formal shards with {cum_eps} episodes and {cum_steps} control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard08; request written at `{request}` and final addendum at `{addendum}`. Sealed test remains closed.
""".format(now=now, episodes=EXPECTED_EPISODES, steps=EXPECTED_CONTROL_STEPS, hash_pass=completed_hash_audit["passed"], trace_pass=audit_raw["episode_trace_hash_audit_passed"], agg_pass=aggregate_replay_passed, s0=learned_horizon_audit["learned_s0"], s1=learned_horizon_audit["learned_s1"], s2=learned_horizon_audit["learned_s2"], cum_eps=cumulative_episodes, cum_steps=cumulative_steps, request=rel(request_path), addendum=rel(addendum_path))
    changed_docs = []
    for name in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"]:
        if append_once(ROOT / name, DOC_MARKER, doc_body):
            changed_docs.append(name)

    blocker_text = """# Post-shard07-audit backup blocker check

Created UTC: `{now}`.

Shard07 formal validation and post-run audit have completed, but shard08 is BLOCKED until a verified external backup proof after this audit/addendum/blocker state and after the finalized audit run registry/stdout/stderr is present locally. The required proof must record `backup_verified=true`, `remaining_changed_files=0`, GitHub release asset/download SHA256 verification, runner SHA `{runner_sha}`, gate SHA `{gate_sha}`, and coverage of shard07 formal outputs, shard07 audit outputs, formal/audit run registry/stdout/stderr, docs/registry updates, backup request, final addendum, and this blocker note.

Latest adequate pre-shard proof used for shard07 was `{pre_proof}`; it is not sufficient for shard08 because it predates shard07 formal validation and this audit. Sealed final test remains closed and unauthorized. This note ran no simulations, no control steps, and no gradient steps.
""".format(now=now, runner_sha=EXPECTED_RUNNER_SHA, gate_sha=EXPECTED_GATE_SHA, pre_proof=rel(PRE_SHARD_BACKUP_PROOF))
    blocker_path.write_text(blocker_text, encoding="utf-8")

    docs_after = doc_hashes(["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"])
    request = {
        "status": "external_backup_requested_after_validation64_shard07_audit",
        "backup_verified": False,
        "created_utc": now,
        "required_before_shard8": True,
        "reason": "Preserve shard07 formal outputs, post-run audit, docs/registry updates, backup request/addendum/blocker, and finalized audit run logs before creating more unique formal validation evidence.",
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "gate_sha256": EXPECTED_GATE_SHA,
        "pre_shard07_backup_proof": {"path": rel(PRE_SHARD_BACKUP_PROOF), "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)]},
        "shard07_formal_artifact_hashes": {rel(p): sha256(p) for p in [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, PROGRESS_JSON, SCHEDULE_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY]},
        "audit_artifact_hashes": {rel(p): sha256(p) for p in [THIS_SCRIPT, raw_path, summary_path, blocker_path]},
        "doc_hashes_after_update": docs_after,
        "validation_accessed": True,
        "validation_bank_reopened_by_this_request": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations_by_request": 0,
        "new_control_steps_by_request": 0,
        "new_gradient_steps_by_request": 0,
        "next_action_after_verified_backup": "Run shard08 with legacy interpreter and sealed final test closed, then audit shard08 before any further shard.",
    }
    write_json(request_path, request)
    addendum = dict(request)
    addendum.update({
        "reason": "Final addendum after writing shard07 audit completed marker, documentation updates, and blocker note; external backup is required before shard08.",
        "backup_request": rel(request_path),
        "backup_request_sha256": sha256(request_path),
        "audit_run_registry_stdout_stderr_required_after_run_experiment_finalizes": True,
        "must_cover_this_final_addendum_itself": True,
    })
    write_json(addendum_path, addendum)

    files_for_completed = [THIS_SCRIPT, raw_path, summary_path, blocker_path, request_path, addendum_path]
    write_json(completed_out, {
        "passed": passed,
        "failures": failures,
        "created_utc": now,
        "validation_accessed": True,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "episodes_audited": EXPECTED_EPISODES,
        "control_steps_audited": EXPECTED_CONTROL_STEPS,
        "completed_hash_audit_passed": completed_hash_audit["passed"],
        "episode_trace_hash_audit_passed": audit_raw["episode_trace_hash_audit_passed"],
        "aggregate_replay_passed": aggregate_replay_passed,
        "learned_horizon_audit": learned_horizon_audit,
        "backup_request": rel(request_path),
        "backup_request_sha256": sha256(request_path),
        "final_addendum": rel(addendum_path),
        "final_addendum_sha256": sha256(addendum_path),
        "changed_docs": changed_docs,
        "hashes": {rel(p): sha256(p) for p in files_for_completed},
    })

    print(json.dumps({
        "passed": passed,
        "completed": rel(completed_out),
        "completed_sha256": sha256(completed_out),
        "raw": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "backup_request": rel(request_path),
        "backup_request_sha256": sha256(request_path),
        "final_addendum": rel(addendum_path),
        "final_addendum_sha256": sha256(addendum_path),
        "episodes_audited": EXPECTED_EPISODES,
        "control_steps_audited": EXPECTED_CONTROL_STEPS,
        "learned_horizon_audit": learned_horizon_audit,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "shard08_blocked_until_backup": True,
        "failures": failures,
    }, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
