#!/usr/bin/env python3
"""Post-run audit for vehicle validation64 shard00.

This is a metadata/result audit of the already completed formal validation
shard.  It does not run simulations, does not train, does not open the sealed
final-test bank, and does not re-open the validation bank.  Reading shard00
outputs is validation-result access and is recorded as such.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard00_audit_20260926"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260926T132324_e8a43cbf/registry.json"
BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260926T132241_after_dryrun_and_blocker_audit.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_VALIDATION_BANK_SHA = "b0ed14f2738a07975a42d5adc12e58ec3970515dea69e6338ed31d69b5fca48f"
EXPECTED_BACKUP_ASSET_SHA = "a1d8d3db38177f3e2c6d57a3924933827213143f1ddf1187c4342b6a94305de4"
EXPECTED_EXPERIMENT_ID = "20260926T132324_e8a43cbf"
EXPECTED_COMMIT = "523ec69d0986ebde3f10f24fd6dd8ad7df30e1f9"
EXPECTED_EPISODES = 224
EXPECTED_CONTROL_UPPER = 33600
DOC_MARKER = "vehicle-validation64-shard00-audit-20260926"

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


def verify_existing_completed(path: Path) -> None:
    done = read_json(path)
    if done.get("passed") is not True:
        raise RuntimeError("existing audit completed marker did not pass")
    for name, expected in sorted((done.get("hashes") or {}).items()):
        actual = sha256(root_path(name))
        if actual != expected:
            raise RuntimeError("existing audit artifact changed: %s" % name)


def assert_fresh() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_existing_completed(completed)
        raise SystemExit("vehicle validation64 shard00 audit already completed and verified; refusing to rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise RuntimeError("partial audit output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:20]))


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
        "passed": (not missing and not mismatches and checked == len(hash_map)),
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def require(condition: bool, failures: List[str], message: str) -> None:
    if not condition:
        failures.append(message)


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
        "constraint_cost_sum", "physical_constraint_cost_sum", "h_penalty_sum",
        "decision_total_s", "decision_gross_total_s", "logging_total_s",
        "construction_total_s", "reset_total_s", "decision_mean_s_per_step",
        "total_cost_mean_episode", "physical_constraint_cost_mean_episode",
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


def read_trace_len_and_last(ep_dir: Path) -> Dict[str, Any]:
    trace_path = ep_dir / "trace.json"
    trace = read_json(trace_path)
    return {
        "trace_len": len(trace),
        "last_termination": trace[-1].get("termination") if trace else None,
        "last_solver_success": trace[-1].get("solver_success") if trace else None,
    }


def summarize_rollout_for_markdown(rollout: Mapping[str, Any]) -> str:
    return (
        "episodes={episodes}, steps={steps}, success={success_count}, failures={episode_failure_count}, "
        "cost_sum={total_cost_sum:.6g}, physical_sum={physical_constraint_cost_sum:.6g}, "
        "decision_mean={decision_mean_s_per_step:.6g}, horizons={horizon_counts}"
    ).format(**rollout)


def main() -> int:
    assert_fresh()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    failures: List[str] = []

    required_files = [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY, BACKUP_PROOF, RUNNER, GATE_JSON]
    missing = [rel(p) for p in required_files if not p.exists()]
    require(not missing, failures, "missing required files: %s" % missing)
    if missing:
        raw_audit = {"created_utc": now, "passed": False, "missing_required_files": missing}
        write_json(OUT_DIR / "raw.json", raw_audit)
        raise SystemExit(2)

    raw = read_json(RAW_JSON)
    completed = read_json(COMPLETED_JSON)
    schedule = read_json(SCHEDULE_JSON)
    progress = read_json(PROGRESS_JSON)
    run_started = read_json(RUN_STARTED_JSON)
    registry = read_json(REGISTRY)
    proof = read_json(BACKUP_PROOF)

    completed_hash_audit = verify_hash_map(completed.get("hashes") or {})
    require(completed_hash_audit["passed"], failures, "top-level shard completed hash audit failed")

    proof_sha = sha256(BACKUP_PROOF)
    top_hashes = {rel(p): sha256(p) for p in [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY, BACKUP_PROOF, RUNNER, GATE_JSON]}

    require(sha256(RUNNER) == EXPECTED_RUNNER_SHA, failures, "runner sha mismatch")
    require(sha256(GATE_JSON) == EXPECTED_GATE_SHA, failures, "gate sha mismatch")
    require(proof.get("backup_verified") is True, failures, "backup proof not verified")
    require(proof.get("remaining_changed_files") == 0, failures, "backup proof remaining_changed_files not zero")
    require(proof.get("runner_sha256") == EXPECTED_RUNNER_SHA, failures, "backup proof runner sha mismatch")
    require(proof.get("gate_sha256") == EXPECTED_GATE_SHA, failures, "backup proof gate sha mismatch")
    require(proof.get("asset_sha256") == EXPECTED_BACKUP_ASSET_SHA, failures, "backup proof asset sha mismatch")

    for key, expected in {
        "validation_accessed": True,
        "validation64_bank_content_opened": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "formal_scientific_evidence_created": True,
    }.items():
        require(raw.get(key) is expected, failures, "raw flag %s expected %s got %s" % (key, expected, raw.get(key)))
        require(completed.get(key) is expected, failures, "completed flag %s expected %s got %s" % (key, expected, completed.get(key)))
    require(raw.get("final_test_authorization_requested") is False, failures, "raw final test authorization flag not false")
    require(completed.get("new_gradient_steps") == 0, failures, "completed new_gradient_steps not zero")
    require(raw.get("new_gradient_steps") == 0, failures, "raw new_gradient_steps not zero")
    require(run_started.get("test_accessed") is False and run_started.get("sealed_test_bank_content_opened") is False, failures, "run_started test flags not false")

    budget_declared = raw.get("budget_declared") or {}
    budget_actual = raw.get("budget_actual") or {}
    require(budget_declared.get("episodes_exact") == EXPECTED_EPISODES, failures, "declared episode count mismatch")
    require(budget_declared.get("control_step_upper_bound") == EXPECTED_CONTROL_UPPER, failures, "declared control upper mismatch")
    require(budget_actual.get("episodes") == EXPECTED_EPISODES, failures, "actual episode count mismatch")
    require(completed.get("episodes") == EXPECTED_EPISODES, failures, "completed episode count mismatch")
    require(int(budget_actual.get("control_steps", -1)) <= EXPECTED_CONTROL_UPPER, failures, "actual control steps exceeds upper bound")
    require(budget_actual.get("control_steps") == completed.get("control_steps"), failures, "raw/completed control steps mismatch")
    require(budget_actual.get("test_bank_cases_opened") == 0, failures, "test bank cases opened not zero")
    require(budget_declared.get("test_episodes") == 0, failures, "declared test episodes not zero")

    validation_bank = raw.get("validation_bank") or {}
    require(validation_bank.get("sha256") == EXPECTED_VALIDATION_BANK_SHA, failures, "validation bank sha in raw mismatch")
    require(str(validation_bank.get("path", "")).endswith("vehicle_validation_bank.json"), failures, "validation bank path unexpected")
    sealed_meta = (((raw.get("gate_verification") or {}).get("bank_metadata_no_content_opened") or {}).get("sealed_test128") or {})
    require(sealed_meta.get("content_opened") is False, failures, "sealed test metadata content_opened not false")
    require(sealed_meta.get("sha256_computed_now") is False, failures, "sealed test metadata sha256_computed_now not false")
    require(str(sealed_meta.get("path", "")).endswith("vehicle_test_bank.json"), failures, "sealed test metadata path missing/changed")

    require(registry.get("experiment_id") == EXPECTED_EXPERIMENT_ID, failures, "registry experiment id mismatch")
    require(registry.get("script_sha256") == EXPECTED_RUNNER_SHA, failures, "registry script sha mismatch")
    require(registry.get("git_commit") == EXPECTED_COMMIT, failures, "registry commit mismatch")
    require(registry.get("exit_status") == 0, failures, "registry exit status not zero")
    require(registry.get("validation_budget", {}).get("episodes_exact") == EXPECTED_EPISODES, failures, "registry validation budget episodes mismatch")
    require(registry.get("test_budget", {}).get("sealed_test_bank_opened") is False, failures, "registry sealed test budget flag not false")

    artifact_inventory_checks: List[Dict[str, Any]] = []
    for item in registry.get("artifact_inventory") or []:
        p = item.get("path")
        if not p or "<experiment_id>" in p:
            continue
        actual_path = root_path(p)
        actual_sha = sha256(actual_path) if actual_path.exists() else None
        artifact_inventory_checks.append({
            "path": p,
            "registry_exists": item.get("exists"),
            "actual_exists": actual_path.exists(),
            "registry_sha256": item.get("sha256"),
            "actual_sha256": actual_sha,
            "passed": item.get("exists") == actual_path.exists() and (not actual_path.exists() or item.get("sha256") == actual_sha),
        })
    bad_inventory = [x for x in artifact_inventory_checks if not x["passed"]]
    require(not bad_inventory, failures, "registry artifact inventory mismatches: %s" % bad_inventory[:3])

    rows = sorted(schedule.get("rows") or [], key=lambda r: int(r["execution_index"]))
    require(len(rows) == EXPECTED_EPISODES, failures, "schedule row count mismatch")
    if rows:
        require([int(r["execution_index"]) for r in rows] == list(range(EXPECTED_EPISODES)), failures, "schedule execution indices not 0..223")
    raw_episodes = sorted(raw.get("episodes") or [], key=lambda e: int(e["execution_index"]))
    require(len(raw_episodes) == EXPECTED_EPISODES, failures, "raw episode count mismatch")
    require(sum(int(e.get("steps", 0)) for e in raw_episodes) == int(budget_actual.get("control_steps", -999)), failures, "raw episode step sum mismatch")
    require(progress.get("episodes_done") == EXPECTED_EPISODES, failures, "progress episodes_done mismatch")
    require(progress.get("control_steps_done") == budget_actual.get("control_steps"), failures, "progress control steps mismatch")
    require(progress.get("test_accessed") is False, failures, "progress test flag not false")

    schedule_mismatches: List[Dict[str, Any]] = []
    for row, ep in zip(rows, raw_episodes):
        for key in ("execution_index", "rollout_key", "case_index"):
            ep_key = "case" if key == "case_index" else key
            if row.get(key) != ep.get(ep_key):
                schedule_mismatches.append({"execution_index": row.get("execution_index"), "key": key, "schedule": row.get(key), "episode": ep.get(ep_key)})
    require(not schedule_mismatches, failures, "schedule/raw episode mismatches: %s" % schedule_mismatches[:5])

    episode_dirs = sorted([p for p in (SHARD_DIR / "episodes").iterdir() if p.is_dir()]) if (SHARD_DIR / "episodes").exists() else []
    require(len(episode_dirs) == EXPECTED_EPISODES, failures, "episode directory count mismatch")
    episode_completed_audits: List[Dict[str, Any]] = []
    episode_summary_by_exec: Dict[int, Dict[str, Any]] = {}
    trace_len_mismatches: List[Dict[str, Any]] = []
    episode_completed_failures: List[str] = []
    for ep_dir in episode_dirs:
        ep_completed = ep_dir / "completed.json"
        ep_summary = ep_dir / "summary.json"
        if not ep_completed.exists() or not ep_summary.exists():
            episode_completed_failures.append(rel(ep_dir))
            continue
        done = read_json(ep_completed)
        if done.get("passed") is not True:
            episode_completed_failures.append(rel(ep_completed))
        ep_hash_audit = verify_hash_map(done.get("hashes") or {})
        if not ep_hash_audit["passed"]:
            episode_completed_failures.append(rel(ep_completed) + " hash audit failed")
        summary = read_json(ep_summary)
        exec_idx = int(summary.get("execution_index"))
        episode_summary_by_exec[exec_idx] = summary
        trace_info = read_trace_len_and_last(ep_dir)
        if trace_info["trace_len"] != int(summary.get("steps", -1)):
            trace_len_mismatches.append({"path": rel(ep_dir), "steps": summary.get("steps"), "trace_len": trace_info["trace_len"]})
        episode_completed_audits.append({
            "path": rel(ep_dir),
            "execution_index": exec_idx,
            "hash_records": ep_hash_audit["hash_records"],
            "bytes_checked": ep_hash_audit["bytes_checked"],
            "trace_len": trace_info["trace_len"],
            "termination": summary.get("termination"),
            "success": bool(summary.get("success")),
            "passed": ep_hash_audit["passed"] and trace_info["trace_len"] == int(summary.get("steps", -1)),
        })
    require(not episode_completed_failures, failures, "episode completed failures: %s" % episode_completed_failures[:5])
    require(not trace_len_mismatches, failures, "episode trace length mismatches: %s" % trace_len_mismatches[:5])
    require(set(episode_summary_by_exec) == set(range(EXPECTED_EPISODES)), failures, "episode summary execution index set mismatch")

    raw_vs_episode_summary_mismatches: List[Dict[str, Any]] = []
    for ep in raw_episodes:
        exec_idx = int(ep["execution_index"])
        summary = episode_summary_by_exec.get(exec_idx)
        if not summary:
            continue
        for key in ("rollout_key", "case", "steps", "success", "termination", "horizon_counts"):
            if ep.get(key) != summary.get(key):
                raw_vs_episode_summary_mismatches.append({"execution_index": exec_idx, "key": key, "raw": ep.get(key), "summary": summary.get(key)})
        for key in ("total_cost", "performance_cost", "constraint_cost", "physical_constraint_cost", "h_penalty"):
            if not close_float(ep.get(key), summary.get(key)):
                raw_vs_episode_summary_mismatches.append({"execution_index": exec_idx, "key": key, "raw": ep.get(key), "summary": summary.get(key)})
    require(not raw_vs_episode_summary_mismatches, failures, "raw vs episode summary mismatches: %s" % raw_vs_episode_summary_mismatches[:5])

    recomputed_overall = aggregate_from_episodes(raw_episodes)
    aggregate_checks = [compare_aggregate("overall", recomputed_overall, raw.get("overall_aggregate") or {}, failures)]
    for family in sorted({e.get("family") for e in raw_episodes}):
        eps = [e for e in raw_episodes if e.get("family") == family]
        aggregate_checks.append(compare_aggregate("family:%s" % family, aggregate_from_episodes(eps), (raw.get("aggregates_by_family") or {}).get(family) or {}, failures))
    for rollout_key in sorted({e.get("rollout_key") for e in raw_episodes}):
        eps = [e for e in raw_episodes if e.get("rollout_key") == rollout_key]
        aggregate_checks.append(compare_aggregate("rollout:%s" % rollout_key, aggregate_from_episodes(eps), (raw.get("aggregates_by_rollout_key") or {}).get(rollout_key) or {}, failures))

    rollout = raw.get("aggregates_by_rollout_key") or {}
    learned = {key: rollout.get(key) for key in ("learned_s0", "learned_s1", "learned_s2") if key in rollout}
    learned_horizon_audit = {}
    for key, agg in learned.items():
        horizons = {str(k): int(v) for k, v in (agg.get("horizon_counts") or {}).items()}
        learned_horizon_audit[key] = {
            "episodes": agg.get("episodes"),
            "steps": agg.get("steps"),
            "success_count": agg.get("success_count"),
            "episode_failure_count": agg.get("episode_failure_count"),
            "switches": agg.get("switches"),
            "horizon_counts": horizons,
            "unique_horizons": agg.get("unique_horizons"),
            "can_support_adaptive_behavior_in_this_shard": len(horizons) >= 2,
        }
    require(learned_horizon_audit.get("learned_s0", {}).get("horizon_counts") == {"25": 408}, failures, "learned_s0 horizon audit unexpected")
    require(learned_horizon_audit.get("learned_s1", {}).get("horizon_counts") == {"25": 390}, failures, "learned_s1 horizon audit unexpected")
    require(learned_horizon_audit.get("learned_s2", {}).get("horizon_counts") == {"25": 287, "35": 60}, failures, "learned_s2 horizon audit unexpected")

    overall = raw.get("overall_aggregate") or {}
    family = raw.get("aggregates_by_family") or {}
    selection_relevant_snapshot = {
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
        "not_model_selection": "Shard00 is only 1/12 of the preregistered validation64 block; do not select/finalize models from this shard alone.",
    }

    # Conservative scan of result artifacts for accidental test-output markers.  The raw
    # JSON is allowed to contain the sealed-test bank *metadata* path from the frozen
    # gate; no file read/hashing of that bank is performed here.
    suspicious_test_output_paths = []
    allowed_metadata_pattern = re.compile(r"bank_metadata_no_content_opened|sealed_test128|vehicle_test_bank\\.json|recorded_sha256_from_gate_metadata")
    for name in completed.get("hashes") or {}:
        if "test" in name.lower() and not allowed_metadata_pattern.search(name):
            suspicious_test_output_paths.append(name)
    require(not suspicious_test_output_paths, failures, "unexpected test-like output paths: %s" % suspicious_test_output_paths[:10])

    runtime_seconds = registry.get("runtime_seconds")
    wall_from_raw = None
    try:
        start = dt.datetime.fromisoformat(str(raw.get("started_utc")))
        end = dt.datetime.fromisoformat(str(raw.get("created_utc")))
        wall_from_raw = (end - start).total_seconds()
    except Exception:
        pass

    formal_artifact_paths = [
        rel(RAW_JSON), rel(SUMMARY_MD), rel(COMPLETED_JSON), rel(SCHEDULE_JSON), rel(PROGRESS_JSON), rel(RUN_STARTED_JSON), rel(TERMINAL_PROGRESS_JSON), rel(REGISTRY), rel(BACKUP_PROOF)
    ]
    formal_artifact_hashes = {path: sha256(root_path(path)) for path in formal_artifact_paths if root_path(path).exists()}

    backup_request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_%s.json" % dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S"))

    raw_audit: Dict[str, Any] = {
        "created_utc": now,
        "purpose": "post-run audit of formal vehicle validation64 shard00; no simulations, no training, no sealed-test access",
        "passed": not failures,
        "failures": failures,
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
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "script_under_audit": {"path": rel(RUNNER), "sha256": top_hashes[rel(RUNNER)]},
        "gate": {"path": rel(GATE_JSON), "sha256": top_hashes[rel(GATE_JSON)]},
        "backup_proof": {"path": rel(BACKUP_PROOF), "sha256": proof_sha, "asset_sha256": proof.get("asset_sha256"), "commit": proof.get("commit")},
        "completed_hash_audit": completed_hash_audit,
        "episode_completed_audit": {
            "episode_dirs": len(episode_dirs),
            "episode_completed_failures": episode_completed_failures,
            "trace_len_mismatches": trace_len_mismatches,
            "sample": episode_completed_audits[:5],
            "all_episode_hash_audits_passed": all(x["passed"] for x in episode_completed_audits) and len(episode_completed_audits) == EXPECTED_EPISODES,
        },
        "registry_audit": {
            "path": rel(REGISTRY),
            "runtime_seconds": runtime_seconds,
            "wall_seconds_from_raw_timestamps": wall_from_raw,
            "exit_status": registry.get("exit_status"),
            "source_tree_sha256": registry.get("source_tree_sha256"),
            "git_commit": registry.get("git_commit"),
            "artifact_inventory_checks": artifact_inventory_checks,
            "cloudwatch_status": registry.get("cloudwatch", {}).get("status"),
        },
        "budget_audit": {
            "declared": budget_declared,
            "actual": budget_actual,
            "completed_control_steps": completed.get("control_steps"),
            "progress": progress,
            "control_step_upper_bound_respected": int(budget_actual.get("control_steps", 10**9)) <= EXPECTED_CONTROL_UPPER,
        },
        "access_audit": {
            "raw_flags": {k: raw.get(k) for k in ("validation_accessed", "validation64_bank_content_opened", "test_accessed", "sealed_test_bank_content_opened", "final_test_authorization_requested")},
            "completed_flags": {k: completed.get(k) for k in ("validation_accessed", "validation64_bank_content_opened", "test_accessed", "sealed_test_bank_content_opened", "formal_scientific_evidence_created")},
            "sealed_test_metadata_from_gate_only": sealed_meta,
            "suspicious_test_output_paths": suspicious_test_output_paths,
        },
        "schedule_audit": {
            "rows": len(rows),
            "execution_index_range": [int(rows[0]["execution_index"]), int(rows[-1]["execution_index"])] if rows else None,
            "schedule_mismatches": schedule_mismatches,
            "raw_vs_episode_summary_mismatches": raw_vs_episode_summary_mismatches,
        },
        "aggregate_audit": {
            "overall_recomputed": recomputed_overall,
            "aggregate_checks": aggregate_checks,
            "all_passed": all(x["passed"] for x in aggregate_checks),
        },
        "selection_relevant_snapshot": selection_relevant_snapshot,
        "formal_artifact_hashes": formal_artifact_hashes,
        "top_file_hashes_current": top_hashes,
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
        "interpretation_limits": [
            "Shard00 is formal validation evidence but only one of twelve preregistered validation shards; no final model-selection or reproduction claim is made here.",
            "This audit did not run simulations and did not open the validation bank or sealed test bank; it read already-created shard00 validation outputs.",
            "Vehicle-only IMPROVED latency-tree evidence, not ORIGINAL SAC and not whole two-task evidence.",
            "All timing is t3a.medium same-host shard timing; CloudWatch/CPU-credit values remain unavailable/unknown, not zero.",
            "A verified external backup covering shard00 formal outputs and this audit is required before accumulating more unique formal validation evidence.",
        ],
    }

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    completed_path = OUT_DIR / "completed.json"
    write_json(raw_path, raw_audit)

    lines = [
        "# Vehicle validation64 shard00 post-run audit",
        "",
        "Created UTC: `%s`." % now,
        "",
        "Validation-result access: validation_accessed=true because this audit reads shard00 outputs; sealed test accessed=false. No simulations or training were run.",
        "",
        "## Gate and budget",
        "",
        "- shard run experiment: `%s`" % EXPECTED_EXPERIMENT_ID,
        "- runner SHA256: `%s`" % top_hashes[rel(RUNNER)],
        "- gate SHA256: `%s`" % top_hashes[rel(GATE_JSON)],
        "- pre-run backup proof SHA256: `%s`" % proof_sha,
        "- completed hash audit passed: `%s` over `%s` records" % (completed_hash_audit["passed"], completed_hash_audit["hash_records"]),
        "- episode dirs: `%s`; episode trace/hash audit passed: `%s`" % (len(episode_dirs), raw_audit["episode_completed_audit"]["all_episode_hash_audits_passed"]),
        "- episodes: `%s` / declared `%s`" % (budget_actual.get("episodes"), budget_declared.get("episodes_exact")),
        "- control steps: `%s` / upper bound `%s`" % (budget_actual.get("control_steps"), budget_declared.get("control_step_upper_bound")),
        "- runtime_seconds from registry: `%s`; CloudWatch status: `%s`" % (runtime_seconds, raw_audit["registry_audit"]["cloudwatch_status"]),
        "",
        "## Aggregate snapshot (not final selection)",
        "",
        "Overall: %s" % summarize_rollout_for_markdown(selection_relevant_snapshot["overall"]),
        "",
        "| family | episodes | steps | success | failures | total cost | physical cost | decision mean s/step | horizons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, val in sorted(selection_relevant_snapshot["families"].items()):
        lines.append(
            "| `%s` | %s | %s | %s | %s | %.6g | %.6g | %.6g | %s |" % (
                key, val["episodes"], val["steps"], val["success_count"], val["episode_failure_count"],
                float(val["total_cost_sum"]), float(val["physical_constraint_cost_sum"]), float(val["decision_mean_s_per_step"]), val["horizon_counts"]
            )
        )
    lines.extend([
        "",
        "## Learned candidate horizon audit",
        "",
        "| rollout | episodes | steps | success | failures | switches | horizons | adaptive in this shard? |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for key, val in sorted(learned_horizon_audit.items()):
        lines.append(
            "| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                key, val["episodes"], val["steps"], val["success_count"], val["episode_failure_count"],
                val["switches"], val["horizon_counts"], val["can_support_adaptive_behavior_in_this_shard"]
            )
        )
    lines.extend([
        "",
        "## Failures / blockers",
        "",
        "- audit passed: `%s`" % raw_audit["passed"],
        "- failures: `%s`" % failures,
        "- backup required before more formal validation: `true`",
        "- sealed final test remains unauthorized/closed.",
        "",
        "Next permitted action after verified external backup: continue with `vehicle_validation64_shard_runner.py --shard 1` using the same frozen gate/proof pattern, or run additional metadata-only analysis; do not open sealed test.",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_request = {
        "status": "external_backup_requested_after_validation64_shard00",
        "backup_verified": False,
        "created_utc": now,
        "reason": "Formal vehicle validation64 shard00 produced unique validation evidence; external recoverable backup is required before accumulating more formal validation shards.",
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations_by_request": 0,
        "formal_scientific_evidence_to_preserve": True,
        "shard00_experiment_id": EXPECTED_EXPERIMENT_ID,
        "shard00_artifact_hashes": formal_artifact_hashes,
        "audit_artifact_hashes": {
            rel(raw_path): sha256(raw_path),
            rel(summary_path): sha256(summary_path),
        },
        "required_next_backup_proof_fields_before_shard1": {
            "backup_verified": True,
            "remaining_changed_files": 0,
            "commit": "new commit or verified state after shard00 formal outputs and this audit/request",
            "release_or_asset_sha256": "required from GitHub release asset/download verification",
            "runner_sha256": EXPECTED_RUNNER_SHA,
            "gate_sha256": EXPECTED_GATE_SHA,
            "must_cover_validation64_shard00_artifacts_and_this_audit": True,
        },
        "next_action_after_verified_backup": "Run vehicle_validation64_shard_runner.py --shard 1 --backup-proof <post-shard00-proof> --i-accept-validation-access with exactly 224 validation episodes and <=33600 control steps; sealed test remains closed.",
    }
    write_json(backup_request_path, backup_request)
    raw_audit["backup_request_sha256"] = sha256(backup_request_path)
    raw_audit["backup_request_path"] = rel(backup_request_path)
    write_json(raw_path, raw_audit)

    doc_body = """
## 2026-09-26 vehicle validation64 shard00 audit

UTC: {now}. Post-run audit of formal shard00 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard00 has {episodes} episodes and {steps} control steps, within the declared 224/33600 budget. Completed hash audit passed={hash_passed}; episode trace/hash audit passed={episode_passed}; aggregate replay checks passed={aggregate_passed}. Learned candidates in shard00: s0 fixed H25 ({s0_steps} steps), s1 fixed H25 ({s1_steps} steps), s2 used H25/H35 ({s2_horizons}) with {s2_switches} switches. This is only 1/12 validation evidence and not final model selection. New formal evidence requires external backup before shard01; request written at `{backup_request}`. Sealed test remains closed.
""".format(
        now=now,
        episodes=budget_actual.get("episodes"),
        steps=budget_actual.get("control_steps"),
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
        "episodes_audited": budget_actual.get("episodes"),
        "control_steps_audited": budget_actual.get("control_steps"),
        "backup_required_before_more_formal_validation": True,
        "backup_request_path": rel(backup_request_path),
        "docs_updated": docs_updated,
        "hashes": {
            rel(raw_path): sha256(raw_path),
            rel(summary_path): sha256(summary_path),
            rel(backup_request_path): sha256(backup_request_path),
        },
    }
    write_json(completed_path, completed_payload)
    completed_payload["hashes"][rel(completed_path)] = sha256(completed_path)
    write_json(completed_path, completed_payload)

    print(json.dumps({
        "passed": raw_audit["passed"],
        "audit_raw": rel(raw_path),
        "audit_summary": rel(summary_path),
        "audit_completed": rel(completed_path),
        "backup_request": rel(backup_request_path),
        "backup_request_sha256": sha256(backup_request_path),
        "validation_accessed": True,
        "test_accessed": False,
        "new_simulations": 0,
        "episodes_audited": budget_actual.get("episodes"),
        "control_steps_audited": budget_actual.get("control_steps"),
        "backup_required_before_more_formal_validation": True,
    }, sort_keys=True))
    if not raw_audit["passed"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
