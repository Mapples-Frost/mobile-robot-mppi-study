#!/usr/bin/env python3
"""Post-run audit for vehicle validation64 shard10.

Reads only already-created shard10 result artifacts. Performs no simulation, no
training, no validation-bank reopen, and no sealed-final-test open/hash. This
uses the repaired validation-budget schema that accepts
``validation_budget.episodes_exact`` as the shard episode count.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard07_audit_v2_schema_repair.py"
SPEC = importlib.util.spec_from_file_location("shard07_audit_v2_helpers", str(HELPER_PATH))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load shard07 v2 helper module")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)  # type: ignore[union-attr]

SHARD_INDEX = 10
SHARD_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation64_shard10_audit_20260927"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REGISTRY = ROOT / "research_artifacts/aws_runs/20260927T034937_4f27027e/registry.json"
PRE_SHARD_BACKUP_PROOF = ROOT / "research_artifacts/aws_backup_proofs/backup_proof_20260927T034828_after_shard09_audit_and_case43_diagnostic_run_finalized.json"
RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard_runner.py"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_validation64_shard10_audit.py"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"

EXPECTED_EXPERIMENT_ID = "20260927T034937_4f27027e"
EXPECTED_RUNNER_SHA = "cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e"
EXPECTED_GATE_SHA = "5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b"
EXPECTED_VALIDATION_BANK_SHA = "b0ed14f2738a07975a42d5adc12e58ec3970515dea69e6338ed31d69b5fca48f"
EXPECTED_PRE_SHARD_ASSET_SHA = "befc97d0145e60c10d3d5a2d9a377876017679f43a1c7ef004ea91306780f00d"
EXPECTED_PRE_SHARD_BACKUP_PROOF_SHA = "fe173b469322b0b7d775c054954aa13943a02aa81891d63965097f7aab90fb6a"
EXPECTED_EPISODES = 224
EXPECTED_CONTROL_STEPS = 19831
EXPECTED_CONTROL_UPPER = 33600
EXPECTED_EXECUTION_START = 2240
EXPECTED_EXECUTION_END = 2463
DOC_MARKER = "vehicle-validation64-shard10-audit-20260927"

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
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_once(path: Path, marker: str, body: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")
    return True


def require(ok: bool, failures: List[str], message: str) -> None:
    if not ok:
        failures.append(message)


def close_float(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def verify_completed_marker(path: Path) -> bool:
    try:
        done = read_json(path)
    except Exception:
        return False
    return bool(done.get("passed") is True and h.v1.verify_hash_map(done.get("hashes") or {}).get("passed"))


def validation_budget_schema(vb: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ["episodes", "validation_episodes", "formal_validation_shard_episodes", "episodes_exact"]
    observed = {k: vb.get(k) for k in keys if k in vb}
    matched: Optional[str] = None
    for key in keys:
        try:
            if vb.get(key) is not None and int(vb.get(key)) == EXPECTED_EPISODES:
                matched = key
                break
        except Exception:
            pass
    control_ok = True
    if "control_step_upper_bound" in vb:
        try:
            control_ok = int(vb.get("control_step_upper_bound")) == EXPECTED_CONTROL_UPPER
        except Exception:
            control_ok = False
    access_ok = True
    if "validation_accessed" in vb:
        access_ok = access_ok and vb.get("validation_accessed") is True
    if "validation64_bank_content_opened" in vb:
        access_ok = access_ok and vb.get("validation64_bank_content_opened") is True
    return {
        "passed": bool(matched and control_ok and access_ok),
        "matched_episode_key": matched,
        "observed_episode_fields": observed,
        "control_step_upper_bound_ok": control_ok,
        "access_flags_ok": access_ok,
        "registry_validation_budget": dict(vb),
        "schema_repair": "accepted episodes_exact=224 as equivalent to a shard episode budget" if matched == "episodes_exact" else "canonical_or_other_episode_key",
    }


def doc_hashes(names: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in names:
        p = ROOT / name
        if p.exists():
            out[name] = sha256(p)
    return out


def write_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle validation64 shard10 post-run audit",
        "",
        "Created UTC: `%s`." % audit["created_utc"],
        "",
        "- Passed: `%s`." % audit["passed"],
        "- Validation accessed: `true` only by reading existing shard10 outputs; validation64 bank reopened: `false`.",
        "- Sealed final test accessed/opened/hashed: `false` / `false` / `false`.",
        "- New simulations/control steps/training steps by audit: `0` / `0` / `0`.",
        "- Shard10 formal episodes/control steps audited: `%s` / `%s` (upper bound `%s`)." % (audit["episodes_audited"], audit["control_steps_audited"], audit["control_step_upper_bound"]),
        "- Completed hash audit: `%s`; episode trace/hash audit: `%s`; aggregate replay: `%s`." % (audit["completed_hash_audit_passed"], audit["episode_trace_hash_audit_passed"], audit["aggregate_replay_passed"]),
        "",
        "## Learned candidate horizon audit",
        "",
        "| candidate | episodes | steps | successes | horizon counts | switches | adaptive in this shard |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for key in sorted(audit["learned_horizon_audit"]):
        row = audit["learned_horizon_audit"][key]
        lines.append("| `%s` | %s | %s | %s | `%s` | %s | `%s` |" % (key, row["episodes"], row["steps"], row["success_count"], row["horizon_counts"], row["switches"], row["adaptive_in_this_shard"]))
    lines.extend([
        "",
        "Shard10 is one shard of the preregistered vehicle validation64 block for the IMPROVED latency-tree controller, not ORIGINAL SAC and not a final reproduction claim.",
        "Shard11 is blocked until a verified external backup proof covers shard10 formal/audit outputs, docs/registry updates, backup requests/addenda/blocker, and finalized audit run registry/stdout/stderr.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    completed_out = OUT_DIR / "completed.json"
    if completed_out.exists():
        if verify_completed_marker(completed_out):
            print(json.dumps({"already_completed": True, "completed": rel(completed_out)}, sort_keys=True))
            return 0
        raise SystemExit("prior shard10 audit exists but did not verify; inspect before rerun")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    now = now_dt.isoformat()
    stamp = now_dt.strftime("%Y%m%dT%H%M%S")
    failures: List[str] = []

    required = [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, SCHEDULE_JSON, PROGRESS_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY, PRE_SHARD_BACKUP_PROOF, RUNNER, THIS_SCRIPT, GATE_JSON, HELPER_PATH]
    missing = [rel(p) for p in required if not p.exists()]
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
    registry_cloudwatch = Path(registry.get("cloudwatch_snapshot", "")) if registry.get("cloudwatch_snapshot") else None
    top_hashes = {rel(p): sha256(p) for p in required}
    for extra in (registry_stdout, registry_stderr, registry_cloudwatch):
        if extra is not None and extra.exists():
            top_hashes[rel(extra)] = sha256(extra)

    completed_hash_audit = h.v1.verify_hash_map(completed.get("hashes") or {})
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
    require(str(validation_bank.get("path", "")).endswith("vehicle_validation_bank.json"), failures, "validation bank path unexpected")
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
    if registry_commit and pre_proof.get("commit"):
        require(registry_commit == pre_proof.get("commit"), failures, "registry commit does not match pre-shard proof commit")
    registry_validation_budget = validation_budget_schema(registry.get("validation_budget") or {})
    require(registry_validation_budget["passed"], failures, "registry validation budget episode mismatch under repaired schema")
    registry_test_closed = h.v1.registry_test_budget_closed(registry.get("test_budget") or {}, raw, completed, run_started)
    require(registry_test_closed["passed"], failures, "registry test budget flags not closed")

    artifact_inventory_checks: List[Dict[str, Any]] = []
    for item in registry.get("artifact_inventory") or []:
        p = item.get("path")
        if not p or "*" in p or "<experiment_id>" in p or "${experiment_id}" in p:
            continue
        actual_path = root_path(p)
        actual_exists = actual_path.exists()
        passed_item = item.get("exists") == actual_exists
        actual_sha: Optional[str] = None
        if actual_exists and actual_path.is_file():
            actual_sha = sha256(actual_path)
            if item.get("sha256") is not None:
                passed_item = passed_item and item.get("sha256") == actual_sha
        artifact_inventory_checks.append({"path": p, "registry_exists": item.get("exists"), "actual_exists": actual_exists, "registry_sha256": item.get("sha256"), "actual_sha256": actual_sha, "passed": passed_item})
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
    raw_summary_mismatches: List[Dict[str, Any]] = []
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
        audit = h.v1.verify_hash_map(done.get("hashes") or {})
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
    aggregate_checks.append(h.v1.compare_aggregate("overall", h.v1.aggregate_from_episodes(episodes), raw.get("overall_aggregate") or {}, failures))
    for key, recorded in sorted((raw.get("aggregates_by_rollout_key") or {}).items()):
        aggregate_checks.append(h.v1.compare_aggregate("rollout:" + key, h.v1.aggregate_from_episodes([e for e in episodes if e.get("rollout_key") == key]), recorded, failures))
    for key, recorded in sorted((raw.get("aggregates_by_family") or {}).items()):
        aggregate_checks.append(h.v1.compare_aggregate("family:" + key, h.v1.aggregate_from_episodes([e for e in episodes if e.get("family") == key]), recorded, failures))
    aggregate_replay_passed = all(x["passed"] for x in aggregate_checks)

    learned_horizon_audit: Dict[str, Any] = {}
    for key in ["learned_s0", "learned_s1", "learned_s2"]:
        eps = [e for e in episodes if e.get("rollout_key") == key]
        agg = h.v1.aggregate_from_episodes(eps)
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

    done_shards = sorted(int(p.name[-2:]) for p in SHARD_DIR.parent.glob("shard[0-9][0-9]") if (p / "completed.json").exists())
    cumulative_episodes = 0
    cumulative_steps = 0
    for idx in done_shards:
        done = read_json(SHARD_DIR.parent / ("shard%02d" % idx) / "completed.json")
        cumulative_episodes += int(done.get("episodes", 0))
        cumulative_steps += int(done.get("control_steps", 0))

    passed = not failures
    audit_raw: Dict[str, Any] = {
        "created_utc": now,
        "passed": passed,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_validation64_shard10_postrun_audit_not_original_SAC",
        "validation_accessed": True,
        "validation_access_type": "existing shard10 result artifacts read/hashed only; validation64 bank not reopened",
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
        "schema_repair": {"registry_validation_budget": registry_validation_budget, "changed_acceptance_only": "validation_budget.episodes_exact is accepted as the episode-budget key; no scientific criterion, simulation output, or final-test gate changed"},
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
        "cumulative_vehicle_validation64_status": {"formal_shards_completed": done_shards, "audited_shards_passed_expected_after_this_audit": list(range(0, SHARD_INDEX + 1)) if passed else list(range(0, SHARD_INDEX)), "formal_episodes_completed": cumulative_episodes, "formal_control_steps_completed": cumulative_steps, "planned_total_shards": 12, "planned_total_validation_episodes": 2688},
        "interpretation_limits": ["Shard-level validation audit only; no final-test evidence.", "IMPROVED latency-tree method, not ORIGINAL SAC.", "No reproduction-success or model-selection conclusion from a single shard."],
    }

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    blocker_path = ROOT / "research_artifacts/aws_diagnostics/post_shard10_audit_backup_blocker_check_20260927.md"
    request_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_%s.json" % stamp)
    addendum_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_FINAL_ADDENDUM_%s.json" % stamp)

    write_json(raw_path, audit_raw)
    write_markdown(summary_path, audit_raw)

    doc_body = """
## 2026-09-27 vehicle validation64 shard10 audit

UTC: {now}. Post-run audit of formal shard10 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard10 has {episodes} episodes and {steps} control steps, within the declared 224/33600 budget. Completed hash audit passed={hash_pass}; episode trace/hash audit passed={trace_pass}; aggregate replay checks passed={agg_pass}. Learned candidates in shard10: s0 {s0}, s1 {s1}, s2 {s2}. Vehicle validation64 progress is now 11/12 completed formal shards with {cum_eps} episodes and {cum_steps} control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard11; request written at `{request}` and final addendum at `{addendum}`. Sealed test remains closed.
""".format(now=now, episodes=EXPECTED_EPISODES, steps=EXPECTED_CONTROL_STEPS, hash_pass=completed_hash_audit["passed"], trace_pass=audit_raw["episode_trace_hash_audit_passed"], agg_pass=aggregate_replay_passed, s0=learned_horizon_audit["learned_s0"], s1=learned_horizon_audit["learned_s1"], s2=learned_horizon_audit["learned_s2"], cum_eps=cumulative_episodes, cum_steps=cumulative_steps, request=rel(request_path), addendum=rel(addendum_path))
    changed_docs = []
    for name in ["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"]:
        if append_once(ROOT / name, DOC_MARKER, doc_body):
            changed_docs.append(name)

    blocker_text = """# Post-shard10-audit backup blocker check

Created UTC: `{now}`.

Shard10 formal validation and post-run audit have completed, but shard11 is BLOCKED until a verified external backup proof after this audit/addendum/blocker state and after the finalized audit run registry/stdout/stderr is present locally. The required proof must record `backup_verified=true`, `remaining_changed_files=0`, GitHub release asset/download SHA256 verification, runner SHA `{runner_sha}`, gate SHA `{gate_sha}`, and coverage of shard10 formal outputs, shard10 audit outputs, formal/audit run registry/stdout/stderr, docs/registry updates, backup request, final addendum, and this blocker note.

Latest adequate pre-shard proof used for shard10 was `{pre_proof}`; it is not sufficient for shard11 because it predates shard10 formal validation and this audit. Sealed final test remains closed and unauthorized. This note ran no simulations, no control steps, and no gradient steps.
""".format(now=now, runner_sha=EXPECTED_RUNNER_SHA, gate_sha=EXPECTED_GATE_SHA, pre_proof=rel(PRE_SHARD_BACKUP_PROOF))
    blocker_path.write_text(blocker_text, encoding="utf-8")

    docs_after = doc_hashes(["STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"])
    request = {
        "status": "external_backup_requested_after_validation64_shard10_audit",
        "backup_verified": False,
        "created_utc": now,
        "required_before_shard11": True,
        "reason": "Preserve shard10 formal outputs, shard10 audit, docs/registry updates, backup request/addendum/blocker, and finalized audit run logs before creating the final validation64 shard.",
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "gate_sha256": EXPECTED_GATE_SHA,
        "pre_shard10_backup_proof": {"path": rel(PRE_SHARD_BACKUP_PROOF), "sha256": top_hashes[rel(PRE_SHARD_BACKUP_PROOF)]},
        "shard10_formal_artifact_hashes": {rel(p): sha256(p) for p in [RAW_JSON, SUMMARY_MD, COMPLETED_JSON, PROGRESS_JSON, SCHEDULE_JSON, RUN_STARTED_JSON, TERMINAL_PROGRESS_JSON, REGISTRY]},
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
        "next_action_after_verified_backup": "Run shard11 with legacy interpreter and sealed final test closed, then audit shard11 before any full-block analysis.",
    }
    write_json(request_path, request)
    addendum = dict(request)
    addendum.update({
        "reason": "Final addendum after writing shard10 audit completed marker, documentation updates, and blocker note; external backup is required before shard11.",
        "backup_request": rel(request_path),
        "backup_request_sha256": sha256(request_path),
        "audit_run_registry_stdout_stderr_required_after_run_experiment_finalizes": True,
        "must_cover_this_final_addendum_itself": True,
    })
    write_json(addendum_path, addendum)

    files_for_completed = [THIS_SCRIPT, HELPER_PATH, raw_path, summary_path, blocker_path, request_path, addendum_path]
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
        "schema_repair": audit_raw["schema_repair"],
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
        "shard11_blocked_until_backup": True,
        "failures": failures,
    }, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
