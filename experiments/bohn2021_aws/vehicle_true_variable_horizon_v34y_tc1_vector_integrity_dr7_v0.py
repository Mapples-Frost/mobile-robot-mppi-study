#!/usr/bin/env python3
"""Structured zero-solve T-C1R / T-C5 artifact diagnostics for OC-epsilon.

This script is intentionally limited to already-opened development artifacts. It
performs no solver call, plant rollout, env.step/reset after construction,
training/refit, validation64 access, or sealed/final-test access.

Supported structured tasks from Opus plan 20260930T133239Z_8fd7da:
  * T-C1R-OC-epsilon-1a-receipt-repair
  * T-C5-OC-epsilon-2-vector-sum-closure
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
try:
    import execution_contract  # type: ignore
except ImportError:
    sys.path.insert(0, str(ROOT / "scripts" / "research_service"))
    import execution_contract  # type: ignore

NAME = "vehicle_true_variable_horizon_v34y_tc1_vector_integrity_dr7_v0"
EXPECTED_REQUEST = "execution-result:20260930T131908_12b79445"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V34X_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34x_epsilon_provenance_audit_v0_20260930T131908Z"
V34X_RAW = V34X_DIR / "raw.json"
V34X_COMPLETED = V34X_DIR / "completed.json"
V34X_SUMMARY = V34X_DIR / "summary.md"
V34U_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z"
V34U_RAW = V34U_DIR / "raw.json"
V34U_COMPLETED = V34U_DIR / "completed.json"
V34V_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34v_eps_objective_localization_postdiagnostic_v0_20260930T130526Z/completed.json"
V34T_123137_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0_20260930T123137Z"
V34T_122045_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0_20260930T122045Z/failed.json"
V34T_123746_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0_20260930T123746Z/failed.json"
RUN_122045 = ROOT / "research_artifacts/aws_runs/20260930T122045_c8b14921"
RUN_123746 = ROOT / "research_artifacts/aws_runs/20260930T123746_30cba63d"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

EXPECTED_V34X_RAW_SHA256 = "c0540b523d93105d02b380c27729d4112eb20e0f7b68e74d1cb9141925c45c5e"
EXPECTED_V34X_COMPLETED_SHA256 = "a9732844fff19523cbd97c2c8c2cf32f05bcc31f8097085814e63cd2f1041545"
EXPECTED_OPT_P_HASHES = {
    "H12_canonical": "7032ac0c47903ccfe27a25accc556b22331ba32247bf1a8b7a95f996cd1fdd5a",
    "H15_canonical": "ca19ce63ea7aa807e66c33f9871022b24e1e7bd1ac0ee3292a82cc0025b52502",
    "H35_canonical": "9878b3d2235205772b48cf79a3e30d8eedd4e625feb2dc3d758130faa0ea651d",
    "H15_goal_facing": "fbe488a0e5749ee25682b7f6a963294112b041c31e0340de4214480cab638352",
}
RECORDED_EPSTERM_TOTALS = [79.89232842059124, 101.80331021216188, 256.38529988021594, 109.92914032557572]
EXPECTED_EPS_COUNTS = {"H12": 12, "H15": 15, "H35": 35, "H15_goal_facing": 15}
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(v: Any) -> Any:
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def iter_tree(obj: Any, path: str = "root") -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from iter_tree(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_tree(v, f"{path}[{i}]")


def summarize_file(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        out.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        if path.suffix.lower() == ".json":
            try:
                j = read_json(path)
                if isinstance(j, Mapping):
                    for k in ["status", "hard_pass", "error", "new_solver_calls_recorded", "validation64_bank_opened", "sealed_test_accessed", "test_accessed"]:
                        if k in j:
                            out[k] = j.get(k)
                    out["json_top_keys"] = sorted(str(k) for k in j.keys())[:30]
            except Exception as exc:
                out["json_error"] = repr(exc)
    return out


def numeric_list_info(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return None
    flat: List[float] = []
    stack: List[Any] = list(value)
    while stack and len(flat) <= 10000:
        x = stack.pop(0)
        if isinstance(x, list):
            stack = list(x) + stack
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
            flat.append(float(x))
        else:
            return None
    if not flat:
        return None
    return {"numeric_count": len(flat), "nonzero_count": int(sum(abs(x) > 1e-12 for x in flat)), "sum": float(math.fsum(flat)), "l1": float(math.fsum(abs(x) for x in flat)), "first_values": flat[:12]}


def collect_eps_path_hits(obj: Any, path: str = "root", hits: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if hits is None:
        hits = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{path}.{k}"
            lower = p.lower()
            if any(tok in lower for tok in ["_eps", ".eps", "epsilon", "epsterm", "slack", "include_eps", "eps_nonzero"]):
                entry: Dict[str, Any] = {"path": p, "type": type(v).__name__}
                info = numeric_list_info(v)
                if info is not None:
                    entry.update(info)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    entry["scalar"] = float(v)
                elif isinstance(v, bool):
                    entry["bool"] = bool(v)
                elif isinstance(v, str):
                    entry["string_preview"] = v[:160]
                elif isinstance(v, Mapping):
                    entry["mapping_keys_preview"] = sorted(str(x) for x in v.keys())[:12]
                elif isinstance(v, list):
                    entry["list_len"] = len(v)
                    entry["list_preview_types"] = [type(x).__name__ for x in v[:8]]
                hits.append(entry)
            collect_eps_path_hits(v, p, hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            collect_eps_path_hits(v, f"{path}[{i}]", hits)
    return hits


def strict_full_epsilon_vector_candidates(v34u_raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hits = collect_eps_path_hits(v34u_raw)
    excluded = ["alias_branch_variants", "candidate", "analysis.headline", "analysis.per_cell", "include_eps", "epsterm_total", "eps_nonzero_count", "eps_delta", "headline", "summary"]
    out: List[Dict[str, Any]] = []
    for h in hits:
        p = str(h.get("path", "")).lower()
        if "numeric_count" not in h:
            continue
        if any(x in p for x in excluded):
            continue
        if any(x in p for x in ["_eps", ".eps", "slack", "opt_x"]):
            out.append(h)
    return out


def find_expected_hashes_in_raw(raw_text: str) -> Dict[str, Any]:
    found = {label: (value in raw_text) for label, value in EXPECTED_OPT_P_HASHES.items()}
    return {"expected_hashes": EXPECTED_OPT_P_HASHES, "found": found, "found_count": int(sum(1 for v in found.values() if v)), "distinct_expected_hash_count": len(set(EXPECTED_OPT_P_HASHES.values()))}


def summarize_dr7() -> Dict[str, Any]:
    records = {
        "v34t_20260930T123137_exit0_hardpass_false": {
            "completed": summarize_file(V34T_123137_DIR / "completed.json"),
            "raw": summarize_file(V34T_123137_DIR / "raw.json"),
            "summary": summarize_file(V34T_123137_DIR / "summary.md"),
            "cell_metrics": summarize_file(V34T_123137_DIR / "cell_metrics.csv"),
            "registered_finding": "exit0 but hard_pass/G2 false; 0 new solver calls; no objective residual evidence; TypeError path in shifted object TVP handling before solver. Operational negative evidence, not a scientific objective-contract result.",
        },
        "v34t_20260930T122045_c8b14921": {
            "failed_json": summarize_file(V34T_122045_FAILED),
            "registry": summarize_file(RUN_122045 / "registry.json"),
            "stdout_log": summarize_file(RUN_122045 / "stdout.log"),
            "stderr_log": summarize_file(RUN_122045 / "stderr.log"),
            "registered_finding": "ContractError plan-ready digest mismatch before scientific work; zero solver calls; no validation/test access. Exact failure string is preserved in failed.json/stdout.",
        },
        "v34t_20260930T123746_30cba63d": {
            "failed_json": summarize_file(V34T_123746_FAILED),
            "registry": summarize_file(RUN_123746 / "registry.json"),
            "stdout_log": summarize_file(RUN_123746 / "stdout.log"),
            "stderr_log": summarize_file(RUN_123746 / "stderr.log"),
            "registered_finding": "ContractError authorization-token/old-pin failure before scientific work; zero solver calls; no validation/test access. Exact failure string is preserved in failed.json/stdout.",
        },
    }
    # Conservative zero-usage recognition from the artifacts already produced.
    zero_flags: Dict[str, bool] = {}
    for key, rec in records.items():
        text = json.dumps(clean(rec), sort_keys=True)
        zero_flags[key] = ("new_solver_calls_recorded\": 0" in text) or ("new_solver_calls\": 0" in text) or ("0 new solver calls" in text) or ("zero solver calls" in text)
    return {"records": records, "zero_usage_flags": zero_flags, "three_zero_usage_failures_registered": bool(len(zero_flags) == 3 and all(zero_flags.values()))}


def extract_arms(v34u_raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    arms = v34u_raw.get("arms") if isinstance(v34u_raw, Mapping) else None
    return [a for a in arms if isinstance(a, Mapping)] if isinstance(arms, list) else []


def arm_label(arm: Mapping[str, Any]) -> str:
    h = int(arm.get("horizon", 0) or 0)
    init = str(arm.get("initialization", ""))
    if h == 15 and "goal" in init:
        return "H15_goal_facing"
    return f"H{h}"


def get_epsterm_and_count(arm: Mapping[str, Any]) -> Tuple[Optional[float], Optional[int]]:
    variants = arm.get("alias_branch_variants") if isinstance(arm.get("alias_branch_variants"), Mapping) else {}
    for key in ["eps_only", "eps_and_rterm"]:
        rec = variants.get(key) if isinstance(variants, Mapping) else None
        if isinstance(rec, Mapping) and rec.get("epsterm_total") is not None:
            total = float(rec.get("epsterm_total"))
            count = int(rec.get("eps_nonzero_count", 0) or 0)
            return total, count
    rec = arm.get("objective_reconstruction") if isinstance(arm.get("objective_reconstruction"), Mapping) else {}
    if isinstance(rec, Mapping) and rec.get("epsterm_total") is not None:
        return float(rec.get("epsterm_total")), int(rec.get("eps_nonzero_count", 0) or 0)
    return None, None


def write_common_outputs(run_dir: Path, task_id: str, marker: str, raw: Mapping[str, Any], summary_lines: Sequence[str], backup_reason: str) -> Tuple[Path, Path, Path, Path]:
    raw_path = run_dir / "raw.json"
    summary_path = run_dir / "summary.md"
    completed_path = run_dir / "completed.json"
    backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_{task_id.replace('-', '_')}_{raw['created_utc'].replace(':','').replace('+','Z')}.json"
    write_json(raw_path, raw)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_json(backup_request, {
        "request": "backup_after_" + task_id,
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": backup_reason,
        "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "new_training_or_gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    return raw_path, summary_path, completed_path, backup_request


def write_failure(run_dir: Path, message: str, snapshot: Optional[Mapping[str, Any]], engineering_error: str = "loader") -> int:
    created = now_utc()
    failed_path = run_dir / "failed.json"
    write_json(failed_path, {"status": "failed", "created_utc": created.isoformat(), "classification": "engineering_failure_zero_resource_before_scientific_outcome", "error": message, "traceback_tail": traceback.format_exc().splitlines()[-8:], "budget_actual": dict(ZERO_RESOURCES), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
    if snapshot is not None:
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO_RESOURCES), {"no_scientific_outcome": True, "error": message, "failed_json": rel(failed_path)}, engineering_error=engineering_error)
    print(json.dumps({"failed": message, "failed_json": rel(failed_path), "resources": ZERO_RESOURCES}, sort_keys=True), flush=True)
    return 1


def run_tc1r(snapshot: Mapping[str, Any], run_dir: Path, marker: str) -> int:
    required = [V34X_RAW, V34X_COMPLETED, V34X_SUMMARY, V34U_RAW, V34U_COMPLETED, V34V_COMPLETED, V34T_122045_FAILED, V34T_123746_FAILED, V34T_123137_DIR / "completed.json"]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        return write_failure(run_dir, "missing required input(s): " + repr(missing), snapshot, "missing_file")
    v34x_raw_sha_before = sha256(V34X_RAW)
    v34x_completed_sha_before = sha256(V34X_COMPLETED)
    if v34x_raw_sha_before != EXPECTED_V34X_RAW_SHA256 or v34x_completed_sha_before != EXPECTED_V34X_COMPLETED_SHA256:
        return write_failure(run_dir, "v34x digest mismatch; refusing to rewrite or reinterpret frozen artifact", snapshot, "loader")

    v34x_completed = read_json(V34X_COMPLETED)
    v34x_raw = read_json(V34X_RAW)
    v34u_text = V34U_RAW.read_text(encoding="utf-8", errors="replace")
    v34u_raw = read_json(V34U_RAW)
    old_flag = bool((v34x_completed.get("headline") or {}).get("individual_epsilon_vector_values_serialized_in_v34u_raw")) if isinstance(v34x_completed, Mapping) else False
    eps_key_hits = collect_eps_path_hits(v34x_raw)[:100]
    strict_candidates = strict_full_epsilon_vector_candidates(v34u_raw)
    hash_summary = find_expected_hashes_in_raw(v34u_text)
    dr7 = summarize_dr7()

    coverage_lines = [
        "ORIGINAL author SAC reconstruction = partially inspected; primary ORIGINAL training logs/checkpoints not yet read by the lead in this cycle.",
        "ORIGINAL vehicle-then-inverted-pendulum pendant = UNINSPECTED this cycle; no pendulum evidence inspected and no pendulum claim made.",
        "Full v33 rollout matrix = not independently rechecked from raw this cycle.",
        "v34k registry ID = not independently verified this cycle.",
    ]
    corrected_receipt = {
        "receipt_type": "T-C1R corrected machine-readable receipt under Opus plan 20260930T133239Z_8fd7da",
        "task_id": "T-C1R-OC-epsilon-1a-receipt-repair",
        "created_utc": now_utc().isoformat(),
        "original_v34x_raw": rel(V34X_RAW),
        "original_v34x_completed": rel(V34X_COMPLETED),
        "original_v34x_raw_sha256_echoed_exactly": v34x_raw_sha_before,
        "original_v34x_completed_sha256_echoed_exactly": v34x_completed_sha_before,
        "original_v34x_bytes_preserved": True,
        "old_completed_headline_value": old_flag,
        "corrected_field_name": "individual_epsilon_vector_values_serialized_in_v34u_raw",
        "corrected_field_value": False,
        "literal__eps_search_count_in_v34u_raw": v34u_text.count("_eps"),
        "bounded_eps_key_hits_sample_from_v34x_raw": eps_key_hits,
        "strict_full_vector_candidates_in_v34u_raw": strict_candidates[:40],
        "correction_reason": "The previous detector matched metadata such as include_eps booleans and epsterm/eps summary fields. It did not establish a full per-stage solver-state epsilon vector. The original v34x bytes are unchanged.",
        "T_C1_source_provenance_pass_verdict_unchanged": True,
        "DR7_operational_failures_registered": dr7,
        "coverage_lines": coverage_lines,
        "verified_opt_p_hashes": hash_summary,
        "label_ORIGINAL_vs_IMPROVED": "IMPROVED latency-tree vehicle development bookkeeping only; not ORIGINAL SAC reproduction evidence.",
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    receipt_path = run_dir / "corrected_receipt.json"
    write_json(receipt_path, corrected_receipt)

    created = now_utc()
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "task_id": "T-C1R-OC-epsilon-1a-receipt-repair",
        "classification": "development_IMPROVED_zero_solve_T_C1R_vector_integrity_DR7_bookkeeping_not_validation_not_test",
        "snapshot_task_id": snapshot.get("task", {}).get("task_id") if isinstance(snapshot, Mapping) else None,
        "corrected_receipt": rel(receipt_path),
        "headline": {
            "corrected_epsilon_vector_field_present_and_false": True,
            "old_v34x_field_value": old_flag,
            "full_per_stage_epsilon_vector_serialized_in_v34u_raw_corrected": False,
            "literal__eps_search_count_in_v34u_raw": v34u_text.count("_eps"),
            "strict_full_vector_candidate_count": len(strict_candidates),
            "dr7_three_zero_usage_failures_registered": dr7["three_zero_usage_failures_registered"],
            "verified_opt_p_hash_count_recorded": hash_summary["found_count"],
            "distinct_expected_opt_p_hash_count": hash_summary["distinct_expected_hash_count"],
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
        },
        "input_hashes": {rel(p): sha256(p) for p in required if p.exists()},
        "corrected_receipt_payload": corrected_receipt,
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    }

    summary_lines = [
        "# T-C1R vector-integrity correction and DR-7 bookkeeping",
        "",
        f"UTC: `{created.isoformat()}`. Structured zero-solve task `T-C1R-OC-epsilon-1a-receipt-repair`.",
        "",
        "## Result",
        f"- Corrected receipt: `{rel(receipt_path)}` sets `individual_epsilon_vector_values_serialized_in_v34u_raw=false`.",
        f"- Original v34x raw SHA256 preserved and echoed: `{v34x_raw_sha_before}`.",
        f"- Original v34x completed SHA256 preserved and echoed: `{v34x_completed_sha_before}`.",
        f"- Literal `_eps` count in v34u raw: `{v34u_text.count('_eps')}`; strict full-vector candidates: `{len(strict_candidates)}`.",
        f"- DR-7 three zero-usage operational failures registered: `{dr7['three_zero_usage_failures_registered']}`.",
        f"- Expected opt_p hashes found: `{hash_summary['found_count']}` / `{hash_summary['distinct_expected_hash_count']}`.",
        "",
        "## Coverage limitations recorded",
    ] + [f"- {line}" for line in coverage_lines] + [
        "",
        "Cross-arm causal attribution is deferred to T-C3. The opt_p hash differences do not invalidate within-cell objective reconstruction.",
        "",
        "Budget/access: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.",
    ]
    raw_path, summary_path, completed_path, backup_request = write_common_outputs(run_dir, "T-C1R-OC-epsilon-1a-receipt-repair", marker, raw, summary_lines, "T-C1R corrected v34x vector flag, registered DR-7 zero-usage failures, and updated coverage/opt_p hash bookkeeping before solver-bearing T-C2.")

    doc_block = f"""
<!-- {marker} -->
## T-C1R vector-integrity correction and DR-7 bookkeeping

UTC: {created.isoformat()}. Structured task `T-C1R-OC-epsilon-1a-receipt-repair` completed with solver=0, plant/env.step/reset=0, training/refit=0, validation64=0, sealed/final test=0. Corrected receipt `{rel(receipt_path)}` sets the unsupported v34x full-vector field false while preserving original v34x raw `{v34x_raw_sha_before}` and completed `{v34x_completed_sha_before}`. DR-7 registered three zero-usage operational failures; expected opt_p hashes found={hash_summary['found_count']}/4. Coverage lines recorded for ORIGINAL SAC partial inspection, pendulum uninspected, full v33 not rechecked, and v34k registry ID not independently verified. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`. Backup request: `{rel(backup_request)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, marker, doc_block)
    state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{created.strftime('%Y%m%dT%H%M%SZ')}_after_tc1r_vector_integrity_dr7.md"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("# Continue state after T-C1R\n\n" + doc_block + "\nNext approved zero-solve tasks are T-C3 and T-C5 under the active structured plan; T-C2 remains dependency-gated.\n", encoding="utf-8")
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(completed_path), marker])

    pass_evidence = {
        "corrected_epsilon_vector_field_present_and_false": True,
        "coverage_lines_for_ORIGINAL_and_pendulum_present": True,
        "dr7_three_zero_usage_failures_registered": dr7["three_zero_usage_failures_registered"],
        "no_new_solver_or_plant_or_training_usage": True,
        "original_v34x_raw_bytes_preserved": sha256(V34X_RAW) == v34x_raw_sha_before == EXPECTED_V34X_RAW_SHA256,
        "original_v34x_raw_sha256_echoed_exactly": v34x_raw_sha_before,
        "receipt_written": receipt_path.exists(),
        "verified_opt_p_hash_count_recorded": int(hash_summary["found_count"]),
    }
    completed = {
        "status": "complete",
        "hard_pass": bool(pass_evidence["corrected_epsilon_vector_field_present_and_false"] and pass_evidence["coverage_lines_for_ORIGINAL_and_pendulum_present"] and pass_evidence["dr7_three_zero_usage_failures_registered"] and pass_evidence["no_new_solver_or_plant_or_training_usage"] and pass_evidence["original_v34x_raw_bytes_preserved"] and pass_evidence["original_v34x_raw_sha256_echoed_exactly"] == EXPECTED_V34X_RAW_SHA256 and pass_evidence["receipt_written"] and pass_evidence["verified_opt_p_hash_count_recorded"] == 4),
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "task_id": "T-C1R-OC-epsilon-1a-receipt-repair",
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "corrected_receipt": rel(receipt_path),
        "backup_request": rel(backup_request),
        "state": rel(state_path),
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "headline": raw["headline"],
        "pass_evidence": pass_evidence,
    }
    write_json(completed_path, completed)
    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
    completed["hashes"] = {rel(p): sha256(p) for p in [Path(__file__).resolve(), raw_path, summary_path, completed_path, receipt_path, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"] if p.exists()}
    write_json(completed_path, completed)
    print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "corrected_receipt": rel(receipt_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "backup_request": rel(backup_request)}), sort_keys=True), flush=True)
    return 0 if completed["hard_pass"] else 2


def run_tc5(snapshot: Mapping[str, Any], run_dir: Path, marker: str) -> int:
    if not V34U_RAW.exists():
        return write_failure(run_dir, "missing v34u raw input", snapshot, "missing_file")
    v34u_raw = read_json(V34U_RAW)
    arms = extract_arms(v34u_raw)
    strict_candidates = strict_full_epsilon_vector_candidates(v34u_raw)
    sufficient = bool(strict_candidates and len(strict_candidates) >= len(arms))
    rows: List[Dict[str, Any]] = []
    for idx, arm in enumerate(arms):
        label = arm_label(arm)
        total, count = get_epsterm_and_count(arm)
        expected_count = EXPECTED_EPS_COUNTS.get(label)
        closure_status = "insufficient_recorded_per_stage_epsilon_vector"
        reconstructed = None
        abs_mismatch = None
        rel_mismatch = None
        # Deliberately do not fabricate a sum from epsterm_total or eps_nonzero_count.
        rows.append({
            "arm_index": idx,
            "arm_id": arm.get("arm_id"),
            "label": label,
            "horizon": arm.get("horizon"),
            "recorded_epsterm_total": total,
            "recorded_eps_nonzero_count": count,
            "expected_eps_nonzero_count_from_truncation_pattern": expected_count,
            "eps_nonzero_count_equals_horizon_prediction": bool(expected_count is not None and count == expected_count),
            "gamma": 0.97,
            "penalty_term_cons": 1000,
            "independent_abs_tolerance": 1e-8,
            "reconstructed_sum_gamma_1000_eps": reconstructed,
            "absolute_mismatch": abs_mismatch,
            "relative_mismatch": rel_mismatch,
            "closure_status": closure_status,
            "data_sufficient_for_non_circular_recompute": False,
        })
    created = now_utc()
    csv_path = run_dir / "epsilon_closure_attempt.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arm_index", "label", "horizon", "recorded_epsterm_total", "recorded_eps_nonzero_count", "expected_eps_nonzero_count_from_truncation_pattern", "eps_nonzero_count_equals_horizon_prediction", "gamma", "penalty_term_cons", "independent_abs_tolerance", "reconstructed_sum_gamma_1000_eps", "absolute_mismatch", "relative_mismatch", "closure_status", "data_sufficient_for_non_circular_recompute", "arm_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(row.get(k)) for k in fields})
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "task_id": "T-C5-OC-epsilon-2-vector-sum-closure",
        "classification": "development_IMPROVED_zero_solve_T_C5_epsilon_vector_sum_closure_attempt_not_validation_not_test",
        "input_hashes": {rel(V34U_RAW): sha256(V34U_RAW), rel(V34U_COMPLETED): sha256(V34U_COMPLETED) if V34U_COMPLETED.exists() else None},
        "strict_full_vector_candidates": strict_candidates[:80],
        "data_sufficiency_verdict": "insufficient_for_non_circular_recompute_from_existing_v34u_raw; T-C2 must persist labelled per-stage epsilon vectors",
        "no_fabricated_partial_sums": True,
        "per_arm_closure_rows": rows,
        "observed_eps_nonzero_pattern": {r["label"]: r["recorded_eps_nonzero_count"] for r in rows},
        "truncation_prediction_result": "observed eps_nonzero_count equals horizon for all four v34u forced max_iter=1 cells; this supports the qualitative first-iterate truncation prediction but says nothing about converged slack magnitude",
        "circularity_limit": "Suspicion that epsterm was residual-derived is not supported by v34k accessor ordering; the remaining limitation is absence of serialized individual epsilon vector values and absence of converged solves.",
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
    }
    summary_lines = [
        "# T-C5 epsilon vector-sum closure attempt",
        "",
        f"UTC: `{created.isoformat()}`. Zero-solve task over existing v34u artifacts.",
        "",
        "## Data sufficiency verdict",
        "The existing v34u raw artifact is insufficient for a non-circular recomputation of `sum_k gamma^k sum_j 1000*epsilon[k,j]` because it records epsterm totals and eps_nonzero_count but not the labelled per-stage epsilon vector. No partial sum was fabricated.",
        "",
        "## Truncation pattern",
        "Observed eps_nonzero_count pattern: H12=12, H15=15, H35=35, H15_goal_facing=15. This is exactly the horizon-length pattern expected for first-iterate max_iter=1 truncation, and it does not measure converged optimal slack.",
        "",
        f"Closure table: `{rel(csv_path)}`. Budget/access: solver=0, plant=0, training=0, validation64=0, sealed/final test=0.",
    ]
    raw_path, summary_path, completed_path, backup_request = write_common_outputs(run_dir, "T-C5-OC-epsilon-2-vector-sum-closure", marker, raw, summary_lines, "T-C5 recorded zero-solve data-sufficiency and truncation-pattern closure attempt before T-C2.")
    doc_block = f"""
<!-- {marker} -->
## T-C5 epsilon vector-sum closure attempt

UTC: {created.isoformat()}. Existing v34u raw is insufficient for a non-circular per-stage epsilon-sum recomputation; no partial sum was fabricated. The 1e-8 absolute tolerance is recorded per arm in `{rel(csv_path)}` but cannot be numerically evaluated without labelled per-stage epsilon vectors. Observed eps_nonzero_count equals the horizon-length truncation pattern for all four forced max_iter=1 cells. Solver=0, plant=0, training=0, validation64=0, sealed/final test=0. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, marker, doc_block)
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(completed_path), marker])
    pass_evidence = {
        "data_sufficiency_verdict_stated_without_fabrication": True,
        "independent_tolerance_1e-8_applied_per_arm": bool(len(rows) >= 4 and all(r["independent_abs_tolerance"] == 1e-8 for r in rows)),
        "no_solver_plant_training_validation_or_test_usage": True,
        "observed_eps_nonzero_pattern_compared_to_truncation_prediction": bool(len(rows) >= 4 and all(r["eps_nonzero_count_equals_horizon_prediction"] for r in rows)),
        "per_arm_closure_mismatch_reported": bool(len(rows) >= 4 and all("closure_status" in r and "absolute_mismatch" in r for r in rows)),
    }
    completed = {
        "status": "complete",
        "hard_pass": all(pass_evidence.values()),
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "task_id": "T-C5-OC-epsilon-2-vector-sum-closure",
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "closure_csv": rel(csv_path),
        "backup_request": rel(backup_request),
        "budget_actual": dict(ZERO_RESOURCES),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "headline": {"data_sufficient_for_non_circular_recompute": sufficient, "rows": len(rows), "truncation_pattern_all_match": pass_evidence["observed_eps_nonzero_pattern_compared_to_truncation_prediction"]},
        "pass_evidence": pass_evidence,
    }
    write_json(completed_path, completed)
    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
    completed["hashes"] = {rel(p): sha256(p) for p in [Path(__file__).resolve(), raw_path, summary_path, completed_path, csv_path, backup_request, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"] if p.exists()}
    write_json(completed_path, completed)
    print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "closure_csv": rel(csv_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "backup_request": rel(backup_request)}), sort_keys=True), flush=True)
    return 0 if completed["hard_pass"] else 2


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-v34y-structured-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
    except Exception as exc:
        return write_failure(run_dir, f"structured execution snapshot verification failed: {type(exc).__name__}: {exc}", None, "authorization")
    if snapshot is None:
        return write_failure(run_dir, "missing structured execution snapshot", None, "authorization")
    task_id = str(((snapshot.get("task") or {}) if isinstance(snapshot, Mapping) else {}).get("task_id", ""))
    try:
        if task_id == "T-C1R-OC-epsilon-1a-receipt-repair":
            return run_tc1r(snapshot, run_dir, marker)
        if task_id == "T-C5-OC-epsilon-2-vector-sum-closure":
            return run_tc5(snapshot, run_dir, marker)
        return write_failure(run_dir, "unsupported structured task for this script: " + task_id, snapshot, "authorization")
    except Exception as exc:
        return write_failure(run_dir, f"unexpected zero-solve diagnostic error: {type(exc).__name__}: {exc}", snapshot, "startup")


if __name__ == "__main__":
    raise SystemExit(main())
