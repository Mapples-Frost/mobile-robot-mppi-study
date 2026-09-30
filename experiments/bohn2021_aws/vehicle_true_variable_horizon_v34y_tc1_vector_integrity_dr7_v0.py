#!/usr/bin/env python3
"""v34y zero-solve correction/DR-7 bookkeeping after v34x.

Purpose:
  * correct the v34x headline around whether full per-stage epsilon vectors were
    actually serialized in the inherited v34u raw file;
  * preserve Opus DR-7 negative/operational-result registrations; and
  * leave the T-C2 execution gate in a clean state without running any solver.

This script performs no solver call, no plant rollout, no env.step/reset after
construction, no training/refit, no validation64 access and no sealed/final-test
access.  It is bookkeeping plus artifact-integrity analysis only.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_v34y_tc1_vector_integrity_dr7_v0"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v34y_tc1_vector_integrity_dr7.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_V34Y_TC1_VECTOR_INTEGRITY_DR7_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
PLAN_READY = ROOT / "docs/bohn2021_takeover/opus_lead/PLAN_READY.json"
OPUS_REPORT = ROOT / "docs/bohn2021_takeover/opus_lead/20260930T130507Z_6f3456.md"
OPUS_REQUEST = "execution-result:20260930T130525_b585d5b0"
OPUS_SHA = "d600ffca526977daac9169b97fe7b5ae1bff18befeb9428b93b9d2e590d8c276"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-v34y-tc1-vector-integrity-dr7-{STAMP}"

V34T_123137_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0_20260930T123137Z"
V34T_122045_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0_20260930T122045Z/failed.json"
V34T_123746_FAILED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34t_nonconverged_objective_contract_probe_v0_20260930T123746Z/failed.json"
V34U_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34u_active_plan_refresh_nonconverged_objective_contract_probe_v0_20260930T125233Z"
V34V_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34v_eps_objective_localization_postdiagnostic_v0_20260930T130526Z"
V34X_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34x_epsilon_provenance_audit_v0_20260930T131908Z"
RUN_122045 = ROOT / "research_artifacts/aws_runs/20260930T122045_c8b14921"
RUN_123746 = ROOT / "research_artifacts/aws_runs/20260930T123746_30cba63d"


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


def fnum(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def summarize_file(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        out.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        if path.suffix.lower() == ".json":
            try:
                j = read_json(path)
                out["json_top_keys"] = sorted([str(k) for k in j.keys()])[:40] if isinstance(j, Mapping) else type(j).__name__
                if isinstance(j, Mapping):
                    for k in ["status", "exit_status", "error", "new_solver_calls_recorded", "validation64_bank_opened", "sealed_test_accessed", "test_accessed"]:
                        if k in j:
                            out[k] = j.get(k)
            except Exception as exc:
                out["json_error"] = repr(exc)
    return out


def numeric_list_info(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return None
    flat: List[float] = []
    stack = list(value)
    while stack and len(flat) <= 5000:
        x = stack.pop(0)
        if isinstance(x, list):
            stack = list(x) + stack
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
            flat.append(float(x))
        else:
            return None
    if not flat:
        return None
    return {"numeric_count": len(flat), "nonzero_count": int(sum(abs(x) > 1e-12 for x in flat)), "first_values": flat[:12], "sum": float(math.fsum(flat)), "l1": float(math.fsum(abs(x) for x in flat))}


def scan_eps_paths(obj: Any, path: str = "root", hits: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if hits is None:
        hits = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k)
            p = f"{path}.{key}"
            lower = p.lower()
            if any(tok in lower for tok in ["_eps", ".eps", "epsilon", "epsterm", "slack"]):
                entry: Dict[str, Any] = {"path": p, "type": type(v).__name__}
                info = numeric_list_info(v)
                if info is not None:
                    entry.update(info)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    entry.update({"scalar": float(v) if math.isfinite(float(v)) else None})
                elif isinstance(v, bool):
                    entry.update({"bool": bool(v)})
                elif isinstance(v, str):
                    entry.update({"string_preview": v[:120]})
                elif isinstance(v, Mapping):
                    entry.update({"mapping_keys_preview": sorted([str(x) for x in v.keys()])[:12], "mapping_size": len(v)})
                elif isinstance(v, list):
                    entry.update({"list_len": len(v), "list_preview_types": [type(x).__name__ for x in v[:8]]})
                hits.append(entry)
            scan_eps_paths(v, p, hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if i >= 2000:
                break
            scan_eps_paths(v, f"{path}[{i}]", hits)
    return hits


def robust_vector_classification(v34u_raw: Mapping[str, Any]) -> Dict[str, Any]:
    hits = scan_eps_paths(v34u_raw)
    numeric_hits = [h for h in hits if "numeric_count" in h]
    # Full per-stage vector evidence is deliberately strict: it must not be a
    # formula summary/candidate table and must contain a numeric sequence tied to
    # _eps/slack solver-state labels.  Formula summaries such as epsterm_total,
    # eps_nonzero_count, include_eps, and candidate aliases are not full vectors.
    excluded_fragments = [
        "alias_branch_variants", "alias_16_candidate", "candidate_source", "candidate_metrics",
        "analysis.headline", "analysis.per_cell", "eps_delta", "epsterm_total", "eps_nonzero_count",
        "include_eps", "summary", "headline",
    ]
    strict_candidates: List[Dict[str, Any]] = []
    for h in numeric_hits:
        lower = h["path"].lower()
        if any(x in lower for x in excluded_fragments):
            continue
        if any(x in lower for x in ["opt_x", "solver", "label", "_eps", "slack"]):
            strict_candidates.append(h)
    arms = v34u_raw.get("arms", []) if isinstance(v34u_raw, Mapping) else []
    per_arm_required: List[Dict[str, Any]] = []
    for idx, arm in enumerate(arms if isinstance(arms, list) else []):
        hval = int(arm.get("horizon", 0) or 0) if isinstance(arm, Mapping) else 0
        arm_prefix = f"root.arms[{idx}]"
        arm_candidates = [h for h in strict_candidates if str(h.get("path", "")).startswith(arm_prefix)]
        enough = any(int(c.get("numeric_count", 0)) >= max(1, hval) for c in arm_candidates)
        per_arm_required.append({
            "arm_index": idx,
            "arm_id": arm.get("arm_id") if isinstance(arm, Mapping) else None,
            "horizon": hval,
            "strict_numeric_vector_candidate_count": len(arm_candidates),
            "has_at_least_horizon_length_candidate": bool(enough),
            "candidate_paths": [c.get("path") for c in arm_candidates[:10]],
        })
    full_available = bool(per_arm_required and all(x["has_at_least_horizon_length_candidate"] for x in per_arm_required))
    return {
        "full_per_stage_epsilon_vector_serialized": full_available,
        "strict_vector_candidate_count": len(strict_candidates),
        "strict_vector_candidates_sample": strict_candidates[:30],
        "eps_path_hit_count_total": len(hits),
        "eps_numeric_list_hit_count_total": len(numeric_hits),
        "eps_path_hits_sample": hits[:80],
        "per_arm_vector_requirements": per_arm_required,
        "classification_rule": "Formula summaries/candidate aliases/booleans/scalars are excluded. A full vector requires a numeric _eps/slack solver-state sequence under each arm with length at least the horizon.",
    }


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def dr7_registration() -> Dict[str, Any]:
    v34t_done = summarize_file(V34T_123137_DIR / "completed.json")
    v34t_raw = summarize_file(V34T_123137_DIR / "raw.json")
    v34u_done = read_json(V34U_DIR / "completed.json")
    v34v_done = read_json(V34V_DIR / "completed.json")
    fail_122045 = read_json(V34T_122045_FAILED)
    fail_123746 = read_json(V34T_123746_FAILED)
    stdout_122045 = RUN_122045 / "stdout.log"
    stderr_122045 = RUN_122045 / "stderr.log"
    stdout_123746 = RUN_123746 / "stdout.log"
    stderr_123746 = RUN_123746 / "stderr.log"
    return {
        "v34t_20260930T123137": {
            "classification": "incomplete_probe_no_objective_evidence",
            "completed": v34t_done,
            "raw": v34t_raw,
            "summary": summarize_file(V34T_123137_DIR / "summary.md"),
            "cell_metrics": summarize_file(V34T_123137_DIR / "cell_metrics.csv"),
            "registered_finding": "exit0 but hard_pass/G2 false; 0 new solver calls; 0/4 residuals; TypeError path in object TVP handling before solver; preserve as operational negative evidence, not objective-contract evidence.",
        },
        "v34u_20260930T125233": {
            "classification": "completed_forced_nonconverged_probe_G2_false_hard_defect_under_noeps_formula",
            "completed": summarize_file(V34U_DIR / "completed.json"),
            "summary": summarize_file(V34U_DIR / "summary.md"),
            "cell_metrics": summarize_file(V34U_DIR / "cell_metrics.csv"),
            "headline_subset": {k: (v34u_done.get("headline") or {}).get(k) for k in ["G2_pass", "G2_hard_objective_contract_defect", "hard_defect_arm_ids_rel_error_gt_1e-4", "new_solver_calls", "forced_nonconverged_or_near_offoptimal_cells", "validation64_episodes", "sealed_test_episodes"]},
            "registered_finding": "v34u remains a failed no-epsilon objective-contract gate: 4 solver calls, all forced Maximum_Iterations_Exceeded, all four cells >1e-4 relative error under frozen no-eps formula. Do not reclassify as passing.",
        },
        "v34v_20260930T130526": {
            "classification": "localization_only_not_gate_pass",
            "completed": summarize_file(V34V_DIR / "completed.json"),
            "summary": summarize_file(V34V_DIR / "summary.md"),
            "eps_csv": summarize_file(V34V_DIR / "eps_alias_localization.csv"),
            "headline_subset": v34v_done.get("headline"),
            "registered_finding": "v34v is zero-solve localization-only. It shows eps-including aliases close the forced-iterate residual, while input v34u G2/hard_pass remains false.",
        },
        "failure_20260930T122045_c8b14921": {
            "classification": "operational_plan_pin_failure_before_work",
            "failed_json": summarize_file(V34T_122045_FAILED),
            "run_registry": summarize_file(RUN_122045 / "registry.json"),
            "stdout_log": summarize_file(stdout_122045),
            "stderr_log": summarize_file(stderr_122045),
            "failed_json_fields": {k: fail_122045.get(k) for k in ["status", "error", "new_solver_calls_recorded", "validation64_bank_opened", "sealed_test_accessed", "test_accessed"]},
            "registered_finding": "PLAN_READY sha mismatch; failed before solver or data access; new_solver_calls_recorded=0.",
        },
        "failure_20260930T123746_30cba63d": {
            "classification": "operational_authorization_token_failure_before_work",
            "failed_json": summarize_file(V34T_123746_FAILED),
            "run_registry": summarize_file(RUN_123746 / "registry.json"),
            "stdout_log": summarize_file(stdout_123746),
            "stderr_log": summarize_file(stderr_123746),
            "failed_json_fields": {k: fail_123746.get(k) for k in ["status", "error", "new_solver_calls_recorded", "validation64_bank_opened", "sealed_test_accessed", "test_accessed"]},
            "registered_finding": "Active Opus report lacked required task token in old pin; failed before solver or data access; new_solver_calls_recorded=0.",
        },
        "original_vs_improved_label": "All v34t/v34u/v34v/v34x/v34y work is IMPROVED latency-tree/development objective-contract work, not ORIGINAL SAC reproduction evidence.",
    }


def verify_active_plan() -> Dict[str, Any]:
    ready = read_json(PLAN_READY)
    report_text = OPUS_REPORT.read_text(encoding="utf-8", errors="replace") if OPUS_REPORT.exists() else ""
    actual_sha = sha256(OPUS_REPORT) if OPUS_REPORT.exists() else None
    checks = {
        "plan_request_matches": ready.get("request_id") == OPUS_REQUEST,
        "plan_report_sha_matches_expected": ready.get("report_sha256") == OPUS_SHA,
        "report_file_sha_matches_expected": actual_sha == OPUS_SHA,
        "report_authorizes_T_C1": "T-C1" in report_text and "OC-" in report_text,
        "report_authorizes_T_C2_after_T_C1": "T-C2" in report_text and "Depends on T-C1 passing" in report_text,
        "report_authorizes_T_C3_DR7": "T-C3" in report_text and "DR-7" in report_text,
    }
    return {"ready": ready, "report": rel(OPUS_REPORT), "report_sha256_actual": actual_sha, "checks": checks, "pass": all(checks.values())}


def write_outputs(raw: Mapping[str, Any]) -> Dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    vector = raw["vector_integrity"]
    dr7 = raw["DR7_registration"]
    # CSV of strict vector candidates and per-arm requirements.
    cand_csv = RUN_DIR / "epsilon_vector_path_scan.csv"
    fields = ["kind", "arm_index", "arm_id", "horizon", "path", "numeric_count", "nonzero_count", "sum", "l1", "has_at_least_horizon_length_candidate"]
    with cand_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in vector["per_arm_vector_requirements"]:
            w.writerow({"kind": "per_arm_requirement", **{k: clean(r.get(k)) for k in ["arm_index", "arm_id", "horizon", "has_at_least_horizon_length_candidate"]}})
        for c in vector["strict_vector_candidates_sample"]:
            w.writerow({"kind": "strict_candidate", "path": c.get("path"), "numeric_count": c.get("numeric_count"), "nonzero_count": c.get("nonzero_count"), "sum": c.get("sum"), "l1": c.get("l1")})
    summary = RUN_DIR / "summary.md"
    h = raw["headline"]
    lines = [
        "# v34y T-C1 vector-integrity correction and DR-7 registration",
        "",
        f"UTC: `{raw['created_utc']}`. Zero-solve bookkeeping/correction under active Opus plan `{OPUS_REQUEST}`.",
        "",
        "## Budget and access",
        "- New solver calls: `0`; plant/env.step/reset-after-construction: `0`; training/refit: `0`; validation64: `0`; sealed/final test: `0`.",
        f"- Active plan self-consistency: `{raw['plan_check']['pass']}`.",
        "",
        "## T-C1 vector-integrity correction",
        f"- v34x recorded `individual_epsilon_vector_values_serialized_in_v34u_raw={h['v34x_recorded_full_vector_flag']}`.",
        f"- Robust reclassification: `full_per_stage_epsilon_vector_serialized={h['full_per_stage_epsilon_vector_serialized']}`.",
        f"- Strict full-vector candidates found: `{vector['strict_vector_candidate_count']}`; per-arm all-horizon coverage: `{h['per_arm_full_vector_coverage_all']}`.",
        f"- Independent epsilon contribution reproduction remains: `{h['independent_contribution_reproduced_count']}/4`; max |alias_pair_delta - epsterm_total| `{h['max_abs_alias_pair_delta_minus_epsterm_total']}`.",
        f"- Corrected T-C1 status: `{h['corrected_T_C1_status']}`.",
        "",
        "Interpretation: the source/provenance and non-circular contribution checks remain satisfied, but v34u did not provide a full per-stage `_eps` vector suitable for reporting every internal slack value. T-C2 remains the planned place to capture those per-stage values at converged and truncated points before any downstream contract-consumer task.",
        "",
        "## DR-7 registrations",
        "- v34t 20260930T123137: incomplete operational negative evidence, 0 solver calls, no objective residual evidence.",
        "- v34u 20260930T125233: completed forced nonconverged probe, G2/hard_pass false under no-eps formula, 4 hard-defect arm IDs, 4 solver calls, no validation/test.",
        "- v34v 20260930T130526: localization-only postdiagnostic, input v34u G2 remains false.",
        "- failures 20260930T122045 and 20260930T123746: pre-work operational failures, both recorded with new_solver_calls_recorded=0 and no validation/test.",
        "- Labeling: IMPROVED development/objective-contract work only; not ORIGINAL SAC reproduction evidence.",
        "",
        f"Raw: `{rel(RUN_DIR/'raw.json')}`; completed: `{rel(RUN_DIR/'completed.json')}`; CSV: `{rel(cand_csv)}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(BACKUP_REQUEST, {
        "request": "backup_after_v34y_tc1_vector_integrity_dr7",
        "created_utc": raw["created_utc"],
        "backup_required_before_more_unique_science": True,
        "reason": "v34y corrected v34x vector-serialization flag, registered DR-7 negative results, and updated state/docs before any T-C2 solver-bearing gate.",
        "must_cover": [rel(Path(__file__).resolve()), rel(RUN_DIR), rel(STATE), rel(BACKUP_REQUEST), rel(RESPONSE_LOG), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "new_solver_calls": 0,
        "new_plant_steps": 0,
        "env_step_calls_after_construction": 0,
        "new_training_or_gradient_steps": 0,
        "selector_refits": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_gate": "After verified backup, if PLAN_READY remains execution-result:20260930T130525_b585d5b0, prepare/run T-C2 bounded converged objective-contract gate. T-C2 must serialize per-stage epsilon vectors and measured timings; no validation64 or sealed test.",
    })
    block = f"""
<!-- {MARKER} -->
## v34y T-C1 vector-integrity correction and DR-7 registration

UTC: {raw['created_utc']}. Zero-solve correction/bookkeeping. Budget: solver=0, plant/env.step/reset-after-construction=0, training/refit=0, validation64=0, sealed/final test=0. Active Opus plan `{OPUS_REQUEST}` verified={raw['plan_check']['pass']}. v34x recorded full-vector flag={h['v34x_recorded_full_vector_flag']}; robust full per-stage vector serialization={h['full_per_stage_epsilon_vector_serialized']}; independent epsilon contribution reproduction remains {h['independent_contribution_reproduced_count']}/4 with max alias-vs-epsterm difference {h['max_abs_alias_pair_delta_minus_epsterm_total']}. Corrected T-C1 status: {h['corrected_T_C1_status']}. DR-7 registered v34t(123137) incomplete 0-solver negative evidence, v34u G2/hard_pass false with four hard-defect forced cells, v34v localization-only, and pre-work failures 122045/123746 with 0 solver calls and no validation/test. Evidence: `{rel(summary)}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(cand_csv)}`, `{rel(RUN_DIR/'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`. Next safe action only after verified backup: T-C2 converged objective-contract gate with explicit per-stage epsilon-vector capture.
"""
    for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
        append_if_missing(doc, MARKER, block)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after v34y T-C1 vector-integrity correction and DR-7\n\n"
        + f"UTC: {raw['created_utc']}\n\n"
        + f"Headline: {json.dumps(clean(h), sort_keys=True)}\n\n"
        + "T-C1 source/non-circular epsilon-contribution evidence remains usable, but full per-stage epsilon vectors were not present in the inherited v34u raw artifact. T-C2 must explicitly serialize them.\n\n"
        + f"Artifacts: `{rel(summary)}`, `{rel(RUN_DIR/'raw.json')}`, `{rel(cand_csv)}`, `{rel(RUN_DIR/'completed.json')}`.\n\n"
        + f"Backup required before T-C2: `{rel(BACKUP_REQUEST)}`. Do not open validation64 or sealed/final test.\n",
        encoding="utf-8",
    )
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([raw["created_utc"], NAME, "development_IMPROVED_zero_solve_T_C1_correction_DR7_bookkeeping_not_validation_not_test", "correct v34x vector flag; register negative results", "opened_development_artifacts_only_no_validation_no_test", 0, 0, 0, 0, 0, False, rel(RUN_DIR/"completed.json"), MARKER])
    files = [Path(__file__).resolve(), RUN_DIR / "raw.json", summary, cand_csv, BACKUP_REQUEST, STATE, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
    completed = {
        "status": "complete",
        "hard_pass": True,
        "created_utc": raw["created_utc"],
        "classification": raw["classification"],
        "summary": rel(summary),
        "raw": rel(RUN_DIR / "raw.json"),
        "epsilon_vector_path_scan_csv": rel(cand_csv),
        "backup_request": rel(BACKUP_REQUEST),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "budget_actual": raw["budget_actual"],
        "headline": h,
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    }
    write_json(RUN_DIR / "completed.json", completed)
    return completed


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    required = [PLAN_READY, OPUS_REPORT, V34U_DIR / "raw.json", V34U_DIR / "completed.json", V34V_DIR / "completed.json", V34X_DIR / "raw.json", V34X_DIR / "completed.json", V34T_122045_FAILED, V34T_123746_FAILED]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("missing required input(s): " + repr(missing))
    plan = verify_active_plan()
    if not plan["pass"]:
        raise RuntimeError("active plan self-consistency failed: " + repr(plan["checks"]))
    v34u_raw = read_json(V34U_DIR / "raw.json")
    v34x_done = read_json(V34X_DIR / "completed.json")
    vector = robust_vector_classification(v34u_raw)
    dr7 = dr7_registration()
    h_x = v34x_done.get("headline") or {}
    indep_count = int(h_x.get("independent_contribution_reproduced_count", 0) or 0)
    max_pair = h_x.get("max_abs_alias_pair_delta_minus_epsterm_total")
    full_serialized = bool(vector["full_per_stage_epsilon_vector_serialized"])
    corrected_status = "source_and_non_circular_contribution_pass_but_full_vector_values_deferred_to_T_C2" if indep_count >= 3 else "T_C1_failed_non_circular_contribution_threshold"
    headline = {
        "v34x_recorded_full_vector_flag": bool(h_x.get("individual_epsilon_vector_values_serialized_in_v34u_raw")),
        "full_per_stage_epsilon_vector_serialized": full_serialized,
        "v34x_full_vector_flag_false_positive": bool(h_x.get("individual_epsilon_vector_values_serialized_in_v34u_raw")) and not full_serialized,
        "per_arm_full_vector_coverage_all": bool(full_serialized),
        "independent_contribution_reproduced_count": indep_count,
        "independent_contribution_threshold": ">=3/4",
        "max_abs_alias_pair_delta_minus_epsterm_total": max_pair,
        "corrected_T_C1_status": corrected_status,
        "T_C1_dependency_for_T_C2": bool(indep_count >= 3),
        "DR7_registered": True,
        "new_solver_calls": 0,
        "plant_steps": 0,
        "env_step_calls_after_construction": 0,
        "training_or_refit": 0,
        "validation64_episodes": 0,
        "sealed_test_episodes": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_IMPROVED_zero_solve_T_C1_vector_integrity_correction_DR7_bookkeeping_not_validation_not_test",
        "active_lead": "claude-opus-5-5",
        "plan_check": plan,
        "purpose": "Correct v34x vector-serialization headline, register DR-7 negative results, and preserve state before T-C2.",
        "input_hashes": {rel(p): sha256(p) for p in required if p.exists()},
        "vector_integrity": vector,
        "DR7_registration": dr7,
        "headline": headline,
        "budget_actual": {
            "new_solver_calls": 0,
            "plant_steps": 0,
            "env_reset_calls_after_construction": 0,
            "env_step_calls_after_construction": 0,
            "new_training_or_gradient_steps": 0,
            "selector_refits": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "next_gate": "External backup first; then T-C2 solver-bearing converged objective-contract gate if PLAN_READY remains unchanged. T-C2 must serialize per-stage epsilon vectors and timing metrics.",
    }
    write_json(RUN_DIR / "raw.json", raw)
    completed = write_outputs(raw)
    print(json.dumps(clean({"completed": completed["raw"].replace("raw.json", "completed.json"), "summary": completed["summary"], "headline": headline, "backup_request": completed["backup_request"]}), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
