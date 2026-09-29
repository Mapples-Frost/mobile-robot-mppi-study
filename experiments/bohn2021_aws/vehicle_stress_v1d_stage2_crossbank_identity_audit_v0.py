#!/usr/bin/env python3
"""Vehicle stress-v1d / Stage2 cross-bank identity audit v0.

Development-only no-simulation diagnostic.  The immediately preceding
state-coverage diagnostic combined stress-v1 Stage1/v1d coverage evidence with
older Stage2 v0b prefix-blocked candidates.  Before freezing any v1e rollout,
verify whether those Stage2 case ids denote the same scenario bank/cases as the
stress-v1 cases that v1d missed.

No rollouts, no candidate resets, no training/refit, no validation64 bank and no
sealed-test access.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_stress_v1d_stage2_crossbank_identity_audit_v0"
STAMP = "20260929T0445Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_{NAME.upper()}_{STAMP}.json"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

COVERAGE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0_20260929T0435Z/raw.json"
STAGE2_V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json"
STAGE2_V0B_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0b_frozen_20260928T1748Z.json"
STRESS_V1_STAGE1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/raw.json"
STRESS_V1_STAGE1_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json"
STRESS_V1_STAGE1_POST_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/summary.md"
V1D_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_jsonable(v) for v in value]
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def parse_time(x: Any) -> Optional[dt.datetime]:
    if not isinstance(x, str):
        return None
    try:
        t = dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def require_inputs() -> Dict[str, str]:
    required = [COVERAGE_RAW, STAGE2_V0B_RAW, STAGE2_V0B_PROTOCOL, STRESS_V1_STAGE1_RAW, STRESS_V1_STAGE1_POST_RAW, STRESS_V1_STAGE1_POST_SUMMARY, V1D_RAW]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    return {rel(p): sha256(p) for p in required}


def count_by(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def v1_case_map_from_coverage(coverage: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in coverage.get("case_coverage_rows") or []:
        case = safe_int(row.get("case"))
        if case >= 0:
            out[case] = {
                "case": case,
                "bank_label": "stress_v1_stage1_postdiagnostic_coverage",
                "group": row.get("group"),
                "source_candidate_index": row.get("source_candidate_index"),
                "theta_r": row.get("theta_r"),
                "traj_steps": row.get("traj_steps"),
                "runner_episode_material": row.get("runner_episode_material"),
                "episode_physical_gain_ge_3_case": row.get("episode_physical_gain_ge_3_case"),
                "material_strict_horizons_vs_H15": row.get("material_strict_horizons_vs_H15"),
                "v1d_selected_target_count": row.get("v1d_selected_target_count"),
            }
    return out


def stage2_case_map_from_protocol(protocol: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for t in protocol.get("targets") or []:
        case = safe_int(t.get("case"))
        if case < 0:
            continue
        meta = t.get("case_metadata") or {}
        cur = out.setdefault(case, {
            "case": case,
            "bank_label": "stage2_v0b_protocol_source_stage1",
            "source_candidate_index": t.get("source_candidate_index"),
            "case_group": t.get("case_group"),
            "theta_r": meta.get("theta_r"),
            "abs_theta_r": meta.get("abs_theta_r"),
            "traj_steps": meta.get("traj_steps"),
            "min_reference_obstacle_clearance": meta.get("min_reference_obstacle_clearance"),
            "target_branch_steps": [],
            "selection_roles": [],
        })
        cur["target_branch_steps"].append(safe_int(t.get("branch_step")))
        role = t.get("selection_role")
        if role not in cur["selection_roles"]:
            cur["selection_roles"].append(role)
    for cur in out.values():
        cur["target_branch_steps"] = sorted(set(cur["target_branch_steps"]))
    return out


def near_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    aa = safe_float(a)
    bb = safe_float(b)
    return math.isfinite(aa) and math.isfinite(bb) and abs(aa - bb) <= tol


def compare_case(case: int, v1: Mapping[str, Any], st2: Mapping[str, Any]) -> Dict[str, Any]:
    source_equal = safe_int(v1.get("source_candidate_index")) == safe_int(st2.get("source_candidate_index"))
    theta_equal = near_equal(v1.get("theta_r"), st2.get("theta_r"), 1e-9)
    traj_equal = safe_int(v1.get("traj_steps")) == safe_int(st2.get("traj_steps"))
    compatible = bool(source_equal and theta_equal and traj_equal)
    return {
        "case": case,
        "v1_coverage": v1,
        "stage2_v0b": st2,
        "source_candidate_index_equal": source_equal,
        "theta_r_equal": theta_equal,
        "traj_steps_equal": traj_equal,
        "same_case_identity_supported": compatible,
        "theta_r_abs_diff": None if not (math.isfinite(safe_float(v1.get("theta_r"))) and math.isfinite(safe_float(st2.get("theta_r")))) else abs(safe_float(v1.get("theta_r")) - safe_float(st2.get("theta_r"))),
    }


def collect_blocked_cases(coverage: Mapping[str, Any]) -> Dict[str, Any]:
    blocked = coverage.get("stage2_prefix_blocked_material_candidates") or []
    return {
        "blocked_candidate_count": len(blocked),
        "blocked_cases": sorted(set(safe_int(r.get("case")) for r in blocked if safe_int(r.get("case")) >= 0)),
        "blocked_candidates": blocked,
    }


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def make_markdown(result: Mapping[str, Any]) -> str:
    h = result["headline"]
    lines = [
        "# Vehicle stress-v1d / Stage2 cross-bank identity audit v0",
        "",
        f"UTC: `{result['created_utc']}`. Development-only no-simulation audit; validation64 and sealed test stayed closed.",
        "",
        "## Headline",
        "",
        f"- Classification: `{h['classification']}`.",
        f"- Stage2 v0b source Stage1 raw: `{h['stage2_source_stage1_raw']}`.",
        f"- Stress-v1 Stage1 raw: `{h['stress_v1_stage1_raw']}`.",
        f"- Stage2 v0b rollout time precedes stress-v1 Stage1/postdiagnostic: `{h['stage2_predates_stress_v1_postdiagnostic']}`.",
        f"- Prefix-blocked Stage2 candidate cases in prior coverage diagnostic: `{h['blocked_candidate_cases']}`.",
        f"- Same-case identity supported for those cases: `{h['same_identity_cases']}`; mismatched cases: `{h['mismatched_cases']}`.",
        f"- Corrective decision: `{h['corrective_decision']}`.",
        f"- Train/refit now: `{h['train_or_refit_now']}`.",
        f"- Next action: `{h['next_action']}`.",
        "",
        "## Case identity comparisons",
        "",
        "| case | v1 source idx/theta/traj/group | stage2-v0b source idx/theta/traj/group | same identity? |",
        "|---:|---|---|---:|",
    ]
    for row in result["case_identity_comparisons"]:
        v1 = row.get("v1_coverage") or {}
        st2 = row.get("stage2_v0b") or {}
        lines.append("| {case} | `{vsrc}` / `{vtheta}` / `{vtraj}` / `{vgroup}` | `{ssrc}` / `{stheta}` / `{straj}` / `{sgroup}` | {same} |".format(
            case=row["case"],
            vsrc=v1.get("source_candidate_index"), vtheta=v1.get("theta_r"), vtraj=v1.get("traj_steps"), vgroup=v1.get("group"),
            ssrc=st2.get("source_candidate_index"), stheta=st2.get("theta_r"), straj=st2.get("traj_steps"), sgroup=st2.get("case_group"),
            same=int(bool(row.get("same_case_identity_supported"))),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "- SCENARIOS/IMPLEMENTATION: the previous state-coverage result correctly shows that v1d missed stress-v1 episode-positive cases 4 and 5, but the Stage2 prefix-blocked case5 candidates came from an older stress-v0 Stage1 bank and must not be treated as same-case evidence for stress-v1/v1d case5.",
        "- REWARD/TERMINAL/VALUE: objective/terminal repair remains unsupported; this audit only corrects candidate provenance and does not create selector labels.",
        "- TRAINING: do not train/refit now. The valid next rollout, if backed up, must target stress-v1 H15 traces for missed cases 4/5 (and controls) directly rather than replaying older Stage2-v0b case ids as positives.",
        "- COMPARISONS: no timing or adaptive-performance claim is made; all future speed claims still require blocked repeated measured timing against fixed-H/Pareto baselines.",
        "",
        "## Revised next discriminating intervention",
        "",
        "Freeze a v1e stress-v1-only targeted common-prefix smoke: choose states from stress-v1 Stage1 H15 traces for cases 4 and 5 (plus runner-material case1/6 and negative/control cases), before any non-H15 branch outcomes; include horizons [10,15,20,25,30,45,50] and terminal-stable modes [zero_terminal,H15_common_terminal]. Acceptance before any refit remains >=2 terminal-stable material positive states across >=2 stress-v1 cases, retained controls, and zero prefix/state/safety artifacts. This replaces using the older Stage2-v0b prefix-blocked candidates as v1d targets.",
        "",
        f"Backup request: `{result['backup_request']}`.",
    ]
    return "\n".join(lines) + "\n"


def run() -> int:
    if OUT.exists() and (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"partial output exists; inspect before rerun: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes = require_inputs()
    created = now_utc()

    coverage = read_json(COVERAGE_RAW)
    stage2_raw = read_json(STAGE2_V0B_RAW)
    stage2_protocol = read_json(STAGE2_V0B_PROTOCOL)
    stress_v1_raw = read_json(STRESS_V1_STAGE1_RAW)
    stress_v1_post = read_json(STRESS_V1_STAGE1_POST_RAW)
    v1d_raw = read_json(V1D_RAW)

    blocked_info = collect_blocked_cases(coverage)
    v1_map = v1_case_map_from_coverage(coverage)
    st2_map = stage2_case_map_from_protocol(stage2_protocol)
    compare_cases = sorted(set(blocked_info["blocked_cases"]).union({1, 4, 5, 6}).intersection(set(v1_map.keys()).union(set(st2_map.keys()))))
    comparisons = []
    for case in compare_cases:
        if case in v1_map and case in st2_map:
            comparisons.append(compare_case(case, v1_map[case], st2_map[case]))
        else:
            comparisons.append({"case": case, "v1_coverage": v1_map.get(case), "stage2_v0b": st2_map.get(case), "same_case_identity_supported": False, "missing_side": "v1" if case not in v1_map else "stage2_v0b"})

    same_cases = [r["case"] for r in comparisons if r.get("same_case_identity_supported")]
    mismatched_cases = [r["case"] for r in comparisons if not r.get("same_case_identity_supported")]

    st2_source = (((stage2_protocol.get("source_stage1") or {}).get("stage1_raw")) or "")
    stress_v1_path = rel(STRESS_V1_STAGE1_RAW)
    st2_time = parse_time(stage2_raw.get("created_utc"))
    v1_post_time = parse_time(stress_v1_post.get("created_utc"))
    stage2_predates = bool(st2_time is not None and v1_post_time is not None and st2_time < v1_post_time)
    path_bank_mismatch = "vehicle_stress_scenario_opportunity_probe_v0_stage1" in st2_source and "vehicle_stress_scenario_opportunity_probe_v1_stage1" in stress_v1_path
    blocked_mismatched = bool(blocked_info["blocked_cases"] and not set(blocked_info["blocked_cases"]).issubset(set(same_cases)))

    if path_bank_mismatch or stage2_predates or blocked_mismatched:
        classification = "stage2_prefix_blocked_candidates_are_cross_bank_not_valid_v1d_targets"
        corrective_decision = "downgrade prior prefix-blocked case5 evidence to legacy/provenance diagnostic only; do not use it as stress-v1/v1d target evidence"
        next_action = "freeze stress-v1-only v1e targeted common-prefix target-preparation/protocol before any rollout"
    else:
        classification = "stage2_prefix_blocked_candidates_same_bank_identity_supported"
        corrective_decision = "prior prefix-blocked candidates may remain candidate targets but still are not positive labels"
        next_action = "freeze v1e common-prefix replay including the verified same-bank blocked targets after backup"

    v1d_cases = []
    try:
        v1d_cases = sorted(set(safe_int(r.get("case")) for r in ((v1d_raw.get("analysis") or {}).get("state_rows") or []) if safe_int(r.get("case")) >= 0))
    except Exception:
        v1d_cases = []

    headline = {
        "classification": classification,
        "stage2_source_stage1_raw": st2_source,
        "stress_v1_stage1_raw": stress_v1_path,
        "stage2_created_utc": None if st2_time is None else st2_time.isoformat(),
        "stress_v1_postdiagnostic_created_utc": None if v1_post_time is None else v1_post_time.isoformat(),
        "stage2_predates_stress_v1_postdiagnostic": stage2_predates,
        "path_bank_mismatch": path_bank_mismatch,
        "blocked_candidate_count": blocked_info["blocked_candidate_count"],
        "blocked_candidate_cases": blocked_info["blocked_cases"],
        "same_identity_cases": same_cases,
        "mismatched_cases": mismatched_cases,
        "v1d_selected_cases": v1d_cases,
        "corrective_decision": corrective_decision,
        "train_or_refit_now": False,
        "next_action": next_action,
    }

    result = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_no_simulation_crossbank_identity_audit_no_validation64_no_test",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_hashes": input_hashes,
        "headline": headline,
        "case_identity_comparisons": comparisons,
        "blocked_stage2_candidates_from_prior_coverage": blocked_info,
        "stage2_protocol_source_stage1": stage2_protocol.get("source_stage1"),
        "stress_v1_stage1_top_keys": sorted(str(k) for k in stress_v1_raw.keys()) if isinstance(stress_v1_raw, Mapping) else [],
        "stress_v1_postdiagnostic_top_keys": sorted(str(k) for k in stress_v1_post.keys()) if isinstance(stress_v1_post, Mapping) else [],
        "decision_rules": {
            "do_not_use_stage2_v0b_case_ids_as_stress_v1_targets_if_mismatch": bool(classification == "stage2_prefix_blocked_candidates_are_cross_bank_not_valid_v1d_targets"),
            "do_not_train_or_refit_current_labels": True,
            "do_not_claim_absence_from_v1d_zero_labels_alone": True,
            "next_rollout_must_freeze_stress_v1_only_targets_before_non_H15_outcomes": True,
        },
        "backup_request": rel(BACKUP_REQ),
    }
    write_json(OUT / "raw.json", result)
    summary = make_markdown(result)
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(summary, encoding="utf-8")
    marker = f"{NAME}-{STAMP}"
    append_docs("\n".join([f"## {marker}", "", summary.strip()]), marker)
    write_json(BACKUP_REQ, {
        "request": "backup_after_vehicle_stress_v1d_stage2_crossbank_identity_audit_v0",
        "created_utc": created.isoformat(),
        "reason": "Preserve no-simulation cross-bank identity correction before any v1e stress-v1-only targeted common-prefix rollout.",
        "artifacts": [rel(OUT / "raw.json"), rel(OUT / "summary.md"), rel(OUT / "completed.json"), rel(STATE), rel(Path(__file__).resolve())],
    })
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": headline,
        "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "state": rel(STATE), "backup_request": rel(BACKUP_REQ)},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "classification": classification,
        "blocked_candidate_cases": blocked_info["blocked_cases"],
        "same_identity_cases": same_cases,
        "mismatched_cases": mismatched_cases,
        "train_or_refit_now": False,
        "next_action": next_action,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": rel(BACKUP_REQ),
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-accept-development-no-simulation", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_no_simulation:
        raise RuntimeError("explicit --i-accept-development-no-simulation required")
    try:
        return run()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "passed": False,
            "created_utc": now_utc().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve failure. Repair only schema/path extraction; do not infer scientific outcome or run simulations.",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
