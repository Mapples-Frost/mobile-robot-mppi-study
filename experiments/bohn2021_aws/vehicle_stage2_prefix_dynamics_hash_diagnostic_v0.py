#!/usr/bin/env python3
"""Stage2 stress-continuation prefix dynamics-hash diagnostic v0.

Development-only analysis of existing Stage2 v0b artifacts.  No simulations,
no training/refit, no historical validation64 access and no sealed-test access.

Why this exists
---------------
The frozen Stage2 label gate reported prefix_clean_sha256 mismatches for every
non-H15 branch.  A later schema/source audit found that the saved clean prefix
still contains decision.branch_horizon metadata, so hashes may differ even when
the replayed H15 dynamics prefix is identical.  This script recomputes a
"dynamics-only" prefix hash from saved traces, ignoring branch-selection
metadata/timing/logging, and recomputes the development-only material label count
with that one diagnostic substitution.  It does not modify the frozen Stage2
result or its acceptance rules.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T1845Z"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stage2_prefix_dynamics_hash_diagnostic_v0_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stage2_prefix_dynamics_hash_diagnostic_v0_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_PREFIX_DYNAMICS_HASH_DIAGNOSTIC_V0_{STAMP}.json"
STAGE2_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z"
STAGE2_RAW = STAGE2_DIR / "raw.json"
STAGE2_COMPLETED = STAGE2_DIR / "completed.json"
STAGE2_SUMMARY = STAGE2_DIR / "summary.md"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_stage2_prefix_dynamics_hash_diagnostic_v0.py"
MARKER = f"vehicle-stage2-prefix-dynamics-hash-diagnostic-v0-{STAMP}"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45]
MATERIAL_GAIN_THRESHOLD = 3.0


class DiagnosticError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=serial) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    # Round below solver/reporting noise while keeping deterministic replay differences visible.
    return round(x, 12)


def norm_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return finite_float(value)
    if isinstance(value, list):
        return [norm_value(v) for v in value]
    if isinstance(value, tuple):
        return [norm_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): norm_value(v) for k, v in sorted(value.items())}
    try:
        return finite_float(value)
    except Exception:
        return str(value)


def select_input(row: Mapping[str, Any]) -> Dict[str, Any]:
    inp = row.get("input") or {}
    return {"u_omega": norm_value(inp.get("u_omega")), "u_s": norm_value(inp.get("u_s"))}


def select_state(row: Mapping[str, Any], key: str) -> Dict[str, Any]:
    state = row.get(key) or {}
    return {"x": norm_value(state.get("x")), "y": norm_value(state.get("y")), "theta": norm_value(state.get("theta"))}


def recovery_dynamics(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    attempts = []
    for attempt in (row.get("recovery") or {}).get("attempts") or []:
        attempts.append({
            "kind": attempt.get("kind"),
            "success": bool(attempt.get("success")) if attempt.get("success") is not None else None,
            "return_status": attempt.get("return_status"),
            "iterations": int(attempt.get("iterations")) if attempt.get("iterations") is not None else None,
            "finite": bool(attempt.get("finite")) if attempt.get("finite") is not None else None,
            "constraint_residual": norm_value(attempt.get("constraint_residual")),
            "bound_residual": norm_value(attempt.get("bound_residual")),
            "accepted": bool(attempt.get("accepted")) if attempt.get("accepted") is not None else None,
        })
    return attempts


def dynamics_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    # Deliberately omit decision/branch metadata, timing, file paths and solver wall time.
    return {
        "horizon": int(row.get("horizon", -1)),
        "previous_state": select_state(row, "previous_state"),
        "state": select_state(row, "state"),
        "input": select_input(row),
        "observation": norm_value(row.get("observation")),
        "next_observation": norm_value(row.get("next_observation")),
        "performance": norm_value(row.get("performance")),
        "constraint": norm_value(row.get("constraint")),
        "compute": norm_value(row.get("compute")),
        "reward": norm_value(row.get("reward")),
        "solver_success": bool(row.get("solver_success")) if row.get("solver_success") is not None else None,
        "termination": norm_value(row.get("termination")),
        "recovery_attempts_dynamic": recovery_dynamics(row),
    }


def dynamics_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    return [dynamics_row(row) for row in list(trace[: min(branch_step, len(trace))])]


def max_abs(a: Iterable[Any], b: Iterable[Any]) -> float:
    out = 0.0
    for x, y in zip(a, b):
        fx, fy = finite_float(x), finite_float(y)
        if fx is None or fy is None:
            continue
        out = max(out, abs(fx - fy))
    return float(out)


def flat_numbers(value: Any) -> List[float]:
    nums: List[float] = []
    if isinstance(value, dict):
        for k in sorted(value):
            nums.extend(flat_numbers(value[k]))
    elif isinstance(value, list):
        for v in value:
            nums.extend(flat_numbers(v))
    else:
        f = finite_float(value)
        if f is not None:
            nums.append(f)
    return nums


def first_dynamics_divergence(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], tol: float = 1e-9) -> Optional[Dict[str, Any]]:
    n = min(len(a), len(b))
    for i in range(n):
        ar, br = dynamics_row(a[i]), dynamics_row(b[i])
        if ar == br:
            continue
        # Find a compact top-level cause.
        details = []
        for key in sorted(set(ar) | set(br)):
            av, bv = ar.get(key), br.get(key)
            if av == bv:
                continue
            delta = max_abs(flat_numbers(av), flat_numbers(bv))
            if delta > tol or not flat_numbers(av) or not flat_numbers(bv):
                details.append({"field": key, "max_abs_numeric_delta": delta, "ref": av, "candidate": bv})
        return {"step": i, "details": details[:5]}
    if len(a) != len(b):
        return {"step": n, "details": [{"field": "prefix_length", "ref": len(a), "candidate": len(b)}]}
    return None


def state_tuple_from_summary(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    s = row.get("branch_previous_state") or {}
    vals = []
    for k in ("x", "y", "theta"):
        v = finite_float(s.get(k))
        vals.append(float("nan") if v is None else float(v))
    return (vals[0], vals[1], vals[2])


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def no_regression(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if bool(reference.get("success")) and not bool(candidate.get("success")):
        return False
    if bool(candidate.get("constraint")) and not bool(reference.get("constraint")):
        return False
    if int(candidate.get("initial_failed_steps", 0)) > int(reference.get("initial_failed_steps", 0)):
        return False
    if int(candidate.get("solver_failure_steps", 0)) > int(reference.get("solver_failure_steps", 0)):
        return False
    return True


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    n = len(vals)
    return {"count": n, "min": vals[0], "median": vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]), "mean": sum(vals) / n, "max": vals[-1]}


def load_trace_for_episode(ep: Mapping[str, Any]) -> List[Dict[str, Any]]:
    p = ROOT / str(ep.get("path", "")) / "trace.json"
    if not p.exists():
        raise DiagnosticError(f"missing trace for episode: {rel(p)}")
    trace = read_json(p)
    if not isinstance(trace, list):
        raise DiagnosticError(f"trace is not a list: {rel(p)}")
    return trace


def verify_inputs() -> Dict[str, Any]:
    for p in (STAGE2_RAW, STAGE2_COMPLETED, STAGE2_SUMMARY, THIS_SCRIPT):
        if not p.exists():
            raise DiagnosticError(f"required input missing: {rel(p)}")
    completed = read_json(STAGE2_COMPLETED)
    if completed.get("passed") is not True or completed.get("hard_pass") is not True:
        raise DiagnosticError("Stage2 completed marker did not pass")
    for flag in ("historical_validation64_bank_opened", "sealed_test_accessed"):
        if completed.get(flag) is not False:
            raise DiagnosticError(f"Stage2 completed marker access flag invalid: {flag}={completed.get(flag)!r}")
    raw = read_json(STAGE2_RAW)
    for flag in ("historical_validation64_bank_opened", "validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened"):
        if raw.get(flag) not in (False, None):
            raise DiagnosticError(f"Stage2 raw access flag invalid: {flag}={raw.get(flag)!r}")
    if raw.get("method") != "vehicle_stress_scenario_stage2_continuation_v0_identical_H15_prefix_branch_fixed_H":
        raise DiagnosticError("unexpected Stage2 method")
    return raw


def analyze(raw: Mapping[str, Any]) -> Dict[str, Any]:
    episodes = raw.get("episodes") or []
    if len(episodes) != 72:
        raise DiagnosticError(f"expected 72 Stage2 episodes, got {len(episodes)}")
    groups: Dict[int, Dict[int, Dict[str, Any]]] = {}
    traces: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    trace_hashes: Dict[str, str] = {}
    for ep in episodes:
        tid, h = int(ep["target_index"]), int(ep["branch_horizon"])
        groups.setdefault(tid, {})[h] = dict(ep)
        trace_path = ROOT / ep["path"] / "trace.json"
        traces[(tid, h)] = load_trace_for_episode(ep)
        trace_hashes[rel(trace_path)] = sha256(trace_path)
    original_prefix_mismatch_non_h15 = 0
    dynamics_prefix_mismatch_non_h15 = 0
    state_distance_gt_tol_non_h15 = 0
    safety_regression_non_h15 = 0
    missing_reference = 0
    missing_horizon = 0
    branch_not_reached = 0
    material_state_count = 0
    positive_cases: List[int] = []
    positive_roles: List[str] = []
    positive_horizon_counts: Dict[str, int] = {}
    large_harms = 0
    all_distances: List[float] = []
    pair_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    divergence_examples: List[Dict[str, Any]] = []
    metadata_artifact_examples: List[Dict[str, Any]] = []
    for tid in sorted(groups):
        by_h = groups[tid]
        ref = by_h.get(PREFIX_H)
        if ref is None:
            missing_reference += 1
            continue
        ref_trace = traces[(tid, PREFIX_H)]
        branch_step = int(ref["branch_step"])
        ref_dyn_hash = canonical_sha(dynamics_prefix(ref_trace, branch_step))
        ref_original_hash = ref.get("prefix_clean_sha256")
        ref_state = state_tuple_from_summary(ref)
        material_horizons: List[int] = []
        comparisons: List[Dict[str, Any]] = []
        for h in BRANCH_HORIZONS:
            cand = by_h.get(h)
            if cand is None:
                missing_horizon += 1
                continue
            cand_trace = traces[(tid, h)]
            cand_dyn_hash = canonical_sha(dynamics_prefix(cand_trace, branch_step))
            original_match = bool(cand.get("prefix_clean_sha256") == ref_original_hash)
            dynamics_match = bool(cand_dyn_hash == ref_dyn_hash)
            if h != PREFIX_H and not original_match:
                original_prefix_mismatch_non_h15 += 1
            if h != PREFIX_H and not dynamics_match:
                dynamics_prefix_mismatch_non_h15 += 1
                if len(divergence_examples) < 5:
                    divergence_examples.append({
                        "target_index": tid,
                        "case": int(cand.get("case", -1)),
                        "branch_step": branch_step,
                        "horizon": h,
                        "first_dynamics_divergence": first_dynamics_divergence(ref_trace[:branch_step], cand_trace[:branch_step]),
                    })
            elif h != PREFIX_H and not original_match and len(metadata_artifact_examples) < 5:
                # Show that decision metadata differs even when dynamics-only prefix matches.
                ex_step = 0 if branch_step > 0 else None
                metadata_artifact_examples.append({
                    "target_index": tid,
                    "case": int(cand.get("case", -1)),
                    "branch_step": branch_step,
                    "horizon": h,
                    "ref_decision_step0": None if ex_step is None else (ref_trace[ex_step].get("decision") or {}),
                    "candidate_decision_step0": None if ex_step is None else (cand_trace[ex_step].get("decision") or {}),
                    "ref_original_prefix_clean_sha256": ref_original_hash,
                    "candidate_original_prefix_clean_sha256": cand.get("prefix_clean_sha256"),
                    "shared_dynamics_prefix_sha256": ref_dyn_hash,
                })
            if not bool(cand.get("branch_reached")):
                branch_not_reached += 1
            dist = state_distance(state_tuple_from_summary(cand), ref_state) if bool(cand.get("branch_reached")) and bool(ref.get("branch_reached")) else float("inf")
            if h != PREFIX_H and bool(cand.get("branch_reached")) and dist > 1e-5:
                state_distance_gt_tol_non_h15 += 1
            if h != PREFIX_H and math.isfinite(dist):
                all_distances.append(dist)
            phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - cand["continuation_physical_constraint_cost_from_branch"]) if bool(cand.get("branch_reached")) and bool(ref.get("branch_reached")) else float("nan")
            total_gain = float(ref["continuation_total_cost_from_branch"] - cand["continuation_total_cost_from_branch"]) if bool(cand.get("branch_reached")) and bool(ref.get("branch_reached")) else float("nan")
            safe = no_regression(cand, ref)
            if h != PREFIX_H and not safe:
                safety_regression_non_h15 += 1
            if h != PREFIX_H and math.isfinite(total_gain) and total_gain <= -MATERIAL_GAIN_THRESHOLD:
                large_harms += 1
            material = bool(h != PREFIX_H and bool(cand.get("branch_reached")) and bool(ref.get("branch_reached")) and dynamics_match and dist <= 1e-5 and safe and (phys_gain >= MATERIAL_GAIN_THRESHOLD or total_gain >= MATERIAL_GAIN_THRESHOLD))
            if material:
                material_horizons.append(h)
                positive_horizon_counts[str(h)] = positive_horizon_counts.get(str(h), 0) + 1
            comp = {
                "horizon": h,
                "original_prefix_clean_matches_H15": original_match,
                "dynamics_prefix_matches_H15": dynamics_match,
                "state_distance_vs_H15_branch_state": dist,
                "gain_vs_H15_physical": phys_gain,
                "gain_vs_H15_total": total_gain,
                "success": bool(cand.get("success")),
                "constraint": bool(cand.get("constraint")),
                "initial_failed_steps": int(cand.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(cand.get("solver_failure_steps", 0)),
                "no_success_constraint_solver_regression_vs_H15": safe,
                "material_positive_with_dynamics_prefix": material,
                "path": cand.get("path"),
            }
            comparisons.append(comp)
            if h != PREFIX_H:
                pair_rows.append(dict(comp, target_index=tid, case=int(cand.get("case", -1)), branch_step=branch_step, selection_role=cand.get("selection_role")))
        if material_horizons:
            material_state_count += 1
            positive_cases.append(int(ref.get("case", -1)))
            positive_roles.append(str(ref.get("selection_role")))
        state_rows.append({
            "target_index": tid,
            "case": int(ref.get("case", -1)),
            "selection_role": ref.get("selection_role"),
            "branch_step": branch_step,
            "material_positive_state_with_dynamics_prefix": bool(material_horizons),
            "material_positive_horizons_with_dynamics_prefix": material_horizons,
            "reference_H15_continuation_physical": float(ref.get("continuation_physical_constraint_cost_from_branch", 0.0)),
            "reference_H15_continuation_total": float(ref.get("continuation_total_cost_from_branch", 0.0)),
            "comparisons": comparisons,
        })
    non_h15_pairs = len(pair_rows)
    corrected_blocking_artifacts = missing_reference + missing_horizon + dynamics_prefix_mismatch_non_h15 + state_distance_gt_tol_non_h15
    negative_controls = sum(1 for r in state_rows if not r["material_positive_state_with_dynamics_prefix"] and r["selection_role"] in ("nonmaterial_stress_control", "lower_stress_control"))
    corrected_label_gate = bool(material_state_count >= 2 and any(c == 5 for c in positive_cases) and negative_controls >= 2 and corrected_blocking_artifacts == 0)
    top_positive = sorted([r for r in pair_rows if r["material_positive_with_dynamics_prefix"]], key=lambda r: max(r["gain_vs_H15_physical"], r["gain_vs_H15_total"]), reverse=True)[:10]
    top_harms = sorted([r for r in pair_rows if math.isfinite(r["gain_vs_H15_total"])], key=lambda r: r["gain_vs_H15_total"])[:10]
    return {
        "input_stage2": {"raw": rel(STAGE2_RAW), "raw_sha256": sha256(STAGE2_RAW), "completed": rel(STAGE2_COMPLETED), "completed_sha256": sha256(STAGE2_COMPLETED), "summary": rel(STAGE2_SUMMARY), "summary_sha256": sha256(STAGE2_SUMMARY)},
        "trace_hashes": trace_hashes,
        "counts": {
            "target_states": len(state_rows),
            "non_H15_pairs": non_h15_pairs,
            "original_prefix_mismatch_non_H15": original_prefix_mismatch_non_h15,
            "dynamics_prefix_mismatch_non_H15": dynamics_prefix_mismatch_non_h15,
            "state_distance_gt_tol_non_H15": state_distance_gt_tol_non_h15,
            "safety_solver_regression_non_H15": safety_regression_non_h15,
            "missing_reference": missing_reference,
            "missing_horizon": missing_horizon,
            "branch_not_reached": branch_not_reached,
            "corrected_blocking_artifact_count": corrected_blocking_artifacts,
            "material_positive_state_count_with_dynamics_prefix": material_state_count,
            "negative_control_state_count_with_dynamics_prefix": negative_controls,
            "large_non_H15_harms": large_harms,
        },
        "state_distance_summary_non_H15": values_summary(all_distances),
        "positive_cases_with_dynamics_prefix": sorted(set(positive_cases)),
        "positive_roles_with_dynamics_prefix": sorted(set(positive_roles)),
        "positive_horizon_counts_with_dynamics_prefix": positive_horizon_counts,
        "corrected_training_refit_label_gate_pass_development_only": corrected_label_gate,
        "gate_rule_applied_diagnostic_only": ">=2 material non-H15 positive matched states, at least one in case 5, >=2 retained negative/control states, and no missing/dynamics-prefix/state-distance blocking artifacts; this is a diagnostic substitution only and does not change the frozen Stage2 result",
        "metadata_artifact_confirmed": bool(original_prefix_mismatch_non_h15 > 0 and dynamics_prefix_mismatch_non_h15 == 0),
        "metadata_artifact_examples": metadata_artifact_examples,
        "dynamics_divergence_examples": divergence_examples,
        "top_positive_branches_with_dynamics_prefix": top_positive,
        "top_harm_branches_by_total_gain": top_harms,
        "state_rows": state_rows,
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    c = a["counts"]
    lines = [
        "# Vehicle Stage2 prefix dynamics-hash diagnostic v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only analysis of existing Stage2 v0b traces. No simulations, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Key result",
        "",
        f"- Original Stage2 saved-prefix mismatches among non-H15 pairs: `{c['original_prefix_mismatch_non_H15']}` / `{c['non_H15_pairs']}`.",
        f"- Dynamics-only prefix mismatches among non-H15 pairs: `{c['dynamics_prefix_mismatch_non_H15']}` / `{c['non_H15_pairs']}`.",
        f"- Metadata-artifact confirmed: `{a['metadata_artifact_confirmed']}`.",
        f"- Corrected blocking artifacts after replacing prefix hash with dynamics-only hash: `{c['corrected_blocking_artifact_count']}`.",
        f"- Material positive states with diagnostic dynamics-prefix matching: `{c['material_positive_state_count_with_dynamics_prefix']}` across cases `{a['positive_cases_with_dynamics_prefix']}`; positive horizon counts `{a['positive_horizon_counts_with_dynamics_prefix']}`.",
        f"- Corrected development-only training/refit label gate: `{a['corrected_training_refit_label_gate_pass_development_only']}`.",
        "",
        "The frozen Stage2 result remains unchanged; this diagnostic only explains whether its prefix hash was over-strict. Even if the metadata artifact is confirmed, a selector/refit is still unjustified unless the corrected label-density gate passes.",
        "",
        "## Top corrected positives",
        "",
        "| case | target | branch step | H | physical gain | total gain | state distance | safe/no-regression |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in a["top_positive_branches_with_dynamics_prefix"]:
        lines.append("| %d | %d | %d | %d | %.6g | %.6g | %.6g | `%s` |" % (int(row["case"]), int(row["target_index"]), int(row["branch_step"]), int(row["horizon"]), float(row["gain_vs_H15_physical"]), float(row["gain_vs_H15_total"]), float(row["state_distance_vs_H15_branch_state"]), row["no_success_constraint_solver_regression_vs_H15"]))
    lines += ["", "## Top harms by total gain", "", "| case | target | branch step | H | physical gain | total gain | state distance | safe/no-regression |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in a["top_harm_branches_by_total_gain"]:
        lines.append("| %d | %d | %d | %d | %.6g | %.6g | %.6g | `%s` |" % (int(row["case"]), int(row["target_index"]), int(row["branch_step"]), int(row["horizon"]), float(row["gain_vs_H15_physical"]), float(row["gain_vs_H15_total"]), float(row["state_distance_vs_H15_branch_state"]), row["no_success_constraint_solver_regression_vs_H15"]))
    lines += [
        "",
        "## Decision",
        "",
    ]
    if a["corrected_training_refit_label_gate_pass_development_only"]:
        lines.append("Correcting the prefix artifact yields enough labels to freeze a compact IMPROVED selector/refit smoke before fresh confirmation.")
    else:
        lines.append("Correcting the prefix artifact does not yield enough label density for refit. Next highest-information step remains a tiny terminal/objective instrumentation smoke on the corrected positive state plus neutral/harm controls, after external backup covers this new diagnostic/source; otherwise move to terminal/reward/modeling or a separately versioned stronger stress design.")
    lines.append("")
    lines.append(f"Backup request: `{rel(REQUEST_BACKUP)}`.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists() and (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json"), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
            return 0
        raise DiagnosticError("existing completed marker did not pass")
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    raw_stage2 = verify_inputs()
    analysis = analyze(raw_stage2)
    raw = {
        "created_utc": created,
        "method": "vehicle_stage2_prefix_dynamics_hash_diagnostic_v0_analysis_only",
        "classification": "development_analysis_only_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "analysis": analysis,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "source_hashes": {rel(THIS_SCRIPT): sha256(THIS_SCRIPT), rel(STAGE2_RAW): sha256(STAGE2_RAW), rel(STAGE2_COMPLETED): sha256(STAGE2_COMPLETED)},
        "interpretation_limits": ["analysis-only over existing Stage2 development traces", "does not alter frozen Stage2 label gate", "does not measure terminal value; terminal/objective smoke still needed if backup permits"],
    }
    write_json(REQUEST_BACKUP, {
        "requested_utc": created,
        "reason": "backup Stage2 prefix dynamics-hash diagnostic source and analysis outputs before terminal/objective smoke or any further simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(THIS_SCRIPT), rel(OUT_DIR), rel(STATE_PATH), rel(REQUEST_BACKUP)],
    })
    raw["backup_request"] = rel(REQUEST_BACKUP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = analysis["counts"]
    STATE_PATH.write_text(
        f"# Vehicle Stage2 prefix dynamics-hash diagnostic v0 ({created})\n\n"
        f"Analysis-only existing Stage2 traces. Original non-H15 prefix mismatches={c['original_prefix_mismatch_non_H15']}/{c['non_H15_pairs']}; dynamics-only prefix mismatches={c['dynamics_prefix_mismatch_non_H15']}/{c['non_H15_pairs']}; metadata_artifact_confirmed={analysis['metadata_artifact_confirmed']}. Corrected positive states={c['material_positive_state_count_with_dynamics_prefix']}; corrected label gate={analysis['corrected_training_refit_label_gate_pass_development_only']}. No simulations/training/validation64/test. Backup requested before further simulations.\n",
        encoding="utf-8",
    )
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle Stage2 prefix dynamics-hash diagnostic v0

UTC: {created}. Analysis-only over existing Stage2 v0b traces. Original saved-prefix mismatches among non-H15 pairs: {c['original_prefix_mismatch_non_H15']}/{c['non_H15_pairs']}; dynamics-only mismatches: {c['dynamics_prefix_mismatch_non_H15']}/{c['non_H15_pairs']}; metadata-artifact confirmed={analysis['metadata_artifact_confirmed']}. Corrected material-positive states={c['material_positive_state_count_with_dynamics_prefix']} with horizons {analysis['positive_horizon_counts_with_dynamics_prefix']}; corrected refit gate={analysis['corrected_training_refit_label_gate_pass_development_only']}. No new rollouts/control steps/training; no validation64 or sealed-test access. Next: after backup, run tiny terminal/objective instrumentation smoke on case5 step18 plus neutral/harm controls, or pivot if instrumentation is blocked.
"""
    append_docs(block)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [THIS_SCRIPT, STATE_PATH, REQUEST_BACKUP, STAGE2_RAW, STAGE2_COMPLETED]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "headline": {
            "metadata_artifact_confirmed": analysis["metadata_artifact_confirmed"],
            "original_prefix_mismatch_non_H15": c["original_prefix_mismatch_non_H15"],
            "dynamics_prefix_mismatch_non_H15": c["dynamics_prefix_mismatch_non_H15"],
            "corrected_material_positive_state_count": c["material_positive_state_count_with_dynamics_prefix"],
            "corrected_refit_gate": analysis["corrected_training_refit_label_gate_pass_development_only"],
            "next_action": "backup_then_terminal_objective_instrumentation_smoke_or_pivot",
        },
        "backup_request": rel(REQUEST_BACKUP),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "metadata_artifact_confirmed": analysis["metadata_artifact_confirmed"],
        "original_prefix_mismatch_non_H15": c["original_prefix_mismatch_non_H15"],
        "dynamics_prefix_mismatch_non_H15": c["dynamics_prefix_mismatch_non_H15"],
        "corrected_material_positive_state_count": c["material_positive_state_count_with_dynamics_prefix"],
        "corrected_refit_gate": analysis["corrected_training_refit_label_gate_pass_development_only"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(REQUEST_BACKUP),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": __import__("traceback").format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve failure; repair analysis-only parser/hash logic only. Do not rerun simulations for this diagnostic.",
        })
        raise
