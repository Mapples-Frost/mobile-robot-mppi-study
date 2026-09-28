#!/usr/bin/env python3
"""Postdiagnostic for vehicle stress-v1 Stage1 fixed-H map.

Analysis-only over the just-completed development diagnostic.  No simulations,
no candidate resets, no validation64-bank access, no sealed-test access and no
training/refit.  The purpose is to extract discriminating evidence from the
Stage1 fixed-H map without changing the frozen protocol or acceptance rules.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T2155Z"
STAGE1_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z"
RAW_PATH = STAGE1_DIR / "raw.json"
SUMMARY_PATH = STAGE1_DIR / "summary.md"
COMPLETED_PATH = STAGE1_DIR / "completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1_stage1_postdiagnostic_and_next_{STAMP}.md"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1_STAGE1_POSTDIAGNOSTIC_{STAMP}.json"
MARKER = f"vehicle-stress-v1-stage1-postdiagnostic-{STAMP}"

HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def eligible(row: Mapping[str, Any], strict_solver: bool) -> bool:
    if not bool(row.get("success")) or bool(row.get("constraint")):
        return False
    if strict_solver and (int(row.get("initial_failed_steps", 0)) > 0 or int(row.get("solver_failure_steps", 0)) > 0):
        return False
    return True


def pct(gain: float, ref: float) -> float:
    return float(gain / ref) if ref and ref > 0 and math.isfinite(ref) else 0.0


def material_gain(phys_gain: float, ref_phys: float, total_gain: float, ref_total: float) -> bool:
    return bool(phys_gain >= 3.0 or pct(phys_gain, ref_phys) >= 0.05 or total_gain >= 3.0 or pct(total_gain, ref_total) >= 0.05)


def nondominated(points: Sequence[Mapping[str, Any]]) -> List[int]:
    keep: List[int] = []
    for p in points:
        h = int(p["horizon"])
        c = safe_float(p["physical_constraint_cost"])
        t = safe_float(p["decision_total_s"])
        dominated = False
        for q in points:
            if int(q["horizon"]) == h:
                continue
            cq = safe_float(q["physical_constraint_cost"])
            tq = safe_float(q["decision_total_s"])
            if cq <= c and tq <= t and (cq < c or tq < t):
                dominated = True
                break
        if not dominated:
            keep.append(h)
    return sorted(set(keep))


def group_counts(rows: Iterable[Mapping[str, Any]], key: str = "group") -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        g = str(r.get(key))
        out[g] = out.get(g, 0) + 1
    return dict(sorted(out.items()))


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (RAW_PATH, SUMMARY_PATH, COMPLETED_PATH):
        if not path.exists():
            raise RuntimeError(f"missing input {rel(path)}")
    raw = read_json(RAW_PATH)
    completed = read_json(COMPLETED_PATH)
    comp_hashes = completed.get("hashes") or {}
    input_hashes = {rel(RAW_PATH): sha256(RAW_PATH), rel(SUMMARY_PATH): sha256(SUMMARY_PATH), rel(COMPLETED_PATH): sha256(COMPLETED_PATH)}
    completed_hash_check = {
        rel(RAW_PATH): comp_hashes.get(rel(RAW_PATH)) in (None, input_hashes[rel(RAW_PATH)]),
        rel(SUMMARY_PATH): comp_hashes.get(rel(SUMMARY_PATH)) in (None, input_hashes[rel(SUMMARY_PATH)]),
    }

    mat = raw["stage1_materiality_analysis"]
    opp = raw["opportunity_analysis"]
    by_case = opp.get("by_case") or {}
    by_h = opp.get("by_horizon") or {}
    selected = raw["bank_selection"]["selected_metadata"]
    selected_by_case = {int(r["selected_case_index"]): r for r in selected}
    material_cases_gate = mat.get("material_cases") or []
    material_case_ids_gate = sorted({int(r["case"]) for r in material_cases_gate})

    selected_group_counts = group_counts(selected, key="selection_group")
    material_group_counts = group_counts(material_cases_gate, key="group")

    case_rows: List[Dict[str, Any]] = []
    all_material_against_h15_strict: List[Dict[str, Any]] = []
    all_material_against_h15_loose: List[Dict[str, Any]] = []
    h5_failures = 0
    for case_id_s, cinfo in sorted(by_case.items(), key=lambda kv: int(kv[0])):
        case_id = int(case_id_s)
        meta = selected_by_case[case_id]
        per = cinfo.get("per_h_summary") or {}
        h15 = per.get("15")
        if not h15:
            continue
        ref_phys = safe_float(h15.get("physical_constraint_cost"))
        ref_total = safe_float(h15.get("total_cost"))
        points_strict: List[Dict[str, Any]] = []
        points_loose: List[Dict[str, Any]] = []
        material_strict: List[Dict[str, Any]] = []
        material_loose: List[Dict[str, Any]] = []
        for hs, row in per.items():
            h = int(hs)
            r = dict(row)
            r["horizon"] = h
            r["decision_total_s"] = safe_float((row.get("decision_total_s") if "decision_total_s" in row else row.get("decision_total")))
            # by_case summaries use decision_total_s already; keep robust fallback.
            if not math.isfinite(r["decision_total_s"]):
                r["decision_total_s"] = safe_float(row.get("decision_total_s"), 0.0)
            if h == 5 and not bool(row.get("success")):
                h5_failures += 1
            if eligible(row, strict_solver=True):
                points_strict.append(r)
            if eligible(row, strict_solver=False):
                points_loose.append(r)
            if h == 15:
                continue
            phys_gain = ref_phys - safe_float(row.get("physical_constraint_cost"), 0.0)
            total_gain = ref_total - safe_float(row.get("total_cost"), 0.0)
            entry = {
                "case": case_id,
                "horizon": h,
                "group": meta.get("selection_group"),
                "source_candidate_index": int(meta.get("candidate_index", -1)),
                "theta_r": safe_float(meta.get("theta_r")),
                "traj_steps": int(meta.get("traj_steps", 0)),
                "min_reference_obstacle_clearance": safe_float(meta.get("min_reference_obstacle_clearance")),
                "physical_gain_vs_H15": float(phys_gain),
                "physical_gain_pct_vs_H15": pct(phys_gain, ref_phys),
                "total_gain_vs_H15": float(total_gain),
                "total_gain_pct_vs_H15": pct(total_gain, ref_total),
                "success": bool(row.get("success")),
                "constraint": bool(row.get("constraint")),
                "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
                "steps": int(row.get("steps", 0)),
                "decision_total_s": safe_float(row.get("decision_total_s"), 0.0),
            }
            if material_gain(phys_gain, ref_phys, total_gain, ref_total) and eligible(row, strict_solver=True) and not (h == 5 and not bool(row.get("success"))):
                material_strict.append(entry)
                all_material_against_h15_strict.append(entry)
            if material_gain(phys_gain, ref_phys, total_gain, ref_total) and eligible(row, strict_solver=False) and not (h == 5 and not bool(row.get("success"))):
                material_loose.append(entry)
                all_material_against_h15_loose.append(entry)
        best_strict_phys = min(points_strict, key=lambda r: (safe_float(r.get("physical_constraint_cost")), safe_float(r.get("decision_total_s")), int(r["horizon"]))) if points_strict else None
        best_strict_total = min(points_strict, key=lambda r: (safe_float(r.get("total_cost")), safe_float(r.get("decision_total_s")), int(r["horizon"]))) if points_strict else None
        fastest_strict = min(points_strict, key=lambda r: (safe_float(r.get("decision_total_s")), safe_float(r.get("physical_constraint_cost")), int(r["horizon"]))) if points_strict else None
        case_rows.append({
            "case": case_id,
            "source_candidate_index": int(meta.get("candidate_index", -1)),
            "group": meta.get("selection_group"),
            "theta_r": safe_float(meta.get("theta_r")),
            "traj_steps": int(meta.get("traj_steps", 0)),
            "min_reference_obstacle_clearance": safe_float(meta.get("min_reference_obstacle_clearance")),
            "best_strict_physical_horizon": None if best_strict_phys is None else int(best_strict_phys["horizon"]),
            "best_strict_total_horizon": None if best_strict_total is None else int(best_strict_total["horizon"]),
            "fastest_strict_horizon": None if fastest_strict is None else int(fastest_strict["horizon"]),
            "nondominated_strict_physical_vs_time": nondominated(points_strict),
            "material_strict_horizons_vs_H15": sorted({int(r["horizon"]) for r in material_strict}),
            "material_loose_horizons_vs_H15": sorted({int(r["horizon"]) for r in material_loose}),
        })

    material_horizon_counts_strict: Dict[str, int] = {}
    for r in all_material_against_h15_strict:
        material_horizon_counts_strict[str(r["horizon"])] = material_horizon_counts_strict.get(str(r["horizon"]), 0) + 1
    material_horizon_counts_strict = dict(sorted(material_horizon_counts_strict.items(), key=lambda kv: int(kv[0])))

    # Stratum-density view is development evidence only; it does not change the
    # predeclared Stage1 gate, which failed because material cases came from one
    # stratum.
    stratum_density: Dict[str, Dict[str, Any]] = {}
    by_group_cases: Dict[str, List[int]] = {}
    for row in case_rows:
        by_group_cases.setdefault(str(row["group"]), []).append(int(row["case"]))
    material_case_set = {int(r["case"]) for r in all_material_against_h15_strict}
    for group, cases in sorted(by_group_cases.items()):
        mat_cases = sorted([c for c in cases if c in material_case_set])
        stratum_density[group] = {
            "selected_cases": sorted(cases),
            "material_case_ids_strict_vs_H15": mat_cases,
            "material_fraction_strict_vs_H15": float(len(mat_cases) / len(cases)) if cases else 0.0,
        }

    horizon_rows = []
    for h in HORIZONS:
        row = by_h[str(h)]
        horizon_rows.append({
            "horizon": h,
            "success_count": int(row.get("success_count", 0)),
            "constraint_count": int(row.get("constraint_count", 0)),
            "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
            "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
            "physical_constraint_cost_sum": safe_float(row.get("physical_constraint_cost_sum")),
            "total_cost_sum": safe_float(row.get("total_cost_sum")),
            "decision_total_s": safe_float(row.get("decision_total_s")),
            "decision_mean_s_per_step": safe_float(row.get("decision_mean_s_per_step")),
        })
    safe_strict_h = [r["horizon"] for r in horizon_rows if r["success_count"] == 20 and r["constraint_count"] == 0 and r["initial_failed_steps"] == 0 and r["solver_failure_steps"] == 0]
    safe_loose_h = [r["horizon"] for r in horizon_rows if r["success_count"] == 20 and r["constraint_count"] == 0]
    best_strict_aggregate_physical = min([r for r in horizon_rows if r["horizon"] in safe_strict_h], key=lambda r: (r["physical_constraint_cost_sum"], r["decision_total_s"], r["horizon"])) if safe_strict_h else None
    fastest_strict_aggregate = min([r for r in horizon_rows if r["horizon"] in safe_strict_h], key=lambda r: (r["decision_total_s"], r["physical_constraint_cost_sum"], r["horizon"])) if safe_strict_h else None

    four_axis = {
        "SCENARIOS": {
            "verified": "Stress-v1 fixed-H map increased episode-level opportunity versus stress-v0, but positives are still concentrated: preregistered Stage2 gate failed because material cases are 4 distinct cases all in the high_heading_long_or_medium stratum.",
            "competing_hypotheses": "(a) useful opportunity exists only in a high-heading long-transient subfamily; (b) episode-level gains reflect one constant longer H per case, not within-episode switching; (c) large-H gains are terminal/solver artifacts despite no constraints/success regression.",
            "missing_evidence": "Matched continuation from identical prefixes for positive high-heading cases and same-stratum negatives, with terminal-mode/solver checks.",
            "discriminating_experiment": "After backup, freeze a v1b postdiagnostic matched-continuation protocol; do not call the failed v1 Stage1 gate passed.",
        },
        "REWARD": {
            "verified": "Physical gains are large for selected cases, while total cost includes synthetic h_penalty; measured decision time remains nonmonotone (H10 fastest aggregate, H15 best strict physical/total aggregate).",
            "competing_hypotheses": "Longer H improves control in a few cases but may not improve measured compute; synthetic total cost can obscure the physical/time Pareto frontier.",
            "missing_evidence": "Paired continuation timing from identical states and randomized final selector timing, if a selector is ever trained.",
            "discriminating_experiment": "Keep physical, total, success/safety and measured wall time separate; no scalar acceleration claim from Stage1.",
        },
        "TRAINING": {
            "verified": "No new training/refit was run. Current stress-v1 labels are not broad enough for a selector because positives are one stratum and episode-level only.",
            "competing_hypotheses": "A longer-H capable selector may help within the high-heading stratum only if matched-state labels are real and not terminal/solver artifacts.",
            "missing_evidence": "Within-episode matched labels with retained negative/control states; if dense, then freeze one IMPROVED selector/refit including longer H.",
            "discriminating_experiment": "Do not train now; matched-continuation v1b is the next gating diagnostic.",
        },
        "COMPARISONS": {
            "verified": "Strong fixed-H grid H5..H50 was run on the same stress-v1 bank. Strict safe aggregate best remains fixed H15; loose large-H physical sums improve but include initial-failure steps.",
            "competing_hypotheses": "A tuned fixed H15 or H10/H15 Pareto choice may absorb most benefit; any adaptive claim needs matched-state labels and fair fixed-H/timing baselines.",
            "missing_evidence": "Same-distribution matched continuation and later fresh validation if a selector is developed.",
            "discriminating_experiment": "Keep Stage1 as a fixed-H opportunity map; no adaptive comparison or reproduction claim.",
        },
    }

    next_plan = {
        "precondition": "verified external backup covering Stage1 rollout and this postdiagnostic before more simulations",
        "recommended_next_action": "freeze vehicle_stress_v1b_matched_continuation_postdiagnostic protocol, then dry-run before rollout",
        "why": "v1 Stage1 failed the preregistered diversity gate, but 4/10 high-heading long/medium cases show material strict H15-counterfactual positives; matched continuation is cheaper and more discriminating than training or another full grid.",
        "not_now": ["no selector/refit/training", "no sealed final test", "no retrospective change to v1 Stage1 gate", "no claim of adaptive acceleration"],
        "candidate_cases_development_only": {
            "positive_cases": material_case_ids_gate,
            "same_stratum_controls_suggested": [int(r["case"]) for r in case_rows if r["group"] == "high_heading_long_or_medium" and int(r["case"]) not in set(material_case_ids_gate)][:4],
            "lower_stress_controls_suggested": [int(r["case"]) for r in case_rows if r["group"] == "lower_stress_control"][:2],
        },
        "bounded_v1b_budget_suggestion": "at most 10 cases x 2 H15-prefix metadata-selected states x 7 branch horizons = 140 branch continuations, <=21000 control steps; exact protocol must be frozen before rollout",
    }

    raw_out = {
        "created_utc": created,
        "method": "vehicle_stress_v1_stage1_postdiagnostic_analysis_only_no_simulation",
        "classification": "development_static_postdiagnostic_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_hashes": input_hashes,
        "completed_hash_check": completed_hash_check,
        "stage1_budget_actual": raw.get("budget_actual"),
        "access_flags_from_stage1": {
            "historical_validation64_bank_opened": raw.get("historical_validation64_bank_opened"),
            "sealed_test_accessed": raw.get("sealed_test_accessed"),
        },
        "predeclared_stage1_gate": {
            "aggregate_guard_development_only": bool(mat.get("aggregate_guard_development_only")),
            "case_diversity_gate_development_only": bool(mat.get("case_diversity_gate_development_only")),
            "stage2_continuation_trigger_candidate_development_only": bool(mat.get("stage2_continuation_trigger_candidate_development_only")),
            "material_case_ids_from_runner": material_case_ids_gate,
            "material_groups_from_runner": mat.get("material_groups"),
            "oracle_total_gain_abs": mat.get("oracle_total_gain_abs"),
            "oracle_physical_gain_abs": mat.get("oracle_physical_gain_abs"),
        },
        "postdiagnostic_case_table": case_rows,
        "strict_material_vs_H15_rows": all_material_against_h15_strict,
        "loose_material_vs_H15_rows": all_material_against_h15_loose,
        "selected_group_counts": selected_group_counts,
        "material_group_counts_runner": material_group_counts,
        "stratum_density_strict_vs_H15": stratum_density,
        "material_horizon_counts_strict_vs_H15": material_horizon_counts_strict,
        "h5_failures": h5_failures,
        "horizon_aggregate_rows": horizon_rows,
        "strict_safe_horizons_recomputed": safe_strict_h,
        "loose_safe_horizons_recomputed": safe_loose_h,
        "best_strict_aggregate_physical": best_strict_aggregate_physical,
        "fastest_strict_aggregate": fastest_strict_aggregate,
        "four_axis_evidence_table": four_axis,
        "decision": "predeclared_stage1_gate_failed_but_high_heading_stratum_signal_warrants_matched_continuation_postdiagnostic_after_backup",
        "next_plan": next_plan,
    }
    write_json(OUT_DIR / "raw.json", raw_out)

    lines = [
        "# Vehicle stress-v1 Stage1 postdiagnostic",
        "",
        f"UTC: `{created}`. Analysis-only over Stage1 outputs: no simulations, no candidate resets, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Gate result preserved",
        "",
        f"- Predeclared aggregate guard: `{mat.get('aggregate_guard_development_only')}`.",
        f"- Predeclared case-diversity gate: `{mat.get('case_diversity_gate_development_only')}`.",
        f"- Predeclared Stage2 trigger candidate: `{mat.get('stage2_continuation_trigger_candidate_development_only')}` (failed; not retrospectively changed).",
        f"- Runner material cases: `{material_case_ids_gate}` in groups `{mat.get('material_groups')}`.",
        "",
        "## Diagnostic findings",
        "",
        f"- Selected group counts: `{selected_group_counts}`; runner material group counts: `{material_group_counts}`.",
        f"- Strict H15-counterfactual material horizons by count: `{material_horizon_counts_strict}`.",
        f"- Stratum density (strict H15-counterfactual): `{stratum_density}`.",
        f"- H5 failed episodes: `{h5_failures}` / 20; H5 is not a useful positive label source.",
        f"- Strict safe horizons recomputed: `{safe_strict_h}`; loose safe horizons recomputed: `{safe_loose_h}`.",
        f"- Best strict aggregate physical fixed H: `{None if best_strict_aggregate_physical is None else best_strict_aggregate_physical['horizon']}`; fastest strict aggregate H: `{None if fastest_strict_aggregate is None else fastest_strict_aggregate['horizon']}`.",
        "",
        "## Four-axis update",
        "",
    ]
    for axis, row in four_axis.items():
        lines += [
            f"### {axis}",
            f"- Verified: {row['verified']}",
            f"- Competing hypotheses: {row['competing_hypotheses']}",
            f"- Missing evidence: {row['missing_evidence']}",
            f"- Discriminating experiment: {row['discriminating_experiment']}",
            "",
        ]
    lines += [
        "## Decision and next action",
        "",
        f"Decision: `{raw_out['decision']}`.",
        f"Next after backup: {next_plan['recommended_next_action']}.",
        f"Rationale: {next_plan['why']}",
        f"Suggested bounded v1b budget: {next_plan['bounded_v1b_budget_suggestion']}.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle stress-v1 Stage1 postdiagnostic and next ({created})\n\n"
        f"No-simulation postdiagnostic completed. Predeclared Stage1 gate remains failed: aggregate={mat.get('aggregate_guard_development_only')}, "
        f"case_diversity={mat.get('case_diversity_gate_development_only')}, stage2_trigger={mat.get('stage2_continuation_trigger_candidate_development_only')}. "
        f"High-heading long/medium stratum has material cases {material_case_ids_gate}; no training/refit warranted yet. "
        f"Next after verified backup: freeze/dry-run a v1b matched-continuation postdiagnostic, not a selector training run. No validation64/test access.\n",
        encoding="utf-8",
    )

    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup Stage1 rollout plus postdiagnostic outputs before any matched-continuation protocol, simulation, or training decision",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(STAGE1_DIR), rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_REQ), rel(Path(__file__).resolve())],
    })
    raw_out["backup_request"] = rel(BACKUP_REQ)
    write_json(OUT_DIR / "raw.json", raw_out)

    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-v1 Stage1 postdiagnostic

UTC: {created}. Analysis-only postdiagnostic over stress-v1 Stage1 completed with no simulations/control steps/training, no validation64-bank access and no sealed-test access. The predeclared Stage1 gate remains failed (aggregate={mat.get('aggregate_guard_development_only')}, case-diversity={mat.get('case_diversity_gate_development_only')}, Stage2 trigger={mat.get('stage2_continuation_trigger_candidate_development_only')}); material cases are {material_case_ids_gate} and remain concentrated in the high_heading_long_or_medium stratum. No selector/refit/training is warranted yet. Next after verified backup: freeze and dry-run a v1b matched-continuation postdiagnostic to test whether the high-heading episode-level signal corresponds to real within-episode adaptive opportunity. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.
"""
    append_docs(block)

    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, BACKUP_REQ, Path(__file__).resolve(), RAW_PATH, SUMMARY_PATH, COMPLETED_PATH]
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
        "predeclared_stage1_gate_failed": not bool(mat.get("stage2_continuation_trigger_candidate_development_only")),
        "decision": raw_out["decision"],
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "predeclared_stage1_trigger": bool(mat.get("stage2_continuation_trigger_candidate_development_only")),
        "decision": raw_out["decision"],
        "next_after_backup": next_plan["recommended_next_action"],
        "backup_request": rel(BACKUP_REQ),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
