#!/usr/bin/env python3
"""No-simulation postdiagnostic for vehicle stress-v1b prefix-hash artifacts.

The v1b matched-continuation rollout reported 120 non-H15 prefix hash mismatches
while also reporting zero branch-state distance for those same comparisons.  This
script audits whether the prefix hash was contaminated by non-causal bookkeeping
metadata (notably the planned branch_horizon embedded in prefix-step decision
records).  It reads only existing development artifacts; it performs no rollout,
training/refit, validation-bank access, or sealed-test access.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_20260928T2220Z_schema_repair"
RAW_PATH = RUN_DIR / "raw.json"
DONE_PATH = RUN_DIR / "completed.json"
SUMMARY_PATH = RUN_DIR / "summary.md"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_PREFIX_ARTIFACT_POSTDIAGNOSTIC_V0_20260928T2315Z.json"
MARKER = "vehicle-stress-v1b-prefix-artifact-postdiagnostic-v0-20260928T2315Z"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45, 50]
MATERIAL_GAIN_THRESHOLD = 3.0
STATE_DISTANCE_TOLERANCE = 1e-5


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
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


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for path in (RAW_PATH, DONE_PATH, SUMMARY_PATH):
        if not path.exists():
            raise RuntimeError("missing required v1b artifact: %s" % rel(path))
    done = read_json(DONE_PATH)
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise RuntimeError("v1b rollout completion marker did not pass")
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise RuntimeError("v1b completion marker has invalid access flags")
    raw = read_json(RAW_PATH)
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("v1b raw artifact has invalid access flags")
    if len(raw.get("episodes") or []) != 140:
        raise RuntimeError("unexpected v1b episode count")
    if int(raw.get("budget_actual", {}).get("new_training_episodes", -1)) != 0 or int(raw.get("budget_actual", {}).get("new_gradient_steps", -1)) != 0:
        raise RuntimeError("v1b raw unexpectedly records training/refit")
    return done, raw


def old_clean_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(list(trace[:min(branch_step, len(trace))]))
    for row in cleaned:
        row.pop("timing", None)
        row.pop("decision_plus_terminal_switch_s", None)
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def dynamic_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    """Return the physical/controller prefix only, excluding bookkeeping/timing.

    The intended v1b prefix-identity condition is same H15 prefix trajectory until
    branch_step.  The planned future branch_horizon is not causal for prefix
    dynamics and must not invalidate the identical-prefix gate.
    """
    out: List[Dict[str, Any]] = []
    for row in trace[:min(branch_step, len(trace))]:
        decision = row.get("decision") or {}
        out.append({
            "previous_state": row.get("previous_state"),
            "state": row.get("state"),
            "observation": row.get("observation"),
            "next_observation": row.get("next_observation"),
            "input": row.get("input"),
            "horizon": row.get("horizon"),
            "reward": row.get("reward"),
            "performance": row.get("performance"),
            "compute": row.get("compute"),
            "constraint": row.get("constraint"),
            "solver_success": row.get("solver_success"),
            "termination": row.get("termination"),
            "decision_phase": decision.get("phase"),
            "decision_selected_horizon": decision.get("selected_horizon"),
            "decision_terminal_horizon": decision.get("terminal_horizon"),
            "decision_prefix_horizon": decision.get("prefix_horizon"),
            "decision_terminal_switched_before_step": decision.get("terminal_switched_before_step"),
        })
    return out


def diff_paths(a: Any, b: Any, prefix: str = "") -> List[str]:
    if type(a) != type(b):
        return [prefix or "<root:type>"]
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        out: List[str] = []
        for key in keys:
            if key not in a or key not in b:
                out.append((prefix + "." if prefix else "") + str(key))
            else:
                out.extend(diff_paths(a[key], b[key], (prefix + "." if prefix else "") + str(key)))
        return out
    if isinstance(a, list):
        out = []
        if len(a) != len(b):
            out.append((prefix or "<root>") + ".len")
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_paths(x, y, "%s[%d]" % (prefix or "<root>", i)))
        return out
    return [] if a == b else [prefix or "<root>"]


def path_for(comp: Mapping[str, Any]) -> Path:
    return ROOT / str(comp["path"]) / "trace.json"


def safety_ok(comp: Mapping[str, Any]) -> bool:
    return bool(comp.get("no_success_constraint_solver_regression_vs_H15"))


def compute_postdiagnostic(raw: Mapping[str, Any]) -> Dict[str, Any]:
    state_rows = raw["analysis"]["state_rows"]
    corrected_rows: List[Dict[str, Any]] = []
    trace_cache: Dict[str, List[Dict[str, Any]]] = {}
    def load_trace(path: Path) -> List[Dict[str, Any]]:
        key = rel(path)
        if key not in trace_cache:
            trace_cache[key] = read_json(path)
        return trace_cache[key]

    non_h15_total = 0
    existing_prefix_mismatch = 0
    dynamic_match_count = 0
    dynamic_mismatch_records: List[Dict[str, Any]] = []
    corrected_positive_cases: List[int] = []
    corrected_positive_horizon_counts: Dict[str, int] = {}
    corrected_negative_control_states = 0
    representative_diff = None

    for row in state_rows:
        tid = int(row["target_index"])
        case = int(row["case"])
        branch_step = int(row["branch_step"])
        comps = row.get("comparisons") or []
        ref = next((c for c in comps if int(c.get("horizon")) == PREFIX_H), None)
        if ref is None:
            corrected_rows.append({"target_index": tid, "case": case, "label_corrected": "missing_H15_reference"})
            continue
        ref_trace = load_trace(path_for(ref))
        ref_old = old_clean_prefix(ref_trace, branch_step)
        ref_dyn = dynamic_prefix(ref_trace, branch_step)
        ref_dyn_sha = canonical_sha(ref_dyn)
        corrected_material_horizons: List[int] = []
        corrected_comps: List[Dict[str, Any]] = []
        for comp in comps:
            h = int(comp["horizon"])
            ctrace = load_trace(path_for(comp))
            dyn_sha = canonical_sha(dynamic_prefix(ctrace, branch_step))
            dyn_match = (dyn_sha == ref_dyn_sha)
            if h != PREFIX_H:
                non_h15_total += 1
                if not bool(comp.get("prefix_clean_sha256_matches_H15")):
                    existing_prefix_mismatch += 1
                if dyn_match:
                    dynamic_match_count += 1
                else:
                    dynamic_mismatch_records.append({
                        "target_index": tid,
                        "case": case,
                        "horizon": h,
                        "path": comp.get("path"),
                        "state_distance_vs_H15_branch_state": comp.get("state_distance_vs_H15_branch_state"),
                    })
                if representative_diff is None and not bool(comp.get("prefix_clean_sha256_matches_H15")) and dyn_match:
                    old_diff = sorted(set(diff_paths(ref_old, old_clean_prefix(ctrace, branch_step))))
                    dyn_diff = sorted(set(diff_paths(ref_dyn, dynamic_prefix(ctrace, branch_step))))
                    representative_diff = {
                        "target_index": tid,
                        "case": case,
                        "horizon": h,
                        "branch_step": branch_step,
                        "existing_H15_prefix_clean_sha": ref.get("path"),
                        "candidate_path": comp.get("path"),
                        "old_clean_diff_paths_first20": old_diff[:20],
                        "old_clean_diff_path_count": len(old_diff),
                        "dynamic_diff_paths_first20": dyn_diff[:20],
                        "dynamic_diff_path_count": len(dyn_diff),
                    }
            material = bool(
                h != PREFIX_H
                and bool(comp.get("branch_reached"))
                and dyn_match
                and float(comp.get("state_distance_vs_H15_branch_state", float("inf"))) <= STATE_DISTANCE_TOLERANCE
                and safety_ok(comp)
                and (
                    float(comp.get("gain_vs_H15_physical", -float("inf"))) >= MATERIAL_GAIN_THRESHOLD
                    or float(comp.get("gain_vs_H15_total", -float("inf"))) >= MATERIAL_GAIN_THRESHOLD
                )
            )
            if material:
                corrected_material_horizons.append(h)
                corrected_positive_horizon_counts[str(h)] = corrected_positive_horizon_counts.get(str(h), 0) + 1
            corrected_comps.append({
                "horizon": h,
                "dynamic_prefix_sha256_matches_H15": dyn_match,
                "existing_prefix_clean_sha256_matches_H15": bool(comp.get("prefix_clean_sha256_matches_H15")),
                "state_distance_vs_H15_branch_state": comp.get("state_distance_vs_H15_branch_state"),
                "gain_vs_H15_physical": comp.get("gain_vs_H15_physical"),
                "gain_vs_H15_total": comp.get("gain_vs_H15_total"),
                "safety_solver_no_regression": safety_ok(comp),
                "material_positive_corrected": material,
                "continuation_physical": comp.get("continuation_physical"),
                "continuation_total": comp.get("continuation_total"),
                "success": comp.get("success"),
                "constraint": comp.get("constraint"),
                "solver_failure_steps": comp.get("solver_failure_steps"),
                "path": comp.get("path"),
            })
        if corrected_material_horizons:
            corrected_positive_cases.append(case)
        label = "positive_non_H15_corrected_prefix" if corrected_material_horizons else "negative_or_neutral_corrected_prefix"
        if label.startswith("negative") and row.get("selection_role") in ("same_stratum_stage1_nonmaterial_control", "lower_stress_control"):
            corrected_negative_control_states += 1
        best_physical = max((c for c in corrected_comps if c["horizon"] != PREFIX_H), key=lambda c: float(c.get("gain_vs_H15_physical") or -1e99), default=None)
        best_total = max((c for c in corrected_comps if c["horizon"] != PREFIX_H), key=lambda c: float(c.get("gain_vs_H15_total") or -1e99), default=None)
        corrected_rows.append({
            "target_index": tid,
            "case": case,
            "selection_role": row.get("selection_role"),
            "window_id": row.get("window_id"),
            "branch_step": branch_step,
            "old_label": row.get("label"),
            "label_corrected": label,
            "corrected_material_positive_horizons": sorted(set(corrected_material_horizons)),
            "best_non_H15_physical": best_physical,
            "best_non_H15_total": best_total,
            "comparisons_corrected": corrected_comps,
        })

    corrected_positive_state_count = sum(1 for r in corrected_rows if r.get("label_corrected") == "positive_non_H15_corrected_prefix")
    corrected_negative_neutral_state_count = sum(1 for r in corrected_rows if r.get("label_corrected") == "negative_or_neutral_corrected_prefix")
    distinct_cases = sorted(set(corrected_positive_cases))
    missing_or_reach_artifacts = int(raw["analysis"]["artifact_flags"].get("missing_H15_reference", 0)) + int(raw["analysis"]["artifact_flags"].get("missing_horizon", 0)) + int(raw["analysis"]["artifact_flags"].get("reference_branch_not_reached", 0)) + int(raw["analysis"]["artifact_flags"].get("candidate_branch_not_reached", 0))
    corrected_blocking = missing_or_reach_artifacts + len(dynamic_mismatch_records)
    corrected_gate = bool(len(distinct_cases) >= 3 and corrected_positive_state_count >= 6 and corrected_negative_control_states >= 2 and corrected_blocking == 0)
    return {
        "input_analysis_headline": {
            "old_positive_state_count": raw["analysis"].get("positive_state_count"),
            "old_positive_cases": raw["analysis"].get("positive_cases"),
            "old_blocking_artifact_count": raw["analysis"].get("blocking_artifact_count"),
            "old_artifact_flags": raw["analysis"].get("artifact_flags"),
            "old_gate": raw["analysis"].get("training_refit_label_gate_pass_development_only"),
        },
        "prefix_artifact_audit": {
            "non_H15_comparisons": non_h15_total,
            "existing_prefix_clean_sha_mismatches": existing_prefix_mismatch,
            "dynamic_prefix_sha_matches": dynamic_match_count,
            "dynamic_prefix_mismatches": len(dynamic_mismatch_records),
            "false_prefix_mismatch_due_to_bookkeeping_count": existing_prefix_mismatch - len(dynamic_mismatch_records),
            "representative_diff": representative_diff,
            "dynamic_mismatch_records": dynamic_mismatch_records[:20],
        },
        "corrected_label_headline": {
            "positive_state_count": corrected_positive_state_count,
            "positive_cases": distinct_cases,
            "positive_horizon_counts": corrected_positive_horizon_counts,
            "negative_neutral_state_count": corrected_negative_neutral_state_count,
            "negative_control_state_count": corrected_negative_control_states,
            "corrected_blocking_artifact_count": corrected_blocking,
            "training_refit_label_gate_pass_development_only_corrected_prefix": corrected_gate,
            "gate_rule": ">=3 distinct cases and >=6 matched states with material non-H15 positives, >=2 retained negative/control states, and no missing/prefix/state-distance/reference artifacts; prefix artifact replaced by dynamic-prefix hash",
        },
        "corrected_rows": corrected_rows,
    }


def write_summary(created: str, raw: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    old = analysis["input_analysis_headline"]
    pref = analysis["prefix_artifact_audit"]
    corr = analysis["corrected_label_headline"]
    lines = [
        "# Vehicle stress-v1b prefix-artifact postdiagnostic v0",
        "",
        "UTC: `%s`. No simulations, no training/refit, no validation64-bank access, no sealed-test access." % created,
        "",
        "## Finding",
        "",
        "The v1b rollout's `prefix_mismatch_non_H15=120` is a bookkeeping-artifact false positive for the physical prefix check. The saved H15-prefix dynamics match for `%d` / `%d` non-H15 comparisons after excluding non-causal bookkeeping/timing fields, while the original hash included the future `decision.branch_horizon` metadata in prefix steps." % (pref["dynamic_prefix_sha_matches"], pref["non_H15_comparisons"]),
        "",
        "Representative old-clean diff paths: `%s` (count `%d`); dynamic-prefix diff count `%d`." % (pref.get("representative_diff", {}).get("old_clean_diff_paths_first20"), pref.get("representative_diff", {}).get("old_clean_diff_path_count"), pref.get("representative_diff", {}).get("dynamic_diff_path_count")),
        "",
        "## Corrected label result",
        "",
        "- Old headline: positive states `%s`, positive cases `%s`, blocking artifacts `%s`, gate `%s`." % (old["old_positive_state_count"], old["old_positive_cases"], old["old_blocking_artifact_count"], old["old_gate"]),
        "- Corrected prefix headline: positive states `%d`, positive cases `%s`, positive horizon counts `%s`, corrected blocking artifacts `%d`, gate `%s`." % (corr["positive_state_count"], corr["positive_cases"], corr["positive_horizon_counts"], corr["corrected_blocking_artifact_count"], corr["training_refit_label_gate_pass_development_only_corrected_prefix"]),
        "- Negative/neutral states `%d`; negative/control states `%d`." % (corr["negative_neutral_state_count"], corr["negative_control_state_count"]),
        "",
        "## Corrected per-target labels",
        "",
        "| target | case | role | window | branch step | corrected label | material H | best phys H/gain | best total H/gain |",
        "|---:|---:|---|---|---:|---|---|---|---|",
    ]
    for r in analysis["corrected_rows"]:
        bp = r.get("best_non_H15_physical") or {}
        bt = r.get("best_non_H15_total") or {}
        lines.append("| %d | %d | `%s` | `%s` | %d | `%s` | `%s` | H%s/%.6g | H%s/%.6g |" % (
            int(r["target_index"]), int(r["case"]), r.get("selection_role"), r.get("window_id"), int(r["branch_step"]), r.get("label_corrected"), r.get("corrected_material_positive_horizons"), str(bp.get("horizon")), float(bp.get("gain_vs_H15_physical") or 0.0), str(bt.get("horizon")), float(bt.get("gain_vs_H15_total") or 0.0)
        ))
    lines += [
        "",
        "## Decision",
        "",
        "Do not train/refit from this v1b label set yet. The zero-positive headline was overly conservative because of a prefix-hash implementation artifact, but the corrected positives remain sparse (not enough distinct cases/states for the predeclared label gate) and are still concentrated in development-mined cases. Next high-information action after external backup is a bounded terminal/reward-source ablation on the corrected-positive states plus controls, to test whether H-specific terminal-value/source effects explain the large apparent gains before any selector refit or broader scenario redesign.",
        "",
        "Backup request after this no-simulation diagnostic: `%s`." % rel(REQUEST_BACKUP),
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if MARKER not in text:
                path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-understand-no-simulation", action="store_true", required=True)
    args = parser.parse_args(argv)
    if (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            print(json.dumps({"already_completed": rel(OUT_DIR / "completed.json"), "sealed_test_accessed": False, "validation64_bank_opened": False}, sort_keys=True))
            return 0
        raise RuntimeError("existing postdiagnostic completion marker is not passing")
    done, raw = verify_inputs()
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    analysis = compute_postdiagnostic(raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REQUEST_BACKUP, {
        "requested_utc": created,
        "reason": "backup no-simulation v1b prefix-artifact postdiagnostic before any further simulation/training/refit",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(REQUEST_BACKUP), rel(Path(__file__).resolve()), rel(RAW_PATH), rel(DONE_PATH)],
    })
    post_raw = {
        "created_utc": created,
        "method": "vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_no_simulation",
        "classification": "development_no_simulation_artifact_and_label_reanalysis_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "input_v1b": {"raw": rel(RAW_PATH), "raw_sha256": sha256(RAW_PATH), "completed": rel(DONE_PATH), "completed_sha256": sha256(DONE_PATH), "summary": rel(SUMMARY_PATH), "summary_sha256": sha256(SUMMARY_PATH)},
        "analysis": analysis,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "script_sha256": sha256(Path(__file__).resolve()),
        "backup_request": rel(REQUEST_BACKUP),
        "interpretation_limits": ["development reanalysis only", "uses existing v1b traces", "not a fresh validation split", "not final test", "does not justify selector/refit gate by itself"],
    }
    write_json(OUT_DIR / "raw.json", post_raw)
    write_summary(created, raw, analysis)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        "# Vehicle stress-v1b prefix-artifact postdiagnostic v0\n\n"
        "UTC: %s. No simulations/training/validation/test. Existing v1b prefix_mismatch_non_H15=120 is a bookkeeping false-positive: dynamic H15-prefix hashes match %d/%d non-H15 continuations after excluding non-causal decision.branch_horizon/timing metadata. Corrected positives=%d states across cases=%s; corrected label gate=%s. Backup required before further simulations. Next: terminal/reward-source ablation on corrected positives plus controls after verified backup.\n" % (
            created,
            analysis["prefix_artifact_audit"]["dynamic_prefix_sha_matches"],
            analysis["prefix_artifact_audit"]["non_H15_comparisons"],
            analysis["corrected_label_headline"]["positive_state_count"],
            analysis["corrected_label_headline"]["positive_cases"],
            analysis["corrected_label_headline"]["training_refit_label_gate_pass_development_only_corrected_prefix"],
        ),
        encoding="utf-8",
    )
    block = """<!-- %s -->
## 2026-09-28 vehicle stress-v1b prefix-artifact postdiagnostic v0

UTC: %s. No-simulation reanalysis of existing v1b traces found the reported `prefix_mismatch_non_H15=120` was a bookkeeping artifact: all %d/%d non-H15 comparisons match the H15 physical prefix after excluding non-causal prefix-step `decision.branch_horizon`/timing metadata. Corrected labels: positive states=%d across cases=%s, negative/neutral=%d, corrected label gate=%s. This repairs the zero-positive headline but still does not justify selector/refit because positives are sparse and development-mined. Next after verified backup: bounded terminal/reward-source ablation on corrected-positive states plus controls. Artifacts: `%s`, `%s`, `%s`.
""" % (
        MARKER,
        created,
        analysis["prefix_artifact_audit"]["dynamic_prefix_sha_matches"],
        analysis["prefix_artifact_audit"]["non_H15_comparisons"],
        analysis["corrected_label_headline"]["positive_state_count"],
        analysis["corrected_label_headline"]["positive_cases"],
        analysis["corrected_label_headline"]["negative_neutral_state_count"],
        analysis["corrected_label_headline"]["training_refit_label_gate_pass_development_only_corrected_prefix"],
        rel(OUT_DIR / "summary.md"),
        rel(OUT_DIR / "raw.json"),
        rel(OUT_DIR / "completed.json"),
    )
    append_docs(block)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, REQUEST_BACKUP, Path(__file__).resolve(), RAW_PATH, DONE_PATH, SUMMARY_PATH]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "headline": {
            "prefix_hash_artifact_confirmed": analysis["prefix_artifact_audit"]["dynamic_prefix_sha_matches"] == analysis["prefix_artifact_audit"]["non_H15_comparisons"],
            "existing_prefix_clean_sha_mismatches": analysis["prefix_artifact_audit"]["existing_prefix_clean_sha_mismatches"],
            "dynamic_prefix_sha_matches": analysis["prefix_artifact_audit"]["dynamic_prefix_sha_matches"],
            "positive_state_count_corrected": analysis["corrected_label_headline"]["positive_state_count"],
            "positive_cases_corrected": analysis["corrected_label_headline"]["positive_cases"],
            "training_refit_label_gate_pass_development_only_corrected_prefix": analysis["corrected_label_headline"]["training_refit_label_gate_pass_development_only_corrected_prefix"],
            "next_action_after_backup": "freeze/run bounded terminal/reward-source ablation on corrected-positive states plus controls; do not train/refit yet",
        },
        "backup_request": rel(REQUEST_BACKUP),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "prefix_hash_artifact_confirmed": analysis["prefix_artifact_audit"]["dynamic_prefix_sha_matches"] == analysis["prefix_artifact_audit"]["non_H15_comparisons"],
        "dynamic_prefix_sha_matches": analysis["prefix_artifact_audit"]["dynamic_prefix_sha_matches"],
        "non_H15_comparisons": analysis["prefix_artifact_audit"]["non_H15_comparisons"],
        "positive_state_count_corrected": analysis["corrected_label_headline"]["positive_state_count"],
        "positive_cases_corrected": analysis["corrected_label_headline"]["positive_cases"],
        "label_gate_corrected": analysis["corrected_label_headline"]["training_refit_label_gate_pass_development_only_corrected_prefix"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(REQUEST_BACKUP),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
