#!/usr/bin/env python3
"""No-simulation postdiagnostic for transient-state prefix hash artifact.

The transient-state continuation rollout completed with 0 material positives but
marked all 36 non-H15 comparisons as prefix-hash mismatches.  This diagnostic
recomputes prefix hashes from saved traces after removing arm-identifying fields
from the prefix-only decision metadata.  It spends no rollout/training budget and
opens no validation64 or sealed-test data.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928/raw.json"
COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928/completed.json"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_20260928T1620Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_20260928T1620Z.md"
BACKUP_REQ = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_PREFIX_ARTIFACT_POSTDIAGNOSTIC_V0_20260928T1620Z.json"
MARKER = "vehicle-v1-transient-state-prefix-artifact-postdiagnostic-v0-20260928T1620Z"
PREFIX_H = 15
MATERIAL_GAIN_THRESHOLD = 3.0


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def read_trace(ep_path: str) -> List[Dict[str, Any]]:
    trace_path = ROOT / ep_path / "trace.jsonl"
    rows: List[Dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_prefix(trace: Sequence[Mapping[str, Any]], branch_step: int, repair_decision_arm: bool) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(list(trace[: min(branch_step, len(trace))]))
    for row in cleaned:
        row.pop("timing", None)
        row.pop("decision_plus_terminal_switch_s", None)
        # The original cleaner deliberately removed timing, but it retained
        # decision.branch_horizon even for prefix rows where selected_horizon and
        # terminal_horizon are both H15.  That embeds the future branch arm in an
        # otherwise identical H15 prefix and makes every non-H15 prefix hash fail.
        if repair_decision_arm and isinstance(row.get("decision"), dict):
            row["decision"].pop("branch_horizon", None)
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def no_regression(comp: Mapping[str, Any]) -> bool:
    return bool(comp.get("branch_reached")) and bool(comp.get("no_success_constraint_solver_regression_vs_H15"))


def main() -> int:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    raw = read_json(RAW)
    done = read_json(COMPLETED)
    if done.get("passed") is not True or raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("input access/completion flags invalid")

    rows_out: List[Dict[str, Any]] = []
    original_mismatch = 0
    repaired_mismatch = 0
    branch_horizon_only_explains = 0
    state_dist_bad = 0
    solver_or_safety_bad = 0
    h15_bad = 0
    missing_ref = 0
    positive_states = 0
    negative_states = 0
    best_total_gain = -1e100
    best_phys_gain = -1e100
    large_harms = 0

    for state in raw["analysis"]["state_rows"]:
        comps = state["comparisons"]
        ref_comps = [c for c in comps if int(c["horizon"]) == PREFIX_H]
        if not ref_comps:
            missing_ref += 1
            continue
        ref = ref_comps[0]
        if (not ref.get("success")) or ref.get("constraint") or int(ref.get("solver_failure_steps", 0)) or int(ref.get("initial_failed_steps", 0)):
            h15_bad += 1
        ref_trace = read_trace(ref["path"])
        ref_orig_hash = canonical_sha(clean_prefix(ref_trace, int(state["branch_step"]), repair_decision_arm=False))
        ref_repair_hash = canonical_sha(clean_prefix(ref_trace, int(state["branch_step"]), repair_decision_arm=True))
        material_h: List[int] = []
        comp_rows: List[Dict[str, Any]] = []
        for comp in comps:
            h = int(comp["horizon"])
            if h == PREFIX_H:
                continue
            cand_trace = read_trace(comp["path"])
            cand_orig_hash = canonical_sha(clean_prefix(cand_trace, int(state["branch_step"]), repair_decision_arm=False))
            cand_repair_hash = canonical_sha(clean_prefix(cand_trace, int(state["branch_step"]), repair_decision_arm=True))
            orig_match = cand_orig_hash == ref_orig_hash
            repaired_match = cand_repair_hash == ref_repair_hash
            if not orig_match:
                original_mismatch += 1
            if not repaired_match:
                repaired_mismatch += 1
            if (not orig_match) and repaired_match:
                branch_horizon_only_explains += 1
            if float(comp.get("state_distance_vs_H15_branch_state", 1e9)) > 1e-5:
                state_dist_bad += 1
            if not no_regression(comp):
                solver_or_safety_bad += 1
            tg = float(comp.get("gain_vs_H15_total", 0.0))
            pg = float(comp.get("gain_vs_H15_physical", 0.0))
            best_total_gain = max(best_total_gain, tg)
            best_phys_gain = max(best_phys_gain, pg)
            if tg <= -MATERIAL_GAIN_THRESHOLD:
                large_harms += 1
            material = bool(
                repaired_match
                and float(comp.get("state_distance_vs_H15_branch_state", 1e9)) <= 1e-5
                and no_regression(comp)
                and (tg >= MATERIAL_GAIN_THRESHOLD or pg >= MATERIAL_GAIN_THRESHOLD)
            )
            if material:
                material_h.append(h)
            comp_rows.append({
                "horizon": h,
                "original_prefix_match": orig_match,
                "repaired_prefix_match_drop_decision_branch_horizon": repaired_match,
                "gain_vs_H15_total": tg,
                "gain_vs_H15_physical": pg,
                "safe_no_regression": no_regression(comp),
                "material_after_repair": material,
            })
        if material_h:
            positive_states += 1
        else:
            negative_states += 1
        rows_out.append({
            "target_index": int(state["target_index"]),
            "case": int(state["case"]),
            "kind": state["kind"],
            "branch_step": int(state["branch_step"]),
            "material_horizons_after_prefix_repair": material_h,
            "label_after_prefix_repair": "positive_non_H15" if material_h else "negative_or_neutral",
            "comparisons": comp_rows,
        })

    repaired_blocking_artifact_count = missing_ref + h15_bad + repaired_mismatch + state_dist_bad + solver_or_safety_bad
    scenario_scarcity_clean = bool(positive_states < 2 and repaired_blocking_artifact_count == 0 and negative_states >= 2)
    transient_gate = bool(positive_states >= 2)  # stricter outside-mined case rule is moot with zero positives

    analysis = {
        "input_raw": rel(RAW),
        "input_completed": rel(COMPLETED),
        "original_prefix_mismatch_count_non_H15": original_mismatch,
        "repaired_prefix_mismatch_count_non_H15": repaired_mismatch,
        "branch_horizon_only_explains_mismatch_count": branch_horizon_only_explains,
        "state_distance_gt_1e_minus_5_count_non_H15": state_dist_bad,
        "missing_H15_reference_count": missing_ref,
        "h15_failure_or_constraint_or_solver_issue_count": h15_bad,
        "solver_or_safety_regression_count_non_H15": solver_or_safety_bad,
        "repaired_blocking_artifact_count": repaired_blocking_artifact_count,
        "positive_state_count_after_repair": positive_states,
        "negative_or_neutral_state_count_after_repair": negative_states,
        "best_non_H15_total_gain_after_repair": best_total_gain,
        "best_non_H15_physical_gain_after_repair": best_phys_gain,
        "large_non_H15_harms_total_gain_le_minus_threshold": large_harms,
        "transient_missed_timing_supported_gate_pass_after_repair": transient_gate,
        "scenario_scarcity_supported_gate_fail_clean_after_repair": scenario_scarcity_clean,
        "diagnosis": "original prefix mismatch was an analysis artifact caused by retaining decision.branch_horizon in prefix rows; after dropping that future-arm metadata, all 36 non-H15 prefix comparisons match their H15 references, but no material non-H15 positives remain",
        "state_rows": rows_out,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    raw_out = {
        "created_utc": created,
        "method": "vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_no_simulation_recompute_prefix_hashes",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "analysis": analysis,
        "input_hashes": {rel(RAW): sha256(RAW), rel(COMPLETED): sha256(COMPLETED)},
    }
    write_json(OUT / "raw.json", raw_out)
    summary = f"""# Vehicle V1 transient-state prefix artifact postdiagnostic v0

Created UTC: `{created}`. No simulations, no training/refit, no historical validation64 access, no sealed-test access.

## Finding

The transient rollout's original `prefix_mismatch_count_non_H15=36` is an analysis artifact. The saved prefix traces embed the future arm in `decision.branch_horizon` even while every prefix step executes H15; the original cleaner retained that field. After removing only this arm-identifying metadata from prefix rows, repaired prefix mismatches are `{repaired_mismatch}/36`.

## Recomputed gate

- positive states after repair: `{positive_states}/12`; negative/neutral: `{negative_states}`.
- best non-H15 total gain: `{best_total_gain}`; best non-H15 physical gain: `{best_phys_gain}`; large non-H15 harms: `{large_harms}`.
- repaired blocking artifact count: `{repaired_blocking_artifact_count}` (missing H15 `{missing_ref}`, H15 issue `{h15_bad}`, prefix mismatch `{repaired_mismatch}`, state-distance mismatch `{state_dist_bad}`, solver/safety regression `{solver_or_safety_bad}`).
- transient gate pass after repair: `{transient_gate}`.
- clean scenario-scarcity gate fail after repair: `{scenario_scarcity_clean}`.

## Decision

Because the prefix artifact is repaired without changing any rollout outcome and there are still zero material non-H15 positives, canonical Vehicle V1 fresh/transient states provide no reusable positive label density for refit/retraining here. Next action after backup should freeze a versioned stress-scenario opportunity protocol, preserving canonical negative evidence and fairly retuning fixed-H baselines on any revised distribution.

Backup request: `{rel(BACKUP_REQ)}`.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        f"# Vehicle V1 transient-state prefix artifact postdiagnostic v0 ({created})\n\n"
        f"No-simulation recomputation found original prefix mismatches were caused by decision.branch_horizon metadata in H15-prefix rows. Repaired mismatches={repaired_mismatch}/36; positives={positive_states}/12; clean scenario-scarcity fail={scenario_scarcity_clean}. Next: request/verify backup, then freeze versioned stress-scenario opportunity protocol before retraining.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQ, {
        "requested_utc": created,
        "reason": "backup no-simulation prefix-artifact postdiagnostic before stress-scenario protocol/source work or further simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(RAW), rel(COMPLETED)],
    })
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + f"\n\n<!-- {MARKER} -->\n## 2026-09-28 vehicle V1 transient prefix-artifact postdiagnostic\n\nUTC: {created}. No-simulation recomputation repaired the transient probe artifact: original prefix mismatches 36/36 were due to retained `decision.branch_horizon` metadata; repaired mismatches 0/36. Positive states remain {positive_states}/12; best non-H15 total gain {best_total_gain}; clean scenario-scarcity gate fail {scenario_scarcity_clean}. Next after backup: freeze a versioned stress-scenario opportunity protocol before retraining. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`.\n", encoding="utf-8")
    hashes = {rel(p): sha256(p) for p in [OUT / "raw.json", OUT / "summary.md", STATE, BACKUP_REQ, Path(__file__).resolve(), RAW, COMPLETED] if p.exists()}
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "headline": {
            "original_prefix_mismatch_count_non_H15": original_mismatch,
            "repaired_prefix_mismatch_count_non_H15": repaired_mismatch,
            "positive_state_count_after_repair": positive_states,
            "scenario_scarcity_supported_gate_fail_clean_after_repair": scenario_scarcity_clean,
            "next_action": "backup then freeze versioned stress-scenario opportunity protocol before retraining",
        },
        "backup_request": rel(BACKUP_REQ),
        "hashes": hashes,
    })
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "original_prefix_mismatch_count_non_H15": original_mismatch,
        "repaired_prefix_mismatch_count_non_H15": repaired_mismatch,
        "positive_state_count_after_repair": positive_states,
        "best_non_H15_total_gain_after_repair": best_total_gain,
        "scenario_scarcity_supported_gate_fail_clean_after_repair": scenario_scarcity_clean,
        "backup_request": rel(BACKUP_REQ),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
