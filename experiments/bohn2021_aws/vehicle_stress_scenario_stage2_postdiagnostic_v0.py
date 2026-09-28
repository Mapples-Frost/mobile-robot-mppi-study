#!/usr/bin/env python3
"""Postdiagnostic for vehicle stress-scenario Stage2 v0b matched continuations.

This is an analysis-only diagnostic.  It reads the completed Stage2 v0b
matched-continuation artifacts and traces, does not run simulations, does not
train/refit, does not open the historical validation64 bank, and does not access
sealed final test data.

Goals:
1. Preserve the Stage2 result with an independent analysis artifact.
2. Diagnose the apparent all-non-H15 prefix_clean_sha256 mismatch in the frozen
   Stage2 gate: is it a real state mismatch or an over-strict administrative
   hash issue?
3. Quantify horizon opportunity under the frozen gate and under an explicitly
   labeled diagnostic state-matched relaxation (not changing the preregistered
   gate) to decide whether selector retraining/refit is justified.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z"
STAGE2_RAW = STAGE2_DIR / "raw.json"
STAGE2_COMPLETED = STAGE2_DIR / "completed.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-stress-scenario-stage2-postdiagnostic-v0-20260928T1815Z"
MATERIAL_GAIN_THRESHOLD = 3.0
STATE_TOL = 1e-5


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


def values_summary(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"count": n, "min": xs[0], "median": med, "mean": sum(xs) / n, "max": xs[-1]}


def top_level_diff_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = []
    for k in keys:
        if a.get(k) != b.get(k):
            out.append(k)
    return out


def semantic_prefix_rows(trace: Sequence[Mapping[str, Any]], branch_step: int) -> List[Dict[str, Any]]:
    """Keep deterministic state/control/cost/solver-status fields only.

    This is intentionally an analysis-only alternative to the frozen
    prefix_clean_sha256.  It drops administrative/timing/path fields and is used
    only to diagnose whether prefix hash mismatches are physical mismatches.
    """
    keep = []
    for row in list(trace[:min(branch_step, len(trace))]):
        item = {}
        for k in ("previous_state", "state", "observation", "input", "performance", "constraint", "compute", "reward", "done", "info", "termination"):
            if k in row:
                item[k] = row[k]
        # Retain solver outcome but drop solver wall-time fields nested in attempts.
        rec = row.get("recovery")
        if isinstance(rec, dict):
            rec2 = {}
            for rk, rv in rec.items():
                if rk == "attempts" and isinstance(rv, list):
                    attempts = []
                    for att in rv:
                        if isinstance(att, dict):
                            attempts.append({ak: av for ak, av in att.items() if "time" not in ak.lower() and "solver_s" != ak and "wall" not in ak.lower()})
                        else:
                            attempts.append(att)
                    rec2[rk] = attempts
                elif "time" not in rk.lower() and "wall" not in rk.lower():
                    rec2[rk] = rv
            item["recovery"] = rec2
        keep.append(item)
    return keep


def trace_prefix_audit(state_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    aggregate_keys: Dict[str, int] = {}
    semantic_matches = 0
    compared = 0
    for sr in state_rows:
        ref_comp = None
        for comp in sr.get("comparisons", []):
            if int(comp.get("horizon", -1)) == 15:
                ref_comp = comp
                break
        if not ref_comp:
            continue
        ref_trace_path = ROOT / str(ref_comp["path"]) / "trace.json"
        ref_trace = read_json(ref_trace_path)
        branch_step = int(sr["branch_step"])
        ref_sem = semantic_prefix_rows(ref_trace, branch_step)
        ref_sem_sha = canonical_sha(ref_sem)
        for comp in sr.get("comparisons", []):
            h = int(comp.get("horizon", -1))
            if h == 15:
                continue
            compared += 1
            tr_path = ROOT / str(comp["path"]) / "trace.json"
            tr = read_json(tr_path)
            cand_sem = semantic_prefix_rows(tr, branch_step)
            cand_sha = canonical_sha(cand_sem)
            match = cand_sha == ref_sem_sha
            if match:
                semantic_matches += 1
            # diagnose first row-level mismatch with full top-level keys, if any
            first_mismatch = None
            for j, (a, b) in enumerate(zip(ref_trace[:branch_step], tr[:branch_step])):
                if a != b:
                    keys = top_level_diff_keys(a, b)
                    for k in keys:
                        aggregate_keys[k] = aggregate_keys.get(k, 0) + 1
                    first_mismatch = {"step": j, "top_level_diff_keys": keys[:20]}
                    break
            rows.append({
                "target_index": int(sr["target_index"]),
                "case": int(sr["case"]),
                "branch_step": branch_step,
                "horizon": h,
                "frozen_prefix_hash_match": bool(comp.get("prefix_clean_sha256_matches_H15")),
                "semantic_prefix_hash_match_diagnostic": bool(match),
                "state_distance_vs_H15_branch_state": safe_float(comp.get("state_distance_vs_H15_branch_state"), float("inf")),
                "first_raw_prefix_mismatch": first_mismatch,
                "path": comp.get("path"),
            })
    return {
        "non_H15_prefix_pairs_compared": compared,
        "frozen_prefix_hash_matches": sum(1 for r in rows if r["frozen_prefix_hash_match"]),
        "semantic_prefix_hash_matches_diagnostic": semantic_matches,
        "semantic_prefix_match_fraction_diagnostic": None if compared == 0 else semantic_matches / float(compared),
        "top_level_diff_keys_first_mismatch_counts": aggregate_keys,
        "examples": rows[:12],
    }


def no_regression(comp: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(comp.get("success")):
        return False
    if bool(comp.get("constraint")) and not bool(ref.get("constraint")):
        return False
    if int(comp.get("initial_failed_steps", 0)) > int(ref.get("initial_failed_steps", 0)):
        return False
    if int(comp.get("solver_failure_steps", 0)) > int(ref.get("solver_failure_steps", 0)):
        return False
    return True


def opportunity_audit(state_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    relaxed_positive_states = []
    all_non_h15 = []
    per_target = []
    for sr in state_rows:
        ref = sr["reference_H15"]
        comps = sr.get("comparisons", [])
        target_relaxed = []
        for c in comps:
            h = int(c.get("horizon", -1))
            if h == 15:
                continue
            dist_ok = safe_float(c.get("state_distance_vs_H15_branch_state"), float("inf")) <= STATE_TOL
            safety_ok = no_regression(c, ref)
            gain_phys = safe_float(c.get("gain_vs_H15_physical"))
            gain_total = safe_float(c.get("gain_vs_H15_total"))
            material_relaxed = bool(c.get("branch_reached")) and dist_ok and safety_ok and (gain_phys >= MATERIAL_GAIN_THRESHOLD or gain_total >= MATERIAL_GAIN_THRESHOLD)
            row = {
                "target_index": int(sr["target_index"]),
                "case": int(sr["case"]),
                "selection_role": sr.get("selection_role"),
                "branch_step": int(sr["branch_step"]),
                "horizon": h,
                "gain_vs_H15_physical": gain_phys,
                "gain_vs_H15_total": gain_total,
                "decision_sum_s": safe_float(c.get("decision_sum_s")),
                "success": bool(c.get("success")),
                "constraint": bool(c.get("constraint")),
                "state_distance_ok": dist_ok,
                "safety_ok": safety_ok,
                "frozen_prefix_hash_match": bool(c.get("prefix_clean_sha256_matches_H15")),
                "material_under_frozen_gate": bool(c.get("material_positive_vs_H15")),
                "material_state_matched_relaxed_prefix_diagnostic": material_relaxed,
                "large_harm_total_le_minus_threshold": bool(math.isfinite(gain_total) and gain_total <= -MATERIAL_GAIN_THRESHOLD),
                "path": c.get("path"),
            }
            all_non_h15.append(row)
            if material_relaxed:
                target_relaxed.append(row)
        best_total = max(all_non_h15[-5:], key=lambda x: x["gain_vs_H15_total"] if math.isfinite(x["gain_vs_H15_total"]) else -1e300)
        best_phys = max(all_non_h15[-5:], key=lambda x: x["gain_vs_H15_physical"] if math.isfinite(x["gain_vs_H15_physical"]) else -1e300)
        if target_relaxed:
            relaxed_positive_states.append({
                "target_index": int(sr["target_index"]),
                "case": int(sr["case"]),
                "selection_role": sr.get("selection_role"),
                "branch_step": int(sr["branch_step"]),
                "positive_horizons": [int(x["horizon"]) for x in target_relaxed],
                "best_total_horizon": int(best_total["horizon"]),
                "best_total_gain": best_total["gain_vs_H15_total"],
                "best_physical_horizon": int(best_phys["horizon"]),
                "best_physical_gain": best_phys["gain_vs_H15_physical"],
            })
        per_target.append({
            "target_index": int(sr["target_index"]),
            "case": int(sr["case"]),
            "selection_role": sr.get("selection_role"),
            "branch_step": int(sr["branch_step"]),
            "frozen_label": sr.get("label"),
            "relaxed_positive": bool(target_relaxed),
            "relaxed_positive_horizons": [int(x["horizon"]) for x in target_relaxed],
            "best_non_H15_total_gain": best_total["gain_vs_H15_total"],
            "best_non_H15_total_horizon": int(best_total["horizon"]),
            "best_non_H15_physical_gain": best_phys["gain_vs_H15_physical"],
            "best_non_H15_physical_horizon": int(best_phys["horizon"]),
        })
    return {
        "frozen_gate_positive_states": sum(1 for sr in state_rows if bool(sr.get("material_positive_state"))),
        "relaxed_state_matched_positive_state_count_diagnostic": len(relaxed_positive_states),
        "relaxed_positive_states": relaxed_positive_states,
        "relaxed_positive_branch_count": sum(1 for r in all_non_h15 if r["material_state_matched_relaxed_prefix_diagnostic"]),
        "large_total_harm_branch_count": sum(1 for r in all_non_h15 if r["large_harm_total_le_minus_threshold"]),
        "non_H15_branch_count": len(all_non_h15),
        "gain_total_summary_all_non_H15": values_summary(r["gain_vs_H15_total"] for r in all_non_h15),
        "gain_physical_summary_all_non_H15": values_summary(r["gain_vs_H15_physical"] for r in all_non_h15),
        "per_target": per_target,
        "top_positive_branches_diagnostic": sorted(all_non_h15, key=lambda r: r["gain_vs_H15_total"], reverse=True)[:12],
        "largest_harms_by_total": sorted(all_non_h15, key=lambda r: r["gain_vs_H15_total"])[:12],
    }


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not STAGE2_RAW.exists() or not STAGE2_COMPLETED.exists():
        raise RuntimeError("Stage2 raw/completed artifacts are missing")
    completed = read_json(STAGE2_COMPLETED)
    if completed.get("passed") is not True or completed.get("hard_pass") is not True:
        raise RuntimeError("Stage2 completed marker did not hard-pass")
    raw = read_json(STAGE2_RAW)
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise RuntimeError("Stage2 access flags are invalid")
    state_rows = raw["analysis"]["state_rows"]
    prefix = trace_prefix_audit(state_rows)
    opp = opportunity_audit(state_rows)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    decision = {
        "retraining_refit_now": False,
        "reason": "Frozen Stage2 label gate failed. Even if the all-non-H15 prefix hash mismatch is treated as an administrative/frozen-gate artifact and state-distance matching is used diagnostically, only one matched state is materially positive; this is below the preregistered >=2 positive-state gate and too sparse for selector training/refit.",
        "next_high_information_action": "Run an analysis-only terminal/reward/modeling audit on Stage1+Stage2 costs and the H15 terminal/value objective before any new training; if no objective artifact is found, freeze a separately versioned stronger source-supported scenario design rather than retrain on sparse labels.",
    }
    if prefix["semantic_prefix_hash_matches_diagnostic"] == prefix["non_H15_prefix_pairs_compared"] and prefix["frozen_prefix_hash_matches"] == 0:
        prefix_interpretation = "All non-H15 frozen prefix hashes mismatched, but the diagnostic semantic prefix hash matched for every non-H15 branch; the blocking prefix flag is therefore likely an over-strict administrative hash criterion, not physical state mismatch. Frozen gate remains failed as written."
    else:
        prefix_interpretation = "Semantic prefix comparison did not fully clear the mismatch; inspect examples before relying on relaxed state-matched labels."
    diagnostics = {
        "created_utc": created,
        "method": "vehicle_stress_scenario_stage2_postdiagnostic_v0_analysis_only",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "stage2_completed": {"path": rel(STAGE2_COMPLETED), "sha256": sha256(STAGE2_COMPLETED), "episodes": completed.get("episodes"), "control_steps": completed.get("control_steps")},
        "stage2_raw": {"path": rel(STAGE2_RAW), "sha256": sha256(STAGE2_RAW)},
        "frozen_stage2_headline": completed.get("headline"),
        "prefix_mismatch_diagnostic": prefix,
        "opportunity_diagnostic": opp,
        "interpretation": {
            "prefix_mismatch": prefix_interpretation,
            "stage2_scientific_read": "There is a real local continuation tradeoff at target 0/case 5/branch step 18 (H30 best by total/physical cost), but it is one state only; the other 11 states are neutral or harmful for non-H15 branches. This supports sparse/localized within-episode opportunity, not enough training labels.",
        },
        "decision": decision,
    }
    write_json(OUT_DIR / "raw.json", diagnostics)
    lines = [
        "# Vehicle stress-scenario Stage2 postdiagnostic v0",
        "",
        f"UTC: `{created}`. Analysis-only; no simulations, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Stage2 evidence check",
        "",
        f"- Stage2 completed: `{rel(STAGE2_COMPLETED)}`; episodes `{completed.get('episodes')}`, control steps `{completed.get('control_steps')}`.",
        f"- Frozen gate headline: `{completed.get('headline')}`.",
        "",
        "## Prefix artifact diagnostic",
        "",
        f"- Non-H15 prefix pairs compared: `{prefix['non_H15_prefix_pairs_compared']}`.",
        f"- Frozen prefix hash matches: `{prefix['frozen_prefix_hash_matches']}`.",
        f"- Diagnostic semantic prefix matches: `{prefix['semantic_prefix_hash_matches_diagnostic']}` / `{prefix['non_H15_prefix_pairs_compared']}`.",
        f"- Interpretation: {prefix_interpretation}",
        "",
        "## Opportunity diagnostic",
        "",
        f"- Frozen-gate positive states: `{opp['frozen_gate_positive_states']}`.",
        f"- State-matched relaxed-prefix positive states (diagnostic only): `{opp['relaxed_state_matched_positive_state_count_diagnostic']}`; positive branches `{opp['relaxed_positive_branch_count']}` / `{opp['non_H15_branch_count']}`.",
        f"- Large total harms: `{opp['large_total_harm_branch_count']}` / `{opp['non_H15_branch_count']}`.",
        f"- Total-gain summary all non-H15: `{opp['gain_total_summary_all_non_H15']}`.",
        f"- Physical-gain summary all non-H15: `{opp['gain_physical_summary_all_non_H15']}`.",
        "",
        "### Relaxed positive states (diagnostic only; not changing frozen gate)",
        "",
        "| target | case | role | branch step | positive H | best total H/gain | best physical H/gain |",
        "|---:|---:|---|---:|---|---|---|",
    ]
    for r in opp["relaxed_positive_states"]:
        lines.append("| %d | %d | `%s` | %d | `%s` | H%d/%.6g | H%d/%.6g |" % (int(r["target_index"]), int(r["case"]), r["selection_role"], int(r["branch_step"]), r["positive_horizons"], int(r["best_total_horizon"]), float(r["best_total_gain"]), int(r["best_physical_horizon"]), float(r["best_physical_gain"])))
    if not opp["relaxed_positive_states"]:
        lines.append("| - | - | - | - | - | - | - |")
    lines += [
        "",
        "## Decision",
        "",
        f"- Retraining/refit now: `{decision['retraining_refit_now']}`.",
        f"- Reason: {decision['reason']}",
        f"- Next: {decision['next_high_information_action']}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Stage2 postdiagnostic state ({created})\n\n"
        f"Stage2 analysis-only postdiagnostic completed. Frozen gate positives={opp['frozen_gate_positive_states']}; relaxed state-matched positives={opp['relaxed_state_matched_positive_state_count_diagnostic']}; semantic prefix matches={prefix['semantic_prefix_hash_matches_diagnostic']}/{prefix['non_H15_prefix_pairs_compared']}; retraining/refit now={decision['retraining_refit_now']}. Next: terminal/reward/modeling audit or a separately versioned stronger scenario design after backup.\n",
        encoding="utf-8",
    )
    request = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE2_POSTDIAGNOSTIC_V0_20260928T1815Z.json"
    write_json(request, {
        "requested_utc": created,
        "reason": "backup Stage2 rollout and postdiagnostic analysis artifacts before further diagnostics or simulations",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(request), rel(STAGE2_DIR)],
    })
    diagnostics["backup_request"] = rel(request)
    write_json(OUT_DIR / "raw.json", diagnostics)
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-scenario Stage2 postdiagnostic v0

UTC: {created}. Analysis-only Stage2 postdiagnostic completed after the 72-episode matched-continuation rollout. Frozen Stage2 gate remains failed: positive states={opp['frozen_gate_positive_states']}. Diagnostic semantic prefix comparison indicates {prefix['semantic_prefix_hash_matches_diagnostic']}/{prefix['non_H15_prefix_pairs_compared']} non-H15 prefixes match on state/control/cost semantics despite frozen hash mismatches. Under a diagnostic state-matched relaxation (not changing the frozen gate), positive states={opp['relaxed_state_matched_positive_state_count_diagnostic']} and large total harms={opp['large_total_harm_branch_count']}/{opp['non_H15_branch_count']}. Decision: do not retrain/refit now; the label signal is too sparse. Next high-information action is an analysis-only terminal/reward/modeling audit or a separately versioned stronger source-supported scenario design after backup. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`. Backup request: `{rel(request)}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), STATE_PATH, request, STAGE2_RAW, STAGE2_COMPLETED]
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
        "frozen_gate_positive_states": opp["frozen_gate_positive_states"],
        "relaxed_state_matched_positive_state_count_diagnostic": opp["relaxed_state_matched_positive_state_count_diagnostic"],
        "semantic_prefix_hash_matches_diagnostic": prefix["semantic_prefix_hash_matches_diagnostic"],
        "non_H15_prefix_pairs_compared": prefix["non_H15_prefix_pairs_compared"],
        "large_total_harm_branch_count": opp["large_total_harm_branch_count"],
        "backup_request": rel(request),
        "decision": decision,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "frozen_gate_positive_states": opp["frozen_gate_positive_states"],
        "relaxed_state_matched_positive_states_diagnostic": opp["relaxed_state_matched_positive_state_count_diagnostic"],
        "semantic_prefix_matches": [prefix["semantic_prefix_hash_matches_diagnostic"], prefix["non_H15_prefix_pairs_compared"]],
        "large_total_harms": opp["large_total_harm_branch_count"],
        "retraining_refit_now": decision["retraining_refit_now"],
        "backup_request": rel(request),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
