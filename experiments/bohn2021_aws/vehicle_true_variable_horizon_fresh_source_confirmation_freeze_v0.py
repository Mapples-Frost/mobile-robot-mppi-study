#!/usr/bin/env python3
"""Freeze a source-independent fresh-source confirmation protocol for true-H H10/H15 selection.

This is a metadata-only, no-simulation protocol freezer.  It follows the
2026-09-29 true-variable-H diagnostics:

* true smaller MPC horizons reduce measured solver/decision time when the
  controller dimension actually shrinks;
* oracle/risk-anchor development banks show a real measured H10-vs-H15/H25
  control/compute signal, but all-profile deployable selector CV remains blocked;
* terminal-profile/source transfer is positive only in terminal-specific subsets.

The next discriminating experiment must therefore use a fresh source: select
new reset cases from the existing stress-v1 candidate pool that have not been
used in Stage1 H-outcome maps, oracle-bank labels, or risk-anchor acquisition;
then, in a future runner, collect H15 traces first, select branch states from
those H15 traces before any new H10 outcome is observed, and only then run
blocked H10/H15 branch comparisons under both terminal profiles.

This script freezes that protocol and writes a backup request.  It performs no
candidate resets, no rollout/simulation, no training/refit, no validation64 and
no sealed-test access.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0"
STAMP = "20260929T1015Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
CONTINUE_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T1015_after_fresh_source_confirmation_freeze.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_FREEZE_V0_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

BANK_PATH = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
BANK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/completed.json"
STAGE1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
STAGE1_POST_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json"
STAGE1_POST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/completed.json"
ORACLE_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_frozen_20260929T0725Z.json"
ORACLE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json"
ORACLE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json"
RISK_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_frozen_20260929T0810Z.json"
RISK_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json"
RISK_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json"
ROBUST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_robust_terminal_label_cv_v0_20260929T0925Z/completed.json"
SAFETY_FAST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_safety_gated_selector_cv_v0b_fast_20260929T0950Z/completed.json"
TRANSFER_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_domain_profile_transfer_audit_v0_20260929T1000Z/completed.json"
TRANSFER_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_domain_profile_transfer_audit_v0_20260929T1000Z/raw.json"

SOURCE = Path(__file__).resolve()
TRUE_HORIZONS = [10, 15]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
STAGE_A_SCAN_H = 15
TARGET_CASES = 8
BRANCH_STATES_PER_CASE = 2
REPEATS = 2
MAX_STEPS = 150
RNG_SEED = 202609291015
GROUP_QUOTAS = {
    "fresh_high_heading_long_or_medium": 4,
    "fresh_high_heading_short": 1,
    "fresh_low_heading_low_clearance_control": 1,
    "fresh_lower_stress_control": 2,
}
MARKER = f"vehicle-true-variable-H-fresh-source-confirmation-freeze-v0-{STAMP}"


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def si(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def percentile(vals: Sequence[float], q: float, default: float = 0.0) -> float:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return default
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return float(xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: " + rel(path))
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "historical_validation64_bank_opened"):
        if key in obj and obj.get(key) is not False:
            raise ContractError(f"{key} flag must be false in {rel(path)}")
    return obj


def require_inputs() -> Dict[str, str]:
    paths = [BANK_PATH, BANK_DONE, STAGE1_DONE, STAGE1_POST_RAW, STAGE1_POST_DONE, ORACLE_PROTOCOL, ORACLE_RAW, ORACLE_DONE, RISK_PROTOCOL, RISK_RAW, RISK_DONE, ROBUST_DONE, SAFETY_FAST_DONE, TRANSFER_DONE, TRANSFER_RAW]
    missing = [rel(p) for p in paths if not p.exists()]
    if missing:
        raise ContractError("missing required development artifacts: " + ", ".join(missing))
    for p in [BANK_DONE, STAGE1_DONE, STAGE1_POST_DONE, ORACLE_DONE, RISK_DONE, ROBUST_DONE, SAFETY_FAST_DONE, TRANSFER_DONE]:
        completed_ok(p)
    return {rel(p): sha256(p) for p in paths + [SOURCE] if p.exists()}


def collect_values(obj: Any, key: str) -> List[Any]:
    out: List[Any] = []
    if isinstance(obj, Mapping):
        if key in obj:
            out.append(obj.get(key))
        for v in obj.values():
            out.extend(collect_values(v, key))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_values(v, key))
    return out


def used_candidate_indices(*objs: Any) -> List[int]:
    vals: List[int] = []
    for obj in objs:
        for v in collect_values(obj, "source_candidate_index"):
            x = si(v, -1)
            if x >= 0:
                vals.append(x)
    return sorted(set(vals))


def selected_stage1_indices(bank: Mapping[str, Any]) -> List[int]:
    sel = bank.get("selection") or {}
    vals = [si(x, -1) for x in (sel.get("selected_indices") or [])]
    for row in (sel.get("selected_metadata") or []):
        vals.append(si(row.get("candidate_index"), -1))
    return sorted(set(x for x in vals if x >= 0))


def thresholds_from_bank(bank: Mapping[str, Any], metas: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    thr = ((bank.get("selection") or {}).get("thresholds") or {})
    abs_vals = [sf(m.get("abs_theta_r"), float("nan")) for m in metas]
    traj_vals = [sf(m.get("traj_steps"), float("nan")) for m in metas]
    clear_vals = [sf(m.get("min_reference_obstacle_clearance"), float("nan")) for m in metas]
    return {
        "abs_q70": sf(thr.get("abs_q70"), percentile(abs_vals, 70.0)),
        "abs_q60": sf(thr.get("abs_q60"), percentile(abs_vals, 60.0)),
        "abs_median": sf(thr.get("abs_median"), percentile(abs_vals, 50.0)),
        "traj_median": sf(thr.get("traj_median"), percentile(traj_vals, 50.0)),
        "traj_q60": sf(thr.get("traj_q60"), percentile(traj_vals, 60.0)),
        "traj_q40": sf(thr.get("traj_q40"), percentile(traj_vals, 40.0)),
        "clear_q30": sf(thr.get("clear_q30"), percentile(clear_vals, 30.0)),
        "clear_q40": sf(thr.get("clear_q40"), percentile(clear_vals, 40.0)),
    }


def classify(row: Mapping[str, Any], thr: Mapping[str, float], relaxed: bool = False) -> str:
    abs_theta = sf(row.get("abs_theta_r"), 0.0)
    traj = sf(row.get("traj_steps"), 0.0)
    clearance = sf(row.get("min_reference_obstacle_clearance"), 1e9)
    abs_hi = thr["abs_q60"] if relaxed else thr["abs_q70"]
    traj_mid = thr["traj_q40"] if relaxed else thr["traj_median"]
    traj_short = thr["traj_q60"] if relaxed else thr["traj_median"]
    clear_low = thr["clear_q40"] if relaxed else thr["clear_q30"]
    if abs_theta >= abs_hi and traj >= traj_mid:
        return "fresh_high_heading_long_or_medium"
    if abs_theta >= abs_hi and traj < traj_short:
        return "fresh_high_heading_short"
    if abs_theta <= thr["abs_median"] and clearance <= clear_low:
        return "fresh_low_heading_low_clearance_control"
    if abs_theta <= thr["abs_median"] and clearance > clear_low:
        return "fresh_lower_stress_control"
    return "fresh_unclassified_metadata_pool"


def diversity_key(row: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], prefer_high: bool) -> Tuple[float, float, float, int]:
    score = sf(row.get("stress_v1_score"), 0.0)
    if not prefer_high:
        score = -score
    z = [sf(row.get("abs_theta_r"), 0.0), sf(row.get("traj_steps"), 0.0), sf(row.get("min_reference_obstacle_clearance"), 0.0)]
    if not selected:
        min_d = 1e9
    else:
        ds = []
        for s in selected:
            zz = [sf(s.get("abs_theta_r"), 0.0), sf(s.get("traj_steps"), 0.0), sf(s.get("min_reference_obstacle_clearance"), 0.0)]
            ds.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(z, zz))))
        min_d = min(ds)
    sign_balance = -abs(sum(-1 if sf(s.get("theta_r"), 0.0) < 0 else 1 for s in selected) + (-1 if sf(row.get("theta_r"), 0.0) < 0 else 1))
    return (float(sign_balance), float(min_d), float(score), -si(row.get("candidate_index"), 0))


def select_group(pool: Sequence[Mapping[str, Any]], k: int, already: set, group: str, prefer_high: bool) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    candidates = [dict(r) for r in pool if si(r.get("candidate_index"), -1) not in already]
    if len(candidates) < k:
        raise ContractError(f"not enough candidates for {group}: have {len(candidates)} need {k}")
    while len(chosen) < k:
        row = max(candidates, key=lambda r: diversity_key(r, chosen, prefer_high=prefer_high))
        candidates = [r for r in candidates if si(r.get("candidate_index"), -1) != si(row.get("candidate_index"), -1)]
        row["fresh_confirmation_group"] = group
        chosen.append(row)
        already.add(si(row.get("candidate_index"), -1))
    return chosen


def build_case_selection(bank: Mapping[str, Any], excluded_indices: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sel = bank.get("selection") or {}
    metas = list(sel.get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) < 64:
        raise ContractError("bank candidate metadata/case dimensions invalid")
    excluded = set(int(x) for x in excluded_indices)
    thr = thresholds_from_bank(bank, metas)
    enriched: List[Dict[str, Any]] = []
    for row0 in metas:
        row = dict(row0)
        idx = si(row.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases):
            continue
        if idx in excluded:
            continue
        row["fresh_confirmation_group_strict"] = classify(row, thr, relaxed=False)
        row["fresh_confirmation_group_relaxed"] = classify(row, thr, relaxed=True)
        row["excluded_from_previous_horizon_outcomes"] = True
        enriched.append(row)
    chosen_rows: List[Dict[str, Any]] = []
    used: set = set()
    fallbacks: List[str] = []
    for group, k in GROUP_QUOTAS.items():
        strict_pool = [r for r in enriched if r["fresh_confirmation_group_strict"] == group]
        relaxed_pool = [r for r in enriched if r["fresh_confirmation_group_relaxed"] == group]
        pool = strict_pool
        fb = None
        if len([r for r in pool if si(r.get("candidate_index"), -1) not in used]) < k:
            pool = relaxed_pool
            fb = group + "_relaxed_thresholds"
        if len([r for r in pool if si(r.get("candidate_index"), -1) not in used]) < k:
            # Last resort stays metadata-only but preserves the intended role in the protocol.
            if group == "fresh_high_heading_long_or_medium":
                pool = sorted(enriched, key=lambda r: (sf(r.get("stress_v1_score"), 0.0), sf(r.get("abs_theta_r"), 0.0)), reverse=True)
            elif group == "fresh_high_heading_short":
                pool = sorted(enriched, key=lambda r: (sf(r.get("abs_theta_r"), 0.0), -sf(r.get("traj_steps"), 0.0)), reverse=True)
            elif group == "fresh_low_heading_low_clearance_control":
                pool = sorted(enriched, key=lambda r: (sf(r.get("min_reference_obstacle_clearance"), 1e9), sf(r.get("abs_theta_r"), 0.0)))
            else:
                pool = sorted(enriched, key=lambda r: (sf(r.get("stress_v1_score"), 0.0), sf(r.get("abs_theta_r"), 0.0)))
            fb = group + "_global_metadata_fallback"
        if fb:
            fallbacks.append(fb)
        chosen = select_group(pool, k, used, group, prefer_high=("control" not in group and "lower_stress" not in group))
        for r in chosen:
            if fb:
                r.setdefault("fresh_confirmation_selection_fallbacks", []).append(fb)
            chosen_rows.append(r)
    if len(chosen_rows) != TARGET_CASES or len({si(r.get("candidate_index"), -1) for r in chosen_rows}) != TARGET_CASES:
        raise ContractError("fresh-source case selection count/uniqueness failed")
    selected_cases: List[Dict[str, Any]] = []
    for order, row in enumerate(chosen_rows):
        idx = si(row.get("candidate_index"), -1)
        selected_cases.append({
            "fresh_case_index": order,
            "source_candidate_index": idx,
            "fresh_confirmation_group": row["fresh_confirmation_group"],
            "strict_metadata_group": row.get("fresh_confirmation_group_strict"),
            "relaxed_metadata_group": row.get("fresh_confirmation_group_relaxed"),
            "theta_r": sf(row.get("theta_r")),
            "abs_theta_r": sf(row.get("abs_theta_r")),
            "traj_steps": sf(row.get("traj_steps")),
            "min_reference_obstacle_clearance": sf(row.get("min_reference_obstacle_clearance")),
            "stress_v1_score": sf(row.get("stress_v1_score")),
            "stress_flags_v1": row.get("stress_flags_v1"),
            "selection_fallbacks": row.get("fresh_confirmation_selection_fallbacks", []),
            "case_snapshot_from_candidate_pool": cases[idx],
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(cases[idx]), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
            "selection_rule": "metadata-only fresh-source selection from stress-v1 candidate pool excluding all Stage1 H-outcome selected cases and oracle/risk-anchor source_candidate_index values; no H10/H15 outcome for this source was observed before selection",
        })
    diagnostics = {
        "thresholds": thr,
        "excluded_source_candidate_indices_count": len(set(excluded_indices)),
        "eligible_after_exclusion_count": len(enriched),
        "strict_pool_counts_after_exclusion": {g: sum(1 for r in enriched if r["fresh_confirmation_group_strict"] == g) for g in sorted(set(list(GROUP_QUOTAS) + ["fresh_unclassified_metadata_pool"]))},
        "relaxed_pool_counts_after_exclusion": {g: sum(1 for r in enriched if r["fresh_confirmation_group_relaxed"] == g) for g in sorted(set(list(GROUP_QUOTAS) + ["fresh_unclassified_metadata_pool"]))},
        "selection_fallbacks": fallbacks,
        "selected_source_candidate_indices": [x["source_candidate_index"] for x in selected_cases],
        "selected_group_counts": {g: sum(1 for x in selected_cases if x["fresh_confirmation_group"] == g) for g in GROUP_QUOTAS},
    }
    return selected_cases, diagnostics


def make_stage_a_schedule(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "stage": "A_h15_trace_scan",
            "execution_index": i,
            "fresh_case_index": int(case["fresh_case_index"]),
            "source_candidate_index": int(case["source_candidate_index"]),
            "true_mpc_n_horizon": STAGE_A_SCAN_H,
            "terminal_profile": "matched_terminal",
            "purpose": "collect H15-only full trace for branch-state selection before any new H10 outcome",
        }
        for i, case in enumerate(selected_cases)
    ]


def make_branch_arm_template(selected_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    rows: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        for case in selected_cases:
            for state_slot in range(BRANCH_STATES_PER_CASE):
                for profile in TERMINAL_PROFILES:
                    hs = list(TRUE_HORIZONS)
                    rng.shuffle(hs)
                    for h in hs:
                        rows.append({
                            "stage": "B_blocked_branch_trueH10_H15",
                            "execution_index": exe,
                            "repeat": rep,
                            "fresh_case_index": int(case["fresh_case_index"]),
                            "source_candidate_index": int(case["source_candidate_index"]),
                            "branch_state_slot": state_slot,
                            "terminal_profile": profile,
                            "true_mpc_n_horizon": h,
                            "blocked_randomization_unit": f"repeat{rep}|fresh_case{case['fresh_case_index']}|slot{state_slot}|{profile}",
                        })
                        exe += 1
    return rows


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        row = f"{now_utc().isoformat()},{NAME},metadata_no_simulation_fresh_source_confirmation_freeze,development_no_validation_no_test,0,0,0,0,0,False,{rel(OUT / 'completed.json')}\n"
        if NAME not in old[-50000:]:
            reg.write_text(old.rstrip() + "\n" + row, encoding="utf-8")


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle true-variable-H fresh-source confirmation freeze v0",
        "",
        f"UTC: `{raw['created_utc']}`. Metadata-only protocol freeze; no simulations, no training/refit, no validation64, no sealed test.",
        "",
        "## Rationale",
        "",
        "Terminal-specific source transfer gave positive development evidence, but all-profile deployment collapsed to all-H15.  This protocol freezes a source-independent confirmation rather than another unchanged label-density/CV sweep.",
        "",
        "## Fresh source selection",
        "",
        f"- Fresh cases: `{len(raw['selected_cases'])}` from the existing stress-v1 candidate pool, before any H outcomes for those candidate IDs.",
        f"- Selected source_candidate_index values: `{raw['case_selection_diagnostics']['selected_source_candidate_indices']}`.",
        f"- Group counts: `{raw['case_selection_diagnostics']['selected_group_counts']}`.",
        f"- Excluded source_candidate_index count: `{raw['case_selection_diagnostics']['excluded_source_candidate_indices_count']}` (Stage1 H-outcome selected plus oracle/risk-anchor sources).",
        "",
        "| fresh_case | source_candidate_index | group | theta_r | traj_steps | clearance | stress_score |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for c in raw["selected_cases"]:
        lines.append("| %d | %d | `%s` | %.6g | %.6g | %.6g | %.6g |" % (int(c["fresh_case_index"]), int(c["source_candidate_index"]), c["fresh_confirmation_group"], sf(c.get("theta_r")), sf(c.get("traj_steps")), sf(c.get("min_reference_obstacle_clearance")), sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Frozen future runner design",
        "",
        f"- Stage A: `{raw['budget_declared']['stage_a_h15_trace_episodes']}` full H15 trace scans; states selected only from H15 traces before any H10 branch outcome.",
        f"- Stage B: `{raw['budget_declared']['stage_b_branch_episodes']}` blocked branch episodes over true horizons `{TRUE_HORIZONS}`, terminal profiles `{TERMINAL_PROFILES}`, repeats `{REPEATS}`.",
        f"- Total cap: `{raw['budget_declared']['total_episodes_exact']}` episodes / `{raw['budget_declared']['control_step_upper_bound']}` control steps.",
        "- Primary confirmation is frozen as a source-trained, shared-H15-terminal H10-abstention guard evaluated on fresh states with no fitting to fresh labels.",
        "",
        "## Decision",
        "",
        "After verified external backup, write/run the development-only fresh-source runner.  If the frozen source-trained rule passes shared-H15-terminal fresh confirmation with zero unsafe/physical-regression false positives and >=5% measured whole-decision saving vs fixed true H15, proceed to an optimized terminal-profile-aware selector/value-calibration smoke. If it fails or only matched/profile-specific labels survive, prioritize terminal-value/objective/representation repair or scenario redesign before any selector rollout.",
        "",
        f"Protocol: `{raw['protocol']}`. Backup request: `{raw['backup_request']}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_continue_state(raw: Mapping[str, Any]) -> None:
    text = f"""# Continue state after fresh-source confirmation freeze v0

UTC: {raw['created_utc']}
Elapsed since first supervisor event: {raw['elapsed_since_first_supervisor_event_seconds']:.1f}s ({raw['elapsed_since_first_supervisor_event_seconds']/3600.0:.2f}h).

Completed this iteration: `{NAME}` metadata/no-simulation freeze. No simulations, no control steps, no training/refit, no validation64, no sealed test.

Artifacts:
- Summary: `{rel(OUT / 'summary.md')}`
- Raw: `{rel(OUT / 'raw.json')}`
- Completed: `{rel(OUT / 'completed.json')}`
- Protocol: `{raw['protocol']}`
- Backup request: `{raw['backup_request']}`

Key decision preserved: fresh-source confirmation is frozen from metadata-only stress-v1 candidate-pool cases that exclude all Stage1 H-outcome selected cases and oracle/risk-anchor source_candidate_index values. Future runner must first run H15 traces, select two branch states per fresh case from those H15 traces before observing any H10 branch outcome, then run blocked H10/H15 branch comparisons under both terminal profiles.

Next action after verified backup: implement/run `vehicle_true_variable_horizon_fresh_source_confirmation_v0_runner.py` with dry-run and then development rollout. Do not open validation64 or sealed test. Do not run selector rollout/formal validation from the current selector before this confirmation or a stronger terminal/value-calibration diagnostic.

Fresh selected source_candidate_index values: {raw['case_selection_diagnostics']['selected_source_candidate_indices']}.
Budget for future runner: {raw['budget_declared']['total_episodes_exact']} episodes, cap {raw['budget_declared']['control_step_upper_bound']} control steps; no training/refit.
"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    CONTINUE_STATE.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--i-accept-no-simulation-fresh-source-confirmation-freeze-v0", action="store_true")
    args = parser.parse_args(argv)
    if not args.freeze or not args.i_accept_no_simulation_fresh_source_confirmation_freeze_v0:
        raise ContractError("requires --freeze and explicit no-simulation fresh-source freeze acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    input_hashes = require_inputs()
    bank = read_json(BANK_PATH)
    oracle_protocol = read_json(ORACLE_PROTOCOL)
    oracle_raw = read_json(ORACLE_RAW)
    risk_protocol = read_json(RISK_PROTOCOL)
    risk_raw = read_json(RISK_RAW)
    transfer_raw = read_json(TRANSFER_RAW)

    stage1_selected = selected_stage1_indices(bank)
    prior_source_used = used_candidate_indices(oracle_protocol, oracle_raw, risk_protocol, risk_raw)
    excluded = sorted(set(stage1_selected) | set(prior_source_used))
    selected_cases, case_diag = build_case_selection(bank, excluded)
    stage_a = make_stage_a_schedule(selected_cases)
    branch_template = make_branch_arm_template(selected_cases)
    stage_a_eps = len(stage_a)
    stage_b_eps = len(branch_template)
    total_eps = stage_a_eps + stage_b_eps
    control_cap = total_eps * MAX_STEPS

    source_trained_policies = [
        {
            "policy_id": "primary_shared_h15_terminal_source_trained_guard_raw_k1_q0.5_m1.25_terminal_agreement_only",
            "terminal_profile": "shared_h15_terminal",
            "robust_label_strategy": "terminal_agreement_only",
            "training_sources": ["oracle_bank", "risk_anchor"],
            "feature_set": "online_observable_no_metadata",
            "model_family": "positive_support_knn_guard",
            "config": {"family": "positive_support_knn_guard", "mode": "raw", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
            "fresh_fit_allowed": False,
            "fresh_threshold_tuning_allowed": False,
            "primary_gate": True,
        },
        {
            "policy_id": "secondary_shared_h15_terminal_source_trained_guard_raw_abs_l2_k1_q0.75_m1.25_terminal_agreement_only",
            "terminal_profile": "shared_h15_terminal",
            "robust_label_strategy": "terminal_agreement_only",
            "training_sources": ["oracle_bank", "risk_anchor"],
            "feature_set": "online_observable_no_metadata",
            "model_family": "positive_support_knn_guard",
            "config": {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.75, "negative_margin": 1.25},
            "fresh_fit_allowed": False,
            "fresh_threshold_tuning_allowed": False,
            "primary_gate": False,
        },
        {
            "policy_id": "diagnostic_matched_terminal_source_trained_guard_raw_abs_l2_k1_q0.75_m1.25_terminal_agreement_only",
            "terminal_profile": "matched_terminal",
            "robust_label_strategy": "terminal_agreement_only",
            "training_sources": ["oracle_bank", "risk_anchor"],
            "feature_set": "online_observable_no_metadata",
            "model_family": "positive_support_knn_guard",
            "config": {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.75, "negative_margin": 1.25},
            "fresh_fit_allowed": False,
            "fresh_threshold_tuning_allowed": False,
            "primary_gate": False,
        },
    ]

    protocol = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_fresh_source_confirmation_freeze_no_simulation_not_validation_not_test",
        "method_label": "IMPROVED",
        "hypothesis": "A terminal-profile-aware H10-abstention selector trained only on existing oracle/risk-anchor development labels captures a source-general online-observable shortening signal under a calibrated terminal profile. On fresh source states selected from H15 traces before any H10 outcome, it should save measured whole-decision time versus fixed true H15 without unsafe or physical-regression H10 false positives. Failure implies source/profile overfit or terminal/value/representation mismatch rather than absent measured compute opportunity.",
        "before_evidence": {
            "domain_profile_transfer_decision": transfer_raw.get("decision"),
            "source_transfer_pass_count": transfer_raw.get("source_transfer_pass_count"),
            "profile_transfer_pass_count": transfer_raw.get("profile_transfer_pass_count"),
            "all_profile_transfer_blocked": "all_profiles transfers selected all_h15 in the latest transfer audit",
            "risk_anchor_labels_terminal_dependent": "risk-anchor acquisition had 7/10 terminal-profile disagreements and fixed H15 absorption despite measured oracle/H25 saving",
        },
        "case_source": {
            "bank_path": rel(BANK_PATH),
            "candidate_pool_created_before_this_freeze": True,
            "candidate_pool_horizon_outcomes_available_only_for_stage1_selected_indices": True,
            "excluded_indices_rule": "exclude all stress-v1 Stage1 selected_indices plus every source_candidate_index appearing in oracle-bank or risk-anchor protocols/runs",
            "stage1_selected_indices_excluded": stage1_selected,
            "oracle_or_risk_source_candidate_indices_excluded": prior_source_used,
            "all_excluded_source_candidate_indices": excluded,
            "selected_fresh_cases": selected_cases,
            "case_selection_diagnostics": case_diag,
        },
        "stage_A_h15_trace_state_selection": {
            "schedule": stage_a,
            "horizon": STAGE_A_SCAN_H,
            "terminal_profile": "matched_terminal",
            "state_selection_time_order": "run all Stage-A H15 traces, write selected_state_manifest, then and only then run any Stage-B H10/H15 branch comparison",
            "branch_states_per_case": BRANCH_STATES_PER_CASE,
            "eligible_step_bounds": {"min_step": 8, "max_step": 90, "reserve_terminal_margin_steps": 3},
            "windows": [
                {"slot": 0, "name": "early_mid", "low_fraction": 0.12, "high_fraction": 0.36, "selection": "max_h15_trace_risk_score"},
                {"slot": 1, "name": "mid_late", "low_fraction": 0.38, "high_fraction": 0.75, "selection": "max_h15_trace_risk_score_with_min_step_separation_8"},
            ],
            "risk_score_formula": "8*sqrt(max(performance,0)) + 2*abs(observation[2]) + 0.6*abs(u_omega) + 0.04*solver_iteration_proxy + 1.0*min(10,constraint) + 0.6*abs(y) + 0.5*abs(theta) + 1.2*phase_fraction",
            "controls_not_deleted": "same rule is used for control groups; low-risk/control cases remain in the future evaluation even if they produce only H15 labels",
        },
        "stage_B_branch_confirmation": {
            "branch_arm_template_after_state_selection": branch_template,
            "true_horizons": TRUE_HORIZONS,
            "terminal_profiles": TERMINAL_PROFILES,
            "repeats": REPEATS,
            "blocked_randomization_seed": RNG_SEED,
            "measured_timing_required": ["whole_decision_time", "solver_attempt_time"],
            "baseline": "fixed true H15 within the same terminal profile and fresh state group",
        },
        "source_trained_selector_confirmation": {
            "policies_frozen_before_fresh_outcomes": source_trained_policies,
            "primary_success_gate": {
                "scope": "shared_h15_terminal fresh branch groups only, using the primary frozen source-trained policy with no fresh fitting/tuning",
                "safety": "zero H10 chosen groups with H10 unsafe, constraint, initial/final solver failure, or solver_failure_steps>0 when H15 is safe",
                "physical": "aggregate predicted-policy physical delta vs fixed true H15 <= max(2*N, 0.05*abs(H15 physical sum)) and no per-group catastrophic H10 physical regression > max(2, 0.25*abs(H15 physical))",
                "compute": ">=5% measured whole-decision saving vs fixed true H15; strong if >=10%; solver timing reported separately",
                "nontriviality": "at least one H10 prediction and nonconstant horizons; all-H15 is a safe fallback but not a pass",
            },
            "secondary_diagnostics": ["matched_terminal profile-specific pass/fail", "terminal-profile disagreement rate", "all-profile pooled deployment remains blocked unless both profiles pass independently", "fixed true H10 and all-H15 baselines reported"],
        },
        "budget_declared": {
            "stage_a_h15_trace_episodes": stage_a_eps,
            "stage_b_branch_episodes": stage_b_eps,
            "total_episodes_exact": total_eps,
            "control_step_upper_bound": control_cap,
            "candidate_pool_resets": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "new_refit_steps": 0,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "decision_after_future_runner": {
            "if_primary_passes": "freeze a small selector-overhead/closed-loop smoke using the terminal-profile-aware true-H controller cache, then require independent validation with fair fixed-H tuning before any final test",
            "if_profile_specific_only": "treat as terminal-value/calibration evidence; run terminal/value representation repair rather than all-profile selector rollout",
            "if_all_h15_or_false_positives": "block selector rollout; prioritize terminal-value/objective/representation/scenario redesign or bounded value-refit",
        },
        "access_rules": {"development_only": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_external_backup_before_future_runner_or_simulation": True},
    }
    write_json(PROTOCOL, protocol)
    write_json(BACKUP_REQ, {
        "requested_utc": created.isoformat(),
        "reason": "backup fresh-source confirmation freeze/source/protocol before writing/running any future H15-trace or H10/H15 branch simulations",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "future_total_episodes_exact": total_eps,
        "future_control_step_upper_bound": control_cap,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(CONTINUE_STATE), rel(BACKUP_REQ)],
    })
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "method": NAME,
        "classification": "metadata_no_simulation_fresh_source_confirmation_protocol_freeze",
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "input_hashes": input_hashes,
        "selected_cases": selected_cases,
        "case_selection_diagnostics": case_diag,
        "budget_declared": protocol["budget_declared"],
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQ),
        "next_action_after_backup": "write/run fresh-source confirmation runner: Stage-A H15 traces, frozen state selection manifest, then blocked true-H10/H15 branch outcomes under both terminal profiles; no validation64/test",
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    write_continue_state(raw)
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation freeze v0

UTC: {created.isoformat()}. Metadata-only/no-simulation protocol freeze completed. Fresh source cases were selected from the stress-v1 candidate pool by metadata only, excluding all Stage1 H-outcome selected indices plus oracle/risk-anchor source_candidate_index values. Future runner budget is {total_eps} episodes / {control_cap} control-step cap: {stage_a_eps} H15 trace scans followed by {stage_b_eps} blocked H10/H15 branch episodes under both terminal profiles. No validation64/test/training/refit. Further simulation/training/refit requires verified backup covering `{rel(BACKUP_REQ)}`. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(PROTOCOL)}`.
""")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, CONTINUE_STATE, BACKUP_REQ, BANK_DONE, STAGE1_DONE, ORACLE_DONE, RISK_DONE, TRANSFER_DONE]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "classification": raw["classification"],
        "formal_scientific_evidence": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "future_total_episodes_exact": total_eps,
        "future_control_step_upper_bound": control_cap,
        "headline": {"fresh_source_protocol_frozen": True, "selected_case_count": len(selected_cases), "selected_group_counts": case_diag["selected_group_counts"], "primary_terminal_profile": "shared_h15_terminal", "requires_backup_before_future_runner": True, "train_or_refit_now": False},
        "backup_request": rel(BACKUP_REQ),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "protocol": rel(PROTOCOL), "headline": completed["headline"], "future_total_episodes_exact": total_eps, "future_control_step_upper_bound": control_cap, "new_simulations": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQ)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
