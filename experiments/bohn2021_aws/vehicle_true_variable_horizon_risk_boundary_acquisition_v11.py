#!/usr/bin/env python3
"""v11 targeted true-variable-H risk-boundary acquisition.

Development-only IMPROVED diagnostic after v9/v10b negative selector-refit
evidence.  v9/v10b showed branch-level measured compute opportunity but simple
online/deployable tree/kNN/poly risk-value selectors could not safely separate
beneficial H10 from catastrophic H10 on opened development banks, especially near
low-clearance and high-heading boundary states.  This bounded acquisition adds a
small, source-independent set of fresh branch labels targeted at those boundary
modes before another selector/value refit.

The implementation reuses the audited v8 true-H10/H15 branch runner but patches
only the pre-outcome case-selection protocol, source namespace, state-count
contract, summary/doc strings, and budgets.  It performs no validation64 access,
no sealed-test access, no gradient training, and no selector refit.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_boundary_acquisition_v11"
STAMP = "20260929T1835Z"
SOURCE = Path(__file__).resolve()
BASE_V8_SOURCE = v8.SOURCE
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

# v8c/v10b development artifacts used only for exclusion/provenance and before-evidence.
V8C_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_preoutcome_frozen_20260929T1320Z.json"
V8C_MANIFEST = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/selected_state_manifest.json"
V8C_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json"
V10B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/completed.json"
V10B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/summary.md"

# Patch the reused runner namespace/budget before calling v8.run().
v8.NAME = NAME
v8.STAMP = STAMP
v8.SOURCE = SOURCE
v8.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
v8.STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
v8.CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1835_after_risk_boundary_acquisition_v11.md"
v8.PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
v8.MARKER = f"vehicle-true-variable-H-risk-boundary-acquisition-v11-{STAMP}"
v8.RNG_SEED = 202609291835
v8.TARGET_CASES = 6
v8.BRANCH_STATES_PER_CASE = 2
v8.REPEATS = 2
v8.TOTAL_EPISODES = v8.TARGET_CASES + v8.TARGET_CASES * v8.BRANCH_STATES_PER_CASE * len(v8.TRUE_HORIZONS) * v8.REPEATS
v8.CONTROL_STEP_CAP = v8.TOTAL_EPISODES * v8.MAX_STEPS

TARGET_ROLES = [
    {"group": "fresh_low_heading_low_clearance_control", "role": "boundary_low_clearance_anchor144_nearest_A", "mode": "nearest_anchor", "anchor": 144},
    {"group": "fresh_low_heading_low_clearance_control", "role": "boundary_low_clearance_anchor144_nearest_B", "mode": "nearest_anchor", "anchor": 144},
    {"group": "fresh_high_heading_long_or_medium", "role": "boundary_high_heading_long_high_stress_A", "mode": "high_stress", "anchor": 94},
    {"group": "fresh_high_heading_long_or_medium", "role": "boundary_high_heading_long_anchor94_nearest_B", "mode": "nearest_anchor", "anchor": 94},
    {"group": "fresh_high_heading_short", "role": "boundary_high_heading_short_anchor22_nearest", "mode": "nearest_anchor", "anchor": 22},
    {"group": "fresh_lower_stress_control", "role": "lower_stress_specificity_control", "mode": "low_stress", "anchor": 107},
]

class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        y = float(value)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def completed_dev_only(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError("missing prerequisite: " + rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True and obj.get("status") not in ("complete", "completed"):
        raise ContractError("prerequisite not complete: " + rel(path))
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError("forbidden access flag in prerequisite %s: %s" % (flag, rel(path)))
    return obj


def metadata_distance(a: Mapping[str, Any], b: Mapping[str, Any], scales: Mapping[str, float]) -> float:
    keys = ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]
    total = 0.0
    for k in keys:
        s = max(sf(scales.get(k), 1.0), 1e-9)
        total += ((sf(a.get(k)) - sf(b.get(k))) / s) ** 2
    return math.sqrt(total / len(keys))


def metadata_scales(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_v1_score"]:
        vals = [sf(r.get(k), float("nan")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if len(vals) <= 1:
            out[k] = 1.0
        else:
            out[k] = max(max(vals) - min(vals), 1e-9)
    return out


def extra_excluded_paths() -> List[Path]:
    return [p for p in (V8C_PROTOCOL, V8C_MANIFEST) if p.exists()]


def choose_cases_v11(bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select fresh source cases using only pre-existing metadata/outcomes.

    The target is not to cherry-pick labels (none are known for these source IDs)
    but to enrich the opened development set around risk modes that v10b could
    not separate: low-heading/low-clearance H10 catastrophes and high-heading
    ambiguous H10 savings.  All previous source_candidate_index values from
    Stage1/oracle/risk/fresh/v8c work are excluded.
    """
    metas = list((bank.get("selection") or {}).get("all_candidate_metadata") or [])
    cases = list(bank.get("candidate_cases") or [])
    if len(metas) != len(cases) or len(metas) < 64:
        raise ContractError("candidate bank dimensions invalid")

    excluded = set(v8.freeze0.selected_stage1_indices(bank))
    excluded.update(v8.freeze0.used_candidate_indices(read_json(v8.freeze0.ORACLE_PROTOCOL), read_json(v8.freeze0.ORACLE_RAW), read_json(v8.freeze0.RISK_PROTOCOL), read_json(v8.freeze0.RISK_RAW)))
    excluded.update(v8.indices_from_paths([v8.V0_PROTOCOL, v8.V1_PROTOCOL, v8.V2_PROTOCOL, v8.V0_RAW, v8.V1_RAW, v8.V2_RAW] + extra_excluded_paths()))

    thr = v8.freeze0.thresholds_from_bank(bank, metas)
    enriched: List[Dict[str, Any]] = []
    for row0 in metas:
        row = dict(row0)
        idx = si(row.get("candidate_index"), -1)
        if idx < 0 or idx >= len(cases) or idx in excluded:
            continue
        row["strict_group"] = v8.freeze0.classify(row, thr, relaxed=False)
        row["relaxed_group"] = v8.freeze0.classify(row, thr, relaxed=True)
        enriched.append(row)
    if len(enriched) < len(TARGET_ROLES):
        raise ContractError("too few eligible candidate cases after exclusion")

    all_rows_by_idx = {si(r.get("candidate_index"), -1): r for r in metas}
    scales = metadata_scales(metas)
    used: set = set()
    selected_rows: List[Dict[str, Any]] = []
    fallbacks: List[str] = []

    def candidate_pool(group: str) -> Tuple[str, List[Mapping[str, Any]]]:
        strict = [r for r in enriched if r["strict_group"] == group and si(r.get("candidate_index"), -1) not in used]
        if strict:
            return "strict", strict
        relaxed = [r for r in enriched if r["relaxed_group"] == group and si(r.get("candidate_index"), -1) not in used]
        if relaxed:
            return "relaxed", relaxed
        global_pool = [r for r in enriched if si(r.get("candidate_index"), -1) not in used]
        return "global", global_pool

    def order_role(pool: Sequence[Mapping[str, Any]], role: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        mode = str(role.get("mode"))
        anchor = all_rows_by_idx.get(si(role.get("anchor"), -1))
        if mode == "nearest_anchor" and anchor:
            return sorted(pool, key=lambda r: (metadata_distance(r, anchor, scales), -sf(r.get("stress_v1_score")), si(r.get("candidate_index"), 9999)))
        if mode == "high_stress":
            return sorted(pool, key=lambda r: (-sf(r.get("stress_v1_score")), -sf(r.get("abs_theta_r")), -sf(r.get("traj_steps")), si(r.get("candidate_index"), 9999)))
        if mode == "low_stress":
            return sorted(pool, key=lambda r: (sf(r.get("stress_v1_score")), sf(r.get("abs_theta_r")), -sf(r.get("min_reference_obstacle_clearance")), si(r.get("candidate_index"), 9999)))
        return sorted(pool, key=lambda r: (si(r.get("candidate_index"), 9999),))

    role_diagnostics: List[Dict[str, Any]] = []
    for role in TARGET_ROLES:
        mode, pool = candidate_pool(str(role["group"]))
        if not pool:
            raise ContractError("no eligible pool for role " + str(role["role"]))
        if mode != "strict":
            fallbacks.append(str(role["role"]) + "_" + mode)
        ordered = order_role(pool, role)
        chosen = dict(ordered[0])
        idx = si(chosen.get("candidate_index"), -1)
        used.add(idx)
        chosen["fresh_case_index"] = len(selected_rows)
        chosen["fresh_confirmation_group"] = role["group"]
        chosen["risk_probe_role"] = role["role"]
        chosen["selection_mode"] = mode
        chosen["selection_anchor_candidate_index"] = role.get("anchor")
        chosen["selection_boundary_mode"] = role.get("mode")
        anchor_row = all_rows_by_idx.get(si(role.get("anchor"), -1))
        chosen["selection_anchor_metadata_distance"] = metadata_distance(chosen, anchor_row, scales) if anchor_row else None
        selected_rows.append(chosen)
        role_diagnostics.append({"role": role["role"], "group": role["group"], "selection_mode": mode, "pool_size": len(pool), "selected_candidate_index": idx, "anchor": role.get("anchor"), "anchor_metadata_distance": chosen.get("selection_anchor_metadata_distance")})

    if len(selected_rows) != len(TARGET_ROLES) or len({si(r.get("candidate_index"), -1) for r in selected_rows}) != len(TARGET_ROLES):
        raise ContractError("v11 case selection count/uniqueness failed")

    selected_cases: List[Dict[str, Any]] = []
    for row in selected_rows:
        idx = si(row.get("candidate_index"), -1)
        selected_cases.append({
            "fresh_case_index": int(row["fresh_case_index"]),
            "source_candidate_index": idx,
            "fresh_confirmation_group": row["fresh_confirmation_group"],
            "risk_probe_role": row["risk_probe_role"],
            "selection_mode": row.get("selection_mode"),
            "selection_boundary_mode": row.get("selection_boundary_mode"),
            "selection_anchor_candidate_index": row.get("selection_anchor_candidate_index"),
            "selection_anchor_metadata_distance": row.get("selection_anchor_metadata_distance"),
            "strict_group": row.get("strict_group"),
            "relaxed_group": row.get("relaxed_group"),
            "theta_r": sf(row.get("theta_r")),
            "abs_theta_r": sf(row.get("abs_theta_r")),
            "traj_steps": sf(row.get("traj_steps")),
            "min_reference_obstacle_clearance": sf(row.get("min_reference_obstacle_clearance"), 0.0),
            "stress_v1_score": sf(row.get("stress_v1_score")),
            "case_snapshot_from_candidate_pool": cases[idx],
            "case_snapshot_sha256": hashlib.sha256(json.dumps(clean(cases[idx]), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
        })
    diag = {
        "excluded_source_candidate_indices_count": len(excluded),
        "eligible_after_exclusion_count": len(enriched),
        "selected_source_candidate_indices": [int(c["source_candidate_index"]) for c in selected_cases],
        "fallbacks": fallbacks,
        "target_roles": TARGET_ROLES,
        "role_diagnostics": role_diagnostics,
        "strict_pool_counts": {g: sum(1 for r in enriched if r["strict_group"] == g) for g in sorted(set(r["group"] for r in TARGET_ROLES))},
        "relaxed_pool_counts": {g: sum(1 for r in enriched if r["relaxed_group"] == g) for g in sorted(set(r["group"] for r in TARGET_ROLES))},
        "additional_exclusion_paths": [rel(p) for p in extra_excluded_paths()],
    }
    return selected_cases, diag


_ORIGINAL_BUILD_PROTOCOL = v8.build_protocol


def build_protocol_v11(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str]) -> Mapping[str, Any]:
    # Add v8c/v10b prerequisite hashes before freezing the protocol.
    for p in (V8C_PROTOCOL, V8C_MANIFEST, V8C_DONE, V10B_DONE, V10B_SUMMARY):
        if p.exists():
            input_hashes[rel(p)] = sha256(p)
    protocol = _ORIGINAL_BUILD_PROTOCOL(created, selected_cases, case_diag, input_hashes)
    protocol["protocol_id"] = f"{NAME}_preoutcome_frozen_{STAMP}"
    protocol["classification"] = "development_IMPROVED_targeted_risk_boundary_acquisition_preoutcome_not_validation_not_test"
    protocol["hypothesis"] = "v10b failed because opened-bank deployable features/data were insufficient near catastrophic/near-safe H10 boundaries. A small source-independent boundary acquisition should add informative catastrophic/noncatastrophic H10 labels from low-clearance and high-heading modes before another risk/value refit."
    protocol["before_evidence"] = [
        "v8c: shared-H15 true-H10/H15 fresh risk probe found 6/8 beneficial rows but 2 catastrophic low-clearance H10 rows; oracle saving existed while fixed H10 failed the physical gate.",
        "v9: opened-bank trees over fresh_v0/v1/v2/v8c had oracle opportunity but no deployable/all-family selector passed weak/strong LOBO gates with v8c included.",
        "v10b: fast deployable kNN/poly risk-value refit found zero global/nested pass; nested selection kept 8.43% decision saving but had 2 catastrophic H10 false positives; conservative global configs collapsed below 5% saving.",
    ]
    protocol["v11_targeting"] = {
        "target_roles": TARGET_ROLES,
        "case_selection_rule": "metadata-only fresh-source selection from stress-v1 bank, excluding all previously used source_candidate_index values and targeting prior risk-boundary modes by group/anchor metadata; no H10/H15 outcome for selected source IDs is observed before selection",
        "state_selection_rule": "two branch states per selected source are chosen from H15 Stage-A traces before any Stage-B H10 outcome using the same H15-trace risk score/window logic as v8c",
        "primary_terminal_profile": v8.PRIMARY_TERMINAL_PROFILE,
        "intended_next_use": "development-only risk/value representation or terminal/objective refit; not direct validation64 rollout",
    }
    protocol.setdefault("execution_repairs", []).append({
        "repair": "v11 wrapper over audited v8 runner with flexible selected-state count and source-independent boundary-targeted case selection",
        "target_cases": v8.TARGET_CASES,
        "branch_states_per_case": v8.BRANCH_STATES_PER_CASE,
        "repeats": v8.REPEATS,
        "total_episodes_exact": v8.TOTAL_EPISODES,
        "control_step_upper_bound": v8.CONTROL_STEP_CAP,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    protocol["source_hashes"] = {
        "base_v8_source": sha256(BASE_V8_SOURCE),
        "v11_wrapper_source": sha256(SOURCE),
    }
    write_json(v8.PROTOCOL, protocol)
    return protocol


def trace_risk_score(row: Mapping[str, Any], n: int) -> float:
    scorer = getattr(v8.engine.fresh0, "trace_risk_score", None)
    if scorer is None:
        raise ContractError("fresh0.trace_risk_score missing; refusing to change state-selection score")
    return float(scorer(row, n))


def select_stage_a_states_v11(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cfg = (protocol.get("stage_A_h15_trace_state_selection") or {})
    bounds = cfg.get("eligible_step_bounds") or {}
    windows = list(cfg.get("windows") or [])
    per_case = si(cfg.get("branch_states_per_case"), len(windows))
    if per_case <= 0 or len(windows) < per_case:
        raise ContractError("invalid v11 branch state/window contract")
    windows = windows[:per_case]
    min_step = si(bounds.get("min_step"), 8)
    max_step_abs = si(bounds.get("max_step"), 90)
    reserve = si(bounds.get("reserve_terminal_margin_steps"), 3)
    selected: List[Dict[str, Any]] = []
    for ep in stage_a_episodes:
        trace_path = ROOT / str(ep.get("path")) / "trace.json"
        trace = read_json(trace_path)
        n = len(trace)
        if n < min_step + reserve + 2:
            raise ContractError("Stage-A trace too short for v11 state selection: " + str(ep.get("state_id")))
        chosen_steps: List[int] = []
        for window in windows:
            slot = si(window.get("slot"), len(chosen_steps))
            lo = max(min_step, int(math.floor(n * sf(window.get("low_fraction"), 0.1))))
            hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * sf(window.get("high_fraction"), 0.8))))
            candidates = [r for r in trace if lo <= si(r.get("step"), -1) <= hi]
            if not candidates:
                candidates = [r for r in trace if min_step <= si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
            if slot == 1 and chosen_steps:
                separated = [r for r in candidates if all(abs(si(r.get("step"), -1) - s) >= 8 for s in chosen_steps)]
                if separated:
                    candidates = separated
            if not candidates:
                raise ContractError("no eligible v11 Stage-A state for " + str(ep.get("state_id")))
            row = max(candidates, key=lambda r: trace_risk_score(r, n))
            step = si(row.get("step"), -1)
            chosen_steps.append(step)
            base_state_id = f"fresh_case{si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{window.get('name', 'window')}"
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": si(ep.get("fresh_case_index"), -1),
                "source_candidate_index": si(ep.get("source_candidate_index"), -1),
                "fresh_confirmation_group": ep.get("fresh_confirmation_group"),
                "branch_state_slot": slot,
                "window": window.get("name"),
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(row.get("previous_state") or row.get("state")),
                "initial_observation_from_h15_trace": copy.deepcopy(row.get("observation") or []),
                "h15_trace_episode_path": ep.get("path"),
                "stage_a_trace_risk_score": trace_risk_score(row, n),
                "selection_rule": "v11 boundary acquisition: selected from H15 trace before any Stage-B H10/H15 branch outcome; scoring/window logic unchanged from v8c",
            })
    expected = len(stage_a_episodes) * per_case
    if len(selected) != expected:
        raise ContractError("expected %d selected branch states under v11 protocol, got %d" % (expected, len(selected)))
    return selected


def fmt(v: Any) -> str:
    return "NA" if v is None else "%.6g" % float(v)


def write_summary_v11(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle true-variable-H targeted risk-boundary acquisition v11",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only source-independent boundary data acquisition; no validation64, no sealed test, no training/refit.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['total_episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- States: `{len(a['state_rows'])}`; H10-beneficial `{a['beneficial_count']}`; catastrophic H10 rows `{len(a['catastrophic_rows'])}`; early catastrophics `{len(a['early_catastrophic_rows'])}`; risk-role catastrophics `{len(a['risk_catastrophic_rows'])}`.",
        f"- Aggregate: `{a['aggregate']}`.",
        f"- Decision: {a['decision']}",
        "",
        "## Selected fresh source cases (pre-outcome)",
        "",
        "| fresh_case | source_candidate_index | group | role | selection | anchor | anchor distance | theta | traj | clearance | stress |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in raw.get("selected_cases") or []:
        lines.append("| %d | %d | `%s` | `%s` | `%s` | %s | %s | %.4g | %.4g | %.4g | %.4g |" % (
            int(c.get("fresh_case_index", -1)), int(c.get("source_candidate_index", -1)), c.get("fresh_confirmation_group"), c.get("risk_probe_role"), c.get("selection_mode"), str(c.get("selection_anchor_candidate_index")), fmt(c.get("selection_anchor_metadata_distance")), sf(c.get("theta_r")), sf(c.get("traj_steps")), sf(c.get("min_reference_obstacle_clearance")), sf(c.get("stress_v1_score"))))
    lines += [
        "",
        "## Per-state rows",
        "",
        "| state | role | slot | H10 ben | H10 cat | physΔ H10-H15 | H10 phys | H15 phys | H10 dec | H15 dec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in a["state_rows"]:
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s | %s | %s |" % (
            r["base_state_id"], r.get("risk_probe_role"), r.get("slot_name"), r.get("h10_beneficial_vs_h15"), r.get("h10_catastrophic_vs_h15"), fmt(r.get("physical_delta_h10_minus_h15")), fmt(r["h10"].get("physical")), fmt(r["h15"].get("physical")), fmt(r["h10"].get("decision_sum_s")), fmt(r["h15"].get("decision_sum_s"))))
    lines += [
        "",
        "## Interpretation",
        "",
        "This diagnostic deliberately adds development-only labels near risk boundaries that simple deployable risk/value selectors failed to separate in v9/v10b.  It is not validation64, not sealed final test, not ORIGINAL SAC, and not a success claim.  The next step should be a bounded risk/value representation or terminal/objective refit that includes these rows, with nested/opened-bank gates before any unused confirmation.",
        "",
        f"Preoutcome protocol: `{raw['protocol']['json']}`.",
        f"Backup request: `{raw['backup_request_after_run']}`.",
    ]
    (v8.RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs_v11(raw: Mapping[str, Any]) -> None:
    block = f"""<!-- {v8.MARKER} -->
## 2026-09-29 vehicle true-variable-H targeted risk-boundary acquisition v11

UTC: {raw['created_utc']}. Development-only source-independent risk-boundary acquisition completed: {raw['budget_actual']['episodes']} episodes, {raw['budget_actual']['control_steps']} control steps, validation64 closed, sealed test closed, no training/refit. Catastrophic H10 rows: {len(raw['analysis']['catastrophic_rows'])}; early catastrophics: {len(raw['analysis']['early_catastrophic_rows'])}; beneficial H10 rows: {raw['analysis']['beneficial_count']}. Decision: {raw['analysis']['decision']}. Artifacts: `{rel(v8.RUN_DIR / 'summary.md')}`, `{rel(v8.RUN_DIR / 'raw.json')}`, `{rel(v8.RUN_DIR / 'completed.json')}`.
"""
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if v8.MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg.exists():
        old = reg.read_text(encoding="utf-8", errors="replace")
        if v8.MARKER not in old[-80000:]:
            reg.write_text(old.rstrip() + f"\n{raw['created_utc']},{NAME},development_targeted_risk_boundary_acquisition,true_variable_H_shared_h15_terminal,{raw['budget_actual']['episodes']},{raw['budget_actual']['control_steps']},0,0,0,False,{rel(v8.RUN_DIR / 'completed.json')},{v8.MARKER}\n", encoding="utf-8")


def postprocess_success() -> None:
    raw_path = v8.RUN_DIR / "raw.json"
    done_path = v8.RUN_DIR / "completed.json"
    if not raw_path.exists() or not done_path.exists():
        return
    raw = read_json(raw_path)
    created = dt.datetime.now(dt.timezone.utc)
    req = v8.BACKUP_DIR / f"REQUEST_BACKUP_AFTER_RISK_BOUNDARY_ACQUISITION_V11_{STAMP}.json"
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup v11 targeted risk-boundary acquisition outputs before any risk/value refit or further simulations",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "episodes": (raw.get("budget_actual") or {}).get("episodes"),
        "control_steps": (raw.get("budget_actual") or {}).get("control_steps"),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(v8.RUN_DIR), rel(v8.PROTOCOL), rel(v8.STATE), rel(v8.CONTINUE_STATE), rel(SOURCE), rel(req)],
    })
    raw["backup_request_after_run"] = rel(req)
    raw["method"] = NAME
    raw["postprocess_v11_backup_request_created_utc"] = created.isoformat()
    write_json(raw_path, raw)
    write_summary_v11(raw)
    v8.STATE.parent.mkdir(parents=True, exist_ok=True)
    v8.STATE.write_text((v8.RUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    v8.CONTINUE_STATE.write_text("# Continue state after risk-boundary acquisition v11\n\nUTC: %s\n\nDecision: %s\n\nBudgets: %s\n\nNext: backup these artifacts, then run a bounded v12 risk/value representation refit including v11 rows; do not open validation64 or sealed test.\n" % (created.isoformat(), (raw.get("analysis") or {}).get("decision"), json.dumps(raw.get("budget_actual"), sort_keys=True)), encoding="utf-8")
    append_docs_v11(raw)
    done = read_json(done_path)
    files = [p for p in v8.RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, v8.PROTOCOL, v8.STATE, v8.CONTINUE_STATE, req]
    done.update({
        "marker": v8.MARKER,
        "backup_request": rel(req),
        "summary": rel(v8.RUN_DIR / "summary.md"),
        "raw": rel(raw_path),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {rel(p): sha256(p) for p in sorted(set(files), key=lambda q: rel(q)) if p.exists()},
    })
    write_json(done_path, done)


# Apply patches used by v8.run().
v8.choose_cases = choose_cases_v11
v8.build_protocol = build_protocol_v11
v8.engine.fresh0.select_stage_a_states = select_stage_a_states_v11
v8.write_summary = write_summary_v11
v8.append_docs = append_docs_v11


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-risk-boundary-v11", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_risk_boundary_v11:
        raise ContractError("requires --run and explicit v11 development risk-boundary acknowledgement")
    for p in (V8C_DONE, V10B_DONE):
        completed_dev_only(p)
    try:
        rc = v8.run(["--run", "--backup-verified-commit", args.backup_verified_commit, "--i-accept-development-risk-probe-v8"])
        if rc == 0:
            postprocess_success()
        return rc
    except Exception as exc:
        v8.RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_json(v8.RUN_DIR / "failed.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False})
        v8.CONTINUE_STATE.parent.mkdir(parents=True, exist_ok=True)
        v8.CONTINUE_STATE.write_text("# Continue state after v11 risk-boundary acquisition failure\n\n" + json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc), "failed_artifact": rel(v8.RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False, "next_action": "inspect failure and partial progress before any rerun; if Stage-B partially ran, backup first and account budgets"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"failed": type(exc).__name__, "message": str(exc), "failed_artifact": rel(v8.RUN_DIR / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
