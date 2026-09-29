#!/usr/bin/env python3
"""True variable-dimension MPC horizon smoke for the v1e case-5 positives.

This IMPROVED development diagnostic follows the v1e/case5 evidence:
H10 repeatedly improved local physical continuation cost, but the deployed AHMPC
implementation still solved a fixed-size H50 NLP (same opt_x size for executed
H10 and H15) and did not show measured decision-time savings.  The next
scientific question is therefore implementation-level, not another label-density
sweep: can controllers constructed with true n_horizon=10 and n_horizon=15 reduce
optimizer dimension and measured solver/decision time on the same development
states without a safety regression?

Modes:
  * --dry-run: no simulation, no TF/legacy import. Freeze the small diagnostic
    protocol from existing v1e case5 artifacts and write a backup request.
  * --run-smoke: after verified external backup, run only the true-H branch
    smoke: two case5 positive states x H10/H15 using H15-common terminal weights
    from direct branch initialisation. Historical validation64 and sealed final
    test remain closed; no training/refit/candidate generation.

The run-smoke branch starts from the saved branch state with the case TVP suffix.
This intentionally tests the true smaller-H controller construction/timing path;
it is not a fresh validation claim and is not assumed identical to the previous
fixed-size prefix-warm-start trajectories.
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
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
REPRO_DIR = ROOT / "experiments/bohn2021_reproduction"
for _p in (AWS_DIR, REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# These imports are metadata-only at module import time. Legacy/TF imports occur
# only inside --run-smoke through runtime.imports()/v1d.import_legacy_modules().
import vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_runner as v1d  # noqa:E402
import vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_runner as v1e_v0b  # noqa:E402

NAME = "vehicle_true_variable_horizon_case5_smoke_v0"
STAMP = "20260929T0640Z"
SOURCE = Path(__file__).resolve()
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

PARENT_CASE5_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z"
PARENT_CASE5_RAW = PARENT_CASE5_DIR / "raw.json"
PARENT_CASE5_DONE = PARENT_CASE5_DIR / "completed.json"
PARENT_CASE5_SUMMARY = PARENT_CASE5_DIR / "summary.md"
PARENT_CASE5_TIMING_DECOMP_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_timing_decomposition_postdiagnostic_v0_20260929T0625Z/raw.json"
PARENT_CASE5_TIMING_DECOMP_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_timing_decomposition_postdiagnostic_v0_20260929T0625Z/completed.json"
SOURCE_AUDIT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_mpc_horizon_source_audit_v0_20260929T0630Z/completed.json"
SOURCE_AUDIT_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_mpc_horizon_source_audit_v0_20260929T0630Z/summary.md"
STAGE1_BANK = v1e_v0b.v0.STAGE1_BANK
STAGE1_DONE = v1e_v0b.v0.STAGE1_DONE

PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/{NAME}_frozen_{STAMP}.json"
DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_dryrun_{STAMP}"
SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_run_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/{NAME}_dryrun_{STAMP}.md"
STATE_SMOKE = ROOT / f"research_artifacts/aws_state/{NAME}_run_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_SMOKE = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_CASE5_SMOKE_V0_RUN_{STAMP}.json"

MARKER_DRYRUN = f"vehicle-true-variable-horizon-case5-smoke-v0-dryrun-{STAMP}"
MARKER_SMOKE = f"vehicle-true-variable-horizon-case5-smoke-v0-run-{STAMP}"

TASK = "vehicle"
TRUE_HORIZONS = [10, 15]
TERMINAL_MODE = "h15_common_terminal"
MAX_BRANCH_STEPS = 150
EXPECTED_TARGET_STATE_IDS = [
    "v1e_t04_case05_cand148_b053_middle",
    "v1e_t05_case05_cand148_b054_late",
]
EPISODES_EXACT = len(EXPECTED_TARGET_STATE_IDS) * len(TRUE_HORIZONS)
CONTROL_STEP_UPPER = EPISODES_EXACT * MAX_BRANCH_STEPS
STATE_DISTANCE_TOL = 1e-5
MIN_DIMENSIONAL_SOLVER_SAVING = 0.15


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


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    return v1d.parse_time(value)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(SOURCE.stat().st_mtime, dt.timezone.utc)


def completed_ok(path: Path, check_hashes: bool = False) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError(f"sealed-test flag is not false in {rel(path)}")
    if obj.get("validation64_bank_opened") not in (False, None):
        raise ContractError(f"validation64 flag is not false in {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError(f"hash mismatch from {rel(path)}: {name}")
    return obj


def file_mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def finite_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * q
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return xs[lo] if lo == hi else xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "median": pct(0.5), "mean": float(math.fsum(xs) / len(xs)), "p95": pct(0.95), "max": xs[-1], "sum": float(math.fsum(xs))}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def arr(value: Any) -> np.ndarray:
    try:
        if hasattr(value, "full"):
            return np.asarray(value.full(), dtype=float).reshape(-1)
        if hasattr(value, "cat"):
            return np.asarray(value.cat, dtype=float).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def state_clean(state: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("x", "y", "theta"):
        val = state.get(key)
        if isinstance(val, list) and val:
            val = val[0]
        out[key] = float(val)
    return out


def state_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    aa = state_clean(a); bb = state_clean(b)
    dtheta = math.atan2(math.sin(aa["theta"] - bb["theta"]), math.cos(aa["theta"] - bb["theta"]))
    return float(math.hypot(aa["x"] - bb["x"], aa["y"] - bb["y"]) + 0.5 * abs(dtheta))


def step_physical(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("performance"), 0.0) + safe_float(row.get("constraint"), 0.0)


def step_total(row: Mapping[str, Any]) -> float:
    return step_physical(row) + safe_float(row.get("compute"), 0.0)


def values_tail_sum(trace: Sequence[Mapping[str, Any]], which: str) -> float:
    if which == "physical":
        return float(math.fsum(step_physical(r) for r in trace))
    if which == "total":
        return float(math.fsum(step_total(r) for r in trace))
    raise ValueError(which)


def verify_inputs_for_dryrun() -> Dict[str, Any]:
    required = [PARENT_CASE5_RAW, PARENT_CASE5_DONE, PARENT_CASE5_SUMMARY, PARENT_CASE5_TIMING_DECOMP_RAW, PARENT_CASE5_TIMING_DECOMP_DONE, SOURCE_AUDIT_DONE, SOURCE_AUDIT_SUMMARY, STAGE1_BANK, STAGE1_DONE]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise ContractError("missing required input(s): " + ", ".join(missing))
    parent_done = completed_ok(PARENT_CASE5_DONE, check_hashes=False)
    timing_done = completed_ok(PARENT_CASE5_TIMING_DECOMP_DONE, check_hashes=False)
    source_done = completed_ok(SOURCE_AUDIT_DONE, check_hashes=False)
    stage1_done = completed_ok(Path(STAGE1_DONE), check_hashes=False)
    parent_raw = read_json(PARENT_CASE5_RAW)
    headline = parent_done.get("headline") or {}
    if int(headline.get("material_pair_count", -1)) != 12:
        raise ContractError("parent case5 run did not record 12/12 material pairs")
    if headline.get("pass_to_next_design_consideration") is not False:
        raise ContractError("parent case5 timing gate unexpectedly passed; this implementation smoke may not be the right next action")
    t_head = timing_done.get("headline") or {}
    if t_head.get("fixed_nlp_size_across_h10_h15") is not True:
        raise ContractError("timing decomposition did not establish fixed-size H10/H15 trace evidence")
    s_head = source_done.get("headline") or {}
    if s_head.get("sealed_test_accessed") is not False or s_head.get("validation64_bank_opened") is not False:
        raise ContractError("source audit access flags invalid")
    if s_head.get("previous_fixed_nlp_trace_evidence") is not True:
        raise ContractError("source audit headline lacks prior fixed-size trace evidence")
    if stage1_done.get("sealed_test_accessed") is not False:
        raise ContractError("stage1 bank marker access flags invalid")
    return {"parent_done": parent_done, "parent_raw": parent_raw, "timing_done": timing_done, "source_done": source_done, "stage1_done": stage1_done}


def extract_target_specs(parent_raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    episodes = parent_raw.get("branch_episodes") or []
    selected: List[Dict[str, Any]] = []
    for sid in EXPECTED_TARGET_STATE_IDS:
        matches = [e for e in episodes if str(e.get("state_id")) == sid and str(e.get("terminal_mode")) == TERMINAL_MODE and int(e.get("branch_horizon", -1)) == 15]
        if not matches:
            raise ContractError(f"no H15/common-terminal parent branch episode for {sid}")
        matches = sorted(matches, key=lambda e: (int(e.get("diagnostic_repeat", 999)), int(e.get("execution_index", 999))))
        e = matches[0]
        if not e.get("branch_reached"):
            raise ContractError(f"parent branch not reached for {sid}")
        branch_state = e.get("branch_previous_state")
        if not isinstance(branch_state, Mapping):
            raise ContractError(f"missing branch state for {sid}")
        selected.append({
            "state_id": sid,
            "case": int(e.get("case")),
            "source_candidate_index": int(e.get("source_candidate_index", -1)),
            "selection_group": e.get("selection_group"),
            "case_role": e.get("case_role"),
            "branch_step": int(e.get("branch_step")),
            "branch_previous_state": state_clean(branch_state),
            "parent_h15_common_terminal_episode_path": e.get("path"),
            "parent_h15_common_terminal_decision_sum_s": safe_float((e.get("decision_timing_s") or {}).get("sum"), 0.0),
            "parent_h15_common_terminal_solver_sum_s": safe_float((e.get("solver_attempt_timing_s") or {}).get("sum"), 0.0),
            "parent_h15_common_terminal_continuation_physical": safe_float(e.get("continuation_physical_constraint_cost_from_branch"), 0.0),
            "parent_h15_common_terminal_success": bool(e.get("success")),
            "parent_h15_common_terminal_solver_failure_steps": int(e.get("solver_failure_steps", 0)),
        })
    if len(selected) != len(EXPECTED_TARGET_STATE_IDS):
        raise ContractError("target extraction count mismatch")
    if sorted({x["case"] for x in selected}) != [5]:
        raise ContractError("expected all true-variable targets to be case 5")
    return selected


def fixed_size_reference_from_existing() -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": rel(PARENT_CASE5_TIMING_DECOMP_RAW), "available": False}
    try:
        raw = read_json(PARENT_CASE5_TIMING_DECOMP_RAW)
    except Exception as exc:
        out["error"] = repr(exc)
        return out
    # Keep this robust to schema changes: prefer explicit headline/analysis keys,
    # then fall back to a compact recursive scan for opt_x sizes.
    out["available"] = True
    out["completed"] = rel(PARENT_CASE5_TIMING_DECOMP_DONE)
    for key in ("headline", "analysis", "summary"):
        if isinstance(raw.get(key), Mapping):
            obj = raw[key]
            for subkey in ("observed_opt_x_sizes_by_horizon", "opt_x_sizes_by_horizon", "opt_x_sizes", "fixed_nlp_size_across_h10_h15", "solver_decision_pair_diff_corr", "median_solver_fraction_of_decision_time", "overall_total_relative_decision_saving", "overall_branch_relative_decision_saving"):
                if subkey in obj:
                    out[subkey] = obj[subkey]
    sizes: Dict[str, List[int]] = {}
    def scan(obj: Any) -> None:
        if isinstance(obj, Mapping):
            h_val = obj.get("horizon") or obj.get("branch_horizon") or obj.get("executed_horizon")
            size_val = obj.get("opt_x_size") or obj.get("opt_x_num_size") or obj.get("optimizer_vector_size")
            if h_val is not None and size_val is not None:
                try:
                    sizes.setdefault(str(int(h_val)), []).append(int(size_val))
                except Exception:
                    pass
            for v in obj.values():
                scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)
    scan(raw)
    if sizes:
        out["recursive_opt_x_sizes_by_horizon"] = {k: sorted(set(v)) for k, v in sorted(sizes.items())}
    return out


def write_protocol(created: dt.datetime, targets: Sequence[Mapping[str, Any]], source_info: Mapping[str, Any]) -> None:
    schedule: List[Dict[str, Any]] = []
    execution = 0
    # Alternate order within each state to reduce monotonic drift confounding in
    # the eventual four-episode smoke while keeping the pair local.
    for i, target in enumerate(targets):
        order = [10, 15] if i % 2 == 0 else [15, 10]
        for h in order:
            schedule.append({
                "execution_index": execution,
                "state_id": target["state_id"],
                "case": int(target["case"]),
                "source_candidate_index": int(target["source_candidate_index"]),
                "branch_step": int(target["branch_step"]),
                "true_mpc_n_horizon": int(h),
                "commanded_horizon": int(h),
                "terminal_mode": TERMINAL_MODE,
                "initialization": "direct_branch_state_with_shifted_case_tvp_cold_mpc_guess",
            })
            execution += 1
    protocol = {
        "protocol_id": f"{NAME}_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_dimension_mpc_smoke_not_validation_not_final_test",
        "hypothesis": "The current fixed-size AHMPC masks shorter horizons inside an H50 NLP; constructing separate MPC controllers with true n_horizon=10 and n_horizon=15 should reduce opt_x dimension and may recover measured solver/decision-time savings on the two stable case5 H10 positives.",
        "why_now": "v1e repeated case5 positives retained physical gains but failed measured timing; timing decomposition showed identical opt_x sizes {'10':[862], '15':[862]} and solver-dominated timing. This discriminates implementation compute path before selector/refit or more label-density sweeps.",
        "targets": list(targets),
        "schedule": schedule,
        "true_horizon_construction": {
            "implemented_in_run_smoke": "instantiate a fresh vehicle environment/controller for each arm with config_kw mpc.params.n_horizon equal to the commanded horizon, instead of AHMPC hend masking in the default H50 controller",
            "terminal_weights": "H15/common terminal weights for both H10 and H15 arms",
            "branch_initialization": "cold branch reset from saved branch_previous_state plus original case TVP suffix; not a formal continuation-validation replay and not used for final claims",
        },
        "budgets": {"development_branch_episodes_exact": EPISODES_EXACT, "development_control_step_upper_bound": CONTROL_STEP_UPPER, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "decision_rules": {
            "dimension_gate": "observed opt_x size for true H10 must be smaller than true H15",
            "timing_gate_for_broadening": f"median relative H10-vs-H15 measured solver-time saving >= {MIN_DIMENSIONAL_SOLVER_SAVING}; decision-time saving is reported separately",
            "safety_gate": "no success, constraint, initial/final solver-failure regression of H10 versus H15 within these two development states",
            "if_pass": "treat true variable-dimension controller cache as viable; next broaden to a small paired development block before any selector/value-learning experiment",
            "if_fail": "do not pursue shorter-H-as-speed selector in this stack; pivot to terminal/value/modeling or scenario-opportunity design",
        },
        "source_artifacts": {
            "parent_case5_run_raw": rel(PARENT_CASE5_RAW),
            "parent_case5_run_completed": rel(PARENT_CASE5_DONE),
            "timing_decomposition_raw": rel(PARENT_CASE5_TIMING_DECOMP_RAW),
            "source_audit_completed": rel(SOURCE_AUDIT_DONE),
            "stage1_bank": rel(Path(STAGE1_BANK)),
        },
        "source_hashes": {rel(p): sha256(p) for p in [SOURCE, PARENT_CASE5_RAW, PARENT_CASE5_DONE, PARENT_CASE5_TIMING_DECOMP_RAW, PARENT_CASE5_TIMING_DECOMP_DONE, SOURCE_AUDIT_DONE, Path(STAGE1_BANK), Path(STAGE1_DONE)] if p.exists()},
        "source_info": source_info,
        "access_rules": {"development_only": True, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "requires_verified_external_backup_after_dryrun_before_smoke": True},
    }
    write_json(PROTOCOL_JSON, protocol)


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle true variable-horizon case5 smoke v0 dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no control steps, no candidate resets, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Evidence used",
        "",
        "- v1e repeated case5 diagnostic retained the local physical-control opportunity but failed the measured timing gate.",
        "- Timing decomposition showed fixed optimizer dimension for H10 and H15 under the current AHMPC mask path and solver-dominated decision time.",
        "- Source audit supports a targeted instrumentation/implementation smoke rather than another unchanged label-density sweep.",
        "",
        "## Frozen next smoke",
        "",
        f"- Targets: `{raw['target_state_ids']}` (two development case5 H15-common-terminal positives).",
        f"- Episodes: `{raw['planned_episodes_exact']}` true-H branch episodes; control-step cap `{raw['planned_control_step_upper_bound']}`.",
        f"- True controller dimensions: construct separate controllers with `mpc.params.n_horizon` in `{TRUE_HORIZONS}`; no AHMPC H50 masking for the test arms.",
        f"- Protocol: `{raw['protocol_json']}`.",
        "",
        "## Decision",
        "",
        "Run the smoke only after a verified external backup covers this source/protocol/dry-run. If true H10 does not reduce opt_x dimension and measured solver time without safety loss, pivot away from shorter-H-as-speed selector design.",
        "",
        f"Backup request before smoke: `{raw['backup_request_before_smoke']}`.",
    ]
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    done_path = DRYRUN_DIR / "completed.json"
    if done_path.exists():
        done = completed_ok(done_path, check_hashes=True)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if DRYRUN_DIR.exists() and any(p.name != "run.lock" for p in DRYRUN_DIR.iterdir()):
        raise ContractError(f"partial dry-run output exists; inspect first: {rel(DRYRUN_DIR)}")
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    inputs = verify_inputs_for_dryrun()
    targets = extract_target_specs(inputs["parent_raw"])
    fixed_ref = fixed_size_reference_from_existing()
    source_info = {"parent_case5_headline": inputs["parent_done"].get("headline"), "timing_decomposition_headline": inputs["timing_done"].get("headline"), "source_audit_headline": inputs["source_done"].get("headline"), "fixed_size_reference": fixed_ref}
    write_protocol(created, targets, source_info)
    write_json(REQUEST_BACKUP_BEFORE_SMOKE, {
        "requested_utc": created.isoformat(),
        "reason": "backup true variable-dimension case5 smoke runner/protocol/dry-run before any true-H branch simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "planned_development_branch_episodes_exact": EPISODES_EXACT,
        "planned_development_control_step_upper_bound": CONTROL_STEP_UPPER,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(SOURCE), rel(PROTOCOL_JSON), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(PARENT_CASE5_DIR), rel(PARENT_CASE5_TIMING_DECOMP_RAW), rel(SOURCE_AUDIT_DONE.parent), rel(REQUEST_BACKUP_BEFORE_SMOKE)],
    })
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": f"{NAME}_dryrun",
        "classification": "development_no_simulation_true_variable_horizon_smoke_readiness",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "protocol_json": rel(PROTOCOL_JSON),
        "target_state_ids": [t["state_id"] for t in targets],
        "targets": targets,
        "planned_episodes_exact": EPISODES_EXACT,
        "planned_control_step_upper_bound": CONTROL_STEP_UPPER,
        "true_horizons": TRUE_HORIZONS,
        "terminal_mode": TERMINAL_MODE,
        "source_info": source_info,
        "source_hashes": {rel(p): sha256(p) for p in [SOURCE, PROTOCOL_JSON, PARENT_CASE5_RAW, PARENT_CASE5_DONE, PARENT_CASE5_TIMING_DECOMP_RAW, PARENT_CASE5_TIMING_DECOMP_DONE, SOURCE_AUDIT_DONE, Path(STAGE1_BANK), Path(STAGE1_DONE)] if p.exists()},
        "backup_request_before_smoke": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "next_action": "after verified backup, run --run-smoke with legacy interpreter and the frozen protocol; then decide whether true variable-dimension H10 warrants a broader development block",
    }
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text((DRYRUN_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-29 vehicle true variable-horizon case5 smoke v0 dry-run

UTC: {created.isoformat()}. Metadata-only dry-run froze a four-episode development smoke to test true variable-dimension MPC controllers on the two v1e case5 H15-common-terminal positive states. Planned smoke: {EPISODES_EXACT} direct branch episodes, cap {CONTROL_STEP_UPPER} control steps, true `mpc.params.n_horizon` in {TRUE_HORIZONS}, no candidate resets, no training/refit, no validation64/sealed-test access. It is motivated by the fixed-size H50 AHMPC timing bottleneck and replaces another unchanged v1c/v1d/v1e label-density sweep. Smoke is blocked until verified backup covers `{rel(REQUEST_BACKUP_BEFORE_SMOKE)}` plus source/protocol/dry-run artifacts.
""", MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL_JSON, STATE_DRYRUN, REQUEST_BACKUP_BEFORE_SMOKE, PARENT_CASE5_RAW, PARENT_CASE5_DONE, PARENT_CASE5_TIMING_DECOMP_RAW, PARENT_CASE5_TIMING_DECOMP_DONE, SOURCE_AUDIT_DONE]
    write_json(done_path, {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "candidate_pool_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "planned_development_branch_episodes_exact": EPISODES_EXACT,
        "planned_development_control_step_upper_bound": CONTROL_STEP_UPPER,
        "backup_required_before_smoke": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "headline": {"true_variable_horizon_smoke_ready": True, "target_state_count": len(targets), "planned_episodes_exact": EPISODES_EXACT, "train_or_refit_now": False, "smoke_blocked_until_verified_backup": True},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(done_path), "summary": rel(DRYRUN_DIR / "summary.md"), "protocol": rel(PROTOCOL_JSON), "target_state_ids": [t["state_id"] for t in targets], "planned_episodes_exact": EPISODES_EXACT, "planned_control_step_upper_bound": CONTROL_STEP_UPPER, "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE)}, sort_keys=True), flush=True)
    return 0


def make_true_horizon_env(true_horizon: int) -> Any:
    from runtime import ART, imports  # legacy import path; imported only in --run-smoke.
    Env, _, _ = imports()

    class ReconstructedEnv(Env):
        def _get_variable_value(self, var):
            if var["type"] == "tvp":
                val = self.control_system.tvps[var["name"]].get_values(self.control_system._step_count)
                return float(np.asarray(val).reshape(-1)[0])
            return super()._get_variable_value(var)

        def get_observation(self):
            obs = super().get_observation()
            obs = obs.astype(float).copy()
            xy = obs[:2].copy()
            obs[3:5] = (obs[3:5] - xy) / 5.0
            for j in range(3):
                obs[5 + 3 * j:7 + 3 * j] = (obs[5 + 3 * j:7 + 3 * j] - xy) / 10.0
            obs[:2] /= 30.0
            obs[2] /= np.pi
            return obs

        def get_reward(self, rew_expr=None, done=False, info=None):
            scope = {"np": np, "done": int(done)}
            for v in self.config["environment"]["reward"]["variables"]:
                value = self._get_variable_value(v)
                scope[v["name"]] = float(np.asarray(value).reshape(-1)[0])
            expression = rew_expr if rew_expr is not None else self.config["environment"]["reward"]["expression"]
            return float(eval(expression, {"__builtins__": {}}, scope))

        def reset(self, **kwargs):
            self.steps_count = 0
            return super().reset(**kwargs)

        def step(self, action):
            hmax = int(self.config["mpc"]["params"]["n_horizon"])
            horizon = int(np.clip(np.rint(action[0]), 1, hmax))
            obs, reward, done, info = super().step(np.array([float(horizon)]))
            mpc = self.control_system.controller.mpc
            info["data"]["mpc_rewards"] = float(mpc.lterm_fun(
                mpc.opt_p_num["_x0"], mpc.opt_x_num_unscaled["_u", 0, 0],
                mpc.opt_x_num_unscaled["_z", 1, 0, -1],
                mpc.opt_p_num["_tvp", 0], mpc.opt_p_num["_p", 0]))
            info["mpc_avg_stage_cost"] = info["data"]["mpc_rewards"]
            ctrl = self.control_system.controller
            collided = any(ctrl.get_obj_distance(self.control_system.current_state, j) <= ctrl.obj_data[j]["r"][0]
                           for j in range(ctrl.n_objects))
            if collided:
                old = info.get("reward/constraint", 0.0)
                penalty = 2 * (self.max_steps - self.steps_count)
                reward += old - penalty
                info["reward/constraint"] = penalty
                info["termination"] = "constraint"
                done = True
            info["executed_horizon"] = horizon
            info["solver_success"] = bool(self.control_system.controller.mpc.solver_stats.get("success", False))
            return obs, reward, done, info

    env = ReconstructedEnv(str(ART / "configs" / "vehicle.json"), config_kw={"mpc": {"params": {"n_horizon": int(true_horizon)}}})
    env.seed(0)
    env.action_space.seed(0)
    return env


def shift_case_tvp(case: Mapping[str, Any], branch_step: int) -> Dict[str, Any]:
    tvp = case.get("tvp") or {}
    if not isinstance(tvp, Mapping) or not tvp:
        raise ContractError("case lacks tvp values for branch initialization")
    shifted: Dict[str, Any] = {}
    for name, values in tvp.items():
        if not isinstance(values, list) or len(values) <= branch_step:
            raise ContractError(f"tvp {name} too short for branch_step {branch_step}")
        shifted[str(name)] = copy.deepcopy(values[branch_step:])
    return shifted


def reset_env_to_branch(env: Any, case: Mapping[str, Any], target: Mapping[str, Any], measured: List[Dict[str, float]], ep_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    # First run the upstream reset once to initialise goal/trajectory/object-noise
    # attributes consistently with the saved case. This warmup is recorded but is
    # not part of the branch episode/control-step budget.
    reset_start = time.perf_counter()
    obs0 = env.reset(**copy.deepcopy(dict(case)))
    reset_s = time.perf_counter() - reset_start
    warmup_measured = measured.pop() if measured else {}
    ctrl = env.control_system.controller
    saved_noise = copy.deepcopy(getattr(ctrl, "object_noise_seed", None))
    saved_goal_x = copy.deepcopy(getattr(ctrl, "goal_x", None))
    saved_goal_y = copy.deepcopy(getattr(ctrl, "goal_y", None))
    saved_env_goal_x = copy.deepcopy(getattr(env, "trajectory_goal_x", None))
    saved_env_goal_y = copy.deepcopy(getattr(env, "trajectory_goal_y", None))
    branch_state = state_clean(target["branch_previous_state"])
    shifted_tvp = shift_case_tvp(case, int(target["branch_step"]))
    for name, values in shifted_tvp.items():
        if name in env.control_system.tvps:
            env.control_system.tvps[name].values = copy.deepcopy(values)
    env.control_system._step_count = 0
    env.control_system.current_state.update(copy.deepcopy(branch_state))
    env.control_system.simulator.reset_history()
    state_vec = env.control_system.get_state_vector(env.control_system.current_state)
    env.control_system.simulator.x0 = state_vec
    try:
        env.control_system.simulator._x0.master = state_vec
    except Exception:
        pass
    ctrl.reset(env.control_system.current_state, reference=None, constraint=None, tvp=None)
    if saved_noise is not None:
        ctrl.object_noise_seed = saved_noise
    if saved_goal_x is not None:
        ctrl.goal_x = saved_goal_x
    if saved_goal_y is not None:
        ctrl.goal_y = saved_goal_y
    if saved_env_goal_x is not None:
        env.trajectory_goal_x = saved_env_goal_x
    if saved_env_goal_y is not None:
        env.trajectory_goal_y = saved_env_goal_y
    env.steps_count = 0
    obs = env.get_observation()
    env.history = {"obs": [obs], "actions": [], "rewards": []}
    after_state = state_clean(env.control_system.current_state)
    meta = {
        "upstream_reset_obs_available": obs0 is not None,
        "upstream_reset_gross_s": float(reset_s),
        "upstream_reset_controller_timing": warmup_measured,
        "branch_state_target": branch_state,
        "branch_state_after_direct_reset": after_state,
        "branch_state_distance_after_reset": state_distance(branch_state, after_state),
        "shifted_tvp_lengths": {k: len(v) for k, v in shifted_tvp.items()},
        "initial_observation_at_branch": obs.tolist(),
        "initialization_note": "upstream reset only initialises saved-case goal/object/tvp state; branch episode starts after direct state/tvp overwrite and controller reset",
    }
    write_json(ep_dir / "branch_reset.json", meta)
    return obs, meta


def summarize_episode(trace: Sequence[Mapping[str, Any]], reset_meta: Mapping[str, Any], item: Mapping[str, Any], ep_dir: Path, counts: Mapping[str, Any]) -> Dict[str, Any]:
    termination = str(trace[-1].get("termination")) if trace else "none"
    success = termination == "goal"
    constraint = termination == "constraint"
    solver_failure_steps = sum(1 for r in trace if not bool(r.get("solver_success")))
    initial_failed = 0
    for r in trace:
        if bool(r.get("solver_success")):
            break
        initial_failed += 1
    final_failed = 0
    for r in reversed(trace):
        if bool(r.get("solver_success")):
            break
        final_failed += 1
    decision_times = [safe_float((r.get("timing") or {}).get("decision_s"), 0.0) for r in trace]
    solver_times: List[float] = []
    iter_counts: List[float] = []
    opt_sizes: List[int] = []
    for r in trace:
        if r.get("opt_x_size") is not None:
            opt_sizes.append(int(r["opt_x_size"]))
        for a in (r.get("recovery") or {}).get("attempts") or []:
            if a.get("solver_s") is not None:
                solver_times.append(float(a["solver_s"]))
            if a.get("iterations") is not None:
                iter_counts.append(float(a["iterations"]))
    horizon_counts: Dict[str, int] = {}
    for r in trace:
        h = str(r.get("horizon"))
        horizon_counts[h] = horizon_counts.get(h, 0) + 1
    summary: Dict[str, Any] = {
        "execution_index": int(item["execution_index"]),
        "state_id": item["state_id"],
        "case": int(item["case"]),
        "source_candidate_index": int(item.get("source_candidate_index", -1)),
        "branch_step_from_original_episode": int(item["branch_step"]),
        "true_mpc_n_horizon": int(item["true_mpc_n_horizon"]),
        "branch_horizon": int(item["commanded_horizon"]),
        "terminal_mode": item["terminal_mode"],
        "initialization": item["initialization"],
        "steps": len(trace),
        "termination": termination,
        "success": bool(success),
        "constraint": bool(constraint),
        "solver_failure_steps": int(solver_failure_steps),
        "initial_failed_steps": int(initial_failed),
        "final_failed_steps": int(final_failed),
        "physical_constraint_cost": values_tail_sum(trace, "physical"),
        "total_cost": values_tail_sum(trace, "total"),
        "decision_timing_s": finite_summary(decision_times),
        "solver_attempt_timing_s": finite_summary(solver_times),
        "solver_iteration_summary": finite_summary(iter_counts),
        "opt_x_sizes_observed": sorted(set(opt_sizes)),
        "horizon_counts": horizon_counts,
        "branch_reset": reset_meta,
        "steps_metered": int(counts.get("step_calls", 0)),
        "resets_metered": int(counts.get("reset_calls", 0)),
        "path": rel(ep_dir),
    }
    return summary


def run_true_h_episode(item: Mapping[str, Any], case: Mapping[str, Any], h15_terminal: Tuple[Any, Any]) -> Dict[str, Any]:
    ep_id = "exec%02d_%s_trueH%02d" % (int(item["execution_index"]), item["state_id"], int(item["true_mpc_n_horizon"]))
    ep_dir = SMOKE_DIR / "episodes" / ep_id
    ep_dir.mkdir(parents=True, exist_ok=False)
    true_h = int(item["true_mpc_n_horizon"])
    env = make_true_horizon_env(true_h)
    env.set_value_function_weights_and_biases(*h15_terminal)
    counts = v1e_v0b.v0.v1d.v1c.import_legacy_modules()[2].v1.meter(env, ep_dir) if False else None
    # Avoid the deliberately unreachable import expression above; use the same
    # metering helper from the stage1 runner imported through v1d below.
    base_smoke, stage1_runner, _ = v1d.import_legacy_modules()
    counts = stage1_runner.base.v1.meter(env, ep_dir)
    controller = env.control_system.controller
    original_get_action = controller.get_action
    measured: List[Dict[str, float]] = []

    def timed_get_action(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        value = original_get_action(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        measured.append({"controller_s": float(elapsed), "decision_s": float(elapsed), "selection_s": 0.0, "logging_s": 0.0})
        return value

    controller.get_action = timed_get_action
    # Reuse the existing instrumented recovery from the terminal/objective smoke.
    import vehicle_stage2_terminal_objective_smoke_v0 as stage2_obj  # noqa:E402
    recovery = stage2_obj.install_instrumented_recovery(controller.mpc, ep_dir)
    recovery.update(enabled=False, events=[], case=int(item["case"]), step=-1, horizon=true_h)
    obs, reset_meta = reset_env_to_branch(env, case, item, measured, ep_dir)
    if reset_meta["branch_state_distance_after_reset"] > STATE_DISTANCE_TOL:
        raise ContractError("direct branch reset did not preserve target branch state")
    recovery.update(enabled=True, events=[], case=int(item["case"]), step=0, horizon=true_h)
    trace: List[Dict[str, Any]] = []
    episode_start = time.perf_counter()
    with (ep_dir / "trace.jsonl").open("x", encoding="utf-8") as stream:
        for t in range(MAX_BRANCH_STEPS):
            recovery["step"] = int(t)
            recovery["horizon"] = int(true_h)
            prev_obs = env.get_observation().tolist()
            prev_state = copy.deepcopy(env.control_system.current_state)
            obs, reward, done, info = env.step(np.array([float(true_h)]))
            if len(measured) != 1:
                raise ContractError("expected exactly one controller timing measurement per branch step")
            timing = measured.pop()
            mpc = env.control_system.controller.mpc
            attempts = (recovery.get("events") or [{}])[-1].get("attempts") if recovery.get("events") else []
            row = {
                "step": int(t),
                "horizon": int(info.get("executed_horizon", true_h)),
                "true_mpc_n_horizon": int(getattr(mpc, "n_horizon", true_h)),
                "previous_state": prev_state,
                "state": copy.deepcopy(env.control_system.current_state),
                "observation": prev_obs,
                "next_observation": obs.tolist(),
                "input": copy.deepcopy(env.control_system.controller.current_input),
                "reward": float(reward),
                "performance": float(info.get("reward/performance", float("nan"))),
                "compute": float(info.get("reward/computation", float("nan"))),
                "constraint": float(info.get("reward/constraint", 0.0)),
                "termination": info.get("termination"),
                "solver_success": bool(info.get("solver_success")),
                "mpc_info_computation_time": None if info.get("mpc_computation_time") is None else float(info.get("mpc_computation_time")),
                "timing": timing,
                "recovery": {"attempts": attempts},
                "opt_x_size": int(arr(getattr(mpc, "opt_x_num", [])).size),
                "opt_g_size": int(arr(getattr(mpc, "opt_g_num", [])).size),
                "model_n_x": int(getattr(mpc.model, "n_x", -1)),
                "model_n_u": int(getattr(mpc.model, "n_u", -1)),
            }
            if row["horizon"] != true_h:
                raise ContractError("executed horizon did not equal true horizon")
            stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
            stream.flush()
            trace.append(row)
            if done:
                break
    if not trace or not trace[-1].get("termination"):
        raise ContractError("true-H branch episode did not terminate within max branch steps")
    summary = summarize_episode(trace, reset_meta, item, ep_dir, counts or {})
    summary["episode_wall_s"] = float(time.perf_counter() - episode_start)
    write_json(ep_dir / "trace.json", trace)
    write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    try:
        if hasattr(env.control_system.controller.mpc, "_optimizer"):
            pass
    finally:
        pass
    return summary


def no_safety_regression(candidate: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(candidate.get("success")):
        return False
    if bool(candidate.get("constraint")) and not bool(ref.get("constraint")):
        return False
    for key in ("initial_failed_steps", "final_failed_steps", "solver_failure_steps"):
        if int(candidate.get(key, 0)) > int(ref.get(key, 0)):
            return False
    return True


def analyze_smoke(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    by_state_h = {(str(e["state_id"]), int(e["true_mpc_n_horizon"])): e for e in episodes}
    pair_rows: List[Dict[str, Any]] = []
    for target in protocol.get("targets") or []:
        sid = str(target["state_id"])
        h10 = by_state_h.get((sid, 10))
        h15 = by_state_h.get((sid, 15))
        if h10 is None or h15 is None:
            pair_rows.append({"state_id": sid, "missing_pair": True, "safe": False})
            continue
        h10_solver = safe_float((h10.get("solver_attempt_timing_s") or {}).get("sum"), 0.0)
        h15_solver = safe_float((h15.get("solver_attempt_timing_s") or {}).get("sum"), 0.0)
        h10_decision = safe_float((h10.get("decision_timing_s") or {}).get("sum"), 0.0)
        h15_decision = safe_float((h15.get("decision_timing_s") or {}).get("sum"), 0.0)
        solver_rel = (h15_solver - h10_solver) / h15_solver if h15_solver > 0 else None
        decision_rel = (h15_decision - h10_decision) / h15_decision if h15_decision > 0 else None
        physical_gain = safe_float(h15.get("physical_constraint_cost"), 0.0) - safe_float(h10.get("physical_constraint_cost"), 0.0)
        total_gain = safe_float(h15.get("total_cost"), 0.0) - safe_float(h10.get("total_cost"), 0.0)
        opt10 = sorted(set(int(x) for x in (h10.get("opt_x_sizes_observed") or [])))
        opt15 = sorted(set(int(x) for x in (h15.get("opt_x_sizes_observed") or [])))
        dim_reduced = bool(opt10 and opt15 and max(opt10) < min(opt15))
        safe = no_safety_regression(h10, h15)
        pair_rows.append({
            "state_id": sid,
            "case": int(target.get("case")),
            "branch_step": int(target.get("branch_step")),
            "missing_pair": False,
            "h10_opt_x_sizes": opt10,
            "h15_opt_x_sizes": opt15,
            "dimension_reduced_H10_vs_H15": dim_reduced,
            "h10_solver_sum_s": h10_solver,
            "h15_solver_sum_s": h15_solver,
            "solver_relative_saving_H10_vs_H15": solver_rel,
            "h10_decision_sum_s": h10_decision,
            "h15_decision_sum_s": h15_decision,
            "decision_relative_saving_H10_vs_H15": decision_rel,
            "physical_gain_H10_vs_H15": physical_gain,
            "total_gain_H10_vs_H15": total_gain,
            "h10_success": bool(h10.get("success")),
            "h15_success": bool(h15.get("success")),
            "h10_constraint": bool(h10.get("constraint")),
            "h15_constraint": bool(h15.get("constraint")),
            "h10_solver_failure_steps": int(h10.get("solver_failure_steps", 0)),
            "h15_solver_failure_steps": int(h15.get("solver_failure_steps", 0)),
            "safety_ok_vs_H15": safe,
            "h10_path": h10.get("path"),
            "h15_path": h15.get("path"),
        })
    solver_rels = [p["solver_relative_saving_H10_vs_H15"] for p in pair_rows if not p.get("missing_pair") and p.get("solver_relative_saving_H10_vs_H15") is not None]
    decision_rels = [p["decision_relative_saving_H10_vs_H15"] for p in pair_rows if not p.get("missing_pair") and p.get("decision_relative_saving_H10_vs_H15") is not None]
    all_dim = all(p.get("dimension_reduced_H10_vs_H15") for p in pair_rows) and len(pair_rows) == len(EXPECTED_TARGET_STATE_IDS)
    all_safe = all(p.get("safety_ok_vs_H15") for p in pair_rows) and len(pair_rows) == len(EXPECTED_TARGET_STATE_IDS)
    median_solver_rel = finite_summary([float(x) for x in solver_rels]).get("median") if solver_rels else None
    median_decision_rel = finite_summary([float(x) for x in decision_rels]).get("median") if decision_rels else None
    pass_to_broaden = bool(all_dim and all_safe and median_solver_rel is not None and float(median_solver_rel) >= MIN_DIMENSIONAL_SOLVER_SAVING)
    return {
        "pair_count": len(pair_rows),
        "pair_rows": pair_rows,
        "all_pairs_dimension_reduced": all_dim,
        "all_pairs_safety_ok": all_safe,
        "median_solver_relative_saving_H10_vs_H15": median_solver_rel,
        "median_decision_relative_saving_H10_vs_H15": median_decision_rel,
        "solver_relative_savings_summary": finite_summary([float(x) for x in solver_rels]),
        "decision_relative_savings_summary": finite_summary([float(x) for x in decision_rels]),
        "pass_to_broader_variable_horizon_block": pass_to_broaden,
        "decision": {
            "train_or_refit_now": False,
            "if_pass": "after backup, broaden true variable-H controller-cache development block with both terminal modes and fair fixed-H timing before any learning/refit",
            "if_fail": "classify shorter-H-as-speed infeasible or too small in this stack and pivot to terminal/value/modeling or scenario-opportunity design",
        },
    }


def write_smoke_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle true variable-horizon case5 smoke v0 run",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only true variable-dimension branch smoke; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- All pairs dimension-reduced: `{a['all_pairs_dimension_reduced']}`.",
        f"- All pairs safety-ok vs H15: `{a['all_pairs_safety_ok']}`.",
        f"- Median solver relative saving H10 vs H15: `{a['median_solver_relative_saving_H10_vs_H15']}`.",
        f"- Median decision relative saving H10 vs H15: `{a['median_decision_relative_saving_H10_vs_H15']}`.",
        f"- Pass to broader variable-H block: `{a['pass_to_broader_variable_horizon_block']}`.",
        "",
        "## Pair table",
        "",
        "| state | opt_x H10 | opt_x H15 | solver rel save | decision rel save | physical gain | safe |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for p in a["pair_rows"]:
        lines.append("| `%s` | `%s` | `%s` | %s | %s | %.6g | `%s` |" % (
            p.get("state_id"), p.get("h10_opt_x_sizes"), p.get("h15_opt_x_sizes"),
            "NA" if p.get("solver_relative_saving_H10_vs_H15") is None else "%.6g" % float(p["solver_relative_saving_H10_vs_H15"]),
            "NA" if p.get("decision_relative_saving_H10_vs_H15") is None else "%.6g" % float(p["decision_relative_saving_H10_vs_H15"]),
            float(p.get("physical_gain_H10_vs_H15", 0.0)), bool(p.get("safety_ok_vs_H15")),
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "This smoke tests implementation feasibility only. Direct branch-state initialisation gives a cold true-H controller comparison; it does not replace independent validation or a fair fixed-H/adaptive campaign. A positive result only justifies a broader development block with repeated timing and fixed-H controls.",
        "",
        f"Backup request: `{raw['backup_request_after_run']}`.",
    ]
    (SMOKE_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(backup_proof: Path) -> int:
    dry_done = completed_ok(DRYRUN_DIR / "completed.json", check_hashes=True)
    if (SMOKE_DIR / "completed.json").exists():
        done = completed_ok(SMOKE_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(SMOKE_DIR / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if SMOKE_DIR.exists() and any(p.name != "run.lock" for p in SMOKE_DIR.iterdir()):
        raise ContractError(f"partial smoke output exists; inspect first: {rel(SMOKE_DIR)}")
    protocol = read_json(PROTOCOL_JSON)
    min_time = max(t for t in [source_mtime_utc(), file_mtime_utc(PROTOCOL_JSON), parse_time(dry_done.get("created_utc"))] if t is not None)
    backup = v1d.verify_backup_proof(backup_proof, min_time, NAME)
    bank = read_json(Path(STAGE1_BANK))
    selected_cases = bank.get("selected_cases_full") or bank.get("selected_cases")
    if not isinstance(selected_cases, list):
        raise ContractError("stage1 selected cases not available")
    # Load terminal grid once. The H15 terminal weights are deliberately shared
    # across true-H10 and true-H15 arms.
    _, stage1_runner, _ = v1d.import_legacy_modules()
    preflight = stage1_runner.runtime_preflight()
    if not preflight.get("passed"):
        raise ContractError(f"legacy runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    terminal_source_protocol = read_json(stage1_runner.TERMINAL_SOURCE_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(terminal_source_protocol["terminal_grid_readiness_reused_from_v1"])
    h15_terminal = terminals[15]
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    started = now_utc().isoformat()
    write_json(SMOKE_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": f"{NAME}_run", "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0})
    write_json(SMOKE_DIR / "runtime_preflight.json", preflight)
    write_json(SMOKE_DIR / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    episodes: List[Dict[str, Any]] = []
    for item in protocol.get("schedule") or []:
        case = selected_cases[int(item["case"])]
        summary = run_true_h_episode(item, case, h15_terminal)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": EPISODES_EXACT, "control_steps_done": int(sum(int(e.get("steps", 0)) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "case", "true_mpc_n_horizon", "steps", "success", "termination", "opt_x_sizes_observed")}, "validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(SMOKE_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    if len(episodes) != EPISODES_EXACT or control_steps > CONTROL_STEP_UPPER:
        raise ContractError("true variable-H smoke budget violation")
    analysis = analyze_smoke(episodes, protocol)
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_CASE5_SMOKE_V0_RUN_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    write_json(req, {"requested_utc": created.isoformat(), "reason": "backup true variable-dimension case5 smoke outputs before further simulation/training/refit", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "artifacts": [rel(SMOKE_DIR), rel(STATE_SMOKE), rel(SOURCE), rel(PROTOCOL_JSON), rel(req)]})
    raw = {
        "created_utc": created.isoformat(),
        "started_utc": started,
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": f"{NAME}_run",
        "classification": "development_IMPROVED_true_variable_dimension_mpc_smoke_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"episodes_exact": EPISODES_EXACT, "control_step_upper_bound": CONTROL_STEP_UPPER, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "episodes": episodes,
        "analysis": analysis,
        "backup_request_after_run": rel(req),
        "interpretation_limits": ["development implementation smoke only", "direct branch-state cold controller initialization", "not validation/model selection", "not final test", "not a trained selector", "not ORIGINAL SAC"],
    }
    write_json(SMOKE_DIR / "raw.json", raw)
    write_smoke_summary(raw)
    STATE_SMOKE.parent.mkdir(parents=True, exist_ok=True)
    STATE_SMOKE.write_text((SMOKE_DIR / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    append_docs(f"""<!-- {MARKER_SMOKE} -->
## 2026-09-29 vehicle true variable-horizon case5 smoke v0 run

UTC: {created.isoformat()}. Development-only true variable-dimension MPC smoke completed: {len(episodes)} episodes, {control_steps} control steps. Dimension reduced for all pairs={analysis['all_pairs_dimension_reduced']}; safety ok for all pairs={analysis['all_pairs_safety_ok']}; median solver relative H10 saving={analysis['median_solver_relative_saving_H10_vs_H15']}; pass to broader variable-H block={analysis['pass_to_broader_variable_horizon_block']}. No validation64-bank or sealed-test access, no training/refit. Artifacts: `{rel(SMOKE_DIR / 'summary.md')}`, `{rel(SMOKE_DIR / 'raw.json')}`, `{rel(SMOKE_DIR / 'completed.json')}`.
""", MARKER_SMOKE)
    files = [p for p in SMOKE_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL_JSON, STATE_SMOKE, req, backup_proof, DRYRUN_DIR / "completed.json"]
    write_json(SMOKE_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(), "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(episodes), "control_steps": control_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "backup_request": rel(req), "headline": {"all_pairs_dimension_reduced": analysis["all_pairs_dimension_reduced"], "all_pairs_safety_ok": analysis["all_pairs_safety_ok"], "median_solver_relative_saving_H10_vs_H15": analysis["median_solver_relative_saving_H10_vs_H15"], "median_decision_relative_saving_H10_vs_H15": analysis["median_decision_relative_saving_H10_vs_H15"], "pass_to_broader_variable_horizon_block": analysis["pass_to_broader_variable_horizon_block"], "train_or_refit_now": False}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(SMOKE_DIR / "completed.json"), "summary": rel(SMOKE_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "all_pairs_dimension_reduced": analysis["all_pairs_dimension_reduced"], "median_solver_relative_saving_H10_vs_H15": analysis["median_solver_relative_saving_H10_vs_H15"], "pass_to_broader_variable_horizon_block": analysis["pass_to_broader_variable_horizon_block"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-smoke", action="store_true")
    ap.add_argument("--backup-proof", type=Path, default=None)
    ap.add_argument("--i-accept-development-true-variable-horizon-case5-smoke-v0", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_accept_development_true_variable_horizon_case5_smoke_v0:
        raise ContractError("explicit --i-accept-development-true-variable-horizon-case5-smoke-v0 required")
    if bool(args.dry_run) == bool(args.run_smoke):
        raise ContractError("exactly one of --dry-run or --run-smoke is required")
    if args.dry_run:
        return run_dry_run()
    if args.backup_proof is None:
        raise ContractError("--run-smoke requires --backup-proof")
    return run_smoke(args.backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR if "--dry-run" in sys.argv else SMOKE_DIR if "--run-smoke" in sys.argv else ROOT / f"research_artifacts/aws_diagnostics/{NAME}_failure_unknown_{STAMP}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {"failed_utc": now_utc().isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_rollouts": 0 if "--dry-run" in sys.argv else None, "new_control_steps": 0 if "--dry-run" in sys.argv else None, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "next_recovery_hint": "Preserve partial output. If dry-run failed, repair source/protocol checks only; if smoke failed, audit partial true-H episodes before any rerun."})
        raise
