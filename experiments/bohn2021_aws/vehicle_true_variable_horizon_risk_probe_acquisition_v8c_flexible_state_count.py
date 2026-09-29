#!/usr/bin/env python3
"""v8c retry for targeted true-variable-H risk-probe acquisition.

This versioned wrapper repairs the v8/v8b execution contract bug found on
2026-09-29: the reused fresh-source helper selected the scientifically intended
2 branch states per fresh case, but then asserted a stale fixed total of 16
states from an earlier 8-case protocol.  v8 targets 4 fresh cases, so the
correct predeclared total is 4 * 2 = 8 selected branch states.

Scientific protocol is otherwise unchanged from v8/v8b:
- development-only IMPROVED diagnostic;
- no validation64 and no sealed final test;
- no gradient training, no selector refit;
- true H10/H15 paired branch continuations under the shared-H15 terminal;
- backup must be verified before execution because v8b produced partial Stage-A
  simulation evidence (4 episodes / 300 control steps) before failing.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_probe_acquisition_v8 as v8  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count"
STAMP = "20260929T1320Z"
BASE_V8_SOURCE = v8.SOURCE

# Fresh output/provenance namespace; never overwrite v8a/v8b evidence.
v8.NAME = NAME
v8.STAMP = STAMP
v8.SOURCE = Path(__file__).resolve()
v8.RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
v8.STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
v8.CONTINUE_STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1320Z_after_risk_probe_acquisition_v8c_flexible_state_count.md"
v8.PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
v8.MARKER = f"vehicle-true-variable-H-risk-probe-acquisition-v8c-flexible-state-count-{STAMP}"

_ORIGINAL_BUILD_PROTOCOL = v8.build_protocol


def _trace_risk_score(row: Mapping[str, Any], n: int) -> float:
    scorer = getattr(v8.engine.fresh0, "trace_risk_score", None)
    if scorer is None:
        # Conservative fallback should never be needed in the current source; it
        # keeps the wrapper self-diagnosing rather than silently changing the
        # scoring rule.
        raise v8.ContractError("fresh0.trace_risk_score missing; refusing to change state-selection score")
    return float(scorer(row, n))


def _select_stage_a_states_v8c(stage_a_episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Select the v8 predeclared number of branch states.

    This is a local copy of the reused helper's selection logic with only the
    terminal contract generalized from a stale hard-coded 16 to
    len(stage_a_episodes) * protocol.branch_states_per_case.  Selection windows,
    risk score, state fields, and the rule that Stage-A traces precede any
    Stage-B H10 outcome remain unchanged.
    """
    cfg = (protocol.get("stage_A_h15_trace_state_selection") or {})
    bounds = cfg.get("eligible_step_bounds") or {}
    windows = list(cfg.get("windows") or [])
    per_case = v8.si(cfg.get("branch_states_per_case"), len(windows))
    if per_case <= 0:
        raise v8.ContractError("invalid branch_states_per_case in v8c protocol")
    if len(windows) < per_case:
        raise v8.ContractError("not enough selection windows for v8c protocol")
    windows = windows[:per_case]
    min_step = v8.si(bounds.get("min_step"), 8)
    max_step_abs = v8.si(bounds.get("max_step"), 90)
    reserve = v8.si(bounds.get("reserve_terminal_margin_steps"), 3)

    selected: List[Dict[str, Any]] = []
    for ep in stage_a_episodes:
        trace_path = v8.ROOT / str(ep.get("path")) / "trace.json"
        trace = v8.read_json(trace_path)
        n = len(trace)
        if n < min_step + reserve + 2:
            raise v8.ContractError("Stage-A trace too short for v8c state selection: " + str(ep.get("state_id")))
        chosen_steps: List[int] = []
        for window in windows:
            slot = v8.si(window.get("slot"), len(chosen_steps))
            lo = max(min_step, int(math.floor(n * v8.sf(window.get("low_fraction"), 0.1))))
            hi = min(max_step_abs, n - reserve - 1, int(math.ceil(n * v8.sf(window.get("high_fraction"), 0.8))))
            candidates = [r for r in trace if lo <= v8.si(r.get("step"), -1) <= hi]
            if not candidates:
                candidates = [r for r in trace if min_step <= v8.si(r.get("step"), -1) <= min(max_step_abs, n - reserve - 1)]
            if slot == 1 and chosen_steps:
                separated = [r for r in candidates if all(abs(v8.si(r.get("step"), -1) - s) >= 8 for s in chosen_steps)]
                if separated:
                    candidates = separated
            if not candidates:
                raise v8.ContractError("no eligible v8c Stage-A state for " + str(ep.get("state_id")))
            row = max(candidates, key=lambda r: _trace_risk_score(r, n))
            step = v8.si(row.get("step"), -1)
            chosen_steps.append(step)
            base_state_id = f"fresh_case{v8.si(ep.get('fresh_case_index'), -1):02d}_slot{slot}_{window.get('name', 'window')}"
            obs = row.get("observation") or []
            selected.append({
                "base_state_id": base_state_id,
                "fresh_case_index": v8.si(ep.get("fresh_case_index"), -1),
                "source_candidate_index": v8.si(ep.get("source_candidate_index"), -1),
                "fresh_confirmation_group": ep.get("fresh_confirmation_group"),
                "branch_state_slot": slot,
                "window": window.get("name"),
                "branch_step": step,
                "branch_previous_state": copy.deepcopy(row.get("previous_state") or row.get("state")),
                "initial_observation_from_h15_trace": copy.deepcopy(obs),
                "h15_trace_episode_path": ep.get("path"),
                "stage_a_trace_risk_score": _trace_risk_score(row, n),
                "selection_rule": "v8c flexible-count repair: selected from H15 trace before any Stage-B H10/H15 branch outcome was run; scoring/window logic unchanged from reused helper",
            })

    expected = len(stage_a_episodes) * per_case
    if len(selected) != expected:
        raise v8.ContractError("expected %d selected branch states under v8c protocol, got %d" % (expected, len(selected)))
    return selected


def _build_protocol_with_amendment(created: dt.datetime, selected_cases: Sequence[Mapping[str, Any]], case_diag: Mapping[str, Any], input_hashes: Mapping[str, str]) -> Mapping[str, Any]:
    protocol = _ORIGINAL_BUILD_PROTOCOL(created, selected_cases, case_diag, input_hashes)
    protocol["amendment"] = {
        "amendment_id": f"{NAME}_after_v8b_stale_state_count_contract_failure",
        "created_utc": created.isoformat(),
        "reason": "v8b ran the legacy runtime and completed Stage-A traces, but the reused fresh-source selector asserted a stale expected total of 16 branch states. The v8 protocol has 4 fresh cases and 2 branch states per case, so the correct total is 8. This wrapper generalizes only that execution contract and preserves selection windows/scoring/budgets.",
        "failed_prior_attempts": [
            {
                "script": "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py",
                "run_registry": "research_artifacts/aws_runs/20260929T130819_cc2e807a/registry.json",
                "failed_artifact": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8_20260929T1258Z/failed.json",
                "scientific_effect": "none; modern interpreter lacked TensorFlow, so no simulation/training/refit/validation/test",
            },
            {
                "script": "experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry.py",
                "run_registry": "research_artifacts/aws_runs/20260929T131628_ec8e3d21/registry.json",
                "failed_artifact": "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry_20260929T1312Z/failed.json",
                "scientific_effect": "partial development Stage-A only: 4 H15 trace episodes / 300 control steps; no Stage-B branch comparisons, no training/refit, no validation64, no sealed test",
            },
        ],
        "unchanged_scientific_protocol_from_v8": True,
        "changed_variables": [
            "NAME", "STAMP", "RUN_DIR", "STATE", "CONTINUE_STATE", "PROTOCOL", "MARKER", "SOURCE for provenance",
            "stage_A selected-state count check generalized from stale constant 16 to len(stage_a_episodes) * branch_states_per_case",
        ],
        "expected_selected_branch_states": {
            "fresh_cases": len(selected_cases),
            "branch_states_per_case": v8.BRANCH_STATES_PER_CASE,
            "total": len(selected_cases) * v8.BRANCH_STATES_PER_CASE,
        },
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "source_hashes": {
            "base_v8_source": v8.sha256(BASE_V8_SOURCE),
            "v8c_wrapper_source": v8.sha256(v8.SOURCE),
        },
    }
    protocol.setdefault("execution_repairs", []).append({
        "repair": "flexible selected-state count for smaller v8 target-case budget",
        "old_stale_expected_total": 16,
        "new_expected_total_formula": "len(stage_a_episodes) * protocol.stage_A_h15_trace_state_selection.branch_states_per_case",
        "selection_score_changed": False,
        "selection_windows_changed": False,
        "case_selection_changed": False,
        "branch_template_changed": False,
    })
    v8.write_json(v8.PROTOCOL, protocol)
    return protocol


# Apply patches before v8.run() uses the helpers.
v8.build_protocol = _build_protocol_with_amendment
v8.engine.fresh0.select_stage_a_states = _select_stage_a_states_v8c


def _write_failure_continue(rc: int) -> None:
    created = dt.datetime.now(dt.timezone.utc)
    v8.CONTINUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    failure_artifact = v8.RUN_DIR / "failed.json"
    payload = {
        "utc": created.isoformat(),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "return_code": rc,
        "failure_artifact": failure_artifact.relative_to(ROOT).as_posix() if failure_artifact.exists() else None,
        "scientific_effect": "inspect progress.json and failed.json before rerun; wrapper writes this only on nonzero return",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "next_action": "If failure occurs before Stage-B, preserve partial H15 traces and repair execution only after backup. If Stage-B partially ran, audit paired budget before any rerun.",
    }
    v8.CONTINUE_STATE.write_text("# Continue state after v8c flexible-state-count retry failure\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    rc = v8.main()
    if rc != 0:
        _write_failure_continue(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
