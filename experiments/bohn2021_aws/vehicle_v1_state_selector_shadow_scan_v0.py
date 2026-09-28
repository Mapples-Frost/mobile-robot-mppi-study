#!/usr/bin/env python3
"""Vehicle V1 state-selector shadow scan v0.

No-simulation diagnostic after the selector-smoke v0b pass.  It scans already
saved fixed-H15 development traces from the V1 fixed-H opportunity probe and
asks where the frozen nearest-state H30 latch would have triggered without
running any new MPC rollouts.  This distinguishes a benign sparse state rule from
an over-broad radius/prototype rule before allocating a broader online selector
confirmation.  It opens no validation64 bank and never accesses sealed test.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.json"
SMOKE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/raw.json"
SMOKE_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/completed.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
NEXT_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928.json"
NEXT_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928.md"
MARKER = "vehicle-v1-state-selector-shadow-scan-v0-20260928"
EXPECTED_CASES = list(range(16))
POSITIVE_CASES = {7, 10}


class DiagnosticError(RuntimeError):
    pass


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


def verify_done(path: Path) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise DiagnosticError("completed marker did not pass: %s" % rel(path))
    if done.get("historical_validation64_bank_opened") not in (False, None):
        raise DiagnosticError("input unexpectedly opened historical validation64: %s" % rel(path))
    if done.get("sealed_test_accessed") not in (False, None):
        raise DiagnosticError("input unexpectedly accessed sealed test: %s" % rel(path))
    return done


def safe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return None


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def state_from_mapping(obj: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    # Direct x/y/theta keys.
    x = safe_float(obj.get("x"))
    y = safe_float(obj.get("y"))
    th = safe_float(obj.get("theta"))
    if x is not None and y is not None and th is not None:
        return (x, y, th)
    # Common nested containers in older diagnostics.
    for key in (
        "previous_state", "state_before", "pre_state", "state", "current_state",
        "vehicle_state", "vehicle_state_before", "system_state", "system_state_before",
        "control_state", "control_state_before", "env_state", "observation_state",
    ):
        val = obj.get(key)
        if isinstance(val, Mapping):
            got = state_from_mapping(val)
            if got is not None:
                return got
        elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)) and len(val) >= 3:
            xs = [safe_float(val[i]) for i in range(3)]
            if all(v is not None for v in xs):
                return (float(xs[0]), float(xs[1]), float(xs[2]))
    # Some traces store state under info/raw/debug.
    for key in ("info", "raw", "debug", "meta"):
        val = obj.get(key)
        if isinstance(val, Mapping):
            got = state_from_mapping(val)
            if got is not None:
                return got
    return None


def proto_state(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    state = row.get("state") or {}
    got = state_from_mapping(state) if isinstance(state, Mapping) else None
    if got is None:
        raise DiagnosticError("prototype lacks state")
    return got


def nearest(state: Tuple[float, float, float], protos: Sequence[Mapping[str, Any]]) -> Tuple[float, Optional[Mapping[str, Any]]]:
    best_d = float("inf")
    best = None
    for p in protos:
        d = state_distance(state, proto_state(p))
        if d < best_d:
            best_d, best = d, p
    return best_d, best


def infer_case(ep: Mapping[str, Any]) -> Optional[int]:
    for key in ("case", "selected_case", "case_id", "bank_case"):
        if key in ep:
            try:
                return int(ep[key])
            except Exception:
                pass
    path = str(ep.get("path", ""))
    m = re.search(r"case[_-]?(\d+)", path, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def infer_horizon(ep: Mapping[str, Any]) -> Optional[int]:
    for key in ("horizon", "fixed_horizon", "H", "h"):
        if key in ep:
            try:
                return int(ep[key])
            except Exception:
                pass
    counts = ep.get("horizon_counts")
    if isinstance(counts, Mapping) and len(counts) == 1:
        try:
            return int(next(iter(counts.keys())))
        except Exception:
            pass
    path = str(ep.get("path", ""))
    for pat in (r"fixed[_-]?H(\d+)", r"[_-]H(\d+)", r"horizon[_-]?(\d+)"):
        m = re.search(pat, path, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def find_episode_list(v1_raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if isinstance(v1_raw.get("episodes"), list):
        return list(v1_raw["episodes"])
    # Fallback: look one level down for a list of dicts with episode-like keys.
    for val in v1_raw.values():
        if isinstance(val, list) and val and isinstance(val[0], Mapping) and ("physical_constraint_cost" in val[0] or "total_cost" in val[0]):
            return list(val)
    return []


def summarize_fixed_grid(episodes: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, Dict[int, Mapping[str, Any]]], Dict[str, Any]]:
    by_case_h: Dict[int, Dict[int, Mapping[str, Any]]] = {}
    by_h: Dict[int, List[Mapping[str, Any]]] = {}
    for ep in episodes:
        case = infer_case(ep)
        h = infer_horizon(ep)
        if case is None or h is None:
            continue
        by_case_h.setdefault(case, {})[h] = ep
        by_h.setdefault(h, []).append(ep)
    aggregate: Dict[str, Any] = {}
    for h, rows in sorted(by_h.items()):
        aggregate[str(h)] = {
            "episodes": len(rows),
            "success_count": int(sum(1 for r in rows if bool(r.get("success")))),
            "constraint_count": int(sum(1 for r in rows if bool(r.get("constraint")))),
            "solver_failure_steps": int(sum(int(r.get("solver_failure_steps", 0) or 0) for r in rows)),
            "initial_failed_steps": int(sum(int(r.get("initial_failed_steps", 0) or 0) for r in rows)),
            "physical_constraint_cost_sum": float(math.fsum(float(r.get("physical_constraint_cost", 0.0) or 0.0) for r in rows)),
            "total_cost_sum": float(math.fsum(float(r.get("total_cost", 0.0) or 0.0) for r in rows)),
            "decision_total_s": float(math.fsum(float(((r.get("decision_timing_s") or {}).get("sum", 0.0)) or 0.0) for r in rows)),
            "steps": int(sum(int(r.get("steps", 0) or 0) for r in rows)),
        }
    return by_case_h, aggregate


def scan_trace(case: int, ep: Mapping[str, Any], selector: Mapping[str, Any]) -> Dict[str, Any]:
    ep_path = ROOT / str(ep.get("path", ""))
    trace_path = ep_path / "trace.json"
    if not trace_path.exists():
        return {"case": case, "error": "missing_trace_json", "path": rel(trace_path), "triggered": False}
    trace = read_json(trace_path)
    pos = list(selector.get("positive_prototypes") or [])
    neg = list(selector.get("negative_veto_prototypes") or [])
    radius = float(selector.get("positive_radius", 0.1))
    veto_radius = float(selector.get("negative_veto_radius", 0.0))
    nearest_seen = {"distance": float("inf"), "step": None, "prototype_case": None, "prototype_branch_step": None}
    first_trigger = None
    missing_state_steps = 0
    for step, row in enumerate(trace):
        if not isinstance(row, Mapping):
            missing_state_steps += 1
            continue
        st = state_from_mapping(row)
        if st is None:
            missing_state_steps += 1
            continue
        pd, pp = nearest(st, pos)
        nd, np_ = nearest(st, neg)
        if pd < nearest_seen["distance"]:
            nearest_seen = {
                "distance": float(pd),
                "step": int(step),
                "prototype_case": None if pp is None else int(pp.get("case")),
                "prototype_branch_step": None if pp is None else int(pp.get("branch_step")),
            }
        veto = bool(veto_radius > 0.0 and nd <= veto_radius)
        if first_trigger is None and pd <= radius and not veto:
            first_trigger = {
                "step": int(step),
                "state": {"x": st[0], "y": st[1], "theta": st[2]},
                "nearest_positive_distance": float(pd),
                "nearest_positive_case": None if pp is None else int(pp.get("case")),
                "nearest_positive_branch_step": None if pp is None else int(pp.get("branch_step")),
                "nearest_negative_distance": float(nd),
                "nearest_negative_case": None if np_ is None else int(np_.get("case")),
                "nearest_negative_branch_step": None if np_ is None else int(np_.get("branch_step")),
            }
            break
    return {
        "case": int(case),
        "path": rel(ep_path),
        "trace_steps": int(len(trace)) if isinstance(trace, list) else None,
        "missing_state_steps_before_trigger_or_end": int(missing_state_steps),
        "triggered": first_trigger is not None,
        "first_trigger": first_trigger,
        "nearest_positive_seen": nearest_seen,
    }


def write_next_protocol(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not raw["shadow_gate_pass"]:
        return None
    created = raw["created_utc"]
    protocol = {
        "protocol_id": "vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_frozen_20260928",
        "created_utc": created,
        "classification": "development_IMPROVED_selector_broad_confirmation_not_validation_not_final_test",
        "motivation": [
            "Smoke v0b passed on opportunity/guard cases; shadow scan of all V1 fixed-H15 traces found sparse triggers without extra-case over-triggering.",
            "Next online rollout checks whether the trace-derived selector remains safe and aggregate-beneficial across the full V1 development bank before any broader refit/training campaign.",
        ],
        "access_rules": {"development_only": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_external_backup_after_shadow_scan_before_rollout": True},
        "selector_source_protocol": rel(PROTOCOL_JSON),
        "selector": read_json(PROTOCOL_JSON)["selector"],
        "split_and_budget": {
            "source_bank": "vehicle_fixed_h_opportunity_probe_v1_fresh64_select16_development_only",
            "rollout_cases": EXPECTED_CASES,
            "arms": ["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"],
            "max_episodes": 64,
            "control_step_upper_bound": 9600,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "acceptance_for_next_refit_or_training": {
            "safety": "selector all-success, no constraints, no solver-failure-step increase vs paired H15, no failure where H15 succeeds",
            "benefit": "aggregate selector physical+constraint and total costs improve versus paired H15 by at least 20 absolute units, with transparent comparison to fixed H30 and H10",
            "scope": "development confirmation only; passing does not establish validation/test success and should trigger a versioned broader refit/training design across independent seeds",
            "timing": "report whole-decision, solver, selection and terminal-switch timing; do not infer compute from horizon counts",
        },
        "shadow_scan_inputs": {"raw": rel(OUT_DIR / "raw.json"), "summary": rel(OUT_DIR / "summary.md")},
    }
    write_json(NEXT_PROTOCOL_JSON, protocol)
    md = [
        "# Vehicle V1 state-continuation selector broad development confirmation v0",
        "",
        f"Frozen UTC: `{created}`.",
        "",
        "Development-only online confirmation protocol. It reuses the frozen nearest-state H30 latch selector and runs it on all 16 V1 development cases against fixed H15/H30/H10. No training, no validation64-bank access, no sealed-test access.",
        "",
        "A verified external backup after the shadow scan/protocol is required before any rollout simulation.",
        "",
        "## Budget",
        "",
        "- Cases: 0..15 from the V1 fresh64/select16 development bank",
        "- Arms: selector_latch_H30, fixed_H15, fixed_H30, fixed_H10",
        "- Max episodes: 64; control-step upper bound: 9600",
        "- Training episodes / gradient steps: 0 / 0",
        "",
        "## Acceptance for next refit/training",
        "",
        "Safety: selector all-success, no constraints, no solver-failure-step increase vs paired H15, no failure where H15 succeeds.",
        "Benefit: aggregate selector physical+constraint and total costs improve versus paired H15 by at least 20 absolute units; fixed H30/H10 remain strong disclosed baselines.",
        "Timing: report actual whole-decision, solver, selection and terminal-switch timing.",
    ]
    NEXT_PROTOCOL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"json": rel(NEXT_PROTOCOL_JSON), "json_sha256": sha256(NEXT_PROTOCOL_JSON), "md": rel(NEXT_PROTOCOL_MD), "md_sha256": sha256(NEXT_PROTOCOL_MD)}


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = str(raw["created_utc"]).replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_STATE_SELECTOR_SHADOW_SCAN_V0_%s.json" % stamp)
    artifacts = [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve())]
    if raw.get("next_protocol"):
        artifacts += [raw["next_protocol"]["json"], raw["next_protocol"]["md"]]
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup no-simulation shadow scan and frozen broad development confirmation protocol before any further selector rollouts",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": artifacts,
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle V1 state-selector shadow scan v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "No-simulation scan of the frozen v0b nearest-state H30 latch over already-saved V1 fixed-H15 development traces. No new rollouts/training, no validation64-bank access, no sealed-test access.",
        "",
        "## Result",
        "",
        f"- H15 traces found: `{len(raw['h15_trace_cases_found'])}` / 16; missing cases: `{raw['missing_h15_trace_cases']}`.",
        f"- Triggered cases: `{raw['triggered_cases']}`; extra trigger cases outside confirmed positives {{7,10}}: `{raw['extra_trigger_cases']}`.",
        f"- Shadow gate pass: `{raw['shadow_gate_pass']}`.",
        f"- Forecast selector aggregate if nontrigger cases remain H15 and smoke deltas transfer: physical `{raw['forecast']['selector_physical_constraint_cost_sum']:.6g}` vs H15 `{raw['forecast']['h15_physical_constraint_cost_sum']:.6g}`; total `{raw['forecast']['selector_total_cost_sum']:.6g}` vs H15 `{raw['forecast']['h15_total_cost_sum']:.6g}`.",
        "",
        "## Per-case shadow triggers",
        "",
        "| case | triggered | first step | nearest prototype | nearest distance seen | H15 physical | H30 physical | smoke/forecast delta phys |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    per_case = {int(r["case"]): r for r in raw["shadow_cases"]}
    for case in EXPECTED_CASES:
        r = per_case.get(case, {"case": case, "triggered": False})
        trig = r.get("first_trigger") or {}
        near = r.get("nearest_positive_seen") or {}
        comp = raw["case_comparisons"].get(str(case), {})
        proto = ""
        if trig:
            proto = f"case{trig.get('nearest_positive_case')} step{trig.get('nearest_positive_branch_step')}"
        else:
            proto = f"nearest case{near.get('prototype_case')} step{near.get('prototype_branch_step')}"
        lines.append("| %d | `%s` | %s | %s | %.6g | %s | %s | %s |" % (
            case,
            str(bool(r.get("triggered"))),
            "" if not trig else str(trig.get("step")),
            proto,
            float((near or {}).get("distance", float("nan"))),
            "" if comp.get("H15_physical") is None else ("%.6g" % float(comp["H15_physical"])),
            "" if comp.get("H30_physical") is None else ("%.6g" % float(comp["H30_physical"])),
            "" if comp.get("forecast_selector_delta_physical_vs_H15") is None else ("%.6g" % float(comp["forecast_selector_delta_physical_vs_H15"])),
        ))
    lines += ["", "## Evidence table", "", "| axis | verified finding | uncertainty / competing hypothesis | discriminating next experiment |", "|---|---|---|---|"]
    lines.append("| scenarios | V1 fixed-H grid and controlled continuation show sparse noninitial H30 opportunity; shadow scan says current rule triggers only confirmed cases if gate passes. | Opportunity may be overfit to four development cases and may not generalize beyond V1 bank. | Run frozen all-16 V1 broad development confirmation after backup, then design fresh refit/confirmation bank. |")
    lines.append("| reward/objective | Smoke gains are physical+constraint and total-cost gains, not only h-penalty; no constraints/solver-fail steps. | Broad aggregate benefit depends on selector not adding hidden failures/timing overhead. | Paired online broad confirmation with actual whole-decision/solver/selection timing. |")
    lines.append("| training/selection | Historical policies collapsed, while state-local selector exploits confirmed labels; this supports state-label/policy-class mismatch. | Nearest-state latch is not trained and may not scale; terminal-value mismatch remains possible. | If broad confirmation passes, freeze a real broader refit/training design over independent seeds rather than more local patches. |")
    lines.append("| comparisons | Fixed H15/H30/H10 remain strong baselines; H30 is strong on the 4 smoke cases but H15 is stronger on all 16 V1 aggregate. | Runtime noise may make solver-time comparisons unreliable at small N. | Blocked all-16 paired rollout with randomized schedule and disclosed full budgets. |")
    lines += ["", "## Decision", ""]
    if raw["shadow_gate_pass"]:
        lines.append("Shadow scan passed. The next informative action is the frozen all-16 broad development confirmation after verified external backup, not another summary-only audit.")
    else:
        lines.append("Shadow scan did not pass. Next work should repair trace extraction if missing, or revise selector representation/radius before any broader online rollout.")
    if raw.get("next_protocol"):
        lines.append(f"Next protocol: `{raw['next_protocol']['md']}` / `{raw['next_protocol']['json']}`.")
    lines.append(f"Backup request before further simulation: `{raw['backup_request']}`.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 state-selector shadow scan v0\n\n"
        f"UTC: {raw['created_utc']}. No-simulation scan over existing V1 H15 traces: "
        f"triggered_cases={raw['triggered_cases']}, extra_trigger_cases={raw['extra_trigger_cases']}, "
        f"shadow_gate={raw['shadow_gate_pass']}. No validation64/test access. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`. "
        f"Backup required before further simulation.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists() and (OUT_DIR / "completed.json").exists():
        raise SystemExit("shadow scan already completed; refusing rerun")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (PROTOCOL_JSON, SMOKE_RAW, SMOKE_DONE, V1_RAW, V1_DONE):
        if not p.exists():
            raise DiagnosticError("missing required input: %s" % rel(p))
    verify_done(SMOKE_DONE)
    verify_done(V1_DONE)
    protocol = read_json(PROTOCOL_JSON)
    smoke = read_json(SMOKE_RAW)
    v1 = read_json(V1_RAW)
    for label, obj in (("smoke", smoke), ("v1", v1)):
        if obj.get("historical_validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise DiagnosticError(label + " access flags invalid")
    episodes = find_episode_list(v1)
    by_case_h, fixed_aggregate = summarize_fixed_grid(episodes)
    selector = protocol["selector"]
    shadow_cases: List[Dict[str, Any]] = []
    found_cases: List[int] = []
    missing_cases: List[int] = []
    for case in EXPECTED_CASES:
        ep = by_case_h.get(case, {}).get(15)
        if ep is None:
            missing_cases.append(case)
            shadow_cases.append({"case": case, "error": "missing_H15_episode_summary", "triggered": False})
            continue
        found_cases.append(case)
        shadow_cases.append(scan_trace(case, ep, selector))
    triggered_cases = sorted(int(r["case"]) for r in shadow_cases if r.get("triggered"))
    extra_trigger_cases = [c for c in triggered_cases if c not in POSITIVE_CASES]
    missing_state_cases = [int(r["case"]) for r in shadow_cases if r.get("missing_state_steps_before_trigger_or_end", 0) and not r.get("triggered")]
    smoke_per_case = (smoke.get("analysis") or {}).get("per_case") or {}
    case_comparisons: Dict[str, Any] = {}
    forecast_phys_delta = 0.0
    forecast_total_delta = 0.0
    unknown_trigger_cases: List[int] = []
    for case in EXPECTED_CASES:
        h15 = by_case_h.get(case, {}).get(15)
        h30 = by_case_h.get(case, {}).get(30)
        dphys = None
        dtotal = None
        if case in triggered_cases:
            sp = smoke_per_case.get(str(case), {})
            if "selector_vs_H15_physical_delta" in sp and "selector_vs_H15_total_delta" in sp:
                dphys = float(sp["selector_vs_H15_physical_delta"])
                dtotal = float(sp["selector_vs_H15_total_delta"])
                forecast_phys_delta += dphys
                forecast_total_delta += dtotal
            else:
                unknown_trigger_cases.append(case)
        case_comparisons[str(case)] = {
            "H15_physical": None if h15 is None else float(h15.get("physical_constraint_cost", 0.0) or 0.0),
            "H15_total": None if h15 is None else float(h15.get("total_cost", 0.0) or 0.0),
            "H30_physical": None if h30 is None else float(h30.get("physical_constraint_cost", 0.0) or 0.0),
            "H30_total": None if h30 is None else float(h30.get("total_cost", 0.0) or 0.0),
            "forecast_selector_delta_physical_vs_H15": dphys,
            "forecast_selector_delta_total_vs_H15": dtotal,
        }
    h15_agg = fixed_aggregate.get("15", {})
    h15_phys = float(h15_agg.get("physical_constraint_cost_sum", 0.0) or 0.0)
    h15_total = float(h15_agg.get("total_cost_sum", 0.0) or 0.0)
    shadow_gate_pass = bool(
        sorted(found_cases) == EXPECTED_CASES
        and not missing_cases
        and set(POSITIVE_CASES).issubset(set(triggered_cases))
        and not extra_trigger_cases
        and not unknown_trigger_cases
    )
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "vehicle_v1_state_selector_shadow_scan_v0_no_simulation",
        "classification": "development_diagnostic_no_simulation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {
            "protocol_json": {"path": rel(PROTOCOL_JSON), "sha256": sha256(PROTOCOL_JSON)},
            "smoke_raw": {"path": rel(SMOKE_RAW), "sha256": sha256(SMOKE_RAW)},
            "smoke_completed": {"path": rel(SMOKE_DONE), "sha256": sha256(SMOKE_DONE)},
            "v1_raw": {"path": rel(V1_RAW), "sha256": sha256(V1_RAW)},
            "v1_completed": {"path": rel(V1_DONE), "sha256": sha256(V1_DONE)},
        },
        "source_hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "selector": {"default_horizon": selector.get("default_horizon"), "trigger_horizon": selector.get("trigger_horizon"), "positive_radius": selector.get("positive_radius"), "negative_veto_radius": selector.get("negative_veto_radius"), "positive_prototype_count": len(selector.get("positive_prototypes") or []), "negative_veto_prototype_count": len(selector.get("negative_veto_prototypes") or [])},
        "v1_fixed_aggregate": fixed_aggregate,
        "h15_trace_cases_found": found_cases,
        "missing_h15_trace_cases": missing_cases,
        "missing_state_cases_without_trigger": missing_state_cases,
        "shadow_cases": shadow_cases,
        "triggered_cases": triggered_cases,
        "extra_trigger_cases": extra_trigger_cases,
        "unknown_trigger_cases_without_smoke_delta": unknown_trigger_cases,
        "case_comparisons": case_comparisons,
        "forecast": {
            "h15_physical_constraint_cost_sum": h15_phys,
            "h15_total_cost_sum": h15_total,
            "selector_physical_constraint_cost_sum": float(h15_phys + forecast_phys_delta),
            "selector_total_cost_sum": float(h15_total + forecast_total_delta),
            "forecast_physical_delta_vs_H15": float(forecast_phys_delta),
            "forecast_total_delta_vs_H15": float(forecast_total_delta),
            "assumption": "nontrigger cases remain identical to H15; triggered cases use already-observed smoke/controlled-continuation deltas; must be confirmed by online rollout",
        },
        "shadow_gate_pass": shadow_gate_pass,
    }
    raw["next_protocol"] = write_next_protocol(raw)
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 state-selector shadow scan v0 ({raw['created_utc']})\n\n"
        f"No-simulation diagnostic completed. Triggered cases={triggered_cases}; extra_trigger_cases={extra_trigger_cases}; shadow_gate_pass={shadow_gate_pass}. "
        f"No validation64/test access. Backup required before more simulation. Next protocol={raw.get('next_protocol')}.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, Path(__file__).resolve(), ROOT / raw["backup_request"]]
    if raw.get("next_protocol"):
        files += [NEXT_PROTOCOL_JSON, NEXT_PROTOCOL_MD]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "headline": {"shadow_gate_pass": shadow_gate_pass, "triggered_cases": triggered_cases, "extra_trigger_cases": extra_trigger_cases, "next_protocol": raw.get("next_protocol")},
        "backup_request": raw["backup_request"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "shadow_gate_pass": shadow_gate_pass, "triggered_cases": triggered_cases, "extra_trigger_cases": extra_trigger_cases, "next_protocol": raw.get("next_protocol"), "backup_request": raw["backup_request"], "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "hint": "No simulations were intended. Preserve failure and inspect trace schema before repair."})
        raise
