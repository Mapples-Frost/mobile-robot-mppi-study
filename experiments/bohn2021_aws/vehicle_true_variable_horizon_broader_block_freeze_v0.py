#!/usr/bin/env python3
"""No-simulation freeze for a broader true variable-H development block.

This follows the v0b true-controller smoke: true H10/H15 construction reduced
NLP dimension and measured solve/decision time, but the cold branch smoke did
not retain the earlier fixed-size common-prefix physical gains.  The next useful
step is therefore not selector/refit yet, but a bounded paired development
protocol that tests whether a real control-vs-compute tradeoff survives with a
true controller cache, matched/shared terminal profiles, repeated randomized
measurement, and controls/negatives.

This script runs no simulations, opens no validation64/sealed-test banks, and
performs no training/refit.  It only inspects already-opened development
artifacts, freezes the next protocol, and writes a backup request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_broader_block_freeze_v0"
STAMP = "20260929T0650Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/vehicle_true_variable_horizon_broader_block_v0_frozen_{STAMP}.json"
BACKUP_REQ = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_BROADER_BLOCK_V0_RUN_{STAMP}.json"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V0B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/completed.json"
V0B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/raw.json"
V0_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_case5_smoke_v0_frozen_20260929T0640Z.json"
V1E_CASE5_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z/raw.json"
V1E_TARGETED_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z/raw.json"
V1D_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json"
STAGE2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json"
CONTINUE_STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T0650_after_true_variable_horizon_v0b_success.md"

TRUE_HORIZONS = [10, 15, 25]
TERMINAL_PROFILES = ["matched_terminal", "shared_h15_terminal"]
REPEATS = 2
MAX_TARGETS = 4
MAX_BRANCH_STEPS = 150
RNG_SEED = 202609290650
MATERIAL_GAIN = 3.0


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def median(xs: Sequence[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def physical_cost(e: Mapping[str, Any]) -> float:
    for k in ("continuation_physical_constraint_cost_from_branch", "physical_constraint_cost"):
        if k in e:
            return sf(e.get(k), 0.0)
    return sf(e.get("performance"), 0.0) + sf(e.get("constraint"), 0.0)


def is_safe(e: Mapping[str, Any]) -> bool:
    return bool(e.get("success")) and not bool(e.get("constraint")) and si(e.get("solver_failure_steps", 0), 999) == 0


def horizon(e: Mapping[str, Any]) -> Optional[int]:
    for k in ("branch_horizon", "true_mpc_n_horizon", "commanded_horizon", "horizon", "executed_horizon"):
        if k in e:
            h = si(e.get(k), -1)
            return h if h > 0 else None
    return None


def term(e: Mapping[str, Any]) -> str:
    return str(e.get("terminal_mode") or e.get("terminal_profile") or "unknown")


def walk_records(obj: Any, source: str, out: List[Dict[str, Any]]) -> None:
    if isinstance(obj, Mapping):
        if obj.get("state_id") is not None and isinstance(obj.get("branch_previous_state"), Mapping) and horizon(obj) is not None:
            rec = dict(obj)
            rec["_source"] = source
            out.append(rec)
        for v in obj.values():
            walk_records(v, source, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_records(v, source, out)


def collect_branch_records(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for p in paths:
        if not p.exists():
            missing.append(rel(p))
            continue
        try:
            walk_records(read_json(p), rel(p), records)
        except Exception as exc:
            missing.append(rel(p) + " read_error=" + repr(exc))
    # De-duplicate exact source/state/horizon/terminal/repeat records conservatively.
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in records:
        key = (r.get("_source"), r.get("state_id"), horizon(r), term(r), r.get("diagnostic_repeat"), r.get("execution_index"), r.get("path"))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique, missing


def target_from_record(r: Mapping[str, Any], role: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    bs = r.get("branch_previous_state") or {}
    return {
        "state_id": str(r.get("state_id")),
        "role": role,
        "case": si(r.get("case")),
        "source_candidate_index": si(r.get("source_candidate_index"), -1),
        "selection_group": r.get("selection_group"),
        "case_role": r.get("case_role") or r.get("target_role"),
        "branch_step": si(r.get("branch_step")),
        "branch_previous_state": {"x": sf((bs.get("x") or [bs.get("x")])[0] if isinstance(bs.get("x"), list) else bs.get("x")), "y": sf((bs.get("y") or [bs.get("y")])[0] if isinstance(bs.get("y"), list) else bs.get("y")), "theta": sf((bs.get("theta") or [bs.get("theta")])[0] if isinstance(bs.get("theta"), list) else bs.get("theta"))},
        "source_record": r.get("_source"),
        "source_path": r.get("path"),
        "development_metrics_from_existing_artifacts": dict(metrics),
    }


def score_states(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_sid: Dict[str, List[Mapping[str, Any]]] = {}
    for r in records:
        if term(r) != "h15_common_terminal":
            continue
        by_sid.setdefault(str(r.get("state_id")), []).append(r)
    info: Dict[str, Dict[str, Any]] = {}
    for sid, rs in by_sid.items():
        h15 = [r for r in rs if horizon(r) == 15]
        h10 = [r for r in rs if horizon(r) == 10]
        if not h15:
            continue
        ref = sorted(h15, key=lambda r: (si(r.get("diagnostic_repeat"), 999), si(r.get("execution_index"), 999)))[0]
        gain = None
        if h10:
            gain = median([physical_cost(a) for a in h15]) - median([physical_cost(a) for a in h10])  # type: ignore[operator]
        safe10 = all(is_safe(r) for r in h10) if h10 else False
        safe15 = all(is_safe(r) for r in h15)
        info[sid] = {
            "ref": ref,
            "case": si(ref.get("case")),
            "branch_step": si(ref.get("branch_step")),
            "h10_count": len(h10),
            "h15_count": len(h15),
            "median_h10_vs_h15_physical_gain_existing": gain,
            "safe_h10_existing": safe10,
            "safe_h15_existing": safe15,
            "available_horizons": sorted(set(horizon(r) for r in rs if horizon(r) is not None)),
        }
    return info


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if marker not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--i-accept-no-simulation-broader-true-variable-horizon-freeze", action="store_true")
    args = ap.parse_args(argv)
    if not args.freeze or not args.i_accept_no_simulation_broader_true_variable_horizon_freeze:
        raise SystemExit("requires --freeze and explicit no-simulation acknowledgement")
    required = [V0B_DONE, V0B_RAW, V0_PROTOCOL, CONTINUE_STATE]
    missing_required = [rel(p) for p in required if not p.exists()]
    if missing_required:
        raise SystemExit("missing required v0b artifacts: " + ", ".join(missing_required))
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()) and not (OUT / "completed.json").exists():
        raise SystemExit("partial output exists; inspect first: " + rel(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    created = now_utc()
    v0b_done = read_json(V0B_DONE)
    v0b_raw = read_json(V0B_RAW)
    v0_protocol = read_json(V0_PROTOCOL)
    if v0b_done.get("validation64_bank_opened") is not False or v0b_done.get("sealed_test_accessed") is not False:
        raise SystemExit("v0b access flags invalid")
    if not ((v0b_done.get("headline") or {}).get("pass_to_broader_variable_horizon_block") is True):
        raise SystemExit("v0b did not pass the implementation timing gate; broader protocol should not be frozen")

    records, missing_optional = collect_branch_records([V1E_TARGETED_RAW, V1E_CASE5_RAW, V1D_RAW, STAGE2_RAW])
    state_info = score_states(records)
    mandatory_ids = [str(t.get("state_id")) for t in (v0_protocol.get("targets") or [])]
    selected: List[Dict[str, Any]] = []
    used = set()
    for t in (v0_protocol.get("targets") or []):
        sid = str(t.get("state_id"))
        metrics = state_info.get(sid, {}).copy()
        selected.append({
            "state_id": sid,
            "role": "mandatory_v0b_case5_timing_positive_physical_unresolved",
            "case": si(t.get("case")),
            "source_candidate_index": si(t.get("source_candidate_index"), -1),
            "selection_group": t.get("selection_group"),
            "case_role": t.get("case_role"),
            "branch_step": si(t.get("branch_step")),
            "branch_previous_state": t.get("branch_previous_state"),
            "source_record": rel(V0_PROTOCOL),
            "source_path": t.get("parent_h15_common_terminal_episode_path"),
            "development_metrics_from_existing_artifacts": metrics,
        })
        used.add(sid)
    positives = []
    controls = []
    for sid, inf in state_info.items():
        if sid in used:
            continue
        gain = inf.get("median_h10_vs_h15_physical_gain_existing")
        if gain is not None and float(gain) >= MATERIAL_GAIN and inf.get("safe_h10_existing") and inf.get("safe_h15_existing"):
            positives.append((sid, inf))
        else:
            controls.append((sid, inf))
    positives.sort(key=lambda kv: (si(kv[1].get("case")) in [5], -sf(kv[1].get("median_h10_vs_h15_physical_gain_existing"), -1e9), si(kv[1].get("case")), si(kv[1].get("branch_step"))))
    controls.sort(key=lambda kv: (si(kv[1].get("case")) in [5], abs(sf(kv[1].get("median_h10_vs_h15_physical_gain_existing"), 0.0)), si(kv[1].get("case")), si(kv[1].get("branch_step"))))
    for sid, inf in positives:
        if len(selected) >= MAX_TARGETS:
            break
        selected.append(target_from_record(inf["ref"], "additional_existing_terminal_stable_positive_if_available", inf))
        used.add(sid)
    for sid, inf in controls:
        if len(selected) >= MAX_TARGETS:
            break
        if sid in used:
            continue
        selected.append(target_from_record(inf["ref"], "negative_or_control_state_for_specificity", inf))
        used.add(sid)

    rng = random.Random(RNG_SEED)
    schedule: List[Dict[str, Any]] = []
    exe = 0
    for rep in range(REPEATS):
        arms = []
        for t in selected:
            for profile in TERMINAL_PROFILES:
                for h in TRUE_HORIZONS:
                    arms.append({"repeat": rep, "state_id": t["state_id"], "case": t["case"], "branch_step": t["branch_step"], "true_mpc_n_horizon": h, "terminal_profile": profile})
        rng.shuffle(arms)
        for arm in arms:
            arm["execution_index"] = exe
            exe += 1
            schedule.append(arm)
    budget_episodes = len(schedule)
    budget_steps = budget_episodes * MAX_BRANCH_STEPS

    v0b_pairs = (v0b_raw.get("analysis") or {}).get("pair_rows") or []
    v0b_physical_gains = [sf(p.get("physical_gain_H10_vs_H15")) for p in v0b_pairs]
    protocol = {
        "protocol_id": f"vehicle_true_variable_horizon_broader_block_v0_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_true_variable_horizon_broader_block_freeze_no_simulation",
        "hypothesis": "True per-H controller construction restores measured compute savings, but useful adaptive horizon control requires those savings to persist with no safety loss and acceptable physical cost on more than two cold-start states. A small randomized repeated true-H block should distinguish a usable control/compute tradeoff from a merely faster but physically worse H10 path.",
        "before_evidence": {
            "v0b_dimension_and_timing_pass": v0b_done.get("headline"),
            "v0b_pair_physical_gains_H10_vs_H15": v0b_physical_gains,
            "interpretation": "implementation speed path is viable locally; physical/control tradeoff remains unresolved and cannot justify selector/refit yet",
        },
        "target_selection": {
            "max_targets": MAX_TARGETS,
            "selected_count": len(selected),
            "mandatory_v0b_state_ids": mandatory_ids,
            "branch_records_scanned": len(records),
            "state_infos_available": len(state_info),
            "missing_optional_inputs": missing_optional,
            "selection_rule": "keep the two v0b case5 states, add existing terminal-stable non-mandatory positives if available, then controls/negatives for specificity; all from already-opened development artifacts only",
            "targets": selected,
        },
        "arms": {"true_horizons": TRUE_HORIZONS, "terminal_profiles": TERMINAL_PROFILES, "repeats": REPEATS, "randomization_seed": RNG_SEED, "schedule": schedule},
        "budget_declared": {"development_branch_episodes_exact": budget_episodes, "development_control_step_upper_bound": budget_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "acceptance_before_learning_or_selector_refit": {
            "implementation": "observed opt_x dimensions increase with true horizon for H10 < H15 < H25 except documented do-mpc constants",
            "safety": "no success/constraint/initial/final/solver-failure regression for shorter H versus H15/H25 within paired state/profile blocks",
            "compute": "paired repeated median solver and whole-decision savings for H10 vs H15 >=15%; report H10 vs H25 and p95 separately",
            "physical_tradeoff": "do not require strict physical improvement everywhere; require H10 physical cost no worse than H15 by more than max(2 cost units, 5%) on at least half of non-control states and identify strict-improvement states separately",
            "if_pass": "then design a selector/value-modeling experiment with true controller cache across >=3 training seeds and fair fixed-H baselines",
            "if_fail": "pivot to terminal-value/modeling/objective or scenario-opportunity redesign; do not repeat sparse label-density sweeps",
        },
        "access_rules": {"development_only": True, "validation64_bank_opened": False, "sealed_test_accessed": False, "requires_verified_backup_before_any_run": True},
    }
    write_json(PROTOCOL, protocol)
    write_json(BACKUP_REQ, {"requested_utc": created.isoformat(), "reason": "backup v0b smoke and broader true-variable-H frozen protocol before any further simulation", "backup_required_before_more_simulations": True, "artifacts": [rel(V0B_RAW), rel(V0B_DONE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQ)], "planned_development_branch_episodes_exact": budget_episodes, "planned_development_control_step_upper_bound": budget_steps, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})

    raw = {"created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "method": NAME, "classification": "metadata_no_simulation_protocol_freeze", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "v0b_headline": v0b_done.get("headline"), "v0b_pair_physical_gains_H10_vs_H15": v0b_physical_gains, "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "selected_targets": selected, "schedule_episode_count": budget_episodes, "control_step_cap": budget_steps, "state_info_count": len(state_info), "missing_optional_inputs": missing_optional, "next_action_after_backup": "run a bounded true-variable-H broader block runner/smoke implementing this frozen protocol; no validation64/test; then decide whether selector/value learning is warranted"}
    write_json(OUT / "raw.json", raw)
    summary = [
        "# Vehicle true variable-H broader block freeze v0",
        "",
        f"UTC: `{created.isoformat()}`. Metadata-only protocol freeze; no simulations/training/refit, no validation64 bank, no sealed test.",
        "",
        "## Why this, not another label-density sweep",
        "",
        "v0b established that true per-H controller construction can reduce optimizer dimension and measured solve/decision time, resolving the fixed-size AHMPC masking hypothesis. However the same cold-start smoke did not retain physical gains, so the discriminating question is now whether a broader true-H controller-cache block shows a real control/compute tradeoff rather than only lower computation.",
        "",
        f"- v0b headline: `{v0b_done.get('headline')}`.",
        f"- v0b physical gains H10 vs H15: `{v0b_physical_gains}`.",
        f"- Frozen targets: `{[t['state_id'] for t in selected]}`.",
        f"- Planned arms: horizons `{TRUE_HORIZONS}`, terminal profiles `{TERMINAL_PROFILES}`, repeats `{REPEATS}`; episodes `{budget_episodes}`, control-step cap `{budget_steps}`.",
        "",
        "## Decision",
        "",
        "Do not train/refit yet. After backup, run the frozen broader development block if resources permit. If speed persists but physical cost regresses, pivot to terminal-value/modeling/objective or scenario-opportunity design before selector learning.",
        "",
        f"Protocol: `{rel(PROTOCOL)}`. Backup request: `{rel(BACKUP_REQ)}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    marker = f"vehicle-true-variable-horizon-broader-block-freeze-v0-{STAMP}"
    append_docs(f"""<!-- {marker} -->
## 2026-09-29 vehicle true variable-H broader block freeze v0

UTC: {created.isoformat()}. Metadata-only/no-simulation protocol freeze completed after v0b. v0b proved true-H construction speed feasibility but physical tradeoff remains unresolved (H10-vs-H15 physical gains {v0b_physical_gains}). Frozen next development protocol `{rel(PROTOCOL)}` has {budget_episodes} planned branch episodes, cap {budget_steps} control steps, horizons {TRUE_HORIZONS}, terminal profiles {TERMINAL_PROFILES}, repeats {REPEATS}, validation64 closed and sealed test closed. More simulation is blocked until backup covers `{rel(BACKUP_REQ)}` and all v0b/freeze artifacts.
""", marker)
    files = [OUT / "raw.json", OUT / "summary.md", STATE, PROTOCOL, BACKUP_REQ, V0B_DONE, V0B_RAW, V0_PROTOCOL, CONTINUE_STATE] + [p for p in [V1E_TARGETED_RAW, V1E_CASE5_RAW, V1D_RAW, STAGE2_RAW] if p.exists()]
    completed = {"passed": True, "hard_pass": True, "created_utc": created.isoformat(), "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(), "classification": "metadata_no_simulation_protocol_freeze", "validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "candidate_pool_resets": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "new_refit_steps": 0, "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "planned_development_branch_episodes_exact": budget_episodes, "planned_development_control_step_upper_bound": budget_steps, "headline": {"v0b_true_h_speed_passed": True, "physical_tradeoff_unresolved": True, "selected_target_count": len(selected), "planned_episodes": budget_episodes, "train_or_refit_now": False, "next_after_backup": "broader_true_variable_H_development_block"}, "hashes": {rel(p): sha256(p) for p in files if p.exists()}}
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQ), "selected_target_count": len(selected), "planned_episodes": budget_episodes, "control_step_cap": budget_steps, "new_rollouts": 0, "new_control_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
