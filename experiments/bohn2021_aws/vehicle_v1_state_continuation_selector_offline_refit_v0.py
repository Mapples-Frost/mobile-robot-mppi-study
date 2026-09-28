#!/usr/bin/env python3
"""Offline refit diagnostic for a Vehicle V1 state-continuation selector.

This script performs a metadata/trace-only bounded diagnostic after the
controlled-continuation V0B result.  It does not construct environments, run MPC,
train/refit neural networks, open historical validation64, or access sealed test
content.

Scientific purpose
------------------
Controlled continuation confirmed sparse noninitial H30 opportunities from
identical H15 prefix states.  Before spending rollout budget on a new IMPROVED
selector smoke, choose an explicit state-level trigger rule and audit whether it
hits the known opportunity windows while avoiding guard/non-opportunity windows
on already-opened V1 development traces.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
CC_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/raw.json"
CC_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/completed.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
TARGET_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z/raw.json"
TARGET_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z/completed.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_offline_refit_v0_20260928T1240Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_state_continuation_selector_offline_refit_v0_20260928T1240Z.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0_frozen_20260928.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
BACKUP_REQUEST = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_V1_STATE_CONTINUATION_SELECTOR_OFFLINE_REFIT_V0_20260928T1240Z.json"
MARKER = "vehicle-v1-state-continuation-selector-offline-refit-v0-20260928T1240Z"

DEFAULT_H = 15
TRIGGER_H = 30
ANGLE_WEIGHT = 0.5
TARGET_WINDOW = 2
CONTROL_STEP_BOUND_NEXT = 2400
NEXT_MAX_EPISODES = 16


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=serial) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_from_mapping(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (safe_float(row.get("x"), float("nan")), safe_float(row.get("y"), float("nan")), safe_float(row.get("theta"), float("nan")))


def step_state(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    state = row.get("previous_state") or row.get("state") or {}
    return state_from_mapping(state)


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(v) for v in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + ANGLE_WEIGHT * abs(angle_diff(a[2], b[2])))


def values_summary(xs: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs)
    if not vals:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "sum": 0.0}
    n = len(vals)
    return {
        "count": n,
        "min": vals[0],
        "mean": float(math.fsum(vals) / n),
        "median": vals[n // 2] if n % 2 else float(0.5 * (vals[n // 2 - 1] + vals[n // 2])),
        "max": vals[-1],
        "sum": float(math.fsum(vals)),
    }


def verify_done(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("Missing completed marker: %s" % rel(path))
    done = read_json(path)
    if done.get("passed") is not True or done.get("hard_pass") is not True:
        raise ContractError("%s did not pass hard marker" % label)
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise ContractError("%s access flags invalid" % label)
    return done


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for path in (CC_RAW, CC_DONE, V1_RAW, V1_DONE, TARGET_RAW, TARGET_DONE):
        if not path.exists():
            raise ContractError("Required input missing: %s" % rel(path))
    cc_done = verify_done(CC_DONE, "controlled_continuation")
    v1_done = verify_done(V1_DONE, "fixed_H_opportunity_v1")
    target_done = verify_done(TARGET_DONE, "continuation_target")
    cc = read_json(CC_RAW)
    v1 = read_json(V1_RAW)
    target = read_json(TARGET_RAW)
    for label, obj in (("cc_raw", cc), ("v1_raw", v1), ("target_raw", target)):
        if obj.get("historical_validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise ContractError("%s access flags invalid" % label)
    if cc.get("analysis", {}).get("selector_smoke_gate_pass") is not True:
        raise ContractError("Controlled-continuation gate did not pass; selector refit is not justified")
    return cc, v1, {"cc_done": cc_done, "v1_done": v1_done, "target_done": target_done, "target_raw": target}


def episode_key(e: Mapping[str, Any]) -> Tuple[int, int, int]:
    return int(e["case"]), int(e["branch_step"]), int(e["branch_horizon"])


def extract_labeled_states(cc: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = {episode_key(e): e for e in cc.get("episodes") or []}
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    for row in cc.get("analysis", {}).get("state_rows", []):
        case_id = int(row["case"])
        branch_step = int(row["branch_step"])
        ref = by_key.get((case_id, branch_step, DEFAULT_H))
        if not ref:
            raise ContractError("Missing H15 reference episode for case=%d branch=%d" % (case_id, branch_step))
        state = state_from_mapping(ref["branch_previous_state"])
        best = row.get("best_total") or row.get("best_physical") or {}
        best_h = int(best.get("horizon", DEFAULT_H))
        label = {
            "case": case_id,
            "branch_step": branch_step,
            "state": {"x": state[0], "y": state[1], "theta": state[2]},
            "source": "controlled_continuation_v0b_H15_prefix_state",
            "h15_continuation_physical": safe_float(row["reference_H15"].get("continuation_physical")),
            "h15_continuation_total": safe_float(row["reference_H15"].get("continuation_total")),
            "best_horizon": best_h,
            "best_gain_physical_vs_H15": safe_float((row.get("best_physical") or {}).get("gain_vs_H15_physical")),
            "best_gain_total_vs_H15": safe_float((row.get("best_total") or {}).get("gain_vs_H15_total")),
            "confirmed_material_noninitial_state": bool(row.get("confirmed_material_noninitial_state")),
        }
        if bool(row.get("confirmed_material_noninitial_state")) and best_h == TRIGGER_H:
            positives.append(label)
        else:
            negatives.append(label)
    if len(positives) < 2:
        raise ContractError("Insufficient H30 positive prototypes for selector refit")
    return positives, negatives


def load_h15_traces(v1: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    traces: Dict[int, Dict[str, Any]] = {}
    for e in v1.get("episodes") or []:
        if int(e.get("horizon", -1)) != DEFAULT_H:
            continue
        case_id = int(e["case"])
        trace_path = ROOT / str(e["path"]) / "trace.json"
        if not trace_path.exists():
            raise ContractError("Missing H15 trace: %s" % rel(trace_path))
        trace = read_json(trace_path)
        traces[case_id] = {"episode": e, "trace": trace, "trace_path": trace_path}
    if len(traces) < 4:
        raise ContractError("Too few H15 V1 traces for offline selector audit")
    return traces


def min_distance(state: Tuple[float, float, float], prototypes: Sequence[Mapping[str, Any]]) -> Tuple[float, Optional[Mapping[str, Any]]]:
    best_d = float("inf")
    best_p: Optional[Mapping[str, Any]] = None
    for p in prototypes:
        ps = state_from_mapping(p["state"])
        d = state_distance(state, ps)
        if d < best_d:
            best_d, best_p = d, p
    return best_d, best_p


def trigger_at_state(state: Tuple[float, float, float], positives: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], radius: float, neg_veto_radius: float) -> Tuple[bool, Dict[str, Any]]:
    d_pos, p_pos = min_distance(state, positives)
    d_neg, p_neg = min_distance(state, negatives) if negatives else (float("inf"), None)
    veto = bool(neg_veto_radius > 0 and d_neg <= neg_veto_radius)
    fire = bool(d_pos <= radius and not veto)
    return fire, {
        "nearest_positive_distance": d_pos,
        "nearest_positive_case": None if p_pos is None else int(p_pos["case"]),
        "nearest_positive_step": None if p_pos is None else int(p_pos["branch_step"]),
        "nearest_negative_distance": d_neg,
        "nearest_negative_case": None if p_neg is None else int(p_neg["case"]),
        "nearest_negative_step": None if p_neg is None else int(p_neg["branch_step"]),
        "negative_veto": veto,
    }


def eval_rule(traces: Mapping[int, Mapping[str, Any]], positives: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], radius: float, neg_veto_radius: float) -> Dict[str, Any]:
    per_case: Dict[str, Any] = {}
    total_trigger_steps = 0
    latched_cases = 0
    for case_id, payload in sorted(traces.items()):
        events: List[Dict[str, Any]] = []
        for step, row in enumerate(payload["trace"]):
            fire, meta = trigger_at_state(step_state(row), positives, negatives, radius, neg_veto_radius)
            if fire:
                item = {"step": int(step)}
                item.update(meta)
                events.append(item)
        total_trigger_steps += len(events)
        if events:
            latched_cases += 1
        per_case[str(case_id)] = {
            "triggered": bool(events),
            "first_trigger_step": None if not events else int(events[0]["step"]),
            "trigger_step_count_before_latch": len(events),
            "events_first10": events[:10],
            "episode_steps": len(payload["trace"]),
        }
    positive_hits = 0
    positive_latched_by_target = 0
    premature_positive_latches = 0
    positive_target_details: List[Dict[str, Any]] = []
    for p in positives:
        case_id = int(p["case"])
        branch_step = int(p["branch_step"])
        row = per_case.get(str(case_id), {})
        first = row.get("first_trigger_step")
        events = [e for e in row.get("events_first10", []) if abs(int(e["step"]) - branch_step) <= TARGET_WINDOW]
        hit = bool(events or (first is not None and abs(int(first) - branch_step) <= TARGET_WINDOW))
        latched_by = bool(first is not None and int(first) <= branch_step + TARGET_WINDOW)
        premature = bool(first is not None and int(first) < branch_step - TARGET_WINDOW)
        positive_hits += int(hit)
        positive_latched_by_target += int(latched_by)
        premature_positive_latches += int(premature)
        positive_target_details.append({
            "case": case_id,
            "branch_step": branch_step,
            "first_trigger_step_on_H15_trace": first,
            "hit_within_window": hit,
            "latched_by_target_window": latched_by,
            "premature_before_window": premature,
        })
    guard_hits = 0
    guard_case_latches = 0
    guard_details: List[Dict[str, Any]] = []
    positive_cases = {int(p["case"]) for p in positives}
    for n in negatives:
        case_id = int(n["case"])
        branch_step = int(n["branch_step"])
        row = per_case.get(str(case_id), {})
        first = row.get("first_trigger_step")
        near = bool(first is not None and abs(int(first) - branch_step) <= TARGET_WINDOW)
        guard_hits += int(near)
        guard_case_latches += int(first is not None and case_id not in positive_cases)
        guard_details.append({
            "case": case_id,
            "branch_step": branch_step,
            "first_trigger_step_on_H15_trace": first,
            "triggered_near_guard_window": near,
            "guard_case_latched": bool(first is not None and case_id not in positive_cases),
        })
    nonpositive_case_latches = sum(1 for case_id, row in per_case.items() if int(case_id) not in positive_cases and row["triggered"])
    return {
        "radius": float(radius),
        "negative_veto_radius": float(neg_veto_radius),
        "positive_hits_within_window": int(positive_hits),
        "positive_latched_by_target_window": int(positive_latched_by_target),
        "premature_positive_latches": int(premature_positive_latches),
        "guard_window_hits": int(guard_hits),
        "guard_case_latches": int(guard_case_latches),
        "nonpositive_case_latches": int(nonpositive_case_latches),
        "latched_cases": int(latched_cases),
        "total_trigger_steps_before_latch": int(total_trigger_steps),
        "positive_target_details": positive_target_details,
        "guard_details": guard_details,
        "per_case": per_case,
    }


def choose_rule(grid: Sequence[Mapping[str, Any]], positive_count: int) -> Dict[str, Any]:
    # Prefer: all positive windows hit/latch, no guard/nonpositive latches, no premature positive latch.
    # Among ties choose the broadest radius that preserves safety, then smaller trigger count.
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            int(row["positive_hits_within_window"]),
            int(row["positive_latched_by_target_window"]),
            -int(row["guard_window_hits"]),
            -int(row["nonpositive_case_latches"]),
            -int(row["premature_positive_latches"]),
            -int(row["guard_case_latches"]),
            float(row["radius"]),
            -int(row["total_trigger_steps_before_latch"]),
            -float(row["negative_veto_radius"]),
        )
    chosen = max(grid, key=key)
    chosen = dict(chosen)
    chosen["offline_gate_pass"] = bool(
        int(chosen["positive_latched_by_target_window"]) >= min(2, positive_count)
        and int(chosen["guard_window_hits"]) == 0
        and int(chosen["nonpositive_case_latches"]) == 0
        and int(chosen["premature_positive_latches"]) == 0
    )
    chosen["selection_key_definition"] = "max positive hits/latches, then minimize guard/nonpositive/premature triggers, then prefer broader radius and fewer trigger steps"
    return chosen


def latest_verified_backup() -> Optional[Dict[str, Any]]:
    rows: List[Tuple[dt.datetime, Path, Dict[str, Any]]] = []
    for path in BACKUP_DIR.glob("backup_proof_*.json"):
        try:
            obj = read_json(path)
        except Exception:
            continue
        if not (obj.get("backup_verified") is True or obj.get("status") == "verified"):
            continue
        if int(obj.get("remaining_changed_files", -1)) != 0:
            continue
        t = None
        for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
            t = parse_time(obj.get(key))
            if t is not None:
                break
        if t is None:
            continue
        rows.append((t, path, obj))
    if not rows:
        return None
    t, path, obj = max(rows, key=lambda x: x[0])
    return {
        "path": rel(path),
        "sha256": sha256(path),
        "time": t.isoformat(),
        "commit": obj.get("commit"),
        "remaining_changed_files": obj.get("remaining_changed_files"),
        "package_sha256": obj.get("package_sha256"),
        "packages_this_run": obj.get("packages_this_run"),
        "status": obj.get("status"),
    }


def write_protocol(created: str, chosen: Mapping[str, Any], positives: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], v1: Mapping[str, Any]) -> Dict[str, Any]:
    smoke_cases = sorted({int(p["case"]) for p in positives} | {5, 12})
    comparator_horizons = [10, 15, 30]
    protocol = {
        "protocol_id": "vehicle_v1_state_continuation_selector_smoke_v0_frozen_20260928",
        "created_utc": created,
        "classification": "development_IMPROVED_state_selector_smoke_not_validation_not_final_test",
        "motivation": [
            "Controlled continuation confirmed material H30 gains from identical noninitial H15 prefix states.",
            "Metadata selectors failed; this freezes a state-level latch rule before collecting selector rollouts.",
            "The rule is intentionally simple and overfitting-prone; success only authorizes broader refit/training, not final claims.",
        ],
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "requires_verified_external_backup_after_this_protocol_before_rollout": True,
        },
        "selector": {
            "name": "nearest_positive_state_H30_latch_v0",
            "default_horizon": DEFAULT_H,
            "trigger_horizon": TRIGGER_H,
            "latch_after_first_trigger_until_termination": True,
            "feature_space": "previous_state=(x,y,theta) from online environment/controller state before MPC decision",
            "distance": "sqrt((x-xp)^2+(y-yp)^2) + %.3f*abs(wrapped(theta-thetap))" % ANGLE_WEIGHT,
            "positive_radius": float(chosen["radius"]),
            "negative_veto_radius": float(chosen["negative_veto_radius"]),
            "positive_prototypes": list(positives),
            "negative_veto_prototypes": list(negatives),
            "terminal_switch": "use H15 terminal before trigger; after trigger switch to H30 terminal/controller, reporting switch overhead separately",
            "gradient_updates": 0,
            "training_episodes": 0,
            "source_of_labels": "controlled-continuation V0B development traces only",
        },
        "split_and_budget": {
            "source_bank": "vehicle_fixed_h_opportunity_probe_v1_fresh64_select16_development_only",
            "rollout_cases": smoke_cases,
            "comparators": ["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"],
            "max_episodes": NEXT_MAX_EPISODES,
            "control_step_upper_bound": CONTROL_STEP_BOUND_NEXT,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "primary_checks": {
            "safety": "No selector constraint episode, no solver-failure-step increase vs paired H15, no failure where H15 succeeds.",
            "opportunity": "Selector must trigger H30 on known opportunity cases 7 and/or 10 early enough to improve paired physical or total cost vs H15 on at least one opportunity case without large guard-case regression.",
            "timing": "Report whole-decision, solver, selection and terminal-switch timing; do not infer compute from horizon count.",
            "scope": "Development smoke only. If passed, freeze broader refit/training across >=3 independent seeds before validation/test.",
        },
        "input_hashes": {
            rel(CC_RAW): sha256(CC_RAW),
            rel(CC_DONE): sha256(CC_DONE),
            rel(V1_RAW): sha256(V1_RAW),
            rel(V1_DONE): sha256(V1_DONE),
        },
    }
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle V1 state-continuation selector smoke v0 frozen protocol",
        "",
        f"Frozen UTC: `{created}`.",
        "",
        "Development-only IMPROVED selector smoke; no validation64-bank or sealed-test access. A verified external backup after this protocol and the offline-refit source/output is required before any rollout.",
        "",
        "## Candidate selector",
        "",
        f"Default H{DEFAULT_H}; if previous-state distance to a positive prototype is <= `{chosen['radius']}` and not inside the negative veto radius `{chosen['negative_veto_radius']}`, switch/latch to H{TRIGGER_H} until termination.",
        "",
        "Positive prototypes are the controlled-continuation confirmed H30 states; negative prototypes are nonconfirmed branch states. This is a deterministic refit/search artifact with 0 gradient updates, not ORIGINAL SAC.",
        "",
        "## First rollout budget",
        "",
        f"- Cases: `{smoke_cases}`",
        "- Arms: selector_latch_H30 plus fixed H15/H30/H10 references",
        f"- Max episodes: `{NEXT_MAX_EPISODES}`; control-step upper bound: `{CONTROL_STEP_BOUND_NEXT}`",
        "- Training episodes / gradient steps: `0 / 0`",
        "",
        "## Acceptance for expansion",
        "",
        protocol["primary_checks"]["safety"],
        protocol["primary_checks"]["opportunity"],
        protocol["primary_checks"]["timing"],
    ]
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return protocol


def write_summary(raw: Mapping[str, Any]) -> None:
    chosen = raw["selected_rule"]
    lines = [
        "# Vehicle V1 state-continuation selector offline refit v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Metadata/trace-only diagnostic: no environment construction, no rollouts, no training/refit episodes, no gradient updates, no validation64-bank access, no sealed-test access.",
        "",
        "## Result",
        "",
        f"- Offline gate pass for next selector smoke: `{chosen['offline_gate_pass']}`.",
        f"- Selected positive radius: `{chosen['radius']}`; negative veto radius: `{chosen['negative_veto_radius']}`.",
        f"- Positive targets latched by window: `{chosen['positive_latched_by_target_window']}` / `{len(raw['positive_prototypes'])}`; guard-window hits: `{chosen['guard_window_hits']}`; nonpositive-case latches: `{chosen['nonpositive_case_latches']}`; premature positive latches: `{chosen['premature_positive_latches']}`.",
        f"- Latch cases on existing H15 traces: `{chosen['latched_cases']}` / `{raw['h15_trace_case_count']}`; trigger steps before latch accounting: `{chosen['total_trigger_steps_before_latch']}`.",
        "",
        "## Positive target details",
        "",
        "| case | target step | first trigger on H15 trace | hit window | latched by target | premature |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in chosen["positive_target_details"]:
        lines.append("| %d | %d | %s | `%s` | `%s` | `%s` |" % (
            row["case"], row["branch_step"], str(row["first_trigger_step_on_H15_trace"]), row["hit_within_window"], row["latched_by_target_window"], row["premature_before_window"]
        ))
    lines += [
        "",
        "## Guard target details",
        "",
        "| case | guard step | first trigger on H15 trace | trigger near guard | guard-case latch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in chosen["guard_details"]:
        lines.append("| %d | %d | %s | `%s` | `%s` |" % (
            row["case"], row["branch_step"], str(row["first_trigger_step_on_H15_trace"]), row["triggered_near_guard_window"], row["guard_case_latched"]
        ))
    lines += [
        "",
        "## Interpretation",
        "",
        "The offline rule is deliberately small and development-data-derived. Passing this gate only means the next bounded rollout is informative; it is not validation evidence and does not establish generalization. If the rollout passes, broader refit/training across independent seeds is still required.",
        "",
        f"Frozen next-smoke protocol: `{raw['next_protocol']['json_path']}`.",
        f"Backup request before any rollout simulation: `{raw['backup_request']}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 state-continuation selector offline refit v0\n\n"
        f"UTC: {raw['created_utc']}. Metadata/trace-only offline selector refit completed with no simulations/training/validation64/test access. "
        f"Offline gate={raw['selected_rule']['offline_gate_pass']}; selected radius={raw['selected_rule']['radius']}; "
        f"positive latches={raw['selected_rule']['positive_latched_by_target_window']}/{len(raw['positive_prototypes'])}; "
        f"guard hits={raw['selected_rule']['guard_window_hits']}; nonpositive latches={raw['selected_rule']['nonpositive_case_latches']}. "
        f"Next frozen smoke protocol: `{rel(PROTOCOL_JSON)}`; backup required before rollout.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists():
        completed = OUT_DIR / "completed.json"
        if completed.exists():
            print(json.dumps({"already_completed": rel(completed)}, sort_keys=True))
            return 0
        leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise ContractError("Partial offline-refit output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:20]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    cc, v1, done_bundle = verify_inputs()
    positives, negatives = extract_labeled_states(cc)
    traces = load_h15_traces(v1)
    radii = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    neg_vetos = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    grid: List[Dict[str, Any]] = []
    for r in radii:
        for nv in neg_vetos:
            grid.append(eval_rule(traces, positives, negatives, r, nv))
    chosen = choose_rule(grid, len(positives))
    protocol = write_protocol(created, chosen, positives, negatives, v1)
    latest_backup = latest_verified_backup()
    backup_needed_before_rollout = True
    write_json(BACKUP_REQUEST, {
        "requested_utc": created,
        "reason": "backup offline selector refit source/output and frozen selector-smoke protocol before any selector rollout simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(Path(__file__).resolve())],
        "latest_verified_backup_seen_by_script": latest_backup,
    })
    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_v1_state_continuation_selector_offline_refit_v0_no_simulation",
        "classification": "development_IMPROVED_selector_refit_metadata_trace_only_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "budget_declared": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "existing_h15_traces_read": len(traces)},
        "positive_prototypes": positives,
        "negative_guard_prototypes": negatives,
        "h15_trace_case_count": len(traces),
        "threshold_grid": grid,
        "selected_rule": chosen,
        "next_protocol": {"path": rel(PROTOCOL_MD), "sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "full": protocol},
        "input_hashes": {
            rel(CC_RAW): sha256(CC_RAW), rel(CC_DONE): sha256(CC_DONE), rel(V1_RAW): sha256(V1_RAW), rel(V1_DONE): sha256(V1_DONE), rel(TARGET_RAW): sha256(TARGET_RAW), rel(TARGET_DONE): sha256(TARGET_DONE)
        },
        "source_hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "latest_verified_backup_seen_by_script": latest_backup,
        "backup_needed_before_rollout": backup_needed_before_rollout,
        "backup_request": rel(BACKUP_REQUEST),
        "interpretation_limits": [
            "offline development refit only",
            "positive labels derive from already-opened controlled-continuation branches",
            "H15 traces are used only to estimate trigger timing/overtrigger before any selector rollout",
            "not gradient RL training",
            "not validation or final-test evidence",
            "may overfit sparse case7/case10 prototypes; fresh confirmation is mandatory",
        ],
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 state-continuation selector offline refit v0 state ({created})\n\n"
        f"No simulations/training/validation64/test access. Offline gate={chosen['offline_gate_pass']}; radius={chosen['radius']}; "
        f"positive_latched={chosen['positive_latched_by_target_window']}/{len(positives)}; guard_hits={chosen['guard_window_hits']}; "
        f"nonpositive_case_latches={chosen['nonpositive_case_latches']}. Next action: obtain verified external backup covering `{rel(Path(__file__).resolve())}`, `{rel(OUT_DIR)}`, `{rel(PROTOCOL_JSON)}`, then run the bounded selector smoke protocol if backup is verified.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, PROTOCOL_JSON, PROTOCOL_MD, BACKUP_REQUEST, Path(__file__).resolve()]
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
        "offline_gate_pass": bool(chosen["offline_gate_pass"]),
        "selected_radius": float(chosen["radius"]),
        "selected_negative_veto_radius": float(chosen["negative_veto_radius"]),
        "backup_request": rel(BACKUP_REQUEST),
        "next_protocol": rel(PROTOCOL_JSON),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "offline_gate_pass": bool(chosen["offline_gate_pass"]),
        "radius": float(chosen["radius"]),
        "negative_veto_radius": float(chosen["negative_veto_radius"]),
        "positive_latched_by_target_window": int(chosen["positive_latched_by_target_window"]),
        "guard_window_hits": int(chosen["guard_window_hits"]),
        "nonpositive_case_latches": int(chosen["nonpositive_case_latches"]),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "backup_request": rel(BACKUP_REQUEST),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
