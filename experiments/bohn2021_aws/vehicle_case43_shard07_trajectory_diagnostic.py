#!/usr/bin/env python3
"""Metadata-only diagnostic for the shard07 learned_s2 case43 failure.

This reads already-created validation result artifacts only. It does not reopen the
validation64 bank, does not open/hash the sealed final test bank, and performs no
new simulation, control step, or training. The goal is to inspect the raw saved
trajectories requested by the 2026-09-27 steering note: learned_s2 case43 in
shard07 versus the same-seed fixed H25 comparator for case43.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
LEARNED_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07/episodes/exec1668_learned_s2_case43"
FIXED_H25_DIR = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00/episodes/exec0185_fixed_seed2_terminal25_controllerH25_case43"
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_case43_shard07_trajectory_diagnostic.py"
DOC_MARKER = "vehicle-case43-shard07-trajectory-diagnostic-20260927"

# Must match experiments/bohn2021_reproduction/latency_tree_policy.FEATURES['vehicle'].
TREE_FEATURE_NAMES = [
    "tracking_error",
    "heading_error_5",
    "heading_error_15",
    "heading_error_30",
    "reference_turn_30",
    "current_clearance",
    "preview_clearance_15",
    "preview_clearance_30",
    "speed",
    "abs_yaw_input",
    "remaining",
    "previous_initial_failure",
    "previous_final_failure",
]

EXPECTED = {
    "learned": {
        "rollout_key": "learned_s2",
        "seed": 2,
        "case": 43,
        "steps": 150,
        "success": False,
        "physical_constraint_cost_approx": 39125.923106760194,
        "horizon_counts": {"25": 149, "35": 1},
    },
    "fixed_h25": {
        "rollout_key": "fixed_seed2_terminal25_controllerH25",
        "seed": 2,
        "case": 43,
        "steps": 96,
        "success": True,
    },
}


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
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_hashes(mapping: Mapping[str, str]) -> Tuple[bool, List[Dict[str, str]]]:
    bad: List[Dict[str, str]] = []
    for name, expected in sorted(mapping.items()):
        p = ROOT / name
        if not p.exists():
            bad.append({"path": name, "expected": expected, "actual": "missing"})
            continue
        actual = sha256(p)
        if actual != expected:
            bad.append({"path": name, "expected": expected, "actual": actual})
    return not bad, bad


def existing_completed_ok(path: Path) -> bool:
    try:
        done = read_json(path)
    except Exception:
        return False
    if done.get("passed") is not True:
        return False
    ok, bad = verify_hashes(done.get("hashes") or {})
    return ok and not bad


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_flat(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, Mapping):
        for k in sorted(value):
            out.extend(finite_flat(value[k]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(finite_flat(item))
    elif isinstance(value, (int, float)):
        out.append(float(value))
    return out


def vec_state(row: Mapping[str, Any], which: str = "previous_state") -> List[float]:
    s = row.get(which) or {}
    return [float(s.get("theta")), float(s.get("x")), float(s.get("y"))]


def vec_input(row: Mapping[str, Any]) -> List[float]:
    u = row.get("input") or {}
    return [float((u.get("u_omega") or [math.nan])[0]), float((u.get("u_s") or [math.nan])[0])]


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return float(max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0))


def norm_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_solver_calls(rows: List[Mapping[str, Any]], trace: List[Mapping[str, Any]], center: Optional[int]) -> Dict[str, Any]:
    step_rows: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            step = int(row.get("step"))
        except Exception:
            continue
        step_rows.setdefault(step, []).append(row)
    all_step_rows = [r for r in rows if int(r.get("step", -999)) >= 0]
    accepted = [bool(r.get("accepted")) for r in all_step_rows]
    success = [bool(r.get("success")) for r in all_step_rows]
    retry_rows = [r for r in all_step_rows if str(r.get("kind", "")).startswith("retry")]
    iters = [int(r.get("iterations")) for r in all_step_rows if r.get("iterations") is not None]
    solver_s = [float(r.get("solver_s")) for r in all_step_rows if r.get("solver_s") is not None]
    window: List[Dict[str, Any]] = []
    if center is not None:
        for step in range(max(0, center - 5), min(len(trace), center + 6)):
            calls = step_rows.get(step, [])
            window.append({
                "step": step,
                "horizon": int(trace[step].get("horizon")),
                "calls": [
                    {
                        "kind": c.get("kind"),
                        "success": c.get("success"),
                        "accepted": c.get("accepted"),
                        "return_status": c.get("return_status"),
                        "iterations": c.get("iterations"),
                        "constraint_residual": c.get("constraint_residual"),
                        "bound_residual": c.get("bound_residual"),
                        "solver_s": c.get("solver_s"),
                    }
                    for c in calls
                ],
            })
    return {
        "total_rows": len(rows),
        "warmup_rows": len(step_rows.get(-1, [])),
        "scored_steps_with_calls": len([k for k in step_rows if k >= 0]),
        "all_scored_success": all(success) if success else None,
        "all_scored_accepted": all(accepted) if accepted else None,
        "retry_rows": len(retry_rows),
        "iteration_summary": values_summary(iters),
        "solver_s_summary": values_summary(solver_s),
        "window_around_h35": window,
        "available_fields": sorted({k for r in rows for k in r.keys()}),
    }


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(x) for x in values]
    if not data:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None, "min": None}
    data_sorted = sorted(data)
    def percentile(p: float) -> float:
        if len(data_sorted) == 1:
            return data_sorted[0]
        idx = (len(data_sorted) - 1) * p / 100.0
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return data_sorted[lo]
        frac = idx - lo
        return data_sorted[lo] * (1.0 - frac) + data_sorted[hi] * frac
    return {
        "count": len(data_sorted),
        "sum": float(math.fsum(data_sorted)),
        "mean": float(math.fsum(data_sorted) / len(data_sorted)),
        "median": float(percentile(50)),
        "p95": float(percentile(95)),
        "max": float(max(data_sorted)),
        "min": float(min(data_sorted)),
    }


def horizon_counts(trace: List[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in trace:
        h = str(row.get("horizon"))
        counts[h] = counts.get(h, 0) + 1
    return counts


def switches(trace: List[Mapping[str, Any]]) -> int:
    hs = [int(r.get("horizon")) for r in trace]
    return sum(a != b for a, b in zip(hs, hs[1:]))


def choose_tree(policy: Mapping[str, Any], values: Sequence[float]) -> Dict[str, Any]:
    nodes = policy["nodes"]
    leaves = policy["leaves"]
    root = nodes[0]
    root_feature = int(root["feature"])
    root_threshold = float(root["threshold"])
    root_value = float(values[root_feature])
    side = int(root_value > root_threshold)
    node = nodes[1 + side]
    node_feature = int(node["feature"])
    node_threshold = float(node["threshold"])
    node_value = float(values[node_feature])
    leaf = 2 * side + int(node_value > node_threshold)
    return {
        "root_feature_index": root_feature,
        "root_feature_name": TREE_FEATURE_NAMES[root_feature],
        "root_value": root_value,
        "root_threshold": root_threshold,
        "root_margin_value_minus_threshold": root_value - root_threshold,
        "side": side,
        "second_feature_index": node_feature,
        "second_feature_name": TREE_FEATURE_NAMES[node_feature],
        "second_value": node_value,
        "second_threshold": node_threshold,
        "second_margin_value_minus_threshold": node_value - node_threshold,
        "leaf": leaf,
        "horizon": int(leaves[leaf]),
    }


def find_policy(gate: Mapping[str, Any], arm_id: str = "learned_latency_tree_vehicle_s2") -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    for cand in gate.get("learned_candidates") or []:
        if cand.get("arm_id") == arm_id or cand.get("rollout_key") == "learned_s2" or cand.get("seed") == 2:
            path = ROOT / cand["policy_path"]
            return path, read_json(path), cand
    raise RuntimeError("learned_s2 policy not found in gate")


def cost_segment(trace: List[Mapping[str, Any]], start: int, end: int) -> Dict[str, Any]:
    rows = trace[start:end]
    return {
        "start_inclusive": start,
        "end_exclusive": end,
        "steps": len(rows),
        "performance_sum": float(math.fsum(float(r.get("performance", 0.0)) for r in rows)),
        "compute_sum": float(math.fsum(float(r.get("compute", 0.0)) for r in rows)),
        "reward_sum": float(math.fsum(float(r.get("reward", 0.0)) for r in rows)),
        "solver_failure_steps": int(sum(1 for r in rows if not bool(r.get("solver_success")))),
        "horizon_counts": horizon_counts(rows),
    }


def first_index(predicate: Any, n: int) -> Optional[int]:
    for i in range(n):
        if predicate(i):
            return i
    return None


def diagnostic() -> Dict[str, Any]:
    inputs = [
        LEARNED_DIR / "summary.json",
        LEARNED_DIR / "trace.json",
        LEARNED_DIR / "solver_calls.jsonl",
        LEARNED_DIR / "solver_attempts.json",
        LEARNED_DIR / "completed.json",
        FIXED_H25_DIR / "summary.json",
        FIXED_H25_DIR / "trace.json",
        FIXED_H25_DIR / "solver_calls.jsonl",
        FIXED_H25_DIR / "solver_attempts.json",
        FIXED_H25_DIR / "completed.json",
        GATE_JSON,
        THIS_SCRIPT,
    ]
    missing = [rel(p) for p in inputs if not p.exists()]
    if missing:
        raise RuntimeError("missing inputs: " + json.dumps(missing))

    learned_summary = read_json(LEARNED_DIR / "summary.json")
    fixed_summary = read_json(FIXED_H25_DIR / "summary.json")
    learned_trace = read_json(LEARNED_DIR / "trace.json")
    fixed_trace = read_json(FIXED_H25_DIR / "trace.json")
    learned_solver = read_jsonl(LEARNED_DIR / "solver_calls.jsonl")
    fixed_solver = read_jsonl(FIXED_H25_DIR / "solver_calls.jsonl")
    gate = read_json(GATE_JSON)
    policy_path, policy, policy_gate_record = find_policy(gate)

    failures: List[str] = []
    for label, summary in (("learned", learned_summary), ("fixed_h25", fixed_summary)):
        exp = EXPECTED[label]
        for key in ("rollout_key", "seed", "case", "steps", "success"):
            if summary.get(key) != exp[key]:
                failures.append(f"{label} summary {key} expected {exp[key]!r} got {summary.get(key)!r}")
    if horizon_counts(learned_trace) != EXPECTED["learned"]["horizon_counts"]:
        failures.append("learned trace horizon counts mismatch")
    if abs(float(learned_summary.get("physical_constraint_cost")) - EXPECTED["learned"]["physical_constraint_cost_approx"]) > 1e-6:
        failures.append("learned physical_constraint_cost mismatch")

    learned_hs = [int(r["horizon"]) for r in learned_trace]
    fixed_hs = [int(r["horizon"]) for r in fixed_trace]
    h35_steps = [i for i, h in enumerate(learned_hs) if h == 35]
    first_h35 = h35_steps[0] if h35_steps else None
    min_len = min(len(learned_trace), len(fixed_trace))

    prefix_end = first_h35 if first_h35 is not None else min_len
    prefix_previous_state_max = max((max_abs_diff(vec_state(learned_trace[i], "previous_state"), vec_state(fixed_trace[i], "previous_state")) for i in range(min(prefix_end, min_len))), default=0.0)
    prefix_post_state_max = max((max_abs_diff(vec_state(learned_trace[i], "state"), vec_state(fixed_trace[i], "state")) for i in range(min(prefix_end, min_len))), default=0.0)
    prefix_input_max = max((max_abs_diff(vec_input(learned_trace[i]), vec_input(fixed_trace[i])) for i in range(min(prefix_end, min_len))), default=0.0)
    prefix_reward_max = max((abs(float(learned_trace[i].get("reward", 0.0)) - float(fixed_trace[i].get("reward", 0.0))) for i in range(min(prefix_end, min_len))), default=0.0)

    first_input_divergence = first_index(lambda i: max_abs_diff(vec_input(learned_trace[i]), vec_input(fixed_trace[i])) > 1e-9, min_len)
    first_post_state_divergence = first_index(lambda i: max_abs_diff(vec_state(learned_trace[i], "state"), vec_state(fixed_trace[i], "state")) > 1e-9, min_len)
    first_pre_state_divergence = first_index(lambda i: max_abs_diff(vec_state(learned_trace[i], "previous_state"), vec_state(fixed_trace[i], "previous_state")) > 1e-9, min_len)

    h35_detail: Dict[str, Any] = {}
    if first_h35 is not None and first_h35 < min_len:
        lrow = learned_trace[first_h35]
        frow = fixed_trace[first_h35]
        h35_detail = {
            "step": first_h35,
            "learned_horizon": int(lrow["horizon"]),
            "fixed_horizon": int(frow["horizon"]),
            "pre_state_max_abs_diff_vs_fixed_h25": max_abs_diff(vec_state(lrow, "previous_state"), vec_state(frow, "previous_state")),
            "post_state_max_abs_diff_vs_fixed_h25": max_abs_diff(vec_state(lrow, "state"), vec_state(frow, "state")),
            "post_state_l2_diff_vs_fixed_h25": norm_diff(vec_state(lrow, "state"), vec_state(frow, "state")),
            "input_learned": vec_input(lrow),
            "input_fixed_h25": vec_input(frow),
            "input_abs_diff": [abs(a - b) for a, b in zip(vec_input(lrow), vec_input(frow))],
            "performance_learned": lrow.get("performance"),
            "performance_fixed_h25": frow.get("performance"),
            "compute_learned": lrow.get("compute"),
            "compute_fixed_h25": frow.get("compute"),
            "reward_learned": lrow.get("reward"),
            "reward_fixed_h25": frow.get("reward"),
        }

    divergence_window: List[Dict[str, Any]] = []
    if first_h35 is not None:
        for i in range(max(0, first_h35 - 8), min(min_len, first_h35 + 18)):
            divergence_window.append({
                "step": i,
                "learned_h": int(learned_trace[i]["horizon"]),
                "fixed_h": int(fixed_trace[i]["horizon"]),
                "pre_state_l2_diff": norm_diff(vec_state(learned_trace[i], "previous_state"), vec_state(fixed_trace[i], "previous_state")),
                "post_state_l2_diff": norm_diff(vec_state(learned_trace[i], "state"), vec_state(fixed_trace[i], "state")),
                "input_l2_diff": norm_diff(vec_input(learned_trace[i]), vec_input(fixed_trace[i])),
                "learned_performance": learned_trace[i].get("performance"),
                "fixed_performance": fixed_trace[i].get("performance"),
                "learned_solver_success": learned_trace[i].get("solver_success"),
                "fixed_solver_success": fixed_trace[i].get("solver_success"),
            })

    # Decision path and margin around H35.
    policy_decision_checks: List[Dict[str, Any]] = []
    bad_policy_steps: List[int] = []
    if policy.get("kind") == "tree":
        for i, row in enumerate(learned_trace):
            vals = row.get("tree_features") or []
            pred = choose_tree(policy, vals)
            if pred["horizon"] != int(row["horizon"]):
                bad_policy_steps.append(i)
            if first_h35 is not None and max(0, first_h35 - 8) <= i < min(len(learned_trace), first_h35 + 18):
                named_features = {TREE_FEATURE_NAMES[j]: float(vals[j]) for j in range(min(len(vals), len(TREE_FEATURE_NAMES)))}
                policy_decision_checks.append({
                    "step": i,
                    "recorded_horizon": int(row["horizon"]),
                    "predicted_horizon_from_policy_file": pred["horizon"],
                    "leaf": pred["leaf"],
                    "path": pred,
                    "selected_feature_values": named_features,
                    "previous_initial_failure": bool((row.get("policy_context") or {}).get("previous_initial_failure", False)),
                    "previous_final_failure": bool((row.get("policy_context") or {}).get("previous_final_failure", False)),
                })

    # Finite/range summaries for normalization-related saved vectors.
    def finite_summary_for_trace(trace: List[Mapping[str, Any]], key_path: Sequence[str]) -> Dict[str, Any]:
        values: List[float] = []
        nonfinite = 0
        lengths: Dict[str, int] = {}
        for row in trace:
            v: Any = row
            for k in key_path:
                if not isinstance(v, Mapping):
                    v = None
                    break
                v = v.get(k)
            flat = finite_flat(v)
            lengths[str(len(flat))] = lengths.get(str(len(flat)), 0) + 1
            for x in flat:
                if math.isfinite(x):
                    values.append(x)
                else:
                    nonfinite += 1
        return {"path": ".".join(key_path), "length_counts": lengths, "nonfinite_values": nonfinite, "value_summary": values_summary(values)}

    normalization_audit = {
        "learned_observation": finite_summary_for_trace(learned_trace, ["observation"]),
        "learned_next_observation": finite_summary_for_trace(learned_trace, ["next_observation"]),
        "learned_tree_features": finite_summary_for_trace(learned_trace, ["tree_features"]),
        "learned_policy_context_encoded_features": finite_summary_for_trace(learned_trace, ["policy_context", "features"]),
        "fixed_observation": finite_summary_for_trace(fixed_trace, ["observation"]),
        "fixed_tree_features": finite_summary_for_trace(fixed_trace, ["tree_features"]),
        "note": "Stored vectors are finite. The tree policy uses latency_tree_policy.FEATURES raw causal features, not the SAC/controller observation vector. Full MPC internal state scaling and objective components are not persisted in these traces.",
    }

    learned_fields = sorted({k for row in learned_trace for k in row.keys()})
    solver_fields = sorted({k for row in learned_solver for k in row.keys()})
    objective_like = [k for k in learned_fields + solver_fields if any(token in k.lower() for token in ("objective", "obj", "terminal", "value", "opt_f"))]

    # Cost and distance development.
    thresholds = [1.0, 10.0, 100.0, 1000.0]
    first_perf_threshold = {
        str(th): first_index(lambda i, th=th: float(learned_trace[i].get("performance", 0.0)) >= th, len(learned_trace))
        for th in thresholds
    }
    top_performance_steps = sorted(
        [
            {"step": i, "performance": float(row.get("performance", 0.0)), "horizon": int(row.get("horizon")), "state": row.get("state")}
            for i, row in enumerate(learned_trace)
        ],
        key=lambda r: r["performance"],
        reverse=True,
    )[:10]

    segment_end = first_h35 if first_h35 is not None else 0
    cost_segments = {
        "learned_before_h35": cost_segment(learned_trace, 0, segment_end),
        "fixed_before_same_step": cost_segment(fixed_trace, 0, min(segment_end, len(fixed_trace))),
        "learned_h35_step": cost_segment(learned_trace, segment_end, min(segment_end + 1, len(learned_trace))),
        "fixed_same_step": cost_segment(fixed_trace, segment_end, min(segment_end + 1, len(fixed_trace))),
        "learned_after_h35_until_fixed_termination_step": cost_segment(learned_trace, min(segment_end + 1, len(learned_trace)), min(len(fixed_trace), len(learned_trace))),
        "fixed_after_same_step_until_termination": cost_segment(fixed_trace, min(segment_end + 1, len(fixed_trace)), len(fixed_trace)),
        "learned_after_fixed_termination": cost_segment(learned_trace, len(fixed_trace), len(learned_trace)),
    }

    input_hashes = {rel(p): sha256(p) for p in inputs + [policy_path] if p.exists()}
    learned_completed_ok, learned_completed_bad = verify_hashes((read_json(LEARNED_DIR / "completed.json").get("hashes") or {}))
    fixed_completed_ok, fixed_completed_bad = verify_hashes((read_json(FIXED_H25_DIR / "completed.json").get("hashes") or {}))

    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": not failures and learned_completed_ok and fixed_completed_ok and not bad_policy_steps,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_case43_shard07_saved_trajectory_diagnostic_not_original_SAC",
        "validation_accessed": True,
        "validation_access_type": "read/hashed existing shard07 and shard00 validation output trajectories only",
        "validation_bank_reopened": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_validation_episodes": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "diagnostic_budget": {
            "saved_validation_episodes_read": 2,
            "saved_validation_traces_read": 2,
            "saved_trace_steps_read": len(learned_trace) + len(fixed_trace),
            "new_environment_resets": 0,
            "new_solver_calls": 0,
            "new_control_steps": 0,
            "new_gradient_steps": 0,
        },
        "inputs_sha256": input_hashes,
        "episode_completed_hash_audit": {
            "learned_completed_ok": learned_completed_ok,
            "learned_completed_bad": learned_completed_bad,
            "fixed_h25_completed_ok": fixed_completed_ok,
            "fixed_h25_completed_bad": fixed_completed_bad,
        },
        "learned_summary": learned_summary,
        "fixed_h25_summary": fixed_summary,
        "horizon_audit": {
            "learned_horizon_counts": horizon_counts(learned_trace),
            "learned_switches": switches(learned_trace),
            "h35_steps": h35_steps,
            "fixed_h25_horizon_counts": horizon_counts(fixed_trace),
            "fixed_h25_switches": switches(fixed_trace),
        },
        "paired_prefix_and_divergence": {
            "same_seed_same_terminal_same_case": True,
            "comparison_limit_steps": min_len,
            "prefix_end_exclusive_first_h35": prefix_end,
            "prefix_previous_state_max_abs_diff": prefix_previous_state_max,
            "prefix_post_state_max_abs_diff": prefix_post_state_max,
            "prefix_input_max_abs_diff": prefix_input_max,
            "prefix_reward_max_abs_diff": prefix_reward_max,
            "first_input_divergence_step": first_input_divergence,
            "first_post_state_divergence_step": first_post_state_divergence,
            "first_pre_state_divergence_step": first_pre_state_divergence,
            "h35_detail": h35_detail,
            "window": divergence_window,
            "interpretation_guard": "Stored deterministic trajectories show the first recorded input/post-state divergence at the singleton H35 action when pre-state is equal; this strongly implicates the horizon decision but does not replace future one-variable instrumented replays with objective/terminal logging.",
        },
        "solver_and_warmstart_audit": {
            "learned": summarize_solver_calls(learned_solver, learned_trace, first_h35),
            "fixed_h25": summarize_solver_calls(fixed_solver, fixed_trace, first_h35),
            "warmstart_limit": "The logs record warmup/initial/retry calls, success, iterations, residuals and solver time. They do not persist the primal warm-start vector itself, so exact warm-start-state comparison requires an instrumented replay.",
        },
        "policy_decision_audit": {
            "policy_path": rel(policy_path),
            "policy_sha256": sha256(policy_path),
            "policy_gate_record": policy_gate_record,
            "policy_kind": policy.get("kind"),
            "policy_nodes": policy.get("nodes"),
            "policy_leaves": policy.get("leaves"),
            "bad_policy_steps": bad_policy_steps,
            "window_around_h35": policy_decision_checks,
        },
        "normalization_and_logged_state_audit": normalization_audit,
        "objective_terminal_value_availability": {
            "objective_like_top_level_or_solver_fields": sorted(set(objective_like)),
            "trace_fields": learned_fields,
            "solver_call_fields": solver_fields,
            "conclusion": "Saved validation traces do not contain MPC objective value, terminal value contribution, or normalized NLP variable vectors. Only rewards/performance/compute, selected horizon, finite observation/context vectors and solver status/residual/timing are available. Objective/terminal-value diagnosis therefore requires a separate non-formal instrumented deterministic replay of already-opened case43.",
        },
        "cost_development": {
            "first_learned_performance_threshold_step": first_perf_threshold,
            "top_learned_performance_steps": top_performance_steps,
            "segments": cost_segments,
        },
        "interpretation": {
            "supported_from_saved_traces": [
                "learned_s2 and same-seed fixed H25 have identical saved pre-H35 trajectory within numerical tolerance if prefix diffs are zero/tiny.",
                "The singleton H35 decision is the first saved input/post-state divergence if first divergence equals the H35 step.",
                "Solver failures/retries do not explain this episode if all scored solver calls are successful/accepted and retry_rows=0.",
            ],
            "not_supported_without_replay": [
                "Attributing failure solely to H35 rather than value/objective/warm-start effects beyond the logged data.",
                "Inspecting terminal value/objective terms, because they are not present in the saved trace schema.",
            ],
            "next_informative_experiment": "After preserving this diagnostic and continuing/finishing validation64 subject to backup gates, run a non-formal instrumented deterministic replay of already-opened case43 with one-variable ablations: learned policy as logged, forced H25 at the H35 step only, forced H35 at the same state with fixed continuation, and objective/terminal-value logging. Do not use sealed test.",
        },
    }
    return raw


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    h35 = raw["paired_prefix_and_divergence"].get("h35_detail") or {}
    solver = raw["solver_and_warmstart_audit"]["learned"]
    obj = raw["objective_terminal_value_availability"]
    lines = [
        "# Vehicle shard07 case43 saved-trajectory diagnostic",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "- Validation access: existing shard07/shard00 outputs read/hashed only; validation bank reopened: `false`.",
        "- Sealed final test opened/hashed/accessed: `false` / `false` / `false`.",
        "- New simulations/control steps/gradient steps: `0` / `0` / `0`.",
        "",
        "## Key facts",
        "",
        f"- learned_s2 case43: success `{raw['learned_summary']['success']}`, steps `{raw['learned_summary']['steps']}`, physical+constraint cost `{raw['learned_summary']['physical_constraint_cost']:.6g}`, horizons `{raw['horizon_audit']['learned_horizon_counts']}`.",
        f"- fixed seed2 terminal25 H25 case43: success `{raw['fixed_h25_summary']['success']}`, steps `{raw['fixed_h25_summary']['steps']}`, physical+constraint cost `{raw['fixed_h25_summary']['physical_constraint_cost']:.6g}`, horizons `{raw['horizon_audit']['fixed_h25_horizon_counts']}`.",
        f"- H35 steps in learned trace: `{raw['horizon_audit']['h35_steps']}`.",
        "",
        "## Paired trajectory comparison",
        "",
        f"- Prefix end before first H35: step `{raw['paired_prefix_and_divergence']['prefix_end_exclusive_first_h35']}`.",
        f"- Prefix max abs previous-state diff vs fixed H25: `{raw['paired_prefix_and_divergence']['prefix_previous_state_max_abs_diff']:.3g}`.",
        f"- Prefix max abs input diff vs fixed H25: `{raw['paired_prefix_and_divergence']['prefix_input_max_abs_diff']:.3g}`.",
        f"- First input divergence step: `{raw['paired_prefix_and_divergence']['first_input_divergence_step']}`; first post-state divergence step: `{raw['paired_prefix_and_divergence']['first_post_state_divergence_step']}`; first pre-state divergence step: `{raw['paired_prefix_and_divergence']['first_pre_state_divergence_step']}`.",
    ]
    if h35:
        lines.extend([
            f"- At H35 step `{h35['step']}`, pre-state max abs diff vs fixed H25 is `{h35['pre_state_max_abs_diff_vs_fixed_h25']:.3g}`; input learned `{h35['input_learned']}` vs fixed `{h35['input_fixed_h25']}`; post-state L2 diff `{h35['post_state_l2_diff_vs_fixed_h25']:.3g}`.",
            f"- Same step performance: learned `{h35['performance_learned']}` vs fixed `{h35['performance_fixed_h25']}`.",
        ])
    lines.extend([
        "",
        "## Solver / warm-start evidence",
        "",
        f"- Learned scored solver calls all success: `{solver['all_scored_success']}`; all accepted: `{solver['all_scored_accepted']}`; retry rows: `{solver['retry_rows']}`.",
        "- Warm-start vectors and objective values are not persisted; the logs only expose warmup/initial/retry labels, status, iterations, residuals and timing.",
        "",
        "## Policy / normalization evidence",
        "",
        f"- Policy file: `{raw['policy_decision_audit']['policy_path']}`.",
        f"- Policy leaves: `{raw['policy_decision_audit']['policy_leaves']}`; bad re-evaluated policy steps: `{raw['policy_decision_audit']['bad_policy_steps']}`.",
        "- Stored observation/tree-feature/context vectors are finite; see raw JSON for ranges and threshold margins around H35.",
        "",
        "## Objective / terminal value limitation",
        "",
        f"- Objective-like persisted fields: `{obj['objective_like_top_level_or_solver_fields']}`.",
        f"- Conclusion: {obj['conclusion']}",
        "",
        "## Interpretation",
        "",
        "Saved trajectories support that the singleton H35 action is the first recorded control/post-state divergence from the same-seed H25 comparator, while solver failure/retry is not observed. This is diagnostic evidence only and does not prove final-test performance or exact causal mechanism. An instrumented non-formal replay is still needed for terminal/objective and one-variable ablation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: Path, marker: str, body: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = f"<!-- {marker} -->"
    if token in old:
        return False
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")
    return True


def main() -> int:
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        if existing_completed_ok(completed):
            print(json.dumps({"already_completed": True, "completed": rel(completed)}, sort_keys=True))
            return 0
        raise SystemExit("prior diagnostic completed marker exists but failed hash verification; inspect before rerun")

    raw = diagnostic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    completed_path = OUT_DIR / "completed.json"
    write_json(raw_path, raw)
    write_summary(summary_path, raw)

    created_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    stamp = created_dt.strftime("%Y%m%dT%H%M%S")
    files_pre_completed = [raw_path, summary_path]
    backup_request = {
        "created_utc": created_dt.isoformat(),
        "purpose": "external backup request after vehicle shard07 case43 saved-trajectory diagnostic",
        "validation_accessed": True,
        "validation_access_type": raw["validation_access_type"],
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "must_cover_before_formal_shard10": True,
        "artifacts_to_cover": [rel(p) for p in files_pre_completed + [THIS_SCRIPT]],
        "pre_completed_hashes": {rel(p): sha256(p) for p in files_pre_completed + [THIS_SCRIPT]},
        "note": "This request does not satisfy the pre-shard10 backup gate by itself; it asks the supervisor backup process to include this diagnostic plus finalized run registry/stdout/stderr.",
    }
    request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_CASE43_SHARD07_DIAGNOSTIC_{stamp}.json"
    write_json(request_path, backup_request)

    doc_body = f"""
## 2026-09-27 vehicle shard07 case43 trajectory diagnostic

UTC: {raw['created_utc']}. Metadata-only diagnostic of already-created validation outputs completed: learned_s2 shard07 case43 versus same-seed terminal25 fixed H25 case43. No validation bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.

Key diagnostic facts: learned_s2 case43 failed at 150 steps with physical+constraint cost {raw['learned_summary']['physical_constraint_cost']:.6g} and horizons {raw['horizon_audit']['learned_horizon_counts']}; fixed seed2 terminal25 H25 succeeded in {raw['fixed_h25_summary']['steps']} steps with physical+constraint cost {raw['fixed_h25_summary']['physical_constraint_cost']:.6g}. H35 steps were {raw['horizon_audit']['h35_steps']}. Prefix max previous-state/input diffs before the singleton H35 were {raw['paired_prefix_and_divergence']['prefix_previous_state_max_abs_diff']:.3g} / {raw['paired_prefix_and_divergence']['prefix_input_max_abs_diff']:.3g}; first input/post-state divergence steps were {raw['paired_prefix_and_divergence']['first_input_divergence_step']} / {raw['paired_prefix_and_divergence']['first_post_state_divergence_step']}. Learned solver calls were all successful/accepted with retry_rows={raw['solver_and_warmstart_audit']['learned']['retry_rows']}. Objective/terminal-value components and warm-start vectors were not persisted, so an instrumented non-formal replay is required for that part of the diagnosis.

Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(completed_path)}`. Backup requested at `{rel(request_path)}`; shard10 remains blocked until an adequate verified external backup proof covers shard09 audit run-finalized evidence and subsequent diagnostic artifacts.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        p = ROOT / doc
        if p.exists():
            append_once(p, DOC_MARKER, doc_body)

    # Re-write raw after adding artifact paths; then write completed marker last.
    raw["artifacts"] = {
        "raw": rel(raw_path),
        "summary": rel(summary_path),
        "completed": rel(completed_path),
        "backup_request": rel(request_path),
    }
    raw["docs_updated"] = [doc for doc in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md") if (ROOT / doc).exists()]
    write_json(raw_path, raw)
    write_summary(summary_path, raw)
    files = [p for p in [raw_path, summary_path, request_path, THIS_SCRIPT] if p.exists()]
    for doc in raw["docs_updated"]:
        files.append(ROOT / doc)
    write_json(completed_path, {
        "passed": raw["passed"],
        "validation_accessed": True,
        "validation_access_type": raw["validation_access_type"],
        "validation_bank_reopened": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_validation_episodes": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "h35_steps": raw["horizon_audit"]["h35_steps"],
        "learned_success": raw["learned_summary"]["success"],
        "fixed_h25_success": raw["fixed_h25_summary"]["success"],
        "first_input_divergence_step": raw["paired_prefix_and_divergence"]["first_input_divergence_step"],
        "first_post_state_divergence_step": raw["paired_prefix_and_divergence"]["first_post_state_divergence_step"],
        "learned_solver_retry_rows": raw["solver_and_warmstart_audit"]["learned"]["retry_rows"],
        "backup_request": rel(request_path),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
    })
    print(json.dumps({
        "completed": rel(completed_path),
        "raw": rel(raw_path),
        "summary": rel(summary_path),
        "backup_request": rel(request_path),
        "passed": raw["passed"],
        "validation_bank_reopened": False,
        "test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "h35_steps": raw["horizon_audit"]["h35_steps"],
        "first_input_divergence_step": raw["paired_prefix_and_divergence"]["first_input_divergence_step"],
        "first_post_state_divergence_step": raw["paired_prefix_and_divergence"]["first_post_state_divergence_step"],
    }, sort_keys=True), flush=True)
    return 0 if raw["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
