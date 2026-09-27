#!/usr/bin/env python3
"""Vehicle learned horizon-policy collapse diagnostic v3 (full validation64).

This is a metadata/trace diagnostic, not a rollout.  It repairs the failed v2
policy locator by using the frozen latency-tree policy files verified by the
vehicle validation gate summary hashes.  It reads only already-created
validation64 learned-controller episode traces/summaries from shards 00-11.
It does not reopen the validation bank, does not open or hash the sealed final
 test bank, and performs no simulation/control/training.

Diagnostic questions:
  1. Are learned_s0 and learned_s1 H25-only because their extracted policies are
     structurally constant H25?
  2. Do stored rollout traces match the frozen policy files when re-evaluating
     the stored tree_features?
  3. How often does learned_s2 actually choose its non-H25 leaf over all 64
     validation cases, and are non-H25/failure cases localized?
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
GATE_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/summary.md"
VALIDATION_ROOT = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_learned_policy_collapse_diagnostic_v3_full_validation64.py"
DOC_MARKER = "vehicle-learned-policy-collapse-diagnostic-v3-full-validation64-20260927"
COMPLETED_SHARDS = list(range(12))
LEARNED_KEYS = ["learned_s0", "learned_s1", "learned_s2"]

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

# Frozen latency-tree vehicle policies as recorded in the 20260926 validation
# gate summary.  These paths are existing registered artifacts; this diagnostic
# only hashes/reads them.
FALLBACK_POLICIES = {
    "learned_s0": {
        "gate_arm": "learned_latency_tree_vehicle_s0",
        "seed": 0,
        "path": "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train/vehicle_s0/policy.json",
        "expected_sha256_from_gate_summary": "10df92c75a7d1d664c9a5b6941671102861f64a66f7d1c49e1f37793dd479d82",
        "expected_gate_structural_class": "behaviorally_fixed_by_structure_H25",
    },
    "learned_s1": {
        "gate_arm": "learned_latency_tree_vehicle_s1",
        "seed": 1,
        "path": "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train/vehicle_s1/policy.json",
        "expected_sha256_from_gate_summary": "37831c5f6ad49c67dcd2996d8abaf8f7dedd7af7b014f0627ec8b8252cc360fa",
        "expected_gate_structural_class": "fixed_H25",
    },
    "learned_s2": {
        "gate_arm": "learned_latency_tree_vehicle_s2",
        "seed": 2,
        "path": "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train/vehicle_s2/policy.json",
        "expected_sha256_from_gate_summary": "56af550d17f2348999edca0bafeedb854b9ad80abce9af6c284892ab2159f6f0",
        "expected_gate_structural_class": "structurally_switching_tree",
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


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not xs:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * p / 100.0
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        frac = idx - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    return {
        "count": len(xs),
        "sum": float(math.fsum(xs)),
        "mean": float(math.fsum(xs) / len(xs)),
        "median": float(pct(50)),
        "p05": float(pct(5)),
        "p95": float(pct(95)),
        "min": float(xs[0]),
        "max": float(xs[-1]),
    }


def inc(mapping: Dict[str, int], key: Any, amount: int = 1) -> None:
    mapping[str(key)] = mapping.get(str(key), 0) + int(amount)


def merge_counts(dst: Dict[str, int], src: Mapping[Any, Any]) -> None:
    for k, v in src.items():
        inc(dst, k, int(v))


def normalize_policy_record(policy: Mapping[str, Any]) -> Dict[str, Any]:
    # Keep the original fields but normalize constant-horizon spelling for
    # downstream evaluation.  Do not mutate the read artifact.
    out = dict(policy)
    if out.get("kind") == "constant" and "horizon" not in out and "h" in out:
        out["horizon"] = out["h"]
    return out


def horizon_options(policy: Mapping[str, Any]) -> List[int]:
    if policy.get("kind") == "constant":
        if "horizon" in policy:
            return [int(policy["horizon"])]
        if "h" in policy:
            return [int(policy["h"])]
    leaves = policy.get("leaves") or []
    return [int(x) for x in leaves]


def choose_policy(policy: Mapping[str, Any], values: Sequence[float]) -> Dict[str, Any]:
    if policy.get("kind") == "constant":
        hs = horizon_options(policy)
        if not hs:
            raise ValueError("constant policy without horizon/h field")
        return {"kind": "constant", "leaf": None, "side": None, "horizon": int(hs[0])}
    if policy.get("kind") != "tree":
        raise ValueError("unsupported policy kind: %r" % (policy.get("kind"),))
    nodes = policy.get("nodes") or []
    leaves = policy.get("leaves") or []
    if len(nodes) < 3 or len(leaves) < 4:
        raise ValueError("expected depth-2 tree with 3 nodes and 4 leaves")
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
        "kind": "tree",
        "root_feature_index": root_feature,
        "root_feature_name": TREE_FEATURE_NAMES[root_feature] if 0 <= root_feature < len(TREE_FEATURE_NAMES) else str(root_feature),
        "root_value": root_value,
        "root_threshold": root_threshold,
        "root_margin": root_value - root_threshold,
        "side": side,
        "second_feature_index": node_feature,
        "second_feature_name": TREE_FEATURE_NAMES[node_feature] if 0 <= node_feature < len(TREE_FEATURE_NAMES) else str(node_feature),
        "second_value": node_value,
        "second_threshold": node_threshold,
        "second_margin": node_value - node_threshold,
        "leaf": leaf,
        "horizon": int(leaves[leaf]),
    }


def policy_summary(policy: Mapping[str, Any]) -> Dict[str, Any]:
    hs = horizon_options(policy)
    out: Dict[str, Any] = {
        "kind": policy.get("kind"),
        "horizon_options": hs,
        "unique_leaf_horizons": sorted(set(hs)),
        "structurally_constant_horizon": bool(hs) and len(set(hs)) == 1,
    }
    if policy.get("kind") == "constant":
        out["constant_horizon"] = hs[0] if hs else None
        return out
    leaves = [int(x) for x in (policy.get("leaves") or [])]
    out["nodes_count"] = len(policy.get("nodes") or [])
    out["leaves"] = leaves
    counts: Dict[str, int] = {}
    for h in leaves:
        inc(counts, h)
    out["leaf_horizon_counts"] = counts
    if policy.get("kind") == "tree" and len(policy.get("nodes") or []) >= 3:
        nodes = policy["nodes"]
        root = nodes[0]
        root_f = int(root["feature"])
        out["root"] = {
            "feature_index": root_f,
            "feature_name": TREE_FEATURE_NAMES[root_f] if 0 <= root_f < len(TREE_FEATURE_NAMES) else str(root_f),
            "threshold": float(root["threshold"]),
        }
        seconds = []
        for side, node in enumerate(nodes[1:3]):
            f = int(node["feature"])
            seconds.append({
                "side": side,
                "feature_index": f,
                "feature_name": TREE_FEATURE_NAMES[f] if 0 <= f < len(TREE_FEATURE_NAMES) else str(f),
                "threshold": float(node["threshold"]),
                "leaf_if_leq": 2 * side,
                "horizon_if_leq": leaves[2 * side] if 2 * side < len(leaves) else None,
                "leaf_if_gt": 2 * side + 1,
                "horizon_if_gt": leaves[2 * side + 1] if 2 * side + 1 < len(leaves) else None,
            })
        out["second_level"] = seconds
    return out


def collect_strings(value: Any, max_items: int = 200000) -> List[str]:
    out: List[str] = []
    stack = [value]
    while stack and len(out) < max_items:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return out


def locate_policies(gate: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Return learned policy metadata using robust frozen-artifact fallback.

    v2 assumed gate['learned_candidates'] was a list with rollout_key fields.
    The frozen gate summary, however, records the policy hashes but the JSON
    shape is not that list.  This v3 first searches for such records if present
    and then falls back to the exact registered latency-tree vehicle policy paths
    whose hashes are recorded in the summary.  Hash mismatches are failures.
    """
    failures: List[str] = []
    out: Dict[str, Dict[str, Any]] = {}

    # Opportunistic recursive search for records containing policy paths and
    # learned arm names.  This is not required for success because the fallback
    # paths are already gate-summary frozen and hash checked.
    candidate_records: List[Mapping[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, Mapping):
            vals = [v for v in x.values() if isinstance(v, str)]
            joined = "\n".join(vals)
            if "policy" in joined and ("learned_s" in joined or "learned_latency_tree_vehicle" in joined or "vehicle_s" in joined):
                candidate_records.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(gate)

    for key, meta in FALLBACK_POLICIES.items():
        path = ROOT / meta["path"]
        if not path.exists():
            failures.append("frozen policy path missing for %s: %s" % (key, rel(path)))
            continue
        policy = normalize_policy_record(read_json(path))
        actual_sha = sha256(path)
        expected_sha = str(meta["expected_sha256_from_gate_summary"])
        if actual_sha != expected_sha:
            failures.append("policy sha mismatch for %s: expected %s got %s at %s" % (key, expected_sha, actual_sha, rel(path)))
        out[key] = {
            "locator": "frozen_gate_summary_hash_fallback_v3",
            "gate_record": meta,
            "path": path,
            "sha256": actual_sha,
            "policy": policy,
            "summary": policy_summary(policy),
        }

    # Record whether the expected hashes/arms are visible anywhere in the gate
    # JSON text data, without making pass/fail depend on a brittle shape.
    all_gate_strings = collect_strings(gate)
    joined_gate_strings = "\n".join(all_gate_strings)
    for key, info in out.items():
        info["gate_json_visibility"] = {
            "expected_sha_string_present": info["gate_record"]["expected_sha256_from_gate_summary"] in joined_gate_strings,
            "gate_arm_string_present": info["gate_record"]["gate_arm"] in joined_gate_strings,
            "fallback_path_string_present": info["gate_record"]["path"] in joined_gate_strings,
            "candidate_records_seen": len(candidate_records),
        }
    return out, failures


def aggregate_one_trace(trace: List[Mapping[str, Any]], policy: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    horizon_counts: Dict[str, int] = {}
    predicted_horizon_counts: Dict[str, int] = {}
    leaf_counts: Dict[str, int] = {}
    side_counts: Dict[str, int] = {}
    root_margins: List[float] = []
    second_margins: List[float] = []
    mismatch_steps: List[Dict[str, Any]] = []
    non25_steps: List[Dict[str, Any]] = []
    near_root = {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0}
    near_second = {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0}
    feature_values: Dict[str, List[float]] = {name: [] for name in TREE_FEATURE_NAMES}
    bad_feature_steps: List[int] = []

    for i, row in enumerate(trace):
        actual_h = row.get("horizon", row.get("H"))
        if actual_h is not None:
            inc(horizon_counts, int(actual_h))
        vals = row.get("tree_features") or (row.get("decision") or {}).get("features") or []
        vals_ok = len(vals) == len(TREE_FEATURE_NAMES) and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals)
        if vals_ok:
            for j, name in enumerate(TREE_FEATURE_NAMES):
                feature_values[name].append(float(vals[j]))
        elif policy is not None and policy.get("kind") == "tree":
            bad_feature_steps.append(i)
            continue
        if policy is not None:
            pred = choose_policy(policy, vals if vals_ok else [])
            inc(predicted_horizon_counts, pred["horizon"])
            if pred.get("leaf") is not None:
                inc(leaf_counts, pred["leaf"])
            if pred.get("side") is not None:
                inc(side_counts, pred["side"])
            if "root_margin" in pred:
                root_margins.append(float(pred["root_margin"]))
                second_margins.append(float(pred["second_margin"]))
                for threshold in near_root:
                    if abs(float(pred["root_margin"])) <= float(threshold):
                        near_root[threshold] += 1
                for threshold in near_second:
                    if abs(float(pred["second_margin"])) <= float(threshold):
                        near_second[threshold] += 1
            if actual_h is not None and int(pred["horizon"]) != int(actual_h):
                mismatch_steps.append({
                    "step": i,
                    "actual_horizon": int(actual_h),
                    "predicted_horizon": int(pred["horizon"]),
                    "predicted_leaf": pred.get("leaf"),
                })
            if actual_h is not None and int(actual_h) != 25:
                non25_steps.append({
                    "step": i,
                    "actual_horizon": int(actual_h),
                    "predicted_horizon": int(pred["horizon"]),
                    "predicted_leaf": pred.get("leaf"),
                    "root_margin": pred.get("root_margin"),
                    "second_margin": pred.get("second_margin"),
                })

    return {
        "steps": len(trace),
        "horizon_counts": horizon_counts,
        "predicted_horizon_counts": predicted_horizon_counts,
        "leaf_counts": leaf_counts,
        "side_counts": side_counts,
        "policy_trace_mismatch_count": len(mismatch_steps),
        "policy_trace_mismatch_first_steps": mismatch_steps[:20],
        "non25_first_steps": non25_steps[:20],
        "non25_step_count": len(non25_steps),
        "root_margin_summary": values_summary(root_margins),
        "second_margin_summary": values_summary(second_margins),
        "near_root_margin_counts": near_root,
        "near_second_margin_counts": near_second,
        "tree_feature_summaries": {name: values_summary(vals) for name, vals in feature_values.items() if vals},
        "nonfinite_or_bad_tree_feature_steps": bad_feature_steps[:20],
        "nonfinite_or_bad_tree_feature_step_count": len(bad_feature_steps),
    }


def empty_aggregate(key: str, policy_info: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rollout_key": key,
        "policy_path": rel(policy_info["path"]) if policy_info else None,
        "policy_sha256": policy_info.get("sha256") if policy_info else None,
        "policy_locator": policy_info.get("locator") if policy_info else None,
        "policy_gate_record": policy_info.get("gate_record") if policy_info else None,
        "policy_gate_json_visibility": policy_info.get("gate_json_visibility") if policy_info else None,
        "policy_summary": policy_info.get("summary") if policy_info else None,
        "episodes": 0,
        "steps": 0,
        "success_count": 0,
        "episode_failure_count": 0,
        "termination_counts": {},
        "case_indices": [],
        "horizon_counts": {},
        "predicted_horizon_counts": {},
        "leaf_counts": {},
        "side_counts": {},
        "policy_trace_mismatch_count": 0,
        "policy_trace_mismatch_examples": [],
        "non25_episode_examples": [],
        "failure_episode_examples": [],
        "total_cost_values": [],
        "performance_cost_values": [],
        "physical_constraint_cost_values": [],
        "decision_mean_s_values": [],
        "decision_sum_s_values": [],
        "selection_sum_s_values": [],
        "solver_sum_s_values": [],
        "root_margins_all": [],
        "second_margins_all": [],
        "near_root_margin_counts": {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0},
        "near_second_margin_counts": {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0},
        "feature_values_all": {name: [] for name in TREE_FEATURE_NAMES},
        "input_hashes_bounded": {},
    }


def add_episode(agg: Dict[str, Any], summary: Mapping[str, Any], trace: List[Mapping[str, Any]], trace_path: Path, summary_path: Path, policy: Optional[Mapping[str, Any]]) -> None:
    agg["episodes"] += 1
    agg["steps"] += int(summary.get("steps", len(trace)))
    agg["success_count"] += int(bool(summary.get("success")))
    agg["episode_failure_count"] += int(bool(summary.get("episode_failure")))
    inc(agg["termination_counts"], summary.get("termination"))
    if summary.get("case") is not None:
        agg["case_indices"].append(int(summary.get("case")))
    for field in ["total_cost", "performance_cost", "physical_constraint_cost"]:
        if summary.get(field) is not None:
            agg[field + "_values"].append(float(summary.get(field)))
    for source_field, out_field in [
        ("decision_timing_s", "decision_mean_s_values"),
        ("decision_timing_s", "decision_sum_s_values"),
        ("selection_timing_s", "selection_sum_s_values"),
        ("solver_attempt_timing_s", "solver_sum_s_values"),
    ]:
        stats = summary.get(source_field) or {}
        if out_field.endswith("mean_s_values") and stats.get("mean") is not None:
            agg[out_field].append(float(stats["mean"]))
        if out_field.endswith("sum_s_values") and stats.get("sum") is not None:
            agg[out_field].append(float(stats["sum"]))

    trace_diag = aggregate_one_trace(trace, policy)
    merge_counts(agg["horizon_counts"], trace_diag["horizon_counts"])
    merge_counts(agg["predicted_horizon_counts"], trace_diag["predicted_horizon_counts"])
    merge_counts(agg["leaf_counts"], trace_diag["leaf_counts"])
    merge_counts(agg["side_counts"], trace_diag["side_counts"])
    agg["policy_trace_mismatch_count"] += trace_diag["policy_trace_mismatch_count"]
    if trace_diag["policy_trace_mismatch_count"] and len(agg["policy_trace_mismatch_examples"]) < 10:
        agg["policy_trace_mismatch_examples"].append({
            "path": rel(trace_path.parent),
            "case": summary.get("case"),
            "first_steps": trace_diag["policy_trace_mismatch_first_steps"],
        })
    for threshold, count in trace_diag["near_root_margin_counts"].items():
        agg["near_root_margin_counts"][threshold] += count
    for threshold, count in trace_diag["near_second_margin_counts"].items():
        agg["near_second_margin_counts"][threshold] += count
    if policy is not None and policy.get("kind") == "tree":
        for row in trace:
            vals = row.get("tree_features") or (row.get("decision") or {}).get("features") or []
            if len(vals) == len(TREE_FEATURE_NAMES) and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals):
                pred = choose_policy(policy, vals)
                agg["root_margins_all"].append(float(pred["root_margin"]))
                agg["second_margins_all"].append(float(pred["second_margin"]))
                for j, name in enumerate(TREE_FEATURE_NAMES):
                    agg["feature_values_all"][name].append(float(vals[j]))
    if any(str(h) != "25" for h in trace_diag["horizon_counts"]):
        agg["non25_episode_examples"].append({
            "path": rel(trace_path.parent),
            "case": summary.get("case"),
            "horizon_counts": trace_diag["horizon_counts"],
            "non25_first_steps": trace_diag["non25_first_steps"],
            "success": summary.get("success"),
            "steps": summary.get("steps"),
            "total_cost": summary.get("total_cost"),
            "physical_constraint_cost": summary.get("physical_constraint_cost"),
        })
    if not bool(summary.get("success")) or bool(summary.get("episode_failure")):
        agg["failure_episode_examples"].append({
            "path": rel(trace_path.parent),
            "case": summary.get("case"),
            "horizon_counts": trace_diag["horizon_counts"],
            "success": summary.get("success"),
            "episode_failure": summary.get("episode_failure"),
            "steps": summary.get("steps"),
            "total_cost": summary.get("total_cost"),
            "physical_constraint_cost": summary.get("physical_constraint_cost"),
        })
    # Bounded hash proof: keep two hashes per episode for learned traces only.
    agg["input_hashes_bounded"][rel(summary_path)] = sha256(summary_path)
    agg["input_hashes_bounded"][rel(trace_path)] = sha256(trace_path)


def finalize_aggregate(agg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(agg)
    out["case_indices"] = sorted(set(out["case_indices"]))
    out["total_cost_summary"] = values_summary(out.pop("total_cost_values"))
    out["performance_cost_summary"] = values_summary(out.pop("performance_cost_values"))
    out["physical_constraint_cost_summary"] = values_summary(out.pop("physical_constraint_cost_values"))
    out["episode_decision_mean_s_summary"] = values_summary(out.pop("decision_mean_s_values"))
    out["episode_decision_sum_s_summary"] = values_summary(out.pop("decision_sum_s_values"))
    out["episode_selection_sum_s_summary"] = values_summary(out.pop("selection_sum_s_values"))
    out["episode_solver_sum_s_summary"] = values_summary(out.pop("solver_sum_s_values"))
    out["root_margin_summary"] = values_summary(out.pop("root_margins_all"))
    out["second_margin_summary"] = values_summary(out.pop("second_margins_all"))
    out["tree_feature_summaries"] = {name: values_summary(vals) for name, vals in out.pop("feature_values_all").items() if vals}
    out["non25_episode_examples"] = out["non25_episode_examples"][:30]
    out["failure_episode_examples"] = out["failure_episode_examples"][:30]
    out["input_hash_count"] = len(out["input_hashes_bounded"])
    return out


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle learned-policy collapse diagnostic v3 (full validation64)",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "This diagnostic reads existing validation64 learned-controller traces from shards 00-11. It performs no new rollout, no control step, no training step, no validation-bank reopen, and no sealed-test open/hash.",
        "",
        "## Main findings",
        "",
    ]
    for key in LEARNED_KEYS:
        diag = raw["diagnosis_by_key"][key]
        concl = raw["conclusions"][key]
        ps = diag.get("policy_summary") or {}
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(f"- Classification: `{concl['classification']}`.")
        lines.append(f"- Frozen policy: `{diag.get('policy_path')}` SHA256 `{diag.get('policy_sha256')}`; locator `{diag.get('policy_locator')}`.")
        lines.append(f"- Gate expected class/hash: `{(diag.get('policy_gate_record') or {}).get('expected_gate_structural_class')}` / `{(diag.get('policy_gate_record') or {}).get('expected_sha256_from_gate_summary')}`.")
        lines.append(f"- Policy kind/options: `{ps.get('kind')}`; options `{ps.get('horizon_options')}`; unique horizons `{ps.get('unique_leaf_horizons')}`; structurally constant: `{ps.get('structurally_constant_horizon')}`.")
        if ps.get("root"):
            lines.append(f"- Root split: `{ps['root']['feature_name']}` > `{ps['root']['threshold']}`; second-level: `{ps.get('second_level')}`.")
        lines.append(f"- Existing validation learned traces read: episodes `{diag['episodes']}`, cases `{len(diag['case_indices'])}`, steps `{diag['steps']}`, successes `{diag['success_count']}`, episode failures `{diag['episode_failure_count']}`.")
        lines.append(f"- Actual horizon counts: `{diag['horizon_counts']}`; policy-predicted counts from stored tree features: `{diag['predicted_horizon_counts']}`; leaf counts: `{diag['leaf_counts']}`; trace/policy mismatches: `{diag['policy_trace_mismatch_count']}`.")
        lines.append(f"- Cost summary: total `{diag['total_cost_summary']}`; physical/control `{diag['physical_constraint_cost_summary']}`.")
        lines.append(f"- Timing summary: episode mean decision seconds `{diag['episode_decision_mean_s_summary']}`; episode decision sum seconds `{diag['episode_decision_sum_s_summary']}`; solver sum seconds `{diag['episode_solver_sum_s_summary']}`.")
        lines.append(f"- Root margin summary: `{diag['root_margin_summary']}`; second-level margin summary: `{diag['second_margin_summary']}`.")
        if diag.get("non25_episode_examples"):
            lines.append(f"- Non-H25 episode examples (bounded): `{diag['non25_episode_examples']}`.")
        if diag.get("failure_episode_examples"):
            lines.append(f"- Failure examples (bounded): `{diag['failure_episode_examples']}`.")
        lines.append("")
    lines.extend([
        "## Interpretation guard",
        "",
        "This diagnostic establishes policy-structure/trace consistency only. It does not estimate adaptive-horizon superiority and does not replace the required full paired validation aggregation against strong fixed-H grids. Timing values are read from existing validation traces and remain development/validation evidence, not final-test evidence.",
        "",
        "If s0/s1 have all horizons fixed at H25 and zero trace/policy mismatch, their H25-only validation behavior is explained at the extracted-policy level; later method revisions should diagnose training/selection objective rather than rollout-time timing noise. learned_s2's non-H25 behavior is not by itself evidence of benefit; paired aggregate analysis and targeted deterministic replays are needed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists() and (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        print(json.dumps({"already_had_completed": True, "passed": done.get("passed"), "completed": rel(OUT_DIR / "completed.json")}, indent=2, sort_keys=True))
        return 0 if done.get("passed") is True else 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    stamp = created.replace("-", "").replace(":", "").replace("+00:00", "")
    failures: List[str] = []

    gate = read_json(GATE_JSON)
    policies, policy_failures = locate_policies(gate)
    failures.extend(policy_failures)
    missing_policies = [k for k in LEARNED_KEYS if k not in policies]
    if missing_policies:
        failures.append("missing learned policy records after v3 fallback: %s" % missing_policies)

    aggregates: Dict[str, Dict[str, Any]] = {k: empty_aggregate(k, policies.get(k)) for k in LEARNED_KEYS}
    shard_episode_counts: Dict[str, Dict[str, int]] = {}
    scanned_episode_dirs = 0
    learned_episode_dirs = 0
    shard_completed_hashes: Dict[str, str] = {}
    shard_raw_hashes: Dict[str, str] = {}

    for shard in COMPLETED_SHARDS:
        shard_dir = VALIDATION_ROOT / ("shard%02d" % shard)
        completed_path = shard_dir / "completed.json"
        raw_path = shard_dir / "raw.json"
        if not completed_path.exists():
            failures.append("completed shard missing: shard%02d" % shard)
            continue
        shard_completed_hashes["shard%02d" % shard] = sha256(completed_path)
        if raw_path.exists():
            shard_raw_hashes["shard%02d" % shard] = sha256(raw_path)
        episodes_dir = shard_dir / "episodes"
        if not episodes_dir.exists():
            failures.append("episodes directory missing: %s" % rel(episodes_dir))
            continue
        shard_counts = {k: 0 for k in LEARNED_KEYS}
        for summary_path in sorted(episodes_dir.glob("*/summary.json")):
            scanned_episode_dirs += 1
            summary = read_json(summary_path)
            key = summary.get("rollout_key")
            if key not in LEARNED_KEYS:
                continue
            learned_episode_dirs += 1
            shard_counts[key] += 1
            trace_path = summary_path.parent / "trace.json"
            if not trace_path.exists():
                failures.append("missing trace for %s" % rel(summary_path.parent))
                continue
            trace = read_json(trace_path)
            policy = (policies.get(key) or {}).get("policy")
            add_episode(aggregates[key], summary, trace, trace_path, summary_path, policy)
        shard_episode_counts["shard%02d" % shard] = shard_counts

    diagnosis_by_key = {k: finalize_aggregate(aggregates[k]) for k in LEARNED_KEYS}
    conclusions: Dict[str, Any] = {}
    for key, diag in diagnosis_by_key.items():
        ps = diag.get("policy_summary") or {}
        structurally_constant = bool(ps.get("structurally_constant_horizon"))
        unique_horizons = ps.get("unique_leaf_horizons") or []
        mismatch_count = int(diag.get("policy_trace_mismatch_count", 0))
        actual_non25_steps = sum(int(v) for h, v in (diag.get("horizon_counts") or {}).items() if str(h) != "25")
        predicted_non25_steps = sum(int(v) for h, v in (diag.get("predicted_horizon_counts") or {}).items() if str(h) != "25")
        if mismatch_count:
            classification = "policy_trace_mismatch_requires_plumbing_diagnosis"
            next_step = "Repair policy execution/trace feature plumbing before scientific interpretation."
        elif structurally_constant and unique_horizons == [25] and actual_non25_steps == 0:
            classification = "extracted_policy_structurally_constant_H25"
            next_step = "Inspect training/selection objective and candidate extraction logs; rollout timing noise is not needed to explain H25-only behavior."
        elif (not structurally_constant) and predicted_non25_steps == 0 and actual_non25_steps == 0:
            classification = "nonH25_leaves_exist_but_validation_states_never_reach_them"
            next_step = "Compare feature distributions with thresholds and training states."
        elif actual_non25_steps > 0 and predicted_non25_steps == actual_non25_steps:
            classification = "adaptive_horizon_used_and_trace_matches_policy"
            next_step = "Use full paired aggregate analysis and targeted deterministic replays to test cost/time benefit and failure causality."
        elif actual_non25_steps > 0:
            classification = "adaptive_horizon_used_but_prediction_count_differs"
            next_step = "Inspect per-step mismatches and decision/feature serialization."
        else:
            classification = "undetermined_from_existing_traces"
            next_step = "Inspect raw traces and policy files manually."
        conclusions[key] = {
            "classification": classification,
            "structurally_constant_policy": structurally_constant,
            "unique_leaf_horizons": unique_horizons,
            "actual_non25_steps": actual_non25_steps,
            "predicted_non25_steps": predicted_non25_steps,
            "policy_trace_mismatch_count": mismatch_count,
            "falsifiable_next_step": next_step,
        }

    for key, diag in diagnosis_by_key.items():
        if diag["episodes"] != 64:
            failures.append("%s expected 64 learned validation episodes, found %s" % (key, diag["episodes"]))
        if len(diag["case_indices"]) != 64:
            failures.append("%s expected 64 distinct validation cases, found %s" % (key, len(diag["case_indices"])))
        if int(diag.get("policy_trace_mismatch_count", 0)) != 0:
            failures.append("%s policy/trace mismatch count %s" % (key, diag.get("policy_trace_mismatch_count")))

    raw = {
        "created_utc": created,
        "script": rel(THIS_SCRIPT),
        "script_sha256": sha256(THIS_SCRIPT),
        "formal_scientific_evidence_created": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation_accessed": True,
        "validation_bank_reopened": False,
        "validation64_existing_outputs_read": True,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "gate_json": {"path": rel(GATE_JSON), "sha256": sha256(GATE_JSON)},
        "gate_summary": {"path": rel(GATE_SUMMARY), "sha256": sha256(GATE_SUMMARY)},
        "completed_shards_read": COMPLETED_SHARDS,
        "shard_completed_hashes": shard_completed_hashes,
        "shard_raw_hashes": shard_raw_hashes,
        "scanned_episode_dirs": scanned_episode_dirs,
        "learned_episode_dirs": learned_episode_dirs,
        "shard_episode_counts": shard_episode_counts,
        "diagnosis_by_key": diagnosis_by_key,
        "conclusions": conclusions,
        "failures": failures,
    }

    raw_path = OUT_DIR / "raw.json"
    summary_path = OUT_DIR / "summary.md"
    completed_path = OUT_DIR / "completed.json"
    write_json(raw_path, raw)
    write_summary(summary_path, raw)

    passed = not failures
    backup_request = {
        "created_utc": created,
        "request": "external_backup_after_vehicle_learned_policy_collapse_diagnostic_v3_full_validation64",
        "reason": "Preserve repaired metadata/trace diagnostic artifacts before subsequent scientific aggregation or formal/final-test gates.",
        "sealed_test_access": "none; sealed final test remains closed/not hashed",
        "validation_access": "existing validation64 learned traces/summaries only; validation bank not reopened",
        "formal_scientific_evidence_created": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "artifacts_to_backup": [
            rel(THIS_SCRIPT),
            rel(raw_path),
            rel(summary_path),
            rel(completed_path),
        ],
    }
    backup_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_LEARNED_POLICY_COLLAPSE_DIAGNOSTIC_V3_FULL_VALIDATION64_%s.json" % stamp)
    write_json(backup_path, backup_request)

    completed = {
        "created_utc": created,
        "passed": passed,
        "raw": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "backup_request": rel(backup_path),
        "backup_request_sha256": sha256(backup_path),
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "validation_accessed": True,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "conclusions": conclusions,
        "failures": failures,
    }
    write_json(completed_path, completed)

    log_body = """
### Vehicle learned-policy collapse diagnostic v3 full validation64 ({created})

- Ran metadata/trace diagnostic only: no simulations, no control steps, no gradient steps, no validation-bank reopen, no sealed-test access/hash.
- Repaired failed v2 policy lookup by hashing exact frozen latency-tree vehicle policy files recorded in the 20260926 validation gate summary.
- Result: `{passed}`; artifacts: `{raw}`, `{summary}`, `{completed}`.
- learned_s0: `{s0}`.
- learned_s1: `{s1}`.
- learned_s2: `{s2}`.
- Next action: run full paired validation64 aggregate/model-selection analysis against the strong fixed-H grid before any final-test gate or method revision.
""".format(
        created=created,
        passed="passed" if passed else "failed",
        raw=rel(raw_path),
        summary=rel(summary_path),
        completed=rel(completed_path),
        s0=conclusions["learned_s0"],
        s1=conclusions["learned_s1"],
        s2=conclusions["learned_s2"],
    )
    append_once(ROOT / "RESEARCH_LOG.md", DOC_MARKER + "-research-log", log_body)
    append_once(ROOT / "STATUS.md", DOC_MARKER + "-status", log_body)
    append_once(ROOT / "DECISIONS.md", DOC_MARKER + "-decisions", """
### Decision: treat learned_s0/s1 as extracted-policy collapse, not runtime noise ({created})

Before evidence: validation64 traces showed learned_s0 and learned_s1 used H25 on every stored step; gate summary listed s0/s1 as fixed-by-structure/fixed-H25, but v2 diagnostic failed policy lookup.

Change in v3: no controller or data change. Only diagnostic locator repaired to hash exact frozen policy files from the gate summary and re-evaluate stored tree features from all 12 validation shards.

Decision rule: if policy hashes match the gate summary and trace/policy mismatches are zero, H25-only behavior is attributed to extracted policy structure. This does not imply adaptive superiority and does not authorize final test.

Outcome: `{passed}` with conclusions `{conclusions}`. Next action is full paired validation64 aggregation against strong fixed-H baselines.
""".format(created=created, passed="passed" if passed else "failed", conclusions=conclusions))

    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
