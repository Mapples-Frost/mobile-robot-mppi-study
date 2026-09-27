#!/usr/bin/env python3
"""Diagnose learned vehicle horizon-policy collapse using existing validation outputs.

This is a metadata/trace diagnostic, not a new rollout. It reads already-created
vehicle validation64 shard outputs for learned_s0/s1/s2 and the frozen gate's
policy-file references. It does not reopen the validation bank, does not open or
hash the sealed final-test bank, and performs no simulation/control/training.

Main question: do learned_s0 and learned_s1 collapse to H25 because the extracted
policy files are structurally constant, because non-H25 leaves are never reached
on validation states, or because trace-time policy execution mismatches the
policy files?
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
GATE_JSON = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json"
VALIDATION_ROOT = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_learned_policy_collapse_diagnostic.py"
DOC_MARKER = "vehicle-learned-policy-collapse-diagnostic-20260927"
COMPLETED_SHARDS = list(range(11))
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
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * p / 100.0
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        frac = idx - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    return {
        "count": len(xs),
        "mean": float(math.fsum(xs) / len(xs)),
        "median": float(pct(50)),
        "p05": float(pct(5)),
        "p95": float(pct(95)),
        "min": float(xs[0]),
        "max": float(xs[-1]),
    }


def inc(mapping: Dict[str, int], key: Any, amount: int = 1) -> None:
    k = str(key)
    mapping[k] = mapping.get(k, 0) + int(amount)


def merge_counts(dst: Dict[str, int], src: Mapping[Any, Any]) -> None:
    for k, v in src.items():
        inc(dst, k, int(v))


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


def policy_tree_summary(policy: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "kind": policy.get("kind"),
        "nodes_count": len(policy.get("nodes") or []),
        "leaves": policy.get("leaves"),
    }
    leaves = [int(x) for x in (policy.get("leaves") or [])]
    out["unique_leaf_horizons"] = sorted(set(leaves))
    counts: Dict[str, int] = {}
    for h in leaves:
        inc(counts, h)
    out["leaf_horizon_counts"] = counts
    out["structurally_constant_horizon"] = len(set(leaves)) == 1
    if policy.get("kind") == "tree" and len(policy.get("nodes") or []) >= 3:
        nodes = policy["nodes"]
        root = nodes[0]
        out["root"] = {
            "feature_index": int(root["feature"]),
            "feature_name": TREE_FEATURE_NAMES[int(root["feature"])],
            "threshold": float(root["threshold"]),
        }
        seconds = []
        for side, node in enumerate(nodes[1:3]):
            seconds.append({
                "side": side,
                "feature_index": int(node["feature"]),
                "feature_name": TREE_FEATURE_NAMES[int(node["feature"])],
                "threshold": float(node["threshold"]),
                "leaf_if_leq": 2 * side,
                "horizon_if_leq": leaves[2 * side] if 2 * side < len(leaves) else None,
                "leaf_if_gt": 2 * side + 1,
                "horizon_if_gt": leaves[2 * side + 1] if 2 * side + 1 < len(leaves) else None,
            })
        out["second_level"] = seconds
    return out


def locate_policies(gate: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cand in gate.get("learned_candidates") or []:
        key = cand.get("rollout_key")
        if key in LEARNED_KEYS:
            path = ROOT / cand["policy_path"]
            policy = read_json(path)
            out[key] = {
                "gate_record": cand,
                "path": path,
                "sha256": sha256(path),
                "policy": policy,
                "summary": policy_tree_summary(policy),
            }
    return out


def aggregate_one_trace(trace: List[Mapping[str, Any]], policy: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    horizon_counts: Dict[str, int] = {}
    predicted_horizon_counts: Dict[str, int] = {}
    leaf_counts: Dict[str, int] = {}
    side_counts: Dict[str, int] = {}
    root_margins: List[float] = []
    second_margins: List[float] = []
    mismatch_steps: List[int] = []
    near_root = {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0}
    near_second = {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0}
    feature_values: Dict[str, List[float]] = {name: [] for name in TREE_FEATURE_NAMES}
    nonfinite_tree_feature_steps: List[int] = []

    for i, row in enumerate(trace):
        actual_h = int(row.get("horizon"))
        inc(horizon_counts, actual_h)
        vals = row.get("tree_features") or []
        if len(vals) != len(TREE_FEATURE_NAMES) or any((not isinstance(v, (int, float)) or not math.isfinite(float(v))) for v in vals):
            nonfinite_tree_feature_steps.append(i)
            continue
        for j, name in enumerate(TREE_FEATURE_NAMES):
            feature_values[name].append(float(vals[j]))
        if policy is not None and policy.get("kind") == "tree":
            pred = choose_tree(policy, vals)
            inc(predicted_horizon_counts, pred["horizon"])
            inc(leaf_counts, pred["leaf"])
            inc(side_counts, pred["side"])
            root_margins.append(float(pred["root_margin"]))
            second_margins.append(float(pred["second_margin"]))
            for threshold in near_root:
                if abs(float(pred["root_margin"])) <= float(threshold):
                    near_root[threshold] += 1
            for threshold in near_second:
                if abs(float(pred["second_margin"])) <= float(threshold):
                    near_second[threshold] += 1
            if int(pred["horizon"]) != actual_h:
                mismatch_steps.append(i)

    feature_summaries = {name: values_summary(vals) for name, vals in feature_values.items() if vals}
    return {
        "steps": len(trace),
        "horizon_counts": horizon_counts,
        "predicted_horizon_counts": predicted_horizon_counts,
        "leaf_counts": leaf_counts,
        "side_counts": side_counts,
        "policy_trace_mismatch_count": len(mismatch_steps),
        "policy_trace_mismatch_first_steps": mismatch_steps[:20],
        "root_margin_summary": values_summary(root_margins),
        "second_margin_summary": values_summary(second_margins),
        "near_root_margin_counts": near_root,
        "near_second_margin_counts": near_second,
        "tree_feature_summaries": feature_summaries,
        "nonfinite_or_bad_tree_feature_steps": nonfinite_tree_feature_steps[:20],
        "nonfinite_or_bad_tree_feature_step_count": len(nonfinite_tree_feature_steps),
    }


def empty_key_aggregate(key: str, policy_info: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rollout_key": key,
        "policy_path": rel(policy_info["path"]) if policy_info else None,
        "policy_sha256": policy_info.get("sha256") if policy_info else None,
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
        "root_margins_all": [],
        "second_margins_all": [],
        "near_root_margin_counts": {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0},
        "near_second_margin_counts": {"1e-06": 0, "0.001": 0, "0.01": 0, "0.1": 0},
        "feature_values_all": {name: [] for name in TREE_FEATURE_NAMES},
        "input_trace_hashes": {},
    }


def add_episode_to_aggregate(agg: Dict[str, Any], summary: Mapping[str, Any], trace: List[Mapping[str, Any]], trace_path: Path, summary_path: Path, policy: Optional[Mapping[str, Any]]) -> None:
    agg["episodes"] += 1
    agg["steps"] += int(summary.get("steps", len(trace)))
    agg["success_count"] += int(bool(summary.get("success")))
    agg["episode_failure_count"] += int(bool(summary.get("episode_failure")))
    inc(agg["termination_counts"], summary.get("termination"))
    agg["case_indices"].append(int(summary.get("case")))
    for field in ["total_cost", "performance_cost", "physical_constraint_cost"]:
        if summary.get(field) is not None:
            agg[field + "_values"].append(float(summary.get(field)))
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
    # Retain margins/features for global summaries; these are small for learned traces.
    if policy is not None and policy.get("kind") == "tree":
        for row in trace:
            vals = row.get("tree_features") or []
            if len(vals) == len(TREE_FEATURE_NAMES) and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals):
                pred = choose_tree(policy, vals)
                agg["root_margins_all"].append(float(pred["root_margin"]))
                agg["second_margins_all"].append(float(pred["second_margin"]))
                for j, name in enumerate(TREE_FEATURE_NAMES):
                    agg["feature_values_all"][name].append(float(vals[j]))
    if any(int(h) != 25 for h in trace_diag["horizon_counts"]):
        agg["non25_episode_examples"].append({
            "path": rel(trace_path.parent),
            "case": summary.get("case"),
            "horizon_counts": trace_diag["horizon_counts"],
            "success": summary.get("success"),
            "steps": summary.get("steps"),
            "total_cost": summary.get("total_cost"),
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
    agg["input_trace_hashes"][rel(summary_path)] = sha256(summary_path)
    agg["input_trace_hashes"][rel(trace_path)] = sha256(trace_path)


def finalize_aggregate(agg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(agg)
    out["case_indices"] = sorted(set(out["case_indices"]))
    out["total_cost_summary"] = values_summary(out.pop("total_cost_values"))
    out["performance_cost_summary"] = values_summary(out.pop("performance_cost_values"))
    out["physical_constraint_cost_summary"] = values_summary(out.pop("physical_constraint_cost_values"))
    out["root_margin_summary"] = values_summary(out.pop("root_margins_all"))
    out["second_margin_summary"] = values_summary(out.pop("second_margins_all"))
    out["tree_feature_summaries"] = {name: values_summary(vals) for name, vals in out.pop("feature_values_all").items() if vals}
    # Keep examples bounded in raw while preserving full counts/hashes.
    out["non25_episode_examples"] = out["non25_episode_examples"][:20]
    out["failure_episode_examples"] = out["failure_episode_examples"][:20]
    out["input_trace_hash_count"] = len(out["input_trace_hashes"])
    return out


def append_once(path: Path, marker: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + body.strip() + "\n", encoding="utf-8")


def write_summary(path: Path, raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle learned-policy collapse diagnostic",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "This diagnostic reads existing validation64 learned-controller traces from shards 00-10 only. It performs no new rollout, no control step, no training step, no validation-bank reopen, and no sealed-test open/hash.",
        "",
        "## Main findings",
        "",
    ]
    for key in LEARNED_KEYS:
        diag = raw["diagnosis_by_key"][key]
        ps = diag.get("policy_summary") or {}
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(f"- Policy file: `{diag.get('policy_path')}` SHA256 `{diag.get('policy_sha256')}`.")
        lines.append(f"- Policy leaf horizons: `{ps.get('leaves')}`; unique leaf horizons `{ps.get('unique_leaf_horizons')}`; structurally constant horizon: `{ps.get('structurally_constant_horizon')}`.")
        if ps.get("root"):
            lines.append(f"- Root split: `{ps['root']['feature_name']}` > `{ps['root']['threshold']}`.")
        lines.append(f"- Existing validation traces read: episodes `{diag['episodes']}`, steps `{diag['steps']}`, successes `{diag['success_count']}`, episode failures `{diag['episode_failure_count']}`.")
        lines.append(f"- Actual horizon counts: `{diag['horizon_counts']}`; policy-predicted counts from stored tree features: `{diag['predicted_horizon_counts']}`; leaf counts: `{diag['leaf_counts']}`; trace/policy mismatches: `{diag['policy_trace_mismatch_count']}`.")
        lines.append(f"- Root margin summary: `{diag['root_margin_summary']}`; second-level margin summary: `{diag['second_margin_summary']}`.")
        if diag.get("non25_episode_examples"):
            lines.append(f"- Non-H25 episode examples (bounded): `{diag['non25_episode_examples']}`.")
        if diag.get("failure_episode_examples"):
            lines.append(f"- Failure examples (bounded): `{diag['failure_episode_examples']}`.")
        lines.append("")
    lines.extend([
        "## Interpretation guard",
        "",
        "If s0/s1 have all four tree leaves equal to H25 and zero trace/policy mismatch, their validation-time H25-only behavior is explained at the extracted-policy level, not by solver timing noise during rollout. This does not by itself identify why training/selection produced constant trees; that requires training-log/objective diagnostics. If a policy has non-H25 leaves but validation leaf counts never reach them, the next diagnostic should focus on feature thresholds/distribution shift. If trace/policy mismatches occur, policy-execution plumbing must be repaired before further scientific use.",
        "",
        "No final-test evidence was created. These are development/validation diagnostics for the IMPROVED latency-tree method only, not ORIGINAL SAC reproduction evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists() and (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json")}, sort_keys=True))
            return 0
        raise SystemExit("prior diagnostic output exists but did not pass; inspect before rerun")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    stamp = created.replace("-", "").replace(":", "").replace("+00:00", "")
    failures: List[str] = []

    gate = read_json(GATE_JSON)
    policies = locate_policies(gate)
    missing_policies = [k for k in LEARNED_KEYS if k not in policies]
    if missing_policies:
        failures.append("missing learned policy records in gate: %s" % missing_policies)

    aggregates: Dict[str, Dict[str, Any]] = {k: empty_key_aggregate(k, policies.get(k)) for k in LEARNED_KEYS}
    shard_episode_counts: Dict[str, Dict[str, int]] = {}
    scanned_episode_dirs = 0
    learned_episode_dirs = 0

    for shard in COMPLETED_SHARDS:
        shard_dir = VALIDATION_ROOT / ("shard%02d" % shard)
        if not (shard_dir / "completed.json").exists():
            failures.append("completed shard missing: shard%02d" % shard)
            continue
        shard_counts = {k: 0 for k in LEARNED_KEYS}
        episodes_dir = shard_dir / "episodes"
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
            add_episode_to_aggregate(aggregates[key], summary, trace, trace_path, summary_path, policy)
        shard_episode_counts["shard%02d" % shard] = shard_counts

    diagnosis_by_key = {k: finalize_aggregate(aggregates[k]) for k in LEARNED_KEYS}

    conclusions: Dict[str, Any] = {}
    for key, diag in diagnosis_by_key.items():
        ps = diag.get("policy_summary") or {}
        structurally_constant = bool(ps.get("structurally_constant_horizon"))
        unique_leaves = ps.get("unique_leaf_horizons") or []
        mismatch_count = int(diag.get("policy_trace_mismatch_count", 0))
        actual_non25_steps = sum(v for h, v in (diag.get("horizon_counts") or {}).items() if str(h) != "25")
        predicted_non25_steps = sum(v for h, v in (diag.get("predicted_horizon_counts") or {}).items() if str(h) != "25")
        if mismatch_count:
            classification = "policy_trace_mismatch_requires_plumbing_diagnosis"
        elif structurally_constant and unique_leaves == [25] and actual_non25_steps == 0:
            classification = "extracted_policy_structurally_constant_H25"
        elif (not structurally_constant) and predicted_non25_steps == 0 and actual_non25_steps == 0:
            classification = "nonH25_leaves_exist_but_validation_states_never_reach_them"
        elif actual_non25_steps > 0:
            classification = "adaptive_horizon_used_on_existing_validation_traces"
        else:
            classification = "undetermined_from_existing_traces"
        conclusions[key] = {
            "classification": classification,
            "structurally_constant_policy": structurally_constant,
            "unique_leaf_horizons": unique_leaves,
            "actual_non25_steps": actual_non25_steps,
            "predicted_non25_steps": predicted_non25_steps,
            "policy_trace_mismatch_count": mismatch_count,
            "falsifiable_next_step": (
                "Inspect training/selection objective and extraction logs for why all leaves were set to H25; runtime timing noise is not needed to explain H25-only validation behavior."
                if classification == "extracted_policy_structurally_constant_H25" else
                "Compare feature distributions with thresholds and training states; non-H25 opportunity may be outside visited validation distribution."
                if classification == "nonH25_leaves_exist_but_validation_states_never_reach_them" else
                "Use failure/benefit case replays to test whether non-H25 decisions improve the cost/time tradeoff or induce divergence."
                if classification == "adaptive_horizon_used_on_existing_validation_traces" else
                "Repair/diagnose policy execution mismatch before scientific interpretation."
                if classification == "policy_trace_mismatch_requires_plumbing_diagnosis" else
                "Collect a more instrumented diagnostic."
            ),
        }

    raw: Dict[str, Any] = {
        "created_utc": created,
        "passed": not failures,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_learned_policy_collapse_diagnostic_existing_validation_outputs_not_original_SAC",
        "validation_accessed": True,
        "validation_access_type": "read existing validation64 shard00-shard10 learned output traces/summaries only",
        "validation_bank_reopened": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "completed_shards_read": COMPLETED_SHARDS,
        "scanned_episode_dirs": scanned_episode_dirs,
        "learned_episode_dirs_read": learned_episode_dirs,
        "shard_learned_episode_counts": shard_episode_counts,
        "input_hashes": {
            rel(THIS_SCRIPT): sha256(THIS_SCRIPT),
            rel(GATE_JSON): sha256(GATE_JSON),
            **{rel(info["path"]): info["sha256"] for info in policies.values()},
        },
        "diagnosis_by_key": diagnosis_by_key,
        "conclusions": conclusions,
        "interpretation_limits": [
            "This diagnostic uses shards 00-10 only because shard11 is not yet complete; it is not full validation model selection.",
            "It reads existing validation outputs and therefore is validation/development diagnostic evidence, not sealed final-test evidence.",
            "It can diagnose structural policy collapse or trace/policy mismatches, but not by itself why training selected those leaves; training-log/objective diagnostics are required next for s0/s1 if structurally constant.",
            "No ORIGINAL SAC reproduction claim follows; this is for the IMPROVED latency-tree method.",
        ],
    }

    raw_path = OUT_DIR / "raw.json"
    write_json(raw_path, raw)
    summary_path = OUT_DIR / "summary.md"
    write_summary(summary_path, raw)

    backup_request = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_LEARNED_POLICY_COLLAPSE_DIAGNOSTIC_%s.json" % stamp)
    backup_payload = {
        "created_utc": created,
        "status": "external_backup_requested_after_vehicle_learned_policy_collapse_diagnostic",
        "backup_verified": False,
        "required_before_shard11": True,
        "reason": "This diagnostic was created after the post-shard10 backup gate recheck; before shard11 formal validation, external backup must cover these new diagnostic artifacts and the finalized diagnostic run logs in addition to shard10 formal/audit evidence.",
        "validation_accessed": True,
        "validation_access_type": "existing validation64 outputs only; validation bank not reopened",
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "artifacts_to_backup": [rel(raw_path), rel(summary_path), rel(OUT_DIR / "completed.json"), rel(THIS_SCRIPT), "this diagnostic run registry/stdout/stderr/cloudwatch/cpu_samples after run_experiment finalizes"],
    }
    write_json(backup_request, backup_payload)

    completed = {
        "created_utc": created,
        "passed": not failures,
        "validation_accessed": True,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
        "raw": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "backup_request": rel(backup_request),
        "backup_request_sha256": sha256(backup_request),
        "conclusions": conclusions,
    }
    completed_path = OUT_DIR / "completed.json"
    write_json(completed_path, completed)
    completed["completed"] = rel(completed_path)
    completed["completed_sha256"] = sha256(completed_path)
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256(completed_path)

    doc_body = f"""
## 2026-09-27 vehicle learned-policy collapse diagnostic

UTC: {created}. Existing-output diagnostic for learned_s0/s1/s2 across validation64 shards 00-10 completed with validation_accessed=true only because saved validation output traces/summaries were read; validation bank reopened=false; sealed final test accessed/opened/hashed=false; simulations/control_steps/gradient_steps=0/0/0. Learned trace episodes read: {learned_episode_dirs}; scanned episode directories: {scanned_episode_dirs}. This is not full validation model selection because shard11 is still incomplete and backup-gated.

Key classifications: {json.dumps(conclusions, sort_keys=True)}

Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(completed_path)}`. Backup request: `{rel(backup_request)}`. Any proof before shard11 must now cover these diagnostic artifacts and the finalized diagnostic run logs as well as the prior shard10 audit/gate artifacts. Method remains IMPROVED latency-tree, not ORIGINAL SAC; no final-test or reproduction-success claim is supported.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / name, DOC_MARKER, doc_body)

    print(json.dumps({
        "completed": rel(completed_path),
        "completed_sha256": completed["completed_sha256"],
        "raw": rel(raw_path),
        "raw_sha256": completed["raw_sha256"],
        "summary": rel(summary_path),
        "summary_sha256": completed["summary_sha256"],
        "backup_request": rel(backup_request),
        "backup_request_sha256": completed["backup_request_sha256"],
        "conclusions": conclusions,
        "validation_accessed": True,
        "validation_bank_reopened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "sealed_test_bank_hashed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_gradient_steps": 0,
    }, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
