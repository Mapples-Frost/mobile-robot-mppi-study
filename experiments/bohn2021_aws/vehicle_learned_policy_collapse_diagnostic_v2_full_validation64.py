#!/usr/bin/env python3
"""Full-validation64 learned vehicle horizon-policy collapse diagnostic (v2).

This is a metadata/trace diagnostic, not a new rollout. It reads the already
completed/audited vehicle validation64 shard outputs for learned_s0/s1/s2 and
the frozen validation gate's policy-file references. It does not reopen the
validation bank, does not open or hash the sealed final-test bank, and performs
no simulation/control/training.

Diagnostic question: did learned_s0 and learned_s1 use H25 only because their
extracted policy files are structurally constant H25, because non-H25 leaves are
present but unreachable on validation states, or because rollout traces mismatch
frozen policy files?  v1 was prepared before shard11 completed; this v2 extends
that diagnostic to all 12 validation64 shards without changing controllers,
checkpoints, selection rules, acceptance criteria, or any rollout output.
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
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
THIS_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_learned_policy_collapse_diagnostic_v2_full_validation64.py"
DOC_MARKER = "vehicle-learned-policy-collapse-diagnostic-v2-full-validation64-20260927"
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
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
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
    mapping[str(key)] = mapping.get(str(key), 0) + int(amount)


def merge_counts(dst: Dict[str, int], src: Mapping[Any, Any]) -> None:
    for k, v in src.items():
        inc(dst, k, int(v))


def horizon_options(policy: Mapping[str, Any]) -> List[int]:
    if policy.get("kind") == "constant":
        if "h" in policy:
            return [int(policy["h"])]
        if "horizon" in policy:
            return [int(policy["horizon"])]
    leaves = policy.get("leaves") or []
    return [int(x) for x in leaves]


def choose_policy(policy: Mapping[str, Any], values: Sequence[float]) -> Dict[str, Any]:
    if policy.get("kind") == "constant":
        hs = horizon_options(policy)
        h = int(hs[0]) if hs else int(policy.get("horizon", 25))
        return {"kind": "constant", "leaf": None, "side": None, "horizon": h}
    if policy.get("kind") != "tree":
        raise ValueError("unsupported policy kind: %r" % (policy.get("kind"),))
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
                "summary": policy_summary(policy),
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
    bad_feature_steps: List[int] = []

    for i, row in enumerate(trace):
        actual_h = row.get("horizon", row.get("H"))
        if actual_h is not None:
            inc(horizon_counts, int(actual_h))
        vals = row.get("tree_features") or []
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
                mismatch_steps.append(i)

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
        "tree_feature_summaries": {name: values_summary(vals) for name, vals in feature_values.items() if vals},
        "nonfinite_or_bad_tree_feature_steps": bad_feature_steps[:20],
        "nonfinite_or_bad_tree_feature_step_count": len(bad_feature_steps),
    }


def empty_aggregate(key: str, policy_info: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
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
            vals = row.get("tree_features") or []
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
    out["non25_episode_examples"] = out["non25_episode_examples"][:25]
    out["failure_episode_examples"] = out["failure_episode_examples"][:25]
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
        "# Vehicle learned-policy collapse diagnostic v2 (full validation64)",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "This diagnostic reads existing validation64 learned-controller traces from all shards 00-11. It performs no new rollout, no control step, no training step, no validation-bank reopen, and no sealed-test open/hash.",
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
        lines.append(f"- Policy file: `{diag.get('policy_path')}` SHA256 `{diag.get('policy_sha256')}`.")
        lines.append(f"- Policy horizons/options: `{ps.get('horizon_options')}`; unique horizons `{ps.get('unique_leaf_horizons')}`; structurally constant horizon: `{ps.get('structurally_constant_horizon')}`.")
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
        "If s0/s1 have all tree leaves equal to H25 and zero trace/policy mismatch, their validation-time H25-only behavior is explained at the extracted-policy level, not by rollout timing noise. This does not by itself identify why training/selection produced constant trees; training-log/objective diagnostics are required next. If a policy has non-H25 leaves but validation leaf counts never reach them, the next diagnostic should focus on feature thresholds/distribution shift. If trace/policy mismatches occur, policy-execution plumbing must be repaired before further scientific use.",
        "",
        "No final-test evidence was created. These are development/validation diagnostics for the IMPROVED latency-tree method only, not ORIGINAL SAC reproduction evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists() and (OUT_DIR / "completed.json").exists():
        done = read_json(OUT_DIR / "completed.json")
        if done.get("passed") is True:
            print(json.dumps({"already_completed": True, "completed": rel(OUT_DIR / "completed.json")}, indent=2, sort_keys=True))
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

    aggregates: Dict[str, Dict[str, Any]] = {k: empty_aggregate(k, policies.get(k)) for k in LEARNED_KEYS}
    shard_episode_counts: Dict[str, Dict[str, int]] = {}
    scanned_episode_dirs = 0
    learned_episode_dirs = 0
    shard_completed_hashes: Dict[str, str] = {}

    for shard in COMPLETED_SHARDS:
        shard_dir = VALIDATION_ROOT / ("shard%02d" % shard)
        completed_path = shard_dir / "completed.json"
        if not completed_path.exists():
            failures.append("completed shard missing: shard%02d" % shard)
            continue
        shard_completed_hashes["shard%02d" % shard] = sha256(completed_path)
        shard_counts = {k: 0 for k in LEARNED_KEYS}
        episodes_dir = shard_dir / "episodes"
        if not episodes_dir.exists():
            failures.append("episodes directory missing: %s" % rel(episodes_dir))
            continue
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
            next_step = "Repair/diagnose policy execution mismatch before scientific interpretation."
        elif structurally_constant and unique_horizons == [25] and actual_non25_steps == 0:
            classification = "extracted_policy_structurally_constant_H25"
            next_step = "Inspect training/selection objective and extraction logs for why all leaves were set to H25; runtime timing noise is not needed to explain H25-only validation behavior."
        elif (not structurally_constant) and predicted_non25_steps == 0 and actual_non25_steps == 0:
            classification = "nonH25_leaves_exist_but_validation_states_never_reach_them"
            next_step = "Compare feature distributions with thresholds and training states; non-H25 opportunity may be outside visited validation distribution."
        elif actual_non25_steps > 0:
            classification = "adaptive_horizon_used_on_existing_validation_traces"
            next_step = "Use paired aggregate validation analysis and targeted replays to decide whether non-H25 decisions improve cost/time tradeoff or induce failures."
        else:
            classification = "undetermined_from_existing_traces"
            next_step = "Collect a more instrumented diagnostic."
        conclusions[key] = {
            "classification": classification,
            "structurally_constant_policy": structurally_constant,
            "unique_leaf_horizons": unique_horizons,
            "actual_non25_steps": actual_non25_steps,
            "predicted_non25_steps": predicted_non25_steps,
            "policy_trace_mismatch_count": mismatch_count,
            "falsifiable_next_step": next_step,
        }

    raw: Dict[str, Any] = {
        "created_utc": created,
        "passed": not failures,
        "failures": failures,
        "method": "IMPROVED_latency_tree_vehicle_learned_policy_collapse_diagnostic_v2_full_validation64_existing_outputs_not_original_SAC",
        "validation_accessed": True,
        "validation_access_type": "read existing validation64 shard00-shard11 learned output traces/summaries only",
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
        "shard_completed_hashes": shard_completed_hashes,
        "input_hashes": {rel(THIS_SCRIPT): sha256(THIS_SCRIPT), rel(GATE_JSON): sha256(GATE_JSON), **{rel(info["path"]): info["sha256"] for info in policies.values()}},
        "diagnosis_by_key": diagnosis_by_key,
        "conclusions": conclusions,
        "interpretation_limits": [
            "This diagnostic reads existing validation outputs and therefore is validation/development diagnostic evidence, not sealed final-test evidence.",
            "It can diagnose structural policy collapse, unreachable non-H25 leaves, or trace/policy mismatches, but not by itself why training selected those leaves; training-log/objective diagnostics are required next for s0/s1 if structurally constant.",
            "Full paired validation aggregation/model selection remains a separate analysis step; this diagnostic is not a final-test gate.",
            "No ORIGINAL SAC reproduction claim follows; this is for the IMPROVED latency-tree method.",
        ],
    }

    raw_path = OUT_DIR / "raw.json"
    write_json(raw_path, raw)
    summary_path = OUT_DIR / "summary.md"
    write_summary(summary_path, raw)
    backup_request = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_LEARNED_POLICY_COLLAPSE_DIAGNOSTIC_V2_FULL_VALIDATION64_%s.json" % stamp)
    backup_payload = {
        "created_utc": created,
        "status": "external_backup_requested_after_vehicle_learned_policy_collapse_diagnostic_v2_full_validation64",
        "backup_verified": False,
        "reason": "Diagnostic artifacts were created after the shard11-audit backup request. External backup should cover these artifacts and finalized run logs before any new formal rollout/final-test gate.",
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
    completed_sha = sha256(completed_path)

    doc_body = f"""
## 2026-09-27 vehicle learned-policy collapse diagnostic v2 (full validation64)

UTC: {created}. Existing-output diagnostic for learned_s0/s1/s2 across validation64 shards 00-11 completed with validation_accessed=true only because saved validation output traces/summaries were read; validation bank reopened=false; sealed final test accessed/opened/hashed=false; simulations/control_steps/gradient_steps=0/0/0. Learned trace episodes read: {learned_episode_dirs}; scanned episode directories: {scanned_episode_dirs}.

Key classifications: {json.dumps(conclusions, sort_keys=True)}

Artifacts: `{rel(raw_path)}`, `{rel(summary_path)}`, `{rel(completed_path)}` (SHA256 `{completed_sha}`). Backup request: `{rel(backup_request)}`. Method remains IMPROVED latency-tree, not ORIGINAL SAC; no final-test or reproduction-success claim is supported. Next scientific step is full paired validation aggregation/model selection and then targeted case43/instrumentation diagnostics or versioned method revision as indicated by the aggregate evidence.
"""
    for name in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        append_once(ROOT / name, DOC_MARKER, doc_body)

    print(json.dumps({
        "completed": rel(completed_path),
        "completed_sha256": completed_sha,
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
