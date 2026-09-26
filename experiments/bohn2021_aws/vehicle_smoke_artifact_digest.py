#!/usr/bin/env python3
"""Metadata-only audit/digest for the AWS vehicle development smoke outputs.

This script reads only the already-created non-formal vehicle smoke diagnostic
bundle under research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing.
It performs no simulations, opens no validation bank/content, and opens no sealed
test bank/content.  Its purpose is to independently verify hashes/replay/budget
accounting and to extract concise timing and seed2 H35 trigger evidence from the
smoke traces.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest"
OUT_JSON = OUT_DIR / "raw.json"
OUT_MD = OUT_DIR / "summary.md"
OUT_COMPLETED = OUT_DIR / "completed.json"
MARKER = "vehicle-smoke-artifact-digest-20260926"
SOURCE_MARKER = "vehicle-development-smoke-pairing-20260926"
VEHICLE_FEATURE_NAMES = [
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
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_record_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


def percentile(values: List[float], q: float) -> float:
    assert values
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def stats(values: Iterable[float]) -> Dict[str, Any]:
    xs = [float(v) for v in values]
    if not xs:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None, "min": None}
    return {
        "count": len(xs),
        "sum": math.fsum(xs),
        "mean": math.fsum(xs) / len(xs),
        "median": percentile(xs, 50.0),
        "p95": percentile(xs, 95.0),
        "max": max(xs),
        "min": min(xs),
    }


def verify_hash_map(hash_map: Dict[str, str]) -> Dict[str, Any]:
    checked = 0
    missing: List[str] = []
    mismatches: List[Dict[str, str]] = []
    by_suffix = Counter()
    for name, expected in sorted(hash_map.items()):
        path = resolve_record_path(name)
        suffix = path.suffix or "<none>"
        by_suffix[suffix] += 1
        if not path.exists():
            missing.append(name)
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    return {
        "hash_records": len(hash_map),
        "checked_existing": checked,
        "missing": missing,
        "mismatches": mismatches,
        "passed": not missing and not mismatches and checked == len(hash_map),
        "suffix_counts": dict(sorted(by_suffix.items())),
    }


def strip_for_replay(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(trace)
    for row in cleaned:
        row.pop("timing", None)
        for attempt in row.get("recovery", {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    if OUT_COMPLETED.exists():
        done = read_json(OUT_COMPLETED)
        assert done.get("passed") is True, "existing completed digest did not pass"
        for name, expected in done.get("hashes", {}).items():
            actual = sha256(resolve_record_path(name))
            assert actual == expected, name
        raise SystemExit("vehicle smoke artifact digest already completed and verified; refusing to rerun")
    leftovers = [p.name for p in OUT_DIR.iterdir() if p.name not in {"run.lock"}]
    assert not leftovers, "Partial digest output exists; inspect before recovery: " + ", ".join(sorted(leftovers))


def load_trace(ep_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = ROOT / ep_summary["path"] / "trace.json"
    trace = read_json(path)
    assert isinstance(trace, list) and len(trace) == int(ep_summary["steps"]), (path, len(trace), ep_summary.get("steps"))
    return trace


def arm_policy(raw: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    for arm in raw.get("arms", []):
        if arm.get("arm_id") == arm_id:
            return arm.get("policy") or {}
    return {}


def tree_path(policy: Dict[str, Any], features: List[float]) -> Dict[str, Any]:
    nodes = policy.get("nodes") or []
    leaves = policy.get("leaves") or []
    if policy.get("kind") != "tree" or len(nodes) != 3 or len(leaves) != 4:
        return {"available": False}
    root = nodes[0]
    root_feature = int(root["feature"])
    root_value = float(features[root_feature])
    root_right = root_value > float(root["threshold"])
    child_index = 1 + int(root_right)
    child = nodes[child_index]
    child_feature = int(child["feature"])
    child_value = float(features[child_feature])
    child_right = child_value > float(child["threshold"])
    leaf = 2 * int(root_right) + int(child_right)
    return {
        "available": True,
        "root": {
            "feature_index": root_feature,
            "feature_name": VEHICLE_FEATURE_NAMES[root_feature],
            "value": root_value,
            "threshold": float(root["threshold"]),
            "went_right_gt_threshold": bool(root_right),
        },
        "child_node_index": child_index,
        "child": {
            "feature_index": child_feature,
            "feature_name": VEHICLE_FEATURE_NAMES[child_feature],
            "value": child_value,
            "threshold": float(child["threshold"]),
            "went_right_gt_threshold": bool(child_right),
        },
        "leaf": leaf,
        "leaf_horizon": leaves[leaf],
    }


def compact_state(row: Dict[str, Any]) -> Dict[str, Any]:
    state = (row.get("policy_context") or {}).get("state") or row.get("previous_state") or {}
    previous_input = (row.get("policy_context") or {}).get("previous_input") or {}
    return {
        "state": {k: state.get(k) for k in ("x", "y", "theta") if k in state},
        "previous_input": previous_input,
        "previous_h": (row.get("policy_context") or {}).get("previous_h"),
        "previous_initial_failure": (row.get("policy_context") or {}).get("previous_initial_failure"),
        "previous_final_failure": (row.get("policy_context") or {}).get("previous_final_failure"),
    }


def make_arm_accumulator(seed: Any = None, family: Any = None) -> Dict[str, Any]:
    return {
        "seed": seed,
        "family": family,
        "episodes": 0,
        "steps": 0,
        "success_count": 0,
        "episode_failure_count": 0,
        "constraint_count": 0,
        "initial_failed_steps": 0,
        "final_solver_failure_steps": 0,
        "retries": 0,
        "recovered_steps": 0,
        "deadline_exceed_steps_summary": 0,
        "resets_metered": 0,
        "environment_constructions": 0,
        "total_cost_sum": 0.0,
        "physical_constraint_cost_sum": 0.0,
        "performance_cost_sum": 0.0,
        "constraint_cost_sum": 0.0,
        "h_penalty_sum": 0.0,
        "switches_summary": 0,
        "horizon_counts": Counter(),
        "decision_s": [],
        "decision_gross_s": [],
        "controller_s": [],
        "selection_s": [],
        "logging_s": [],
        "solver_s": [],
        "episode_indices": [],
    }


def add_episode_to_acc(acc: Dict[str, Any], summary: Dict[str, Any], trace: List[Dict[str, Any]]) -> None:
    acc["episodes"] += 1
    acc["steps"] += int(summary.get("steps", 0))
    acc["success_count"] += int(bool(summary.get("success")))
    acc["episode_failure_count"] += int(bool(summary.get("episode_failure")))
    acc["constraint_count"] += int(bool(summary.get("constraint")))
    acc["initial_failed_steps"] += int(summary.get("initial_failed_steps", 0))
    acc["final_solver_failure_steps"] += int(summary.get("solver_failure_steps", 0))
    acc["retries"] += int(summary.get("retries", 0))
    acc["recovered_steps"] += int(summary.get("recovered_steps", 0))
    acc["deadline_exceed_steps_summary"] += int(summary.get("deadline_exceed_steps", 0))
    acc["resets_metered"] += int(summary.get("resets_metered", 0))
    acc["environment_constructions"] += 1
    for key in ("total_cost", "physical_constraint_cost", "performance_cost", "constraint_cost", "h_penalty"):
        acc[key + "_sum"] += float(summary.get(key, 0.0))
    acc["switches_summary"] += int(summary.get("switches", 0))
    acc["episode_indices"].append(int(summary.get("episode_index", -1)))
    for h, n in (summary.get("horizon_counts") or {}).items():
        acc["horizon_counts"][str(h)] += int(n)
    for row in trace:
        timing = row.get("timing") or {}
        for key in ("decision_s", "decision_gross_s", "controller_s", "selection_s", "logging_s"):
            if timing.get(key) is not None:
                acc[key].append(float(timing[key]))
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                acc["solver_s"].append(float(attempt["solver_s"]))


def finalize_arm_acc(acc: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in acc.items() if k not in {"horizon_counts", "decision_s", "decision_gross_s", "controller_s", "selection_s", "logging_s", "solver_s"}}
    out["horizon_counts"] = dict(sorted(acc["horizon_counts"].items(), key=lambda kv: int(kv[0])))
    out["unique_horizons"] = [int(k) for k in out["horizon_counts"].keys()]
    out["decision_timing_s_exact_from_traces"] = stats(acc["decision_s"])
    out["decision_gross_timing_s_exact_from_traces"] = stats(acc["decision_gross_s"])
    out["controller_timing_s_exact_from_traces"] = stats(acc["controller_s"])
    out["selection_timing_s_exact_from_traces"] = stats(acc["selection_s"])
    out["logging_timing_s_exact_from_traces"] = stats(acc["logging_s"])
    out["solver_attempt_timing_s_exact_from_traces"] = stats(acc["solver_s"])
    out["deadline_exceed_steps_recomputed_from_traces"] = sum(1 for v in acc["decision_s"] if v > 0.1)
    out["total_cost_mean_episode"] = out["total_cost_sum"] / out["episodes"] if out["episodes"] else None
    out["decision_mean_s_per_step_exact"] = out["decision_timing_s_exact_from_traces"]["mean"]
    return out


def replay_audit(episode_summaries: List[Dict[str, Any]], traces_by_episode: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for summary in episode_summaries:
        grouped[(summary["arm_id"], int(summary["case"]))].append(summary)
    pairs_checked = 0
    mismatches: List[Dict[str, Any]] = []
    clean_hashes: Dict[str, Dict[str, str]] = {}
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: int(r["repeat"]))
        clean_hashes[str(key)] = {}
        if len(rows) != 2:
            mismatches.append({"arm_case": str(key), "reason": "expected two repeats", "count": len(rows)})
            continue
        left = strip_for_replay(traces_by_episode[int(rows[0]["episode_index"])])
        right = strip_for_replay(traces_by_episode[int(rows[1]["episode_index"])])
        left_hash = canonical_hash(left)
        right_hash = canonical_hash(right)
        clean_hashes[str(key)] = {"repeat_%s" % rows[0]["repeat"]: left_hash, "repeat_%s" % rows[1]["repeat"]: right_hash}
        pairs_checked += 1
        if left != right:
            mismatches.append({
                "arm_id": key[0],
                "case": key[1],
                "repeat_a": rows[0]["repeat"],
                "repeat_b": rows[1]["repeat"],
                "clean_hash_a": left_hash,
                "clean_hash_b": right_hash,
            })
    return {"pairs_checked": pairs_checked, "mismatches": mismatches, "passed": not mismatches, "clean_trace_hashes": clean_hashes}


def trigger_intervals(triggers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []
    by_episode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for trig in triggers:
        by_episode[int(trig["episode_index"])].append(trig)
    for episode_index, rows in sorted(by_episode.items()):
        rows = sorted(rows, key=lambda r: int(r["step_index"]))
        start = prev = int(rows[0]["step_index"])
        repeat = rows[0]["repeat"]
        case = rows[0]["case"]
        for row in rows[1:]:
            step = int(row["step_index"])
            if step == prev + 1:
                prev = step
                continue
            intervals.append({"episode_index": episode_index, "repeat": repeat, "case": case, "start_step": start, "end_step": prev, "length": prev - start + 1})
            start = prev = step
        intervals.append({"episode_index": episode_index, "repeat": repeat, "case": case, "start_step": start, "end_step": prev, "length": prev - start + 1})
    return intervals


def documentation_markers() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        out[name] = {
            "exists": path.exists(),
            "source_smoke_marker_present_before_digest_append": SOURCE_MARKER in text,
            "digest_marker_present_before_append": MARKER in text,
        }
    return out


def append_docs(payload: Dict[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-26 vehicle smoke artifact digest\n\n"
        f"UTC: {payload['created_utc']}. Metadata-only audit of the non-formal AWS vehicle smoke outputs completed. "
        f"Top-level completed hash check passed={payload['completed_hash_audit']['top_level']['passed']}; "
        f"episode completed hash check passed={payload['completed_hash_audit']['episodes']['passed']}; "
        f"independent replay passed={payload['replay_audit_independent']['passed']} with {payload['replay_audit_independent']['pairs_checked']} pairs. "
        f"Seed2 H35 trigger count={payload['seed2_h35']['count']} and intervals={payload['seed2_h35']['intervals']}. "
        "No simulations were run; validation_accessed=false; test_accessed=false. This remains engineering/development evidence only, not model selection or reproduction evidence. "
        f"Artifacts: `{rel(OUT_JSON)}`, `{rel(OUT_MD)}`, `{rel(OUT_COMPLETED)}`. New digest artifacts require backup before formal evidence.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# Vehicle smoke artifact digest",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "Scope: metadata-only audit of `vehicle_development_smoke_pairing` outputs. No simulations, no validation reads, no sealed-test reads.",
        "",
        "## Hash/replay/budget audit",
        "",
        f"- top-level completed hash check: {payload['completed_hash_audit']['top_level']['passed']} ({payload['completed_hash_audit']['top_level']['checked_existing']}/{payload['completed_hash_audit']['top_level']['hash_records']} files checked)",
        f"- episode completed hash check: {payload['completed_hash_audit']['episodes']['passed']} ({payload['completed_hash_audit']['episodes']['checked_existing']}/{payload['completed_hash_audit']['episodes']['hash_records']} files checked)",
        f"- independent replay check: {payload['replay_audit_independent']['passed']} ({payload['replay_audit_independent']['pairs_checked']} pairs)",
        f"- reconstructed budget: {payload['budget_reconstructed']}",
        f"- raw flags: validation_accessed={payload['input_flags']['validation_accessed']}, test_accessed={payload['input_flags']['test_accessed']}, formal_scientific_evidence={payload['input_flags']['formal_scientific_evidence']}",
        "",
        "## Exact per-arm timing from traces",
        "",
        "| arm | episodes | steps | success | constraints | final fail steps | deadlines | total cost | physical cost | decision mean | decision median | decision p95 | solver mean | solver p95 | horizons | switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm_id in sorted(payload["arm_audit"]):
        arm = payload["arm_audit"][arm_id]
        decision = arm["decision_timing_s_exact_from_traces"]
        solver = arm["solver_attempt_timing_s_exact_from_traces"]
        lines.append(
            f"| `{arm_id}` | {arm['episodes']} | {arm['steps']} | {arm['success_count']} | {arm['constraint_count']} | "
            f"{arm['final_solver_failure_steps']} | {arm['deadline_exceed_steps_recomputed_from_traces']} | "
            f"{arm['total_cost_sum']:.6g} | {arm['physical_constraint_cost_sum']:.6g} | "
            f"{decision['mean']:.6g} | {decision['median']:.6g} | {decision['p95']:.6g} | "
            f"{solver['mean']:.6g} | {solver['p95']:.6g} | {arm['horizon_counts']} | {arm['switches_summary']} |"
        )
    lines.extend([
        "",
        "## Seed2 H35 triggers",
        "",
        f"- trigger count: {payload['seed2_h35']['count']}",
        f"- trigger intervals: {payload['seed2_h35']['intervals']}",
        "",
        "| episode | repeat | case | step | leaf | heading_error_5 | abs_yaw_input | decision_s | solver_s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for trig in payload["seed2_h35"]["triggers"]:
        f = trig["features"]
        lines.append(
            f"| {trig['episode_index']} | {trig['repeat']} | {trig['case']} | {trig['step_index']} | {trig['leaf']} | "
            f"{f['heading_error_5']:.6g} | {f['abs_yaw_input']:.6g} | {trig['decision_s']:.6g} | {trig['solver_s']:.6g} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This digest strengthens engineering auditability of the AWS smoke run only. It confirms the smoke bundle is internally hash-consistent and deterministic after timing/solver-time stripping, and that seed2 really selected H35 in two short intervals. It does not open validation64 or sealed test and must not be used as formal model-selection evidence.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    assert_no_prior_partial()
    if not IN_DIR.exists():
        raise FileNotFoundError(IN_DIR)
    raw_path = IN_DIR / "raw.json"
    completed_path = IN_DIR / "completed.json"
    schedule_path = IN_DIR / "schedule.json"
    raw = read_json(raw_path)
    completed = read_json(completed_path)
    schedule = read_json(schedule_path)
    flags = {
        "validation_accessed": bool(raw.get("validation_accessed")),
        "validation64_bank_content_opened": bool(raw.get("validation64_bank_content_opened")),
        "test_accessed": bool(raw.get("test_accessed")),
        "sealed_test_bank_content_opened": bool(raw.get("sealed_test_bank_content_opened")),
        "formal_scientific_evidence": bool(raw.get("formal_scientific_evidence")),
        "split": raw.get("split"),
        "new_simulations_in_source_smoke": int((raw.get("budget_actual") or {}).get("episodes", 0)),
    }
    assert flags["validation_accessed"] is False
    assert flags["validation64_bank_content_opened"] is False
    assert flags["test_accessed"] is False
    assert flags["sealed_test_bank_content_opened"] is False
    assert flags["formal_scientific_evidence"] is False
    assert raw.get("split") == "vehicle_smoke_bank_only"
    assert completed.get("passed") is True

    top_audit = verify_hash_map(completed.get("hashes") or {})
    episode_dirs = sorted((IN_DIR / "episodes").iterdir())
    episode_hash_records: Dict[str, str] = {}
    episode_completed_files = 0
    for ep_dir in episode_dirs:
        ep_completed = ep_dir / "completed.json"
        if ep_completed.exists():
            episode_completed_files += 1
            ep_done = read_json(ep_completed)
            assert ep_done.get("passed") is True, ep_completed
            episode_hash_records.update(ep_done.get("hashes") or {})
    episode_hash_audit = verify_hash_map(episode_hash_records)

    episode_summaries = sorted(raw.get("episodes") or [], key=lambda e: int(e["episode_index"]))
    assert len(episode_summaries) == 24
    schedule_rows = sorted(schedule.get("episodes") or [], key=lambda e: int(e["episode_index"]))
    assert len(schedule_rows) == len(episode_summaries)
    schedule_mismatches: List[Dict[str, Any]] = []
    for row, summary in zip(schedule_rows, episode_summaries):
        for key in ("episode_index", "repeat", "case", "arm_id", "seed"):
            if row.get(key) != summary.get(key):
                schedule_mismatches.append({"episode_index": row.get("episode_index"), "key": key, "schedule": row.get(key), "summary": summary.get(key)})

    traces_by_episode: Dict[int, List[Dict[str, Any]]] = {}
    arm_acc: Dict[str, Dict[str, Any]] = {}
    seed2_policy = arm_policy(raw, "learned_latency_tree_vehicle_s2")
    h35_triggers: List[Dict[str, Any]] = []
    for summary in episode_summaries:
        ep_index = int(summary["episode_index"])
        trace = load_trace(summary)
        traces_by_episode[ep_index] = trace
        arm_id = summary["arm_id"]
        if arm_id not in arm_acc:
            arm_acc[arm_id] = make_arm_accumulator(summary.get("seed"), summary.get("family"))
        add_episode_to_acc(arm_acc[arm_id], summary, trace)
        if arm_id == "learned_latency_tree_vehicle_s2":
            for step_index, row in enumerate(trace):
                if int(row.get("horizon")) != 35:
                    continue
                feat_values = row.get("tree_features") or (row.get("decision") or {}).get("features") or []
                features_by_name = {name: float(feat_values[i]) for i, name in enumerate(VEHICLE_FEATURE_NAMES)}
                path = tree_path(seed2_policy, list(feat_values))
                solver_values = [float(a["solver_s"]) for a in (row.get("recovery") or {}).get("attempts", []) if a.get("solver_s") is not None]
                h35_triggers.append({
                    "episode_index": ep_index,
                    "repeat": int(summary["repeat"]),
                    "case": int(summary["case"]),
                    "step_index": int(step_index),
                    "horizon": 35,
                    "leaf": (row.get("decision") or {}).get("leaf"),
                    "features": features_by_name,
                    "tree_path": path,
                    "compact_state": compact_state(row),
                    "decision_s": float((row.get("timing") or {}).get("decision_s")),
                    "solver_s": solver_values[0] if solver_values else None,
                    "performance": float(row.get("performance", 0.0)),
                    "constraint": float(row.get("constraint", 0.0)),
                    "solver_success": bool(row.get("solver_success")),
                })

    replay = replay_audit(episode_summaries, traces_by_episode)
    arm_audit = {arm_id: finalize_arm_acc(acc) for arm_id, acc in sorted(arm_acc.items())}
    reconstructed_budget = {
        "episodes": len(episode_summaries),
        "control_steps_from_summaries": sum(int(e.get("steps", 0)) for e in episode_summaries),
        "control_steps_from_traces": sum(len(t) for t in traces_by_episode.values()),
        "resets_metered": sum(int(e.get("resets_metered", 0)) for e in episode_summaries),
        "environment_constructions": len(episode_summaries),
        "new_gradient_steps": 0,
        "validation_episodes": 0,
        "test_episodes": 0,
    }
    raw_budget = raw.get("budget_actual") or {}
    budget_match = {
        "episodes_match_raw": reconstructed_budget["episodes"] == raw_budget.get("episodes"),
        "control_steps_match_raw": reconstructed_budget["control_steps_from_summaries"] == raw_budget.get("control_steps"),
        "resets_match_raw": reconstructed_budget["resets_metered"] == raw_budget.get("resets"),
        "environment_constructions_match_raw": reconstructed_budget["environment_constructions"] == raw_budget.get("environment_constructions"),
    }

    payload: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "metadata-only vehicle smoke artifact digest; no simulation; IMPROVED latency-tree audit only",
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_directory": rel(IN_DIR),
        "input_files": {
            "raw_json": {"path": rel(raw_path), "sha256": sha256(raw_path)},
            "summary_md": {"path": rel(IN_DIR / "summary.md"), "sha256": sha256(IN_DIR / "summary.md")},
            "completed_json": {"path": rel(completed_path), "sha256": sha256(completed_path)},
            "schedule_json": {"path": rel(schedule_path), "sha256": sha256(schedule_path)},
        },
        "input_flags": flags,
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "formal_scientific_evidence": False,
        "completed_hash_audit": {
            "top_level": top_audit,
            "episodes": dict(episode_hash_audit, episode_completed_files=episode_completed_files),
        },
        "schedule_audit": {
            "rows": len(schedule_rows),
            "order_seed": schedule.get("order_seed"),
            "mismatches": schedule_mismatches,
            "passed": not schedule_mismatches,
        },
        "replay_audit_from_raw": raw.get("replay"),
        "replay_audit_independent": replay,
        "budget_raw": raw_budget,
        "budget_reconstructed": reconstructed_budget,
        "budget_match": budget_match,
        "arm_audit": arm_audit,
        "paired_comparisons_from_raw": raw.get("paired_comparisons"),
        "seed2_h35": {
            "count": len(h35_triggers),
            "intervals": trigger_intervals(h35_triggers),
            "triggers": h35_triggers,
            "policy": seed2_policy,
            "interpretation": "Seed2 structurally switching tree used H35 only where leaf0 was reached; smoke shows actual switching but no smoke-level benefit and remains non-formal.",
        },
        "documentation_markers_before_append": documentation_markers(),
        "interpretation_boundary": "Engineering/development audit only; not validation/model selection; final sealed test remains unauthorized.",
    }
    assert payload["completed_hash_audit"]["top_level"]["passed"] is True
    assert payload["completed_hash_audit"]["episodes"]["passed"] is True
    assert payload["schedule_audit"]["passed"] is True
    assert payload["replay_audit_independent"]["passed"] is True
    assert all(budget_match.values())
    assert len(h35_triggers) == int((raw.get("aggregates") or {}).get("learned_latency_tree_vehicle_s2", {}).get("horizon_counts", {}).get("35", 0))

    write_json(OUT_JSON, payload)
    write_summary(payload)
    write_json(OUT_COMPLETED, {
        "passed": True,
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "formal_scientific_evidence": False,
        "hashes": {rel(OUT_JSON): sha256(OUT_JSON), rel(OUT_MD): sha256(OUT_MD)},
    })
    append_docs(payload)
    print(json.dumps({
        "wrote": rel(OUT_JSON),
        "summary": rel(OUT_MD),
        "completed": rel(OUT_COMPLETED),
        "top_hash_passed": top_audit["passed"],
        "episode_hash_passed": episode_hash_audit["passed"],
        "replay_passed": replay["passed"],
        "h35_trigger_count": len(h35_triggers),
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
