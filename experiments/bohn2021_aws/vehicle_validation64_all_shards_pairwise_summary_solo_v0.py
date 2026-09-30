#!/usr/bin/env python3
"""Zero-resource all-shard numerical summary for vehicle validation64 outputs.

This diagnostic reads only completed validation result artifacts that have already
been produced under the frozen validation64 campaign.  It does not open the
validation bank, does not run the plant/controller, does not train or refit any
model, and does not access sealed/final tests.  Its purpose is to consolidate the
available validation evidence into a reproducible numerical table before choosing
another nonzero-resource intervention.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STATE = Path("/data/openai-agent/state")
TASK_ID = "S-VAL64-all-shards-pairwise-summary-v0"
NAME = "vehicle_validation64_all_shards_pairwise_summary_solo_v0"
INPUT_ROOT = ROOT / "research_artifacts/aws_formal_validation/vehicle_validation64_20260926"
OUTPUT_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_PROOF_ROOT = ROOT / "research_artifacts/aws_backup_proofs"
EXPECTED_SHARDS = list(range(12))
RESOURCE_ZERO = {
    "solver_calls": 0,
    "plant_steps": 0,
    "training_steps": 0,
    "validation_episodes": 0,
    "test_episodes": 0,
}
SUM_FIELDS = [
    "constraint_cost_sum",
    "constraint_count",
    "construction_total_s",
    "deadline_exceed_steps",
    "decision_gross_total_s",
    "decision_total_s",
    "episode_failure_count",
    "episodes",
    "h_penalty_sum",
    "initial_failed_steps",
    "final_failed_steps",
    "logging_total_s",
    "performance_cost_sum",
    "physical_constraint_cost_sum",
    "recovered_steps",
    "reset_total_s",
    "retries",
    "solver_failure_steps",
    "steps",
    "success_count",
    "switches",
    "total_cost_sum",
]
NUMERIC_HINTS = (
    "cost",
    "steps",
    "success",
    "failure",
    "constraint",
    "decision",
    "solver",
    "horizon",
    "episode",
    "time",
    "wall",
    "physical",
    "performance",
)


def _import_execution_contract() -> Any:
    service_dir = ROOT / "scripts/research_service"
    if str(service_dir) not in sys.path:
        sys.path.insert(0, str(service_dir))
    import execution_contract  # type: ignore

    return execution_contract


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def clean(obj: Any, max_list: Optional[int] = None) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v, max_list=max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        values = obj if max_list is None else obj[:max_list]
        out = [clean(v, max_list=max_list) for v in values]
        if max_list is not None and len(obj) > max_list:
            out.append({"truncated_count": len(obj) - max_list})
        return out
    if isinstance(obj, tuple):
        return [clean(v, max_list=max_list) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return str(obj)
    return str(obj)


def add_numeric(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    for key in SUM_FIELDS:
        value = src.get(key, 0)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            dst[key] = dst.get(key, 0.0) + value
    hc = src.get("horizon_counts") or {}
    if isinstance(hc, Mapping):
        out_hc = dst.setdefault("horizon_counts", {})
        for h, count in hc.items():
            try:
                out_hc[str(h)] = int(out_hc.get(str(h), 0)) + int(count)
            except Exception:
                pass
    unique = set(dst.get("unique_horizons", []))
    for h in src.get("unique_horizons", []) or []:
        try:
            unique.add(int(h))
        except Exception:
            unique.add(str(h))
    if unique:
        try:
            dst["unique_horizons"] = sorted(unique, key=lambda x: int(x))
        except Exception:
            dst["unique_horizons"] = sorted(unique, key=str)


def finalize_aggregate(agg: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(agg))
    episodes = float(out.get("episodes", 0) or 0)
    steps = float(out.get("steps", 0) or 0)
    success = float(out.get("success_count", 0) or 0)
    failures = float(out.get("episode_failure_count", 0) or 0)
    if episodes > 0:
        out["success_rate"] = success / episodes
        out["episode_failure_rate"] = failures / episodes
        for name in ("physical_constraint_cost_sum", "total_cost_sum", "performance_cost_sum"):
            out[name.replace("_sum", "_mean_episode")] = float(out.get(name, 0.0) or 0.0) / episodes
    if steps > 0:
        for total_name, mean_name in (
            ("decision_total_s", "decision_mean_s_per_step_recomputed"),
            ("decision_gross_total_s", "decision_gross_mean_s_per_step_recomputed"),
            ("construction_total_s", "construction_mean_s_per_step"),
            ("reset_total_s", "reset_mean_s_per_step"),
            ("logging_total_s", "logging_mean_s_per_step"),
        ):
            out[mean_name] = float(out.get(total_name, 0.0) or 0.0) / steps
    return clean(out)


def merge_aggregates(raws: Sequence[Mapping[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw in raws:
        section = raw.get(key_name) or {}
        if not isinstance(section, Mapping):
            continue
        for key, agg in section.items():
            if not isinstance(agg, Mapping):
                continue
            add_numeric(merged.setdefault(str(key), {}), agg)
    return {k: finalize_aggregate(v) for k, v in sorted(merged.items())}


def best_by_success_then_cost(entries: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = []
    for key, agg in entries.items():
        episodes = int(agg.get("episodes", 0) or 0)
        if episodes <= 0:
            continue
        success = int(agg.get("success_count", 0) or 0)
        failures = int(agg.get("episode_failure_count", 0) or 0)
        phys = float(agg.get("physical_constraint_cost_mean_episode", float("inf")))
        decision = float(agg.get("decision_mean_s_per_step_recomputed", float("inf")))
        candidates.append((success == episodes and failures == 0, success / float(episodes), -phys, -decision, key, agg))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _all_success, _rate, _neg_phys, _neg_decision, key, agg = candidates[0]
    return {"key": key, "aggregate": clean(agg)}


def lowest_cost_all_success(entries: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = []
    for key, agg in entries.items():
        episodes = int(agg.get("episodes", 0) or 0)
        if episodes <= 0:
            continue
        success = int(agg.get("success_count", 0) or 0)
        failures = int(agg.get("episode_failure_count", 0) or 0)
        if success == episodes and failures == 0:
            candidates.append((float(agg.get("physical_constraint_cost_mean_episode", float("inf"))), key, agg))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    _phys, key, agg = candidates[0]
    return {"key": key, "aggregate": clean(agg)}


def matched_fixed_grid(rollouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    by_seed: Dict[str, Dict[str, Dict[str, Any]]] = {"0": {}, "1": {}, "2": {}}
    pattern = re.compile(r"^fixed_seed(?P<seed>\d+)_terminal25_controllerH(?P<h>\d+)$")
    for key, agg in rollouts.items():
        match = pattern.match(key)
        if match:
            by_seed.setdefault(match.group("seed"), {})["H" + match.group("h")] = agg
    summary: Dict[str, Any] = {}
    for seed, entries in sorted(by_seed.items(), key=lambda kv: kv[0]):
        seed_summary = {
            "arms": entries,
            "best_by_success_then_physical_cost": best_by_success_then_cost(entries),
            "lowest_physical_cost_among_all_success": lowest_cost_all_success(entries),
        }
        summary["seed" + seed] = seed_summary
    return summary


def independent_terminal_grid(rollouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    entries: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r"^fixed_seed0_terminal(?P<t>\d+)_controllerH(?P<h>\d+)$")
    for key, agg in rollouts.items():
        match = pattern.match(key)
        if match and match.group("t") == match.group("h"):
            entries["H" + match.group("h")] = agg
    return {
        "arms": entries,
        "best_by_success_then_physical_cost": best_by_success_then_cost(entries),
        "lowest_physical_cost_among_all_success": lowest_cost_all_success(entries),
    }


def compare_learned_to_seed_fixed(rollouts: Mapping[str, Mapping[str, Any]], fixed_grid: Mapping[str, Any]) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    for seed in ("0", "1", "2"):
        learned_key = "learned_s" + seed
        learned = rollouts.get(learned_key)
        seed_grid = (fixed_grid.get("seed" + seed) or {}) if isinstance(fixed_grid, Mapping) else {}
        arms = (seed_grid.get("arms") or {}) if isinstance(seed_grid, Mapping) else {}
        fixed_h25 = arms.get("H25") if isinstance(arms, Mapping) else None
        best = seed_grid.get("lowest_physical_cost_among_all_success") if isinstance(seed_grid, Mapping) else None
        comparisons[learned_key] = {
            "learned": learned,
            "matched_fixed_H25": fixed_h25,
            "best_all_success_matched_fixed": best,
            "deltas_vs_fixed_H25": delta_metrics(learned, fixed_h25),
            "deltas_vs_best_all_success_matched_fixed": delta_metrics(learned, best.get("aggregate") if isinstance(best, Mapping) else None),
        }
    return comparisons


def delta_metrics(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return None
    out: Dict[str, Any] = {}
    for key in (
        "success_rate",
        "episode_failure_rate",
        "physical_constraint_cost_mean_episode",
        "total_cost_mean_episode",
        "decision_mean_s_per_step_recomputed",
        "decision_gross_mean_s_per_step_recomputed",
        "steps",
        "switches",
    ):
        try:
            out[key + "_delta_learned_minus_comparator"] = float(a.get(key, 0.0) or 0.0) - float(b.get(key, 0.0) or 0.0)
        except Exception:
            pass
    if b.get("physical_constraint_cost_mean_episode"):
        try:
            out["physical_constraint_cost_mean_episode_ratio"] = float(a.get("physical_constraint_cost_mean_episode", 0.0) or 0.0) / float(b.get("physical_constraint_cost_mean_episode"))
        except Exception:
            pass
    if b.get("decision_mean_s_per_step_recomputed"):
        try:
            out["decision_mean_s_per_step_ratio"] = float(a.get("decision_mean_s_per_step_recomputed", 0.0) or 0.0) / float(b.get("decision_mean_s_per_step_recomputed"))
        except Exception:
            pass
    return clean(out)


def possible_episode_record(obj: Mapping[str, Any]) -> bool:
    keys = set(str(k) for k in obj.keys())
    if "rollout_key" in keys and ("case" in keys or "case_index" in keys or "case_id" in keys):
        return True
    if "rollout_key" in keys and ("steps" in keys or "steps_executed" in keys) and ("success" in keys or "done" in keys):
        return True
    if "episode" in keys and "rollout_key" in keys:
        return True
    return False


def iter_dicts(obj: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def compact_episode_record(record: Mapping[str, Any], shard_index: int) -> Dict[str, Any]:
    preferred = [
        "shard_index",
        "rollout_key",
        "family",
        "controller_key",
        "case",
        "case_index",
        "case_id",
        "episode_index",
        "repeat",
        "success",
        "is_success",
        "constraint",
        "constraint_violation",
        "episode_failure",
        "failed",
        "steps",
        "control_steps",
        "steps_executed",
        "total_cost",
        "total_cost_sum",
        "physical_constraint_cost",
        "physical_constraint_cost_sum",
        "performance_cost",
        "performance_cost_sum",
        "decision_mean_s",
        "decision_mean_s_per_step",
        "decision_total_s",
        "horizon_counts",
        "horizons",
        "switches",
        "initial_failed_steps",
        "final_failed_steps",
        "solver_failure_steps",
        "terminated_reason",
        "stopped_reason",
    ]
    out: Dict[str, Any] = {"shard_index": shard_index}
    for key in preferred:
        if key in record:
            out[key] = clean(record[key], max_list=30)
    for key, value in record.items():
        if key in out:
            continue
        skey = str(key).lower()
        if any(hint in skey for hint in NUMERIC_HINTS) and not isinstance(value, (dict, list)):
            out[str(key)] = clean(value)
    return out


def record_case_value(record: Mapping[str, Any]) -> Optional[int]:
    for key in ("case", "case_index", "case_id", "validation_case", "validation_case_index"):
        if key in record:
            try:
                return int(record[key])
            except Exception:
                pass
    return None


def extract_case_records(raws: Sequence[Mapping[str, Any]], target_case: int) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    found_keys = set()
    for raw in raws:
        shard_index = int(raw.get("shard_index", -1) if isinstance(raw.get("shard_index", -1), int) else -1)
        for record in iter_dicts(raw):
            if not possible_episode_record(record):
                continue
            case_value = record_case_value(record)
            if case_value != target_case:
                continue
            compact = compact_episode_record(record, shard_index)
            key = json.dumps(compact, sort_keys=True, default=str)
            if key not in found_keys:
                found_keys.add(key)
                records.append(compact)
    by_rollout: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        rollout = str(record.get("rollout_key", record.get("controller_key", "unknown")))
        by_rollout.setdefault(rollout, []).append(record)
    focus_keys = [
        "learned_s2",
        "learned_s0",
        "learned_s1",
        "fixed_seed2_terminal25_controllerH25",
        "fixed_seed2_terminal25_controllerH35",
        "fixed_seed0_terminal25_controllerH25",
        "fixed_seed1_terminal25_controllerH25",
    ]
    return {
        "target_case": target_case,
        "records_found": len(records),
        "rollout_keys_found": sorted(by_rollout.keys()),
        "focus_records": {key: by_rollout.get(key, []) for key in focus_keys if key in by_rollout},
        "all_records_compact": sorted(records, key=lambda r: (str(r.get("rollout_key")), int(r.get("shard_index", -1)), str(r.get("episode_index", ""))))[:300],
        "truncated_all_records": max(0, len(records) - 300),
    }


def make_markdown(result: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Vehicle validation64 all-shard pairwise summary (solo v0)")
    lines.append("")
    lines.append("This is a zero-resource diagnostic over already-created validation output artifacts. It did not open the validation bank, did not run controllers, and did not access sealed/final tests.")
    lines.append("")
    lines.append(f"Created UTC: {result['created_utc']}")
    lines.append(f"Input shards read: {len(result['input_files']['raw_files'])} raw files and {len(result['input_files']['summary_files'])} summary files")
    lines.append("")
    lines.append("## Learned candidates across all 12 shards")
    lines.append("")
    lines.append("| rollout | episodes | success | failures | steps | physical+constraint mean/episode | total mean/episode | decision mean s/step | horizons | switches |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    learned = result.get("learned_rollouts", {})
    for key in ("learned_s0", "learned_s1", "learned_s2"):
        agg = learned.get(key, {}) if isinstance(learned, Mapping) else {}
        lines.append(
            "| `{}` | {} | {} | {} | {} | {:.6g} | {:.6g} | {:.6g} | {} | {} |".format(
                key,
                int(agg.get("episodes", 0) or 0),
                int(agg.get("success_count", 0) or 0),
                int(agg.get("episode_failure_count", 0) or 0),
                int(agg.get("steps", 0) or 0),
                float(agg.get("physical_constraint_cost_mean_episode", float("nan"))),
                float(agg.get("total_cost_mean_episode", float("nan"))),
                float(agg.get("decision_mean_s_per_step_recomputed", float("nan"))),
                agg.get("horizon_counts", {}),
                int(agg.get("switches", 0) or 0),
            )
        )
    lines.append("")
    lines.append("## Learned versus matched terminal25 fixed-H comparators")
    lines.append("")
    comparisons = result.get("learned_vs_matched_fixed", {})
    for key in ("learned_s0", "learned_s1", "learned_s2"):
        comp = comparisons.get(key, {}) if isinstance(comparisons, Mapping) else {}
        fixed = comp.get("matched_fixed_H25") or {}
        best = comp.get("best_all_success_matched_fixed") or {}
        delta_h25 = comp.get("deltas_vs_fixed_H25") or {}
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"Matched fixed H25 physical mean/episode: {fixed.get('physical_constraint_cost_mean_episode')}; decision mean s/step: {fixed.get('decision_mean_s_per_step_recomputed')}")
        if isinstance(best, Mapping):
            lines.append(f"Best all-success matched fixed arm: `{best.get('key')}` with physical mean/episode {((best.get('aggregate') or {}).get('physical_constraint_cost_mean_episode') if isinstance(best.get('aggregate'), Mapping) else None)}")
        lines.append(f"Delta learned minus fixed H25: {json.dumps(delta_h25, sort_keys=True)}")
        lines.append("")
    case43 = result.get("case43_records", {})
    lines.append("## Case 43 record scan")
    lines.append("")
    lines.append(f"Records found for case 43 in existing raw outputs: {case43.get('records_found')}")
    lines.append(f"Rollout keys found: {case43.get('rollout_keys_found')}")
    focus = case43.get("focus_records", {}) if isinstance(case43, Mapping) else {}
    for key, records in focus.items():
        lines.append(f"- `{key}`: {json.dumps(records, sort_keys=True)[:1800]}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- These are validation/development outputs, not sealed/final test evidence.")
    lines.append("- Solo GPT-5.5 self-review is not independent Opus/Astra acceptance.")
    lines.append("- The current external backup service remains blocked for new nonzero-resource evidence; a backup request was written after this diagnostic.")
    lines.append("")
    return "\n".join(lines)


def run() -> Tuple[Dict[str, Any], Path]:
    execution_contract = _import_execution_contract()
    snapshot = execution_contract.runtime_snapshot(ROOT)
    if snapshot is None:
        raise RuntimeError("No structured execution snapshot is available")
    if snapshot.get("task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("Unexpected task_id in runtime snapshot: %r" % (snapshot.get("task", {}).get("task_id"),))
    created = now_utc()
    out_dir = OUTPUT_ROOT / (NAME + "_" + created.strftime("%Y%m%dT%H%M%SZ"))
    raw_files = [INPUT_ROOT / ("shard%02d" % i) / "raw.json" for i in EXPECTED_SHARDS]
    summary_files = [INPUT_ROOT / ("shard%02d" % i) / "summary.md" for i in EXPECTED_SHARDS]
    missing = [p for p in raw_files + summary_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing expected all-shard validation output files: " + ", ".join(rel(p) for p in missing))
    raws = [read_json(p) for p in raw_files]
    raw_hashes = {rel(p): sha256_file(p) for p in raw_files}
    summary_hashes = {rel(p): sha256_file(p) for p in summary_files}
    rollout_merged = merge_aggregates(raws, "aggregates_by_rollout_key")
    family_merged = merge_aggregates(raws, "aggregates_by_family")
    learned = {key: rollout_merged.get(key, {}) for key in ("learned_s0", "learned_s1", "learned_s2")}
    matched_grid = matched_fixed_grid(rollout_merged)
    independent_grid = independent_terminal_grid(rollout_merged)
    comparisons = compare_learned_to_seed_fixed(rollout_merged, matched_grid)
    case43 = extract_case_records(raws, 43)
    result: Dict[str, Any] = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "authority_mode": "temporary_user_authorized_solo_self_review",
        "analysis_type": "zero_resource_existing_validation_output_summary",
        "input_files": {
            "raw_files": [rel(p) for p in raw_files],
            "summary_files": [rel(p) for p in summary_files],
            "raw_sha256": raw_hashes,
            "summary_sha256": summary_hashes,
        },
        "safety_and_split": {
            "no_solver_calls": True,
            "no_plant_steps": True,
            "no_training_or_refit": True,
            "validation_bank_content_opened": False,
            "existing_validation_output_artifacts_read": True,
            "sealed_or_final_test_accessed": False,
            "new_validation_episodes": 0,
            "test_episodes": 0,
        },
        "resources": dict(RESOURCE_ZERO),
        "families_merged": family_merged,
        "learned_rollouts": learned,
        "matched_terminal25_fixed_h_grid": matched_grid,
        "independent_terminal_same_h_grid_seed0": independent_grid,
        "learned_vs_matched_fixed": comparisons,
        "case43_records": case43,
        "all_rollout_keys": sorted(rollout_merged.keys()),
    }
    raw_out = out_dir / "raw.json"
    summary_out = out_dir / "summary.md"
    completed_out = out_dir / "completed.json"
    backup_request = BACKUP_PROOF_ROOT / ("REQUEST_BACKUP_AFTER_VAL64_ALLSHARDS_SOLO_V0_" + created.strftime("%Y%m%dT%H%M%SZ") + ".json")
    write_json(raw_out, result)
    write_text(summary_out, make_markdown(result))
    completed = {
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "raw": rel(raw_out),
        "summary": rel(summary_out),
        "raw_sha256": sha256_file(raw_out),
        "summary_sha256": sha256_file(summary_out),
        "resources": dict(RESOURCE_ZERO),
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
    }
    write_json(completed_out, completed)
    backup_payload = {
        "request": "backup_after_vehicle_validation64_all_shards_pairwise_summary_solo_v0",
        "created_utc": created.isoformat(),
        "reason": "Zero-resource consolidation of existing validation64 outputs was created while nonzero source242 microcontinuation remained blocked by external backup HTTP-422/asset-cap status.",
        "required_coverage": [rel(raw_out), rel(summary_out), rel(completed_out), rel(Path(__file__).resolve())],
        "resources": dict(RESOURCE_ZERO),
        "sealed_or_final_test_accessed": False,
        "validation_bank_content_opened": False,
    }
    write_json(backup_request, backup_payload)
    result["outputs"] = {
        "raw": rel(raw_out),
        "summary": rel(summary_out),
        "completed": rel(completed_out),
        "backup_request": rel(backup_request),
        "completed_sha256": sha256_file(completed_out),
        "backup_request_sha256": sha256_file(backup_request),
    }
    write_json(raw_out, result)
    completed["raw_sha256"] = sha256_file(raw_out)
    completed["backup_request"] = rel(backup_request)
    completed["backup_request_sha256"] = sha256_file(backup_request)
    write_json(completed_out, completed)
    evidence = {
        "zero_solver_plant_training_validation_test_resources": True,
        "all_12_shard_raw_files_read": True,
        "all_12_shard_summaries_read": True,
        "existing_validation_outputs_read_only": True,
        "validation_bank_content_opened": False,
        "sealed_or_final_test_accessed": False,
        "aggregates_by_learned_seed_persisted": True,
        "matched_fixed_h_grid_summary_persisted": True,
        "case43_validation_development_record_summarized_if_present": True,
        "outputs_persisted": True,
        "backup_request_written": True,
        "raw_output": rel(raw_out),
        "summary_output": rel(summary_out),
        "completed_output": rel(completed_out),
        "backup_request": rel(backup_request),
        "learned_s2_episodes": int((learned.get("learned_s2") or {}).get("episodes", 0) or 0),
        "case43_records_found": int(case43.get("records_found", 0) or 0),
    }
    execution_contract.record_outcome(ROOT, "scientific_result", dict(RESOURCE_ZERO), evidence, engineering_error=None)
    return result, out_dir


def main() -> int:
    execution_contract = _import_execution_contract()
    try:
        _result, _out_dir = run()
        return 0
    except Exception as exc:
        created = now_utc()
        out_dir = OUTPUT_ROOT / (NAME + "_FAILED_" + created.strftime("%Y%m%dT%H%M%SZ"))
        failed_path = out_dir / "failed.json"
        payload = {
            "created_utc": created.isoformat(),
            "task_id": TASK_ID,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=20),
            "resources": dict(RESOURCE_ZERO),
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
        }
        write_json(failed_path, payload)
        evidence = {
            "no_scientific_outcome": True,
            "zero_solver_plant_training_validation_test_resources": True,
            "validation_bank_content_opened": False,
            "sealed_or_final_test_accessed": False,
            "failed_json": rel(failed_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(RESOURCE_ZERO), evidence, engineering_error="loader")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
