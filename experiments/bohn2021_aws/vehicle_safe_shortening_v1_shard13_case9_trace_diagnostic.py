#!/usr/bin/env python3
"""Metadata-only diagnostic for shard13 seed2 case9 rare off-H failure.

This script reads only the already-created fresh development-validation shard13
raw artifact. It performs no rollout, no training, no historical validation64
bank access, and no sealed-test access. It is intended to answer a narrow
question raised by the frozen safe-shortening v1 campaign: seed2 adaptive on
case9 made one H10 decision and failed, while same-seed fixed H25 succeeded.

The script inspects whatever per-step traces are present in raw.json, compares
adaptive/fixed-H25 episode summaries, inventories available fields relevant to
raw-request/executed-H/guard/fallback/solver/value/objective/normalization, and
records whether existing inputs suffice for a causal diagnosis or whether an
instrumented deterministic replay is required after the frozen campaign.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1"
DIAG_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SERVICE_START = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
H_GRID = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
TERMS = (
    "horizon", "raw", "request", "execut", "guard", "fallback", "clamp",
    "candidate", "value", "terminal", "normal", "score", "objective",
    "solver", "warm", "status", "fail", "retry", "cost", "state", "obs",
    "action", "control", "constraint",
)


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
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def fsum(values: Iterable[float]) -> float:
    return float(math.fsum(float(v) for v in values))


def mean(xs: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def parse_h_from_arm(arm_id: str) -> Optional[int]:
    m = re.search(r"_H(\d+)", arm_id)
    return int(m.group(1)) if m else None


def decision_sum(episode: Mapping[str, Any]) -> float:
    timing = episode.get("decision_timing_s") or {}
    if isinstance(timing, Mapping):
        val = finite_float(timing.get("sum"))
        if val is not None:
            return val
    val = finite_float(episode.get("decision_total_s"))
    return 0.0 if val is None else val


def numeric_vector_summary(v: Any, max_items: int = 6) -> Any:
    if not isinstance(v, list):
        return v
    nums: List[float] = []
    ok = True
    for item in v:
        y = finite_float(item)
        if y is None:
            ok = False
            break
        nums.append(y)
    if ok:
        norm = math.sqrt(fsum(x * x for x in nums)) if nums else 0.0
        return {"len": len(nums), "first": nums[:max_items], "l2_norm": norm, "min": min(nums) if nums else None, "max": max(nums) if nums else None}
    return {"len": len(v), "first": [compact(x, max_depth=1) for x in v[:max_items]]}


def compact(value: Any, max_depth: int = 2) -> Any:
    if max_depth <= 0:
        if isinstance(value, Mapping):
            return {"type": "dict", "keys": list(value.keys())[:12], "n_keys": len(value)}
        if isinstance(value, list):
            return numeric_vector_summary(value)
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k in list(value.keys())[:30]:
            out[str(k)] = compact(value[k], max_depth=max_depth - 1)
        if len(value) > 30:
            out["__truncated_keys__"] = len(value) - 30
        return out
    if isinstance(value, list):
        return numeric_vector_summary(value)
    return value


def iter_paths(value: Any, prefix: str = "", max_depth: int = 8) -> Iterable[Tuple[str, Any]]:
    if max_depth < 0:
        return
    yield prefix or "$", value
    if isinstance(value, Mapping):
        for k, v in value.items():
            key = str(k)
            child = f"{prefix}.{key}" if prefix else key
            yield from iter_paths(v, child, max_depth - 1)
    elif isinstance(value, list):
        # Do not recursively scan every large item. For list-of-dicts, scan a
        # few representative elements under stable synthetic indices.
        if len(value) <= 6:
            indices = range(len(value))
        else:
            indices = [0, 1, len(value) // 2, len(value) - 2, len(value) - 1]
        seen = set()
        for i in indices:
            if i in seen or i < 0 or i >= len(value):
                continue
            seen.add(i)
            yield from iter_paths(value[i], f"{prefix}[{i}]", max_depth - 1)


def summarize_top_level_episode(ep: Mapping[str, Any]) -> Dict[str, Any]:
    keys = sorted(str(k) for k in ep.keys())
    return {
        "arm_id": ep.get("arm_id"),
        "case": ep.get("case"),
        "steps": ep.get("steps"),
        "success": ep.get("success"),
        "termination": ep.get("termination"),
        "episode_failure": ep.get("episode_failure"),
        "constraint": ep.get("constraint"),
        "physical_constraint_cost": ep.get("physical_constraint_cost"),
        "total_cost": ep.get("total_cost"),
        "h_penalty": ep.get("h_penalty"),
        "horizon_counts": ep.get("horizon_counts"),
        "raw_horizon_counts_before_clamp": ep.get("raw_horizon_counts_before_clamp"),
        "clamped_steps": ep.get("clamped_steps"),
        "solver_failure_fallback_steps": ep.get("solver_failure_fallback_steps"),
        "solver_failure_steps": ep.get("solver_failure_steps"),
        "initial_failed_steps": ep.get("initial_failed_steps"),
        "retries": ep.get("retries"),
        "switches": ep.get("switches"),
        "decision_sum_s": decision_sum(ep),
        "decision_mean_s_per_step": (decision_sum(ep) / int(ep.get("steps", 0)) if int(ep.get("steps", 0) or 0) else None),
        "available_top_level_keys": keys,
    }


def list_summary(path: str, value: list, steps: int) -> Dict[str, Any]:
    item_types = Counter(type(x).__name__ for x in value[:20])
    out: Dict[str, Any] = {"path": path, "len": len(value), "item_types_first20": dict(item_types)}
    scalar_vals: List[float] = []
    scalar_ok = True
    for x in value[: min(len(value), 2000)]:
        y = finite_float(x)
        if y is None:
            scalar_ok = False
            break
        scalar_vals.append(y)
    if scalar_ok and scalar_vals:
        out.update({"numeric_min": min(scalar_vals), "numeric_max": max(scalar_vals), "numeric_unique_first2000": sorted(set(scalar_vals))[:20]})
    if value and isinstance(value[0], Mapping):
        key_counts: Counter[str] = Counter()
        for item in value[: min(len(value), 20)]:
            if isinstance(item, Mapping):
                key_counts.update(str(k) for k in item.keys())
        out["dict_keys_first20"] = sorted(key_counts.keys())[:80]
    if len(value) in (steps, steps + 1):
        out["step_aligned"] = True
    return out


def find_step_series(ep: Mapping[str, Any]) -> List[Dict[str, Any]]:
    steps = int(ep.get("steps", 0) or 0)
    found: List[Dict[str, Any]] = []
    for path, value in iter_paths(ep, max_depth=7):
        if isinstance(value, list) and len(value) in (steps, steps + 1):
            found.append(list_summary(path, value, steps))
    found.sort(key=lambda x: (x.get("len", 0), x.get("path", "")))
    return found[:120]


def flatten_record(record: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in record.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_record(v, key))
        else:
            out[key] = v
    return out


def values_as_horizons(values: Sequence[Any]) -> Optional[List[int]]:
    hs: List[int] = []
    for x in values:
        if isinstance(x, bool):
            return None
        if isinstance(x, str):
            if not re.fullmatch(r"\d+", x.strip()):
                return None
            h = int(x.strip())
        else:
            y = finite_float(x)
            if y is None or abs(y - round(y)) > 1e-9:
                return None
            h = int(round(y))
        if h not in H_GRID:
            return None
        hs.append(h)
    return hs


def find_horizon_candidates(ep: Mapping[str, Any]) -> List[Dict[str, Any]]:
    steps = int(ep.get("steps", 0) or 0)
    candidates: List[Dict[str, Any]] = []
    for path, value in iter_paths(ep, max_depth=7):
        path_l = path.lower()
        if not isinstance(value, list) or len(value) != steps:
            continue
        # Direct scalar horizon list.
        hs = values_as_horizons(value)
        if hs is not None and ("h" in path_l or "horizon" in path_l):
            candidates.append({"path": path, "mode": "scalar_list", "counts": dict(Counter(map(str, hs))), "sequence": hs})
            continue
        # Step-record list. Look for any scalar field that is horizon-like.
        if value and all(isinstance(x, Mapping) for x in value[: min(len(value), 10)]):
            field_values: Dict[str, List[Any]] = defaultdict(list)
            for item in value:
                flat = flatten_record(item) if isinstance(item, Mapping) else {}
                for k, v in flat.items():
                    field_values[k].append(v)
            for k, vals in field_values.items():
                k_l = k.lower()
                if not ("h" == k_l or "horizon" in k_l or "raw_h" in k_l or "request" in k_l or "execut" in k_l):
                    continue
                hs2 = values_as_horizons(vals)
                if hs2 is not None:
                    candidates.append({"path": f"{path}[*].{k}", "mode": "dict_field", "counts": dict(Counter(map(str, hs2))), "sequence": hs2})
    # Prefer executed-ish over raw-ish when sorting, but keep all for audit.
    def score(c: Mapping[str, Any]) -> Tuple[int, str]:
        p = str(c["path"]).lower()
        s = 0
        if "execut" in p or "selected" in p or "horizon" in p:
            s -= 2
        if "raw" in p or "request" in p:
            s -= 1
        return (s, str(c["path"]))
    candidates.sort(key=score)
    return candidates[:20]


def context_from_step_records(ep: Mapping[str, Any], indices: Sequence[int], window: int = 3) -> List[Dict[str, Any]]:
    steps = int(ep.get("steps", 0) or 0)
    contexts: List[Dict[str, Any]] = []
    # Candidate list-of-dict step traces with relevant keys.
    list_paths: List[Tuple[str, List[Any], int]] = []
    for path, value in iter_paths(ep, max_depth=6):
        if isinstance(value, list) and len(value) == steps and value and isinstance(value[0], Mapping):
            keys = set()
            for item in value[: min(len(value), 10)]:
                if isinstance(item, Mapping):
                    keys.update(str(k).lower() for k in flatten_record(item).keys())
            relevance = sum(1 for k in keys for term in TERMS if term in k)
            if relevance:
                list_paths.append((path, value, relevance))
    list_paths.sort(key=lambda x: (-x[2], x[0]))
    for idx in indices:
        lo = max(0, int(idx) - window)
        hi = min(steps - 1, int(idx) + window)
        block: Dict[str, Any] = {"center_index": int(idx), "window": [lo, hi], "records_by_path": {}}
        for path, value, _rel in list_paths[:5]:
            rows = []
            for j in range(lo, hi + 1):
                rows.append({"step_index": j, "record": compact(value[j], max_depth=3)})
            block["records_by_path"][path] = rows
        contexts.append(block)
    return contexts


def key_inventory(ep: Mapping[str, Any]) -> Dict[str, Any]:
    matches: Dict[str, List[str]] = {term: [] for term in TERMS}
    for path, value in iter_paths(ep, max_depth=8):
        leaf = path.lower()
        for term in TERMS:
            if term in leaf and len(matches[term]) < 80:
                if isinstance(value, Mapping):
                    desc = f"{path}:dict[{len(value)}]"
                elif isinstance(value, list):
                    desc = f"{path}:list[{len(value)}]"
                else:
                    desc = f"{path}:{type(value).__name__}={repr(value)[:120]}"
                matches[term].append(desc)
    return {k: v for k, v in matches.items() if v}


def episode_row(ep: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": ep.get("arm_id"),
        "H": parse_h_from_arm(str(ep.get("arm_id", ""))),
        "success": bool(ep.get("success")),
        "termination": ep.get("termination"),
        "steps": int(ep.get("steps", 0) or 0),
        "physical_constraint_cost": float(ep.get("physical_constraint_cost", 0.0)),
        "total_cost": float(ep.get("total_cost", 0.0)),
        "h_penalty": float(ep.get("h_penalty", 0.0)),
        "decision_sum_s": decision_sum(ep),
        "decision_mean_s_per_step": (decision_sum(ep) / int(ep.get("steps", 0)) if int(ep.get("steps", 0) or 0) else None),
        "horizon_counts": ep.get("horizon_counts"),
        "raw_horizon_counts_before_clamp": ep.get("raw_horizon_counts_before_clamp"),
        "clamped_steps": int(ep.get("clamped_steps", 0) or 0),
        "solver_failure_fallback_steps": int(ep.get("solver_failure_fallback_steps", 0) or 0),
        "solver_failure_steps": int(ep.get("solver_failure_steps", 0) or 0),
        "initial_failed_steps": int(ep.get("initial_failed_steps", 0) or 0),
        "retries": int(ep.get("retries", 0) or 0),
        "switches": int(ep.get("switches", 0) or 0),
    }


def append_once(path: Path, token: str, text: str) -> None:
    if not path.exists():
        return
    old = path.read_text(encoding="utf-8")
    if token in old:
        return
    path.write_text(old.rstrip() + "\n\n" + token + "\n" + text.strip() + "\n", encoding="utf-8")


def safe_append_registry(row: Mapping[str, Any]) -> str:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    if not path.exists():
        return "missing"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
    except Exception as exc:
        return f"read_failed:{exc}"
    if not header or len(header) < 3:
        return "unrecognized_header"
    allowed = set(header)
    out = {k: "" for k in header}
    for k, v in row.items():
        if k in allowed:
            out[k] = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
    try:
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writerow(out)
        return "appended"
    except Exception as exc:
        return f"append_failed:{exc}"


def write_summary(raw: Mapping[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Vehicle safe-shortening v1 shard13 case9 seed2 trace diagnostic")
    lines.append("")
    lines.append(f"Created UTC: `{raw['created_utc']}`.")
    lines.append("")
    lines.append("Metadata-only diagnostic over already-created shard13 raw evidence. No rollout, no training, no historical validation64 bank reopen, and no sealed-test access.")
    lines.append("")
    lines.append("## Target comparison")
    lines.append("")
    for label in ("adaptive", "matched_H25"):
        e = raw["target_episodes"][label]
        lines.append(
            f"- {label}: arm=`{e.get('arm_id')}`, success={e.get('success')}, termination={e.get('termination')}, steps={e.get('steps')}, phys={e.get('physical_constraint_cost')}, total={e.get('total_cost')}, horizons={e.get('horizon_counts')}, raw_horizons={e.get('raw_horizon_counts_before_clamp')}, clamped={e.get('clamped_steps')}, solver_fallback={e.get('solver_failure_fallback_steps')}, solver_fail_steps={e.get('solver_failure_steps')}, retries={e.get('retries')}."
        )
    d = raw["target_delta_adaptive_minus_H25"]
    lines.append(
        f"- Adaptive-minus-H25: success_delta={d['success_delta']}, steps_delta={d['steps_delta']}, physical_delta={d['physical_constraint_delta']:.9g}, total_delta={d['total_delta']:.9g}, decision_sum_delta_s={d['decision_sum_delta_s']:.9g}."
    )
    lines.append("")
    lines.append("## Horizon sequence and off-H localization")
    lines.append("")
    lines.append(f"- Adaptive horizon candidates found: `{len(raw['adaptive_horizon_candidates'])}`.")
    if raw["off_h_indices_by_candidate"]:
        for item in raw["off_h_indices_by_candidate"]:
            lines.append(f"  - `{item['path']}` counts={item['counts']} off_H_indices={item['off_h_indices'][:20]} total_off_H={item['total_off_h']}")
    else:
        lines.append("  - No exact step-level horizon sequence was recoverable from the raw episode; only aggregate horizon counts are available.")
    lines.append("")
    lines.append("## Field availability relevant to requested diagnosis")
    lines.append("")
    fa = raw["field_availability"]
    for key in sorted(fa):
        lines.append(f"- {key}: {fa[key]}")
    lines.append("")
    lines.append("## Seed2 case9 comparator grid")
    lines.append("")
    lines.append("Matched-terminal seed2 fixed-H arms on the same case:")
    for row in raw["same_case_seed2_matched_fixed_grid"]:
        lines.append(
            f"- H{row['H']}: success={row['success']}, steps={row['steps']}, phys={row['physical_constraint_cost']:.9g}, total={row['total_cost']:.9g}, decision_mean_s_per_step={row['decision_mean_s_per_step']}, solver_fail_steps={row['solver_failure_steps']}, retries={row['retries']}"
        )
    lines.append("")
    lines.append("Adaptive arms on the same case:")
    for row in raw["same_case_adaptive_rows"]:
        lines.append(
            f"- {row['arm_id']}: success={row['success']}, steps={row['steps']}, phys={row['physical_constraint_cost']:.9g}, total={row['total_cost']:.9g}, horizons={row['horizon_counts']}, raw_horizons={row['raw_horizon_counts_before_clamp']}, solver_fail_steps={row['solver_failure_steps']}, retries={row['retries']}"
        )
    lines.append("")
    lines.append("## Diagnostic conclusion")
    lines.append("")
    for item in raw["diagnostic_conclusions"]:
        lines.append("- " + item)
    lines.append("")
    lines.append("## Next action")
    lines.append("")
    lines.append(raw["next_action"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=13)
    parser.add_argument("--case", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    shard_dir = CAMPAIGN / ("shard%02d" % args.shard)
    raw_path_in = shard_dir / "raw.json"
    completed_path_in = shard_dir / "completed.json"
    if not raw_path_in.exists() or not completed_path_in.exists():
        raise RuntimeError(f"Required shard artifacts missing under {rel(shard_dir)}")

    done = read_json(completed_path_in)
    if done.get("passed") is not True:
        raise RuntimeError(f"Completed marker did not pass: {rel(completed_path_in)}")
    raw_hash = sha256(raw_path_in)
    done_hashes = done.get("hashes") or {}
    expected_raw_hash = done_hashes.get(rel(raw_path_in)) or done_hashes.get(str(raw_path_in))
    if expected_raw_hash and expected_raw_hash != raw_hash:
        raise RuntimeError(f"Raw hash mismatch: expected {expected_raw_hash} got {raw_hash}")

    raw = read_json(raw_path_in)
    flags = raw.get("access_flags") or {}
    if flags.get("sealed_test_bank_opened") or raw.get("sealed_test_bank_opened"):
        raise RuntimeError("Unexpected sealed-test flag in input raw")
    if flags.get("historical_validation64_bank_opened") or raw.get("historical_validation64_bank_opened"):
        raise RuntimeError("Unexpected historical validation64 flag in input raw")

    episodes = raw.get("episodes") or []
    target_arm = f"safe_shortening_v1_vehicle_s{args.seed}"
    fixed_h25_arm = f"matched_terminal_fixed_H25_vehicle_s{args.seed}"
    target = [e for e in episodes if e.get("arm_id") == target_arm and int(e.get("case", -1)) == args.case]
    fixed = [e for e in episodes if e.get("arm_id") == fixed_h25_arm and int(e.get("case", -1)) == args.case]
    if len(target) != 1 or len(fixed) != 1:
        raise RuntimeError(f"Expected exactly one target/fixed episode, got {len(target)} / {len(fixed)}")
    target_ep = target[0]
    fixed_ep = fixed[0]

    target_summary = summarize_top_level_episode(target_ep)
    fixed_summary = summarize_top_level_episode(fixed_ep)

    horizon_candidates = find_horizon_candidates(target_ep)
    off_h: List[Dict[str, Any]] = []
    candidate_sequences: List[List[int]] = []
    for c in horizon_candidates:
        seq = list(c.get("sequence") or [])
        if not seq:
            continue
        candidate_sequences.append(seq)
        idxs = [i for i, h in enumerate(seq) if h != 25]
        off_h.append({"path": c["path"], "counts": c["counts"], "off_h_indices": idxs, "total_off_h": len(idxs)})
    # Do not store whole sequences in raw_out; counts and indices suffice.
    horizon_candidates_slim = [{k: v for k, v in c.items() if k != "sequence"} for c in horizon_candidates]
    representative_indices = sorted(set(i for item in off_h for i in item["off_h_indices"][:10]))

    step_series = find_step_series(target_ep)
    fixed_step_series = find_step_series(fixed_ep)
    contexts = context_from_step_records(target_ep, representative_indices, window=3) if representative_indices else []

    # Field availability categories requested by the user.
    inv = key_inventory(target_ep)
    field_availability: Dict[str, Any] = {}
    for category, needles in {
        "raw_request_or_executed_horizon_fields": ("raw", "request", "execut", "horizon"),
        "guard_clamp_fallback_fields": ("guard", "clamp", "fallback"),
        "solver_status_retry_warmstart_fields": ("solver", "status", "retry", "warm", "fail"),
        "terminal_value_objective_fields": ("terminal", "value", "objective", "score", "candidate"),
        "normalization_state_feature_fields": ("normal", "state", "obs"),
        "control_action_cost_constraint_fields": ("control", "action", "cost", "constraint"),
    }.items():
        paths: List[str] = []
        for needle in needles:
            paths.extend(inv.get(needle, []))
        # Preserve order while removing duplicates.
        seen = set()
        uniq = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        field_availability[category] = {"count": len(uniq), "examples": uniq[:30]}

    same_case_seed2_matched = []
    for e in episodes:
        arm = str(e.get("arm_id", ""))
        if int(e.get("case", -1)) == args.case and arm.startswith("matched_terminal_fixed_H") and arm.endswith(f"vehicle_s{args.seed}"):
            same_case_seed2_matched.append(episode_row(e))
    same_case_seed2_matched.sort(key=lambda r: (r["H"] if r["H"] is not None else 999))

    same_case_adaptive = []
    for e in episodes:
        arm = str(e.get("arm_id", ""))
        if int(e.get("case", -1)) == args.case and arm.startswith("safe_shortening_v1_vehicle_s"):
            same_case_adaptive.append(episode_row(e))
    same_case_adaptive.sort(key=lambda r: r["arm_id"])

    delta = {
        "success_delta": int(bool(target_ep.get("success"))) - int(bool(fixed_ep.get("success"))),
        "steps_delta": int(target_ep.get("steps", 0) or 0) - int(fixed_ep.get("steps", 0) or 0),
        "physical_constraint_delta": float(target_ep.get("physical_constraint_cost", 0.0)) - float(fixed_ep.get("physical_constraint_cost", 0.0)),
        "total_delta": float(target_ep.get("total_cost", 0.0)) - float(fixed_ep.get("total_cost", 0.0)),
        "decision_sum_delta_s": decision_sum(target_ep) - decision_sum(fixed_ep),
    }

    conclusions: List[str] = []
    if target_summary.get("clamped_steps") == 0 and target_summary.get("solver_failure_fallback_steps") == 0 and target_summary.get("solver_failure_steps") == 0 and target_summary.get("retries") == 0:
        conclusions.append("The episode-level fields show no clamp, solver-failure fallback, solver-failure steps, or retries for the adaptive failure; an execution fallback/clamp explanation is not supported by these aggregate raw fields.")
    else:
        conclusions.append("The adaptive failure has nonzero clamp/fallback/solver/retry aggregate fields; inspect step-level status before attributing failure to policy selection alone.")
    if off_h:
        exact_indices = sorted(set(i for item in off_h for i in item["off_h_indices"]))
        conclusions.append(f"A step-level horizon sequence is available and localizes non-H25 decisions at indices {exact_indices}; context records were extracted where available.")
    else:
        conclusions.append("The raw artifact did not expose a recoverable step-level horizon sequence despite aggregate horizon_counts showing H10 once; this prevents exact localization from raw metadata alone.")
    if int(target_ep.get("steps", 0) or 0) > int(fixed_ep.get("steps", 0) or 0) and not target_ep.get("success") and fixed_ep.get("success"):
        conclusions.append("The adaptive case9 failure is a real paired degradation relative to same-seed H25 in this development shard: adaptive exhausted the 150-step cap, while H25 reached the goal earlier.")
    # Compare fixed grid outcomes to see whether short horizons are risky on this case.
    failed_short = [r for r in same_case_seed2_matched if (r["H"] or 999) <= 10 and not r["success"]]
    successful_h25plus = [r for r in same_case_seed2_matched if (r["H"] or 0) >= 25 and r["success"]]
    if failed_short and successful_h25plus:
        conclusions.append("Same-case fixed-grid evidence suggests very short horizons are risky on this scenario while H25+ succeeds, consistent with but not proving that the rare H10 decision could be harmful.")
    if not contexts:
        conclusions.append("Existing raw fields appear insufficient for terminal-value/objective/state-normalization causality; a bounded instrumented deterministic replay is the appropriate next diagnostic after the frozen campaign/backup gate.")
    else:
        conclusions.append("Step-record context is present in the artifact; review raw diagnostic context before deciding whether a replay is still needed.")

    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = DIAG_ROOT / f"vehicle_safe_shortening_v1_shard{args.shard:02d}_case{args.case}_seed{args.seed}_trace_diagnostic_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_raw = out_dir / "raw.json"
    out_summary = out_dir / "summary.md"
    out_completed = out_dir / "completed.json"
    backup_request = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_SHARD{args.shard:02d}_CASE{args.case}_SEED{args.seed}_TRACE_DIAGNOSTIC_{stamp}.json"

    elapsed_s = (now - SERVICE_START).total_seconds()
    raw_out: Dict[str, Any] = {
        "created_utc": now.isoformat(),
        "method": "IMPROVED_vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_metadata_only",
        "script": {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "input": {"shard_raw": rel(raw_path_in), "shard_raw_sha256": raw_hash, "shard_completed": rel(completed_path_in), "shard_completed_sha256": sha256(completed_path_in)},
        "access_control": {"sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollout_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
        "target": {"shard": args.shard, "case": args.case, "seed": args.seed, "adaptive_arm": target_arm, "matched_H25_arm": fixed_h25_arm},
        "target_episodes": {"adaptive": target_summary, "matched_H25": fixed_summary},
        "target_delta_adaptive_minus_H25": delta,
        "adaptive_horizon_candidates": horizon_candidates_slim,
        "off_h_indices_by_candidate": off_h,
        "adaptive_step_aligned_series_inventory": step_series,
        "fixed_H25_step_aligned_series_inventory": fixed_step_series,
        "off_h_context_records": contexts,
        "field_availability": field_availability,
        "same_case_seed2_matched_fixed_grid": same_case_seed2_matched,
        "same_case_adaptive_rows": same_case_adaptive,
        "diagnostic_conclusions": conclusions,
        "next_action": "Do not modify the frozen safe-shortening v1 validation campaign. After backup verification, finish shards14-15. Then run the full 16-shard aggregate and, if safe-shortening v1 is rejected or diagnosed as collapsed, run an instrumented deterministic replay for shard13 case9 seed2 and the earlier case43 lead before designing a versioned retraining/selection revision.",
        "service_elapsed_at_diagnostic": {"since_2026-09-26T10:55:29.419331Z_seconds": elapsed_s, "since_2026-09-26T10:55:29.419331Z_hours": elapsed_s / 3600.0, "research_sqlite_total_tokens": "unknown; research.sqlite path unavailable to repository tools"},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        "artifacts": {"raw_json": rel(out_raw), "summary_md": rel(out_summary), "completed_json": rel(out_completed), "backup_request": rel(backup_request)},
    }

    write_json(out_raw, raw_out)
    write_summary(raw_out, out_summary)
    write_json(backup_request, {
        "requested_utc": now.isoformat(),
        "reason": "backup metadata-only diagnostic for vehicle safe-shortening v1 shard13 case9 seed2 rare H10 failure",
        "artifacts": [rel(out_raw), rel(out_summary), rel(out_completed), rel(Path(__file__).resolve()), rel(raw_path_in), rel(completed_path_in), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "EXPERIMENT_REGISTRY.csv"],
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "backup_gate_note": "Created after latest known verified backup; verify before accumulating more unique validation/formal evidence.",
    })

    token = f"<!-- vehicle-safe-shortening-v1-shard13-case9-seed2-trace-diagnostic-{stamp} -->"
    doc_text = (
        "## 2026-09-27 vehicle safe-shortening v1 shard13 case9 seed2 trace diagnostic\n\n"
        f"UTC: {now.isoformat()}. Ran a metadata-only diagnostic over already-created shard13 raw evidence; no rollout/training, no historical validation64 bank reopen, and no sealed-test access. "
        f"Adaptive seed2 case9 failed with summary horizons {target_summary.get('horizon_counts')} and raw horizons {target_summary.get('raw_horizon_counts_before_clamp')}, clamped={target_summary.get('clamped_steps')}, solver_fallback={target_summary.get('solver_failure_fallback_steps')}, solver_fail_steps={target_summary.get('solver_failure_steps')}, retries={target_summary.get('retries')}; same-seed H25 succeeded. "
        f"Adaptive-minus-H25 deltas: steps {delta['steps_delta']}, physical {delta['physical_constraint_delta']:.9g}, total {delta['total_delta']:.9g}. "
        f"Conclusions: {conclusions}. Artifacts: `{rel(out_summary)}`, `{rel(out_raw)}`, `{rel(out_completed)}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md"):
        append_once(ROOT / name, token, doc_text)
    reg_status = safe_append_registry({
        "timestamp": now.isoformat(),
        "phase": "vehicle_safe_shortening_v1_diagnostic",
        "method": raw_out["method"],
        "split": "fresh_devval_shard13_metadata_only_no_sealed_test",
        "seed": f"case{args.case}_seed{args.seed}",
        "status": "complete",
        "artifacts": raw_out["artifacts"],
        "episodes": 0,
        "control_steps": 0,
        "notes": conclusions,
    })

    completed_payload = {
        "passed": True,
        "method": raw_out["method"],
        "metadata_only": True,
        "formal_final_test_evidence": False,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "registry_append_status": reg_status,
        "backup_request": rel(backup_request),
        "hashes": {rel(p): sha256(p) for p in [Path(__file__).resolve(), raw_path_in, completed_path_in, out_raw, out_summary, backup_request]},
    }
    write_json(out_completed, completed_payload)
    # Rewrite raw to include completed hash path now that completed exists.
    raw_out["artifacts"]["completed_json_sha256"] = sha256(out_completed)
    write_json(out_raw, raw_out)
    # Completed must track final raw hash.
    completed_payload["hashes"][rel(out_raw)] = sha256(out_raw)
    write_json(out_completed, completed_payload)

    print(json.dumps({
        "completed": rel(out_completed),
        "summary": rel(out_summary),
        "raw": rel(out_raw),
        "backup_request": rel(backup_request),
        "registry_append_status": reg_status,
        "target_delta_adaptive_minus_H25": delta,
        "adaptive_horizon_counts": target_summary.get("horizon_counts"),
        "raw_horizon_counts_before_clamp": target_summary.get("raw_horizon_counts_before_clamp"),
        "off_h_indices_by_candidate": off_h,
        "field_availability_counts": {k: v["count"] for k, v in field_availability.items()},
        "same_case_seed2_matched_fixed_grid_brief": [{"H": r["H"], "success": r["success"], "steps": r["steps"], "phys": r["physical_constraint_cost"], "solver_fail_steps": r["solver_failure_steps"]} for r in same_case_seed2_matched],
        "conclusions": conclusions,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
