#!/usr/bin/env python3
"""Development metadata diagnostic: where would v2 H10 dwell alter v1 traces?

This diagnostic reads only already-opened fresh development-validation outputs from
safe-shortening v1.  It performs no rollout, no training, no fresh-bank
creation, no historical validation64-bank reopen, and no sealed-test access.

Question: the v2 transition-hold amendment adds a 3-step minimum H10 dwell after
raw H10 requests.  Before running the frozen v2 devval campaign (currently
backup-gated), quantify the v1 trace situations where this logic would actually
change an executed schedule, and cross-reference affected cases against fixed
H10/H25 comparators already present in the same fresh devval campaign.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = REPO_ROOT / "research_artifacts" / "aws_development_validation" / "vehicle_safe_shortening_v1_devval64_20260927_v1"
DIAG_ROOT = REPO_ROOT / "research_artifacts" / "aws_diagnostics"
BACKUP_DIR = REPO_ROOT / "research_artifacts" / "aws_backup_proofs"

H10_MIN_DWELL = 3
BASE_H = 25
TARGET_SHORT_H = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_label(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "Z").replace("Z", "Z")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group(0))
    return None


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def first_present(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_counts(value: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            hk = as_int(k)
            cv = as_int(v)
            if hk is not None and cv is not None:
                out[str(hk)] = out.get(str(hk), 0) + cv
    return out


def parse_seed(arm_id: str, d: Dict[str, Any]) -> Optional[int]:
    for key in ("training_seed", "learned_seed", "seed", "policy_seed", "terminal_seed"):
        if key in d:
            val = as_int(d[key])
            if val is not None and 0 <= val <= 99:
                return val
    patterns = [r"learned[_-]?s(\d+)", r"adaptive[_-]?seed(\d+)", r"seed(\d+)", r"seed_(\d+)", r"s(\d+)"]
    for pat in patterns:
        m = re.search(pat, arm_id)
        if m:
            return int(m.group(1))
    return None


def parse_controller_h(arm_id: str, d: Dict[str, Any], horizon_counts: Dict[str, int]) -> Optional[int]:
    for key in ("controller_h", "controller_H", "fixed_h", "fixed_H", "horizon", "h"):
        if key in d:
            val = as_int(d[key])
            if val is not None and 1 <= val <= 200:
                return val
    for pat in (r"controllerH(\d+)", r"controller_H(\d+)", r"fixed[_-]?H(\d+)", r"fixed_h(\d+)", r"\bH(\d+)\b"):
        m = re.search(pat, arm_id)
        if m:
            return int(m.group(1))
    if len(horizon_counts) == 1:
        return as_int(next(iter(horizon_counts.keys())))
    return None


def parse_case(d: Dict[str, Any]) -> Optional[int]:
    for key in ("case_id", "case_index", "case", "bank_case", "scenario_id", "scenario_index"):
        if key in d:
            val = as_int(d[key])
            if val is not None:
                return val
    return None


def extract_success(d: Dict[str, Any]) -> Optional[bool]:
    for key in ("success", "succeeded", "reached_goal", "goal_reached", "is_success"):
        if key in d:
            v = d[key]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                if v.lower() in ("true", "yes", "1", "success", "succeeded"):
                    return True
                if v.lower() in ("false", "no", "0", "failure", "failed"):
                    return False
    for key in ("episode_failure", "failed", "failure"):
        if key in d:
            v = d[key]
            if isinstance(v, bool):
                return not v
            if isinstance(v, (int, float)):
                return not bool(v)
    return None


def sequence_from_list(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or not value:
        return None
    seq: List[int] = []
    if all(not isinstance(x, (dict, list)) for x in value):
        for x in value:
            hx = as_int(x)
            if hx is None:
                return None
            seq.append(hx)
        return seq if seq else None
    if all(isinstance(x, dict) for x in value):
        for step in value:
            hv = first_present(step, (
                "executed_horizon", "applied_horizon", "actual_horizon", "controller_horizon",
                "horizon", "H", "h", "executed_H", "selected_horizon", "requested_horizon",
            ))
            if hv is None:
                # Try one nested diagnostic dict per step.
                for nested_key in ("policy", "decision", "diagnostics", "mpc", "info"):
                    nested = step.get(nested_key)
                    if isinstance(nested, dict):
                        hv = first_present(nested, ("executed_horizon", "horizon", "H", "h", "selected_horizon", "requested_horizon"))
                        if hv is not None:
                            break
            hi = as_int(hv)
            if hi is None:
                return None
            seq.append(hi)
        return seq if seq else None
    return None


def extract_horizon_sequence(d: Dict[str, Any]) -> Tuple[Optional[List[int]], str]:
    # Direct sequence fields.
    for key in (
        "executed_horizon_sequence", "horizon_sequence", "horizons", "executed_horizons",
        "actual_horizons", "selected_horizons", "raw_horizons", "requested_horizons",
    ):
        if key in d:
            seq = sequence_from_list(d[key])
            if seq:
                return seq, key
    # Common trace containers.
    for key in ("step_records", "steps", "trace", "trajectory", "rollout", "records", "step_trace"):
        if key in d:
            seq = sequence_from_list(d[key])
            if seq:
                return seq, key
    # One level nested containers.
    for outer_key in ("episode", "result", "summary", "details", "raw", "diagnostics"):
        nested = d.get(outer_key)
        if isinstance(nested, dict):
            seq, src = extract_horizon_sequence(nested)
            if seq:
                return seq, outer_key + "." + src
    return None, "none"


def find_episode_dicts(obj: Any, path: Tuple[Any, ...] = ()) -> Iterable[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    if isinstance(obj, dict):
        arm = first_present(obj, ("arm_id", "arm", "policy_id", "controller_id", "arm_name", "method"))
        case = parse_case(obj)
        counts = normalize_counts(first_present(obj, ("horizon_counts", "executed_horizon_counts", "actual_horizon_counts")))
        seq, _ = extract_horizon_sequence(obj)
        has_outcome = any(k in obj for k in ("success", "succeeded", "reached_goal", "episode_failure", "failed", "control_steps", "step_count", "num_steps", "physical_cost", "total_cost"))
        if isinstance(arm, str) and case is not None and (counts or seq or has_outcome):
            yield path, obj
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                yield from find_episode_dicts(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                yield from find_episode_dicts(v, path + (i,))


def is_adaptive_arm(arm_id: str, d: Dict[str, Any]) -> bool:
    family = str(first_present(d, ("family", "role", "kind", "policy_kind")) or "").lower()
    text = (arm_id + " " + family).lower()
    return ("adaptive" in text or "learned" in text or "safe_shortening" in text) and "fixed" not in text


def is_fixed_arm(arm_id: str, d: Dict[str, Any]) -> bool:
    family = str(first_present(d, ("family", "role", "kind", "policy_kind")) or "").lower()
    text = (arm_id + " " + family).lower()
    return "fixed" in text or "constant" in text or "h_grid" in text or "terminal" in text


def extract_episode(path: Path, shard: int, path_tuple: Tuple[Any, ...], d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    arm = str(first_present(d, ("arm_id", "arm", "policy_id", "controller_id", "arm_name", "method")) or "")
    if not arm:
        return None
    case = parse_case(d)
    if case is None:
        return None
    counts = normalize_counts(first_present(d, ("horizon_counts", "executed_horizon_counts", "actual_horizon_counts")))
    seq, seq_src = extract_horizon_sequence(d)
    if not counts and seq:
        counts = {str(k): v for k, v in Counter(seq).items()}
    seed = parse_seed(arm, d)
    h = parse_controller_h(arm, d, counts)
    steps_value = first_present(d, ("control_steps", "step_count", "num_steps", "episode_steps", "steps_count", "n_steps"))
    steps = as_int(steps_value)
    if steps is None and seq:
        steps = len(seq)
    success = extract_success(d)
    total_cost = as_float(first_present(d, ("total_cost", "complete_episode_cost", "episode_cost", "cost")))
    physical_cost = as_float(first_present(d, ("physical_cost", "physical_control_cost", "control_cost", "physical_constraint_cost")))
    decision_s = as_float(first_present(d, ("total_decision_s", "decision_wall_time_s", "decision_time_s", "solver_wall_time_s")))
    return {
        "source": rel(path),
        "shard": shard,
        "json_path": "/".join(str(x) for x in path_tuple),
        "arm_id": arm,
        "case_id": case,
        "seed": seed,
        "controller_h": h,
        "is_adaptive": is_adaptive_arm(arm, d),
        "is_fixed": is_fixed_arm(arm, d),
        "success": success,
        "steps": steps,
        "total_cost": total_cost,
        "physical_cost": physical_cost,
        "decision_s": decision_s,
        "horizon_counts": counts,
        "horizon_sequence": seq,
        "horizon_sequence_source": seq_src,
    }


def h_segments(seq: Sequence[int], h: int) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    i = 0
    n = len(seq)
    while i < n:
        if seq[i] != h:
            i += 1
            continue
        j = i
        while j + 1 < n and seq[j + 1] == h:
            j += 1
        out.append({"start": i, "end": j, "length": j - i + 1})
        i = j + 1
    return out


def emulate_min_h10_dwell(seq: Sequence[int], dwell: int = H10_MIN_DWELL) -> Dict[str, Any]:
    # Treat v1 executed H10 steps as the only available proxy for raw H10
    # requests.  For each H10 request, force H10 for dwell steps starting at that
    # index, unless the episode ends.
    new_seq = list(seq)
    forced_until = -1
    forced_indices: List[int] = []
    raw_h10_indices = [i for i, h in enumerate(seq) if h == TARGET_SHORT_H]
    for i, h in enumerate(seq):
        if h == TARGET_SHORT_H:
            forced_until = max(forced_until, i + dwell - 1)
        if i <= forced_until and new_seq[i] != TARGET_SHORT_H:
            new_seq[i] = TARGET_SHORT_H
            forced_indices.append(i)
    return {
        "raw_h10_indices": raw_h10_indices,
        "added_h10_indices": forced_indices,
        "added_h10_count": len(forced_indices),
        "original_counts": {str(k): v for k, v in Counter(seq).items()},
        "emulated_counts": {str(k): v for k, v in Counter(new_seq).items()},
        "changed": bool(forced_indices),
    }


def summarize_episode(ep: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if ep is None:
        return None
    return {
        "arm_id": ep["arm_id"],
        "success": ep["success"],
        "steps": ep["steps"],
        "physical_cost": ep["physical_cost"],
        "total_cost": ep["total_cost"],
        "horizon_counts": ep["horizon_counts"],
        "shard": ep["shard"],
    }


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def append_registry(timestamp: str, record_path: str) -> None:
    path = REPO_ROOT / "EXPERIMENT_REGISTRY.csv"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if record_path in existing:
        return
    row = {
        "experiment_id": "",
        "timestamp": timestamp,
        "method": "IMPROVED_vehicle_safe_shortening_v2_transition_hold_scope_diagnostic_metadata_only",
        "seed": "metadata_existing_v1_devval_no_rng",
        "split": "already_opened_fresh_v1_devval_existing_outputs_no_validation64_no_sealed_test",
        "commit_sha": "",
        "status": "completed_development_metadata_diagnostic",
        "exit_status": "",
        "runtime_seconds": "",
        "peak_process_rss_kb": "",
        "record": record_path,
    }
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existing.strip():
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    created = utc_now()
    label = safe_label(created)
    out_dir = DIAG_ROOT / f"vehicle_safe_shortening_v2_transition_hold_scope_diagnostic_{label}"
    out_dir.mkdir(parents=True, exist_ok=False)

    raw_files = sorted(V1_ROOT.glob("shard*/raw.json"))
    episodes: List[Dict[str, Any]] = []
    shard_file_hashes: List[Dict[str, Any]] = []
    for raw_path in raw_files:
        m = re.search(r"shard(\d+)", raw_path.as_posix())
        shard = int(m.group(1)) if m else -1
        shard_file_hashes.append({"path": rel(raw_path), "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size})
        data = load_json(raw_path)
        seen = set()
        for path_tuple, d in find_episode_dicts(data):
            ep = extract_episode(raw_path, shard, path_tuple, d)
            if ep is None:
                continue
            # De-duplicate exact path/arm/case/horizon-count candidates.
            key = (ep["source"], ep["json_path"], ep["arm_id"], ep["case_id"], ep["seed"], tuple(sorted(ep["horizon_counts"].items())))
            if key in seen:
                continue
            seen.add(key)
            episodes.append(ep)

    adaptive_eps = [e for e in episodes if e["is_adaptive"] and e["seed"] in (0, 1, 2)]
    fixed_eps = [e for e in episodes if e["is_fixed"]]
    fixed_index: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for e in fixed_eps:
        if e["seed"] is not None and e["controller_h"] is not None:
            fixed_index[(int(e["seed"]), int(e["case_id"]), int(e["controller_h"]))].append(e)

    seed_summary: Dict[str, Any] = {}
    affected_cases: List[Dict[str, Any]] = []
    sequence_missing_h10 = 0
    for seed in (0, 1, 2):
        eps = [e for e in adaptive_eps if e["seed"] == seed]
        h_counts = Counter()
        h10_eps = 0
        h10_steps = 0
        sequence_available = 0
        changed_eps = 0
        added_steps = 0
        singleton_segments = 0
        short_return_segments = 0
        failures_with_h10 = 0
        for ep in eps:
            h_counts.update({int(k): v for k, v in ep["horizon_counts"].items()})
            if int(ep["horizon_counts"].get(str(TARGET_SHORT_H), 0)) > 0:
                h10_eps += 1
                h10_steps += int(ep["horizon_counts"].get(str(TARGET_SHORT_H), 0))
                if ep["success"] is False:
                    failures_with_h10 += 1
                seq = ep["horizon_sequence"]
                if seq:
                    sequence_available += 1
                    segs = h_segments(seq, TARGET_SHORT_H)
                    singleton_segments += sum(1 for s in segs if s["length"] == 1)
                    short_return_segments += sum(1 for s in segs if s["end"] + 1 < len(seq) and seq[s["end"] + 1] == BASE_H and s["length"] < H10_MIN_DWELL)
                    emu = emulate_min_h10_dwell(seq)
                    if emu["changed"]:
                        changed_eps += 1
                        added_steps += emu["added_h10_count"]
                        fixed_h25 = fixed_index.get((seed, int(ep["case_id"]), BASE_H), [])
                        fixed_h10 = fixed_index.get((seed, int(ep["case_id"]), TARGET_SHORT_H), [])
                        # Prefer same-seed fixed arms with matching seed; if duplicates, use first sorted by arm id.
                        fixed_h25_sorted = sorted(fixed_h25, key=lambda x: x["arm_id"])
                        fixed_h10_sorted = sorted(fixed_h10, key=lambda x: x["arm_id"])
                        affected_cases.append({
                            "seed": seed,
                            "case_id": ep["case_id"],
                            "adaptive_arm_id": ep["arm_id"],
                            "adaptive_success": ep["success"],
                            "adaptive_steps": ep["steps"],
                            "adaptive_physical_cost": ep["physical_cost"],
                            "adaptive_total_cost": ep["total_cost"],
                            "adaptive_horizon_counts": ep["horizon_counts"],
                            "h10_segments": segs,
                            "emulated_v2_added_h10_indices": emu["added_h10_indices"],
                            "emulated_v2_added_h10_count": emu["added_h10_count"],
                            "emulated_v2_horizon_counts": emu["emulated_counts"],
                            "fixed_H25": summarize_episode(fixed_h25_sorted[0] if fixed_h25_sorted else None),
                            "fixed_H10": summarize_episode(fixed_h10_sorted[0] if fixed_h10_sorted else None),
                            "source": ep["source"],
                        })
                else:
                    sequence_missing_h10 += 1
        seed_summary[str(seed)] = {
            "adaptive_episode_count": len(eps),
            "horizon_counts": {str(k): v for k, v in sorted(h_counts.items())},
            "h10_episode_count": h10_eps,
            "h10_step_count": h10_steps,
            "h10_episode_sequence_available_count": sequence_available,
            "h10_episode_sequence_missing_count": h10_eps - sequence_available,
            "episodes_changed_by_v2_dwell_emulation": changed_eps,
            "additional_h10_steps_emulated_by_v2": added_steps,
            "singleton_h10_segments": singleton_segments,
            "short_h10_return_to_H25_segments": short_return_segments,
            "failures_with_h10": failures_with_h10,
        }

    # Compact risk cross-reference for affected cases.
    for row in affected_cases:
        h25 = row.get("fixed_H25") or {}
        h10 = row.get("fixed_H10") or {}
        row["fixed_H25_success"] = h25.get("success")
        row["fixed_H25_steps"] = h25.get("steps")
        row["fixed_H25_total_cost"] = h25.get("total_cost")
        row["fixed_H10_success"] = h10.get("success")
        row["fixed_H10_steps"] = h10.get("steps")
        row["fixed_H10_total_cost"] = h10.get("total_cost")
        row["risk_note"] = (
            "adaptive_failed_with_short_horizon" if row["adaptive_success"] is False else
            "fixed_H10_failed_on_same_case" if h10.get("success") is False else
            "no_failure_flag_in_cross_reference"
        )

    csv_path = out_dir / "affected_cases.csv"
    csv_fields = [
        "seed", "case_id", "adaptive_arm_id", "adaptive_success", "adaptive_steps",
        "adaptive_physical_cost", "adaptive_total_cost", "adaptive_horizon_counts",
        "h10_segments", "emulated_v2_added_h10_indices", "emulated_v2_added_h10_count",
        "emulated_v2_horizon_counts", "fixed_H25_success", "fixed_H25_steps", "fixed_H25_total_cost",
        "fixed_H10_success", "fixed_H10_steps", "fixed_H10_total_cost", "risk_note", "source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in sorted(affected_cases, key=lambda r: (r["seed"], r["case_id"], r["adaptive_arm_id"])):
            writer.writerow({k: json.dumps(row.get(k), sort_keys=True) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in csv_fields})

    # Aggregate fixed-H cross-reference for affected cases.
    affected_by_seed = Counter(str(r["seed"]) for r in affected_cases)
    affected_failures = [r for r in affected_cases if r["adaptive_success"] is False]
    affected_fixed_h10_failures = [r for r in affected_cases if r.get("fixed_H10_success") is False]
    affected_fixed_h25_failures = [r for r in affected_cases if r.get("fixed_H25_success") is False]

    backup_request_path = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_TRANSITION_HOLD_SCOPE_DIAGNOSTIC_{label}.json"
    backup_request = {
        "created_utc": created,
        "reason": "metadata-only v2 transition-hold scope diagnostic over already-opened v1 fresh devval outputs; backup before additional unique evidence accumulates",
        "artifacts": [rel(out_dir), rel(csv_path)],
        "validation64_bank_opened": False,
        "fresh_devval_bank_generated": False,
        "sealed_test_accessed": False,
        "rollout_episodes": 0,
        "control_steps": 0,
        "training_episodes": 0,
        "gradient_steps": 0,
    }
    write_json(backup_request_path, backup_request)

    raw = {
        "created_utc": created,
        "purpose": "Quantify where v2 H10 minimum-dwell logic would alter already-opened v1 safe-shortening devval traces.",
        "method_classification": "IMPROVED development diagnostic, not ORIGINAL SAC and not final-test evidence",
        "input_root": rel(V1_ROOT),
        "input_raw_files": shard_file_hashes,
        "raw_file_count": len(raw_files),
        "episode_candidate_count": len(episodes),
        "adaptive_episode_count": len(adaptive_eps),
        "fixed_episode_count": len(fixed_eps),
        "h10_min_dwell_steps": H10_MIN_DWELL,
        "base_horizon": BASE_H,
        "target_short_horizon": TARGET_SHORT_H,
        "seed_summary": seed_summary,
        "affected_case_count": len(affected_cases),
        "affected_by_seed": dict(affected_by_seed),
        "affected_adaptive_failure_count": len(affected_failures),
        "affected_fixed_h10_failure_count": len(affected_fixed_h10_failures),
        "affected_fixed_h25_failure_count": len(affected_fixed_h25_failures),
        "affected_cases": sorted(affected_cases, key=lambda r: (r["seed"], r["case_id"], r["adaptive_arm_id"])),
        "sequence_missing_h10_episode_count": sequence_missing_h10,
        "access_flags": {
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_bank_opened": False,
            "fresh_devval_bank_generated": False,
            "sealed_test_accessed": False,
            "sealed_test_bank_content_opened": False,
            "sealed_test_bank_hashed": False,
        },
        "backup_request": rel(backup_request_path),
    }
    raw_path = out_dir / "raw.json"
    write_json(raw_path, raw)

    interpretation = []
    total_added = sum(v.get("additional_h10_steps_emulated_by_v2", 0) for v in seed_summary.values())
    total_changed_eps = sum(v.get("episodes_changed_by_v2_dwell_emulation", 0) for v in seed_summary.values())
    if total_changed_eps == 0:
        interpretation.append("No v1 adaptive trace with recoverable step-level H10 sequence would be changed by the v2 dwell emulation; v2 dwell may not be exercised on these traces.")
    else:
        interpretation.append(f"V2 dwell emulation would change {total_changed_eps} adaptive episode(s), adding {total_added} H10 step(s) relative to v1 executed schedules (using v1 executed H10 as the raw-request proxy).")
    if affected_failures:
        interpretation.append(f"Affected set includes {len(affected_failures)} adaptive failure(s); these remain development diagnostics, not proof that v2 will repair them without rollout.")
    if affected_fixed_h10_failures:
        interpretation.append(f"Fixed-H10 failed on {len(affected_fixed_h10_failures)} affected same-seed case reference(s), warning that extended H10 dwell can also be risky depending on state.")
    if affected_fixed_h25_failures:
        interpretation.append(f"Fixed-H25 failed on {len(affected_fixed_h25_failures)} affected same-seed case reference(s), so not all affected cases have a clean H25 rescue comparator.")
    interpretation.append("This metadata diagnostic cannot replace the frozen v2 devval rollout because it does not re-solve MPC under the altered horizon schedule; it only bounds the likely intervention scope and risk cross-reference from existing v1 data.")

    summary_lines = [
        "# Vehicle safe-shortening v2 transition-hold scope diagnostic",
        "",
        f"UTC: `{created}`.",
        "",
        "Metadata-only diagnostic over already-opened v1 fresh development-validation outputs. It created no rollout/control steps, no training, no fresh devval bank, no historical validation64 bank reopen, and no sealed-test access/hash.",
        "",
        "## Question",
        "",
        "Where would the frozen v2 rule (minimum 3-step dwell after H10 requests) actually alter the v1 adaptive traces, and do the affected cases show obvious fixed-H10/H25 risk from existing paired comparators?",
        "",
        "## Aggregate findings",
        "",
        f"- Raw shard files parsed: `{len(raw_files)}`.",
        f"- Episode candidates parsed: `{len(episodes)}`; adaptive candidates: `{len(adaptive_eps)}`; fixed candidates: `{len(fixed_eps)}`.",
        f"- Affected adaptive episodes under v2 dwell emulation: `{total_changed_eps}`.",
        f"- Additional H10 steps emulated: `{total_added}`.",
        f"- Affected cases by seed: `{dict(affected_by_seed)}`.",
        f"- Affected adaptive failures: `{len(affected_failures)}`.",
        f"- Affected same-seed fixed-H10 failures: `{len(affected_fixed_h10_failures)}`.",
        f"- Affected same-seed fixed-H25 failures: `{len(affected_fixed_h25_failures)}`.",
        "",
        "## Seed summaries",
        "",
    ]
    for seed in (0, 1, 2):
        summary_lines.append(f"- seed {seed}: `{seed_summary.get(str(seed), {})}`")
    summary_lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    for item in interpretation:
        summary_lines.append(f"- {item}")
    summary_lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Raw JSON: `{rel(raw_path)}`",
        f"- Affected cases CSV: `{rel(csv_path)}`",
        f"- Backup request: `{rel(backup_request_path)}`",
        "",
        "Next: if/when the post-preflight backup gate is satisfied, run the preregistered v2 devval shard00 rather than further optimizing on these already-opened v1 cases.",
    ])
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    completed = {
        "created_utc": created,
        "passed": bool(raw_files) and len(adaptive_eps) > 0,
        "raw_file_count": len(raw_files),
        "episode_candidate_count": len(episodes),
        "adaptive_episode_count": len(adaptive_eps),
        "affected_case_count": len(affected_cases),
        "affected_by_seed": dict(affected_by_seed),
        "total_changed_episodes": total_changed_eps,
        "total_emulated_added_h10_steps": total_added,
        "access_flags": raw["access_flags"],
        "raw_path": rel(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "summary_path": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "affected_cases_csv": rel(csv_path),
        "affected_cases_csv_sha256": sha256_file(csv_path),
        "backup_request": rel(backup_request_path),
        "backup_request_sha256": sha256_file(backup_request_path),
    }
    completed_path = out_dir / "completed.json"
    write_json(completed_path, completed)
    completed["completed_path"] = rel(completed_path)
    completed["completed_sha256"] = sha256_file(completed_path)
    write_json(completed_path, completed)
    completed["completed_sha256"] = sha256_file(completed_path)

    marker = f"<!-- vehicle-safe-shortening-v2-transition-hold-scope-diagnostic-{label} -->"
    doc = f"""
{marker}
## 2026-09-28 vehicle safe-shortening v2 transition-hold scope diagnostic

UTC: {created}. Metadata-only diagnostic over already-opened v1 fresh devval outputs; no rollout/control steps, no training, no fresh devval bank generation, no historical validation64 bank reopen, and no sealed-test access/hash. Parsed {len(raw_files)} raw shard files and {len(adaptive_eps)} adaptive episode candidates. Emulating the frozen v2 3-step H10 dwell on recoverable v1 H10 traces would change {total_changed_eps} adaptive episode(s), adding {total_added} H10 step(s), with affected cases by seed {dict(affected_by_seed)}. Affected adaptive failures={len(affected_failures)}, affected same-seed fixed-H10 failures={len(affected_fixed_h10_failures)}, affected same-seed fixed-H25 failures={len(affected_fixed_h25_failures)}. This bounds intervention scope/risk from existing development data only and does not replace v2 rollout. Artifacts: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(csv_path)}`, `{rel(completed_path)}`. Backup request: `{rel(backup_request_path)}`. Next remains: after verified post-preflight backup, run frozen v2 devval shard00 with legacy interpreter; do not tune on sealed test.
""".strip()
    for doc_name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md"):
        append_once(REPO_ROOT / doc_name, marker, doc)
    decision_marker = f"<!-- decision-v2-scope-diagnostic-{label} -->"
    decision_doc = f"""
{decision_marker}
## 2026-09-28 decision: use v2 scope diagnostic only as development context

Evidence from `{rel(summary_path)}`: the metadata-only emulation of 3-step H10 dwell on already-opened v1 traces changed {total_changed_eps} adaptive episode(s) and added {total_added} H10 step(s). Because no MPC was re-solved under the altered schedule, this diagnostic cannot establish v2 control performance or timing. Decision: keep the frozen v2 protocol unchanged; once the backup gate is satisfied, proceed with preregistered fresh v2 devval shard00 rather than altering thresholds based on this already-opened v1 diagnostic.
""".strip()
    append_once(REPO_ROOT / "DECISIONS.md", decision_marker, decision_doc)
    append_registry(created, rel(completed_path))

    print(json.dumps({
        "created_utc": created,
        "completed_path": rel(completed_path),
        "passed": completed["passed"],
        "raw_file_count": len(raw_files),
        "adaptive_episode_count": len(adaptive_eps),
        "total_changed_episodes": total_changed_eps,
        "total_emulated_added_h10_steps": total_added,
        "affected_by_seed": dict(affected_by_seed),
        "affected_adaptive_failures": len(affected_failures),
        "affected_fixed_h10_failures": len(affected_fixed_h10_failures),
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "sealed_test_accessed": False,
        "backup_request": rel(backup_request_path),
    }, indent=2, sort_keys=True))
    return 0 if completed["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
