#!/usr/bin/env python3
"""Metadata-only inventory of latency-tree pendulum training artifacts.

Safeguards:
- Restricts traversal to results/latency_tree_2026-09-26/train/pendulum_s*.
- Does not open validation or test directories.
- Runs no simulations.
- Parses only training metadata/marker JSON files; raw trace contents are counted
  but not read or summarized beyond filename/size/mtime metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO / "research_artifacts" / "bohn2021_reproduction_2026-09-17" / "results" / "latency_tree_2026-09-26" / "train"
OUT_DIR = REPO / "research_artifacts" / "aws_diagnostics" / "pendulum_inventory_metadata"
RAW = OUT_DIR / "raw.json"
SUMMARY = OUT_DIR / "summary.md"
EXPECTED = ["pendulum_s0", "pendulum_s1", "pendulum_s2"]

# Files whose content may contain trajectory step series; keep metadata-only for these.
RAW_TRACE_PATTERNS = ("trace_", "solver_calls.jsonl")


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def pid_alive(pid: Any) -> Optional[bool]:
    try:
        ipid = int(pid)
    except Exception:  # noqa: BLE001
        return None
    if ipid <= 0:
        return None
    try:
        os.kill(ipid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return None


def is_raw_trace(path: Path) -> bool:
    name = path.name
    return name.endswith(".jsonl") or any(pat in name for pat in RAW_TRACE_PATTERNS)


def file_meta(path: Path, rel_root: Path) -> Dict[str, Any]:
    st = path.stat()
    meta: Dict[str, Any] = {
        "path": str(path.relative_to(rel_root)),
        "bytes": st.st_size,
        "mtime_utc": iso_mtime(path),
    }
    if is_raw_trace(path):
        meta["sha256"] = None
        meta["sha256_skipped_reason"] = "raw_trace_or_jsonl_metadata_only"
    else:
        meta["sha256"] = sha256_file(path)
    return meta


def collect_files(job_dir: Path) -> Dict[str, Any]:
    all_files: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "files": 0,
        "dirs": 0,
        "bytes": 0,
        "json_files": 0,
        "jsonl_files": 0,
        "trace_json_files": 0,
        "trace_jsonl_files": 0,
        "completed_json_files": 0,
        "policy_json_files": 0,
        "lock_files": 0,
        "tmp_files": 0,
    }
    for root, dirs, files in os.walk(job_dir):
        root_path = Path(root)
        counts["dirs"] += len(dirs)
        for name in sorted(files):
            p = root_path / name
            st = p.stat()
            counts["files"] += 1
            counts["bytes"] += st.st_size
            if name.endswith(".json"):
                counts["json_files"] += 1
            if name.endswith(".jsonl"):
                counts["jsonl_files"] += 1
            if "trace_" in name and name.endswith(".json"):
                counts["trace_json_files"] += 1
            if "trace_" in name and name.endswith(".jsonl"):
                counts["trace_jsonl_files"] += 1
            if name == "completed.json":
                counts["completed_json_files"] += 1
            if name == "policy.json":
                counts["policy_json_files"] += 1
            if name == "run.lock":
                counts["lock_files"] += 1
            if name.endswith(".tmp"):
                counts["tmp_files"] += 1
            all_files.append(file_meta(p, job_dir))
    all_files.sort(key=lambda m: m["path"])
    return {"counts": counts, "files": all_files}


def summarize_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"valid_json_object": False}
    out: Dict[str, Any] = {
        "valid_json_object": True,
        "kind": policy.get("kind"),
        "task": policy.get("task"),
    }
    if policy.get("kind") == "constant":
        out["h"] = policy.get("h")
    if policy.get("kind") == "tree":
        leaves = policy.get("leaves") or []
        out["leaves"] = leaves
        out["unique_leaves"] = sorted(set(leaves)) if isinstance(leaves, list) else []
        out["node_count"] = len(policy.get("nodes") or [])
    return out


def summarize_completed(path: Path) -> Dict[str, Any]:
    data, err = safe_json(path)
    out: Dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path.relative_to(TRAIN_ROOT)),
        "sha256": sha256_file(path) if path.exists() else None,
        "json_error": err,
    }
    if isinstance(data, dict):
        out.update(
            {
                "passed": data.get("passed"),
                "selected": data.get("selected"),
                "training_only": data.get("training_only"),
                "hash_entries": len(data.get("hashes") or {}) if isinstance(data.get("hashes"), dict) else None,
                "mismatches": data.get("mismatches"),
            }
        )
        # Condition-level completed files usually store summary metrics. Keep a small key inventory only.
        for key in ["summary", "episodes", "success", "constraint_violations", "solver_failures", "initial_solver_failures", "final_solver_failures"]:
            if key in data:
                out[key] = data.get(key)
    return out


def summarize_attempt_files(job_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(job_dir.rglob("attempt_*.json")):
        data, err = safe_json(p)
        row: Dict[str, Any] = {
            "path": str(p.relative_to(job_dir)),
            "sha256": sha256_file(p),
            "json_error": err,
        }
        if isinstance(data, dict):
            row.update({"pid": data.get("pid"), "pid_alive": pid_alive(data.get("pid")), "keys": sorted(data.keys())})
        rows.append(row)
    return rows


def generation_summary(job_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gen_dir in sorted(job_dir.glob("generation*")):
        if not gen_dir.is_dir():
            continue
        arm_dirs = [p for p in sorted(gen_dir.iterdir()) if p.is_dir()]
        completed = [p for p in arm_dirs if (p / "completed.json").exists()]
        missing = [p.name for p in arm_dirs if not (p / "completed.json").exists()]
        selection_json = gen_dir / "selection.json"
        registration_json = gen_dir / "registration.json"
        selection_data, selection_err = safe_json(selection_json) if selection_json.exists() else (None, None)
        selected_ids: List[Any] = []
        if isinstance(selection_data, dict):
            # Historical files have varied key names; preserve a key inventory and selected-like values only.
            for key in ["selected", "finalists", "elite", "elites", "best"]:
                value = selection_data.get(key)
                if value is not None:
                    if isinstance(value, list):
                        selected_ids.extend([v.get("id") if isinstance(v, dict) else v for v in value[:8]])
                    elif isinstance(value, dict):
                        selected_ids.append(value.get("id", value))
                    else:
                        selected_ids.append(value)
        rows.append(
            {
                "generation": gen_dir.name,
                "arm_dirs": [p.name for p in arm_dirs],
                "arm_dir_count": len(arm_dirs),
                "completed_count": len(completed),
                "missing_completed": missing,
                "has_selection_json": selection_json.exists(),
                "selection_json_sha256": sha256_file(selection_json) if selection_json.exists() else None,
                "selection_json_error": selection_err,
                "selection_keys": sorted(selection_data.keys()) if isinstance(selection_data, dict) else None,
                "selected_like_values": selected_ids,
                "has_registration_json": registration_json.exists(),
                "registration_json_sha256": sha256_file(registration_json) if registration_json.exists() else None,
            }
        )
    return rows


def threshold_reference_summary(job_dir: Path) -> Dict[str, Any]:
    tr = job_dir / "threshold_reference"
    out: Dict[str, Any] = {"exists": tr.exists()}
    if not tr.exists():
        return out
    progress_path = tr / "progress.json"
    solver_path = tr / "solver_attempts.json"
    completed_path = tr / "completed.json"
    policy_path = tr / "policy.json"
    for label, path in [("progress", progress_path), ("solver_attempts", solver_path), ("completed", completed_path), ("policy", policy_path)]:
        if path.exists():
            data, err = safe_json(path)
            out[label] = {
                "path": str(path.relative_to(job_dir)),
                "sha256": sha256_file(path),
                "json_error": err,
                "data": data if label in {"progress", "solver_attempts", "policy"} else None,
            }
            if label == "completed" and isinstance(data, dict):
                out[label].update(
                    {
                        "passed": data.get("passed"),
                        "episodes": data.get("episodes"),
                        "summary_keys": sorted((data.get("summary") or {}).keys()) if isinstance(data.get("summary"), dict) else None,
                    }
                )
        else:
            out[label] = {"path": str(path.relative_to(job_dir)), "exists": False}
    trace_json = sorted(tr.glob("*trace_*.json"))
    trace_jsonl = sorted(tr.glob("*trace_*.jsonl"))
    resets = sorted(tr.glob("*reset_*.json"))
    tmps = sorted(tr.glob("*.tmp"))
    out["raw_trace_counts"] = {
        "trace_json": len(trace_json),
        "trace_jsonl": len(trace_jsonl),
        "reset_json": len(resets),
        "tmp_files": len(tmps),
        "solver_calls_jsonl_exists": (tr / "solver_calls.jsonl").exists(),
        "solver_calls_jsonl_bytes": (tr / "solver_calls.jsonl").stat().st_size if (tr / "solver_calls.jsonl").exists() else None,
    }
    out["partial_trace_indicators"] = {
        "jsonl_without_json": sorted([p.name for p in trace_jsonl if not (tr / p.name.replace(".jsonl", ".json")).exists()]),
        "tmp_files": [p.name for p in tmps],
        "has_run_lock": (tr / "run.lock").exists(),
    }
    return out


def selection_summary(job_dir: Path) -> Dict[str, Any]:
    sel = job_dir / "selection"
    out: Dict[str, Any] = {"exists": sel.exists()}
    reg = job_dir / "selection_registration.json"
    if reg.exists():
        data, err = safe_json(reg)
        out["registration"] = {
            "exists": True,
            "sha256": sha256_file(reg),
            "json_error": err,
            "keys": sorted(data.keys()) if isinstance(data, dict) else None,
        }
        if isinstance(data, dict):
            arms = data.get("arms") or []
            finalists = data.get("finalists") or []
            order = data.get("order") or []
            out["registration"].update(
                {
                    "arm_ids": [a.get("id") for a in arms if isinstance(a, dict)],
                    "finalist_ids": [a.get("id") for a in finalists if isinstance(a, dict)],
                    "order_count": len(order) if isinstance(order, list) else None,
                    "repeat_arm_order": order,
                }
            )
    else:
        out["registration"] = {"exists": False}
    if sel.exists():
        arm_dirs = [p for p in sorted(sel.iterdir()) if p.is_dir()]
        complete_dirs = [p for p in arm_dirs if (p / "completed.json").exists()]
        out["arm_dirs"] = [p.name for p in arm_dirs]
        out["arm_dir_count"] = len(arm_dirs)
        out["completed_count"] = len(complete_dirs)
        out["missing_completed"] = [p.name for p in arm_dirs if not (p / "completed.json").exists()]
        out["completed_markers"] = [summarize_completed(p / "completed.json") for p in complete_dirs]
    return out


def lock_and_partial_summary(job_dir: Path) -> Dict[str, Any]:
    locks = sorted(job_dir.rglob("run.lock"))
    tmps = sorted(job_dir.rglob("*.tmp"))
    progress_files = sorted(job_dir.rglob("progress.json"))
    progress_rows = []
    for p in progress_files:
        data, err = safe_json(p)
        progress_rows.append({"path": str(p.relative_to(job_dir)), "sha256": sha256_file(p), "json_error": err, "data": data})
    return {
        "run_locks": [{"path": str(p.relative_to(job_dir)), "bytes": p.stat().st_size, "mtime_utc": iso_mtime(p)} for p in locks],
        "tmp_files": [{"path": str(p.relative_to(job_dir)), "bytes": p.stat().st_size, "mtime_utc": iso_mtime(p)} for p in tmps],
        "progress_files": progress_rows,
        "attempt_files": summarize_attempt_files(job_dir),
    }


def inventory_job(name: str) -> Dict[str, Any]:
    job_dir = TRAIN_ROOT / name
    out: Dict[str, Any] = {"job": name, "path": str(job_dir.relative_to(REPO)), "exists": job_dir.exists()}
    if not job_dir.exists():
        out["status"] = "absent_unstarted_at_train_root"
        return out
    if not str(job_dir.resolve()).startswith(str(TRAIN_ROOT.resolve())):
        raise RuntimeError(f"Refusing to inspect outside train root: {job_dir}")
    if "validation" in job_dir.parts or "test" in job_dir.parts:
        raise RuntimeError(f"Refusing to inspect validation/test path: {job_dir}")

    top_entries = sorted(job_dir.iterdir(), key=lambda p: p.name)
    out["top_level"] = [{"name": p.name, "kind": "dir" if p.is_dir() else "file", "bytes": None if p.is_dir() else p.stat().st_size} for p in top_entries]

    started = job_dir / "started.json"
    completed = job_dir / "completed.json"
    policy = job_dir / "policy.json"
    fit = job_dir / "fit.json"
    thresholds = job_dir / "thresholds.json"
    out["started"] = None
    if started.exists():
        data, err = safe_json(started)
        out["started"] = {"sha256": sha256_file(started), "json_error": err, "data": data}
        if isinstance(data, dict):
            out["started"]["pid_alive"] = pid_alive(data.get("pid"))
    out["completed"] = summarize_completed(completed) if completed.exists() else {"exists": False}
    out["fit"] = {"exists": fit.exists(), "sha256": sha256_file(fit) if fit.exists() else None}
    out["thresholds"] = {"exists": thresholds.exists(), "sha256": sha256_file(thresholds) if thresholds.exists() else None}
    if policy.exists():
        pdata, perr = safe_json(policy)
        out["policy"] = {"exists": True, "sha256": sha256_file(policy), "json_error": perr, "summary": summarize_policy(pdata)}
    else:
        out["policy"] = {"exists": False}

    out["threshold_reference"] = threshold_reference_summary(job_dir)
    out["generations"] = generation_summary(job_dir)
    out["selection"] = selection_summary(job_dir)
    out["locks_and_partials"] = lock_and_partial_summary(job_dir)
    out["file_inventory"] = collect_files(job_dir)

    root_completed = bool(completed.exists())
    generation_complete = len(out["generations"]) == 4 and all(g["completed_count"] == 13 for g in out["generations"])
    selection_complete = out["selection"].get("completed_count") == out["selection"].get("arm_dir_count") and out["selection"].get("arm_dir_count") is not None
    threshold_complete = bool(out["threshold_reference"].get("completed", {}).get("passed")) if isinstance(out["threshold_reference"].get("completed"), dict) else False
    has_partial_indicators = bool(out["locks_and_partials"]["tmp_files"]) or any(
        row.get("pid_alive") is False for row in out["locks_and_partials"].get("attempt_files", [])
    )
    if root_completed:
        status = "completed_training_artifact"
    elif started.exists():
        status = "started_partial_or_interrupted"
    else:
        status = "directory_exists_without_started_marker"
    out["derived_status"] = {
        "status": status,
        "root_completed": root_completed,
        "threshold_complete": threshold_complete,
        "generation_complete": generation_complete,
        "selection_complete": selection_complete,
        "has_partial_or_stale_indicators": has_partial_indicators,
    }
    return out


def write_summary(raw: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Pendulum latency-tree training inventory (metadata only)")
    lines.append("")
    lines.append(f"Created UTC: {raw['created_utc']}")
    lines.append("")
    lines.append("Scope: `latency_tree_2026-09-26/train/pendulum_s*` only. No simulations, no validation reads, and no sealed test reads were performed. Raw trace and JSONL contents were not read; they were counted by filename/size metadata.")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| job | exists | derived status | root completed | selected/policy | top-level markers | generation completed | selection completed | partial/stale indicators |")
    lines.append("|---|---:|---|---:|---|---|---|---|---|")
    for job in raw["jobs"]:
        if not job.get("exists"):
            lines.append(f"| `{job['job']}` | false | {job.get('status')} | false | NA | absent | NA | NA | absent/unstarted |")
            continue
        ds = job.get("derived_status", {})
        policy = job.get("policy", {}).get("summary", {})
        selected = job.get("completed", {}).get("selected")
        if selected:
            policy_s = f"selected `{selected}`; {policy.get('kind')} {policy.get('unique_leaves') or policy.get('h') or ''}"
        elif policy.get("kind"):
            policy_s = f"{policy.get('kind')} {policy.get('unique_leaves') or policy.get('h') or ''}"
        else:
            policy_s = "none"
        markers = []
        for key in ["started", "completed", "fit", "thresholds", "policy"]:
            val = job.get(key)
            if isinstance(val, dict) and val.get("exists", True) is not False and val is not None:
                markers.append(key)
            elif key == "started" and val:
                markers.append(key)
        gen_s = ", ".join([f"{g['generation']} {g['completed_count']}/{g['arm_dir_count']}" for g in job.get("generations", [])]) or "none"
        sel_s = "none"
        if job.get("selection", {}).get("exists"):
            sel_s = f"{job['selection'].get('completed_count')}/{job['selection'].get('arm_dir_count')} dirs; finalists {job['selection'].get('registration', {}).get('finalist_ids')}"
        partial = []
        if job.get("started", {}).get("pid_alive") is False:
            partial.append("started PID dead")
        for a in job.get("locks_and_partials", {}).get("attempt_files", []):
            if a.get("pid_alive") is False:
                partial.append(f"dead {a.get('path')} pid {a.get('pid')}")
        if job.get("locks_and_partials", {}).get("tmp_files"):
            partial.append(f"tmp={len(job['locks_and_partials']['tmp_files'])}")
        ptr = job.get("threshold_reference", {}).get("partial_trace_indicators", {})
        if ptr.get("jsonl_without_json"):
            partial.append("jsonl_without_json=" + ",".join(ptr["jsonl_without_json"]))
        if not partial:
            partial.append("none detected")
        lines.append(
            f"| `{job['job']}` | true | {ds.get('status')} | {ds.get('root_completed')} | {policy_s} | {', '.join(markers) or 'none'} | {gen_s} | {sel_s} | {'; '.join(partial)} |"
        )
    lines.append("")
    lines.append("## Detailed notes")
    for job in raw["jobs"]:
        lines.append("")
        lines.append(f"### {job['job']}")
        if not job.get("exists"):
            lines.append("- Directory absent at train root; treated as unstarted unless future evidence shows otherwise.")
            continue
        counts = job.get("file_inventory", {}).get("counts", {})
        lines.append(f"- Files/dirs/bytes under job directory: {counts.get('files')} files, {counts.get('dirs')} dirs, {counts.get('bytes')} bytes.")
        lines.append(f"- Completed markers: {counts.get('completed_json_files')}; policy.json files: {counts.get('policy_json_files')}; run.lock files: {counts.get('lock_files')}; tmp files: {counts.get('tmp_files')}.")
        tr = job.get("threshold_reference", {})
        lines.append(f"- Threshold reference exists={tr.get('exists')}; raw trace counts={tr.get('raw_trace_counts')}.")
        if job.get("selection", {}).get("exists"):
            reg = job["selection"].get("registration", {})
            lines.append(f"- Selection registration arms={reg.get('arm_ids')}; finalists={reg.get('finalist_ids')}; order_count={reg.get('order_count')}.")
        if job.get("completed", {}).get("exists"):
            lines.append(f"- Root completed sha256={job['completed'].get('sha256')}; selected={job['completed'].get('selected')}; hash_entries={job['completed'].get('hash_entries')}; passed={job['completed'].get('passed')}.")
        else:
            lines.append("- Root completed marker absent.")
        if job.get("locks_and_partials", {}).get("progress_files"):
            progress_s = "; ".join([f"{p['path']}={p.get('data')}" for p in job["locks_and_partials"]["progress_files"]])
            lines.append(f"- Progress files: {progress_s}.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("This inventory is not a validation result, not a sealed-test result, and not evidence of reproduced control performance. It is used only to decide safe recovery boundaries.")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    # Explicit path guard to keep sealed splits out of scope.
    resolved = TRAIN_ROOT.resolve()
    if "validation" in resolved.parts or "test" in resolved.parts:
        raise RuntimeError(f"Refusing unsafe root: {resolved}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(REPO)),
        "train_root": str(TRAIN_ROOT.relative_to(REPO)),
        "simulations_run": 0,
        "validation_accessed": False,
        "test_accessed": False,
        "raw_trace_contents_read": False,
        "jobs": [inventory_job(name) for name in EXPECTED],
    }
    RAW.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(raw)
    compact = {
        "created_utc": raw["created_utc"],
        "outputs": [str(RAW.relative_to(REPO)), str(SUMMARY.relative_to(REPO))],
        "validation_accessed": False,
        "test_accessed": False,
        "jobs": [
            {
                "job": j["job"],
                "exists": j["exists"],
                "status": j.get("derived_status", {}).get("status", j.get("status")),
                "root_completed": j.get("derived_status", {}).get("root_completed", False),
                "selected": j.get("completed", {}).get("selected") if j.get("exists") else None,
                "files": j.get("file_inventory", {}).get("counts", {}).get("files") if j.get("exists") else 0,
                "completed_markers": j.get("file_inventory", {}).get("counts", {}).get("completed_json_files") if j.get("exists") else 0,
            }
            for j in raw["jobs"]
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
