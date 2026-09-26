#!/usr/bin/env python3
"""Post-pendulum-inventory documentation and recovery preflight.

Metadata-only diagnostic.  It consumes the already-produced pendulum inventory,
inspects training-root metadata for completed/partial candidate availability, and
writes the required post-inventory interpretation blocks.  It performs no
simulation and deliberately does not read validation or sealed-test directories
or outcome files.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO / "research_artifacts" / "bohn2021_reproduction_2026-09-17" / "results" / "latency_tree_2026-09-26" / "train"
INV_RAW = REPO / "research_artifacts" / "aws_diagnostics" / "pendulum_inventory_metadata" / "raw.json"
INV_SUMMARY = REPO / "research_artifacts" / "aws_diagnostics" / "pendulum_inventory_metadata" / "summary.md"
OUT_DIR = REPO / "research_artifacts" / "aws_diagnostics" / "post_inventory_recovery_preflight"
OUT_RAW = OUT_DIR / "raw.json"
OUT_SUMMARY = OUT_DIR / "summary.md"
MARKER = "<!-- pendulum-inventory-interpretation-20260926 -->"
TASKS = ("vehicle", "pendulum")
SEEDS = (0, 1, 2)
BASE_H = {"vehicle": 25, "pendulum": 30}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_json(path: Path) -> tuple[Optional[Any], Optional[str]]:
    try:
        return load_json(path), None
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


def summarize_policy(policy: Any, task: str) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"valid": False, "type": type(policy).__name__}
    out: Dict[str, Any] = {"valid": True, "kind": policy.get("kind"), "task": policy.get("task")}
    if policy.get("kind") == "constant":
        h = policy.get("h")
        out.update({"h": h, "structurally_fixed": True, "structurally_switching": False})
    elif policy.get("kind") == "tree":
        leaves = policy.get("leaves") if isinstance(policy.get("leaves"), list) else []
        uniq = sorted(set(leaves))
        out.update(
            {
                "leaves": leaves,
                "unique_leaves": uniq,
                "node_count": len(policy.get("nodes") or []),
                "structurally_fixed": len(uniq) == 1,
                "structurally_switching": len(uniq) > 1,
                "fixed_equiv_base_h_if_all_leaves_base": bool(uniq == [BASE_H[task]]),
            }
        )
    else:
        out.update({"structurally_fixed": None, "structurally_switching": None})
    out["note"] = "Structural policy summary only; actual used horizons require paired rollout audit."
    return out


def count_metadata_files(job_dir: Path) -> Dict[str, int]:
    counts = {"files": 0, "dirs": 0, "completed_json": 0, "policy_json": 0, "run_locks": 0, "tmp_files": 0}
    if not job_dir.exists():
        return counts
    for root, dirs, files in os.walk(job_dir):
        counts["dirs"] += len(dirs)
        for name in files:
            counts["files"] += 1
            if name == "completed.json":
                counts["completed_json"] += 1
            if name == "policy.json":
                counts["policy_json"] += 1
            if name == "run.lock":
                counts["run_locks"] += 1
            if name.endswith(".tmp"):
                counts["tmp_files"] += 1
    return counts


def job_metadata(task: str, seed: int) -> Dict[str, Any]:
    name = f"{task}_s{seed}"
    job_dir = TRAIN_ROOT / name
    out: Dict[str, Any] = {"job": name, "task": task, "seed": seed, "path": rel(job_dir), "exists": job_dir.exists()}
    if not job_dir.exists():
        out["derived_status"] = "absent_unstarted_at_train_root"
        return out
    completed = job_dir / "completed.json"
    policy = job_dir / "policy.json"
    started = job_dir / "started.json"
    out["counts"] = count_metadata_files(job_dir)
    out["started"] = {"exists": started.exists()}
    if started.exists():
        data, err = safe_json(started)
        out["started"].update({"sha256": sha256_file(started), "json_error": err})
        if isinstance(data, dict):
            out["started"].update({"pid": data.get("pid"), "pid_alive": pid_alive(data.get("pid")), "started": data.get("started")})
    out["completed"] = {"exists": completed.exists()}
    if completed.exists():
        data, err = safe_json(completed)
        out["completed"].update({"sha256": sha256_file(completed), "json_error": err})
        if isinstance(data, dict):
            out["completed"].update(
                {
                    "passed": data.get("passed"),
                    "training_only": data.get("training_only"),
                    "selected": data.get("selected"),
                    "hash_entries": len(data.get("hashes") or {}) if isinstance(data.get("hashes"), dict) else None,
                }
            )
    out["policy"] = {"exists": policy.exists()}
    if policy.exists():
        data, err = safe_json(policy)
        out["policy"].update({"sha256": sha256_file(policy), "json_error": err, "summary": summarize_policy(data, task)})
    if completed.exists():
        out["derived_status"] = "completed_training_artifact"
    elif started.exists():
        out["derived_status"] = "started_partial_or_interrupted"
    else:
        out["derived_status"] = "directory_exists_without_started_marker"
    return out


def classify_inventory(inv: Dict[str, Any]) -> Dict[str, Any]:
    jobs = {j.get("job"): j for j in inv.get("jobs", []) if isinstance(j, dict)}
    p0 = jobs.get("pendulum_s0", {})
    p1 = jobs.get("pendulum_s1", {})
    p2 = jobs.get("pendulum_s2", {})
    return {
        "pendulum_s0_completed": bool(p0.get("derived_status", {}).get("root_completed")),
        "pendulum_s0_selected": p0.get("completed", {}).get("selected"),
        "pendulum_s0_completed_markers": p0.get("file_inventory", {}).get("counts", {}).get("completed_json_files"),
        "pendulum_s1_status": p1.get("derived_status", {}).get("status") if p1.get("exists") else p1.get("status"),
        "pendulum_s1_threshold_progress": p1.get("threshold_reference", {}).get("progress", {}).get("data"),
        "pendulum_s1_tmp_files": len(p1.get("locks_and_partials", {}).get("tmp_files") or []),
        "pendulum_s1_dead_attempt_pids": sorted(
            {
                a.get("pid")
                for a in p1.get("locks_and_partials", {}).get("attempt_files", [])
                if a.get("pid_alive") is False
            }
        ),
        "pendulum_s2_status": p2.get("status") if not p2.get("exists") else p2.get("derived_status", {}).get("status"),
        "inventory_validation_accessed": inv.get("validation_accessed"),
        "inventory_test_accessed": inv.get("test_accessed"),
        "inventory_raw_trace_contents_read": inv.get("raw_trace_contents_read"),
    }


def read_status_metadata() -> Dict[str, Any]:
    status_path = TRAIN_ROOT.parent / "status.json"
    out: Dict[str, Any] = {"path": rel(status_path), "exists": status_path.exists()}
    if not status_path.exists():
        return out
    data, err = safe_json(status_path)
    out.update({"sha256": sha256_file(status_path), "json_error": err})
    if isinstance(data, dict):
        out.update(
            {
                "pid": data.get("pid"),
                "pid_alive_now": pid_alive(data.get("pid")),
                "active_field": data.get("active"),
                "complete_field": data.get("complete"),
                "completed_count": len(data.get("completed") or []) if isinstance(data.get("completed"), list) else None,
                "stage": data.get("stage"),
                "test_accessed": data.get("test_accessed"),
            }
        )
    return out


def append_once(path: Path, block: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER in current:
        return False
    path.write_text(current.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    return True


def doc_blocks(summary: Dict[str, Any]) -> Dict[str, str]:
    utc = summary["created_utc"]
    inv = summary["inventory_classification"]
    common = (
        f"{MARKER}\n"
        f"## 2026-09-26 pendulum inventory finalized\n\n"
        f"UTC: {utc}. Metadata-only inventory `experiments/bohn2021_aws/pendulum_inventory_metadata.py` "
        f"completed with no simulations, no validation reads, and no sealed-test reads. "
        f"pendulum_s0 is a completed training artifact (selected `{inv.get('pendulum_s0_selected')}`, "
        f"{inv.get('pendulum_s0_completed_markers')} completed markers). pendulum_s1 is a stale/interrupted "
        f"threshold-reference run (progress {inv.get('pendulum_s1_threshold_progress')}, dead PIDs "
        f"{inv.get('pendulum_s1_dead_attempt_pids')}, tmp files {inv.get('pendulum_s1_tmp_files')}). "
        f"pendulum_s2 is absent/unstarted at train root. This is not control-performance evidence. "
        f"The partial WSL pendulum_s1 timing-sensitive work must be counted as interrupted budget and must not be "
        f"spliced into AWS timing objectives. Final test remains sealed/unauthorized; validation64 remains unopened "
        f"for post-amendment model selection.\n"
    )
    return {
        "DECISIONS.md": common + "\nDecision: treat pendulum_s1 as failed/interrupted historical work and pendulum_s2 as unstarted unless future metadata contradicts this. Formal all-seed pendulum latency-tree evidence requires an AWS-only fresh recovery block or an explicitly vehicle-only development scope; behaviorally fixed trees remain fixed-H comparators.",
        "RESEARCH_LOG.md": common + "\nNext research step: run an AWS-only recovery/candidate preflight and then choose either a vehicle-only paired development remeasurement/freeze or a fresh AWS pendulum recovery block. No validation/test outcomes were opened by this inventory.",
        "RESULTS_AUDIT.md": common + "\nAudit interpretation: inventory establishes artifact state and budget accounting only. It cannot support adaptive-horizon or reproduction claims; it records completed/interrupted/unstarted training boundaries for later fair-budget reporting.",
        "STATUS.md": common + "\nCurrent next action: use the post-inventory preflight output to freeze the next bounded experiment. Do not rerun completed vehicle diagnostics or the pendulum inventory unless artifact hashes are missing.",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INV_RAW.exists() or not INV_SUMMARY.exists():
        raise FileNotFoundError("pendulum inventory outputs are required before this preflight")
    inv = load_json(INV_RAW)
    train_jobs = [job_metadata(task, seed) for task in TASKS for seed in SEEDS]
    completed_by_task = {
        task: [j["seed"] for j in train_jobs if j["task"] == task and j.get("completed", {}).get("passed") is True]
        for task in TASKS
    }
    policy_candidates = [
        {
            "job": j["job"],
            "selected": j.get("completed", {}).get("selected"),
            "policy_sha256": j.get("policy", {}).get("sha256"),
            "policy_summary": j.get("policy", {}).get("summary"),
            "status": j.get("derived_status"),
        }
        for j in train_jobs
        if j.get("policy", {}).get("exists")
    ]
    readiness = {
        "training_status": read_status_metadata(),
        "completed_by_task": completed_by_task,
        "vehicle_all_three_training_policies_available": completed_by_task.get("vehicle") == [0, 1, 2],
        "pendulum_all_three_training_policies_available": completed_by_task.get("pendulum") == [0, 1, 2],
        "formal_two_task_all_seed_validation_blocked_by_pendulum": completed_by_task.get("pendulum") != [0, 1, 2],
        "validation_read_performed": False,
        "test_read_performed": False,
        "recommended_next_experiment": (
            "Freeze and run a no-validation AWS-only candidate/timing design for vehicle completed policies first, "
            "or start fresh AWS pendulum_s1/s2 recovery under a new run root if the immediate goal is all-seed pendulum evidence. "
            "Do not combine WSL and AWS timings; do not open validation64 until candidate set, metrics, and selection rules are frozen."
        ),
    }
    raw: Dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {"inventory_raw": rel(INV_RAW), "inventory_raw_sha256": sha256_file(INV_RAW), "inventory_summary": rel(INV_SUMMARY), "inventory_summary_sha256": sha256_file(INV_SUMMARY)},
        "simulations_run": 0,
        "validation_accessed": False,
        "test_accessed": False,
        "raw_trace_contents_read": False,
        "inventory_classification": classify_inventory(inv),
        "train_jobs": train_jobs,
        "policy_candidates": policy_candidates,
        "readiness": readiness,
    }
    OUT_RAW.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: List[str] = [
        "# Post-inventory recovery preflight",
        "",
        f"Created UTC: {raw['created_utc']}",
        "",
        "Scope: training-root metadata and already-generated pendulum inventory only. No simulations, no validation reads, no sealed-test reads.",
        "",
        "## Candidate/training availability",
        "",
        f"- completed_by_task: {completed_by_task}",
        f"- vehicle_all_three_training_policies_available: {readiness['vehicle_all_three_training_policies_available']}",
        f"- pendulum_all_three_training_policies_available: {readiness['pendulum_all_three_training_policies_available']}",
        f"- formal_two_task_all_seed_validation_blocked_by_pendulum: {readiness['formal_two_task_all_seed_validation_blocked_by_pendulum']}",
        "",
        "## Policy candidates (structural metadata only)",
        "",
        "| job | status | selected | policy sha256 | structural summary |",
        "|---|---|---|---|---|",
    ]
    for c in policy_candidates:
        lines.append(f"| `{c['job']}` | {c['status']} | {c.get('selected')} | `{c.get('policy_sha256')}` | {c.get('policy_summary')} |")
    lines.extend(
        [
            "",
            "## Inventory interpretation",
            "",
            json.dumps(raw["inventory_classification"], indent=2, sort_keys=True),
            "",
            "## Recommended next experiment",
            "",
            readiness["recommended_next_experiment"],
            "",
            "Non-claim: this preflight is metadata and documentation only; it is not validation/test evidence and not a control-performance result.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    appended: Dict[str, bool] = {}
    for name, block in doc_blocks(raw).items():
        appended[name] = append_once(REPO / name, block)
    raw["doc_appends"] = appended
    OUT_RAW.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [rel(OUT_RAW), rel(OUT_SUMMARY)], "doc_appends": appended, "validation_accessed": False, "test_accessed": False, "simulations_run": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
