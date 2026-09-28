#!/usr/bin/env python3
"""Metadata/runtime preflight for v2 transition-hold development validation.

This diagnostic verifies the newly frozen v2 development-validation runner,
protocols, smoke readiness, arm/schedule dimensions and terminal-source metadata
without generating/opening the fresh v2 devval bank, without running rollouts,
and without opening sealed final-test content.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import vehicle_safe_shortening_v2_transition_hold_devval_shard_runner as r  # noqa:E402

ROOT = r.ROOT
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z"
SMOKE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-safe-shortening-v2-transition-hold-devval-preflight-20260928T004500Z"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return rel(value)
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_completed_marker(path: Path, max_hashes: int | None = None) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise RuntimeError("Completed marker did not pass: %s" % rel(path))
    hashes = done.get("hashes") or {}
    checked = 0
    for name, expected in sorted(hashes.items()):
        actual = sha256(ROOT / name)
        if actual != expected:
            raise RuntimeError("Hash mismatch for %s" % name)
        checked += 1
        if max_hashes is not None and checked >= max_hashes:
            break
    done["hashes_checked_now"] = checked
    done["hashes_total"] = len(hashes)
    done["hash_check_truncated"] = max_hashes is not None and checked < len(hashes)
    return done


def verify_terminal_metadata(source_rel: str) -> Dict[str, Any]:
    folder = ROOT / source_rel
    manifest_path = folder / "manifest.json"
    completed_path = folder / "completed.json"
    model_path = folder / "model.zip"
    if not manifest_path.exists() or not completed_path.exists() or not model_path.exists():
        raise RuntimeError("Terminal source missing required files: %s" % source_rel)
    manifest = read_json(manifest_path)
    done = read_json(completed_path)
    if manifest.get("task") != r.TASK or done.get("status") != "complete" or int(done.get("steps", -1)) != 15000:
        raise RuntimeError("Terminal source is not a complete vehicle 15k model: %s" % source_rel)
    return {
        "source": source_rel,
        "fixed_horizon": manifest.get("fixed_horizon"),
        "seed": manifest.get("seed"),
        "model_zip_sha256": sha256(model_path),
        "model_zip_size_bytes": model_path.stat().st_size,
        "manifest_sha256": sha256(manifest_path),
        "completed_sha256": sha256(completed_path),
        "weights_hash": done.get("final_hash"),
    }


def case_schedule_audit(schedule: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(schedule["rows"])
    shards = list(schedule["shards"])
    cases_seen = [int(row["case_index"]) for row in rows]
    per_case: Dict[str, int] = {}
    for case in cases_seen:
        per_case[str(case)] = per_case.get(str(case), 0) + 1
    shard_lengths = [int(s["episodes"]) for s in shards]
    shard_case_counts = [len(s["cases"]) for s in shards]
    return {
        "episode_count": len(rows),
        "shard_count": len(shards),
        "episodes_per_full_shard": int(schedule["episodes_per_full_shard"]),
        "shard_lengths_unique": sorted(set(shard_lengths)),
        "shard_case_counts_unique": sorted(set(shard_case_counts)),
        "unique_case_count": len(per_case),
        "per_case_episode_count_unique": sorted(set(per_case.values())),
        "case_ids_min_max": [min(int(k) for k in per_case), max(int(k) for k in per_case)],
        "arms_count": len(schedule["arms"]),
        "adaptive_arm_count": sum(1 for a in schedule["arms"] if str(a["arm_id"]).startswith("safe_shortening_v2_hold3")),
        "matched_fixed_grid_arm_count": sum(1 for a in schedule["arms"] if a.get("role") == "matched_terminal_fixed_grid"),
        "independent_seed0_grid_arm_count": sum(1 for a in schedule["arms"] if a.get("role") == "independent_terminal_seed0_grid"),
        "first_shard": shards[0],
        "last_shard": shards[-1],
    }


def write_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle safe-shortening v2 transition-hold devval preflight",
        "",
        "Created UTC: `%s`." % raw["created_utc"],
        "",
        "Metadata/runtime diagnostic only: no fresh devval bank generation, no rollout episodes, no control steps, no historical validation64 bank reopen and no sealed-test access.",
        "",
        "## Checks",
        "",
        "- Legacy TF/runtime preflight passed: `%s` (`%s`)." % (raw["runtime_preflight"].get("passed"), raw["runtime_preflight"].get("tensorflow_version")),
        "- V2 smoke completed marker passed: `%s`; smoke episodes/control steps: `%s` / `%s`; validation64 opened: `%s`; sealed test opened: `%s`." % (
            raw["smoke"]["completed_passed"], raw["smoke"].get("episodes"), raw["smoke"].get("control_steps"), raw["smoke"].get("validation64_bank_opened"), raw["smoke"].get("sealed_test_accessed")),
        "- Devval bank absent before shard run: `%s`; incomplete bank detected: `%s`." % (raw["bank_state"]["bank_absent"], raw["bank_state"]["incomplete_bank_detected"]),
        "- Schedule: `%s` episodes, `%s` shards, `%s` arms; shard lengths `%s`; per-case episode counts `%s`." % (
            raw["schedule_audit"]["episode_count"], raw["schedule_audit"]["shard_count"], raw["schedule_audit"]["arms_count"], raw["schedule_audit"]["shard_lengths_unique"], raw["schedule_audit"]["per_case_episode_count_unique"]),
        "- Arms: adaptive `%s`, matched fixed grid `%s`, independent seed0 grid `%s`." % (
            raw["schedule_audit"]["adaptive_arm_count"], raw["schedule_audit"]["matched_fixed_grid_arm_count"], raw["schedule_audit"]["independent_seed0_grid_arm_count"]),
        "- Unique terminal sources verified from metadata/hash: `%s`." % len(raw["terminal_sources"]),
        "",
        "## Decision",
        "",
        raw["decision"],
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    text = (
        "\n<!-- %s -->\n" % MARKER +
        "## 2026-09-28 vehicle safe-shortening v2 transition-hold devval preflight\n\n" +
        "UTC: %s. Ran metadata/runtime preflight for the frozen IMPROVED v2 transition-hold development-validation campaign. No devval bank generation, no rollout/control steps, no historical validation64 bank reopen, and no sealed-test access. Schedule/arm dimensions verified: %d episodes, %d shards, %d arms; terminal-source metadata verified for %d unique sources. Decision: %s Artifacts: `%s`, `%s`, `%s`.\n" % (
            raw["created_utc"], raw["schedule_audit"]["episode_count"], raw["schedule_audit"]["shard_count"], raw["schedule_audit"]["arms_count"], len(raw["terminal_sources"]), raw["decision"],
            rel(OUT_DIR / "summary.md"), rel(OUT_DIR / "raw.json"), rel(OUT_DIR / "completed.json"))
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def write_backup_request(raw: Mapping[str, Any]) -> str:
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_DEVVAL_PREFLIGHT_%s.json" % raw["created_utc"].replace("-", "").replace(":", "").split(".")[0])
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup v2 transition-hold devval protocols/source/preflight/smoke before any devval shard rollout",
        "artifacts": [
            rel(OUT_DIR),
            rel(r.DEVVAL_PROTOCOL),
            rel(r.TRANSITION_PROTOCOL),
            rel(Path(r.__file__).resolve()),
            rel(Path(__file__).resolve()),
            rel(SMOKE_DIR),
        ],
        "rollout_episodes": 0,
        "control_steps": 0,
        "sealed_test_accessed": False,
        "historical_validation64_bank_opened": False,
    })
    return rel(path)


def main() -> int:
    if OUT_DIR.exists():
        completed = OUT_DIR / "completed.json"
        if completed.exists():
            verify_completed_marker(completed)
            raise SystemExit("v2 devval preflight already completed and verified; refusing rerun")
        leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise RuntimeError("Partial preflight output exists; inspect before recovery: " + ", ".join(rel(p) for p in leftovers[:10]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {"started_utc": created, "pid": os.getpid(), "sealed_test_accessed": False, "historical_validation64_bank_opened": False})

    runtime = r.verify_runtime()
    if not runtime.get("passed"):
        raise RuntimeError(runtime.get("diagnosis") + ": " + runtime.get("exception", ""))
    r.v1.latency_verify()
    for protocol in (r.TRANSITION_PROTOCOL, r.DEVVAL_PROTOCOL):
        if not protocol.exists():
            raise RuntimeError("Missing frozen protocol: %s" % rel(protocol))

    smoke_completed = verify_completed_marker(SMOKE_DIR / "completed.json")
    smoke_raw = read_json(SMOKE_DIR / "raw.json")
    if smoke_raw.get("sealed_test_accessed") is not False or smoke_raw.get("validation64_bank_opened") is not False:
        raise RuntimeError("V2 smoke access flags are not closed")
    if not smoke_raw.get("replay", {}).get("passed"):
        raise RuntimeError("V2 smoke replay did not pass")
    for seed, comp in (smoke_raw.get("paired_comparisons") or {}).items():
        if comp.get("unsafe_above_H25_dispatch"):
            raise RuntimeError("V2 smoke dispatched H>25 for seed %s" % seed)

    arms = r.build_arms()
    schedule = r.make_schedule(arms)
    sched_audit = case_schedule_audit(schedule)
    expected = {
        "episode_count": r.DEVVAL_CASES * len(arms),
        "shard_count": r.DEVVAL_CASES // r.CASES_PER_SHARD,
        "arms_count": 43,
        "adaptive_arm_count": 3,
        "matched_fixed_grid_arm_count": 30,
        "independent_seed0_grid_arm_count": 10,
    }
    for key, value in expected.items():
        if int(sched_audit[key]) != int(value):
            raise RuntimeError("Schedule audit mismatch %s: %r != %r" % (key, sched_audit[key], value))
    if sched_audit["per_case_episode_count_unique"] != [43] or sched_audit["shard_lengths_unique"] != [172]:
        raise RuntimeError("Schedule case/shard balance mismatch")

    terminal_sources: Dict[str, Any] = {}
    for source in sorted({str(arm["terminal_source"]) for arm in arms}):
        terminal_sources[source] = verify_terminal_metadata(source)

    bank_state = {
        "bank_path": rel(r.BANK_PATH),
        "bank_completed_path": rel(r.BANK_COMPLETED),
        "bank_exists": r.BANK_PATH.exists(),
        "bank_completed_exists": r.BANK_COMPLETED.exists(),
        "bank_absent": (not r.BANK_PATH.exists() and not r.BANK_COMPLETED.exists()),
        "incomplete_bank_detected": (r.BANK_PATH.exists() != r.BANK_COMPLETED.exists()),
        "bank_generation_performed_now": False,
    }
    if bank_state["incomplete_bank_detected"]:
        raise RuntimeError("Incomplete v2 devval bank state detected")
    if r.GATE_PATH.exists() != r.GATE_COMPLETED.exists():
        raise RuntimeError("Incomplete v2 devval gate state detected")

    decision = "Preflight passed. Do not run v2 devval shard until supervisor reports verified external backup for smoke, protocols, runner and this preflight. After backup, next concrete action is shard00 with legacy interpreter."
    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "IMPROVED_vehicle_safe_shortening_v2_transition_hold_devval_preflight_engineering_only",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "runtime_preflight": runtime,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "protocols": {
            "transition": {"path": rel(r.TRANSITION_PROTOCOL), "sha256": sha256(r.TRANSITION_PROTOCOL)},
            "devval": {"path": rel(r.DEVVAL_PROTOCOL), "sha256": sha256(r.DEVVAL_PROTOCOL)},
        },
        "source_hashes": r.source_hashes(),
        "smoke": {
            "path": rel(SMOKE_DIR),
            "completed_passed": True,
            "completed_hashes_checked_now": smoke_completed.get("hashes_checked_now"),
            "completed_hashes_total": smoke_completed.get("hashes_total"),
            "episodes": smoke_raw.get("budget_actual", {}).get("episodes"),
            "control_steps": smoke_raw.get("budget_actual", {}).get("control_steps"),
            "replay_passed": smoke_raw.get("replay", {}).get("passed"),
            "paired_comparisons": smoke_raw.get("paired_comparisons"),
            "validation64_bank_opened": smoke_raw.get("validation64_bank_opened"),
            "sealed_test_accessed": smoke_raw.get("sealed_test_accessed"),
        },
        "bank_state": bank_state,
        "schedule_audit": sched_audit,
        "arms": arms,
        "terminal_sources": terminal_sources,
        "budget_actual": {"rollout_episodes": 0, "control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_episodes": 0},
        "next_action": {
            "requires_verified_external_backup_first": True,
            "script": "experiments/bohn2021_aws/vehicle_safe_shortening_v2_transition_hold_devval_shard_runner.py",
            "args": ["--shard", "0"],
            "interpreter": "legacy",
            "declared_episode_budget": 172,
            "declared_control_step_upper_bound": 25800,
            "sealed_test_accessed": False,
        },
        "decision": decision,
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    files: List[Path] = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"]
    files += [ROOT / raw["backup_request"], r.DEVVAL_PROTOCOL, r.TRANSITION_PROTOCOL, Path(r.__file__).resolve(), Path(__file__).resolve(), SMOKE_DIR / "completed.json", SMOKE_DIR / "raw.json", SMOKE_DIR / "summary.md"]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "rollout_episodes": 0,
        "control_steps": 0,
        "backup_request": raw["backup_request"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files))},
        "headline": {
            "schedule_episode_count": sched_audit["episode_count"],
            "shard_count": sched_audit["shard_count"],
            "arms_count": sched_audit["arms_count"],
            "bank_generation_performed_now": False,
            "decision": decision,
        },
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "rollout_episodes": 0,
        "control_steps": 0,
        "schedule_episode_count": sched_audit["episode_count"],
        "shard_count": sched_audit["shard_count"],
        "arms_count": sched_audit["arms_count"],
        "bank_generation_performed_now": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_request": raw["backup_request"],
        "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "rollout_episodes": 0,
            "control_steps": 0,
            "next_recovery_hint": "Inspect failure, preserve output, then create one-variable repair. Do not open sealed test.",
        })
        raise
