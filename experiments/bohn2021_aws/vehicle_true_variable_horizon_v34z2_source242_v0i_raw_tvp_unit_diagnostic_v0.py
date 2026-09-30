#!/usr/bin/env python3
"""Zero-resource unit diagnostic for the S-TC2H8 v0i raw-TVP repair.

This does not run MPC, advance a plant, train/refit, open validation64, or touch
sealed/final tests.  It checks the actual authored v0i repair function on a fake
LetMPCEnv-shaped object and performs source-level guards for the one-variable
high-level horizon-action repair.  The purpose is to obtain concrete
implementation evidence while nonzero solver/plant work remains blocked by the
external-backup gate.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments" / "bohn2021_aws"
SERVICE_DIR = ROOT / "scripts" / "research_service"
for path in (SERVICE_DIR, AWS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import execution_contract  # type: ignore  # noqa:E402

NAME = "vehicle_true_variable_horizon_v34z2_source242_v0i_raw_tvp_unit_diagnostic_v0"
TASK_ID = "S-TC2H8-v0i-raw-tvp-unit-diagnostic-v0"
ZERO = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0I_SOURCE = AWS_DIR / "vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair.py"
RESPONSE_LOG = ROOT / "docs" / "bohn2021_takeover" / "astra_reviews" / "RESPONSE_LOG.md"
DOCS = [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


class FakeTVP:
    def __init__(self, values: Any) -> None:
        self.values = values


class FakeControlSystem:
    def __init__(self) -> None:
        self.tvps = {
            "speed": FakeTVP([0.0, 1.0, 2.0]),
            "curvature": FakeTVP([3.0, 4.0, 5.0]),
        }


class FakeEnv:
    def __init__(self) -> None:
        self.control_system = FakeControlSystem()


def run_success(created: dt.datetime, out_dir: Path) -> int:
    snapshot = execution_contract.runtime_snapshot(ROOT)
    if (snapshot.get("task") or {}).get("task_id") != TASK_ID:
        raise RuntimeError("unexpected structured task_id")

    source_text = V0I_SOURCE.read_text(encoding="utf-8")
    static_checks = {
        "v0i_source_exists": V0I_SOURCE.exists(),
        "high_level_numpy_horizon_action_present": "high_level_action = np.asarray([float(h)], dtype=float)" in source_text,
        "env_step_receives_high_level_action": "env.step(high_level_action)" in source_text,
        "raw_restore_function_defined": "def _restore_raw_env_tvps" in source_text,
        "raw_restore_called_before_history_initialization": "raw_tvp_restore_meta = _restore_raw_env_tvps(env, raw_shifted_tvp)" in source_text and source_text.find("raw_tvp_restore_meta = _restore_raw_env_tvps(env, raw_shifted_tvp)") < source_text.find("history_meta = v0h._initialize_env_histories(env)"),
        "scalar_tvp_kept_for_controller_metadata": "ctrl._tvp_data = copy.deepcopy(scalar_tvp)" in source_text,
    }

    import vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair as v0i  # type: ignore  # noqa:E402

    fake_env = FakeEnv()
    raw_shifted_tvp: Dict[str, Any] = {
        "speed": [
            {"true": [1.0, 2.0], "forecast": [[1.1, 2.1], [1.2, 2.2]]},
            {"true": [3.0, 4.0], "forecast": [[3.1, 4.1]]},
        ],
        "curvature": [
            {"true": [0.01], "forecast": [[0.02], [0.03]]},
        ],
        "not_installed_on_env": [
            {"true": [99.0], "forecast": [[100.0]]},
        ],
    }
    meta = v0i._restore_raw_env_tvps(fake_env, raw_shifted_tvp)
    raw_shifted_tvp["speed"][0]["true"][0] = -12345.0
    restore_checks = {
        "speed_values_are_list": isinstance(fake_env.control_system.tvps["speed"].values, list),
        "curvature_values_are_list": isinstance(fake_env.control_system.tvps["curvature"].values, list),
        "speed_first_entry_is_dict": isinstance(fake_env.control_system.tvps["speed"].values[0], dict),
        "curvature_first_entry_is_dict": isinstance(fake_env.control_system.tvps["curvature"].values[0], dict),
        "speed_first_entry_has_true_forecast": set(fake_env.control_system.tvps["speed"].values[0].keys()) >= {"true", "forecast"},
        "curvature_first_entry_has_true_forecast": set(fake_env.control_system.tvps["curvature"].values[0].keys()) >= {"true", "forecast"},
        "restore_meta_marks_speed_raw_format": bool(meta.get("speed", {}).get("first_has_true_forecast_keys")),
        "restore_meta_marks_curvature_raw_format": bool(meta.get("curvature", {}).get("first_has_true_forecast_keys")),
        "unknown_tvp_key_not_added_to_env": "not_installed_on_env" not in fake_env.control_system.tvps,
        "deepcopy_isolation_from_raw_input": fake_env.control_system.tvps["speed"].values[0]["true"][0] == 1.0,
    }
    evidence = {
        "runtime_snapshot_verified": True,
        "v0i_raw_tvp_restore_unit_check_completed": True,
        "fake_raw_tvp_restore_keeps_list_of_dicts": bool(all(restore_checks.values())),
        "deepcopy_isolation_verified": bool(restore_checks["deepcopy_isolation_from_raw_input"]),
        "high_level_horizon_action_static_check": bool(static_checks["high_level_numpy_horizon_action_present"] and static_checks["env_step_receives_high_level_action"]),
        "raw_restore_before_observation_history_static_check": bool(static_checks["raw_restore_called_before_history_initialization"]),
        "no_solver_plant_training_validation_test_usage": True,
        "no_validation64_or_sealed_test_access": True,
        "artifacts_persisted": True,
    }
    hard_pass = all(evidence.values())

    raw_path = out_dir / "raw.json"
    summary_path = out_dir / "summary.md"
    completed_path = out_dir / "completed.json"
    backup_request = ROOT / "research_artifacts" / "aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_S_TC2H8_V0I_RAW_TVP_UNIT_DIAGNOSTIC_{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    state_path = ROOT / "research_artifacts" / "aws_state" / f"continue_state_{created.strftime('%Y%m%dT%H%M%SZ')}_after_s_tc2h8_v0i_raw_tvp_unit_diagnostic.md"

    raw = {
        "status": "complete",
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "task_id": TASK_ID,
        "classification": "zero_resource_implementation_unit_diagnostic_not_validation_not_test",
        "v0i_source": {"path": rel(V0I_SOURCE), "sha256": sha256(V0I_SOURCE)},
        "static_checks": static_checks,
        "restore_meta": meta,
        "restore_checks": restore_checks,
        "pass_evidence": evidence,
        "hard_pass": hard_pass,
        "resources": dict(ZERO),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "test_accessed": False,
        "limitations": [
            "This checks the v0i raw-TVP repair on a fake LetMPCEnv-shaped object only.",
            "It does not show that CasADi/IPOPT solves or plant transitions will succeed.",
            "S-TC2H8 nonzero solver/plant evidence remains blocked until external backup is verified.",
        ],
    }
    write_json(raw_path, raw)
    summary_lines = [
        "# S-TC2H8 v0i raw-TVP unit diagnostic v0",
        "",
        f"UTC: `{created.isoformat()}`. Zero-resource implementation diagnostic; no solver, plant, training/refit, validation64, or sealed/final-test use.",
        "",
        f"Local hard_pass: `{hard_pass}`.",
        "",
        "## Static checks",
        "",
    ]
    for key, value in static_checks.items():
        summary_lines.append(f"- {key}: `{value}`")
    summary_lines += ["", "## Fake-env raw TVP restore checks", ""]
    for key, value in restore_checks.items():
        summary_lines.append(f"- {key}: `{value}`")
    summary_lines += ["", "## Limitation", "", "This is not solver/control evidence; it only verifies the one-variable implementation repair before the backup-blocked S-TC2H8 launch."]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_json(backup_request, {
        "request": "backup_after_s_tc2h8_v0i_raw_tvp_unit_diagnostic",
        "created_utc": created.isoformat(),
        "must_cover": [rel(Path(__file__).resolve()), rel(out_dir), rel(backup_request), rel(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", rel(RESPONSE_LOG)],
        "resources": dict(ZERO),
        "backup_required_before_nonzero_s_tc2h8_solver_or_plant_resources": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    block = f"""<!-- s-tc2h8-v0i-raw-tvp-unit-diagnostic-{created.strftime('%Y%m%dT%H%M%SZ')} -->
## S-TC2H8 v0i raw-TVP unit diagnostic

UTC: {created.isoformat()}. Zero-resource implementation unit diagnostic completed with hard_pass `{hard_pass}`. It verified on a fake LetMPCEnv-shaped object that `_restore_raw_env_tvps` writes list-of-dicts entries containing `true` and `forecast` into environment TVP objects and deep-copies them, and it statically confirmed the high-level `env.step(high_level_action)` horizon-action path. This is not solver/control evidence and does not remove the external-backup blocker. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(completed_path)}`.
"""
    for doc in DOCS:
        append_if_missing(doc, "s-tc2h8-v0i-raw-tvp-unit-diagnostic-" + created.strftime('%Y%m%dT%H%M%SZ'), block)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("# Continue state after S-TC2H8 v0i raw-TVP unit diagnostic\n\n" + block + "\nNext: wait for or repair verified external backup, then run the already-authored S-TC2H8 v0i bounded microcontinuation under the frozen diagnostic budget.\n", encoding="utf-8")
    completed_files = [Path(__file__).resolve(), V0I_SOURCE, raw_path, summary_path, backup_request, state_path] + [p for p in DOCS if p.exists()]
    completed = {
        "status": "complete",
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "classification": raw["classification"],
        "summary": rel(summary_path),
        "raw": rel(raw_path),
        "backup_request": rel(backup_request),
        "state": rel(state_path),
        "resources": dict(ZERO),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hard_pass": hard_pass,
        "pass_evidence": evidence,
        "hashes": {rel(p): sha256(p) for p in sorted(set(completed_files)) if p.exists() and p.is_file()},
    }
    write_json(completed_path, completed)
    execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), evidence)
    print(json.dumps({"completed": rel(completed_path), "summary": rel(summary_path), "hard_pass": hard_pass, "resources": dict(ZERO), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def run_failure(created: dt.datetime, out_dir: Path, exc: BaseException) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / "failed.json"
    write_json(failed_path, {
        "status": "failed",
        "created_utc": created.isoformat(),
        "task_id": TASK_ID,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback_tail": traceback.format_exc().splitlines()[-24:],
        "resources": dict(ZERO),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
    })
    try:
        execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), {"no_scientific_outcome": True, "failed_json": rel(failed_path), "no_solver_plant_training_validation_test_usage": True, "no_validation64_or_sealed_test_access": True}, engineering_error="loader")
    except Exception:
        pass
    print(json.dumps({"failed": rel(failed_path), "error": f"{type(exc).__name__}: {exc}", "resources": dict(ZERO)}, sort_keys=True), flush=True)
    return 1


def main() -> int:
    created = now_utc()
    out_dir = ROOT / "research_artifacts" / "aws_diagnostics" / f"{NAME}_{created.strftime('%Y%m%dT%H%M%SZ')}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "run_started.json", {"started_utc": created.isoformat(), "task_id": TASK_ID, "resources": dict(ZERO), "validation64_bank_opened": False, "sealed_test_accessed": False})
        return run_success(created, out_dir)
    except BaseException as exc:
        return run_failure(created, out_dir, exc)


if __name__ == "__main__":
    raise SystemExit(main())
