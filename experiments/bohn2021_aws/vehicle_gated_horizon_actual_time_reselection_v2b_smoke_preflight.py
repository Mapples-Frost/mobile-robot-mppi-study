#!/usr/bin/env python3
"""Metadata-only preflight for actual-time V2b smoke.

This checks that the newly frozen V2b smoke source/protocol/nominations are
internally consistent before any rollout.  It performs no environment import, no
TensorFlow import, no simulation, no training, no validation64-bank access, and
no sealed-test access.
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2b_smoke.py"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_protocol_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_protocol_20260928.json"
V2B_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z/completed.json"
V2B_NOMINATED = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair_20260928T0915Z/nominated_policies.json"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z.md"
BACKUP = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_ACTUAL_TIME_RESELECTION_V2B_SMOKE_PREFLIGHT_20260928T0910Z.json"
DOCS = [ROOT / x for x in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]]
MARKER = "<!-- vehicle-gated-horizon-actual-time-reselection-v2b-smoke-preflight-20260928 -->"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def sha256(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def const_assignments(py: Path) -> Dict[str, Any]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out


def append_once(path: Path, block: str) -> None:
    if not path.exists():
        return
    old = path.read_text(encoding="utf-8")
    if MARKER in old:
        return
    path.write_text(old.rstrip() + "\n" + block + "\n", encoding="utf-8")


def main() -> int:
    if OUT.exists():
        raise RuntimeError("Output directory already exists; refusing overwrite: %s" % OUT)
    required = [SMOKE, PROTOCOL_MD, PROTOCOL_JSON, V2B_COMPLETED, V2B_NOMINATED]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required inputs: %r" % missing)

    protocol = read_json(PROTOCOL_JSON)
    completed = read_json(V2B_COMPLETED)
    nominated = read_json(V2B_NOMINATED)
    smoke_text = SMOKE.read_text(encoding="utf-8")
    consts = const_assignments(SMOKE)

    expected_ids = protocol["inputs"]["nominated_policy_ids"]
    actual_ids: Dict[str, str] = {}
    for s in ["0", "1", "2"]:
        actual_ids[s] = str(nominated["nominations"][s]["nominated"]["candidate_id"])

    checks: Dict[str, Any] = {}
    checks["v2b_completed_passed"] = completed.get("passed") is True
    checks["v2b_completed_acceptance_for_smoke"] = completed.get("acceptance_for_smoke_met") is True
    checks["v2b_nomination_acceptance_for_smoke"] = nominated.get("acceptance_for_smoke_met") is True
    checks["nomination_ids_match_protocol"] = actual_ids == expected_ids
    checks["seed0_is_fixed_fallback"] = actual_ids.get("0") == "fixed" and nominated["nominations"]["0"].get("fallback_to_fixed") is True
    checks["seed1_seed2_are_adaptive_h15_p1_g5"] = actual_ids.get("1") == "h15_p1_g5" and actual_ids.get("2") == "h15_p1_g5"
    checks["protocol_episode_budget_matches_source"] = protocol["split"]["episodes_exact"] == 36 and protocol["budgets"]["control_step_upper_bound"] == 5400
    checks["source_constants_match_protocol"] = (consts.get("CASES_EXPECTED") == 2 and consts.get("REPEATS") == 2 and consts.get("MAX_STEPS") == 150 and consts.get("BASE_H") == 25)
    checks["source_records_no_validation_or_test_access"] = (
        "historical_validation64_bank_opened" in smoke_text
        and "validation64_bank_opened" in smoke_text
        and "sealed_test_accessed" in smoke_text
        and "sealed_test_bank_opened" in smoke_text
    )
    checks["source_has_no_above_H25_dispatch_clause"] = "selected = min(raw_h, BASE_H)" in smoke_text
    checks["source_uses_runtime_preflight"] = "base.runtime_preflight()" in smoke_text
    checks["source_has_no_direct_tensorflow_import"] = "import tensorflow" not in smoke_text and "from tensorflow" not in smoke_text
    checks["smoke_output_not_started"] = not (ROOT / "research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/completed.json").exists()
    checks["no_validation_or_test_access"] = True
    checks["new_rollouts"] = 0
    checks["new_training_episodes"] = 0
    checks["new_gradient_steps"] = 0
    hard_pass = all(bool(v) for k, v in checks.items() if k not in ["new_rollouts", "new_training_episodes", "new_gradient_steps"])

    OUT.mkdir(parents=True, exist_ok=False)
    raw = {
        "created_utc": now(),
        "method": "metadata_only_actual_time_v2b_smoke_preflight",
        "classification": "IMPROVED preflight; zero rollouts/training; not ORIGINAL SAC",
        "checks": checks,
        "hard_pass": hard_pass,
        "protocol_summary": {
            "episodes_exact": protocol["split"]["episodes_exact"],
            "control_step_upper_bound": protocol["budgets"]["control_step_upper_bound"],
            "bank_rng": protocol["split"]["bank_rng"],
            "order_seed": protocol["split"]["order_seed"],
            "nominated_policy_ids": expected_ids,
        },
        "actual_nomination_ids": actual_ids,
        "access_flags": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_bank_opened": False, "sealed_test_accessed": False},
        "input_hashes": {rel(p): sha256(p) for p in required + [Path(__file__).resolve()]},
        "next_action": "after_verified_backup_run_legacy_actual_time_v2b_smoke" if hard_pass else "repair_preflight_failures_before_any_simulation",
    }
    write_json(OUT / "raw.json", raw)
    summary = (
        "# Vehicle actual-time-aware V2b smoke preflight\n\n"
        f"UTC: `{raw['created_utc']}`. Metadata-only; no simulations, no training, no validation64-bank access, no sealed-test access.\n\n"
        f"Hard pass: `{hard_pass}`.\n\n"
        f"Nominations: `{actual_ids}`.\n\n"
        f"Checks: `{checks}`.\n\n"
        f"Next action: `{raw['next_action']}`.\n"
    )
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    completed_obj = {"created_utc": now(), "passed": True, "hard_pass": hard_pass, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollouts": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(BACKUP), "next_action": raw["next_action"]}
    write_json(OUT / "completed.json", completed_obj)
    STATE.write_text(
        "# V2b smoke preflight state\n\n"
        f"UTC: {now()}\n\n"
        f"- hard_pass={hard_pass}\n"
        f"- nominations={actual_ids}\n"
        "- No simulations/training/validation64/test access.\n"
        f"- Next action: {raw['next_action']} after verified backup.\n",
        encoding="utf-8",
    )
    backup_obj = {"created_utc": now(), "reason": "Backup required after V2b smoke preflight/source/protocol before any smoke simulation.", "backup_required_before_more_simulations": True, "artifacts_requiring_backup": [rel(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "completed.json", STATE, BACKUP, SMOKE, PROTOCOL_MD, PROTOCOL_JSON, Path(__file__).resolve()]], "sha256": {rel(p): sha256(p) for p in [OUT / "summary.md", OUT / "raw.json", OUT / "completed.json", STATE, SMOKE, PROTOCOL_MD, PROTOCOL_JSON, Path(__file__).resolve()]}, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_rollouts": 0}
    write_json(BACKUP, backup_obj)
    block = (
        f"\n{MARKER}\n"
        "## 2026-09-28 vehicle actual-time-aware V2b smoke preflight\n\n"
        f"UTC: {raw['created_utc']}. Metadata-only preflight for the frozen IMPROVED V2b smoke completed: hard_pass={hard_pass}, nominations={actual_ids}. "
        "No simulations, training, validation64 bank access, or sealed-test access. "
        f"Next action after verified backup: `{raw['next_action']}`. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.\n"
    )
    for d in DOCS:
        append_once(d, block)
    print(summary)
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
