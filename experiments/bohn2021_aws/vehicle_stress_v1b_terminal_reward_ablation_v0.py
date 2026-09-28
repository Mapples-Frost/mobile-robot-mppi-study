#!/usr/bin/env python3
"""Vehicle stress-v1b terminal/reward-source ablation v0.

Development-only diagnostic following the stress-v1b matched-continuation prefix
artifact reanalysis.  It freezes a small matched-prefix branch ablation over the
corrected-positive v1b states and deterministic controls, then (after external
backup) can run a smoke or full ablation.

Scientific question
-------------------
The corrected v1b labels contain only four positive states across two mined
cases.  Before any selector/refit, test whether those local gains are robust to
terminal-value source choices, or whether H-specific learned terminal values
create apparent horizon opportunities.  Physical cost, synthetic total cost and
measured decision/solver timing are reported separately; total cost is not used
as a measured-runtime claim.

Access/budget contract
----------------------
No validation64-bank access, no sealed-test access, and no training/refit or
gradient updates.  --dry-run performs no simulation and writes the frozen
protocol plus a backup request.  --run-smoke/--run-full require a verified
external backup proof postdating the dry-run/source before running rollouts.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stage2_terminal_objective_smoke_v0 as base_smoke  # noqa:E402
import vehicle_stress_scenario_opportunity_probe_v1_runner as stage1_runner  # noqa:E402

TASK = "vehicle"
STAMP = "20260928T2320Z"
PREFIX_H = 15
MAX_STEPS = 150
MATERIAL_GAIN = 3.0
STATE_DISTANCE_TOL = 1e-5

V1B_RUN = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_20260928T2220Z_schema_repair"
V1B_RUN_RAW = V1B_RUN / "raw.json"
V1B_RUN_DONE = V1B_RUN / "completed.json"
PREFIX_POST = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z"
PREFIX_POST_RAW = PREFIX_POST / "raw.json"
PREFIX_POST_DONE = PREFIX_POST / "completed.json"
STAGE1_BANK = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
STAGE1_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json"
STAGE1_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json"
V1B_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_20260928T2210Z_schema_repair.json"

DRYRUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_{STAMP}"
SMOKE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_smoke_{STAMP}"
FULL_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_full_{STAMP}"
STATE_DRYRUN = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_{STAMP}.md"
STATE_SMOKE = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0_smoke_{STAMP}.md"
STATE_FULL = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_terminal_reward_ablation_v0_full_{STAMP}.md"
PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_{STAMP}.json"
PROTOCOL_MD = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP_BEFORE_SMOKE = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0_SMOKE_{STAMP}.json"
MARKER_DRYRUN = f"vehicle-stress-v1b-terminal-reward-ablation-v0-dryrun-{STAMP}"
MARKER_RUN = f"vehicle-stress-v1b-terminal-reward-ablation-v0-run-{STAMP}"

# Deterministic controls: two same-stratum nonmaterial controls and two lower
# stress controls, chosen by frozen target_index rather than post-hoc favorable
# adaptive outcome.  This preserves unfavorable/neutral controls in the ablation.
CONTROL_TARGET_INDICES = [8, 9, 16, 18]
SMOKE_TARGET_INDICES = [2, 4, 8, 16]


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return clean_jsonable(value.tolist())
    if hasattr(value, "item"):
        return clean_jsonable(value.item())
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def completed_passed(path: Path, check_hashes: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError(f"sealed-test flag not false: {rel(path)}")
    if obj.get("historical_validation64_bank_opened") not in (False, None):
        raise ContractError(f"validation64 flag not false: {rel(path)}")
    if check_hashes:
        for name, expected in (obj.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError(f"hash mismatch from {rel(path)} for {name}")
    return obj


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime], purpose: str) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError(f"backup proof does not exist: {rel(path)}")
    proof = read_json(path)
    if not (proof.get("backup_verified") is True or proof.get("status") == "verified"):
        raise ContractError("backup proof is not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof remaining_changed_files != 0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    packages = proof.get("packages_this_run") or []
    if not (proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256") or packages):
        raise ContractError("backup proof lacks package/release sha")
    for pkg in packages:
        if not pkg.get("sha256") or not pkg.get("bytes") or not pkg.get("verification"):
            raise ContractError("backup package entry lacks sha256/bytes/verification")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError("backup proof has no parseable time")
        if proof_time < min_time:
            raise ContractError(f"backup proof predates required inputs for {purpose}: {proof_time.isoformat()} < {min_time.isoformat()}")
    return {"path": rel(path), "sha256": sha256(path), "time": None if proof_time is None else proof_time.isoformat(), "commit": proof.get("commit"), "remaining_changed_files": proof.get("remaining_changed_files"), "packages_this_run": packages, "purpose": purpose}


def verify_inputs(check_prefix_hashes: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (V1B_RUN_RAW, V1B_RUN_DONE, PREFIX_POST_RAW, PREFIX_POST_DONE, STAGE1_BANK, STAGE1_PROTOCOL, STAGE1_DONE, V1B_PROTOCOL):
        if not p.exists():
            raise ContractError(f"required input missing: {rel(p)}")
    run_done = completed_passed(V1B_RUN_DONE, check_hashes=False)
    prefix_done = completed_passed(PREFIX_POST_DONE, check_hashes=check_prefix_hashes)
    stage1_done = completed_passed(STAGE1_DONE, check_hashes=False)
    if run_done.get("new_training_episodes", 0) != 0 or run_done.get("new_gradient_steps", 0) != 0:
        raise ContractError("v1b input unexpectedly records training")
    if prefix_done.get("new_rollouts", 0) != 0 or prefix_done.get("new_training_episodes", 0) != 0 or prefix_done.get("new_gradient_steps", 0) != 0:
        raise ContractError("prefix postdiagnostic unexpectedly records simulation/training")
    raw = read_json(PREFIX_POST_RAW)
    v1b_raw = read_json(V1B_RUN_RAW)
    if raw.get("sealed_test_accessed") is not False or raw.get("historical_validation64_bank_opened") is not False:
        raise ContractError("prefix raw access flags invalid")
    if v1b_raw.get("sealed_test_accessed") is not False or v1b_raw.get("historical_validation64_bank_opened") is not False:
        raise ContractError("v1b raw access flags invalid")
    return run_done, prefix_done, raw, v1b_raw


def choose_candidate_horizons(row: Mapping[str, Any], positive: bool) -> List[int]:
    out: List[int] = []
    for key in ("best_non_H15_physical", "best_non_H15_total"):
        item = row.get(key) or {}
        try:
            h = int(item.get("horizon"))
        except Exception:
            continue
        if h != PREFIX_H and h not in out:
            out.append(h)
    if positive:
        for h in row.get("corrected_material_positive_horizons") or []:
            h = int(h)
            if h != PREFIX_H and h not in out:
                out.append(h)
            if len(out) >= 2:
                break
    if not out:
        out = [10]
    return out[:2]


def make_state_specs(prefix_raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = list((prefix_raw.get("analysis") or {}).get("corrected_rows") or [])
    if not rows:
        raise ContractError("no corrected rows in prefix postdiagnostic raw")
    by_tid = {int(r["target_index"]): r for r in rows}
    positives = [r for r in rows if r.get("label_corrected") == "positive_non_H15_corrected_prefix"]
    positives = sorted(positives, key=lambda r: int(r["target_index"]))
    controls = []
    for tid in CONTROL_TARGET_INDICES:
        r = by_tid.get(tid)
        if r is None:
            raise ContractError(f"control target_index {tid} missing")
        if r.get("label_corrected") != "negative_or_neutral_corrected_prefix":
            raise ContractError(f"control target_index {tid} is not negative/neutral")
        controls.append(r)
    if len(positives) != 4:
        raise ContractError(f"expected 4 corrected positive states from v1b; found {len(positives)}")
    specs: List[Dict[str, Any]] = []
    for r in positives + controls:
        positive = r.get("label_corrected") == "positive_non_H15_corrected_prefix"
        candidates = choose_candidate_horizons(r, positive=positive)
        specs.append({
            "state_id": "target%02d_case%02d_b%03d" % (int(r["target_index"]), int(r["case"]), int(r["branch_step"])),
            "target_index": int(r["target_index"]),
            "case": int(r["case"]),
            "branch_step": int(r["branch_step"]),
            "window_id": r.get("window_id"),
            "selection_role": r.get("selection_role"),
            "corrected_label": r.get("label_corrected"),
            "state_role": "corrected_positive" if positive else "deterministic_control_negative_or_neutral",
            "candidate_horizons": candidates,
            "material_horizons_from_v1b": [int(h) for h in (r.get("corrected_material_positive_horizons") or [])],
            "best_physical_from_v1b": r.get("best_non_H15_physical"),
            "best_total_from_v1b": r.get("best_non_H15_total"),
        })
    return specs


def modes_for_candidate(h: int, state_role: str, full: bool) -> List[str]:
    if state_role == "corrected_positive":
        modes = ["per_h", "h15_terminal", "h25_terminal", "zero_terminal"]
    else:
        modes = ["per_h", "h15_terminal", "zero_terminal"] if full else ["per_h", "zero_terminal"]
    # Avoid exact duplicate of per_h when the candidate horizon already uses H25 terminal.
    if h == 25:
        modes = [m for m in modes if m != "h25_terminal"]
    return modes


def build_plan(specs: Sequence[Mapping[str, Any]], target_indices: Optional[Sequence[int]], full: bool) -> List[Dict[str, Any]]:
    allowed = None if target_indices is None else set(int(x) for x in target_indices)
    plan: List[Dict[str, Any]] = []
    for spec in specs:
        if allowed is not None and int(spec["target_index"]) not in allowed:
            continue
        sid = str(spec["state_id"])
        plan.append({"state_id": sid, "target_index": int(spec["target_index"]), "case": int(spec["case"]), "branch_step": int(spec["branch_step"]), "role": spec["state_role"], "horizon": PREFIX_H, "terminal_mode": "per_h"})
        for h in spec["candidate_horizons"]:
            for mode in modes_for_candidate(int(h), str(spec["state_role"]), full=full):
                plan.append({"state_id": sid, "target_index": int(spec["target_index"]), "case": int(spec["case"]), "branch_step": int(spec["branch_step"]), "role": spec["state_role"], "horizon": int(h), "terminal_mode": mode})
    for i, item in enumerate(plan):
        item["execution_index"] = i
    return plan


def state_tuple(obj: Any) -> Tuple[float, float, float]:
    if not isinstance(obj, Mapping):
        return (float("nan"), float("nan"), float("nan"))
    return (float(obj.get("x", float("nan"))), float(obj.get("y", float("nan"))), float(obj.get("theta", float("nan"))))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def no_regression(c: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(c.get("success")):
        return False
    if bool(c.get("constraint")) and not bool(ref.get("constraint")):
        return False
    for key in ("initial_failed_steps", "solver_failure_steps", "final_failed_steps"):
        if int(c.get(key, 0)) > int(ref.get(key, 0)):
            return False
    return True


def finite_summary(xs: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    n = len(vals)
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"count": n, "min": vals[0], "median": median, "mean": float(math.fsum(vals) / n), "max": vals[-1]}


def analyze_episodes(episodes: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    spec_by_sid = {str(s["state_id"]): s for s in specs}
    by_state_mode: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    by_state_h: Dict[Tuple[str, int], Dict[str, Mapping[str, Any]]] = {}
    for e in episodes:
        sid = str(e["state_id"]); mode = str(e["terminal_mode"]); h = int(e["branch_horizon"])
        by_state_mode.setdefault((sid, mode), {})[h] = e
        by_state_h.setdefault((sid, h), {})[mode] = e
    rows: List[Dict[str, Any]] = []
    terminal_sensitivity: List[Dict[str, Any]] = []
    prefix_state_distances: List[float] = []
    positive_counts_by_terminal_mode: Dict[str, int] = {}
    for sid in sorted({str(e["state_id"]) for e in episodes}):
        ref = by_state_mode.get((sid, "per_h"), {}).get(PREFIX_H)
        if ref is None:
            raise ContractError(f"missing per_h H15 reference for {sid}")
        ref_state = state_tuple(ref.get("branch_previous_state"))
        for mode in sorted({m for (s, m) in by_state_mode if s == sid}):
            for h, e in sorted(by_state_mode[(sid, mode)].items()):
                dist = state_distance(state_tuple(e.get("branch_previous_state")), ref_state)
                prefix_state_distances.append(dist)
                gain_phys = float(ref["continuation_physical_constraint_cost_from_branch"] - e["continuation_physical_constraint_cost_from_branch"])
                gain_total = float(ref["continuation_total_cost_from_branch"] - e["continuation_total_cost_from_branch"])
                synthetic_gain_component = gain_total - gain_phys
                material_physical = bool(h != PREFIX_H and gain_phys >= MATERIAL_GAIN and no_regression(e, ref) and dist <= STATE_DISTANCE_TOL)
                material_total = bool(h != PREFIX_H and gain_total >= MATERIAL_GAIN and no_regression(e, ref) and dist <= STATE_DISTANCE_TOL)
                material = bool(material_physical or material_total)
                if material:
                    positive_counts_by_terminal_mode[mode] = positive_counts_by_terminal_mode.get(mode, 0) + 1
                bdiag = e.get("branch_step_mpc_diag") or {}
                rows.append({
                    "state_id": sid,
                    "target_index": int(e.get("target_index", spec_by_sid.get(sid, {}).get("target_index", -1))),
                    "state_role": spec_by_sid.get(sid, {}).get("state_role"),
                    "case": int(e["case"]),
                    "branch_step": int(e["branch_step"]),
                    "horizon": h,
                    "terminal_mode": mode,
                    "effective_terminal": e.get("effective_branch_terminal_label"),
                    "success": bool(e.get("success")),
                    "constraint": bool(e.get("constraint")),
                    "steps": int(e.get("steps", 0)),
                    "initial_failed_steps": int(e.get("initial_failed_steps", 0)),
                    "solver_failure_steps": int(e.get("solver_failure_steps", 0)),
                    "branch_state_distance_vs_per_h_H15": dist,
                    "continuation_physical": float(e["continuation_physical_constraint_cost_from_branch"]),
                    "continuation_total": float(e["continuation_total_cost_from_branch"]),
                    "synthetic_compute_from_branch": float(e["continuation_total_cost_from_branch"] - e["continuation_physical_constraint_cost_from_branch"]),
                    "gain_vs_per_h_H15_physical": gain_phys,
                    "gain_vs_per_h_H15_total": gain_total,
                    "gain_synthetic_component_vs_H15": synthetic_gain_component,
                    "material_physical": material_physical,
                    "material_total": material_total,
                    "material_any": material,
                    "no_safety_solver_regression_vs_H15": no_regression(e, ref),
                    "branch_mpc_value_fn": bdiag.get("info_mpc_value_fn"),
                    "branch_mpc_avg_stage_cost": bdiag.get("info_mpc_avg_stage_cost"),
                    "branch_objective_opt_f_num": bdiag.get("objective_opt_f_num"),
                    "decision_sum_s": (e.get("decision_timing_s") or {}).get("sum"),
                    "decision_plus_terminal_switch_sum_s": (e.get("decision_timing_s") or {}).get("sum"),
                    "solver_attempt_sum_s": (e.get("solver_attempt_timing_s") or {}).get("sum"),
                    "path": e.get("path"),
                })
    row_by_key = {(r["state_id"], int(r["horizon"]), r["terminal_mode"]): r for r in rows}
    for (sid, h), modes in sorted(by_state_h.items()):
        per = row_by_key.get((sid, h, "per_h"))
        if per is None:
            continue
        for mode in sorted(modes):
            if mode == "per_h":
                continue
            r = row_by_key[(sid, h, mode)]
            terminal_sensitivity.append({
                "state_id": sid,
                "target_index": r["target_index"],
                "state_role": r["state_role"],
                "horizon": h,
                "mode": mode,
                "physical_delta_vs_per_h": float(r["continuation_physical"] - per["continuation_physical"]),
                "total_delta_vs_per_h": float(r["continuation_total"] - per["continuation_total"]),
                "value_fn_delta_vs_per_h_at_branch": None if r["branch_mpc_value_fn"] is None or per["branch_mpc_value_fn"] is None else float(r["branch_mpc_value_fn"] - per["branch_mpc_value_fn"]),
                "objective_delta_vs_per_h_at_branch": None if r["branch_objective_opt_f_num"] is None or per["branch_objective_opt_f_num"] is None else float(r["branch_objective_opt_f_num"] - per["branch_objective_opt_f_num"]),
                "per_h_material_any": bool(per["material_any"]),
                "mode_material_any": bool(r["material_any"]),
                "material_label_flip": bool(per["material_any"]) != bool(r["material_any"]),
            })
    positive_state_ids = sorted({r["state_id"] for r in rows if r["state_role"] == "corrected_positive"})
    positive_per_h_material_states = sorted({r["state_id"] for r in rows if r["state_role"] == "corrected_positive" and r["terminal_mode"] == "per_h" and r["material_any"]})
    positive_states_robust_to_zero = []
    for sid in positive_state_ids:
        per_material = any(r["state_id"] == sid and r["terminal_mode"] == "per_h" and r["material_any"] for r in rows)
        zero_material = any(r["state_id"] == sid and r["terminal_mode"] == "zero_terminal" and r["material_any"] for r in rows)
        if per_material and zero_material:
            positive_states_robust_to_zero.append(sid)
    flips = [x for x in terminal_sensitivity if x["material_label_flip"]]
    major_terminal_delta = [x for x in terminal_sensitivity if abs(float(x["physical_delta_vs_per_h"])) >= MATERIAL_GAIN or abs(float(x["total_delta_vs_per_h"])) >= MATERIAL_GAIN]
    synthetic_only_material = [r for r in rows if r["material_total"] and not r["material_physical"]]
    return {
        "comparison_rows": rows,
        "terminal_sensitivity_rows": terminal_sensitivity,
        "terminal_material_label_flips": flips,
        "major_terminal_cost_deltas_abs_ge_material_threshold": major_terminal_delta,
        "positive_counts_by_terminal_mode": positive_counts_by_terminal_mode,
        "positive_state_ids": positive_state_ids,
        "positive_per_h_material_states": positive_per_h_material_states,
        "positive_states_robust_to_zero_terminal": positive_states_robust_to_zero,
        "synthetic_only_material_rows": synthetic_only_material,
        "prefix_branch_state_distance_summary": finite_summary(prefix_state_distances),
        "terminal_artifact_hypothesis_strength": "strong" if flips or len(major_terminal_delta) >= 2 else "weak_or_inconclusive",
        "decision": {
            "retrain_or_selector_refit_now": False,
            "reason": "Ablation is development-mined and diagnostic. Refit requires dense artifact-free labels and fresh confirmation; terminal/reward-source sensitivity must be interpreted before training.",
            "if_strong_terminal_artifact": "freeze terminal-source/matched-terminal label protocol or terminal-value refit ablation before any selector training",
            "if_weak_terminal_artifact": "scenario opportunity remains sparse; freeze next scenario/design probe or compact selector only if labels pass density gate on fresh development data",
        },
    }


def source_hashes(extra: Sequence[Path] = ()) -> Dict[str, str]:
    paths = [Path(__file__).resolve(), Path(base_smoke.__file__).resolve(), Path(stage1_runner.__file__).resolve(), V1B_RUN_RAW, V1B_RUN_DONE, PREFIX_POST_RAW, PREFIX_POST_DONE, STAGE1_BANK, STAGE1_PROTOCOL, STAGE1_DONE, V1B_PROTOCOL, PROTOCOL_JSON, PROTOCOL_MD] + list(extra)
    return {rel(p): sha256(p) for p in paths if p.exists()}


def write_protocol(created: str, specs: Sequence[Mapping[str, Any]], smoke_plan: Sequence[Mapping[str, Any]], full_plan: Sequence[Mapping[str, Any]]) -> None:
    protocol = {
        "protocol_id": f"vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_{STAMP}",
        "created_utc": created,
        "classification": "development_IMPROVED_terminal_reward_source_ablation_not_validation_not_final_test",
        "hypothesis": "Corrected v1b local gains may be caused by H-specific terminal-value/source/objective artifacts or synthetic total-cost effects rather than robust physical state-dependent horizon opportunity.",
        "input_label_artifact_correction": {"prefix_postdiagnostic": rel(PREFIX_POST_DONE), "corrected_positive_state_count_expected": 4, "corrected_positive_cases_expected": [4, 5]},
        "state_selection_rule": {
            "positives": "all corrected v1b positive states after dynamic-prefix repair",
            "controls": "fixed target_index list [8,9,16,18]: two same-stratum nonmaterial controls and two lower-stress controls, chosen before this ablation and not by favorable adaptive outcome",
        },
        "state_specs": specs,
        "candidate_horizon_rule": "For each state, compare H15 per-H reference with up to two non-H15 candidate horizons: best physical and best total v1b corrected horizons, supplemented by material horizons for positives when needed.",
        "terminal_modes": ["per_h", "h15_terminal", "h25_terminal", "zero_terminal"],
        "metrics": ["continuation_physical_constraint_cost_from_branch", "continuation_total_cost_from_branch", "synthetic_compute_from_branch", "success", "constraint", "solver_failure_steps", "decision_sum_s", "solver_attempt_sum_s", "mpc_value_fn", "objective_opt_f_num"],
        "analysis_rules": {
            "reference": "same state H15 branch with per-H H15 terminal",
            "prefix_identity_check": "branch_previous_state distance <= 1e-5 vs per-H H15 reference; dynamic prefix from v1b already repaired non-causal branch_horizon metadata",
            "material_gain_threshold": MATERIAL_GAIN,
            "terminal_artifact_evidence": "material-label flip or abs physical/total delta >= material threshold when changing terminal mode at identical state+horizon",
            "reward_artifact_evidence": "material total gain without material physical gain, or ranking dominated by synthetic_compute_from_branch rather than physical cost",
            "decision": "No selector/refit from this ablation alone; use it to choose terminal refit/matched-terminal labeling vs scenario redesign/training only after dense labels.",
        },
        "budgets": {
            "smoke": {"episodes_exact": len(smoke_plan), "control_step_upper_bound": len(smoke_plan) * MAX_STEPS, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
            "full": {"episodes_exact": len(full_plan), "control_step_upper_bound": len(full_plan) * MAX_STEPS, "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        },
        "smoke_plan": smoke_plan,
        "full_plan": full_plan,
    }
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle stress-v1b terminal/reward-source ablation v0",
        "",
        f"Frozen UTC: `{created}`. Development-only; no validation64-bank or sealed-test access; no training/refit.",
        "",
        "## Hypothesis",
        "",
        protocol["hypothesis"],
        "",
        "## State selection",
        "",
        "- Positives: all corrected v1b positive states after dynamic-prefix repair.",
        "- Controls: fixed target_index `[8, 9, 16, 18]` (two same-stratum nonmaterial controls and two lower-stress controls).",
        "",
        "## State specs",
        "",
        "| target | state_id | case | branch step | role | corrected label | candidate H | material H from v1b |",
        "|---:|---|---:|---:|---|---|---|---|",
    ]
    for s in specs:
        lines.append("| %d | `%s` | %d | %d | `%s` | `%s` | `%s` | `%s` |" % (int(s["target_index"]), s["state_id"], int(s["case"]), int(s["branch_step"]), s["state_role"], s["corrected_label"], s["candidate_horizons"], s["material_horizons_from_v1b"]))
    lines += [
        "",
        "## Budgets",
        "",
        f"- Smoke: `{len(smoke_plan)}` episodes, control-step cap `{len(smoke_plan) * MAX_STEPS}`.",
        f"- Full: `{len(full_plan)}` episodes, control-step cap `{len(full_plan) * MAX_STEPS}`.",
        "- Both modes use zero new training episodes and zero gradient steps.",
        "",
        "## Smoke plan",
        "",
        "| exec | target | state_id | H | terminal mode |",
        "|---:|---:|---|---:|---|",
    ]
    for item in smoke_plan:
        lines.append("| %d | %d | `%s` | %d | `%s` |" % (int(item["execution_index"]), int(item["target_index"]), item["state_id"], int(item["horizon"]), item["terminal_mode"]))
    lines += ["", "## Full plan", "", "| exec | target | state_id | H | terminal mode |", "|---:|---:|---|---:|---|"]
    for item in full_plan:
        lines.append("| %d | %d | `%s` | %d | `%s` |" % (int(item["execution_index"]), int(item["target_index"]), item["state_id"], int(item["horizon"]), item["terminal_mode"]))
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if marker not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def write_dryrun_summary(raw: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle stress-v1b terminal/reward-source ablation v0 dry-run",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations; no training/refit; validation64 and sealed test stayed closed.",
        "",
        "## Frozen design",
        "",
        f"- Input corrected v1b positives: `{raw['input_corrected_headline']}`.",
        f"- State count: `{len(raw['state_specs'])}`; smoke episodes `{len(raw['smoke_plan'])}`; full episodes `{len(raw['full_plan'])}`.",
        f"- Protocol: `{rel(PROTOCOL_JSON)}`, `{rel(PROTOCOL_MD)}`.",
        "- Simulation modes remain blocked until a verified external backup postdates this dry-run/source.",
        "",
        "## State specs",
        "",
        "| target | case | role | candidate H | material H |",
        "|---:|---:|---|---|---|",
    ]
    for s in raw["state_specs"]:
        lines.append("| %d | %d | `%s` | `%s` | `%s` |" % (int(s["target_index"]), int(s["case"]), s["state_role"], s["candidate_horizons"], s["material_horizons_from_v1b"]))
    lines += ["", f"Backup request before smoke: `{raw['backup_request_before_smoke']}`."]
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    (DRYRUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry_run() -> int:
    if (DRYRUN_DIR / "completed.json").exists():
        completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"already_completed": rel(DRYRUN_DIR / "completed.json"), "sealed_test_accessed": False, "validation64_bank_opened": False}, sort_keys=True))
        return 0
    run_done, prefix_done, prefix_raw, v1b_raw = verify_inputs(check_prefix_hashes=True)
    specs = make_state_specs(prefix_raw)
    smoke_plan = build_plan(specs, SMOKE_TARGET_INDICES, full=False)
    full_plan = build_plan(specs, None, full=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    write_protocol(created, specs, smoke_plan, full_plan)
    write_json(REQUEST_BACKUP_BEFORE_SMOKE, {
        "requested_utc": created,
        "reason": "backup prefix postdiagnostic and frozen v1b terminal/reward-source ablation dry-run before any smoke rollouts",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "planned_smoke_episodes": len(smoke_plan),
        "planned_smoke_control_step_upper_bound": len(smoke_plan) * MAX_STEPS,
        "planned_full_episodes": len(full_plan),
        "planned_full_control_step_upper_bound": len(full_plan) * MAX_STEPS,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(Path(__file__).resolve()), rel(DRYRUN_DIR), rel(STATE_DRYRUN), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(PREFIX_POST), rel(V1B_RUN), rel(REQUEST_BACKUP_BEFORE_SMOKE)],
    })
    corr = (prefix_raw.get("analysis") or {}).get("corrected_label_headline") or {}
    raw = {
        "created_utc": created,
        "method": "vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_no_simulation",
        "classification": "development_protocol_freeze_no_simulation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {"v1b_completed": rel(V1B_RUN_DONE), "v1b_completed_sha256": sha256(V1B_RUN_DONE), "prefix_completed": rel(PREFIX_POST_DONE), "prefix_completed_sha256": sha256(PREFIX_POST_DONE), "stage1_bank": rel(STAGE1_BANK), "stage1_bank_sha256": sha256(STAGE1_BANK)},
        "input_corrected_headline": corr,
        "state_specs": specs,
        "smoke_plan": smoke_plan,
        "full_plan": full_plan,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD)},
        "budget_declared": {"smoke_episodes": len(smoke_plan), "smoke_control_step_upper_bound": len(smoke_plan) * MAX_STEPS, "full_episodes": len(full_plan), "full_control_step_upper_bound": len(full_plan) * MAX_STEPS, "new_training_episodes": 0, "new_gradient_steps": 0},
        "source_hashes": source_hashes(extra=[REQUEST_BACKUP_BEFORE_SMOKE]),
        "backup_request_before_smoke": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "next": "await verified external backup, then run --run-smoke with the backup proof; do not run simulations before backup",
    }
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DRYRUN_DIR / "raw.json", raw)
    write_dryrun_summary(raw)
    STATE_DRYRUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_DRYRUN.write_text(
        f"# Vehicle stress-v1b terminal/reward ablation v0 dry-run\n\nUTC: {created}. No simulations/training/validation/test. Frozen smoke plan={len(smoke_plan)} episodes, full plan={len(full_plan)} episodes. Corrected v1b positives remain {corr.get('positive_state_count')} states across cases={corr.get('positive_cases')}; label gate={corr.get('training_refit_label_gate_pass_development_only_corrected_prefix')}. Backup required before any smoke rollout: {rel(REQUEST_BACKUP_BEFORE_SMOKE)}.\n",
        encoding="utf-8",
    )
    block = f"""<!-- {MARKER_DRYRUN} -->
## 2026-09-28 vehicle stress-v1b terminal/reward-source ablation v0 dry-run

UTC: {created}. No-simulation protocol freeze completed after prefix-artifact postdiagnostic. Corrected v1b positives remain {corr.get('positive_state_count')} states across cases={corr.get('positive_cases')}, label gate={corr.get('training_refit_label_gate_pass_development_only_corrected_prefix')}. Froze terminal/reward-source ablation over all four corrected positives plus deterministic controls [8,9,16,18]; smoke plan={len(smoke_plan)} episodes, full plan={len(full_plan)} episodes. No validation64/test access, no training/refit. Simulation remains blocked until verified external backup covers `{rel(REQUEST_BACKUP_BEFORE_SMOKE)}`, `{rel(PROTOCOL_JSON)}`, dry-run outputs and source. Artifacts: `{rel(DRYRUN_DIR / 'summary.md')}`, `{rel(DRYRUN_DIR / 'raw.json')}`, `{rel(DRYRUN_DIR / 'completed.json')}`.
"""
    append_docs(block, MARKER_DRYRUN)
    files = [p for p in DRYRUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_DRYRUN, REQUEST_BACKUP_BEFORE_SMOKE, Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, V1B_RUN_DONE, PREFIX_POST_DONE, STAGE1_BANK]
    write_json(DRYRUN_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "state_count": len(specs),
        "smoke_episodes": len(smoke_plan),
        "full_episodes": len(full_plan),
        "backup_required_before_smoke": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE),
        "next_after_backup": "run --run-smoke with verified proof postdating this dry-run/source",
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(DRYRUN_DIR / "completed.json"), "summary": rel(DRYRUN_DIR / "summary.md"), "state_count": len(specs), "smoke_episodes": len(smoke_plan), "full_episodes": len(full_plan), "backup_request": rel(REQUEST_BACKUP_BEFORE_SMOKE), "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0}, sort_keys=True), flush=True)
    return 0


def write_run_summary(out_dir: Path, raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        f"# Vehicle stress-v1b terminal/reward-source ablation v0 {raw['run_kind']}",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only; no training/refit, no validation64 bank, no sealed test.",
        "",
        f"Budget: `{raw['budget_actual']['episodes']}` episodes / `{raw['budget_declared']['episodes_exact']}`; `{raw['budget_actual']['control_steps']}` control steps / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        "",
        "## Headline",
        "",
        f"- Positive counts by terminal mode: `{a['positive_counts_by_terminal_mode']}`.",
        f"- Positive per-H material states: `{a['positive_per_h_material_states']}`.",
        f"- Positive states robust to zero terminal: `{a['positive_states_robust_to_zero_terminal']}`.",
        f"- Terminal material-label flips: `{len(a['terminal_material_label_flips'])}`; major terminal deltas: `{len(a['major_terminal_cost_deltas_abs_ge_material_threshold'])}`.",
        f"- Synthetic-only material rows: `{len(a['synthetic_only_material_rows'])}`.",
        f"- Branch-state distance summary: `{a['prefix_branch_state_distance_summary']}`.",
        f"- Terminal-artifact hypothesis strength: `{a['terminal_artifact_hypothesis_strength']}`.",
        "",
        "## Comparison rows",
        "",
        "| state | target | role | H | terminal mode | success | phys gain | total gain | synthetic gain component | material | value_fn | objective |",
        "|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in a["comparison_rows"]:
        lines.append("| `%s` | %d | `%s` | %d | `%s` | `%s` | %.6g | %.6g | %.6g | `%s` | %s | %s |" % (
            r["state_id"], int(r["target_index"]), r["state_role"], int(r["horizon"]), r["terminal_mode"], r["success"], float(r["gain_vs_per_h_H15_physical"]), float(r["gain_vs_per_h_H15_total"]), float(r["gain_synthetic_component_vs_H15"]), r["material_any"], "NA" if r["branch_mpc_value_fn"] is None else "%.6g" % float(r["branch_mpc_value_fn"]), "NA" if r["branch_objective_opt_f_num"] is None else "%.6g" % float(r["branch_objective_opt_f_num"])
        ))
    lines += ["", "## Terminal sensitivity versus per-H terminal", "", "| state | H | mode | phys delta | total delta | value delta | objective delta | flip |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in a["terminal_sensitivity_rows"]:
        lines.append("| `%s` | %d | `%s` | %.6g | %.6g | %s | %s | `%s` |" % (
            r["state_id"], int(r["horizon"]), r["mode"], float(r["physical_delta_vs_per_h"]), float(r["total_delta_vs_per_h"]), "NA" if r["value_fn_delta_vs_per_h_at_branch"] is None else "%.6g" % float(r["value_fn_delta_vs_per_h_at_branch"]), "NA" if r["objective_delta_vs_per_h_at_branch"] is None else "%.6g" % float(r["objective_delta_vs_per_h_at_branch"]), r["material_label_flip"]
        ))
    lines += ["", "## Decision", "", f"- Retrain/refit now: `{a['decision']['retrain_or_selector_refit_now']}`.", f"- Reason: {a['decision']['reason']}", f"- If strong terminal artifact: {a['decision']['if_strong_terminal_artifact']}", f"- If weak terminal artifact: {a['decision']['if_weak_terminal_artifact']}", "", f"Backup request: `{raw['backup_request_after_run']}`."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(kind: str, backup_proof: Path) -> int:
    out_dir = SMOKE_DIR if kind == "smoke" else FULL_DIR
    state_path = STATE_SMOKE if kind == "smoke" else STATE_FULL
    if not (DRYRUN_DIR / "completed.json").exists():
        raise ContractError("run requires completed dry-run marker")
    dry_done = completed_passed(DRYRUN_DIR / "completed.json", check_hashes=True)
    if (out_dir / "completed.json").exists():
        completed_passed(out_dir / "completed.json", check_hashes=True)
        raise SystemExit(f"{kind} ablation already completed; refusing rerun")
    if out_dir.exists() and any(p.name != "run.lock" for p in out_dir.iterdir()):
        raise ContractError(f"partial output exists; inspect before rerun: {rel(out_dir)}")
    run_done, prefix_done, prefix_raw, v1b_raw = verify_inputs(check_prefix_hashes=True)
    min_time = max(parse_time(dry_done.get("created_utc")) or dt.datetime.now(dt.timezone.utc), source_mtime_utc())
    backup = verify_backup_proof(backup_proof, min_time, f"v1b_terminal_reward_ablation_{kind}")
    protocol = read_json(PROTOCOL_JSON)
    specs = list(protocol["state_specs"])
    plan = list(protocol["smoke_plan"] if kind == "smoke" else protocol["full_plan"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out_dir / "run_started.json", {"started_utc": started, "pid": os.getpid(), "run_kind": kind, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_training_episodes": 0, "new_gradient_steps": 0})
    preflight = stage1_runner.runtime_preflight()
    write_json(out_dir / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(f"runtime preflight failed: {preflight}")
    stage1_runner.base.v1.latency_verify()
    bank = read_json(STAGE1_BANK)
    selected_cases = bank["selected_cases"]
    stage1_protocol = read_json(STAGE1_PROTOCOL)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid(stage1_protocol["terminal_grid_readiness_reused_from_v1"])
    write_json(out_dir / "terminal_sources.json", {str(k): v for k, v in terminal_receipts.items()})
    # Reuse the already instrumented branch runner from the previous terminal
    # smoke, but route all outputs to this v1b ablation folder.
    base_smoke.OUT = out_dir
    base_smoke.PREFIX_H = PREFIX_H
    base_smoke.MAX_STEPS = MAX_STEPS
    base_smoke.MATERIAL_GAIN = MATERIAL_GAIN
    write_json(out_dir / "schedule.json", {"episodes": plan, "prefix_horizon": PREFIX_H, "max_steps": MAX_STEPS, "run_kind": kind})
    episodes: List[Dict[str, Any]] = []
    for item in plan:
        summary = base_smoke.run_one(item, selected_cases[int(item["case"])], terminals, terminal_receipts)
        summary.update({"target_index": int(item["target_index"])})
        # base_smoke already wrote summary.json before this update; rewrite the
        # per-episode summary so target_index is preserved in both places.
        write_json(ROOT / summary["path"] / "summary.json", summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "run_kind": kind, "episodes_done": len(episodes), "episodes_expected": len(plan), "control_steps_done": int(sum(int(e.get("steps", 0)) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "state_id", "target_index", "case", "branch_step", "branch_horizon", "terminal_mode", "steps", "success", "termination")}, "historical_validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(out_dir / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    cap = len(plan) * MAX_STEPS
    if len(episodes) != len(plan) or control_steps > cap:
        raise ContractError("ablation budget violation")
    analysis = analyze_episodes(episodes, specs)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0_%s_%s.json" % (kind.upper(), created.replace("-", "").replace(":", "").replace("+00:00", "+0000")))
    write_json(req, {"requested_utc": created, "reason": f"backup v1b terminal/reward-source ablation {kind} outputs before further simulation/training/refit", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "artifacts": [rel(out_dir), rel(state_path), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(req)]})
    raw = {
        "created_utc": created,
        "started_utc": started,
        "method": "vehicle_stress_v1b_terminal_reward_ablation_v0",
        "run_kind": kind,
        "classification": "development_IMPROVED_terminal_reward_source_ablation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "input_backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD)},
        "inputs": {"v1b_raw": rel(V1B_RUN_RAW), "v1b_raw_sha256": sha256(V1B_RUN_RAW), "prefix_raw": rel(PREFIX_POST_RAW), "prefix_raw_sha256": sha256(PREFIX_POST_RAW), "stage1_bank": rel(STAGE1_BANK), "stage1_bank_sha256": sha256(STAGE1_BANK)},
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"episodes_exact": len(plan), "control_step_upper_bound": cap, "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "terminal_sources": terminal_receipts,
        "schedule": plan,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": ["development diagnostic only", "states selected from development-mined corrected v1b labels", "not validation/model selection", "not final test", "not enough by itself for selector/refit", "total cost includes synthetic horizon penalty and is not measured runtime"],
        "backup_request_after_run": rel(req),
    }
    write_json(out_dir / "raw.json", raw)
    write_run_summary(out_dir, raw)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"# Vehicle stress-v1b terminal/reward-source ablation v0 {kind}\n\nUTC: {created}. Completed {len(episodes)} episodes / {control_steps} control steps. No validation64/test/training. Terminal artifact strength={analysis['terminal_artifact_hypothesis_strength']}; flips={len(analysis['terminal_material_label_flips'])}; synthetic-only material rows={len(analysis['synthetic_only_material_rows'])}. Backup required before further simulations: {rel(req)}.\n", encoding="utf-8")
    block = f"""<!-- {MARKER_RUN}-{kind} -->
## 2026-09-28 vehicle stress-v1b terminal/reward-source ablation v0 {kind}

UTC: {created}. Development-only ablation completed: {len(episodes)} episodes, {control_steps} control steps. No validation64/test access and no training/refit. Terminal-artifact strength={analysis['terminal_artifact_hypothesis_strength']}; material-label flips={len(analysis['terminal_material_label_flips'])}; synthetic-only material rows={len(analysis['synthetic_only_material_rows'])}. Decision remains no selector/refit from this diagnostic alone. Artifacts: `{rel(out_dir / 'summary.md')}`, `{rel(out_dir / 'raw.json')}`, `{rel(out_dir / 'completed.json')}`. Backup request: `{rel(req)}`.
"""
    append_docs(block, MARKER_RUN + "-" + kind)
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "completed.json"] + [state_path, req, Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, backup_proof]
    write_json(out_dir / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(req), "headline": {"run_kind": kind, "positive_counts_by_terminal_mode": analysis["positive_counts_by_terminal_mode"], "terminal_artifact_hypothesis_strength": analysis["terminal_artifact_hypothesis_strength"], "terminal_material_label_flips": len(analysis["terminal_material_label_flips"]), "synthetic_only_material_rows": len(analysis["synthetic_only_material_rows"]), "next_action": "inspect sensitivity; then freeze terminal-source/refit or scenario-design follow-up"}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(out_dir / "completed.json"), "summary": rel(out_dir / "summary.md"), "run_kind": kind, "episodes": len(episodes), "control_steps": control_steps, "terminal_artifact_hypothesis_strength": analysis["terminal_artifact_hypothesis_strength"], "terminal_material_label_flips": len(analysis["terminal_material_label_flips"]), "synthetic_only_material_rows": len(analysis["synthetic_only_material_rows"]), "backup_request": rel(req), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run-smoke", action="store_true")
    group.add_argument("--run-full", action="store_true")
    ap.add_argument("--input-backup-proof", type=Path)
    ap.add_argument("--i-accept-development-terminal-reward-ablation", action="store_true", required=True)
    args = ap.parse_args(argv)
    if args.dry_run:
        return run_dry_run()
    if args.input_backup_proof is None:
        raise ContractError("--input-backup-proof is required for simulation runs")
    return run_ablation("smoke" if args.run_smoke else "full", args.input_backup_proof)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = DRYRUN_DIR
        if "--run-smoke" in sys.argv:
            target = SMOKE_DIR
        elif "--run-full" in sys.argv:
            target = FULL_DIR
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {"failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exception": repr(exc), "traceback": traceback.format_exc(), "historical_validation64_bank_opened": False, "validation64_bank_opened": False, "sealed_test_accessed": False, "sealed_test_bank_opened": False, "new_training_episodes": 0, "new_gradient_steps": 0, "next_recovery_hint": "Preserve failure. If dry-run failed, repair schema/source with a new version; if run failed after episodes, audit partial outputs before rerun."})
        raise
