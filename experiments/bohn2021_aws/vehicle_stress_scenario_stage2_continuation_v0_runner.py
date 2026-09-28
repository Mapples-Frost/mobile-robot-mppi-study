#!/usr/bin/env python3
"""Vehicle stress-scenario Stage2 identical-state continuation runner v0.

Purpose
-------
Development-only IMPROVED diagnostic following the frozen Stage1 stress-scenario
fixed-H map.  Stage1 found material episode-level opportunity concentrated in
case 5, but episode-level fixed-H oracle labels do not prove useful
within-episode adaptive switching.  This script first freezes a Stage2 protocol
and target list from H15 reference traces only, then (after a new external backup
covers this source + prepare artifacts) can run matched H15-prefix continuations
with branch horizons [10, 15, 25, 30, 35, 45].

Access policy
-------------
No historical validation64 bank access, no sealed final-test access, no training
or gradient updates.  --prepare-only performs no simulations.  --run-stage2 is
refused unless a verified backup proof postdates the prepare/source artifacts.
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

TASK = "vehicle"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45]
MAX_STEPS = 150
ORDER_SEED = 2609289001
MATERIAL_GAIN_THRESHOLD = 3.0
PROTOCOL_STAMP = "20260928T1740Z"
MARKER_PREPARE = "vehicle-stress-scenario-stage2-continuation-v0-prepare-20260928T1740Z"
MARKER_RUN = "vehicle-stress-scenario-stage2-continuation-v0-run-20260928T1740Z"

STAGE1_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928"
STAGE1_RAW = STAGE1_DIR / "raw.json"
STAGE1_COMPLETED = STAGE1_DIR / "completed.json"
STAGE1_BANK = STAGE1_DIR / "bank/vehicle_stress_scenario_opportunity_probe_v0_bank.json"
STAGE1_POST_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z"
STAGE1_POST_COMPLETED = STAGE1_POST_DIR / "completed.json"
STAGE1_POST_RAW = STAGE1_POST_DIR / "raw.json"
STAGE1_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
STAGE1_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.md"

PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0_frozen_{PROTOCOL_STAMP}.json"
PROTOCOL_MD = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0_frozen_{PROTOCOL_STAMP}.md"
PREPARE_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0_prepare_{PROTOCOL_STAMP}"
RUN_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0_{PROTOCOL_STAMP}"
STATE_PREPARE = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_stage2_continuation_v0_prepare_{PROTOCOL_STAMP}.md"
STATE_RUN = ROOT / f"research_artifacts/aws_state/vehicle_stress_scenario_stage2_continuation_v0_run_{PROTOCOL_STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SUPERVISOR_PROOF_PATH = BACKUP_DIR / "backup_proof_20260928T173312_from_supervisor_context_after_stage1_postdiagnostic.json"
REQUEST_BACKUP_BEFORE_RUN = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_SCENARIO_STAGE2_CONTINUATION_V0_ROLLOUT_{PROTOCOL_STAMP}.json"

SUPERVISOR_BACKUP_AFTER_STAGE1_POSTDIAG = {
    "source": "user_supplied_current_supervisor_context",
    "note": "Persisted from supervisor context. It verifies an external backup after Stage1 stress-scenario rollout/postdiagnostic and before Stage2 prepare. The local proof copy and Stage2 prepare artifacts require the next backup before Stage2 rollouts.",
    "time": "2026-09-28T17:33:12.799853+00:00",
    "status": "verified",
    "backup_verified": True,
    "remaining_changed_files": 0,
    "commit": "40ea0bec9590942b49c6e36978f7fd940bbf2bd4",
    "changed_files": 23,
    "packages_this_run": [
        {
            "name": "20260928T173310_d31190f1.tar.gz",
            "url": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-aws-evidence-20260926/20260928T173310_d31190f1.tar.gz",
            "id": 595990186,
            "sha256": "5b82432a47c0d534b4b24490600a21bfa2d7b606f1067dc2c7ec438678ca850b",
            "bytes": 6812281,
            "verification": "github_server_sha256",
        }
    ],
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
    "tracked_files": 126508,
}


class ContractError(RuntimeError):
    pass


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
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=serial) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=serial).encode("utf-8")).hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def source_mtime_utc() -> dt.datetime:
    return dt.datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime, dt.timezone.utc)


def source_hashes(extra: Sequence[Path] = ()) -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(), STAGE1_RAW, STAGE1_COMPLETED, STAGE1_POST_COMPLETED,
        STAGE1_POST_RAW, STAGE1_BANK, STAGE1_PROTOCOL_JSON, STAGE1_PROTOCOL_MD,
        PROTOCOL_JSON, PROTOCOL_MD, SUPERVISOR_PROOF_PATH,
    ] + list(extra)
    return {rel(p): sha256(p) for p in paths if p.exists()}


def verify_completed_basic(path: Path, check_hashes: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: %s" % rel(path))
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("completed marker did not pass: %s" % rel(path))
    if check_hashes:
        for name, expected in (done.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists() or sha256(p) != expected:
                raise ContractError("completed marker hash mismatch: %s" % name)
    return done


def verify_stage1_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (STAGE1_RAW, STAGE1_COMPLETED, STAGE1_BANK, STAGE1_POST_COMPLETED, STAGE1_POST_RAW, STAGE1_PROTOCOL_JSON, STAGE1_PROTOCOL_MD):
        if not p.exists():
            raise ContractError("required Stage1 input missing: %s" % rel(p))
    stage1_done = verify_completed_basic(STAGE1_COMPLETED, check_hashes=False)
    post_done = verify_completed_basic(STAGE1_POST_COMPLETED, check_hashes=False)
    if stage1_done.get("hard_pass") is not True or post_done.get("hard_pass") is not True:
        raise ContractError("Stage1 or postdiagnostic did not hard-pass")
    for label, obj in (("stage1_done", stage1_done), ("post_done", post_done)):
        if obj.get("historical_validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise ContractError(label + " access flags invalid")
    if post_done.get("stage2_identical_state_continuation_should_be_frozen_after_backup") is not True:
        raise ContractError("Stage1 postdiagnostic did not authorize freezing Stage2")
    # Check key hashes without walking the huge episode tree.
    expected_raw = (stage1_done.get("hashes") or {}).get(rel(STAGE1_RAW))
    if expected_raw and sha256(STAGE1_RAW) != expected_raw:
        raise ContractError("Stage1 raw hash mismatch against completed marker")
    expected_post_raw = (post_done.get("hashes") or {}).get(rel(STAGE1_POST_RAW))
    if expected_post_raw and sha256(STAGE1_POST_RAW) != expected_post_raw:
        raise ContractError("Stage1 postdiagnostic raw hash mismatch against completed marker")
    return read_json(STAGE1_RAW), stage1_done, read_json(STAGE1_POST_RAW), post_done


def persist_and_verify_supervisor_backup(stage1_post_done: Mapping[str, Any]) -> Dict[str, Any]:
    write_json(SUPERVISOR_PROOF_PATH, SUPERVISOR_BACKUP_AFTER_STAGE1_POSTDIAG)
    proof = read_json(SUPERVISOR_PROOF_PATH)
    verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
    if not verified:
        raise ContractError("supervisor backup proof is not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("supervisor backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("supervisor backup proof lacks commit")
    packages = proof.get("packages_this_run") or []
    if not packages or not all(p.get("sha256") and p.get("verification") and p.get("bytes") for p in packages):
        raise ContractError("supervisor backup proof lacks verified package SHA information")
    proof_time = parse_time(proof.get("time"))
    post_time = parse_time(stage1_post_done.get("created_utc"))
    if proof_time is None or post_time is None or proof_time < post_time:
        raise ContractError("supervisor backup proof does not postdate Stage1 postdiagnostic")
    return {
        "path": rel(SUPERVISOR_PROOF_PATH),
        "sha256": sha256(SUPERVISOR_PROOF_PATH),
        "time": proof_time.isoformat(),
        "commit": proof.get("commit"),
        "remaining_changed_files": proof.get("remaining_changed_files"),
        "packages_this_run": packages,
        "postdates_stage1_postdiagnostic": True,
        "note": "This proof covers Stage1 rollout/postdiagnostic, not the new Stage2 prepare/source artifacts; another backup is required before --run-stage2.",
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = list(float(x) for x in values if math.isfinite(float(x)))
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    xs_sorted = sorted(xs)
    mid = len(xs_sorted) // 2
    median = xs_sorted[mid] if len(xs_sorted) % 2 else 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])
    return {"count": len(xs_sorted), "min": xs_sorted[0], "median": median, "mean": sum(xs_sorted) / len(xs_sorted), "max": xs_sorted[-1]}


def trace_path_for(stage1_raw: Mapping[str, Any], case_id: int, horizon: int) -> Path:
    rows = [e for e in stage1_raw.get("episodes", []) if int(e.get("case", -1)) == case_id and int(e.get("horizon", -1)) == horizon]
    if len(rows) != 1:
        raise ContractError("expected exactly one Stage1 episode summary for case %d H%d, got %d" % (case_id, horizon, len(rows)))
    path = ROOT / rows[0]["path"] / "trace.json"
    if not path.exists():
        raise ContractError("missing Stage1 trace: %s" % rel(path))
    return path


def score_trace_row(row: Mapping[str, Any], step: int, trace_len: int) -> Dict[str, Any]:
    obs = row.get("observation") or []
    omega = 0.0
    try:
        omega = abs(float(((row.get("input") or {}).get("u_omega") or [0.0])[0]))
    except Exception:
        omega = 0.0
    heading_proxy = abs(safe_float(obs[2], 0.0)) if len(obs) > 2 else 0.0
    perf = safe_float(row.get("performance"), 0.0)
    solver_iter = 0.0
    attempts = (row.get("recovery") or {}).get("attempts") or []
    if attempts:
        solver_iter = max(safe_float(a.get("iterations"), 0.0) for a in attempts)
    # A deterministic, H15-only transient proxy: high turn effort/heading error,
    # step cost and solver iteration are emphasized; central episode states are
    # modestly favored to avoid only near-terminal labels.
    phase = float(step) / float(max(trace_len - 1, 1))
    central = 1.0 - min(1.0, abs(phase - 0.5) / 0.5)
    score = 1.5 * omega + 1.0 * heading_proxy + 25.0 * perf + 0.01 * solver_iter + 0.2 * central
    return {
        "transient_score_from_H15_only": float(score),
        "turn_effort_abs_u_omega": float(omega),
        "heading_proxy_abs_obs2": float(heading_proxy),
        "performance_step_cost": float(perf),
        "solver_iteration_proxy": float(solver_iter),
        "phase_fraction": float(phase),
    }


def choose_step_from_range(trace: Sequence[Mapping[str, Any]], lo: int, hi: int) -> Tuple[int, Dict[str, Any]]:
    max_step = max(0, min(int(hi), len(trace) - 2, 90))
    min_step = max(1, min(int(lo), max_step))
    if min_step > max_step:
        min_step = max_step
    candidates = []
    for s in range(min_step, max_step + 1):
        features = score_trace_row(trace[s], s, len(trace))
        candidates.append((features["transient_score_from_H15_only"], -abs(features["phase_fraction"] - 0.5), -s, s, features))
    if not candidates:
        raise ContractError("no candidate branch step in requested range")
    _, _, _, step, features = max(candidates)
    return int(step), features


def build_targets(stage1_raw: Mapping[str, Any], stage1_post_raw: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bank = read_json(STAGE1_BANK)
    selected_meta = bank["selection"]["selected_metadata"]
    material_cases = set(int(x) for x in (stage1_post_raw.get("material_non_H15_positive_cases") or [5]))
    # Conditional development design: case 5 supplies localized opportunity states;
    # nonmaterial stress and control cases guard against overfitting to a single case.
    case_plan = [
        {"case": 5, "role": "material_stage1_case5_multi_phase", "ranges": [(8, 18), (19, 32), (33, 48), (49, 64), (65, 89)]},
        {"case": 0, "role": "nonmaterial_stress_control", "ranges": [(8, 89)]},
        {"case": 4, "role": "nonmaterial_stress_control", "ranges": [(8, 89)]},
        {"case": 7, "role": "nonmaterial_stress_control", "ranges": [(8, 89)]},
        {"case": 8, "role": "lower_stress_control", "ranges": [(8, 89)]},
        {"case": 9, "role": "lower_stress_control", "ranges": [(8, 89)]},
        {"case": 10, "role": "lower_stress_control", "ranges": [(8, 89)]},
        {"case": 11, "role": "lower_stress_control", "ranges": [(8, 89)]},
    ]
    targets: List[Dict[str, Any]] = []
    seen: set = set()
    for plan in case_plan:
        case_id = int(plan["case"])
        h15_trace_path = trace_path_for(stage1_raw, case_id, PREFIX_H)
        h15_trace = read_json(h15_trace_path)
        trace_hash = sha256(h15_trace_path)
        for r_index, (lo, hi) in enumerate(plan["ranges"]):
            step, features = choose_step_from_range(h15_trace, lo, hi)
            # If a single case/range collision occurs, move to the best available adjacent unused step.
            if (case_id, step) in seen:
                candidates = []
                for s in range(max(1, lo), min(hi, len(h15_trace) - 2, 90) + 1):
                    if (case_id, s) in seen:
                        continue
                    f = score_trace_row(h15_trace[s], s, len(h15_trace))
                    candidates.append((f["transient_score_from_H15_only"], -abs(f["phase_fraction"] - 0.5), -s, s, f))
                if not candidates:
                    continue
                _, _, _, step, features = max(candidates)
            seen.add((case_id, step))
            row = h15_trace[step]
            meta = selected_meta[case_id]
            targets.append({
                "target_index": len(targets),
                "case": case_id,
                "source_candidate_index": int(meta.get("candidate_index", -1)),
                "case_group": meta.get("selection_group"),
                "selection_role": plan["role"],
                "stage1_material_positive_case": bool(case_id in material_cases),
                "branch_step": int(step),
                "phase_bucket_index": int(r_index),
                "range_requested": [int(lo), int(hi)],
                "branch_horizons": list(BRANCH_HORIZONS),
                "prefix_horizon": PREFIX_H,
                "h15_trace_path": rel(h15_trace_path),
                "h15_trace_sha256": trace_hash,
                "h15_trace_length": len(h15_trace),
                "h15_reference_previous_state": row.get("previous_state"),
                "h15_reference_observation": row.get("observation"),
                "h15_reference_step_costs": {"performance": row.get("performance"), "constraint": row.get("constraint"), "compute": row.get("compute"), "reward": row.get("reward")},
                "h15_only_selection_features": features,
                "case_metadata": {k: meta.get(k) for k in ("theta_r", "abs_theta_r", "traj_steps", "min_reference_obstacle_clearance", "stress_flags_count", "stress_score")},
            })
    if len(targets) != 12:
        raise ContractError("Stage2 target count must be exactly 12, got %d" % len(targets))
    if len({(int(t["case"]), int(t["branch_step"])) for t in targets}) != len(targets):
        raise ContractError("Stage2 targets are not unique by case/branch_step")
    role_counts: Dict[str, int] = {}
    for t in targets:
        role_counts[str(t["selection_role"])] = role_counts.get(str(t["selection_role"]), 0) + 1
    diagnostics = {
        "target_count": len(targets),
        "episode_budget": len(targets) * len(BRANCH_HORIZONS),
        "control_step_upper_bound": len(targets) * len(BRANCH_HORIZONS) * MAX_STEPS,
        "case_counts": {str(c): sum(1 for t in targets if int(t["case"]) == c) for c in sorted(set(int(t["case"]) for t in targets))},
        "role_counts": role_counts,
        "feature_score_summary": values_summary(t["h15_only_selection_features"]["transient_score_from_H15_only"] for t in targets),
        "branch_step_summary": values_summary(int(t["branch_step"]) for t in targets),
        "selection_rule": "Targets selected after Stage1 materiality decision but before Stage2 branch rollouts: 5 H15-only transient/phase states from material case 5, plus one H15-only high-transient state from three nonmaterial stress cases and four lower-stress controls. Non-H15 branch outcomes are not used for within-case branch-step selection.",
    }
    return targets, diagnostics


def write_protocol_and_summary(raw: Mapping[str, Any]) -> None:
    protocol = raw["protocol_full"]
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle stress-scenario Stage2 identical-state continuation v0 protocol",
        "",
        f"Frozen UTC: `{raw['created_utc']}`. Development-only IMPROVED diagnostic; not validation, not model selection, not final test.",
        "",
        "## Hypothesis",
        "",
        "Stage1 stress cases show material episode-level fixed-H opportunity concentrated in case 5. Stage2 tests whether this is a reusable within-episode state-dependent horizon signal or merely a case-level fixed-H effect.",
        "",
        "## Fixed design",
        "",
        f"- Prefix horizon: H{PREFIX_H}; branch horizons: `{BRANCH_HORIZONS}`.",
        f"- Targets: `{len(protocol['targets'])}`; rollout episodes if executed: `{protocol['budget']['rollout_episodes_exact']}`; control-step cap: `{protocol['budget']['control_step_upper_bound']}`.",
        "- Target selection uses Stage1 H15 traces and metadata only for branch-step selection. No Stage2 branch outcomes exist at freeze time.",
        "- Run is blocked until an external backup covers this runner, frozen protocol, prepare outputs and backup request.",
        "",
        "## Targets",
        "",
        "| target | case | role | branch step | H15 trace len | score | theta_r | traj_steps | clearance |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in protocol["targets"]:
        meta = t["case_metadata"]
        lines.append("| %d | %d | `%s` | %d | %d | %.6g | %.6g | %d | %.6g |" % (
            int(t["target_index"]), int(t["case"]), t["selection_role"], int(t["branch_step"]), int(t["h15_trace_length"]),
            float(t["h15_only_selection_features"]["transient_score_from_H15_only"]),
            float(meta.get("theta_r", float("nan"))), int(meta.get("traj_steps", 0)), float(meta.get("min_reference_obstacle_clearance", float("nan"))),
        ))
    lines += [
        "",
        "## Stage2 analysis gates (frozen before rollout)",
        "",
        "- Material positive state: non-H15 branch has identical clean H15 prefix, branch-state distance <=1e-5, no success/constraint/initial/final solver regression versus the H15 branch, and improves continuation physical or total cost by >=3 absolute units.",
        "- Training/refit label gate: >=2 material non-H15 positive matched states, at least one in the Stage1 material stress case, and >=2 retained negative/control matched states, with no blocking prefix/state/safety artifact.",
        "- If the gate fails, preserve as evidence of sparse within-episode opportunity and pivot to terminal/reward/modeling or a separately versioned stronger scenario design rather than retraining on sparse labels.",
        "",
        f"Backup request before Stage2 rollout: `{rel(REQUEST_BACKUP_BEFORE_RUN)}`.",
    ]
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prepare_summary(raw: Mapping[str, Any]) -> None:
    tdiag = raw["target_diagnostics"]
    lines = [
        "# Vehicle stress-scenario Stage2 continuation prepare v0",
        "",
        f"UTC: `{raw['created_utc']}`. No simulations, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Backup/input gate",
        "",
        f"- Supervisor Stage1 backup proof: `{raw['stage1_backup_proof']['path']}` commit `{raw['stage1_backup_proof']['commit']}`, postdates Stage1 postdiagnostic: `{raw['stage1_backup_proof']['postdates_stage1_postdiagnostic']}`.",
        f"- Stage1 postdiagnostic authorized Stage2 freeze: `{raw['stage1_postdiagnostic']['stage2_identical_state_continuation_should_be_frozen_after_backup']}`.",
        "",
        "## Frozen Stage2 target diagnostic",
        "",
        f"- Target count: `{tdiag['target_count']}`; planned episodes: `{tdiag['episode_budget']}`; control-step cap: `{tdiag['control_step_upper_bound']}`.",
        f"- Case counts: `{tdiag['case_counts']}`; role counts: `{tdiag['role_counts']}`.",
        f"- Branch-step summary: `{tdiag['branch_step_summary']}`.",
        f"- H15-only score summary: `{tdiag['feature_score_summary']}`.",
        "",
        "## Decision",
        "",
        "Prepared/froze Stage2 identical-state continuation protocol and runner source. The actual 72-episode Stage2 rollout is still blocked until an external backup covers this new source, frozen protocol, prepare artifacts and backup request.",
        "",
        f"Frozen protocol JSON: `{rel(PROTOCOL_JSON)}`.",
        f"Frozen protocol MD: `{rel(PROTOCOL_MD)}`.",
        f"Backup request before rollout: `{rel(REQUEST_BACKUP_BEFORE_RUN)}`.",
    ]
    PREPARE_DIR.mkdir(parents=True, exist_ok=True)
    (PREPARE_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str, marker: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if marker not in old:
                path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def prepare_only() -> int:
    if PREPARE_DIR.exists() and (PREPARE_DIR / "completed.json").exists():
        verify_completed_basic(PREPARE_DIR / "completed.json", check_hashes=True)
        print(json.dumps({"prepare_already_completed": rel(PREPARE_DIR / "completed.json"), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    stage1_raw, stage1_done, stage1_post_raw, stage1_post_done = verify_stage1_inputs()
    backup = persist_and_verify_supervisor_backup(stage1_post_done)
    targets, target_diag = build_targets(stage1_raw, stage1_post_raw)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    protocol_full = {
        "protocol_id": f"vehicle_stress_scenario_stage2_continuation_v0_frozen_{PROTOCOL_STAMP}",
        "created_utc": created,
        "classification": "development_IMPROVED_identical_state_continuation_not_validation_not_final_test",
        "source_stage1": {
            "stage1_raw": rel(STAGE1_RAW),
            "stage1_raw_sha256": sha256(STAGE1_RAW),
            "stage1_completed": rel(STAGE1_COMPLETED),
            "stage1_completed_sha256": sha256(STAGE1_COMPLETED),
            "stage1_postdiagnostic_raw": rel(STAGE1_POST_RAW),
            "stage1_postdiagnostic_raw_sha256": sha256(STAGE1_POST_RAW),
            "stage1_postdiagnostic_completed": rel(STAGE1_POST_COMPLETED),
            "stage1_postdiagnostic_completed_sha256": sha256(STAGE1_POST_COMPLETED),
            "stage1_bank": rel(STAGE1_BANK),
            "stage1_bank_sha256": sha256(STAGE1_BANK),
        },
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "final_test_authorized": False,
            "requires_verified_external_backup_after_prepare_before_rollout": True,
            "do_not_modify_stage1_or_current_validation_campaign": True,
        },
        "hypothesis": "Stage1 material episode-level horizon opportunity is localized; matched H15-prefix continuations test whether safe non-H15 gains exist from identical intermediate states.",
        "target_selection_rule": target_diag["selection_rule"],
        "targets": targets,
        "rollout_design_after_backup": {
            "prefix_horizon": PREFIX_H,
            "branch_horizons": BRANCH_HORIZONS,
            "rollout_episodes_exact": len(targets) * len(BRANCH_HORIZONS),
            "control_step_upper_bound": len(targets) * len(BRANCH_HORIZONS) * MAX_STEPS,
            "max_steps_per_episode": MAX_STEPS,
            "order_seed": ORDER_SEED,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
        },
        "analysis_rules_frozen_before_rollout": {
            "material_gain_abs_threshold": MATERIAL_GAIN_THRESHOLD,
            "prefix_clean_sha256_must_match_H15": True,
            "branch_state_distance_tolerance": 1e-5,
            "no_success_constraint_initial_or_solver_regression_vs_H15": True,
            "training_refit_label_gate": ">=2 material non-H15 positive matched states, at least one in case 5, and >=2 negative/control states, with no blocking artifacts",
        },
        "decision_after_stage2": {
            "if_gate_passes": "freeze compact IMPROVED supervised selector/value-refit smoke, then fresh confirmation with fair same-distribution fixed-H baselines",
            "if_gate_fails": "preserve scenario-opportunity scarcity evidence and pivot to terminal-value/reward/modeling diagnosis or separately versioned source-supported scenario design; do not retrain on sparse labels",
        },
        "stage1_backup_proof": backup,
    }
    raw = {
        "created_utc": created,
        "method": "vehicle_stress_scenario_stage2_continuation_v0_prepare_no_simulation",
        "classification": "development_prepare_protocol_target_selection_no_rollouts",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "stage1_backup_proof": backup,
        "stage1_completed": {"path": rel(STAGE1_COMPLETED), "sha256": sha256(STAGE1_COMPLETED), "episodes": stage1_done.get("episodes"), "control_steps": stage1_done.get("control_steps")},
        "stage1_postdiagnostic": {"path": rel(STAGE1_POST_COMPLETED), "sha256": sha256(STAGE1_POST_COMPLETED), "stage2_identical_state_continuation_should_be_frozen_after_backup": stage1_post_done.get("stage2_identical_state_continuation_should_be_frozen_after_backup"), "material_non_H15_positive_cases": stage1_post_done.get("material_non_H15_positive_cases")},
        "target_diagnostics": target_diag,
        "protocol_full": protocol_full,
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "source_hashes_before_protocol_write": source_hashes(),
        "interpretation_limits": ["prepare-only", "no new simulation", "no training/refit", "not validation/model selection", "not final test"],
    }
    write_protocol_and_summary(raw)
    write_json(REQUEST_BACKUP_BEFORE_RUN, {
        "requested_utc": created,
        "reason": "backup Stage2 continuation runner source, frozen protocol and prepare outputs before any 72-episode rollout simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "planned_rollout_episodes": protocol_full["rollout_design_after_backup"]["rollout_episodes_exact"],
        "planned_control_step_upper_bound": protocol_full["rollout_design_after_backup"]["control_step_upper_bound"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(PREPARE_DIR), rel(STATE_PREPARE), rel(SUPERVISOR_PROOF_PATH), rel(REQUEST_BACKUP_BEFORE_RUN)],
    })
    raw["backup_request_before_stage2_rollout"] = rel(REQUEST_BACKUP_BEFORE_RUN)
    raw["source_hashes"] = source_hashes(extra=[REQUEST_BACKUP_BEFORE_RUN])
    PREPARE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PREPARE_DIR / "raw.json", raw)
    write_prepare_summary(raw)
    STATE_PREPARE.parent.mkdir(parents=True, exist_ok=True)
    STATE_PREPARE.write_text(
        f"# Vehicle stress-scenario Stage2 prepare state ({created})\n\n"
        f"Prepared/froze Stage2 matched-continuation protocol with {len(targets)} targets and {len(targets) * len(BRANCH_HORIZONS)} planned episodes. "
        "No simulations/training/validation64/test access occurred. Stage2 rollout is backup-blocked until a verified proof covers the new source/protocol/prepare artifacts and backup request.\n",
        encoding="utf-8",
    )
    block = f"""<!-- {MARKER_PREPARE} -->
## 2026-09-28 vehicle stress-scenario Stage2 continuation prepare v0

UTC: {created}. No-simulation Stage2 matched-continuation target/protocol freeze completed after a verified Stage1-postdiagnostic backup proof. Targets={len(targets)}, planned branch horizons={BRANCH_HORIZONS}, planned episodes={len(targets) * len(BRANCH_HORIZONS)}, control-step cap={len(targets) * len(BRANCH_HORIZONS) * MAX_STEPS}. No validation64-bank or sealed-test access and no training/refit. Rollout is blocked until external backup covers `experiments/bohn2021_aws/vehicle_stress_scenario_stage2_continuation_v0_runner.py`, `{rel(PROTOCOL_JSON)}`, `{rel(PROTOCOL_MD)}`, `{rel(PREPARE_DIR)}` and `{rel(REQUEST_BACKUP_BEFORE_RUN)}`.
"""
    append_docs(block, MARKER_PREPARE)
    files = [p for p in PREPARE_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, STATE_PREPARE, SUPERVISOR_PROOF_PATH, REQUEST_BACKUP_BEFORE_RUN, STAGE1_POST_COMPLETED, STAGE1_COMPLETED]
    write_json(PREPARE_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "target_count": len(targets),
        "planned_rollout_episodes": len(targets) * len(BRANCH_HORIZONS),
        "planned_control_step_upper_bound": len(targets) * len(BRANCH_HORIZONS) * MAX_STEPS,
        "backup_required_before_stage2_rollout": True,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN),
        "protocol_json": rel(PROTOCOL_JSON),
        "protocol_json_sha256": sha256(PROTOCOL_JSON),
        "protocol_md": rel(PROTOCOL_MD),
        "protocol_md_sha256": sha256(PROTOCOL_MD),
        "next_after_backup": "run --run-stage2 under legacy interpreter with a verified backup proof postdating this prepare",
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(PREPARE_DIR / "completed.json"),
        "summary": rel(PREPARE_DIR / "summary.md"),
        "protocol_json": rel(PROTOCOL_JSON),
        "target_count": len(targets),
        "planned_episodes": len(targets) * len(BRANCH_HORIZONS),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(REQUEST_BACKUP_BEFORE_RUN),
        "next": "await_verified_external_backup_before_stage2_rollout",
    }, sort_keys=True), flush=True)
    return 0


def verify_backup_proof_for_run(path: Path, min_time: dt.datetime) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof path does not exist: %s" % rel(path))
    proof = read_json(path)
    verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
    if not verified:
        raise ContractError("backup proof is not verified")
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    packages = proof.get("packages_this_run") or []
    if not (proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256") or packages):
        raise ContractError("backup proof lacks package/release SHA")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if proof_time is None or proof_time < min_time:
        raise ContractError("backup proof predates Stage2 prepare/source artifacts")
    return {"path": rel(path), "sha256": sha256(path), "time": proof_time.isoformat(), "commit": proof.get("commit"), "remaining_changed_files": proof.get("remaining_changed_files"), "packages_this_run": packages}


def build_schedule(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in targets:
        for h in BRANCH_HORIZONS:
            rows.append({
                "target_index": int(t["target_index"]),
                "case": int(t["case"]),
                "branch_step": int(t["branch_step"]),
                "branch_horizon": int(h),
                "selection_role": t["selection_role"],
            })
    # deterministic local permutation without numpy dependency in prepare mode
    import random
    rng = random.Random(ORDER_SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [dict(rows[i], execution_index=j, schedule_base_index=i) for j, i in enumerate(order)]


def no_regression(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if bool(reference.get("success")) and not bool(candidate.get("success")):
        return False
    if bool(candidate.get("constraint")) and not bool(reference.get("constraint")):
        return False
    if int(candidate.get("initial_failed_steps", 0)) > int(reference.get("initial_failed_steps", 0)):
        return False
    if int(candidate.get("solver_failure_steps", 0)) > int(reference.get("solver_failure_steps", 0)):
        return False
    return True


def state_tuple(row: Mapping[str, Any], key: str = "branch_previous_state") -> Tuple[float, float, float]:
    s = row.get(key) or {}
    return (safe_float(s.get("x"), float("nan")), safe_float(s.get("y"), float("nan")), safe_float(s.get("theta"), float("nan")))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def analyze_stage2(episodes: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[int, List[Mapping[str, Any]]] = {}
    target_by_index = {int(t["target_index"]): t for t in targets}
    for ep in episodes:
        groups.setdefault(int(ep["target_index"]), []).append(ep)
    state_rows: List[Dict[str, Any]] = []
    artifact_flags = {"missing_H15_reference": 0, "missing_horizon": 0, "prefix_mismatch_non_H15": 0, "state_distance_gt_tol_non_H15": 0, "branch_not_reached": 0, "safety_solver_regression_non_H15": 0}
    positive_count = 0
    negative_count = 0
    positive_cases: List[int] = []
    positive_roles: List[str] = []
    positive_horizon_counts: Dict[str, int] = {}
    large_harms = 0
    best_non_h15_total_gain = -float("inf")
    best_non_h15_physical_gain = -float("inf")
    for tid in sorted(target_by_index):
        target = target_by_index[tid]
        rows = sorted(groups.get(tid, []), key=lambda r: int(r.get("branch_horizon", -1)))
        by_h = {int(r["branch_horizon"]): r for r in rows}
        for h in BRANCH_HORIZONS:
            if h not in by_h:
                artifact_flags["missing_horizon"] += 1
        ref = by_h.get(PREFIX_H)
        if ref is None:
            artifact_flags["missing_H15_reference"] += 1
            continue
        if not bool(ref.get("branch_reached")):
            artifact_flags["branch_not_reached"] += 1
        ref_state = state_tuple(ref)
        comparisons: List[Dict[str, Any]] = []
        material_horizons: List[int] = []
        best_physical = None
        best_total = None
        for h in BRANCH_HORIZONS:
            row = by_h.get(h)
            if row is None:
                continue
            branch_reached = bool(row.get("branch_reached"))
            if not branch_reached:
                artifact_flags["branch_not_reached"] += 1
            cand_state = state_tuple(row)
            dist = state_distance(cand_state, ref_state) if branch_reached and bool(ref.get("branch_reached")) else float("inf")
            prefix_match = bool(row.get("prefix_clean_sha256") == ref.get("prefix_clean_sha256"))
            if h != PREFIX_H and not prefix_match:
                artifact_flags["prefix_mismatch_non_H15"] += 1
            if h != PREFIX_H and branch_reached and dist > 1e-5:
                artifact_flags["state_distance_gt_tol_non_H15"] += 1
            phys_gain = float(ref["continuation_physical_constraint_cost_from_branch"] - row["continuation_physical_constraint_cost_from_branch"]) if branch_reached and bool(ref.get("branch_reached")) else float("nan")
            total_gain = float(ref["continuation_total_cost_from_branch"] - row["continuation_total_cost_from_branch"]) if branch_reached and bool(ref.get("branch_reached")) else float("nan")
            if h != PREFIX_H and math.isfinite(total_gain):
                best_non_h15_total_gain = max(best_non_h15_total_gain, total_gain)
                if total_gain <= -MATERIAL_GAIN_THRESHOLD:
                    large_harms += 1
            if h != PREFIX_H and math.isfinite(phys_gain):
                best_non_h15_physical_gain = max(best_non_h15_physical_gain, phys_gain)
            safety_ok = no_regression(row, ref)
            if h != PREFIX_H and not safety_ok:
                artifact_flags["safety_solver_regression_non_H15"] += 1
            material = bool(h != PREFIX_H and branch_reached and bool(ref.get("branch_reached")) and prefix_match and dist <= 1e-5 and safety_ok and (phys_gain >= MATERIAL_GAIN_THRESHOLD or total_gain >= MATERIAL_GAIN_THRESHOLD))
            if material:
                material_horizons.append(h)
                positive_horizon_counts[str(h)] = positive_horizon_counts.get(str(h), 0) + 1
            comp = {
                "horizon": h,
                "branch_reached": branch_reached,
                "success": bool(row.get("success")),
                "constraint": bool(row.get("constraint")),
                "steps": int(row.get("steps", 0)),
                "continuation_physical": float(row.get("continuation_physical_constraint_cost_from_branch", 0.0)),
                "continuation_total": float(row.get("continuation_total_cost_from_branch", 0.0)),
                "gain_vs_H15_physical": phys_gain,
                "gain_vs_H15_total": total_gain,
                "decision_sum_s": float(row["decision_timing_s"]["sum"]),
                "decision_plus_terminal_switch_sum_s": float(row["decision_plus_terminal_switch_timing_s"]["sum"]),
                "state_distance_vs_H15_branch_state": dist,
                "prefix_clean_sha256_matches_H15": prefix_match,
                "no_success_constraint_solver_regression_vs_H15": safety_ok,
                "material_positive_vs_H15": material,
                "initial_failed_steps": int(row.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(row.get("solver_failure_steps", 0)),
                "path": row.get("path"),
            }
            comparisons.append(comp)
            if branch_reached:
                if best_physical is None or (comp["continuation_physical"], comp["decision_sum_s"], h) < (best_physical["continuation_physical"], best_physical["decision_sum_s"], best_physical["horizon"]):
                    best_physical = comp
                if best_total is None or (comp["continuation_total"], comp["decision_sum_s"], h) < (best_total["continuation_total"], best_total["decision_sum_s"], best_total["horizon"]):
                    best_total = comp
        if material_horizons:
            positive_count += 1
            positive_cases.append(int(target["case"]))
            positive_roles.append(str(target["selection_role"]))
        else:
            negative_count += 1
        state_rows.append({
            "target_index": tid,
            "case": int(target["case"]),
            "selection_role": target["selection_role"],
            "branch_step": int(target["branch_step"]),
            "stage1_material_positive_case": bool(target.get("stage1_material_positive_case")),
            "material_positive_state": bool(material_horizons),
            "material_positive_horizons": material_horizons,
            "label": "positive_non_H15" if material_horizons else "negative_or_neutral",
            "reference_H15": {
                "success": bool(ref.get("success")),
                "constraint": bool(ref.get("constraint")),
                "steps": int(ref.get("steps", 0)),
                "continuation_physical": float(ref.get("continuation_physical_constraint_cost_from_branch", 0.0)),
                "continuation_total": float(ref.get("continuation_total_cost_from_branch", 0.0)),
                "branch_previous_state": ref.get("branch_previous_state"),
                "initial_failed_steps": int(ref.get("initial_failed_steps", 0)),
                "solver_failure_steps": int(ref.get("solver_failure_steps", 0)),
            },
            "best_physical": best_physical,
            "best_total": best_total,
            "comparisons": comparisons,
        })
    blocking_artifacts = artifact_flags["missing_H15_reference"] + artifact_flags["missing_horizon"] + artifact_flags["prefix_mismatch_non_H15"] + artifact_flags["state_distance_gt_tol_non_H15"]
    positive_case5 = any(int(c) == 5 for c in positive_cases)
    negative_controls = sum(1 for r in state_rows if not r["material_positive_state"] and r["selection_role"] in ("nonmaterial_stress_control", "lower_stress_control"))
    gate = bool(positive_count >= 2 and positive_case5 and negative_controls >= 2 and blocking_artifacts == 0)
    return {
        "state_count": len(state_rows),
        "positive_state_count": int(positive_count),
        "negative_neutral_state_count": int(negative_count),
        "positive_cases": sorted(set(positive_cases)),
        "positive_roles": sorted(set(positive_roles)),
        "positive_horizon_counts": positive_horizon_counts,
        "negative_control_state_count": int(negative_controls),
        "artifact_flags": artifact_flags,
        "blocking_artifact_count": int(blocking_artifacts),
        "best_non_H15_total_gain": None if best_non_h15_total_gain == -float("inf") else float(best_non_h15_total_gain),
        "best_non_H15_physical_gain": None if best_non_h15_physical_gain == -float("inf") else float(best_non_h15_physical_gain),
        "large_non_H15_harms": int(large_harms),
        "training_refit_label_gate_pass_development_only": gate,
        "gate_rule": ">=2 material non-H15 positive matched states, at least one in case 5, >=2 retained negative/control states, and no missing/prefix/state-distance blocking artifacts",
        "state_rows": state_rows,
    }


def write_run_summary(raw: Mapping[str, Any]) -> None:
    a = raw["analysis"]
    lines = [
        "# Vehicle stress-scenario Stage2 identical-state continuation v0",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only matched-continuation diagnostic; no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Budget/access",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / `{raw['budget_declared']['rollout_episodes_exact']}`; control steps: `{raw['budget_actual']['control_steps']}` / cap `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Label result",
        "",
        f"- Positive matched states: `{a['positive_state_count']}` across cases `{a['positive_cases']}`; negative/neutral states: `{a['negative_neutral_state_count']}`.",
        f"- Positive horizon counts: `{a['positive_horizon_counts']}`; negative/control states: `{a['negative_control_state_count']}`.",
        f"- Blocking artifact count: `{a['blocking_artifact_count']}`; flags: `{a['artifact_flags']}`.",
        f"- Best non-H15 total gain: `{a['best_non_H15_total_gain']}`; best physical gain: `{a['best_non_H15_physical_gain']}`; large harms: `{a['large_non_H15_harms']}`.",
        f"- training_refit_label_gate_pass_development_only: `{a['training_refit_label_gate_pass_development_only']}`.",
        "",
        "## Per-target labels",
        "",
        "| target | case | role | branch step | label | material H | H15 cont phys | H15 cont total | best phys H/gain | best total H/gain |",
        "|---:|---:|---|---:|---|---|---:|---:|---|---|",
    ]
    for r in a["state_rows"]:
        ref = r["reference_H15"]
        bp = r.get("best_physical") or {}
        bt = r.get("best_total") or {}
        lines.append("| %d | %d | `%s` | %d | `%s` | `%s` | %.6g | %.6g | H%s/%.6g | H%s/%.6g |" % (
            int(r["target_index"]), int(r["case"]), r["selection_role"], int(r["branch_step"]), r["label"], r["material_positive_horizons"],
            float(ref["continuation_physical"]), float(ref["continuation_total"]), str(bp.get("horizon")), float(bp.get("gain_vs_H15_physical", 0.0)), str(bt.get("horizon")), float(bt.get("gain_vs_H15_total", 0.0)),
        ))
    lines += ["", "## Decision", ""]
    if a["training_refit_label_gate_pass_development_only"]:
        lines.append("Stage2 found enough matched safe positives to justify freezing a compact IMPROVED selector/value-refit smoke before any broader validation.")
    else:
        lines.append("Stage2 did not meet the label-density/artifact gate; preserve sparse within-episode opportunity evidence and pivot to terminal/reward/modeling or stronger scenario design rather than retraining on sparse labels.")
    lines.append("")
    lines.append(f"Backup request after Stage2 rollout: `{raw['backup_request']}`.")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2(args: argparse.Namespace) -> int:
    prepare_done = verify_completed_basic(PREPARE_DIR / "completed.json", check_hashes=True)
    protocol = read_json(PROTOCOL_JSON)
    if canonical_sha(protocol.get("targets")) != canonical_sha(read_json(PREPARE_DIR / "raw.json")["protocol_full"]["targets"]):
        raise ContractError("protocol target list changed after prepare")
    min_time = max(parse_time(prepare_done["created_utc"]) or dt.datetime.now(dt.timezone.utc), source_mtime_utc())
    backup = verify_backup_proof_for_run(args.backup_proof, min_time)
    if RUN_DIR.exists() and (RUN_DIR / "completed.json").exists():
        verify_completed_basic(RUN_DIR / "completed.json", check_hashes=True)
        raise SystemExit("Stage2 rollout already completed and verified; refusing rerun")
    if RUN_DIR.exists():
        leftovers = [p for p in RUN_DIR.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise ContractError("partial Stage2 output exists; inspect before rerun: " + ", ".join(rel(p) for p in leftovers[:20]))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(RUN_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "vehicle_stress_scenario_stage2_continuation_v0", "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "training_gradient_steps": 0})
    # Import legacy environment only for real rollout mode.
    import vehicle_stress_scenario_opportunity_probe_v0_runner as stage1_runner  # noqa:E402
    import vehicle_v1_fresh_continuation_label_probe_v0_runner as fresh  # noqa:E402
    fresh.OUT_DIR = RUN_DIR
    fresh.STATE_PATH = STATE_RUN
    fresh.PREFIX_H = PREFIX_H
    fresh.BRANCH_HORIZONS = list(BRANCH_HORIZONS)
    fresh.MAX_STEPS = MAX_STEPS
    fresh.IMPROVEMENT_ABS_THRESHOLD = MATERIAL_GAIN_THRESHOLD
    preflight = stage1_runner.runtime_preflight()
    write_json(RUN_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError("legacy runtime preflight failed: " + str(preflight))
    stage1_runner.base.v1.latency_verify()
    bank = read_json(STAGE1_BANK)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    stage1_protocol = read_json(STAGE1_PROTOCOL_JSON)
    terminals, terminal_receipts = stage1_runner.load_terminal_grid_for_stage1(stage1_protocol)
    write_json(RUN_DIR / "terminal_sources.json", terminal_receipts)
    schedule = build_schedule(protocol["targets"])
    write_json(RUN_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "branch_horizons": BRANCH_HORIZONS, "prefix_horizon": PREFIX_H})
    episodes: List[Dict[str, Any]] = []
    target_by_index = {int(t["target_index"]): t for t in protocol["targets"]}
    for item in schedule:
        target = target_by_index[int(item["target_index"])]
        case_id = int(item["case"])
        summary = fresh.run_episode(item, selected_cases[case_id], selected_meta[case_id], terminals, terminal_receipts)
        summary.update({"target_index": int(item["target_index"]), "selection_role": target["selection_role"], "stage1_material_positive_case": bool(target.get("stage1_material_positive_case")), "target_h15_only_selection_features": target.get("h15_only_selection_features")})
        # Rewrite augmented per-episode summary for traceability.
        write_json(ROOT / summary["path"] / "summary.json", summary)
        episodes.append(summary)
        progress = {"pid": os.getpid(), "episodes_done": len(episodes), "episodes_expected": len(schedule), "control_steps_done": int(sum(int(e.get("steps", 0)) for e in episodes)), "last_episode": {k: summary.get(k) for k in ("execution_index", "target_index", "case", "branch_step", "branch_horizon", "steps", "success", "termination", "branch_reached")}, "historical_validation64_bank_opened": False, "sealed_test_accessed": False}
        write_json(RUN_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    budget = protocol["rollout_design_after_backup"]
    if len(episodes) != int(budget["rollout_episodes_exact"]) or control_steps > int(budget["control_step_upper_bound"]):
        raise ContractError("Stage2 rollout budget violation")
    analysis = analyze_stage2(episodes, protocol["targets"])
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    backup_request = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE2_CONTINUATION_V0_%s.json" % created.replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created,
        "started_utc": started,
        "method": "vehicle_stress_scenario_stage2_continuation_v0_identical_H15_prefix_branch_fixed_H",
        "classification": "development_IMPROVED_matched_continuation_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"json": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD)},
        "prepare": {"completed": rel(PREPARE_DIR / "completed.json"), "completed_sha256": sha256(PREPARE_DIR / "completed.json")},
        "source_hashes": source_hashes(extra=[args.backup_proof]),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": budget,
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": ["development diagnostic only", "matched continuations from replayed H15 prefix", "not online adaptive validation", "not new training/refit", "not final test"],
    }
    write_json(backup_request, {"requested_utc": created, "reason": "backup Stage2 matched-continuation rollout before selector/refit or further simulations", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "artifacts": [rel(RUN_DIR), rel(STATE_RUN), rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(backup_request)]})
    raw["backup_request"] = rel(backup_request)
    write_json(RUN_DIR / "raw.json", raw)
    write_run_summary(raw)
    STATE_RUN.parent.mkdir(parents=True, exist_ok=True)
    STATE_RUN.write_text(f"# Vehicle stress-scenario Stage2 run state ({created})\n\nCompleted {len(episodes)} matched-continuation episodes / {control_steps} control steps. Positive states={analysis['positive_state_count']}; negative/neutral={analysis['negative_neutral_state_count']}; label gate={analysis['training_refit_label_gate_pass_development_only']}. No validation64/test/training. Backup required before next simulation.\n", encoding="utf-8")
    block = f"""<!-- {MARKER_RUN} -->
## 2026-09-28 vehicle stress-scenario Stage2 continuation v0

UTC: {created}. Development-only matched-continuation rollout completed: {len(episodes)} episodes, {control_steps} control steps. Positive states={analysis['positive_state_count']}, negative/neutral={analysis['negative_neutral_state_count']}, label gate={analysis['training_refit_label_gate_pass_development_only']}, blocking artifacts={analysis['blocking_artifact_count']}. No training, no validation64-bank access, no sealed-test access. Artifacts: `{rel(RUN_DIR / 'summary.md')}`, `{rel(RUN_DIR / 'raw.json')}`, `{rel(RUN_DIR / 'completed.json')}`.
"""
    append_docs(block, MARKER_RUN)
    files = [p for p in RUN_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, STATE_RUN, backup_request, args.backup_proof]
    write_json(RUN_DIR / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "episodes": len(episodes), "control_steps": control_steps, "new_training_episodes": 0, "new_gradient_steps": 0, "backup_request": rel(backup_request), "headline": {"positive_state_count": analysis["positive_state_count"], "negative_neutral_state_count": analysis["negative_neutral_state_count"], "training_refit_label_gate_pass_development_only": analysis["training_refit_label_gate_pass_development_only"], "blocking_artifact_count": analysis["blocking_artifact_count"], "next_action": "if gate passed freeze selector/refit smoke; otherwise pivot to terminal/reward/modeling or stronger scenario design"}, "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()}})
    print(json.dumps({"completed": rel(RUN_DIR / "completed.json"), "summary": rel(RUN_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "positive_state_count": analysis["positive_state_count"], "training_refit_label_gate_pass_development_only": analysis["training_refit_label_gate_pass_development_only"], "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(backup_request)}, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true", help="freeze protocol/targets only; no simulations")
    ap.add_argument("--run-stage2", action="store_true", help="run 72 matched-continuation episodes after post-prepare backup")
    ap.add_argument("--backup-proof", type=Path, help="verified post-prepare backup proof required for --run-stage2")
    ap.add_argument("--i-accept-stage2-development-diagnostic", action="store_true", help="acknowledge development-only diagnostic, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_stage2_development_diagnostic:
        raise ContractError("explicit --i-accept-stage2-development-diagnostic is required")
    if args.prepare_only == args.run_stage2:
        raise ContractError("choose exactly one of --prepare-only or --run-stage2")
    if args.prepare_only:
        return prepare_only()
    if args.backup_proof is None:
        raise ContractError("--run-stage2 requires --backup-proof")
    return run_stage2(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        target = PREPARE_DIR if "--prepare-only" in sys.argv else RUN_DIR
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0 if "--prepare-only" in sys.argv else None,
            "new_control_steps": 0 if "--prepare-only" in sys.argv else None,
            "next_recovery_hint": "Preserve partial output. If prepare failed, repair protocol/target selection only; if rollout failed, audit partial episodes before rerun.",
        })
        raise
