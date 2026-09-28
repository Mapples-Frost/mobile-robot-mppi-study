#!/usr/bin/env python3
"""Prepare/dry-run vehicle stress-v1b matched-continuation diagnostic.

This analysis-only script freezes the next bounded diagnostic after stress-v1
Stage1.  It performs **no rollout simulations**, no training/refit, no validation
bank access and no sealed-test access.  It selects matched-continuation branch
states from already saved H15 traces using H15-only metadata, so branch outcomes
remain unknown until a separately backed-up rollout runner is executed.

Motivation
----------
Stress-v1 Stage1 increased episode-level fixed-H opportunity but failed the
predeclared Stage2 diversity gate: material H15-counterfactual cases were
[1, 4, 5, 6], all in one high-heading long/medium stratum.  The user requested
concrete bounded diagnostics rather than another unchanged validation batch.
This prepare step freezes a cheaper discriminating experiment: matched
continuations from identical H15-prefix states in the four positive cases,
same-stratum negative controls and lower-stress controls.

Rollout remains blocked until an external backup covers this source, frozen
protocol, dry-run outputs and backup request.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

# Reuse only pure-Python utilities from the prior Stage2 runner.  No environment
# or TensorFlow imports are triggered in prepare mode.
import vehicle_stress_scenario_stage2_continuation_v0_runner as stage2_base  # noqa:E402

STAMP = "20260928T2205Z"
METHOD = "vehicle_stress_v1b_matched_continuation_prepare"
PREFIX_H = 15
BRANCH_HORIZONS = [10, 15, 25, 30, 35, 45, 50]
MAX_STEPS = 150
TARGET_CASES_POSITIVE = [1, 4, 5, 6]
TARGET_CASES_SAME_STRATUM_CONTROLS = [0, 2, 3, 7]
TARGET_CASES_LOWER_STRESS_CONTROLS = [16, 17]
TARGET_CASES = TARGET_CASES_POSITIVE + TARGET_CASES_SAME_STRATUM_CONTROLS + TARGET_CASES_LOWER_STRESS_CONTROLS
ORDER_SEED = 2609289501
MATERIAL_GAIN_THRESHOLD = 3.0
STATE_DISTANCE_TOLERANCE = 1e-5

STAGE1_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z"
STAGE1_RAW = STAGE1_DIR / "raw.json"
STAGE1_COMPLETED = STAGE1_DIR / "completed.json"
STAGE1_SUMMARY = STAGE1_DIR / "summary.md"
STAGE1_BANK = STAGE1_DIR / "bank/vehicle_stress_scenario_opportunity_probe_v1_bank.json"
POST_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z"
POST_RAW = POST_DIR / "raw.json"
POST_COMPLETED = POST_DIR / "completed.json"
POST_SUMMARY = POST_DIR / "summary.md"
STAGE1_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json"
STAGE1_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.md"

OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_prepare_{STAMP}"
PROTOCOL_JSON = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_{STAMP}.json"
PROTOCOL_MD = ROOT / f"research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_{STAMP}.md"
STATE_PATH = ROOT / f"research_artifacts/aws_state/vehicle_stress_v1b_matched_continuation_prepare_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
REQUEST_BACKUP = BACKUP_DIR / f"REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_{STAMP}.json"
MARKER = f"vehicle-stress-v1b-matched-continuation-prepare-{STAMP}"


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
        out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def source_hashes(extra: Sequence[Path] = ()) -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(), STAGE1_RAW, STAGE1_COMPLETED, STAGE1_SUMMARY,
        STAGE1_BANK, POST_RAW, POST_COMPLETED, POST_SUMMARY,
        STAGE1_PROTOCOL_JSON, STAGE1_PROTOCOL_MD, PROTOCOL_JSON, PROTOCOL_MD,
        STATE_PATH, REQUEST_BACKUP,
    ] + list(extra)
    return {rel(p): sha256(p) for p in paths if p.exists()}


def completed_passed(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("missing completed marker: %s" % rel(path))
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError("completed marker did not pass: %s" % rel(path))
    if obj.get("sealed_test_accessed") is not False:
        raise ContractError("sealed-test access flag is not false in %s" % rel(path))
    return obj


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for p in (STAGE1_RAW, STAGE1_COMPLETED, STAGE1_SUMMARY, STAGE1_BANK, POST_RAW, POST_COMPLETED, POST_SUMMARY, STAGE1_PROTOCOL_JSON, STAGE1_PROTOCOL_MD):
        if not p.exists():
            raise ContractError("required input missing: %s" % rel(p))
    stage1_done = completed_passed(STAGE1_COMPLETED)
    post_done = completed_passed(POST_COMPLETED)
    if stage1_done.get("historical_validation64_bank_opened") is not False:
        raise ContractError("Stage1 unexpectedly opened historical validation64 bank")
    if post_done.get("historical_validation64_bank_opened") is not False:
        raise ContractError("postdiagnostic unexpectedly opened historical validation64 bank")
    stage1_raw = read_json(STAGE1_RAW)
    post_raw = read_json(POST_RAW)
    bank = read_json(STAGE1_BANK)
    protocol = read_json(STAGE1_PROTOCOL_JSON)
    if int(stage1_done.get("episodes", -1)) != 200 or int(stage1_done.get("control_steps", -1)) <= 0:
        raise ContractError("Stage1 budget markers unexpected")
    if post_done.get("predeclared_stage1_gate_failed") is not True:
        raise ContractError("postdiagnostic did not preserve the failed Stage1 gate")
    selected_meta = (bank.get("selection") or {}).get("selected_metadata") or []
    if len(selected_meta) < max(TARGET_CASES) + 1:
        raise ContractError("bank selected_metadata too short for v1b target cases")
    material_cases = [int(x) for x in (post_raw.get("runner_material_cases") or post_raw.get("material_non_H15_positive_cases") or [])]
    if sorted(material_cases) != TARGET_CASES_POSITIVE:
        raise ContractError("unexpected material cases in postdiagnostic: %s" % material_cases)
    if bool(post_raw.get("predeclared_stage2_trigger_candidate", False)) is not False:
        raise ContractError("predeclared Stage2 gate unexpectedly passed; v1b should remain postdiagnostic")
    return stage1_raw, stage1_done, post_raw, post_done, bank, protocol


def trace_path_for(stage1_raw: Mapping[str, Any], case_id: int, horizon: int) -> Path:
    rows = [e for e in stage1_raw.get("episodes", []) if int(e.get("case", -1)) == int(case_id) and int(e.get("horizon", -1)) == int(horizon)]
    if len(rows) != 1:
        raise ContractError("expected exactly one Stage1 episode for case %d H%d, got %d" % (case_id, horizon, len(rows)))
    path = ROOT / rows[0]["path"] / "trace.json"
    if not path.exists():
        raise ContractError("missing H15 trace: %s" % rel(path))
    return path


def choose_step_from_range(trace: Sequence[Mapping[str, Any]], lo: int, hi: int, used_steps: Iterable[int]) -> Tuple[int, Dict[str, Any]]:
    used = set(int(x) for x in used_steps)
    max_step = max(1, min(int(hi), len(trace) - 2, 95))
    min_step = max(1, min(int(lo), max_step))
    candidates = []
    for step in range(min_step, max_step + 1):
        if step in used:
            continue
        features = stage2_base.score_trace_row(trace[step], step, len(trace))
        # Break ties toward earlier phases inside each window, reducing the chance
        # that two cases only test near-terminal continuation behavior.
        candidates.append((features["transient_score_from_H15_only"], -features["phase_fraction"], -step, step, features))
    if not candidates:
        raise ContractError("no unused H15 branch-step candidates in [%d, %d] for trace length %d" % (lo, hi, len(trace)))
    _, _, _, step, features = max(candidates)
    return int(step), features


def case_role(case_id: int) -> str:
    if case_id in TARGET_CASES_POSITIVE:
        return "stage1_material_high_heading_long_or_medium"
    if case_id in TARGET_CASES_SAME_STRATUM_CONTROLS:
        return "same_stratum_stage1_nonmaterial_control"
    if case_id in TARGET_CASES_LOWER_STRESS_CONTROLS:
        return "lower_stress_control"
    return "unexpected"


def build_targets(stage1_raw: Mapping[str, Any], bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_meta = (bank.get("selection") or {}).get("selected_metadata") or []
    targets: List[Dict[str, Any]] = []
    windows = [
        {"window_id": "early_transient", "range": [8, 32]},
        {"window_id": "mid_transient", "range": [33, 70]},
    ]
    for case_id in TARGET_CASES:
        trace_path = trace_path_for(stage1_raw, case_id, PREFIX_H)
        trace = read_json(trace_path)
        used_steps: List[int] = []
        for window in windows:
            step, features = choose_step_from_range(trace, int(window["range"][0]), int(window["range"][1]), used_steps)
            used_steps.append(step)
            row = trace[step]
            meta = copy.deepcopy(dict(selected_meta[case_id]))
            targets.append({
                "target_index": len(targets),
                "case": int(case_id),
                "source_candidate_index": int(meta.get("candidate_index", -1)),
                "selection_role": case_role(case_id),
                "stage1_material_positive_case": bool(case_id in TARGET_CASES_POSITIVE),
                "branch_step": int(step),
                "window_id": window["window_id"],
                "window_requested": list(window["range"]),
                "prefix_horizon": PREFIX_H,
                "branch_horizons": list(BRANCH_HORIZONS),
                "h15_trace_path": rel(trace_path),
                "h15_trace_sha256": sha256(trace_path),
                "h15_trace_length": len(trace),
                "h15_reference_previous_state": row.get("previous_state"),
                "h15_reference_observation": row.get("observation"),
                "h15_reference_step_costs": {
                    "performance": row.get("performance"),
                    "constraint": row.get("constraint"),
                    "compute": row.get("compute"),
                    "reward": row.get("reward"),
                },
                "h15_only_selection_features": features,
                "case_metadata": {k: meta.get(k) for k in (
                    "selection_group", "stratum", "theta_r", "abs_theta_r", "traj_steps",
                    "min_reference_obstacle_clearance", "stress_flags_count", "stress_v1_score", "stress_score",
                )},
            })
    if len(targets) != 20:
        raise ContractError("expected 20 targets, got %d" % len(targets))
    if len({(int(t["case"]), int(t["branch_step"])) for t in targets}) != len(targets):
        raise ContractError("duplicate target case/branch_step pairs")
    role_counts: Dict[str, int] = {}
    window_counts: Dict[str, int] = {}
    for target in targets:
        role_counts[target["selection_role"]] = role_counts.get(target["selection_role"], 0) + 1
        window_counts[target["window_id"]] = window_counts.get(target["window_id"], 0) + 1
    score_vals = [float(t["h15_only_selection_features"]["transient_score_from_H15_only"]) for t in targets]
    step_vals = [float(t["branch_step"]) for t in targets]
    diagnostics = {
        "target_count": len(targets),
        "branch_horizons": list(BRANCH_HORIZONS),
        "planned_branch_continuations": len(targets) * len(BRANCH_HORIZONS),
        "control_step_upper_bound": len(targets) * len(BRANCH_HORIZONS) * MAX_STEPS,
        "target_cases_positive": list(TARGET_CASES_POSITIVE),
        "target_cases_same_stratum_controls": list(TARGET_CASES_SAME_STRATUM_CONTROLS),
        "target_cases_lower_stress_controls": list(TARGET_CASES_LOWER_STRESS_CONTROLS),
        "role_counts": role_counts,
        "window_counts": window_counts,
        "branch_step_range": [min(step_vals), max(step_vals)],
        "branch_step_mean": sum(step_vals) / len(step_vals),
        "h15_only_score_range": [min(score_vals), max(score_vals)],
        "h15_only_score_mean": sum(score_vals) / len(score_vals),
        "selection_rule": "For each case in positives [1,4,5,6], same-stratum controls [0,2,3,7], and lower-stress controls [16,17], select exactly two H15-prefix states before branch outcomes: one highest H15-only transient score in steps 8-32 and one in steps 33-70 (clipped by trace length). Score is the existing H15-only turn/heading/performance/solver proxy from the prior Stage2 runner. No non-H15 branch outcomes or validation/test cases are used.",
    }
    return targets, diagnostics


def write_protocol_and_summary(raw: Mapping[str, Any]) -> None:
    protocol = raw["protocol"]
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle stress-v1b matched-continuation protocol/dry-run",
        "",
        f"Frozen UTC: `{raw['created_utc']}`. Analysis-only prepare/dry-run; no simulations, no training/refit, no validation64-bank access, no sealed-test access.",
        "",
        "## Why this diagnostic",
        "",
        "Stress-v1 Stage1 failed its predeclared diversity gate, so this is not the preregistered Stage2 success path.  However, four high-heading long/medium cases showed material episode-level H15-counterfactual positives.  v1b tests whether that one-stratum episode-level signal corresponds to identical-state within-episode labels before any selector/refit or scenario-claim expansion.",
        "",
        "## Frozen rollout design (blocked until backup)",
        "",
        f"- Prefix horizon: H{PREFIX_H}.",
        f"- Branch horizons: `{BRANCH_HORIZONS}`.",
        f"- Targets: `{len(protocol['targets'])}` = 10 cases x 2 H15-prefix states.",
        f"- Planned branch continuations: `{protocol['budget']['rollout_episodes_exact']}`; control-step cap: `{protocol['budget']['control_step_upper_bound']}`.",
        "- Target branch states are selected from H15 traces only, using deterministic early/mid transient windows and not branch outcomes.",
        "- Actual rollout must record physical, total (synthetic h_penalty), success/safety/solver flags, whole-decision timing and branch state identity separately.",
        "",
        "## Targets",
        "",
        "| target | case | role | window | step | trace len | H15 score | theta_r | traj_steps | clearance |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in protocol["targets"]:
        meta = target["case_metadata"]
        lines.append("| %d | %d | `%s` | `%s` | %d | %d | %.6g | %.6g | %s | %.6g |" % (
            int(target["target_index"]), int(target["case"]), target["selection_role"], target["window_id"],
            int(target["branch_step"]), int(target["h15_trace_length"]),
            float(target["h15_only_selection_features"]["transient_score_from_H15_only"]),
            float(meta.get("theta_r", float("nan"))), str(meta.get("traj_steps")),
            float(meta.get("min_reference_obstacle_clearance", float("nan"))),
        ))
    lines += [
        "",
        "## Frozen analysis gates for the later rollout",
        "",
        f"- Material matched positive: non-H15 branch has clean identical H15 prefix, branch-state distance <= {STATE_DISTANCE_TOLERANCE:g}, no success/constraint/initial/final/solver regression versus H15, and improves continuation physical or total cost by >= {MATERIAL_GAIN_THRESHOLD:g} absolute units.",
        "- Selector/refit gate: >=3 distinct cases and >=6 matched states with material non-H15 positives, with at least two same-stratum/control negative states retained and no blocking prefix/state/terminal/solver artifact.",
        "- If gate fails: do not train on sparse labels; preserve the negative evidence and choose between scenario redesign, terminal-value modeling, reward/objective changes, or other evidence-supported causes.",
        "",
        f"Backup request before rollout: `{rel(REQUEST_BACKUP)}`.",
    ]
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if MARKER not in text:
            path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def prepare_and_dry_run() -> int:
    if (OUT_DIR / "completed.json").exists():
        done = completed_passed(OUT_DIR / "completed.json")
        print(json.dumps({
            "already_completed": rel(OUT_DIR / "completed.json"),
            "target_count": done.get("target_count"),
            "planned_branch_continuations": done.get("planned_rollout_episodes"),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }, sort_keys=True))
        return 0
    stage1_raw, stage1_done, post_raw, post_done, bank, stage1_protocol = verify_inputs()
    targets, target_diagnostics = build_targets(stage1_raw, bank)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    protocol = {
        "protocol_id": f"vehicle_stress_v1b_matched_continuation_frozen_{STAMP}",
        "created_utc": created,
        "classification": "development_IMPROVED_matched_continuation_postdiagnostic_not_validation_not_final_test",
        "motivation": {
            "stress_v1_stage1_outcome": "predeclared gate failed; aggregate fixed-H oracle gains were large but material cases [1,4,5,6] were all in high_heading_long_or_medium",
            "why_not_training_now": "episode-level labels are one-stratum and not yet matched-state labels; training/refit is gated on this v1b continuation evidence",
            "why_this_next": "cheaper and more discriminating than another unchanged validation batch because it tests candidate horizon choices from identical H15-prefix states",
        },
        "access_rules": {
            "development_only": True,
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "final_test_authorized": False,
            "do_not_modify_current_frozen_validation_campaign": True,
            "rollout_requires_verified_external_backup_after_prepare": True,
        },
        "source_inputs": {
            "stage1_raw": rel(STAGE1_RAW),
            "stage1_raw_sha256": sha256(STAGE1_RAW),
            "stage1_completed": rel(STAGE1_COMPLETED),
            "stage1_completed_sha256": sha256(STAGE1_COMPLETED),
            "postdiagnostic_raw": rel(POST_RAW),
            "postdiagnostic_raw_sha256": sha256(POST_RAW),
            "postdiagnostic_completed": rel(POST_COMPLETED),
            "postdiagnostic_completed_sha256": sha256(POST_COMPLETED),
            "bank": rel(STAGE1_BANK),
            "bank_sha256": sha256(STAGE1_BANK),
            "stage1_protocol": rel(STAGE1_PROTOCOL_JSON),
            "stage1_protocol_sha256": sha256(STAGE1_PROTOCOL_JSON),
        },
        "target_selection_rule": target_diagnostics["selection_rule"],
        "targets": targets,
        "budget": {
            "rollout_episodes_exact": len(targets) * len(BRANCH_HORIZONS),
            "control_step_upper_bound": len(targets) * len(BRANCH_HORIZONS) * MAX_STEPS,
            "max_steps_per_episode": MAX_STEPS,
            "prefix_horizon": PREFIX_H,
            "branch_horizons": list(BRANCH_HORIZONS),
            "order_seed": ORDER_SEED,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "analysis_rules_frozen_before_rollout": {
            "material_gain_abs_threshold": MATERIAL_GAIN_THRESHOLD,
            "branch_state_distance_tolerance": STATE_DISTANCE_TOLERANCE,
            "material_positive_no_regression_fields": ["success", "constraint", "initial_failed_steps", "final_failed_steps", "solver_failure_steps"],
            "report_metrics_separately": ["physical_control_plus_constraint_cost", "total_cost_with_synthetic_h_penalty", "success_safety_solver", "whole_decision_wall_time", "solver_attempt_wall_time"],
            "selector_refit_gate": ">=3 distinct cases and >=6 states with material non-H15 positives; >=2 negative/control states retained; no blocking prefix/state/terminal/solver artifact",
        },
        "next_decision_rule": {
            "if_gate_passes": "freeze a compact IMPROVED longer-H selector/refit smoke with fair fixed-H baselines and fresh confirmation; count this v1b evidence as development only",
            "if_gate_fails": "do not retrain on sparse labels; preserve evidence and prioritize scenario-design or terminal/reward/modeling diagnostics by evidence",
        },
    }
    raw = {
        "created_utc": created,
        "method": METHOD,
        "classification": protocol["classification"],
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_candidate_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "stage1_gate_preserved_failed": True,
        "target_diagnostics": target_diagnostics,
        "protocol": protocol,
        "stage1_marker_snapshot": {
            "stage1_created_utc": stage1_done.get("created_utc"),
            "stage1_episodes": stage1_done.get("episodes"),
            "stage1_control_steps": stage1_done.get("control_steps"),
            "postdiagnostic_created_utc": post_done.get("created_utc"),
            "postdiagnostic_decision": post_done.get("decision"),
            "runner_material_cases": post_raw.get("runner_material_cases"),
            "predeclared_stage2_trigger_candidate": post_raw.get("predeclared_stage2_trigger_candidate"),
        },
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pid": os.getpid()},
        "interpretation_limits": ["prepare/dry-run only", "no branch continuation results yet", "not validation/model selection", "not final test", "does not rescue failed Stage1 gate"],
    }
    write_protocol_and_summary(raw)
    write_json(REQUEST_BACKUP, {
        "requested_utc": created,
        "reason": "backup v1b matched-continuation prepare/protocol/source before any 140-episode continuation rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "planned_rollout_episodes": protocol["budget"]["rollout_episodes_exact"],
        "planned_control_step_upper_bound": protocol["budget"]["control_step_upper_bound"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(Path(__file__).resolve()), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(OUT_DIR), rel(STATE_PATH), rel(REQUEST_BACKUP)],
    })
    raw["backup_request_before_rollout"] = rel(REQUEST_BACKUP)
    raw["source_hashes"] = source_hashes()
    write_json(OUT_DIR / "raw.json", raw)
    # Rewrite summaries after backup-request path/hash is present.
    write_protocol_and_summary(raw)
    state_text = (
        f"# Vehicle stress-v1b matched-continuation prepare ({created})\n\n"
        f"Prepared/dry-ran the v1b matched-continuation diagnostic with {len(targets)} H15-prefix targets "
        f"and {protocol['budget']['rollout_episodes_exact']} planned branch continuations (cap {protocol['budget']['control_step_upper_bound']} control steps). "
        "No simulations, no training/refit, no validation64-bank access and no sealed-test access. "
        "The stress-v1 Stage1 diversity gate remains failed; this diagnostic is a development postdiagnostic to decide whether within-episode labels exist before any selector/refit. "
        f"Rollout is blocked until a verified backup covers `{rel(REQUEST_BACKUP)}` and the new protocol/source artifacts.\n"
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(state_text, encoding="utf-8")
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-v1b matched-continuation prepare

UTC: {created}. Analysis-only prepare/dry-run froze a matched-continuation postdiagnostic after stress-v1 Stage1 failed its diversity gate. Targets={len(targets)} (positive cases {TARGET_CASES_POSITIVE}, same-stratum controls {TARGET_CASES_SAME_STRATUM_CONTROLS}, lower-stress controls {TARGET_CASES_LOWER_STRESS_CONTROLS}); branch horizons={BRANCH_HORIZONS}; planned continuations={protocol['budget']['rollout_episodes_exact']}; control-step cap={protocol['budget']['control_step_upper_bound']}. No simulations, no training/refit, no validation64-bank access and no sealed-test access. Rollout is backup-blocked until external backup covers `{rel(REQUEST_BACKUP)}`, `{rel(PROTOCOL_JSON)}`, `{rel(PROTOCOL_MD)}`, `{rel(OUT_DIR)}` and this runner source.
"""
    append_docs(block)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), PROTOCOL_JSON, PROTOCOL_MD, STATE_PATH, REQUEST_BACKUP, STAGE1_COMPLETED, POST_COMPLETED, STAGE1_BANK]
    write_json(OUT_DIR / "completed.json", {
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
        "new_candidate_resets": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "stage1_gate_preserved_failed": True,
        "target_count": len(targets),
        "planned_rollout_episodes": protocol["budget"]["rollout_episodes_exact"],
        "planned_control_step_upper_bound": protocol["budget"]["control_step_upper_bound"],
        "backup_required_before_rollout": True,
        "backup_request": rel(REQUEST_BACKUP),
        "protocol_json": rel(PROTOCOL_JSON),
        "protocol_json_sha256": sha256(PROTOCOL_JSON),
        "protocol_md": rel(PROTOCOL_MD),
        "protocol_md_sha256": sha256(PROTOCOL_MD),
        "next_after_backup": "write/run the bounded v1b continuation rollout runner or extend the prepared source to rollout mode under legacy interpreter; do not train until label gate is met",
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "protocol_json": rel(PROTOCOL_JSON),
        "target_count": len(targets),
        "planned_branch_continuations": protocol["budget"]["rollout_episodes_exact"],
        "control_step_upper_bound": protocol["budget"]["control_step_upper_bound"],
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_request": rel(REQUEST_BACKUP),
        "next": "await_verified_external_backup_before_v1b_rollout",
    }, sort_keys=True), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-and-dry-run", action="store_true", help="freeze v1b protocol/targets; no simulations")
    ap.add_argument("--i-accept-development-diagnostic", action="store_true", help="acknowledge development-only, no validation/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_development_diagnostic:
        raise ContractError("explicit --i-accept-development-diagnostic required")
    if not args.prepare_and_dry_run:
        raise ContractError("only --prepare-and-dry-run mode is implemented in this source")
    return prepare_and_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
