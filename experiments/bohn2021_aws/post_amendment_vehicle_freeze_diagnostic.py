#!/usr/bin/env python3
"""Post-amendment vehicle candidate/timing freeze diagnostic.

Metadata-only diagnostic.  It freezes the next *development* vehicle-only
AWS paired timing-smoke design and audits candidate/comparator availability after
migration.  It performs no simulation, does not read validation64 scenario files,
does not read sealed test files, and does not inspect any validation/test outcome
folders.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17"
LAT = ART / "results/latency_tree_2026-09-26"
TRAIN = LAT / "train"
BANKS = LAT / "banks"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic"
OUT_JSON = OUT_DIR / "raw.json"
OUT_MD = OUT_DIR / "summary.md"
GATE_JSON = OUT_DIR / "vehicle_development_timing_freeze.json"
SEEDS = (0, 1, 2)
H_GRID = tuple(range(5, 51, 5))
MARKER = "post-amendment-vehicle-freeze-diagnostic-20260926"


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_only(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists(), "sha256_computed": False}
    if path.exists():
        st = path.stat()
        out.update(size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return out


def file_record(path: Path, *, hash_file: bool = True) -> Dict[str, Any]:
    out = stat_only(path)
    if path.is_file() and hash_file:
        out["sha256"] = sha256(path)
        out["sha256_computed"] = True
    return out


def classify_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    kind = policy.get("kind")
    if kind == "constant":
        h = policy.get("h")
        return {
            "kind": "constant",
            "leaves": [h],
            "structural_class": f"fixed_H{h}",
            "requires_evaluation_for_actual_switching": False,
            "prevalidation_adaptive_candidate": False,
        }
    if kind == "tree":
        leaves = list(policy.get("leaves") or [])
        distinct = sorted(set(leaves))
        if len(distinct) <= 1:
            label = f"behaviorally_fixed_by_structure_H{distinct[0] if distinct else 'unknown'}"
            return {
                "kind": "tree",
                "leaves": leaves,
                "distinct_leaves": distinct,
                "structural_class": label,
                "requires_evaluation_for_actual_switching": False,
                "prevalidation_adaptive_candidate": False,
            }
        return {
            "kind": "tree",
            "leaves": leaves,
            "distinct_leaves": distinct,
            "structural_class": "structurally_switching_tree",
            "requires_evaluation_for_actual_switching": True,
            "prevalidation_adaptive_candidate": True,
        }
    return {
        "kind": kind,
        "structural_class": "unknown_policy_kind",
        "requires_evaluation_for_actual_switching": True,
        "prevalidation_adaptive_candidate": False,
    }


def compact_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    if policy.get("kind") == "constant":
        return {"kind": "constant", "task": policy.get("task"), "h": policy.get("h")}
    if policy.get("kind") == "tree":
        return {"kind": "tree", "task": policy.get("task"), "nodes": policy.get("nodes"), "leaves": policy.get("leaves")}
    return dict(policy)


def policy_sha256(policy: Dict[str, Any]) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_record(path: Path, *, family: str, seed: int, h: int, role: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"role": role, "family": family, "seed": seed, "h": h, "path": rel(path), "exists": path.exists()}
    for name in ("manifest.json", "completed.json", "model.zip"):
        rec[name] = file_record(path / name) if (path / name).exists() else {"path": rel(path / name), "exists": False, "sha256_computed": False}
    if (path / "manifest.json").exists():
        try:
            manifest = read_json(path / "manifest.json")
            rec["manifest_summary"] = {
                "task": manifest.get("task"),
                "seed": manifest.get("seed"),
                "fixed_horizon": manifest.get("fixed_horizon"),
                "steps": manifest.get("steps"),
                "adaptations": manifest.get("adaptations"),
            }
        except Exception as exc:  # preserve diagnostic information without hiding failure
            rec["manifest_error"] = repr(exc)
    if (path / "completed.json").exists():
        try:
            done = read_json(path / "completed.json")
            rec["completed_summary"] = {
                "status": done.get("status"),
                "steps": done.get("steps"),
                "elapsed_s": done.get("elapsed_s"),
                "train_episodes": done.get("train_episodes"),
                "updates": done.get("updates"),
                "final_hash": done.get("final_hash"),
            }
        except Exception as exc:
            rec["completed_error"] = repr(exc)
    rec["available_complete"] = bool(
        rec["exists"]
        and rec["manifest.json"].get("exists")
        and rec["completed.json"].get("exists")
        and rec["model.zip"].get("exists")
        and (rec.get("completed_summary") or {}).get("status") == "complete"
        and (rec.get("completed_summary") or {}).get("steps") == 15000
    )
    return rec


def primary_model_path(seed: int) -> Path:
    if seed == 0:
        return ART / "results/paper_exact_grid_2026-09-23/vehicle_fixed_h25"
    return ART / f"results/min_q_training_2026-09-24/vehicle_fixed_h25_s{seed}"


def independent_seed0_grid_path(h: int) -> Path:
    if h in (5, 10, 15):
        return ART / f"results/paper_defaults/vehicle_fixed_h{h}"
    return ART / f"results/paper_exact_grid_2026-09-23/vehicle_fixed_h{h}"


def load_vehicle_candidates() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        folder = TRAIN / f"vehicle_s{seed}"
        policy_path = folder / "policy.json"
        completed_path = folder / "completed.json"
        selection_path = folder / "selection_registration.json"
        policy = read_json(policy_path)
        done = read_json(completed_path)
        selection = read_json(selection_path)
        rows.append({
            "arm_id": f"learned_latency_tree_vehicle_s{seed}",
            "task": "vehicle",
            "seed": seed,
            "family": "learned_latency_tree_IMPROVED",
            "policy": compact_policy(policy),
            "policy_file": file_record(policy_path),
            "policy_canonical_sha256": policy_sha256(policy),
            "completed_file": file_record(completed_path),
            "selection_registration_file": file_record(selection_path),
            "selected_training_arm": done.get("selected"),
            "registered_arm_ids": [a.get("id") for a in selection.get("arms", []) if isinstance(a, dict)],
            "registered_finalist_ids": [a.get("id") for a in selection.get("finalists", []) if isinstance(a, dict)],
            "classification": classify_policy(policy),
            "interpretation_before_validation": "candidate only; behavior and effect must be remeasured on AWS paired block before any adaptive claim",
        })
    return rows


def fixed_comparator_inventory() -> Dict[str, Any]:
    primary = [model_record(primary_model_path(seed), family="primary_fixed_H25_seed_terminal", seed=seed, h=25, role="fixed_h25_primary") for seed in SEEDS]
    matched_grid = []
    for seed in SEEDS:
        source = primary_model_path(seed)
        for h in H_GRID:
            rec = model_record(source, family="matched_terminal_fixed_H_grid", seed=seed, h=h, role=f"matched_terminal_H{h}")
            rec["controller_horizon_varies_without_new_terminal_training"] = True
            matched_grid.append(rec)
    independent_seed0_grid = [
        model_record(independent_seed0_grid_path(h), family="independent_terminal_seed0_H_grid", seed=0, h=h, role=f"independent_seed0_H{h}")
        for h in H_GRID
    ]
    independent_selected_seed1_seed2 = {
        "available_now": [model_record(primary_model_path(seed), family="independent_terminal_selected_if_H25", seed=seed, h=25, role="independent_seed1_seed2_H25_available") for seed in (1, 2)],
        "non_H25_status": "not preselected and not trained in this post-amendment gate; if validation selects an independent-terminal H other than 25, seed1/2 must be trained on AWS/frozen provenance before final comparison",
    }
    return {
        "primary_fixed_h25_all_seeds": primary,
        "matched_terminal_full_h_grid_all_seeds": matched_grid,
        "independent_terminal_full_h_grid_seed0": independent_seed0_grid,
        "independent_terminal_seed1_seed2_followup_rule": independent_selected_seed1_seed2,
        "availability_summary": {
            "primary_h25_all_three_complete": all(r["available_complete"] for r in primary),
            "matched_grid_all_three_has_terminal_source": all(r["available_complete"] for r in matched_grid),
            "independent_seed0_full_grid_complete": all(r["available_complete"] for r in independent_seed0_grid),
        },
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "experiments/bohn2021_reproduction/latency_tree_protocol.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_policy.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_run.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_evaluate.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_evaluation_spec.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_iteration_timing.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_shared_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/relative_policy_features.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/min_q_eval_suite.py",
        ROOT / "docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md",
        ROOT / "docs/protocols/bohn2021_latency_tree_2026-09-26.md",
        ART / "configs/vehicle.json",
        LAT / "registration.json",
        LAT / "status.json",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def split_records() -> Dict[str, Any]:
    # Intentionally hash only the smoke bank.  Validation/test bank content is not
    # opened by this diagnostic; existence and byte size are enough for ID/gate
    # accounting until a formal validation-read gate is published.
    return {
        "development_smoke_bank": {
            "id": "latency_tree_2026-09-26 vehicle_smoke_bank",
            "record": file_record(BANKS / "vehicle_smoke_bank.json", hash_file=True),
            "outcome_split": False,
            "may_be_used_for_engineering_smoke": True,
        },
        "validation64": {
            "id": "latency_tree_2026-09-26 vehicle_validation_bank",
            "record": stat_only(BANKS / "vehicle_validation_bank.json"),
            "content_opened_by_this_diagnostic": False,
            "outcomes_opened_by_this_diagnostic": False,
            "authorization_to_rollout": False,
        },
        "sealed_test128": {
            "id": "latency_tree_2026-09-26 vehicle_test_bank",
            "record": stat_only(BANKS / "vehicle_test_bank.json"),
            "content_opened_by_this_diagnostic": False,
            "outcomes_opened_by_this_diagnostic": False,
            "authorization_to_rollout": False,
        },
    }


def build_plan(candidates: List[Dict[str, Any]], comparators: Dict[str, Any]) -> Dict[str, Any]:
    dev_arms_minimal = []
    for seed in SEEDS:
        dev_arms_minimal.append({"arm_id": f"learned_latency_tree_vehicle_s{seed}", "seed": seed, "policy_source": "train/vehicle_s{seed}/policy.json"})
        dev_arms_minimal.append({"arm_id": f"fixed_H25_primary_vehicle_s{seed}", "seed": seed, "h": 25, "policy": {"kind": "constant", "task": "vehicle", "h": 25}})
    dev_arms_full_grid = []
    for seed in SEEDS:
        dev_arms_full_grid.append({"arm_id": f"learned_latency_tree_vehicle_s{seed}", "seed": seed, "family": "learned"})
        for h in H_GRID:
            dev_arms_full_grid.append({"arm_id": f"matched_terminal_fixed_H{h}_vehicle_s{seed}", "seed": seed, "family": "matched_terminal_fixed_grid", "h": h})
    cases = 2
    repeats = 2
    return {
        "decision": "Do not open validation64 yet. Next informative actual experiment is an AWS-only no-validation vehicle timing/control smoke block; pendulum_s1/s2 recovery remains queued but vehicle is prioritized because all three vehicle policies exist.",
        "development_block_minimal": {
            "status": "frozen_by_this_metadata_diagnostic_for_next_run",
            "formal_scientific_evidence": False,
            "split": "vehicle_smoke_bank only; not validation64 and not sealed test128",
            "cases": cases,
            "repeats": repeats,
            "randomized_order_seed": 2609268200,
            "arms": dev_arms_minimal,
            "episode_budget_exact": len(dev_arms_minimal) * cases * repeats,
            "control_step_upper_bound": len(dev_arms_minimal) * cases * repeats * 150,
            "purpose": "check AWS timing boundaries, replay determinism, solver/failure accounting, and whether seed2 structurally switching policy actually changes H on unseen smoke states; not model selection",
        },
        "development_block_expanded_if_minimal_passes": {
            "formal_scientific_evidence": False,
            "split": "vehicle_smoke_bank only",
            "cases": cases,
            "repeats": repeats,
            "arms": dev_arms_full_grid,
            "episode_budget_exact": len(dev_arms_full_grid) * cases * repeats,
            "control_step_upper_bound": len(dev_arms_full_grid) * cases * repeats * 150,
            "purpose": "pre-validation timing sanity across full matched-terminal H grid; still not validation/model selection",
        },
        "formal_validation_prerequisites": {
            "vehicle_only_candidate_set_freezable_now": True,
            "whole_two_task_gate_blocked_by_pendulum": True,
            "blocking_reason": "pendulum_s1 interrupted and pendulum_s2 unstarted, so all-seed two-task formal validation is incomplete unless AWS-only pendulum recovery is run or scope is explicitly vehicle-only development",
            "before_any_validation64_rollout": [
                "publish validation gate JSON with immutable code/config/model hashes",
                "freeze exact candidate and comparator arms",
                "freeze checkpoint policy: final saved policies only; no validation checkpoint tuning",
                "freeze model-selection rule before observing validation results",
                "confirm backup of new gate artifacts",
                "run no-validation smoke/replay audit if using amended AWS timing code",
            ],
        },
        "formal_vehicle_validation_candidate_set_draft": {
            "adaptive_candidates": [c["arm_id"] for c in candidates],
            "primary_comparator": "fixed_H25_primary all three vehicle seeds",
            "matched_terminal_grid": "H=5,10,...,50 for all three vehicle seeds using each seed's primary terminal",
            "independent_terminal_grid": "seed0 H=5,10,...,50; if validation nominates non-H25 independent H, train seed1/2 terminals under new AWS/frozen budget before final comparison",
            "model_selection_metric_draft": "safety first (success nondecrease; constraints and initial/final solver failures nonincrease), then raw complete-episode total cost; timing reported separately unless a predeclared timing route is selected",
            "behavior_rule": "a tree with only one actually used H is classified as fixed-H for that evaluation block; seed0 all-H25 tree and seed1 constant are not adaptive evidence unless behavior changes, which their structures preclude",
        },
        "timing_boundaries_to_use": {
            "include": ["policy feature/context extraction", "tree/constant H selection", "controller get_action", "all NLP retries", "feasibility/recovery checks"],
            "exclude_but_report": ["environment construction", "episode reset", "outer trace write/audit overhead"],
            "report": ["decision wall time mean/median/p95", "solver/controller gross and logging-deducted time", "reset time", "deadline exceed steps", "initial/final solver failures", "episode failures", "constraint episodes", "complete episode cost"],
        },
    }


def append_docs(summary: Dict[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-26 post-amendment vehicle freeze diagnostic\n\n"
        f"UTC: {summary['created_utc']}. Metadata-only diagnostic completed with no simulations, no validation64 content/outcome read, and no sealed-test read. "
        "Vehicle learned candidates s0/s1/s2 and fixed-H comparator inventory were hashed. "
        "Next actual experiment is frozen as a non-formal AWS-only vehicle smoke paired timing/control block versus fixed H25; formal validation remains unopened and whole two-task validation remains blocked by pendulum_s1/s2 recovery. "
        f"Artifacts: `{rel(OUT_JSON)}`, `{rel(OUT_MD)}`, `{rel(GATE_JSON)}`. New artifacts require backup before unique formal evidence accumulates.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def write_summary(payload: Dict[str, Any]) -> None:
    candidates = payload["vehicle_candidates"]
    comp = payload["fixed_comparators"]["availability_summary"]
    plan = payload["frozen_next_plan"]
    lines: List[str] = []
    lines.extend([
        "# Post-amendment vehicle candidate/timing freeze diagnostic",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "Scope: metadata only. No simulations were run. The script did not open validation64 bank content or sealed test bank content, and did not inspect validation/test outcome folders.",
        "",
        "## Candidate policies",
        "",
        "| seed | selected training arm | policy sha256 | structural class | leaves | adaptive candidate before validation? |",
        "|---:|---|---|---|---|---:|",
    ])
    for c in candidates:
        cls = c["classification"]
        lines.append(
            f"| {c['seed']} | `{c.get('selected_training_arm')}` | `{c['policy_file'].get('sha256')}` | "
            f"{cls.get('structural_class')} | {cls.get('leaves')} | {cls.get('prevalidation_adaptive_candidate')} |"
        )
    lines.extend([
        "",
        "## Fixed comparator availability",
        "",
    ])
    for k, v in comp.items():
        lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "## Frozen next actual experiment",
        "",
        f"Decision: {plan['decision']}",
        "",
        "Minimal smoke block (non-formal):",
        f"- split: {plan['development_block_minimal']['split']}",
        f"- arms: {len(plan['development_block_minimal']['arms'])}",
        f"- episodes: {plan['development_block_minimal']['episode_budget_exact']}",
        f"- control-step upper bound: {plan['development_block_minimal']['control_step_upper_bound']}",
        f"- randomized order seed: {plan['development_block_minimal']['randomized_order_seed']}",
        "",
        "Expanded no-validation grid block if minimal smoke passes:",
        f"- arms: {len(plan['development_block_expanded_if_minimal_passes']['arms'])}",
        f"- episodes: {plan['development_block_expanded_if_minimal_passes']['episode_budget_exact']}",
        f"- control-step upper bound: {plan['development_block_expanded_if_minimal_passes']['control_step_upper_bound']}",
        "",
        "Formal validation remains unopened. This diagnostic is not a final validation gate and not test authorization.",
        "",
        "## Timing/metric boundaries",
        "",
    ])
    for k, values in plan["timing_boundaries_to_use"].items():
        lines.append(f"- {k}: {values}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_JSON.exists() or OUT_MD.exists() or GATE_JSON.exists():
        raise SystemExit("Refusing to overwrite existing post-amendment vehicle freeze diagnostic; inspect existing artifacts first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_vehicle_candidates()
    comparators = fixed_comparator_inventory()
    plan = build_plan(candidates, comparators)
    payload: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "IMPROVED latency-tree vehicle post-amendment metadata freeze; not original SAC",
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "source_hashes": source_hashes(),
        "split_records": split_records(),
        "vehicle_candidates": candidates,
        "fixed_comparators": comparators,
        "frozen_next_plan": plan,
        "unresolved_gates": [
            "Backup new diagnostic artifacts before accumulating unique formal evidence.",
            "Implement/run the no-validation vehicle timing smoke block; do not treat it as validation evidence.",
            "Do not open validation64 until a separate validation gate JSON freezes exact runnable code/config/model hashes and checkpoint/model-selection policy.",
            "Final sealed test remains unauthorized until final_test_gate.json and explicit update_state authorization request after independent validation gate.",
            "Whole two-task latency-tree validation remains blocked by pendulum_s1 interrupted and pendulum_s2 unstarted unless scope is explicitly vehicle-only development.",
        ],
    }
    write_json(OUT_JSON, payload)
    write_json(GATE_JSON, {
        "created_utc": payload["created_utc"],
        "status": "development_timing_smoke_freeze_only_not_formal_validation_gate",
        "validation_accessed": False,
        "test_accessed": False,
        "candidate_policy_hashes": {c["arm_id"]: c["policy_file"].get("sha256") for c in candidates},
        "candidate_policy_canonical_hashes": {c["arm_id"]: c["policy_canonical_sha256"] for c in candidates},
        "comparator_availability_summary": comparators["availability_summary"],
        "minimal_development_block": plan["development_block_minimal"],
        "formal_validation_prerequisites": plan["formal_validation_prerequisites"],
    })
    write_summary(payload)
    append_docs(payload)
    print(json.dumps({
        "created": rel(OUT_JSON),
        "summary": rel(OUT_MD),
        "freeze": rel(GATE_JSON),
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "minimal_smoke_episodes": plan["development_block_minimal"]["episode_budget_exact"],
        "expanded_smoke_episodes": plan["development_block_expanded_if_minimal_passes"]["episode_budget_exact"],
        "primary_h25_all_three_complete": comparators["availability_summary"]["primary_h25_all_three_complete"],
        "matched_grid_all_three_has_terminal_source": comparators["availability_summary"]["matched_grid_all_three_has_terminal_source"],
        "independent_seed0_full_grid_complete": comparators["availability_summary"]["independent_seed0_full_grid_complete"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
