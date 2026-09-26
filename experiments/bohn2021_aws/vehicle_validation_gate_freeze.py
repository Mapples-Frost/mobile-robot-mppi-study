#!/usr/bin/env python3
"""Freeze a no-validation vehicle formal-validation gate for latency-tree work.

This is a metadata/protocol diagnostic only.  It freezes the exact vehicle-only
validation candidate set, comparator set, model/source hashes, validation split
identifier, selection rules, metrics, budgets, sharding plan, and success/failure
thresholds before any post-amendment validation64 rollout is performed.

It intentionally does not read vehicle_validation_bank.json or vehicle_test_bank
content.  It uses only registration metadata, banks/completed.json hash records,
training/model metadata, source files, and already completed no-validation smoke
artifacts.  It performs no simulations, no gradient updates, no validation
rollouts, and no sealed-test access.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17"
LAT = ART / "results/latency_tree_2026-09-26"
BANKS = LAT / "banks"
TRAIN = LAT / "train"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926"
GATE_JSON = OUT_DIR / "vehicle_validation_gate_20260926.json"
SUMMARY_MD = OUT_DIR / "summary.md"
COMPLETED_JSON = OUT_DIR / "completed.json"
MARKER = "vehicle-validation-gate-freeze-20260926"
SEEDS = (0, 1, 2)
H_GRID = tuple(range(5, 51, 5))
VALIDATION_CASES = 64
TEST_CASES = 128
VALIDATION_SCHEDULE_SEED = 2609268400
TIMING_REPLAY_SCHEDULE_SEED = 2609268401
BOOTSTRAP_COST_SEED = 2609268100
BOOTSTRAP_TIMING_SEED = 2609268101
BOOTSTRAP_DRAWS = 10000
SHARD_MAX_EPISODES = 224


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
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def stat_only(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists(), "content_opened": False, "sha256_computed_now": False}
    if path.exists():
        st = path.stat()
        out.update(size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return out


def file_record(path: Path, *, role: str | None = None, read_hash: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if role is not None:
        out["role"] = role
    if path.exists():
        st = path.stat()
        out.update(size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        if read_hash:
            out["sha256"] = sha256(path)
            out["sha256_computed_now"] = True
        else:
            out["content_opened"] = False
            out["sha256_computed_now"] = False
    return out


def assert_no_prior_partial() -> None:
    if not OUT_DIR.exists():
        return
    if COMPLETED_JSON.exists():
        done = read_json(COMPLETED_JSON)
        assert done.get("passed") is True, "existing gate completed marker did not pass"
        for name, expected in done.get("hashes", {}).items():
            actual = sha256(ROOT / name)
            assert actual == expected, name
        raise SystemExit("vehicle validation gate already completed and verified; refusing to rerun")
    leftovers = [p.name for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    assert not leftovers, "Partial vehicle validation gate output exists; inspect before recovery: " + ", ".join(sorted(leftovers))


def policy_classification(policy: Dict[str, Any]) -> Dict[str, Any]:
    kind = policy.get("kind")
    if kind == "constant":
        h = int(policy.get("h"))
        return {
            "kind": "constant",
            "structural_class": f"fixed_H{h}",
            "leaves": [h],
            "distinct_leaves": [h],
            "can_be_adaptive_on_validation": False,
            "predeclared_interpretation": "fixed-H comparator-like learned selection; not adaptive evidence",
        }
    if kind == "tree":
        leaves = [int(x) for x in (policy.get("leaves") or [])]
        distinct = sorted(set(leaves))
        if len(distinct) <= 1:
            h = distinct[0] if distinct else "unknown"
            return {
                "kind": "tree",
                "structural_class": f"behaviorally_fixed_by_structure_H{h}",
                "leaves": leaves,
                "distinct_leaves": distinct,
                "can_be_adaptive_on_validation": False,
                "predeclared_interpretation": "tree JSON but all leaves identical; classify as fixed-H for every block",
            }
        return {
            "kind": "tree",
            "structural_class": "structurally_switching_tree",
            "leaves": leaves,
            "distinct_leaves": distinct,
            "can_be_adaptive_on_validation": True,
            "predeclared_interpretation": "adaptive candidate only if validation trace actually uses >=2 horizons and has within-block/within-episode changes",
        }
    return {
        "kind": kind,
        "structural_class": "unknown_policy_kind",
        "leaves": [],
        "distinct_leaves": [],
        "can_be_adaptive_on_validation": False,
        "predeclared_interpretation": "unknown; cannot support adaptive claim without explicit audit",
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "experiments/bohn2021_aws/post_amendment_vehicle_freeze_diagnostic.py",
        ROOT / "experiments/bohn2021_aws/vehicle_development_smoke_pairing.py",
        ROOT / "experiments/bohn2021_aws/vehicle_smoke_artifact_digest.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_protocol.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_policy.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_run.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_evaluate.py",
        ROOT / "experiments/bohn2021_reproduction/latency_tree_evaluation_spec.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/relative_policy_features.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
        ROOT / "docs/protocols/bohn2021_latency_tree_2026-09-26.md",
        ROOT / "docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md",
        ART / "configs/vehicle.json",
        LAT / "registration.json",
        LAT / "banks/completed.json",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def vehicle_bank_record(hash_map: Dict[str, str], suffix: str) -> Dict[str, Any]:
    matches = [(name, digest) for name, digest in hash_map.items() if name.endswith("/" + suffix) or name.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    current = BANKS / suffix
    return {
        "bank_file": stat_only(current),
        "recorded_hash_source_path": matches[0][0],
        "recorded_sha256_from_banks_completed": matches[0][1],
        "content_opened_by_this_gate": False,
        "outcomes_opened_by_this_gate": False,
    }


def primary_model_path(seed: int) -> Path:
    if seed == 0:
        return ART / "results/paper_exact_grid_2026-09-23/vehicle_fixed_h25"
    return ART / f"results/min_q_training_2026-09-24/vehicle_fixed_h25_s{seed}"


def independent_seed0_model_path(h: int) -> Path:
    if h in (5, 10, 15):
        return ART / f"results/paper_defaults/vehicle_fixed_h{h}"
    return ART / f"results/paper_exact_grid_2026-09-23/vehicle_fixed_h{h}"


def model_record(path: Path, *, seed: int, h: int, family: str, role: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"role": role, "family": family, "seed": seed, "h": h, "path": rel(path), "exists": path.exists()}
    for name in ("manifest.json", "completed.json", "model.zip"):
        rec[name] = file_record(path / name, role=name)
    if (path / "manifest.json").exists():
        manifest = read_json(path / "manifest.json")
        rec["manifest_summary"] = {
            "task": manifest.get("task"),
            "seed": manifest.get("seed"),
            "fixed_horizon": manifest.get("fixed_horizon"),
            "steps": manifest.get("steps"),
            "adaptations": manifest.get("adaptations"),
        }
    if (path / "completed.json").exists():
        done = read_json(path / "completed.json")
        rec["completed_summary"] = {
            "status": done.get("status"),
            "steps": done.get("steps"),
            "elapsed_s": done.get("elapsed_s"),
            "train_episodes": done.get("train_episodes"),
            "updates": done.get("updates"),
            "final_hash": done.get("final_hash"),
        }
    rec["available_complete"] = bool(
        rec["exists"]
        and rec["manifest.json"].get("exists")
        and rec["completed.json"].get("exists")
        and rec["model.zip"].get("exists")
        and (rec.get("manifest_summary") or {}).get("task") == "vehicle"
        and (rec.get("manifest_summary") or {}).get("seed") == seed
        and (rec.get("manifest_summary") or {}).get("fixed_horizon") == h
        and (rec.get("manifest_summary") or {}).get("steps") == 15000
        and (rec.get("completed_summary") or {}).get("status") == "complete"
        and (rec.get("completed_summary") or {}).get("steps") == 15000
    )
    return rec


def load_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for seed in SEEDS:
        folder = TRAIN / f"vehicle_s{seed}"
        policy_path = folder / "policy.json"
        completed_path = folder / "completed.json"
        fit_path = folder / "fit.json"
        selection_path = folder / "selection_registration.json"
        policy = read_json(policy_path)
        completed = read_json(completed_path)
        fit = read_json(fit_path)
        selection = read_json(selection_path)
        assert completed.get("passed") is True
        assert fit.get("validation_access") is False and fit.get("test_access") is False
        candidates.append({
            "arm_id": f"learned_latency_tree_vehicle_s{seed}",
            "method_class": "IMPROVED_latency_tree_not_ORIGINAL_SAC",
            "task": "vehicle",
            "seed": seed,
            "policy": policy,
            "policy_path": rel(policy_path),
            "policy_file_sha256": sha256(policy_path),
            "policy_canonical_sha256": canonical_sha(policy),
            "completed_json": file_record(completed_path, role="latency_tree_training_completed"),
            "fit_json": file_record(fit_path, role="training_fit_no_validation_no_test"),
            "selection_registration_json": file_record(selection_path, role="training_selection_registration"),
            "selected_training_arm": completed.get("selected"),
            "training_validation_access": bool(fit.get("validation_access")),
            "training_test_access": bool(fit.get("test_access")),
            "registered_finalists": [x.get("id") for x in selection.get("finalists", []) if isinstance(x, dict)],
            "classification": policy_classification(policy),
        })
    return candidates


def comparator_inventory() -> Dict[str, Any]:
    primary = [
        model_record(primary_model_path(seed), seed=seed, h=25, family="primary_fixed_H25_independent_terminal", role="primary_fixed_H25")
        for seed in SEEDS
    ]
    matched_grid: List[Dict[str, Any]] = []
    for seed in SEEDS:
        terminal_source = primary_model_path(seed)
        for h in H_GRID:
            rec = model_record(terminal_source, seed=seed, h=25, family="matched_terminal_fixed_H_grid", role=f"matched_terminal_fixed_H{h}")
            rec["controller_h"] = h
            rec["terminal_h"] = 25
            rec["policy"] = {"kind": "constant", "task": "vehicle", "h": h}
            rec["policy_canonical_sha256"] = canonical_sha(rec["policy"])
            rec["available_for_rollout"] = bool(rec["available_complete"])
            matched_grid.append(rec)
    independent_seed0_grid: List[Dict[str, Any]] = []
    for h in H_GRID:
        source = independent_seed0_model_path(h)
        rec = model_record(source, seed=0, h=h, family="independent_terminal_seed0_H_grid", role=f"independent_terminal_seed0_H{h}")
        rec["controller_h"] = h
        rec["terminal_h"] = h
        rec["policy"] = {"kind": "constant", "task": "vehicle", "h": h}
        rec["policy_canonical_sha256"] = canonical_sha(rec["policy"])
        rec["available_for_rollout"] = bool(rec["available_complete"])
        independent_seed0_grid.append(rec)
    return {
        "primary_fixed_H25_per_seed": primary,
        "matched_terminal_full_grid_all_seeds": matched_grid,
        "independent_terminal_full_grid_seed0": independent_seed0_grid,
        "independent_terminal_seed1_seed2_followup_rule": {
            "rule": "If validation nominates an independent-terminal H other than 25, do not use seed0-only evidence for a three-seed claim. First freeze a new AWS training amendment, train vehicle seed1/seed2 fixed-H terminals for that nominated H with the original 15k-step settings, then run fresh validation for those comparators before any final-test gate. If H25 is nominated, the existing primary seed1/seed2 terminals are the independent-terminal comparators.",
            "existing_H25_seed1_seed2_sources": [model_record(primary_model_path(seed), seed=seed, h=25, family="independent_terminal_H25_followup_available", role="independent_terminal_seed1_seed2_H25") for seed in (1, 2)],
        },
        "availability_summary": {
            "primary_fixed_H25_all_three_complete": all(r["available_complete"] for r in primary),
            "matched_terminal_grid_all_entries_have_complete_H25_terminal_source": all(r["available_for_rollout"] for r in matched_grid),
            "independent_terminal_seed0_full_grid_complete": all(r["available_for_rollout"] for r in independent_seed0_grid),
        },
    }


def logical_arms() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return logical arms and deduplicated rollout arms without reading validation cases."""
    logical: List[Dict[str, Any]] = []
    rollout_by_key: Dict[str, Dict[str, Any]] = {}

    def add(arm: Dict[str, Any]) -> None:
        logical.append(arm)
        key = arm["rollout_key"]
        if key not in rollout_by_key:
            rollout_by_key[key] = dict(arm, logical_aliases=[arm["arm_id"]])
        else:
            rollout_by_key[key]["logical_aliases"].append(arm["arm_id"])

    for seed in SEEDS:
        add({
            "arm_id": f"learned_latency_tree_vehicle_s{seed}",
            "role": "learned_candidate",
            "seed": seed,
            "family": "learned_latency_tree_IMPROVED",
            "controller_h": "policy_selected_each_step",
            "terminal_source": rel(primary_model_path(seed)),
            "rollout_key": f"learned_s{seed}",
        })
    for seed in SEEDS:
        for h in H_GRID:
            role = "primary_fixed_H25" if h == 25 else "matched_terminal_fixed_grid"
            add({
                "arm_id": f"matched_terminal_fixed_H{h}_vehicle_s{seed}",
                "role": role,
                "seed": seed,
                "family": "matched_terminal_fixed_H_grid",
                "controller_h": h,
                "terminal_h": 25,
                "terminal_source": rel(primary_model_path(seed)),
                "rollout_key": f"fixed_seed{seed}_terminal25_controllerH{h}",
            })
    for h in H_GRID:
        if h == 25:
            rollout_key = "fixed_seed0_terminal25_controllerH25"
            duplicate_of = "matched_terminal_fixed_H25_vehicle_s0"
        else:
            rollout_key = f"fixed_seed0_terminal{h}_controllerH{h}"
            duplicate_of = None
        add({
            "arm_id": f"independent_terminal_seed0_fixed_H{h}",
            "role": "independent_terminal_seed0_grid",
            "seed": 0,
            "family": "independent_terminal_seed0_H_grid",
            "controller_h": h,
            "terminal_h": h,
            "terminal_source": rel(independent_seed0_model_path(h)),
            "rollout_key": rollout_key,
            "duplicate_rollout_of": duplicate_of,
        })
    return logical, list(rollout_by_key.values())


def build_schedule(rollout_arms: List[Dict[str, Any]]) -> Dict[str, Any]:
    episodes = []
    for arm_index, arm in enumerate(rollout_arms):
        for case in range(VALIDATION_CASES):
            episodes.append({
                "rollout_index": len(episodes),
                "arm_index": arm_index,
                "rollout_key": arm["rollout_key"],
                "primary_arm_id": arm["arm_id"],
                "logical_aliases": arm["logical_aliases"],
                "seed": arm["seed"],
                "case_index": case,
            })
    rng = np.random.RandomState(VALIDATION_SCHEDULE_SEED)
    order = rng.permutation(len(episodes)).tolist()
    randomized = []
    for execution_index, original_index in enumerate(order):
        row = dict(episodes[original_index])
        row["execution_index"] = execution_index
        randomized.append(row)
    shards = []
    for start in range(0, len(randomized), SHARD_MAX_EPISODES):
        rows = randomized[start:start + SHARD_MAX_EPISODES]
        shards.append({
            "shard_index": len(shards),
            "start_execution_index": rows[0]["execution_index"],
            "end_execution_index_inclusive": rows[-1]["execution_index"],
            "episodes": len(rows),
            "status": "planned_not_run",
        })
    return {
        "schedule_seed": VALIDATION_SCHEDULE_SEED,
        "case_indices_only_no_validation_content": True,
        "logical_arm_count": len({alias for arm in rollout_arms for alias in arm["logical_aliases"]}),
        "unique_rollout_arm_count": len(rollout_arms),
        "episodes": randomized,
        "episode_count": len(randomized),
        "shard_max_episodes": SHARD_MAX_EPISODES,
        "shards": shards,
        "expected_shards": len(shards),
    }


def extract_bank_specs(registration: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in registration["spec"]["bank_specs"]:
        if item.get("task") == "vehicle" and item.get("split") in ("smoke", "validation", "test"):
            result[item["split"]] = dict(item)
    assert result["validation"]["cases"] == VALIDATION_CASES and result["test"]["cases"] == TEST_CASES
    return result


def protocol_rules() -> Dict[str, Any]:
    return {
        "scope": "vehicle-only formal validation gate for the IMPROVED latency-tree policies; not sufficient for whole two-task Bohn reproduction and not ORIGINAL SAC",
        "validation_use": "Open vehicle_validation_bank only after this gate is backed up and intentionally invoked by a validation runner. Validation may select fixed-H baselines and decide whether vehicle-only evidence is negative/partial/improved-only; it may not tune learned-tree parameters or checkpoints.",
        "test_use": "Final sealed vehicle_test_bank remains closed. This gate is not final_test_gate.json and does not request test authorization.",
        "checkpoint_policy": "Use final saved train/vehicle_s{0,1,2}/policy.json only. No validation checkpoint selection, no tree retuning, no policy edits after seeing validation outcomes.",
        "behavior_classification": "A tree that emits one horizon throughout validation is classified as fixed-H for that block. Adaptive success requires actual use of >=2 horizons and nonzero within-block/within-episode horizon changes for every learned seed supporting the claim.",
        "fixed_baseline_selection": {
            "primary": "Primary comparator is fixed H25 for each seed using the inherited independent H25 terminal.",
            "matched_terminal": "Evaluate H=5,10,...,50 for each seed using that seed's H25 terminal. Nominate one common H among candidates safe for all three seeds vs primary; choose lowest pooled mean raw complete-episode total cost, exact ties smaller H.",
            "independent_terminal": "Evaluate seed0 H=5,10,...,50 with independently trained seed0 terminals. Nominate a candidate safe vs seed0 primary; choose lowest seed0 mean raw complete-episode total cost, exact ties smaller H. If nominated H != 25, seed1/2 terminal training is required before any all-seed independent-terminal validation/final claim.",
        },
        "metrics_primary": [
            "complete episode raw total_cost",
            "physical_constraint_cost = performance_cost + constraint_cost",
            "success count and episode_failure count",
            "constraint episode count",
            "initial solver failure steps/rate",
            "final solver failure steps/rate",
        ],
        "metrics_secondary": [
            "decision wall time mean/median/p95 and deadline exceed steps",
            "solver attempt wall time mean/median/p95",
            "controller/logging/reset/construction timing boundaries",
            "horizon distribution and switches as behavior audit only",
            "per-seed paired scenario tables and all failed/interrupted work",
        ],
        "paired_uncertainty": {
            "unit": "validation scenario/case index, paired across arms and retained across trained seeds",
            "not_independent_units": ["control steps", "solver attempts", "timing repeats", "episode repeats within same case"],
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "cost_bootstrap_seed": BOOTSTRAP_COST_SEED,
            "timing_bootstrap_seed": BOOTSTRAP_TIMING_SEED,
            "report": "paired mean deltas/ratios with 95% intervals plus individual seed summaries; intervals are conditional on the trained models",
        },
        "success_thresholds": {
            "safety_noninferiority_each_seed_vs_each_nominated_comparator": "success nondecrease; constraint count nonincrease; initial and final solver-failure rates nonincrease by exact integer/rate comparison",
            "physical_noninferiority": "physical_constraint_cost no worse than +2% vs comparator, with exact per-protocol tolerance recorded by evaluator",
            "adaptive_behavior": "actual >=2 horizons and nonzero horizon changes; structurally fixed s0/s1 cannot satisfy adaptive evidence under this gate",
            "cost_route": ">=3% aggregate raw total-cost improvement for the task and paired 95% upper cost-delta < 0, while all safety/physical gates pass",
            "timing_route": ">=10% actual decision-time improvement, paired 95% upper timing ratio < 1, both formal timing repeats favorable, total_cost noninferiority within 2%, and no safety degradation",
            "vehicle_only_limitation": "Even a vehicle pass would be partial/improved-only evidence until pendulum recovery and sealed final-test gate requirements are satisfied.",
        },
        "failure_abort_rules": [
            "Any existing partial validation output must be preserved and audited before resume.",
            "Any hash mismatch in frozen sources/models/policies/bank metadata aborts validation.",
            "Any final-test access without final_test_gate.json and explicit update_state authorization invalidates independence.",
            "If behaviorally fixed learned seeds remain fixed on validation, report adaptive-learning failure rather than adaptive success.",
        ],
        "timing_replay_plan": {
            "initial_validation_rollout_records_actual_timing": True,
            "formal_timing_success_requires_replay": "After fixed-H nominations, run a same-host randomized two-repeat exact-replay block for learned arms and nominated comparators before any timing-route claim.",
            "timing_replay_schedule_seed": TIMING_REPLAY_SCHEDULE_SEED,
            "do_not_infer_speed_from_H_distribution": True,
        },
    }


def docs_append(payload: Dict[str, Any]) -> None:
    text = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-26 vehicle validation gate freeze\n\n"
        f"UTC: {payload['created_utc']}. Metadata-only no-validation gate frozen at `{rel(GATE_JSON)}`. "
        f"Validation bank content opened=false; sealed test content opened=false; simulations=0. "
        f"Gate froze {payload['validation_schedule']['unique_rollout_arm_count']} unique rollout arms, "
        f"{payload['validation_schedule']['episode_count']} planned validation episodes over case indices only, and "
        f"{payload['validation_schedule']['expected_shards']} bounded shards. "
        "Vehicle learned s0 and s1 are preclassified as fixed/nonadaptive by structure; only s2 is structurally switching. "
        "External backup of this new gate is required before formal validation64 rollout; final test remains unauthorized.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n" + text, encoding="utf-8")


def write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# Vehicle validation gate 20260926",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "Scope: metadata-only freeze. No simulations, no validation-bank content reads, no sealed-test content reads, and no validation/test outcomes.",
        "",
        "## Access flags",
        "",
        f"- validation_accessed: {payload['validation_accessed']}",
        f"- validation64_bank_content_opened: {payload['validation64_bank_content_opened']}",
        f"- test_accessed: {payload['test_accessed']}",
        f"- sealed_test_bank_content_opened: {payload['sealed_test_bank_content_opened']}",
        "",
        "## Learned candidate classification",
        "",
        "| arm | seed | policy sha256 | structural class | distinct leaves | can support adaptive claim? |",
        "|---|---:|---|---|---|---:|",
    ]
    for cand in payload["learned_candidates"]:
        cls = cand["classification"]
        lines.append(
            f"| `{cand['arm_id']}` | {cand['seed']} | `{cand['policy_file_sha256']}` | "
            f"{cls['structural_class']} | {cls['distinct_leaves']} | {cls['can_be_adaptive_on_validation']} |"
        )
    lines.extend([
        "",
        "## Comparator availability",
        "",
    ])
    for key, value in payload["fixed_comparators"]["availability_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Frozen validation block",
        "",
        f"- validation split ID: {payload['splits']['validation64']['generator_spec']}",
        f"- validation bank recorded sha256 (from banks/completed metadata, content not opened now): `{payload['splits']['validation64']['recorded_sha256_from_banks_completed']}`",
        f"- logical arms: {payload['validation_schedule']['logical_arm_count']}",
        f"- unique rollout arms after exact duplicate aliases: {payload['validation_schedule']['unique_rollout_arm_count']}",
        f"- planned episodes: {payload['validation_schedule']['episode_count']}",
        f"- shard max episodes: {payload['validation_schedule']['shard_max_episodes']}",
        f"- planned shards: {payload['validation_schedule']['expected_shards']}",
        "",
        "## Interpretation gates",
        "",
        "- Vehicle validation may be run only after this gate is externally backed up and intentionally invoked.",
        "- This is not final-test authorization; sealed test remains closed.",
        "- Because learned s0/s1 are fixed by structure, a vehicle adaptive-horizon success claim is precluded unless the report explicitly treats those seeds as fixed/nonadaptive failures; no best-seed-only claim is allowed.",
        "- Timing claims require actual wall-time paired replay; H distribution is behavior audit only.",
    ])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    assert_no_prior_partial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registration = read_json(LAT / "registration.json")
    bank_completed = read_json(BANKS / "completed.json")
    assert bank_completed.get("passed") is True
    candidates = load_candidates()
    comparators = comparator_inventory()
    assert comparators["availability_summary"]["primary_fixed_H25_all_three_complete"] is True
    assert comparators["availability_summary"]["matched_terminal_grid_all_entries_have_complete_H25_terminal_source"] is True
    assert comparators["availability_summary"]["independent_terminal_seed0_full_grid_complete"] is True
    logical, rollout_arms = logical_arms()
    schedule = build_schedule(rollout_arms)
    source = source_hashes()

    hash_map = bank_completed.get("hashes") or {}
    bank_specs = extract_bank_specs(registration)
    smoke_digest_completed = ROOT / "research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/completed.json"
    smoke_pairing_completed = ROOT / "research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing/completed.json"
    smoke_inputs = {
        "development_smoke_pairing_completed": file_record(smoke_pairing_completed, role="required_no_validation_smoke_completed"),
        "smoke_artifact_digest_completed": file_record(smoke_digest_completed, role="required_smoke_digest_completed"),
    }
    for item in smoke_inputs.values():
        assert item.get("exists"), item
    assert read_json(smoke_pairing_completed).get("passed") is True
    assert read_json(smoke_digest_completed).get("passed") is True

    payload: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "IMPROVED_latency_tree_vehicle_validation_gate_metadata_only_not_original_SAC",
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "validation_rollouts_run": 0,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "test_rollouts_run": 0,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "final_test_authorization_requested": False,
        "source_hashes": source,
        "smoke_preconditions": smoke_inputs,
        "splits": {
            "smoke": {"generator_spec": bank_specs["smoke"], **vehicle_bank_record(hash_map, "vehicle_smoke_bank.json")},
            "validation64": {"generator_spec": bank_specs["validation"], **vehicle_bank_record(hash_map, "vehicle_validation_bank.json")},
            "sealed_test128": {"generator_spec": bank_specs["test"], **vehicle_bank_record(hash_map, "vehicle_test_bank.json")},
        },
        "learned_candidates": candidates,
        "fixed_comparators": comparators,
        "logical_arms": logical,
        "unique_rollout_arms": rollout_arms,
        "validation_schedule": schedule,
        "budget_freeze": {
            "primary_validation_effect_block": {
                "validation_episodes_planned": schedule["episode_count"],
                "validation_cases": VALIDATION_CASES,
                "logical_arms": schedule["logical_arm_count"],
                "unique_rollout_arms": schedule["unique_rollout_arm_count"],
                "control_step_upper_bound_vehicle_150_steps": schedule["episode_count"] * 150,
                "environment_construction_upper_bound": schedule["episode_count"],
                "reset_upper_bound": schedule["episode_count"],
                "new_training_episodes": 0,
                "new_gradient_steps": 0,
                "test_episodes": 0,
            },
            "bounded_execution": {
                "reason": "full block is likely longer than the 4h experiment cap on t3a.medium; run contiguous frozen shards only",
                "shard_max_episodes": SHARD_MAX_EPISODES,
                "planned_shards": schedule["expected_shards"],
                "one_experiment_at_a_time": True,
                "resume_requires_hash_and_partial_audit": True,
            },
            "future_timing_replay_block": {
                "status": "not scheduled until validation fixed-H nominations exist",
                "minimum_jobs": "learned s0/s1/s2 plus nominated matched-terminal comparators; independent-terminal all-seed jobs only after required seed1/2 training if H != 25",
                "repeats": 2,
                "case_count": VALIDATION_CASES,
                "schedule_seed": TIMING_REPLAY_SCHEDULE_SEED,
            },
        },
        "protocol_rules": protocol_rules(),
        "gate_integrity_passed": True,
        "authorization_after_backup": {
            "vehicle_validation64_rollout": "allowed only after external backup verifies this gate artifact and runner hashes; validation access must be explicit in run metadata",
            "final_test128_rollout": "not authorized; requires final_test_gate.json and one explicit update_state request after validation audit",
        },
        "known_limitations_before_validation": [
            "Vehicle-only validation cannot establish whole two-task reproduction because pendulum_s1 is interrupted and pendulum_s2 unstarted.",
            "The method is IMPROVED latency-tree search, not the ORIGINAL SAC method from Bøhn et al. 2021.",
            "Learned vehicle s0 and s1 are fixed/nonadaptive by structure; this is expected to prevent any all-seed adaptive success claim.",
            "All smoke decision times exceeded the nominal 0.1s vehicle deadline on t3a.medium; formal timing must use actual same-host paired wall time, not horizon counts.",
        ],
    }

    # Safety assertions for the no-validation contract and exact budget arithmetic.
    assert payload["validation_accessed"] is False
    assert payload["validation64_bank_content_opened"] is False
    assert payload["test_accessed"] is False
    assert payload["sealed_test_bank_content_opened"] is False
    assert payload["new_simulations"] == 0
    assert schedule["episode_count"] == schedule["unique_rollout_arm_count"] * VALIDATION_CASES
    assert len(schedule["episodes"]) == schedule["episode_count"]
    assert schedule["expected_shards"] == int(math.ceil(schedule["episode_count"] / SHARD_MAX_EPISODES))
    assert any(c["classification"]["can_be_adaptive_on_validation"] for c in candidates)
    assert sum(c["classification"]["can_be_adaptive_on_validation"] for c in candidates) == 1

    write_json(GATE_JSON, payload)
    write_summary(payload)
    write_json(COMPLETED_JSON, {
        "passed": True,
        "validation_accessed": False,
        "validation64_bank_content_opened": False,
        "test_accessed": False,
        "sealed_test_bank_content_opened": False,
        "new_simulations": 0,
        "new_gradient_steps": 0,
        "formal_scientific_evidence_created": False,
        "hashes": {rel(GATE_JSON): sha256(GATE_JSON), rel(SUMMARY_MD): sha256(SUMMARY_MD)},
    })
    docs_append(payload)
    print(json.dumps({
        "gate": rel(GATE_JSON),
        "summary": rel(SUMMARY_MD),
        "completed": rel(COMPLETED_JSON),
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "unique_rollout_arms": schedule["unique_rollout_arm_count"],
        "planned_validation_episodes": schedule["episode_count"],
        "planned_shards": schedule["expected_shards"],
        "adaptive_capable_learned_candidates": sum(c["classification"]["can_be_adaptive_on_validation"] for c in candidates),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
