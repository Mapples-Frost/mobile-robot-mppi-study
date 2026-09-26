#!/usr/bin/env python3
"""Digest saved vehicle latency-tree selection-race outcomes.

This is an engineering/development diagnostic.  It reads only already-created
training selection metadata/results under train/vehicle_s*/selection and does no
simulation.  It deliberately does not inspect validation or sealed test outputs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TRAIN_BASE = Path("research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train")
OUT_DIR = Path("research_artifacts/aws_diagnostics/vehicle_selection_race_outcome_digest")
OUT_JSON = OUT_DIR / "raw.json"
OUT_MD = OUT_DIR / "summary.md"
SEEDS = [0, 1, 2]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def sum_key(items: Iterable[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        val = as_float(item.get(key))
        if val is not None:
            total += val
    return total


def count_bool(items: Iterable[Dict[str, Any]], key: str, value: bool = True) -> int:
    return sum(1 for item in items if bool(item.get(key)) is value)


def weighted_mean(items: Iterable[Dict[str, Any]], value_key: str, weight_key: str = "steps") -> Optional[float]:
    num = 0.0
    den = 0.0
    for item in items:
        val = as_float(item.get(value_key))
        wt = as_float(item.get(weight_key))
        if val is not None and wt is not None and wt > 0:
            num += val * wt
            den += wt
    return (num / den) if den > 0 else None


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def infer_final_arm(final_policy: Dict[str, Any], arms: List[Dict[str, Any]]) -> Optional[str]:
    """Match train/vehicle_s*/policy.json to a registered arm policy."""
    final_kind = final_policy.get("kind") if isinstance(final_policy, dict) else None
    if final_kind == "constant" and final_policy.get("h") == 25:
        return "fixed"
    final_canon = canon(final_policy)
    for arm in arms:
        if canon(arm.get("policy")) == final_canon:
            return arm.get("id")
    return None


def arm_training_rank(reg: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    for arm in reg.get("arms", []):
        if arm.get("id") == arm_id:
            return dict(arm.get("rank") or {})
    return {}


def compact_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"type": type(policy).__name__}
    out: Dict[str, Any] = {"kind": policy.get("kind"), "task": policy.get("task")}
    if policy.get("kind") == "constant":
        out["h"] = policy.get("h")
    if policy.get("kind") == "tree":
        out["leaves"] = policy.get("leaves")
        out["nodes"] = policy.get("nodes")
    return out


def summarize_repeat(path: Path, seed: int, repeat: int, arm_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "seed": seed,
        "repeat": repeat,
        "arm": arm_id,
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        out["missing_reason"] = "selection directory absent"
        return out

    file_hashes: Dict[str, Optional[str]] = {}
    for fname in ("summary.json", "completed.json", "progress.json", "policy.json", "environment.json", "solver_attempts.json"):
        f = path / fname
        file_hashes[fname] = sha256_file(f) if f.exists() else None
    out["file_hashes"] = file_hashes

    summary_path = path / "summary.json"
    if not summary_path.exists():
        out["missing_reason"] = "summary.json absent"
        return out
    summary = load_json(summary_path)
    episodes = summary.get("episodes") or []
    if not isinstance(episodes, list):
        episodes = []
    completed = load_json(path / "completed.json") if (path / "completed.json").exists() else {}
    progress = load_json(path / "progress.json") if (path / "progress.json").exists() else {}
    environment = load_json(path / "environment.json") if (path / "environment.json").exists() else {}

    steps = int(sum_key(episodes, "steps"))
    total_cost_sum = sum_key(episodes, "total_cost")
    perf_sum = sum_key(episodes, "performance_cost")
    phys_sum = sum_key(episodes, "physical_constraint_cost")
    h_penalty_sum = sum_key(episodes, "h_penalty")
    decision_total = sum_key(episodes, "decision_total_s")
    gross_total = sum_key(episodes, "gross_total_s")
    logging_total = sum_key(episodes, "logging_total_s")
    p95_values = [as_float(ep.get("decision_p95_s")) for ep in episodes]
    p95_values = [x for x in p95_values if x is not None]

    out.update({
        "passed": completed.get("passed"),
        "training_only": completed.get("training_only"),
        "record_audit_pending": completed.get("record_audit_pending"),
        "summary_test_accessed": summary.get("test_accessed"),
        "summary_split": summary.get("split"),
        "summary_task": summary.get("task"),
        "summary_seed": summary.get("seed"),
        "environment_keys": sorted(environment.keys()) if isinstance(environment, dict) else [],
        "progress": progress,
        "episodes": len(episodes),
        "steps": steps,
        "resets": summary.get("resets"),
        "summary_steps": summary.get("steps"),
        "solver_counts": summary.get("solver_counts"),
        "success_count": count_bool(episodes, "success", True),
        "failure_count": count_bool(episodes, "success", False),
        "constraint_count": count_bool(episodes, "constraint", True),
        "termination_counts": {},
        "total_cost_sum": total_cost_sum,
        "total_cost_mean": total_cost_sum / len(episodes) if episodes else None,
        "performance_cost_sum": perf_sum,
        "performance_cost_mean": perf_sum / len(episodes) if episodes else None,
        "physical_constraint_cost_sum": phys_sum,
        "physical_constraint_cost_mean": phys_sum / len(episodes) if episodes else None,
        "h_penalty_sum": h_penalty_sum,
        "h_penalty_mean": h_penalty_sum / len(episodes) if episodes else None,
        "decision_total_s": decision_total,
        "decision_mean_s_per_step": decision_total / steps if steps else None,
        "gross_total_s": gross_total,
        "logging_total_s": logging_total,
        "decision_episode_p95_s_mean": statistics.mean(p95_values) if p95_values else None,
        "decision_episode_p95_s_max": max(p95_values) if p95_values else None,
        "mean_horizon_weighted": weighted_mean(episodes, "mean_horizon"),
        "switches": int(sum_key(episodes, "switches")),
        "solver_failure_steps": int(sum_key(episodes, "solver_failure_steps")),
        "initial_failed_steps": int(sum_key(episodes, "initial_failed_steps")),
        "retries": int(sum_key(episodes, "retries")),
        "recovered_steps": int(sum_key(episodes, "recovered_steps")),
        "deadline_exceed_steps": int(sum_key(episodes, "deadline_exceed_steps")),
        "case_ids": [ep.get("case") for ep in episodes],
        "repeat_ids_in_episodes": sorted({ep.get("repeat") for ep in episodes}),
    })
    term_counts: Dict[str, int] = {}
    for ep in episodes:
        term = str(ep.get("termination"))
        term_counts[term] = term_counts.get(term, 0) + 1
    out["termination_counts"] = term_counts
    return out


def aggregate_arm(repeats: List[Dict[str, Any]], training_rank: Dict[str, Any]) -> Dict[str, Any]:
    present = [r for r in repeats if r.get("exists") and r.get("episodes") is not None]
    episodes = int(sum(r.get("episodes") or 0 for r in present))
    steps = int(sum(r.get("steps") or 0 for r in present))
    decision_total = sum((r.get("decision_total_s") or 0.0) for r in present)
    gross_total = sum((r.get("gross_total_s") or 0.0) for r in present)
    logging_total = sum((r.get("logging_total_s") or 0.0) for r in present)
    total_cost_sum = sum((r.get("total_cost_sum") or 0.0) for r in present)
    perf_sum = sum((r.get("performance_cost_sum") or 0.0) for r in present)
    phys_sum = sum((r.get("physical_constraint_cost_sum") or 0.0) for r in present)
    hpen_sum = sum((r.get("h_penalty_sum") or 0.0) for r in present)
    mean_h_num = sum((r.get("mean_horizon_weighted") or 0.0) * (r.get("steps") or 0) for r in present)
    p95_mean_vals = [r.get("decision_episode_p95_s_mean") for r in present if r.get("decision_episode_p95_s_mean") is not None]
    p95_max_vals = [r.get("decision_episode_p95_s_max") for r in present if r.get("decision_episode_p95_s_max") is not None]
    termination_counts: Dict[str, int] = {}
    for r in present:
        for k, v in (r.get("termination_counts") or {}).items():
            termination_counts[k] = termination_counts.get(k, 0) + int(v)
    agg: Dict[str, Any] = {
        "repeat_count": len(present),
        "complete_repeat_count": sum(1 for r in present if r.get("passed") is True),
        "missing_repeat_count": len(repeats) - len(present),
        "all_passed": bool(repeats and all(r.get("passed") is True for r in present) and len(present) == len(repeats)),
        "summary_test_accessed_any": any(r.get("summary_test_accessed") for r in present),
        "training_only_all": bool(present and all(r.get("training_only") is True for r in present)),
        "episodes": episodes,
        "steps": steps,
        "success_count": int(sum(r.get("success_count") or 0 for r in present)),
        "failure_count": int(sum(r.get("failure_count") or 0 for r in present)),
        "constraint_count": int(sum(r.get("constraint_count") or 0 for r in present)),
        "termination_counts": termination_counts,
        "solver_failure_steps": int(sum(r.get("solver_failure_steps") or 0 for r in present)),
        "initial_failed_steps": int(sum(r.get("initial_failed_steps") or 0 for r in present)),
        "retries": int(sum(r.get("retries") or 0 for r in present)),
        "recovered_steps": int(sum(r.get("recovered_steps") or 0 for r in present)),
        "deadline_exceed_steps": int(sum(r.get("deadline_exceed_steps") or 0 for r in present)),
        "switches": int(sum(r.get("switches") or 0 for r in present)),
        "total_cost_sum": total_cost_sum,
        "total_cost_mean_episode": total_cost_sum / episodes if episodes else None,
        "performance_cost_sum": perf_sum,
        "performance_cost_mean_episode": perf_sum / episodes if episodes else None,
        "physical_constraint_cost_sum": phys_sum,
        "physical_constraint_cost_mean_episode": phys_sum / episodes if episodes else None,
        "h_penalty_sum": hpen_sum,
        "h_penalty_mean_episode": hpen_sum / episodes if episodes else None,
        "decision_total_s": decision_total,
        "decision_mean_s_per_step": decision_total / steps if steps else None,
        "decision_episode_p95_s_mean_across_repeats": statistics.mean(p95_mean_vals) if p95_mean_vals else None,
        "decision_episode_p95_s_max": max(p95_max_vals) if p95_max_vals else None,
        "gross_total_s": gross_total,
        "logging_total_s": logging_total,
        "mean_horizon_weighted": mean_h_num / steps if steps else None,
        "training_rank": training_rank,
    }
    return agg


def add_fixed_comparisons(arms: Dict[str, Dict[str, Any]]) -> None:
    fixed = arms.get("fixed")
    if not fixed:
        return
    f_total = fixed.get("total_cost_sum")
    f_perf = fixed.get("performance_cost_sum")
    f_phys = fixed.get("physical_constraint_cost_sum")
    f_time = fixed.get("decision_total_s")
    for arm_id, arm in arms.items():
        total_ratio = safe_ratio(arm.get("total_cost_sum"), f_total)
        perf_ratio = safe_ratio(arm.get("performance_cost_sum"), f_perf)
        phys_ratio = safe_ratio(arm.get("physical_constraint_cost_sum"), f_phys)
        time_ratio = safe_ratio(arm.get("decision_total_s"), f_time)
        arm["relative_to_fixed"] = {
            "total_cost_ratio": total_ratio,
            "performance_cost_ratio": perf_ratio,
            "physical_constraint_cost_ratio": phys_ratio,
            "decision_total_time_ratio": time_ratio,
            "total_cost_change": (total_ratio - 1.0) if total_ratio is not None else None,
            "performance_cost_change": (perf_ratio - 1.0) if perf_ratio is not None else None,
            "physical_constraint_cost_change": (phys_ratio - 1.0) if phys_ratio is not None else None,
            "decision_time_change": (time_ratio - 1.0) if time_ratio is not None else None,
            # Training-rank objective pattern appears to be cost_change + 0.5*(time_ratio-1).
            "objective_total_plus_half_time": ((total_ratio - 1.0) + 0.5 * (time_ratio - 1.0)) if total_ratio is not None and time_ratio is not None else None,
            "objective_performance_plus_half_time": ((perf_ratio - 1.0) + 0.5 * (time_ratio - 1.0)) if perf_ratio is not None and time_ratio is not None else None,
            "objective_physical_plus_half_time": ((phys_ratio - 1.0) + 0.5 * (time_ratio - 1.0)) if phys_ratio is not None and time_ratio is not None else None,
        }


def eligible_for_ranking(arm: Dict[str, Any]) -> bool:
    return bool(
        arm.get("repeat_count", 0) > 0
        and arm.get("episodes", 0) > 0
        and arm.get("failure_count", 1) == 0
        and arm.get("constraint_count", 1) == 0
        and arm.get("solver_failure_steps", 1) == 0
    )


def best_by(arms: Dict[str, Dict[str, Any]], key_path: Tuple[str, ...]) -> Optional[str]:
    best_id: Optional[str] = None
    best_val: Optional[float] = None
    for arm_id, arm in arms.items():
        cur: Any = arm
        for k in key_path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        val = as_float(cur)
        if val is None or not eligible_for_ranking(arm):
            continue
        if best_val is None or val < best_val:
            best_id = arm_id
            best_val = val
    return best_id


def sorted_arm_ids(arms: Dict[str, Dict[str, Any]], key_path: Tuple[str, ...]) -> List[str]:
    vals: List[Tuple[float, str]] = []
    for arm_id, arm in arms.items():
        cur: Any = arm
        for k in key_path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        val = as_float(cur)
        if val is not None:
            vals.append((val, arm_id))
    return [arm_id for _, arm_id in sorted(vals)]


def digest_seed(seed: int) -> Dict[str, Any]:
    seed_dir = TRAIN_BASE / f"vehicle_s{seed}"
    reg_path = seed_dir / "selection_registration.json"
    final_policy_path = seed_dir / "policy.json"
    reg = load_json(reg_path)
    final_policy = load_json(final_policy_path)
    arms_reg = reg.get("arms") or []
    arm_ids = [a.get("id") for a in arms_reg if a.get("id")]
    final_arm = infer_final_arm(final_policy, arms_reg)

    order = reg.get("order") or []
    expected_pairs = [(int(item.get("repeat")), str(item.get("arm"))) for item in order if "repeat" in item and "arm" in item]
    pairs_by_arm: Dict[str, List[int]] = {arm_id: [] for arm_id in arm_ids}
    for repeat, arm_id in expected_pairs:
        pairs_by_arm.setdefault(arm_id, []).append(repeat)
    for arm_id in arm_ids:
        if not pairs_by_arm.get(arm_id):
            # Conservative fallback to the historical two-repeat layout.
            pairs_by_arm[arm_id] = [0, 1]

    repeat_details: Dict[str, List[Dict[str, Any]]] = {}
    arm_aggregates: Dict[str, Dict[str, Any]] = {}
    for arm_id in arm_ids:
        repeats = []
        for repeat in sorted(set(pairs_by_arm.get(arm_id, [0, 1]))):
            repeats.append(summarize_repeat(seed_dir / "selection" / f"r{repeat}_{arm_id}", seed, repeat, arm_id))
        repeat_details[arm_id] = repeats
        arm_aggregates[arm_id] = aggregate_arm(repeats, arm_training_rank(reg, arm_id))
        arm_aggregates[arm_id]["policy"] = compact_policy(next((a.get("policy") for a in arms_reg if a.get("id") == arm_id), {}))
    add_fixed_comparisons(arm_aggregates)

    ranking_keys = {
        "total_cost_sum": ("total_cost_sum",),
        "performance_cost_sum": ("performance_cost_sum",),
        "physical_constraint_cost_sum": ("physical_constraint_cost_sum",),
        "h_penalty_sum": ("h_penalty_sum",),
        "decision_total_s": ("decision_total_s",),
        "objective_total_plus_half_time": ("relative_to_fixed", "objective_total_plus_half_time"),
        "objective_performance_plus_half_time": ("relative_to_fixed", "objective_performance_plus_half_time"),
        "objective_physical_plus_half_time": ("relative_to_fixed", "objective_physical_plus_half_time"),
    }
    best = {name: best_by(arm_aggregates, path) for name, path in ranking_keys.items()}
    sorted_by_objective = sorted_arm_ids(arm_aggregates, ("relative_to_fixed", "objective_total_plus_half_time"))

    # Capture top training-objective arm from registration for comparison.
    train_ranked: List[Tuple[float, str]] = []
    for arm in arms_reg:
        if arm.get("id") == "fixed":
            continue
        obj = as_float((arm.get("rank") or {}).get("objective"))
        if obj is not None:
            train_ranked.append((obj, arm.get("id")))
    train_ranked.sort()
    chosen_metrics = arm_aggregates.get(final_arm or "", {})

    explanation_flags = {
        "final_policy_matched_registered_arm": final_arm is not None,
        "final_matches_best_total_cost": final_arm == best.get("total_cost_sum"),
        "final_matches_best_performance_cost": final_arm == best.get("performance_cost_sum"),
        "final_matches_best_physical_cost": final_arm == best.get("physical_constraint_cost_sum"),
        "final_matches_fastest_arm": final_arm == best.get("decision_total_s"),
        "final_matches_best_total_plus_half_time_objective": final_arm == best.get("objective_total_plus_half_time"),
        "final_matches_best_performance_plus_half_time_objective": final_arm == best.get("objective_performance_plus_half_time"),
        "final_matches_best_physical_plus_half_time_objective": final_arm == best.get("objective_physical_plus_half_time"),
        "final_matches_best_training_objective_candidate": bool(train_ranked and final_arm == train_ranked[0][1]),
        "all_selection_repeats_passed": all(a.get("all_passed") for a in arm_aggregates.values()),
        "any_summary_test_accessed_flag": any(a.get("summary_test_accessed_any") for a in arm_aggregates.values()),
    }

    return {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "file_hashes": {
            "selection_registration.json": sha256_file(reg_path),
            "policy.json": sha256_file(final_policy_path),
        },
        "selection_registration_order": order,
        "registered_arm_ids": arm_ids,
        "registered_finalist_ids": [f.get("id") for f in reg.get("finalists", []) if isinstance(f, dict)],
        "final_policy": compact_policy(final_policy),
        "final_arm_inferred_from_policy_json": final_arm,
        "best_candidate_by_training_objective": train_ranked[0][1] if train_ranked else None,
        "training_objective_order": [{"arm": arm_id, "objective": obj} for obj, arm_id in train_ranked],
        "arm_aggregates": arm_aggregates,
        "repeat_details": repeat_details,
        "best_by_metric": best,
        "arms_sorted_by_total_plus_half_time_objective": sorted_by_objective,
        "explanation_flags": explanation_flags,
        "final_arm_metrics": chosen_metrics,
    }


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{float(x):.{digits}f}"
    return str(x)


def write_summary(digest: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.extend([
        "# Vehicle selection-race outcome digest",
        "",
        f"Created UTC: {digest['created_utc']}",
        "",
        "Scope: saved vehicle training selection-race metadata only. No simulations, no validation reads, and no sealed test reads were performed.",
        "",
        "## Cross-seed findings",
        "",
    ])
    for k, v in digest["cross_seed_findings"].items():
        lines.append(f"- {k}: {v}")
    for seed_block in digest["seeds"]:
        lines.extend([
            "",
            f"## Seed {seed_block['seed']}",
            "",
            f"- final policy arm inferred from policy.json: `{seed_block['final_arm_inferred_from_policy_json']}`",
            f"- best candidate by saved fit/training objective: `{seed_block['best_candidate_by_training_objective']}`",
            f"- best-by-selection metrics: {seed_block['best_by_metric']}",
            f"- explanation flags: {seed_block['explanation_flags']}",
            "",
            "| arm | final | train_obj | episodes | success | constr | solver_fail | init_fail | deadlines | total_cost | perf_cost | h_pen | decision_s | time_ratio | obj_total+0.5time | obj_phys+0.5time | switches | meanH |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        final_arm = seed_block["final_arm_inferred_from_policy_json"]
        arms = seed_block["arm_aggregates"]
        for arm_id in sorted(arms.keys(), key=lambda a: (arms[a].get("relative_to_fixed", {}).get("objective_total_plus_half_time") is None, arms[a].get("relative_to_fixed", {}).get("objective_total_plus_half_time") or 1e99, a)):
            arm = arms[arm_id]
            rel = arm.get("relative_to_fixed") or {}
            train_obj = (arm.get("training_rank") or {}).get("objective")
            lines.append(
                f"| `{arm_id}` | {arm_id == final_arm} | {fmt(train_obj, 6)} | {arm.get('episodes')} | "
                f"{arm.get('success_count')}/{arm.get('episodes')} | {arm.get('constraint_count')} | {arm.get('solver_failure_steps')} | "
                f"{arm.get('initial_failed_steps')} | {arm.get('deadline_exceed_steps')} | {fmt(arm.get('total_cost_sum'), 3)} | "
                f"{fmt(arm.get('performance_cost_sum'), 3)} | {fmt(arm.get('h_penalty_sum'), 3)} | {fmt(arm.get('decision_total_s'), 3)} | "
                f"{fmt(rel.get('decision_total_time_ratio'), 4)} | {fmt(rel.get('objective_total_plus_half_time'), 6)} | "
                f"{fmt(rel.get('objective_physical_plus_half_time'), 6)} | {arm.get('switches')} | {fmt(arm.get('mean_horizon_weighted'), 3)} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest: Dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "method": "IMPROVED latency-tree saved-selection-race metadata digest; no simulation",
        "input_train_base": str(TRAIN_BASE),
        "seeds_requested": SEEDS,
        "validation_accessed": False,
        "test_accessed": False,
        "new_simulations": 0,
        "seeds": [],
    }
    for seed in SEEDS:
        block = digest_seed(seed)
        digest["seeds"].append(block)
        print(json.dumps({
            "seed": seed,
            "final_arm": block["final_arm_inferred_from_policy_json"],
            "best_training_objective": block["best_candidate_by_training_objective"],
            "best_by_metric": block["best_by_metric"],
            "explanation_flags": block["explanation_flags"],
        }, sort_keys=True))

    n = len(digest["seeds"])
    final_match_training = sum(1 for s in digest["seeds"] if s["explanation_flags"].get("final_matches_best_training_objective_candidate"))
    final_match_obj_total = sum(1 for s in digest["seeds"] if s["explanation_flags"].get("final_matches_best_total_plus_half_time_objective"))
    final_match_obj_phys = sum(1 for s in digest["seeds"] if s["explanation_flags"].get("final_matches_best_physical_plus_half_time_objective"))
    all_repeats_passed = all(s["explanation_flags"].get("all_selection_repeats_passed") for s in digest["seeds"])
    any_test_flag = any(s["explanation_flags"].get("any_summary_test_accessed_flag") for s in digest["seeds"])
    fixed_final = sum(1 for s in digest["seeds"] if s["final_arm_inferred_from_policy_json"] == "fixed")
    constant_tree_final = 0
    switching_tree_final = 0
    for s in digest["seeds"]:
        arm_id = s["final_arm_inferred_from_policy_json"]
        pol = (s["arm_aggregates"].get(arm_id, {}) if arm_id else {}).get("policy") or {}
        if pol.get("kind") == "tree":
            leaves = pol.get("leaves") or []
            if len(set(leaves)) <= 1:
                constant_tree_final += 1
            else:
                switching_tree_final += 1
    digest["cross_seed_findings"] = {
        "seeds": n,
        "final_matches_best_training_objective_count": final_match_training,
        "final_matches_best_total_plus_half_time_selection_objective_count": final_match_obj_total,
        "final_matches_best_physical_plus_half_time_selection_objective_count": final_match_obj_phys,
        "fixed_final_policy_count": fixed_final,
        "constant_tree_final_policy_count": constant_tree_final,
        "switching_tree_final_policy_count": switching_tree_final,
        "all_selection_repeats_passed": all_repeats_passed,
        "any_summary_test_accessed_flag": any_test_flag,
        "interpretation": (
            "Selection-race summaries are training/development evidence only. The digest compares final policy.json "
            "against saved selection-race aggregate metrics and training fit-objective ranks; it does not establish "
            "validation/test performance."
        ),
    }
    OUT_JSON.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(digest)
    print(json.dumps({"wrote": str(OUT_JSON), "summary": str(OUT_MD), "cross_seed_findings": digest["cross_seed_findings"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
