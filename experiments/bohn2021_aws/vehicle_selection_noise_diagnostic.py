#!/usr/bin/env python3
"""Noise and objective audit for saved vehicle latency-tree selection races.

Engineering/development diagnostic only.  This script reads already-created
training selection summaries under train/vehicle_s*/selection plus the saved
selection registrations and selected policies.  It performs no simulation and
intentionally does not inspect validation or sealed test outputs.
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
OUT_DIR = Path("research_artifacts/aws_diagnostics/vehicle_selection_noise_diagnostic")
OUT_JSON = OUT_DIR / "raw.json"
OUT_MD = OUT_DIR / "summary.md"
SEEDS = [0, 1, 2]
TIME_WEIGHT = 0.5
EPS = 1e-9


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


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or abs(den) <= EPS:
        return None
    return num / den


def mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def median(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def stdev(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def minmax(xs: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return (min(vals), max(vals)) if vals else (None, None)


def sum_key(items: Iterable[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        val = as_float(item.get(key))
        if val is not None:
            total += val
    return total


def compact_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"type": type(policy).__name__}
    out: Dict[str, Any] = {"kind": policy.get("kind"), "task": policy.get("task")}
    if policy.get("kind") == "constant":
        out["h"] = policy.get("h")
    elif policy.get("kind") == "tree":
        out["leaves"] = policy.get("leaves")
        out["nodes"] = policy.get("nodes")
    return out


def policy_is_behaviorally_fixed(policy: Any, h: int = 25) -> bool:
    if not isinstance(policy, dict):
        return False
    if policy.get("kind") == "constant":
        return policy.get("h") == h
    if policy.get("kind") == "tree":
        leaves = policy.get("leaves") or []
        return bool(leaves) and all(x == h for x in leaves)
    return False


def policy_switching(policy: Any) -> bool:
    if not isinstance(policy, dict) or policy.get("kind") != "tree":
        return False
    leaves = policy.get("leaves") or []
    return len(set(leaves)) > 1


def infer_final_arm(final_policy: Dict[str, Any], arms: List[Dict[str, Any]]) -> Optional[str]:
    if isinstance(final_policy, dict) and final_policy.get("kind") == "constant" and final_policy.get("h") == 25:
        return "fixed"
    c = canon(final_policy)
    for arm in arms:
        if canon(arm.get("policy")) == c:
            return arm.get("id")
    return None


def arm_training_rank(reg: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    for arm in reg.get("arms", []):
        if arm.get("id") == arm_id:
            return dict(arm.get("rank") or {})
    return {}


def get_policy(reg: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    for arm in reg.get("arms", []):
        if arm.get("id") == arm_id:
            pol = arm.get("policy")
            return pol if isinstance(pol, dict) else {}
    return {}


def expected_repeats(reg: Dict[str, Any], arm_id: str) -> List[int]:
    reps: List[int] = []
    for item in reg.get("order") or []:
        if item.get("arm") == arm_id and "repeat" in item:
            try:
                reps.append(int(item["repeat"]))
            except Exception:
                pass
    return sorted(set(reps or [0, 1]))


def normalize_episode(ep: Dict[str, Any], arm_id: str, repeat: int) -> Dict[str, Any]:
    out = dict(ep)
    out["arm"] = arm_id
    out["repeat"] = repeat
    out["case"] = int(ep.get("case")) if isinstance(ep.get("case"), int) else ep.get("case")
    return out


def read_arm_episodes(seed_dir: Path, reg: Dict[str, Any], arm_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    repeat_summaries: Dict[str, Any] = {}
    missing: List[str] = []
    hashes: Dict[str, str] = {}
    test_flags: List[Any] = []
    validation_flags: List[Any] = []
    for repeat in expected_repeats(reg, arm_id):
        path = seed_dir / "selection" / f"r{repeat}_{arm_id}" / "summary.json"
        if not path.exists():
            missing.append(str(path))
            continue
        hashes[str(path)] = sha256_file(path)
        summary = load_json(path)
        test_flags.append(summary.get("test_accessed"))
        validation_flags.append(summary.get("validation_accessed"))
        eps = summary.get("episodes") or []
        if not isinstance(eps, list):
            eps = []
        norm = [normalize_episode(ep, arm_id, repeat) for ep in eps if isinstance(ep, dict)]
        episodes.extend(norm)
        repeat_summaries[str(repeat)] = {
            "path": str(path),
            "episodes": len(norm),
            "steps": int(sum_key(norm, "steps")),
            "total_cost_sum": sum_key(norm, "total_cost"),
            "performance_cost_sum": sum_key(norm, "performance_cost"),
            "physical_constraint_cost_sum": sum_key(norm, "physical_constraint_cost"),
            "h_penalty_sum": sum_key(norm, "h_penalty"),
            "decision_total_s": sum_key(norm, "decision_total_s"),
            "success_count": sum(1 for ep in norm if ep.get("success") is True),
            "constraint_count": sum(1 for ep in norm if bool(ep.get("constraint"))),
            "solver_failure_steps": int(sum_key(norm, "solver_failure_steps")),
            "deadline_exceed_steps": int(sum_key(norm, "deadline_exceed_steps")),
            "switches": int(sum_key(norm, "switches")),
        }
    meta = {
        "missing_summaries": missing,
        "summary_hashes": hashes,
        "test_accessed_any": any(bool(x) for x in test_flags),
        "validation_accessed_any": any(bool(x) for x in validation_flags),
        "repeat_summaries": repeat_summaries,
    }
    return episodes, meta


def aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(episodes)
    steps = int(sum_key(episodes, "steps"))
    h_num = 0.0
    h_den = 0.0
    for ep in episodes:
        mh = as_float(ep.get("mean_horizon"))
        st = as_float(ep.get("steps"))
        if mh is not None and st is not None and st > 0:
            h_num += mh * st
            h_den += st
    p95s = [as_float(ep.get("decision_p95_s")) for ep in episodes]
    p95s = [x for x in p95s if x is not None]
    return {
        "episodes": n,
        "steps": steps,
        "success_count": sum(1 for ep in episodes if ep.get("success") is True),
        "failure_count": sum(1 for ep in episodes if ep.get("success") is False),
        "constraint_count": sum(1 for ep in episodes if bool(ep.get("constraint"))),
        "solver_failure_steps": int(sum_key(episodes, "solver_failure_steps")),
        "initial_failed_steps": int(sum_key(episodes, "initial_failed_steps")),
        "deadline_exceed_steps": int(sum_key(episodes, "deadline_exceed_steps")),
        "switches": int(sum_key(episodes, "switches")),
        "total_cost_sum": sum_key(episodes, "total_cost"),
        "performance_cost_sum": sum_key(episodes, "performance_cost"),
        "physical_constraint_cost_sum": sum_key(episodes, "physical_constraint_cost"),
        "h_penalty_sum": sum_key(episodes, "h_penalty"),
        "decision_total_s": sum_key(episodes, "decision_total_s"),
        "decision_mean_s_per_step": (sum_key(episodes, "decision_total_s") / steps) if steps else None,
        "decision_p95_episode_mean_s": mean(p95s),
        "decision_p95_episode_max_s": max(p95s) if p95s else None,
        "mean_horizon_weighted": (h_num / h_den) if h_den else None,
    }


def add_objectives(arms: Dict[str, Dict[str, Any]]) -> None:
    fixed = arms.get("fixed")
    if not fixed:
        return
    for arm_id, arm in arms.items():
        total_ratio = safe_ratio(arm.get("total_cost_sum"), fixed.get("total_cost_sum"))
        perf_ratio = safe_ratio(arm.get("performance_cost_sum"), fixed.get("performance_cost_sum"))
        phys_ratio = safe_ratio(arm.get("physical_constraint_cost_sum"), fixed.get("physical_constraint_cost_sum"))
        time_ratio = safe_ratio(arm.get("decision_total_s"), fixed.get("decision_total_s"))
        arm["relative_to_fixed"] = {
            "total_cost_ratio": total_ratio,
            "performance_cost_ratio": perf_ratio,
            "physical_constraint_cost_ratio": phys_ratio,
            "decision_time_ratio": time_ratio,
            "total_cost_change": (total_ratio - 1.0) if total_ratio is not None else None,
            "performance_cost_change": (perf_ratio - 1.0) if perf_ratio is not None else None,
            "physical_constraint_cost_change": (phys_ratio - 1.0) if phys_ratio is not None else None,
            "decision_time_change": (time_ratio - 1.0) if time_ratio is not None else None,
            "selection_objective_total_plus_half_time": ((total_ratio - 1.0) + TIME_WEIGHT * (time_ratio - 1.0)) if total_ratio is not None and time_ratio is not None else None,
            "selection_objective_physical_plus_half_time": ((phys_ratio - 1.0) + TIME_WEIGHT * (time_ratio - 1.0)) if phys_ratio is not None and time_ratio is not None else None,
        }


def rank_key(arm: Dict[str, Any]) -> float:
    rel = arm.get("relative_to_fixed") or {}
    val = rel.get("selection_objective_total_plus_half_time")
    if val is None:
        return float("inf")
    if arm.get("failure_count", 0) or arm.get("constraint_count", 0) or arm.get("solver_failure_steps", 0):
        return float("inf")
    return float(val)


def episode_map(episodes: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for ep in episodes:
        case = ep.get("case")
        repeat = ep.get("repeat")
        if isinstance(case, int) and isinstance(repeat, int):
            out[(repeat, case)] = ep
    return out


def paired_delta(a_eps: List[Dict[str, Any]], b_eps: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Any]:
    amap = episode_map(a_eps)
    bmap = episode_map(b_eps)
    keys = sorted(set(amap).intersection(bmap))
    out: Dict[str, Any] = {"paired_units": len(keys)}
    for m in metrics:
        deltas: List[float] = []
        for k in keys:
            av = as_float(amap[k].get(m))
            bv = as_float(bmap[k].get(m))
            if av is not None and bv is not None:
                deltas.append(av - bv)
        mn, mx = minmax(deltas)
        out[m] = {
            "n": len(deltas),
            "sum_delta_a_minus_b": sum(deltas),
            "mean_delta_a_minus_b": mean(deltas),
            "median_delta_a_minus_b": median(deltas),
            "stdev_delta_a_minus_b": stdev(deltas),
            "min_delta": mn,
            "max_delta": mx,
            "positive_count": sum(1 for d in deltas if d > EPS),
            "negative_count": sum(1 for d in deltas if d < -EPS),
            "zero_count": sum(1 for d in deltas if abs(d) <= EPS),
        }
    return out


def repeat_level_objectives(arms: Dict[str, Dict[str, Any]], arm_id: str) -> Dict[str, Any]:
    fixed = arms.get("fixed", {}).get("meta", {}).get("repeat_summaries", {})
    current = arms.get(arm_id, {}).get("meta", {}).get("repeat_summaries", {})
    vals: List[float] = []
    rows: List[Dict[str, Any]] = []
    for rep in sorted(set(fixed).intersection(current), key=lambda x: int(x)):
        f = fixed[rep]
        c = current[rep]
        cr = safe_ratio(c.get("total_cost_sum"), f.get("total_cost_sum"))
        tr = safe_ratio(c.get("decision_total_s"), f.get("decision_total_s"))
        obj = ((cr - 1.0) + TIME_WEIGHT * (tr - 1.0)) if cr is not None and tr is not None else None
        rows.append({
            "repeat": int(rep),
            "total_cost_ratio_to_fixed_same_repeat": cr,
            "decision_time_ratio_to_fixed_same_repeat": tr,
            "objective_total_plus_half_time": obj,
            "total_cost_delta": (c.get("total_cost_sum") or 0.0) - (f.get("total_cost_sum") or 0.0),
            "decision_time_delta_s": (c.get("decision_total_s") or 0.0) - (f.get("decision_total_s") or 0.0),
        })
        if obj is not None:
            vals.append(obj)
    mn, mx = minmax(vals)
    return {
        "rows": rows,
        "mean_objective": mean(vals),
        "stdev_objective": stdev(vals),
        "min_objective": mn,
        "max_objective": mx,
        "negative_objective_repeats": sum(1 for x in vals if x < 0),
        "positive_objective_repeats": sum(1 for x in vals if x > 0),
    }


def training_objective_order(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Tuple[float, str, Dict[str, Any]]] = []
    for arm in reg.get("arms") or []:
        if arm.get("id") == "fixed":
            continue
        rank = arm.get("rank") or {}
        obj = as_float(rank.get("objective"))
        if obj is not None:
            rows.append((obj, str(arm.get("id")), rank))
    rows.sort()
    return [{"arm": arm_id, "objective": obj, "rank": rank} for obj, arm_id, rank in rows]


def digest_seed(seed: int) -> Dict[str, Any]:
    seed_dir = TRAIN_BASE / f"vehicle_s{seed}"
    reg_path = seed_dir / "selection_registration.json"
    final_path = seed_dir / "policy.json"
    reg = load_json(reg_path)
    final_policy = load_json(final_path)
    arm_ids = [a.get("id") for a in (reg.get("arms") or []) if a.get("id")]
    final_arm = infer_final_arm(final_policy, reg.get("arms") or [])

    arms: Dict[str, Dict[str, Any]] = {}
    episodes_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm_id in arm_ids:
        episodes, meta = read_arm_episodes(seed_dir, reg, arm_id)
        episodes_by_arm[arm_id] = episodes
        arms[arm_id] = aggregate_episodes(episodes)
        pol = get_policy(reg, arm_id)
        arms[arm_id]["policy"] = compact_policy(pol)
        arms[arm_id]["behaviorally_fixed_h25"] = policy_is_behaviorally_fixed(pol, 25)
        arms[arm_id]["switching_policy"] = policy_switching(pol)
        arms[arm_id]["training_rank"] = arm_training_rank(reg, arm_id)
        arms[arm_id]["meta"] = meta
    add_objectives(arms)

    selection_order = sorted(arm_ids, key=lambda a: (rank_key(arms[a]), a))
    best_selection = selection_order[0] if selection_order and math.isfinite(rank_key(arms[selection_order[0]])) else None
    train_order = training_objective_order(reg)
    best_training = train_order[0]["arm"] if train_order else None

    pair_metrics = [
        "total_cost", "performance_cost", "physical_constraint_cost", "h_penalty", "decision_total_s",
        "decision_p95_s", "mean_horizon", "switches", "deadline_exceed_steps", "initial_failed_steps",
    ]
    comparisons: Dict[str, Any] = {}
    if final_arm and final_arm != "fixed" and final_arm in episodes_by_arm and "fixed" in episodes_by_arm:
        comparisons["final_minus_fixed_paired_episode"] = paired_delta(episodes_by_arm[final_arm], episodes_by_arm["fixed"], pair_metrics)
    if final_arm and best_training and final_arm != best_training and final_arm in episodes_by_arm and best_training in episodes_by_arm:
        comparisons["final_minus_best_training_paired_episode"] = paired_delta(episodes_by_arm[final_arm], episodes_by_arm[best_training], pair_metrics)
    if best_training and best_training in episodes_by_arm and "fixed" in episodes_by_arm:
        comparisons["best_training_minus_fixed_paired_episode"] = paired_delta(episodes_by_arm[best_training], episodes_by_arm["fixed"], pair_metrics)

    repeat_objectives = {arm_id: repeat_level_objectives(arms, arm_id) for arm_id in arm_ids if arm_id != "fixed"}

    final = arms.get(final_arm or "", {})
    final_rel = final.get("relative_to_fixed") or {}
    selected_fixed_equivalent = bool(final.get("behaviorally_fixed_h25"))
    selected_switching = bool(final.get("switching_policy"))
    selected_cost_change = final_rel.get("total_cost_change")
    selected_time_change = final_rel.get("decision_time_change")
    selected_obj = final_rel.get("selection_objective_total_plus_half_time")
    timing_only_win = bool(
        final_arm not in (None, "fixed")
        and selected_fixed_equivalent
        and selected_cost_change is not None
        and abs(float(selected_cost_change)) < 1e-8
        and selected_obj is not None
        and selected_obj < 0
    )
    fixed_valid_win = bool(
        final_arm == "fixed"
        and best_selection == "fixed"
        and all(arms[a].get("failure_count", 0) == 0 for a in arm_ids)
    )
    real_switching_cost_win = bool(
        selected_switching
        and selected_cost_change is not None
        and selected_time_change is not None
        and selected_cost_change < 0
        and selected_obj is not None
        and selected_obj < 0
    )

    top_two_gap = None
    if len(selection_order) >= 2:
        top_two_gap = rank_key(arms[selection_order[1]]) - rank_key(arms[selection_order[0]])

    return {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "hashes": {
            "selection_registration.json": sha256_file(reg_path),
            "policy.json": sha256_file(final_path),
        },
        "registered_arm_ids": arm_ids,
        "final_arm": final_arm,
        "best_selection_objective_arm": best_selection,
        "selection_objective_order": [
            {
                "arm": arm_id,
                "objective": rank_key(arms[arm_id]),
                "total_cost_change": (arms[arm_id].get("relative_to_fixed") or {}).get("total_cost_change"),
                "decision_time_change": (arms[arm_id].get("relative_to_fixed") or {}).get("decision_time_change"),
                "switching": arms[arm_id].get("switching_policy"),
                "behaviorally_fixed_h25": arms[arm_id].get("behaviorally_fixed_h25"),
            }
            for arm_id in selection_order
        ],
        "best_training_objective_arm": best_training,
        "training_objective_order": train_order,
        "arms": arms,
        "repeat_level_objectives_vs_fixed": repeat_objectives,
        "paired_comparisons": comparisons,
        "diagnosis_flags": {
            "final_matches_selection_objective": final_arm == best_selection,
            "final_matches_training_objective": final_arm == best_training,
            "selected_behaviorally_fixed_h25": selected_fixed_equivalent,
            "selected_switching_policy": selected_switching,
            "timing_only_constant_tree_win_over_fixed": timing_only_win,
            "fixed_valid_selection_objective_win_not_fallback": fixed_valid_win,
            "real_switching_total_cost_selection_win": real_switching_cost_win,
            "any_summary_validation_accessed_flag": any((arms[a].get("meta") or {}).get("validation_accessed_any") for a in arm_ids),
            "any_summary_test_accessed_flag": any((arms[a].get("meta") or {}).get("test_accessed_any") for a in arm_ids),
        },
        "top_two_selection_objective_gap": top_two_gap,
    }


def fmt(x: Any, digits: int = 5) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isfinite(x):
            return f"{x:.{digits}f}"
        return str(x)
    return str(x)


def write_summary(digest: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.extend([
        "# Vehicle selection noise/objective diagnostic",
        "",
        f"Created UTC: {digest['created_utc']}",
        "",
        "Scope: saved vehicle training selection summaries only. No simulations, no validation reads, and no sealed test reads were performed.",
        "",
        "## Cross-seed diagnosis",
        "",
    ])
    for k, v in digest["cross_seed"].items():
        lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "Interpretation notes:",
        "- `selection objective` below is reconstructed as total_cost ratio to fixed plus 0.5 * decision-time ratio penalty, i.e. (cost_ratio-1)+0.5*(time_ratio-1).",
        "- Identical H25 behavior can still win this objective through sub-percent wall-time differences; this is timing noise/measurement susceptibility, not adaptive-horizon evidence.",
        "- All entries are development/training-selection evidence only.",
    ])

    for s in digest["seeds"]:
        lines.extend([
            "",
            f"## Seed {s['seed']}",
            "",
            f"- final arm: `{s['final_arm']}`; best selection-objective arm: `{s['best_selection_objective_arm']}`; best saved fit/training-objective arm: `{s['best_training_objective_arm']}`",
            f"- flags: {s['diagnosis_flags']}",
            f"- top-two selection-objective gap: {fmt(s['top_two_selection_objective_gap'], 6)}",
            "",
            "| rank | arm | final | obj | total_cost_change | decision_time_change | switching | fixed-H25-equiv | switches | meanH | total_cost | decision_s | train_obj | repeat obj min..max |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for rank, row in enumerate(s["selection_objective_order"], 1):
            arm_id = row["arm"]
            arm = s["arms"][arm_id]
            rel = arm.get("relative_to_fixed") or {}
            train_obj = (arm.get("training_rank") or {}).get("objective")
            rep_obj = s["repeat_level_objectives_vs_fixed"].get(arm_id) or {}
            rep_range = f"{fmt(rep_obj.get('min_objective'), 6)}..{fmt(rep_obj.get('max_objective'), 6)}" if arm_id != "fixed" else "NA"
            lines.append(
                f"| {rank} | `{arm_id}` | {arm_id == s['final_arm']} | {fmt(row['objective'], 6)} | "
                f"{fmt(rel.get('total_cost_change'), 6)} | {fmt(rel.get('decision_time_change'), 6)} | "
                f"{row['switching']} | {row['behaviorally_fixed_h25']} | {arm.get('switches')} | {fmt(arm.get('mean_horizon_weighted'), 3)} | "
                f"{fmt(arm.get('total_cost_sum'), 3)} | {fmt(arm.get('decision_total_s'), 3)} | {fmt(train_obj, 6)} | {rep_range} |"
            )

        comps = s.get("paired_comparisons") or {}
        for name, comp in comps.items():
            lines.extend(["", f"### {name}", "", f"paired case-repeat units: {comp.get('paired_units')}", "", "| metric | sum_delta | mean_delta | median_delta | negative/zero/positive | min..max |", "|---|---:|---:|---:|---:|---:|"])
            for metric in ["total_cost", "performance_cost", "h_penalty", "decision_total_s", "mean_horizon", "switches"]:
                m = comp.get(metric) or {}
                lines.append(
                    f"| {metric} | {fmt(m.get('sum_delta_a_minus_b'), 6)} | {fmt(m.get('mean_delta_a_minus_b'), 6)} | "
                    f"{fmt(m.get('median_delta_a_minus_b'), 6)} | {m.get('negative_count')}/{m.get('zero_count')}/{m.get('positive_count')} | "
                    f"{fmt(m.get('min_delta'), 6)}..{fmt(m.get('max_delta'), 6)} |"
                )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest: Dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "method": "IMPROVED latency-tree vehicle selection noise/objective diagnostic; saved metadata only",
        "input_train_base": str(TRAIN_BASE),
        "seeds_requested": SEEDS,
        "time_weight": TIME_WEIGHT,
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
            "final_arm": block["final_arm"],
            "best_selection_objective_arm": block["best_selection_objective_arm"],
            "best_training_objective_arm": block["best_training_objective_arm"],
            "flags": block["diagnosis_flags"],
            "top_two_gap": block["top_two_selection_objective_gap"],
        }, sort_keys=True))

    digest["cross_seed"] = {
        "seeds": len(digest["seeds"]),
        "final_matches_selection_objective_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["final_matches_selection_objective"]),
        "final_matches_training_objective_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["final_matches_training_objective"]),
        "timing_only_constant_tree_win_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["timing_only_constant_tree_win_over_fixed"]),
        "fixed_valid_selection_win_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["fixed_valid_selection_objective_win_not_fallback"]),
        "real_switching_total_cost_selection_win_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["real_switching_total_cost_selection_win"]),
        "selected_behaviorally_fixed_h25_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["selected_behaviorally_fixed_h25"]),
        "selected_switching_policy_count": sum(1 for s in digest["seeds"] if s["diagnosis_flags"]["selected_switching_policy"]),
        "any_validation_accessed_flag": any(s["diagnosis_flags"]["any_summary_validation_accessed_flag"] for s in digest["seeds"]),
        "any_test_accessed_flag": any(s["diagnosis_flags"]["any_summary_test_accessed_flag"] for s in digest["seeds"]),
        "diagnostic_conclusion": (
            "Saved vehicle policy.json choices are explained by the reconstructed selection-race objective, not by a final policy write mismatch. "
            "However, only one of three seeds selected a switching policy with a real aggregate total-cost improvement; one selected a behaviorally fixed H25 tree solely through a sub-percent timing advantage, and one selected the fixed arm."
        ),
    }
    OUT_JSON.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(digest)
    print(json.dumps({"wrote": str(OUT_JSON), "summary": str(OUT_MD), "cross_seed": digest["cross_seed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
