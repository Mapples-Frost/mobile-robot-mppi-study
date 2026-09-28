#!/usr/bin/env python3
"""V2 read-only opportunity/runtime diagnostic for vehicle risk-reselection devval.

Repairs v1 limitations:
  * parse shard cases from completed.json (not only markdown lines with backticks),
  * parse actual per-episode decision/solver timing from nested timing summaries
    (decision_timing_s.mean, solver_attempt_timing_s.mean),
  * mark v1's fixed-H timing-opportunity conclusion as superseded.

No simulation, no training, no sealed final-test access, and no historical validation64
bank access.  Reads only already materialized fresh devval shard outputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path("research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1")
OUT_DEFAULT = Path("research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z")
STATE_DIR = Path("research_artifacts/aws_state")
BACKUP_DIR = Path("research_artifacts/aws_backup_proofs")
POLICY_SHORT_H = {0: 20, 1: 20, 2: 15}
RE_EPISODE = re.compile(r"exec(?P<exec>\d+)_case(?P<case>\d+)_(?P<arm>.+)$")
RE_MATCHED = re.compile(r"matched_terminal_fixed_H(?P<H>\d+)_vehicle_s(?P<seed>\d+)$")
RE_INDEP = re.compile(r"independent_terminal_seed(?P<seed>\d+)_fixed_H(?P<H>\d+)$")
RE_RISK = re.compile(r"risk_reselected_v1_vehicle_s(?P<seed>\d+)$")
RE_CURRENT = re.compile(r"current_gated_vehicle_s(?P<seed>\d+)$")
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RE_SUMMARY_RISK = re.compile(
    r"^- seed (?P<seed>\d+) risk-vs-H25: .*?success risk/fixed=(?P<succ>[^,]+), "
    r"physical_delta_sum=(?P<phys>" + NUM + r"), total_delta_sum=(?P<total>" + NUM + r"), "
    r"decision_ratio_mean=(?P<ratio>" + NUM + r")"
)
RE_SUMMARY_CURRENT = re.compile(
    r"\s*current-vs-H25 physical_delta_sum=(?P<phys>" + NUM + r"), total_delta_sum=(?P<total>" + NUM + r"), "
    r"decision_ratio_mean=(?P<ratio>" + NUM + r"); risk-vs-current physical_delta_sum=(?P<rphys>" + NUM + r"), "
    r"total_delta_sum=(?P<rtotal>" + NUM + r"), decision_ratio_mean=(?P<rratio>" + NUM + r")"
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_path(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nested_mean(obj: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
        if isinstance(val, dict):
            m = val.get("mean")
            if isinstance(m, (int, float)) and math.isfinite(float(m)):
                return float(m)
    return None


def as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def fnum(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    try:
        return ("%.*g" % (nd, float(x)))
    except Exception:
        return str(x)


def mean_or_none(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    return mean(vals) if vals else None


def median_or_none(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    return median(vals) if vals else None


def safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b <= 0:
        return None
    return float(a) / float(b)


def weighted_mean(pairs: Iterable[Tuple[Optional[float], int]]) -> Optional[float]:
    total_w = 0
    total = 0.0
    for val, w in pairs:
        if val is None or w <= 0:
            continue
        total += float(val) * int(w)
        total_w += int(w)
    return total / total_w if total_w else None


def simple_stats(vals: List[float]) -> Dict[str, Any]:
    return {
        "n": len(vals),
        "mean": mean_or_none(vals),
        "median": median_or_none(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "lt_0_count": sum(1 for v in vals if v < -1e-6),
        "near_le_0p05_count": sum(1 for v in vals if v <= 0.05),
    }


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    with path.open("a", encoding="utf-8") as f:
        if old and not old.endswith("\n"):
            f.write("\n")
        f.write("\n" + block.strip() + "\n")


def detect_shards(max_shard: int) -> List[int]:
    shards: List[int] = []
    for s in range(max_shard + 1):
        if (ROOT / ("shard%02d" % s) / "completed.json").exists():
            shards.append(s)
    missing = [s for s in range(max_shard + 1) if s not in shards]
    if missing:
        raise RuntimeError("Missing completed shards up to max_shard=%d: %s" % (max_shard, missing))
    return shards


def parse_shard_inventory(shards: List[int]) -> Dict[str, Any]:
    inv = {"shards": {}, "episodes": 0, "control_steps": 0, "cases": []}
    summary_risk = defaultdict(lambda: {"phys": 0.0, "total": 0.0, "ratio": [], "succ": []})
    summary_current = defaultdict(lambda: {"phys": 0.0, "total": 0.0, "ratio": [], "risk_phys": 0.0, "risk_total": 0.0, "risk_ratio": []})
    for s in shards:
        cpath = ROOT / ("shard%02d" % s) / "completed.json"
        cobj = json.loads(cpath.read_text(encoding="utf-8"))
        cases = [int(x) for x in cobj.get("cases", [])]
        episodes = int(cobj.get("episodes", 0))
        steps = int(cobj.get("control_steps", 0))
        inv["cases"].extend(cases)
        inv["episodes"] += episodes
        inv["control_steps"] += steps
        inv["shards"][str(s)] = {"completed_path": str(cpath), "summary_path": str(ROOT / ("shard%02d" % s) / "summary.md"), "cases": cases, "episodes": episodes, "control_steps": steps}
        spath = ROOT / ("shard%02d" % s) / "summary.md"
        if spath.exists():
            last_seed = None
            for line in spath.read_text(encoding="utf-8").splitlines():
                m = RE_SUMMARY_RISK.match(line)
                if m:
                    seed = int(m.group("seed")); last_seed = seed
                    summary_risk[seed]["phys"] += float(m.group("phys"))
                    summary_risk[seed]["total"] += float(m.group("total"))
                    summary_risk[seed]["ratio"].append(float(m.group("ratio")))
                    summary_risk[seed]["succ"].append(m.group("succ"))
                m = RE_SUMMARY_CURRENT.match(line)
                if m and last_seed is not None:
                    seed = last_seed
                    summary_current[seed]["phys"] += float(m.group("phys"))
                    summary_current[seed]["total"] += float(m.group("total"))
                    summary_current[seed]["ratio"].append(float(m.group("ratio")))
                    summary_current[seed]["risk_phys"] += float(m.group("rphys"))
                    summary_current[seed]["risk_total"] += float(m.group("rtotal"))
                    summary_current[seed]["risk_ratio"].append(float(m.group("rratio")))
    inv["unique_cases"] = sorted(set(inv["cases"]))
    inv["summary_risk_vs_h25_by_seed"] = {str(k): {"physical_delta_sum": v["phys"], "total_delta_sum": v["total"], "decision_ratio_mean_unweighted_shards": mean_or_none(v["ratio"]), "decision_ratio_median_unweighted_shards": median_or_none(v["ratio"]), "success_strings": v["succ"]} for k, v in sorted(summary_risk.items())}
    inv["summary_current_vs_h25_by_seed"] = {str(k): {"physical_delta_sum": v["phys"], "total_delta_sum": v["total"], "decision_ratio_mean_unweighted_shards": mean_or_none(v["ratio"]), "risk_vs_current_physical_delta_sum": v["risk_phys"], "risk_vs_current_total_delta_sum": v["risk_total"], "risk_vs_current_decision_ratio_mean_unweighted_shards": mean_or_none(v["risk_ratio"])} for k, v in sorted(summary_current.items())}
    return inv


def classify_arm(arm: str) -> Tuple[str, Optional[int], Optional[int]]:
    m = RE_MATCHED.match(arm)
    if m:
        return "matched", int(m.group("seed")), int(m.group("H"))
    m = RE_INDEP.match(arm)
    if m:
        return "independent", int(m.group("seed")), int(m.group("H"))
    m = RE_RISK.match(arm)
    if m:
        return "risk", int(m.group("seed")), None
    m = RE_CURRENT.match(arm)
    if m:
        return "current", int(m.group("seed")), None
    return "other", None, None


def load_episode_records(shards: List[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing = Counter()
    for s in shards:
        pattern = str(ROOT / ("shard%02d" % s) / "episodes" / "*" / "summary.json")
        for name in sorted(glob.glob(pattern)):
            p = Path(name)
            m = RE_EPISODE.match(p.parent.name)
            if not m:
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            arm = m.group("arm")
            family, seed, H = classify_arm(arm)
            horizons = obj.get("horizon_counts") or obj.get("raw_horizon_counts_before_clamp") or {}
            horizons = {str(k): int(v) for k, v in horizons.items()}
            steps = int(obj.get("steps") or sum(horizons.values()) or 0)
            rec = {
                "path": str(p),
                "shard": s,
                "case": int(m.group("case")),
                "execution_index": int(m.group("exec")),
                "arm": arm,
                "family": family,
                "seed": seed,
                "H": H,
                "success": bool(obj.get("success")),
                "episode_failure": bool(obj.get("episode_failure")),
                "termination": obj.get("termination"),
                "physical": as_float(obj.get("physical_constraint_cost", obj.get("performance_cost"))),
                "total": as_float(obj.get("total_cost")),
                "h_penalty": as_float(obj.get("h_penalty")),
                "decision_mean": nested_mean(obj, "decision_mean_s_per_step", "decision_timing_s"),
                "decision_gross_mean": nested_mean(obj, "decision_gross_timing_s"),
                "solver_attempt_mean": nested_mean(obj, "solver_mean_s_per_attempt", "solver_attempt_timing_s"),
                "controller_logging_deducted_mean": nested_mean(obj, "controller_timing_s_logging_deducted"),
                "selection_mean": nested_mean(obj, "selection_timing_s"),
                "logging_mean": nested_mean(obj, "logging_timing_s"),
                "construction_s": as_float(obj.get("construction_s")),
                "reset_gross_s": as_float((obj.get("reset") or {}).get("reset_gross_s") if isinstance(obj.get("reset"), dict) else None),
                "steps": steps,
                "horizons": horizons,
                "solver_failure_steps": int(obj.get("solver_failure_steps") or 0),
                "initial_failed_steps": int(obj.get("initial_failed_steps") or 0),
                "constraint": bool(obj.get("constraint")),
            }
            for key in ["physical", "total", "decision_mean", "solver_attempt_mean", "steps"]:
                if rec.get(key) is None:
                    missing[key] += 1
            records.append(rec)
    return records, {"episode_summary_records": len(records), "missing_field_counts": dict(missing)}


def aggregate_adaptive(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = defaultdict(lambda: {"episodes": 0, "success": 0, "failures": 0, "constraints": 0, "init_fail": 0, "solver_fail_steps": 0, "steps": 0, "horizons": Counter(), "decision_weighted": [], "solver_weighted": [], "selection_weighted": [], "physical": 0.0, "total": 0.0})
    for r in records:
        if r["family"] not in ("risk", "current"):
            continue
        key = "%s_seed%d" % (r["family"], r["seed"])
        a = out[key]
        a["episodes"] += 1
        a["success"] += int(bool(r["success"]))
        a["failures"] += int(bool(r["episode_failure"]))
        a["constraints"] += int(bool(r["constraint"]))
        a["init_fail"] += int(r["initial_failed_steps"] > 0)
        a["solver_fail_steps"] += int(r["solver_failure_steps"])
        a["steps"] += int(r["steps"])
        a["horizons"].update(r["horizons"])
        a["decision_weighted"].append((r["decision_mean"], r["steps"]))
        a["solver_weighted"].append((r["solver_attempt_mean"], r["steps"]))
        a["selection_weighted"].append((r["selection_mean"], r["steps"]))
        if r["physical"] is not None:
            a["physical"] += r["physical"]
        if r["total"] is not None:
            a["total"] += r["total"]
    final = {}
    for key, a in sorted(out.items()):
        short = sum(v for k, v in a["horizons"].items() if int(k) < 25)
        long = sum(v for k, v in a["horizons"].items() if int(k) > 25)
        final[key] = {
            "episodes": a["episodes"], "success": a["success"], "failures": a["failures"],
            "constraints": a["constraints"], "init_fail_episodes": a["init_fail"], "solver_failure_steps": a["solver_fail_steps"],
            "steps": a["steps"], "horizons": {str(k): int(v) for k, v in sorted(a["horizons"].items(), key=lambda kv: int(kv[0]))},
            "short_steps": short, "short_fraction": short / a["steps"] if a["steps"] else None,
            "long_steps": long, "long_fraction": long / a["steps"] if a["steps"] else None,
            "decision_mean_s_per_step_weighted": weighted_mean(a["decision_weighted"]),
            "solver_attempt_mean_s_weighted": weighted_mean(a["solver_weighted"]),
            "selection_mean_s_per_step_weighted": weighted_mean(a["selection_weighted"]),
            "physical_sum": a["physical"], "total_sum": a["total"],
        }
    return final


def fixed_groups(records: List[Dict[str, Any]], family: str) -> Dict[Tuple[int, int], Dict[int, Dict[str, Any]]]:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        if r["family"] == family and r["seed"] is not None and r["H"] is not None:
            groups[(int(r["case"]), int(r["seed"]))][int(r["H"])] = r
    return groups


def opportunity(records: List[Dict[str, Any]], family: str) -> Dict[str, Any]:
    groups = fixed_groups(records, family)
    best_phys = Counter(); best_total = Counter()
    deltas_phys = defaultdict(list); deltas_total = defaultdict(list)
    ratios_decision = defaultdict(list); ratios_solver = defaultdict(list)
    counts = Counter(); examples = {"short_near_faster": [], "short_better_slow": [], "long_better": [], "no_short_near": []}
    policy_counts_by_seed = defaultdict(Counter)
    for (case, seed), hs in sorted(groups.items()):
        counts["groups"] += 1
        base = hs.get(25)
        if not base or base["physical"] is None:
            continue
        counts["with_H25"] += 1
        success_recs = [r for r in hs.values() if r["success"] and r["physical"] is not None]
        if success_recs:
            bp = min(success_recs, key=lambda r: (float(r["physical"]), int(r["H"])))
            best_phys[int(bp["H"])] += 1
            totalable = [r for r in success_recs if r["total"] is not None]
            if totalable:
                bt = min(totalable, key=lambda r: (float(r["total"]), int(r["H"])))
                best_total[int(bt["H"])] += 1
        any_short_better = any_short_near = any_short_total = any_short_fast = any_short_near_fast = False
        any_long_better = any_long_total = False
        best_short_row = None; best_long_row = None
        for H, r in sorted(hs.items()):
            if H == 25 or (not r["success"]) or r["physical"] is None:
                continue
            dphys = float(r["physical"]) - float(base["physical"])
            dtot = (float(r["total"]) - float(base["total"])) if r["total"] is not None and base["total"] is not None else None
            dr = safe_ratio(r["decision_mean"], base["decision_mean"])
            sr = safe_ratio(r["solver_attempt_mean"], base["solver_attempt_mean"])
            deltas_phys[H].append(dphys)
            if dtot is not None: deltas_total[H].append(dtot)
            if dr is not None: ratios_decision[H].append(dr)
            if sr is not None: ratios_solver[H].append(sr)
            row = {"case": case, "seed": seed, "H": H, "physical_delta_vs_H25": dphys, "total_delta_vs_H25": dtot, "decision_ratio_vs_H25": dr, "solver_attempt_ratio_vs_H25": sr, "decision_mean": r["decision_mean"], "H25_decision_mean": base["decision_mean"], "steps": r["steps"], "H25_steps": base["steps"]}
            if H < 25:
                any_short_better = any_short_better or dphys < -1e-6
                any_short_near = any_short_near or dphys <= 0.05
                any_short_total = any_short_total or (dtot is not None and dtot < -1e-6)
                any_short_fast = any_short_fast or (dr is not None and dr < 0.98)
                any_short_near_fast = any_short_near_fast or (dphys <= 0.05 and dr is not None and dr < 0.98)
                if H == POLICY_SHORT_H.get(seed):
                    pc = policy_counts_by_seed[seed]
                    pc["available"] += 1
                    pc["physical_better"] += int(dphys < -1e-6)
                    pc["near"] += int(dphys <= 0.05)
                    pc["total_better"] += int(dtot is not None and dtot < -1e-6)
                    pc["faster_2pct"] += int(dr is not None and dr < 0.98)
                    pc["near_faster"] += int(dphys <= 0.05 and dr is not None and dr < 0.98)
                if best_short_row is None or (dphys, H) < (best_short_row["physical_delta_vs_H25"], best_short_row["H"]):
                    best_short_row = row
            else:
                any_long_better = any_long_better or dphys < -1e-6
                any_long_total = any_long_total or (dtot is not None and dtot < -1e-6)
                if best_long_row is None or (dphys, H) < (best_long_row["physical_delta_vs_H25"], best_long_row["H"]):
                    best_long_row = row
        counts["short_any_physical_better"] += int(any_short_better)
        counts["short_any_near"] += int(any_short_near)
        counts["short_any_total_better"] += int(any_short_total)
        counts["short_any_fast"] += int(any_short_fast)
        counts["short_any_near_fast"] += int(any_short_near_fast)
        counts["long_any_physical_better"] += int(any_long_better)
        counts["long_any_total_better"] += int(any_long_total)
        if any_short_near_fast and len(examples["short_near_faster"]) < 12: examples["short_near_faster"].append(best_short_row)
        if any_short_better and not any_short_fast and len(examples["short_better_slow"]) < 12: examples["short_better_slow"].append(best_short_row)
        if any_long_better and len(examples["long_better"]) < 12: examples["long_better"].append(best_long_row)
        if not any_short_near and len(examples["no_short_near"]) < 12: examples["no_short_near"].append({"case": case, "seed": seed, "best_short_by_physical": best_short_row})
    fixed_record_count = sum(1 for r in records if r["family"] == family)
    return {
        "family": family,
        "fixed_episode_records": fixed_record_count,
        "case_seed_groups": counts["groups"], "case_seed_groups_with_H25": counts["with_H25"],
        "best_physical_horizon_counts": {str(k): int(v) for k, v in sorted(best_phys.items())},
        "best_total_horizon_counts": {str(k): int(v) for k, v in sorted(best_total.items())},
        "H25_physical_best_count": int(best_phys.get(25, 0)), "H25_total_best_count": int(best_total.get(25, 0)),
        "short_H_any_physical_better_count": counts["short_any_physical_better"],
        "short_H_any_physical_near_equal_le_0p05_count": counts["short_any_near"],
        "short_H_any_total_better_count": counts["short_any_total"],
        "short_H_any_measured_faster_by_2pct_count": counts["short_any_fast"],
        "short_H_any_near_equal_and_faster_count": counts["short_any_near_fast"],
        "long_H_any_physical_better_count": counts["long_any_physical_better"],
        "long_H_any_total_better_count": counts["long_any_total"],
        "policy_short_counts_by_seed": {str(k): dict(v) for k, v in sorted(policy_counts_by_seed.items())},
        "decision_ratios_vs_H25_by_H": {str(k): {"n": len(v), "mean": mean_or_none(v), "median": median_or_none(v), "lt_0p98_count": sum(1 for x in v if x < 0.98), "gt_1p02_count": sum(1 for x in v if x > 1.02), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in sorted(ratios_decision.items())},
        "solver_ratios_vs_H25_by_H": {str(k): {"n": len(v), "mean": mean_or_none(v), "median": median_or_none(v), "lt_0p98_count": sum(1 for x in v if x < 0.98), "gt_1p02_count": sum(1 for x in v if x > 1.02), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in sorted(ratios_solver.items())},
        "physical_delta_vs_H25_by_H": {str(k): simple_stats(v) for k, v in sorted(deltas_phys.items())},
        "total_delta_vs_H25_by_H": {str(k): simple_stats(v) for k, v in sorted(deltas_total.items())},
        "examples": examples,
    }


def pair_adaptive(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fixed = {(r["case"], r["seed"], r["H"]): r for r in records if r["family"] == "matched" and r["H"] is not None}
    adaptive = [r for r in records if r["family"] in ("risk", "current")]
    by_family_seed = defaultdict(lambda: {"episodes": 0, "success": 0, "physical_delta": 0.0, "total_delta": 0.0, "decision_ratios": [], "solver_ratios": [], "selection_means": [], "identical_H25_only_episodes": 0, "identical_H25_only_decision_ratios": [], "short_steps": 0, "steps": 0})
    examples = []
    for r in adaptive:
        base = fixed.get((r["case"], r["seed"], 25))
        if not base:
            continue
        key = "%s_seed%d" % (r["family"], r["seed"])
        a = by_family_seed[key]
        a["episodes"] += 1
        a["success"] += int(bool(r["success"]))
        if r["physical"] is not None and base["physical"] is not None:
            a["physical_delta"] += float(r["physical"]) - float(base["physical"])
        if r["total"] is not None and base["total"] is not None:
            a["total_delta"] += float(r["total"]) - float(base["total"])
        dr = safe_ratio(r["decision_mean"], base["decision_mean"])
        sr = safe_ratio(r["solver_attempt_mean"], base["solver_attempt_mean"])
        if dr is not None: a["decision_ratios"].append(dr)
        if sr is not None: a["solver_ratios"].append(sr)
        if r["selection_mean"] is not None: a["selection_means"].append(r["selection_mean"])
        h = {int(k): int(v) for k, v in r["horizons"].items()}
        short = sum(v for H, v in h.items() if H < 25)
        a["short_steps"] += short; a["steps"] += int(r["steps"])
        if h == {25: int(r["steps"])}:
            a["identical_H25_only_episodes"] += 1
            if dr is not None: a["identical_H25_only_decision_ratios"].append(dr)
        if len(examples) < 20:
            examples.append({"family": r["family"], "seed": r["seed"], "case": r["case"], "horizons": r["horizons"], "physical_delta_vs_H25": (float(r["physical"]) - float(base["physical"])) if r["physical"] is not None and base["physical"] is not None else None, "total_delta_vs_H25": (float(r["total"]) - float(base["total"])) if r["total"] is not None and base["total"] is not None else None, "decision_ratio_vs_H25": dr, "solver_ratio_vs_H25": sr, "selection_mean_s": r["selection_mean"], "adaptive_decision_mean": r["decision_mean"], "H25_decision_mean": base["decision_mean"]})
    final = {}
    for key, a in sorted(by_family_seed.items()):
        final[key] = {
            "episodes": a["episodes"], "success": a["success"],
            "physical_delta_sum": a["physical_delta"], "total_delta_sum": a["total_delta"],
            "decision_ratio_mean": mean_or_none(a["decision_ratios"]), "decision_ratio_median": median_or_none(a["decision_ratios"]),
            "solver_ratio_mean": mean_or_none(a["solver_ratios"]), "selection_mean_s": mean_or_none(a["selection_means"]),
            "identical_H25_only_episodes": a["identical_H25_only_episodes"],
            "identical_H25_only_decision_ratio_mean": mean_or_none(a["identical_H25_only_decision_ratios"]),
            "short_steps": a["short_steps"], "steps": a["steps"], "short_fraction": a["short_steps"] / a["steps"] if a["steps"] else None,
        }
    return {"adaptive_vs_matched_H25_by_family_seed": final, "pair_examples_first20": examples}


def opportunity_sets(records: List[Dict[str, Any]]) -> Tuple[set, set, set, set]:
    groups = fixed_groups(records, "matched")
    any_near, any_near_fast, pol_near, pol_near_fast = set(), set(), set(), set()
    for (case, seed), hs in groups.items():
        base = hs.get(25)
        if not base or base["physical"] is None:
            continue
        for H, r in hs.items():
            if H >= 25 or not r["success"] or r["physical"] is None:
                continue
            dphys = float(r["physical"]) - float(base["physical"])
            dr = safe_ratio(r["decision_mean"], base["decision_mean"])
            if dphys <= 0.05:
                any_near.add((case, seed))
            if dphys <= 0.05 and dr is not None and dr < 0.98:
                any_near_fast.add((case, seed))
            if H == POLICY_SHORT_H.get(seed):
                if dphys <= 0.05:
                    pol_near.add((case, seed))
                if dphys <= 0.05 and dr is not None and dr < 0.98:
                    pol_near_fast.add((case, seed))
    return any_near, any_near_fast, pol_near, pol_near_fast


def alignment(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    any_near, any_near_fast, pol_near, pol_near_fast = opportunity_sets(records)
    out = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r["family"] not in ("risk", "current") or r["seed"] is None:
            continue
        key = "%s_seed%d" % (r["family"], r["seed"])
        h = {int(k): int(v) for k, v in r["horizons"].items()}
        steps = int(r["steps"]); short = sum(v for H, v in h.items() if H < 25)
        pair = (r["case"], r["seed"])
        a = out[key]
        a["episodes"] += 1; a["steps"] += steps; a["short_steps"] += short; a["episodes_with_short"] += int(short > 0)
        a["any_short_near_opp_episodes"] += int(pair in any_near)
        a["any_short_near_fast_opp_episodes"] += int(pair in any_near_fast)
        a["policy_short_near_opp_episodes"] += int(pair in pol_near)
        a["policy_short_near_fast_opp_episodes"] += int(pair in pol_near_fast)
        if short > 0:
            a["short_episode_with_any_near_opp"] += int(pair in any_near)
            a["short_episode_without_any_near_opp"] += int(pair not in any_near)
            a["short_episode_with_policy_near_opp"] += int(pair in pol_near)
            a["short_episode_without_policy_near_opp"] += int(pair not in pol_near)
    final = {}
    for key, a in sorted(out.items()):
        d = dict(a); d["short_fraction"] = a["short_steps"] / a["steps"] if a["steps"] else None
        final[key] = d
    return {"opportunity_set_sizes": {"any_short_near": len(any_near), "any_short_near_fast": len(any_near_fast), "policy_short_near": len(pol_near), "policy_short_near_fast": len(pol_near_fast)}, "by_family_seed": final}


def interpret(inv: Dict[str, Any], recdiag: Dict[str, Any], adaptive: Dict[str, Any], matched: Dict[str, Any], indep: Dict[str, Any], align: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], str]:
    verified: List[str] = []
    hypotheses: List[str] = []
    limitations: List[str] = []
    if recdiag.get("missing_field_counts", {}).get("decision_mean", 0) == 0:
        verified.append("V2 repaired the v1 timing parser: per-episode decision timing was read from decision_timing_s.mean for all parsed episode summaries.")
    else:
        limitations.append("Some per-episode decision timing is still missing: %s" % recdiag.get("missing_field_counts"))
    for seed in [0, 1, 2]:
        row = adaptive["adaptive_vs_matched_H25_by_family_seed"].get("risk_seed%d" % seed, {})
        if row:
            verified.append("Risk seed%d vs matched H25 over completed shards: physical_delta_sum=%s, total_delta_sum=%s, decision_ratio_mean=%s, solver_ratio_mean=%s, short_fraction=%s." % (seed, fnum(row.get("physical_delta_sum")), fnum(row.get("total_delta_sum")), fnum(row.get("decision_ratio_mean")), fnum(row.get("solver_ratio_mean")), fnum(row.get("short_fraction"), 4)))
            if (row.get("decision_ratio_mean") or 0) > 1.0:
                verified.append("Risk seed%d is not actually faster than H25 in measured decision time on this partial devval set." % seed)
            if (row.get("selection_mean_s") or 0) > 0:
                verified.append("Risk seed%d incurs measurable selection overhead (mean %s s/step) even when many episodes execute H25." % (seed, fnum(row.get("selection_mean_s"), 4)))
    groups = matched.get("case_seed_groups_with_H25", 0) or 0
    nf = matched.get("short_H_any_near_equal_and_faster_count", 0) or 0
    near = matched.get("short_H_any_physical_near_equal_le_0p05_count", 0) or 0
    faster = matched.get("short_H_any_measured_faster_by_2pct_count", 0) or 0
    verified.append("Matched fixed-grid opportunity on identical starts: %d/%d case-seed groups have a short H within 0.05 physical cost of H25; %d/%d have a short H at least 2%% faster; %d/%d have both." % (near, groups, faster, groups, nf, groups))
    if nf == 0:
        verified.append("On completed shards, measured wall-time eliminates the nominal short-H opportunity under the >=2%% speed criterion; synthetic h_penalty benefits should not be treated as real acceleration.")
    elif nf < groups / 3.0:
        hypotheses.append("Some real near/faster short-H opportunity exists but is sparse; a selector must target a small subset of states/cases rather than broadly shorten.")
    else:
        hypotheses.append("There is substantial real short-H opportunity in fixed-grid episodes; current gates likely fail to exploit it consistently.")
    long_better = matched.get("long_H_any_physical_better_count", 0) or 0
    short_better = matched.get("short_H_any_physical_better_count", 0) or 0
    if long_better > short_better:
        verified.append("Longer horizons improve physical cost in more matched fixed-grid groups than shorter horizons (%d vs %d), so strong fixed-H tuning remains a serious competitor." % (long_better, short_better))
    # Alignment details
    for seed in [0, 1, 2]:
        key = "risk_seed%d" % seed
        row = align.get("by_family_seed", {}).get(key, {})
        if row:
            if row.get("policy_short_near_opp_episodes", 0) > row.get("episodes_with_short", 0):
                hypotheses.append("Risk seed%d has more policy-short near-opportunity episodes than episodes with any short execution (%d vs %d), indicating restrictive or misaligned gates." % (seed, row.get("policy_short_near_opp_episodes", 0), row.get("episodes_with_short", 0)))
            if row.get("short_episode_without_policy_near_opp", 0) > 0:
                hypotheses.append("Risk seed%d sometimes executes short H where the policy-nominated short fixed episode is not near-equal to H25 (%d episodes), indicating possible misalignment." % (seed, row.get("short_episode_without_policy_near_opp", 0)))
    limitations.append("This is partial development-validation evidence from shards00-05 only (%d episodes, %d control steps; %d unique cases), not final test and not fresh confirmation validation." % (inv.get("episodes", 0), inv.get("control_steps", 0), len(inv.get("unique_cases", []))))
    limitations.append("Fixed-H opportunity is full-episode from identical initial states, not one-step continuation rollouts from every adaptive decision state; terminal-value bias may remain.")
    limitations.append("This v2 diagnostic supersedes v1's fixed-H timing-opportunity statements because v1 failed to parse nested per-episode decision_timing_s.mean.")
    if nf == 0 or all((adaptive["adaptive_vs_matched_H25_by_family_seed"].get("risk_seed%d" % seed, {}).get("decision_ratio_mean") or 0) >= 0.995 for seed in [0, 1, 2]):
        decision = "Pause additional unchanged long risk-reselection devval shards after backup; freeze an outcome-informed scheduling amendment and design a smaller versioned training/selection or scenario-opportunity experiment with actual measured-time objectives."
    else:
        decision = "After backup, either finish the frozen devval64 campaign for precision or run a targeted selector-refit on the identified near/faster opportunity set; do not open sealed test."
    return verified, hypotheses, limitations, decision


def write_outputs(out: Path, shards: List[int], inv: Dict[str, Any], records: List[Dict[str, Any]], recdiag: Dict[str, Any], adaptive_agg: Dict[str, Any], adaptive_pair: Dict[str, Any], matched: Dict[str, Any], indep: Dict[str, Any], align: Dict[str, Any], verified: List[str], hypotheses: List[str], limitations: List[str], decision: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    raw = {
        "created_utc": now(),
        "diagnostic_type": "read_only_existing_fresh_devval_partial_opportunity_runtime_v2",
        "supersedes_v1_due_to_timing_parse_bug": True,
        "method_classification": "IMPROVED finite direct gated-horizon risk re-selection; no new gradient training; not ORIGINAL SAC",
        "access_flags": {"new_simulations": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_bank_opened": False, "historical_validation64_bank_opened": False, "fresh_devval_existing_outputs_read": True},
        "root": str(ROOT), "completed_shards": shards,
        "inventory": inv,
        "episode_record_diagnostics": recdiag,
        "adaptive_aggregates": adaptive_agg,
        "adaptive_vs_matched_H25": adaptive_pair,
        "matched_terminal_opportunity": matched,
        "independent_terminal_opportunity": indep,
        "policy_alignment": align,
        "verified_findings": verified,
        "live_hypotheses": hypotheses,
        "missing_evidence_or_limitations": limitations,
        "next_allocation_decision": decision,
        "hashes": {str(Path(__file__)): sha256_path(Path(__file__))},
    }
    for p in [ROOT / "gate_completed.json", ROOT / "bank" / "completed.json", ROOT / "bank" / "vehicle_gated_horizon_risk_reselection_v1_devval64_bank.json"]:
        if p.exists(): raw["hashes"][str(p)] = sha256_path(p)
    for s in shards:
        for name in ["summary.md", "completed.json", "raw.json"]:
            p = ROOT / ("shard%02d" % s) / name
            if p.exists(): raw["hashes"][str(p)] = sha256_path(p)
    raw_path = out / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Vehicle risk-reselection v1 partial opportunity/runtime diagnostic V2",
        "", "Created UTC: `%s`." % now(), "",
        "Read-only diagnostic over already generated fresh development-validation shards. No simulations, no training, no sealed-test access, and no historical validation64 bank access.",
        "", "V2 supersedes the timing-opportunity statements from the first diagnostic because v1 did not parse nested per-episode `decision_timing_s.mean` and therefore had empty fixed-H timing ratios.",
        "", "## Evidence inventory", "",
        "- Shards analyzed: `%s` (%d / 16 risk-reselection devval shards)." % (shards, len(shards)),
        "- Case slots / unique cases: `%d` / `%d`; cases: `%s`." % (len(inv.get("cases", [])), len(inv.get("unique_cases", [])), inv.get("unique_cases", [])),
        "- Shard-reported episodes/control steps: `%s` / `%s`." % (inv.get("episodes"), inv.get("control_steps")),
        "- Episode summary records parsed: `%s`; missing fields: `%s`." % (recdiag.get("episode_summary_records"), recdiag.get("missing_field_counts")),
        "", "## Adaptive risk-reselected vs matched H25 (per-episode timing)", "",
    ]
    for key, row in sorted(adaptive_pair["adaptive_vs_matched_H25_by_family_seed"].items()):
        if key.startswith("risk_seed"):
            agg = adaptive_agg.get(key, {})
            lines.append("- %s: episodes=%s, physical_delta_sum=%s, total_delta_sum=%s, decision_ratio_mean=%s, solver_ratio_mean=%s, selection_mean_s=%s, short_fraction=%s, horizons=%s, identical_H25_only_episodes=%s, identical_H25_only_decision_ratio_mean=%s" % (key, row.get("episodes"), fnum(row.get("physical_delta_sum")), fnum(row.get("total_delta_sum")), fnum(row.get("decision_ratio_mean")), fnum(row.get("solver_ratio_mean")), fnum(row.get("selection_mean_s"), 4), fnum(row.get("short_fraction"), 4), agg.get("horizons"), row.get("identical_H25_only_episodes"), fnum(row.get("identical_H25_only_decision_ratio_mean"))))
    lines += ["", "## Matched-terminal fixed-H opportunity with actual timing", ""]
    lines.append("- Groups with H25: `%s`; fixed records: `%s`." % (matched.get("case_seed_groups_with_H25"), matched.get("fixed_episode_records")))
    lines.append("- Best physical horizon counts: `%s`; best total horizon counts: `%s`." % (matched.get("best_physical_horizon_counts"), matched.get("best_total_horizon_counts")))
    lines.append("- Short-H counts: physical_better=%s, near_equal<=0.05=%s, total_better=%s, measured_faster>=2%%=%s, near_equal_and_faster=%s." % (matched.get("short_H_any_physical_better_count"), matched.get("short_H_any_physical_near_equal_le_0p05_count"), matched.get("short_H_any_total_better_count"), matched.get("short_H_any_measured_faster_by_2pct_count"), matched.get("short_H_any_near_equal_and_faster_count")))
    lines.append("- Long-H counts: physical_better=%s, total_better=%s." % (matched.get("long_H_any_physical_better_count"), matched.get("long_H_any_total_better_count")))
    lines.append("- Decision ratios vs H25 by H: `%s`." % matched.get("decision_ratios_vs_H25_by_H"))
    lines.append("- Solver ratios vs H25 by H: `%s`." % matched.get("solver_ratios_vs_H25_by_H"))
    lines += ["", "## Independent-terminal seed0 fixed-H opportunity", ""]
    lines.append("- Groups with H25: `%s`; fixed records: `%s`; best physical horizon counts: `%s`." % (indep.get("case_seed_groups_with_H25"), indep.get("fixed_episode_records"), indep.get("best_physical_horizon_counts")))
    lines.append("- Short-H counts: physical_better=%s, near_equal<=0.05=%s, total_better=%s, measured_faster>=2%%=%s, near_equal_and_faster=%s." % (indep.get("short_H_any_physical_better_count"), indep.get("short_H_any_physical_near_equal_le_0p05_count"), indep.get("short_H_any_total_better_count"), indep.get("short_H_any_measured_faster_by_2pct_count"), indep.get("short_H_any_near_equal_and_faster_count")))
    lines += ["", "## Policy/opportunity alignment", "", "- Opportunity set sizes: `%s`." % align.get("opportunity_set_sizes")]
    for key, row in sorted(align.get("by_family_seed", {}).items()):
        if key.startswith("risk_seed"):
            lines.append("- %s: `%s`" % (key, row))
    lines += ["", "## Verified findings", ""] + ["- " + s for s in verified]
    lines += ["", "## Live hypotheses", ""] + (["- " + s for s in hypotheses] if hypotheses else ["- None promoted beyond the verified findings in this bounded read-only diagnostic."])
    lines += ["", "## Missing evidence / limitations", ""] + ["- " + s for s in limitations]
    lines += ["", "## Decision for next allocation", "", decision, ""]
    summary_path = out / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    completed_path = out / "completed.json"
    backup_path = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_RISK_RESELECTION_PARTIAL_OPPORTUNITY_DIAGNOSTIC_V2_20260928T0855Z.json"
    completed = {"created_utc": now(), "passed": True, "summary_path": str(summary_path), "raw_path": str(raw_path), "completed_shards": shards, "episodes_read_from_summaries": inv.get("episodes"), "episode_summary_records": recdiag.get("episode_summary_records"), "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_simulations": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "next_allocation_decision": decision, "backup_request": str(backup_path)}
    completed_path.write_text(json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8")
    state_path = STATE_DIR / "vehicle_risk_reselection_partial_opportunity_diagnostic_v2_20260928T0855Z.md"
    state_lines = ["# Vehicle risk-reselection partial opportunity/runtime diagnostic V2 state", "", "UTC: %s" % now(), "", "- Completed read-only V2 diagnostic over shards: %s" % shards, "- V2 supersedes v1 fixed-H timing-opportunity claims; v1 missed nested decision_timing_s.mean.", "- No new simulations/training/gradient steps; sealed test not opened; historical validation64 bank not opened.", "- Summary: `%s`" % summary_path, "- Raw: `%s`" % raw_path, "- Completed marker: `%s`" % completed_path, "- Decision: %s" % decision, "", "Verified findings:"] + ["- " + s for s in verified] + ["", "Limitations:"] + ["- " + s for s in limitations]
    state_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")
    backup = {"created_utc": now(), "reason": "Backup required after V2 corrected read-only partial opportunity/runtime diagnostic before any further simulations.", "backup_required_before_more_simulations": True, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_simulations": 0, "artifacts_requiring_backup": [str(summary_path), str(raw_path), str(completed_path), str(state_path), str(backup_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "EXPERIMENT_REGISTRY.csv", str(Path(__file__))], "summary_sha256": sha256_path(summary_path), "raw_sha256": sha256_path(raw_path), "completed_sha256": sha256_path(completed_path), "state_sha256": sha256_path(state_path)}
    backup_path.write_text(json.dumps(backup, indent=2, sort_keys=True), encoding="utf-8")
    completed["completed_sha256_pre_backup_field"] = sha256_path(completed_path)
    completed["backup_request_sha256"] = sha256_path(backup_path)
    completed_path.write_text(json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8")
    marker = "<!-- vehicle-risk-reselection-partial-opportunity-diagnostic-v2-20260928 -->"
    block = "%s\n## 2026-09-28 risk-reselection partial opportunity/runtime diagnostic V2\n\nUTC: %s. Corrected read-only diagnostic over fresh devval shards %s; no simulations, no training, no sealed-test access, and no historical validation64 bank access. V2 supersedes v1 fixed-H timing-opportunity statements because v1 did not parse nested per-episode timing. Parsed %s per-episode summaries and shard-reported %s episodes / %s control steps.\n\nDecision: %s\n\nArtifacts: `%s`, `%s`, `%s`. New diagnostic artifacts and doc updates require external backup before further simulations.\n" % (marker, now(), shards, recdiag.get("episode_summary_records"), inv.get("episodes"), inv.get("control_steps"), decision, summary_path, raw_path, completed_path)
    for doc in [Path("STATUS.md"), Path("RESEARCH_LOG.md"), Path("DECISIONS.md"), Path("RESULTS_AUDIT.md")]:
        append_once(doc, marker, block)
    print("\n".join(lines))
    print("Artifacts:")
    for p in [summary_path, raw_path, completed_path, state_path, backup_path]:
        print("- %s" % p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shard", type=int, default=5)
    ap.add_argument("--output-dir", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    shards = detect_shards(args.max_shard)
    inv = parse_shard_inventory(shards)
    records, recdiag = load_episode_records(shards)
    adaptive_agg = aggregate_adaptive(records)
    adaptive_pair = pair_adaptive(records)
    matched = opportunity(records, "matched")
    indep = opportunity(records, "independent")
    align = alignment(records)
    verified, hypotheses, limitations, decision = interpret(inv, recdiag, adaptive_pair, matched, indep, align)
    write_outputs(Path(args.output_dir), shards, inv, records, recdiag, adaptive_agg, adaptive_pair, matched, indep, align, verified, hypotheses, limitations, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
