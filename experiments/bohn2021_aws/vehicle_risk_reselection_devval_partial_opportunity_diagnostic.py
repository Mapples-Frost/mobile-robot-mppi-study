#!/usr/bin/env python3
"""Read-only partial opportunity/runtime diagnostic for vehicle risk reselection.

This script reads only already-generated fresh development-validation outputs from
vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1.  It runs no
simulation, no training, opens no sealed final test, and does not open the historical
validation64 bank.
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path("research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1")
OUT_DEFAULT = Path("research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z")
STATE_DIR = Path("research_artifacts/aws_state")
BACKUP_DIR = Path("research_artifacts/aws_backup_proofs")

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RE_CASES = re.compile(r"^- Shard cases: \[(?P<cases>[^\]]*)\]")
RE_BUDGET = re.compile(r"^- Shard episodes/control steps: `(?P<eps>\d+)` / `(?P<steps>\d+)`")
RE_RISK = re.compile(
    r"^- seed (?P<seed>\d+) risk-vs-H25: .*?success risk/fixed=(?P<succ>[^,]+), "
    r"physical_delta_sum=(?P<phys>" + NUM + r"), total_delta_sum=(?P<total>" + NUM + r"), "
    r"decision_ratio_mean=(?P<ratio>" + NUM + r")"
)
RE_CURRENT = re.compile(
    r"\s*current-vs-H25 physical_delta_sum=(?P<phys>" + NUM + r"), total_delta_sum=(?P<total>" + NUM + r"), "
    r"decision_ratio_mean=(?P<ratio>" + NUM + r"); risk-vs-current physical_delta_sum=(?P<rphys>" + NUM + r"), "
    r"total_delta_sum=(?P<rtotal>" + NUM + r"), decision_ratio_mean=(?P<rratio>" + NUM + r")"
)
RE_ADAPT = re.compile(
    r"^- `(?P<arm>[^`]+)`: episodes=(?P<episodes>\d+), success=(?P<success>\d+), failures=(?P<failures>\d+), "
    r"constraints=(?P<constraints>\d+), init_fail=(?P<init>\d+), final_fail=(?P<final>\d+), phys=(?P<phys>" + NUM + r"), "
    r"total=(?P<total>" + NUM + r"), decision_mean_s_per_step=(?P<decision>" + NUM + r"), horizons=(?P<horizons>\{.*?\}),"
)
RE_EPISODE = re.compile(r"exec(?P<exec>\d+)_case(?P<case>\d+)_(?P<arm>.+)$")
RE_MATCHED = re.compile(r"matched_terminal_fixed_H(?P<H>\d+)_vehicle_s(?P<seed>\d+)$")
RE_INDEPENDENT = re.compile(r"independent_terminal_seed(?P<seed>\d+)_fixed_H(?P<H>\d+)$")
RE_RISK_ARM = re.compile(r"risk_reselected_v1_vehicle_s(?P<seed>\d+)$")
RE_CURRENT_ARM = re.compile(r"current_gated_vehicle_s(?P<seed>\d+)$")
POLICY_SHORT_H = {0: 20, 1: 20, 2: 15}


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


def fnum(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return "NA"
    return ("%.*g" % (nd, x))


def parse_horizon_counts(s: str) -> Dict[str, int]:
    return {str(int(k)): int(v) for k, v in re.findall(r"'?(\d+)'?\s*:\s*(\d+)", s)}


def mean_or_none(vals: List[float]) -> Optional[float]:
    return mean(vals) if vals else None


def median_or_none(vals: List[float]) -> Optional[float]:
    return median(vals) if vals else None


def safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b <= 0:
        return None
    return float(a) / float(b)


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    with path.open("a", encoding="utf-8") as f:
        if old and not old.endswith("\n"):
            f.write("\n")
        f.write("\n" + text.strip() + "\n")


def parse_summaries(shards: List[int]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {"shards": {}, "episodes": 0, "control_steps": 0, "cases": []}
    risk = defaultdict(lambda: {"phys": 0.0, "total": 0.0, "ratios": [], "success": []})
    current = defaultdict(lambda: {"phys": 0.0, "total": 0.0, "ratios": [], "risk_phys": 0.0, "risk_total": 0.0, "risk_ratios": []})
    arms = defaultdict(lambda: {"episodes": 0, "success": 0, "failures": 0, "constraints": 0, "init_fail": 0, "final_fail": 0, "phys": 0.0, "total": 0.0, "steps": 0, "decision_weighted": 0.0, "horizons": Counter()})
    last_seed: Optional[int] = None
    for shard in shards:
        p = ROOT / ("shard%02d" % shard) / "summary.md"
        text = p.read_text(encoding="utf-8")
        sinfo: Dict[str, Any] = {"summary_path": str(p), "cases": [], "episodes": None, "control_steps": None, "risk_vs_h25": {}, "current_vs_h25": {}}
        for line in text.splitlines():
            m = RE_CASES.match(line)
            if m:
                cases = [int(x.strip()) for x in m.group("cases").split(",") if x.strip()]
                sinfo["cases"] = cases
                parsed["cases"].extend(cases)
            m = RE_BUDGET.match(line)
            if m:
                eps, steps = int(m.group("eps")), int(m.group("steps"))
                sinfo["episodes"] = eps
                sinfo["control_steps"] = steps
                parsed["episodes"] += eps
                parsed["control_steps"] += steps
            m = RE_RISK.match(line)
            if m:
                seed = int(m.group("seed")); last_seed = seed
                phys, total, rr = float(m.group("phys")), float(m.group("total")), float(m.group("ratio"))
                risk[seed]["phys"] += phys
                risk[seed]["total"] += total
                risk[seed]["ratios"].append(rr)
                risk[seed]["success"].append(m.group("succ"))
                sinfo["risk_vs_h25"][str(seed)] = {"physical_delta_sum": phys, "total_delta_sum": total, "decision_ratio_mean": rr, "success": m.group("succ")}
            m = RE_CURRENT.match(line)
            if m and last_seed is not None:
                seed = last_seed
                phys, total, rr = float(m.group("phys")), float(m.group("total")), float(m.group("ratio"))
                rphys, rtotal, rratio = float(m.group("rphys")), float(m.group("rtotal")), float(m.group("rratio"))
                current[seed]["phys"] += phys
                current[seed]["total"] += total
                current[seed]["ratios"].append(rr)
                current[seed]["risk_phys"] += rphys
                current[seed]["risk_total"] += rtotal
                current[seed]["risk_ratios"].append(rratio)
                sinfo["current_vs_h25"][str(seed)] = {"physical_delta_sum": phys, "total_delta_sum": total, "decision_ratio_mean": rr, "risk_vs_current_physical_delta_sum": rphys, "risk_vs_current_total_delta_sum": rtotal, "risk_vs_current_decision_ratio_mean": rratio}
            m = RE_ADAPT.match(line)
            if m:
                arm = m.group("arm")
                h = parse_horizon_counts(m.group("horizons"))
                steps = sum(h.values())
                a = arms[arm]
                a["episodes"] += int(m.group("episodes"))
                a["success"] += int(m.group("success"))
                a["failures"] += int(m.group("failures"))
                a["constraints"] += int(m.group("constraints"))
                a["init_fail"] += int(m.group("init"))
                a["final_fail"] += int(m.group("final"))
                a["phys"] += float(m.group("phys"))
                a["total"] += float(m.group("total"))
                a["steps"] += steps
                a["decision_weighted"] += float(m.group("decision")) * steps
                a["horizons"].update(h)
        parsed["shards"][str(shard)] = sinfo
    parsed["risk_vs_h25_by_seed"] = {}
    for seed, r in sorted(risk.items()):
        parsed["risk_vs_h25_by_seed"][str(seed)] = {"physical_delta_sum": r["phys"], "total_delta_sum": r["total"], "decision_ratio_mean_unweighted_shards": mean_or_none(r["ratios"]), "decision_ratio_median_unweighted_shards": median_or_none(r["ratios"]), "success_strings": r["success"]}
    parsed["current_vs_h25_by_seed"] = {}
    for seed, r in sorted(current.items()):
        parsed["current_vs_h25_by_seed"][str(seed)] = {"physical_delta_sum": r["phys"], "total_delta_sum": r["total"], "decision_ratio_mean_unweighted_shards": mean_or_none(r["ratios"]), "risk_vs_current_physical_delta_sum": r["risk_phys"], "risk_vs_current_total_delta_sum": r["risk_total"], "risk_vs_current_decision_ratio_mean_unweighted_shards": mean_or_none(r["risk_ratios"])}
    parsed["adaptive_arm_counts"] = {}
    for arm, a in sorted(arms.items()):
        steps = a["steps"]
        short = sum(v for k, v in a["horizons"].items() if int(k) < 25)
        parsed["adaptive_arm_counts"][arm] = {"episodes": a["episodes"], "success": a["success"], "failures": a["failures"], "constraints": a["constraints"], "init_fail": a["init_fail"], "final_fail": a["final_fail"], "phys": a["phys"], "total": a["total"], "steps": steps, "decision_mean_s_per_step_weighted": (a["decision_weighted"] / steps if steps else None), "horizons": {str(k): int(v) for k, v in sorted(a["horizons"].items(), key=lambda kv: int(kv[0]))}, "short_horizon_steps": short, "short_horizon_fraction": (short / steps if steps else None)}
    return parsed


def load_records(shards: List[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing = Counter()
    for shard in shards:
        for name in sorted(glob.glob(str(ROOT / ("shard%02d" % shard) / "episodes" / "*" / "summary.json"))):
            p = Path(name)
            m = RE_EPISODE.match(p.parent.name)
            if not m:
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rec = {
                "path": str(p),
                "shard": shard,
                "case": int(m.group("case")),
                "arm": m.group("arm"),
                "success": obj.get("success"),
                "episode_failure": obj.get("episode_failure"),
                "physical": obj.get("physical_constraint_cost", obj.get("performance_cost")),
                "total": obj.get("total_cost"),
                "decision": obj.get("decision_mean_s_per_step"),
                "solver_mean": obj.get("solver_mean_s_per_attempt"),
                "selection_mean": (obj.get("selection_timing_s") or {}).get("mean") if isinstance(obj.get("selection_timing_s"), dict) else None,
                "steps": obj.get("steps"),
                "horizons": obj.get("horizon_counts") or obj.get("raw_horizon_counts_before_clamp") or {},
                "solver_failure_steps": obj.get("solver_failure_steps"),
                "initial_failed_steps": obj.get("initial_failed_steps"),
            }
            for key in ["success", "physical", "total", "decision", "steps"]:
                if rec.get(key) is None:
                    missing[key] += 1
            records.append(rec)
    return records, {"episode_summary_records": len(records), "missing_field_counts": dict(missing)}


def simple_stats(vals: List[float]) -> Dict[str, Any]:
    return {"n": len(vals), "mean": mean_or_none(vals), "median": median_or_none(vals), "min": min(vals) if vals else None, "max": max(vals) if vals else None, "lt_0_count": sum(1 for v in vals if v < -1e-6), "near_le_0p05_count": sum(1 for v in vals if v <= 0.05)}


def opportunity(records: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    regex = RE_MATCHED if kind == "matched" else RE_INDEPENDENT
    fixed: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in records:
        m = regex.match(r["arm"])
        if m and r.get("physical") is not None:
            fixed[(int(r["case"]), int(m.group("seed")), int(m.group("H")))] = r
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for (case, seed, H), r in fixed.items():
        groups[(case, seed)][H] = r
    best_phys, best_total = Counter(), Counter()
    time_ratios, phys_delta, total_delta = defaultdict(list), defaultdict(list), defaultdict(list)
    counts = Counter()
    examples_short, examples_anti, examples_long = [], [], []
    for (case, seed), hs in sorted(groups.items()):
        counts["groups"] += 1
        base = hs.get(25)
        if not base or base.get("physical") is None:
            continue
        counts["with_H25"] += 1
        bphys = float(base["physical"])
        btotal = float(base["total"]) if base.get("total") is not None else None
        btime = float(base["decision"]) if base.get("decision") is not None else None
        successful = [r for r in hs.values() if r.get("success") is not False and r.get("physical") is not None]
        if successful:
            bp = min(successful, key=lambda r: (float(r["physical"]), int(regex.match(r["arm"]).group("H"))))
            best_phys[int(regex.match(bp["arm"]).group("H"))] += 1
            totalable = [r for r in successful if r.get("total") is not None]
            if totalable:
                bt = min(totalable, key=lambda r: (float(r["total"]), int(regex.match(r["arm"]).group("H"))))
                best_total[int(regex.match(bt["arm"]).group("H"))] += 1
        any_short_better = any_short_near = any_short_total = any_short_fast = any_short_near_fast = False
        any_long_better = any_long_total = False
        best_short_row, best_long_row = None, None
        for H, r in sorted(hs.items()):
            if H == 25 or r.get("success") is False or r.get("physical") is None:
                continue
            dphys = float(r["physical"]) - bphys
            dtot = (float(r["total"]) - btotal) if (r.get("total") is not None and btotal is not None) else None
            tr = safe_ratio(r.get("decision"), btime)
            phys_delta[H].append(dphys)
            if dtot is not None:
                total_delta[H].append(dtot)
            if tr is not None:
                time_ratios[H].append(tr)
            row = {"case": case, "seed": seed, "H": H, "physical_delta_vs_H25": dphys, "total_delta_vs_H25": dtot, "decision_ratio_vs_H25": tr}
            if H < 25:
                any_short_better = any_short_better or dphys < -1e-6
                any_short_near = any_short_near or dphys <= 0.05
                any_short_total = any_short_total or (dtot is not None and dtot < -1e-6)
                any_short_fast = any_short_fast or (tr is not None and tr < 0.98)
                any_short_near_fast = any_short_near_fast or (dphys <= 0.05 and tr is not None and tr < 0.98)
                if H == POLICY_SHORT_H.get(seed):
                    counts["policy_short_physical_better"] += int(dphys < -1e-6)
                    counts["policy_short_total_better"] += int(dtot is not None and dtot < -1e-6)
                    counts["policy_short_near"] += int(dphys <= 0.05)
                    counts["policy_short_near_fast"] += int(dphys <= 0.05 and tr is not None and tr < 0.98)
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
        if any_short_near_fast and len(examples_short) < 12:
            examples_short.append(best_short_row)
        if not any_short_near and len(examples_anti) < 12:
            examples_anti.append({"case": case, "seed": seed, "best_short_by_physical": best_short_row})
        if any_long_better and len(examples_long) < 12:
            examples_long.append(best_long_row)
    return {
        "fixed_kind": kind,
        "fixed_episode_records": len(fixed),
        "case_seed_groups": counts["groups"],
        "case_seed_groups_with_H25": counts["with_H25"],
        "short_H_any_physical_better_count": counts["short_any_physical_better"],
        "short_H_any_physical_near_equal_le_0p05_count": counts["short_any_near"],
        "short_H_any_total_better_count": counts["short_any_total_better"],
        "short_H_any_measured_faster_by_2pct_count": counts["short_any_fast"],
        "short_H_any_near_equal_and_faster_count": counts["short_any_near_fast"],
        "policy_nominated_short_H_physical_better_episode_count": counts["policy_short_physical_better"],
        "policy_nominated_short_H_total_better_episode_count": counts["policy_short_total_better"],
        "policy_nominated_short_H_near_equal_episode_count": counts["policy_short_near"],
        "policy_nominated_short_H_near_equal_and_faster_episode_count": counts["policy_short_near_fast"],
        "long_H_any_physical_better_count": counts["long_any_physical_better"],
        "long_H_any_total_better_count": counts["long_any_total_better"],
        "H25_physical_best_count": best_phys.get(25, 0),
        "H25_total_best_count": best_total.get(25, 0),
        "best_physical_horizon_counts": {str(k): int(v) for k, v in sorted(best_phys.items())},
        "best_total_horizon_counts": {str(k): int(v) for k, v in sorted(best_total.items())},
        "decision_ratios_vs_H25_by_H": {str(k): {"n": len(v), "mean": mean_or_none(v), "median": median_or_none(v), "lt_0p98_count": sum(1 for x in v if x < 0.98), "gt_1p02_count": sum(1 for x in v if x > 1.02)} for k, v in sorted(time_ratios.items())},
        "physical_delta_vs_H25_by_H": {str(k): simple_stats(v) for k, v in sorted(phys_delta.items())},
        "total_delta_vs_H25_by_H": {str(k): simple_stats(v) for k, v in sorted(total_delta.items())},
        "opportunity_examples_short_near_equal_and_faster": examples_short,
        "anti_examples_no_near_equal_short": examples_anti,
        "long_horizon_physical_better_examples": examples_long,
    }


def alignment(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fixed: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in records:
        m = RE_MATCHED.match(r["arm"])
        if m and r.get("physical") is not None:
            fixed[(int(r["case"]), int(m.group("seed")), int(m.group("H")))] = r
    any_near, any_near_fast, pol_near, pol_near_fast = set(), set(), set(), set()
    for (case, seed, H), r in fixed.items():
        if H >= 25 or r.get("success") is False:
            continue
        base = fixed.get((case, seed, 25))
        if not base or base.get("physical") is None:
            continue
        dphys = float(r["physical"]) - float(base["physical"])
        tr = safe_ratio(r.get("decision"), base.get("decision"))
        if dphys <= 0.05:
            any_near.add((case, seed))
        if dphys <= 0.05 and tr is not None and tr < 0.98:
            any_near_fast.add((case, seed))
        if H == POLICY_SHORT_H.get(seed):
            if dphys <= 0.05:
                pol_near.add((case, seed))
            if dphys <= 0.05 and tr is not None and tr < 0.98:
                pol_near_fast.add((case, seed))
    fam_records: Dict[str, List[Dict[str, Any]]] = {"risk": [], "current": []}
    for r in records:
        fam, seed = None, None
        m = RE_RISK_ARM.match(r["arm"])
        if m:
            fam, seed = "risk", int(m.group("seed"))
        else:
            m = RE_CURRENT_ARM.match(r["arm"])
            if m:
                fam, seed = "current", int(m.group("seed"))
        if fam is None or seed is None:
            continue
        h = Counter({int(k): int(v) for k, v in (r.get("horizons") or {}).items() if str(k).lstrip("-").isdigit()})
        steps = sum(h.values()) or int(r.get("steps") or 0)
        short = sum(v for hh, v in h.items() if hh < 25)
        case = int(r["case"])
        fam_records[fam].append({"case": case, "seed": seed, "steps": steps, "short_steps": short, "short_fraction": short / steps if steps else None, "horizon_counts": {str(k): int(v) for k, v in sorted(h.items())}, "any_short_near_opportunity": (case, seed) in any_near, "any_short_near_fast_opportunity": (case, seed) in any_near_fast, "policy_short_near_opportunity": (case, seed) in pol_near, "policy_short_near_fast_opportunity": (case, seed) in pol_near_fast})

    def agg(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(lambda: {"episodes": 0, "steps": 0, "short_steps": 0, "episodes_with_short": 0, "any_short_near_opp_episodes": 0, "any_short_near_fast_opp_episodes": 0, "policy_short_near_opp_episodes": 0, "policy_short_near_fast_opp_episodes": 0, "short_episode_with_any_near_opp": 0, "short_episode_without_any_near_opp": 0, "short_episode_with_policy_near_opp": 0, "short_episode_without_policy_near_opp": 0})
        for r in recs:
            a = out[r["seed"]]
            a["episodes"] += 1; a["steps"] += int(r["steps"]); a["short_steps"] += int(r["short_steps"])
            has_short = bool(r["short_steps"])
            a["episodes_with_short"] += int(has_short)
            for key in ["any_short_near", "any_short_near_fast", "policy_short_near", "policy_short_near_fast"]:
                a[key + "_opp_episodes"] += int(bool(r[key + "_opportunity"]))
            if has_short:
                a["short_episode_with_any_near_opp"] += int(bool(r["any_short_near_opportunity"]))
                a["short_episode_without_any_near_opp"] += int(not bool(r["any_short_near_opportunity"]))
                a["short_episode_with_policy_near_opp"] += int(bool(r["policy_short_near_opportunity"]))
                a["short_episode_without_policy_near_opp"] += int(not bool(r["policy_short_near_opportunity"]))
        final = {}
        for seed, a in sorted(out.items()):
            d = dict(a); d["short_fraction"] = a["short_steps"] / a["steps"] if a["steps"] else None
            final[str(seed)] = d
        return final
    return {"opportunity_sets_size": {"any_short_near": len(any_near), "any_short_near_fast": len(any_near_fast), "policy_short_near": len(pol_near), "policy_short_near_fast": len(pol_near_fast)}, "risk_by_seed": agg(fam_records["risk"]), "current_by_seed": agg(fam_records["current"]), "risk_records": fam_records["risk"][:300], "current_records": fam_records["current"][:300]}


def interpret(parsed: Dict[str, Any], matched: Dict[str, Any], indep: Dict[str, Any], align: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], str]:
    verified, hypotheses, missing = [], [], []
    for seed_s, row in sorted(parsed["risk_vs_h25_by_seed"].items(), key=lambda kv: int(kv[0])):
        ratio_mean, phys, total = row.get("decision_ratio_mean_unweighted_shards"), row.get("physical_delta_sum"), row.get("total_delta_sum")
        if ratio_mean is not None and ratio_mean > 1.0:
            verified.append("Risk seed%s measured decision time is slower than matched fixed H25 in unweighted shard mean (%.4f)." % (seed_s, ratio_mean))
        if phys is not None and phys > 0.05:
            verified.append("Risk seed%s is physically worse than matched fixed H25 over completed shards (delta %.6g)." % (seed_s, phys))
        elif phys is not None and abs(phys) <= 0.05 and total is not None and total < -0.05:
            verified.append("Risk seed%s total-cost benefit over H25 is mostly horizon/work-penalty driven: physical delta %.6g, total delta %.6g." % (seed_s, phys, total))
    for seed in [0, 1, 2]:
        arm = "risk_reselected_v1_vehicle_s%d" % seed
        row = parsed["adaptive_arm_counts"].get(arm, {})
        frac = row.get("short_horizon_fraction")
        if frac is not None and frac < 0.20:
            verified.append("Risk seed%d remains sparse-switching on completed shards: short-horizon fraction %.3f, horizons %s." % (seed, frac, row.get("horizons")))
    if matched.get("short_H_any_near_equal_and_faster_count", 0) == 0:
        verified.append("Matched-terminal fixed-grid episodes show no completed case/seed group where any short H is both physical-near-equal (<=0.05) and at least 2%% faster than H25.")
    else:
        hypotheses.append("Some short-H near/faster opportunity exists in matched fixed-grid episodes, but policy alignment and continuation-cost checks are needed.")
    if matched.get("policy_nominated_short_H_near_equal_and_faster_episode_count", 0) == 0:
        verified.append("The nominated policy short horizons (s0/s1 H20, s2 H15) are never both physical-near-equal and >=2%% faster than H25 on completed matched-terminal fixed-grid groups.")
    if matched.get("long_H_any_physical_better_count", 0) > matched.get("short_H_any_physical_better_count", 0):
        verified.append("Longer fixed horizons (>25) more often improve physical cost than shorter horizons on completed matched-terminal groups, so the current scenarios/candidates may favor strong fixed-H tuning rather than safe shortening.")
    for seed_s, a in sorted(align.get("risk_by_seed", {}).items(), key=lambda kv: int(kv[0])):
        if a.get("policy_short_near_opp_episodes", 0) > a.get("episodes_with_short", 0):
            hypotheses.append("Risk seed%s has nominated short-H near-opportunity episodes that are not always executed with short H, consistent with restrictive gates or state-dependent conservatism." % seed_s)
    missing.append("This uses completed risk-reselection fresh devval shards00-05 only (24/64 cases), not sealed final test and not fresh confirmation validation.")
    missing.append("Fixed-H comparisons are full episodes from identical scenario resets, not one-step continuation rollouts from every adaptive state; transition-consequence diagnosis remains separate.")
    if not indep.get("fixed_episode_records"):
        missing.append("Independent-terminal opportunity records were unavailable; matched-terminal fixed grid is the main opportunity evidence.")
    all_slowish = all((parsed["risk_vs_h25_by_seed"].get(str(i), {}).get("decision_ratio_mean_unweighted_shards") or 0) >= 0.995 for i in [0, 1, 2])
    if any("physically worse" in s for s in verified) or all_slowish:
        decision = "Do not allocate another unchanged long risk-reselection shard by default. Preserve the partial campaign and prioritize a versioned training/selection or scenario-opportunity diagnostic that can create a stronger candidate or explain absent adaptive opportunity."
    else:
        decision = "Continue the frozen devval64 shard campaign after backup if precise model-selection estimates are still the objective; keep evidence scoped as development-validation only."
    return verified, hypotheses, missing, decision


def write_outputs(out: Path, shards: List[int], parsed: Dict[str, Any], recdiag: Dict[str, Any], matched: Dict[str, Any], indep: Dict[str, Any], align: Dict[str, Any], verified: List[str], hypotheses: List[str], missing: List[str], decision: str) -> None:
    raw = {"created_utc": now(), "diagnostic_type": "read_only_existing_fresh_devval_partial_opportunity_runtime", "method_classification": "IMPROVED finite direct gated-horizon risk re-selection; no new gradient training; not ORIGINAL SAC", "access_flags": {"new_simulations": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "sealed_test_bank_opened": False, "historical_validation64_bank_opened": False, "fresh_devval_existing_outputs_read": True}, "root": str(ROOT), "completed_shards": shards, "parsed_summaries": parsed, "episode_record_diagnostics": recdiag, "matched_terminal_opportunity": matched, "independent_terminal_opportunity": indep, "policy_alignment": align, "verified_findings": verified, "live_hypotheses": hypotheses, "missing_evidence_or_limitations": missing, "next_allocation_decision": decision, "hashes": {}}
    for p in [Path(__file__), ROOT / "gate_completed.json", ROOT / "bank" / "completed.json", ROOT / "bank" / "vehicle_gated_horizon_risk_reselection_v1_devval64_bank.json"]:
        if p.exists(): raw["hashes"][str(p)] = sha256_path(p)
    for shard in shards:
        for name in ["summary.md", "completed.json", "raw.json"]:
            p = ROOT / ("shard%02d" % shard) / name
            if p.exists(): raw["hashes"][str(p)] = sha256_path(p)
    raw_path = out / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Vehicle risk-reselection v1 partial opportunity/runtime diagnostic", "", "Created UTC: `%s`." % now(), "", "Read-only diagnostic over already generated fresh development-validation shards. No simulations, no training, no sealed-test access, and no historical validation64 bank access.", "", "## Evidence inventory", "", "- Shards analyzed: `%s` (%d / 16 risk-reselection devval shards; %d case slots, %d unique cases)." % (shards, len(shards), len(parsed.get("cases", [])), len(set(parsed.get("cases", [])))), "- Shard cases: `%s`." % parsed.get("cases"), "- Shard-reported episodes/control steps: `%s` / `%s`." % (parsed.get("episodes"), parsed.get("control_steps")), "- Episode summary records parsed: `%s`; missing fields: `%s`." % (recdiag.get("episode_summary_records"), recdiag.get("missing_field_counts")), "", "## Risk-reselected policy vs matched fixed H25 (partial)", ""]
    for seed_s, row in sorted(parsed["risk_vs_h25_by_seed"].items(), key=lambda kv: int(kv[0])):
        arm = "risk_reselected_v1_vehicle_s%s" % seed_s
        a = parsed["adaptive_arm_counts"].get(arm, {})
        lines.append("- seed %s: physical_delta_sum=%s, total_delta_sum=%s, decision_ratio_mean_unweighted=%s, short_fraction=%s, horizons=%s, success_strings=%s" % (seed_s, fnum(row.get("physical_delta_sum")), fnum(row.get("total_delta_sum")), fnum(row.get("decision_ratio_mean_unweighted_shards")), fnum(a.get("short_horizon_fraction"), 4), a.get("horizons"), row.get("success_strings")))
    lines += ["", "## Fixed-horizon opportunity from identical scenario starts", ""]
    for title, opp in [("matched-terminal fixed grid", matched), ("independent-terminal seed0 fixed grid", indep)]:
        lines += ["### " + title, "", "- Case/seed groups with H25: `%s`; fixed episode records: `%s`." % (opp.get("case_seed_groups_with_H25"), opp.get("fixed_episode_records")), "- Best physical horizon counts: `%s`; H25 best count: `%s`." % (opp.get("best_physical_horizon_counts"), opp.get("H25_physical_best_count")), "- Best total horizon counts: `%s`; H25 total-best count: `%s`." % (opp.get("best_total_horizon_counts"), opp.get("H25_total_best_count")), "- Short-H opportunity counts: physical_better=%s, near_equal<=0.05=%s, total_better=%s, measured_faster>=2%%=%s, near_equal_and_faster=%s." % (opp.get("short_H_any_physical_better_count"), opp.get("short_H_any_physical_near_equal_le_0p05_count"), opp.get("short_H_any_total_better_count"), opp.get("short_H_any_measured_faster_by_2pct_count"), opp.get("short_H_any_near_equal_and_faster_count"))]
        if title.startswith("matched"):
            lines.append("- Nominated policy short-H counts (s0/s1 H20, s2 H15): physical_better=%s, near_equal=%s, near_equal_and_faster=%s." % (opp.get("policy_nominated_short_H_physical_better_episode_count"), opp.get("policy_nominated_short_H_near_equal_episode_count"), opp.get("policy_nominated_short_H_near_equal_and_faster_episode_count")))
        lines += ["- Long-H opportunity counts: physical_better=%s, total_better=%s." % (opp.get("long_H_any_physical_better_count"), opp.get("long_H_any_total_better_count")), "- Decision ratios vs H25 by H: `%s`." % opp.get("decision_ratios_vs_H25_by_H"), ""]
    lines += ["## Policy/opportunity alignment", "", "- Opportunity-set sizes: `%s`." % align.get("opportunity_sets_size")]
    for seed_s, row in sorted(align.get("risk_by_seed", {}).items(), key=lambda kv: int(kv[0])):
        lines.append("- risk seed %s: `%s`" % (seed_s, row))
    lines += ["", "## Verified findings", ""] + ["- " + s for s in verified] + ["", "## Live hypotheses", ""] + ["- " + s for s in hypotheses] + ["", "## Missing evidence / limitations", ""] + ["- " + s for s in missing] + ["", "## Decision for next allocation", "", decision, ""]
    md = "\n".join(lines)
    summary_path = out / "summary.md"
    summary_path.write_text(md, encoding="utf-8")

    completed_path = out / "completed.json"
    completed = {"created_utc": now(), "passed": True, "summary_path": str(summary_path), "raw_path": str(raw_path), "completed_shards": shards, "episodes_read_from_summaries": parsed.get("episodes"), "episode_summary_records": recdiag.get("episode_summary_records"), "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_simulations": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "next_allocation_decision": decision}
    completed_path.write_text(json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8")

    marker = "<!-- vehicle-risk-reselection-partial-opportunity-diagnostic-20260928 -->"
    block = "%s\n## 2026-09-28 risk-reselection partial opportunity/runtime diagnostic\n\nUTC: %s. Read-only diagnostic over fresh devval shards %s; no simulations, no training, no sealed-test access, and no historical validation64 bank access. Parsed %s per-episode summaries and shard-reported %s episodes / %s control steps.\n\nKey result: %s\n\nArtifacts: `%s`, `%s`, `%s`. New diagnostic artifacts and doc updates require external backup before further simulations.\n" % (marker, now(), shards, recdiag.get("episode_summary_records"), parsed.get("episodes"), parsed.get("control_steps"), decision, summary_path, raw_path, completed_path)
    for doc in [Path("STATUS.md"), Path("RESEARCH_LOG.md"), Path("DECISIONS.md"), Path("RESULTS_AUDIT.md")]:
        append_once(doc, marker, block)

    state_path = STATE_DIR / "vehicle_risk_reselection_partial_opportunity_diagnostic_20260928T0845Z.md"
    state_lines = ["# Vehicle risk-reselection partial opportunity/runtime diagnostic state", "", "UTC: %s" % now(), "", "- Completed read-only diagnostic over shards: %s" % shards, "- No new simulations/training/gradient steps; sealed test not opened; historical validation64 bank not opened.", "- Summary: `%s`" % summary_path, "- Raw: `%s`" % raw_path, "- Completed marker: `%s`" % completed_path, "- Decision: %s" % decision, "", "Verified findings:"] + ["- " + s for s in verified] + ["", "Missing evidence / limitations:"] + ["- " + s for s in missing]
    state_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")

    backup_path = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_RISK_RESELECTION_PARTIAL_OPPORTUNITY_DIAGNOSTIC_20260928T0845Z.json"
    backup = {"created_utc": now(), "reason": "Backup required after read-only partial opportunity/runtime diagnostic before any further simulations.", "backup_required_before_more_simulations": True, "sealed_test_accessed": False, "historical_validation64_bank_opened": False, "new_simulations": 0, "artifacts_requiring_backup": [str(summary_path), str(raw_path), str(completed_path), str(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "EXPERIMENT_REGISTRY.csv", "experiments/bohn2021_aws/vehicle_risk_reselection_devval_partial_opportunity_diagnostic.py"], "summary_sha256": sha256_path(summary_path), "raw_sha256": sha256_path(raw_path), "completed_sha256": sha256_path(completed_path), "state_sha256": sha256_path(state_path)}
    backup_path.write_text(json.dumps(backup, indent=2, sort_keys=True), encoding="utf-8")
    completed["backup_request"] = str(backup_path)
    completed["backup_request_sha256"] = sha256_path(backup_path)
    completed_path.write_text(json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8")
    print(md)
    print("Artifacts:")
    for p in [summary_path, raw_path, completed_path, state_path, backup_path]:
        print("- %s" % p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--max-shard", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True); BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shards = []
    for p in sorted(ROOT.glob("shard*/completed.json")):
        m = re.search(r"shard(\d+)", str(p))
        if m and int(m.group(1)) <= args.max_shard:
            shards.append(int(m.group(1)))
    shards = sorted(set(shards))
    missing = [s for s in range(args.max_shard + 1) if s not in shards]
    if missing:
        raise RuntimeError("Missing completed shards needed for partial diagnostic: %s" % missing)
    parsed = parse_summaries(shards)
    records, recdiag = load_records(shards)
    matched = opportunity(records, "matched")
    indep = opportunity(records, "independent")
    align = alignment(records)
    verified, hypotheses, missing_ev, decision = interpret(parsed, matched, indep, align)
    write_outputs(out, shards, parsed, recdiag, matched, indep, align, verified, hypotheses, missing_ev, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
