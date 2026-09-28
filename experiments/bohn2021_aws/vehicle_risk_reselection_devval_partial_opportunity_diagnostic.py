#!/usr/bin/env python3
"""Partial devval opportunity/runtime diagnostic for vehicle gated-horizon risk reselection.

This is a metadata/read-only diagnostic over already-generated fresh development-validation
outputs. It runs no simulation, no training, and opens no sealed-test or historical
validation64 bank. It is intended to answer, before allocating another long shard by
default, whether the completed fresh devval shards already indicate: (1) scenario/horizon
opportunity, (2) policy/selection bottlenecks, and (3) measured runtime overhead issues.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path("research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1")
DEFAULT_OUT = Path("research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z")
STATE_DIR = Path("research_artifacts/aws_state")
BACKUP_DIR = Path("research_artifacts/aws_backup_proofs")

_SUMMARY_RISK_RE = re.compile(
    r"^- seed (?P<seed>\d+) risk-vs-H25: .*?success risk/fixed=(?P<succ>[^,]+), "
    r"physical_delta_sum=(?P<phys>[-+0-9.eE]+), total_delta_sum=(?P<total>[-+0-9.eE]+), "
    r"decision_ratio_mean=(?P<ratio>[-+0-9.eE]+)"
)
_SUMMARY_CASES_RE = re.compile(r"^- Shard cases: \[(?P<cases>[^\]]*)\]")
_SUMMARY_BUDGET_RE = re.compile(r"^- Shard episodes/control steps: `(?P<eps>\d+)` / `(?P<steps>\d+)`")
_ADAPTIVE_LINE_RE = re.compile(
    r"^- `(?P<arm>[^`]+)`: episodes=(?P<episodes>\d+), success=(?P<success>\d+), failures=(?P<failures>\d+), "
    r"constraints=(?P<constraints>\d+), init_fail=(?P<init>\d+), final_fail=(?P<final>\d+), phys=(?P<phys>[-+0-9.eE]+), "
    r"total=(?P<total>[-+0-9.eE]+), decision_mean_s_per_step=(?P<decision>[-+0-9.eE]+), horizons=(?P<horizons>\{.*?\}),"
)
_EPISODE_DIR_RE = re.compile(r"exec(?P<exec>\d+)_case(?P<case>\d+)_(?P<arm>.+)$")
_MATCHED_FIXED_RE = re.compile(r"matched_terminal_fixed_H(?P<H>\d+)_vehicle_s(?P<seed>\d+)$")
_RISK_RE = re.compile(r"risk_reselected_v1_vehicle_s(?P<seed>\d+)$")
_CURRENT_RE = re.compile(r"current_gated_vehicle_s(?P<seed>\d+)$")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_path(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def flatten_scalars(obj: Any, prefix: str = "") -> Iterable[Tuple[str, str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_scalars(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from flatten_scalars(v, p)
    else:
        leaf = prefix.split(".")[-1]
        yield prefix, leaf, obj


def find_number(obj: Any, candidates: List[str]) -> Optional[float]:
    flat = list(flatten_scalars(obj))
    candidate_norms = [norm_key(c) for c in candidates]
    # Exact normalized leaf match in priority order.
    for cand in candidate_norms:
        for path, leaf, value in flat:
            if norm_key(leaf) == cand and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    # Suffix/path match in priority order.
    for cand in candidate_norms:
        for path, leaf, value in flat:
            if cand in norm_key(path) and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def find_bool(obj: Any, candidates: List[str]) -> Optional[bool]:
    flat = list(flatten_scalars(obj))
    candidate_norms = [norm_key(c) for c in candidates]
    for cand in candidate_norms:
        for path, leaf, value in flat:
            if norm_key(leaf) == cand and isinstance(value, bool):
                return bool(value)
    for cand in candidate_norms:
        for path, leaf, value in flat:
            if cand in norm_key(path) and isinstance(value, bool):
                return bool(value)
    return None


def find_int(obj: Any, candidates: List[str]) -> Optional[int]:
    value = find_number(obj, candidates)
    if value is None or not math.isfinite(value):
        return None
    return int(round(value))


def find_horizon_counts(obj: Any) -> Dict[str, int]:
    best: Dict[str, int] = {}

    def rec(x: Any, path: str = "") -> None:
        nonlocal best
        if isinstance(x, dict):
            # Prefer explicitly named horizon-count dictionaries.
            if "horizon" in norm_key(path) and all(str(k).lstrip("-").isdigit() for k in x.keys()):
                cand = {}
                ok = True
                for k, v in x.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        cand[str(k)] = int(round(v))
                    else:
                        ok = False
                        break
                if ok and sum(cand.values()) >= sum(best.values()):
                    best = cand
            for k, v in x.items():
                rec(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            # Some traces store the horizon per step as a list.
            if "horizon" in norm_key(path):
                vals = []
                for v in x:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(str(int(round(v))))
                if vals and len(vals) >= sum(best.values()):
                    best = dict(Counter(vals))
            for i, v in enumerate(x[:3]):
                rec(v, f"{path}[{i}]")

    rec(obj)
    return best


def read_solver_call_mean(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    vals: List[float] = []
    candidates = [
        "decision_time_s", "decision_wall_time_s", "wall_time_s", "elapsed_wall_time_s",
        "elapsed_s", "runtime_s", "duration_s", "solve_time_s", "solver_time_s",
    ]
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                value = find_number(obj, candidates)
                if value is not None and value >= 0 and math.isfinite(value):
                    vals.append(value)
    except Exception:
        return None
    return mean(vals) if vals else None


def parse_md_horizon_counts(s: str) -> Dict[str, int]:
    # Markdown prints a Python dict with quoted numeric keys; avoid eval.
    out: Dict[str, int] = {}
    for k, v in re.findall(r"'?(\d+)'?\s*:\s*(\d+)", s):
        out[str(int(k))] = int(v)
    return out


def parse_shard_summaries(shards: List[int]) -> Dict[str, Any]:
    parsed = {
        "shards": {},
        "risk_vs_h25_by_seed": defaultdict(lambda: {"physical_delta_sum": 0.0, "total_delta_sum": 0.0, "decision_ratios": [], "success_strings": []}),
        "adaptive_arm_counts": defaultdict(lambda: {"episodes": 0, "success": 0, "failures": 0, "constraints": 0, "init_fail": 0, "final_fail": 0, "phys": 0.0, "total": 0.0, "steps": 0, "decision_weighted_sum": 0.0, "horizons": Counter()}),
        "episodes": 0,
        "control_steps": 0,
        "cases": [],
    }
    for shard in shards:
        path = ROOT / f"shard{shard:02d}" / "summary.md"
        txt = path.read_text(encoding="utf-8")
        shard_info: Dict[str, Any] = {"summary_path": str(path), "cases": [], "episodes": None, "control_steps": None, "risk_vs_h25": {}}
        for line in txt.splitlines():
            m = _SUMMARY_CASES_RE.match(line)
            if m:
                cases = [int(x.strip()) for x in m.group("cases").split(",") if x.strip()]
                shard_info["cases"] = cases
                parsed["cases"].extend(cases)
            m = _SUMMARY_BUDGET_RE.match(line)
            if m:
                eps = int(m.group("eps")); steps = int(m.group("steps"))
                shard_info["episodes"] = eps; shard_info["control_steps"] = steps
                parsed["episodes"] += eps; parsed["control_steps"] += steps
            m = _SUMMARY_RISK_RE.match(line)
            if m:
                seed = int(m.group("seed"))
                phys = float(m.group("phys")); total = float(m.group("total")); ratio = float(m.group("ratio"))
                shard_info["risk_vs_h25"][str(seed)] = {"physical_delta_sum": phys, "total_delta_sum": total, "decision_ratio_mean": ratio, "success": m.group("succ")}
                acc = parsed["risk_vs_h25_by_seed"][seed]
                acc["physical_delta_sum"] += phys
                acc["total_delta_sum"] += total
                acc["decision_ratios"].append(ratio)
                acc["success_strings"].append(m.group("succ"))
            m = _ADAPTIVE_LINE_RE.match(line)
            if m:
                arm = m.group("arm")
                horizons = parse_md_horizon_counts(m.group("horizons"))
                steps = sum(horizons.values())
                acc = parsed["adaptive_arm_counts"][arm]
                acc["episodes"] += int(m.group("episodes"))
                acc["success"] += int(m.group("success"))
                acc["failures"] += int(m.group("failures"))
                acc["constraints"] += int(m.group("constraints"))
                acc["init_fail"] += int(m.group("init"))
                acc["final_fail"] += int(m.group("final"))
                acc["phys"] += float(m.group("phys"))
                acc["total"] += float(m.group("total"))
                acc["steps"] += steps
                acc["decision_weighted_sum"] += float(m.group("decision")) * steps
                acc["horizons"].update(horizons)
        parsed["shards"][str(shard)] = shard_info
    # Materialize defaultdicts/counters into normal JSONable dicts.
    rvh = {}
    for seed, acc in parsed["risk_vs_h25_by_seed"].items():
        ratios = acc["decision_ratios"]
        rvh[str(seed)] = {
            "physical_delta_sum": acc["physical_delta_sum"],
            "total_delta_sum": acc["total_delta_sum"],
            "decision_ratio_mean_unweighted_shards": mean(ratios) if ratios else None,
            "decision_ratio_median_unweighted_shards": median(ratios) if ratios else None,
            "success_strings": acc["success_strings"],
        }
    arms = {}
    for arm, acc in parsed["adaptive_arm_counts"].items():
        steps = acc["steps"]
        arms[arm] = {
            "episodes": acc["episodes"],
            "success": acc["success"],
            "failures": acc["failures"],
            "constraints": acc["constraints"],
            "init_fail": acc["init_fail"],
            "final_fail": acc["final_fail"],
            "phys": acc["phys"],
            "total": acc["total"],
            "steps": steps,
            "decision_mean_s_per_step_weighted": (acc["decision_weighted_sum"] / steps) if steps else None,
            "horizons": dict(sorted(acc["horizons"].items(), key=lambda kv: int(kv[0]))),
            "short_horizon_steps": sum(v for k, v in acc["horizons"].items() if int(k) < 25),
            "short_horizon_fraction": (sum(v for k, v in acc["horizons"].items() if int(k) < 25) / steps) if steps else None,
        }
    parsed["risk_vs_h25_by_seed"] = rvh
    parsed["adaptive_arm_counts"] = arms
    parsed["cases"] = parsed["cases"]
    return parsed


def episode_record_from_dir(summary_path: Path) -> Optional[Dict[str, Any]]:
    m = _EPISODE_DIR_RE.match(summary_path.parent.name)
    if not m:
        return None
    try:
        obj = load_json(summary_path)
    except Exception:
        return None
    arm = m.group("arm")
    case = int(m.group("case"))
    shard_m = re.search(r"shard(\d+)", str(summary_path))
    shard = int(shard_m.group(1)) if shard_m else None
    decision = find_number(obj, [
        "decision_mean_s_per_step", "mean_decision_time_s", "decision_time_mean_s",
        "mean_controller_wall_time_s", "mean_solver_wall_time_s", "solver_time_mean_s",
    ])
    if decision is None:
        decision = read_solver_call_mean(summary_path.parent / "solver_calls.jsonl")
    rec = {
        "path": str(summary_path),
        "shard": shard,
        "case": case,
        "arm": arm,
        "success": find_bool(obj, ["success", "episode_success", "succeeded"]),
        "constraint": find_bool(obj, ["constraint", "constraint_violation", "any_constraint_violation"]),
        "physical": find_number(obj, [
            "physical_constraint_cost", "physical_cost", "phys", "physical_total_cost",
            "tracking_cost_with_constraints", "closed_loop_physical_cost",
        ]),
        "total": find_number(obj, ["total_cost", "augmented_total_cost", "objective_total_cost", "total"]),
        "decision_mean_s_per_step": decision,
        "steps": find_int(obj, ["control_steps", "steps", "n_steps", "episode_steps"]),
        "horizon_counts": find_horizon_counts(obj),
    }
    return rec


def collect_episode_records(shards: List[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing = Counter()
    for shard in shards:
        for p in sorted((ROOT / f"shard{shard:02d}" / "episodes").glob("*/summary.json")):
            rec = episode_record_from_dir(p)
            if not rec:
                continue
            for field in ["success", "physical", "total", "decision_mean_s_per_step", "steps"]:
                if rec.get(field) is None:
                    missing[field] += 1
            records.append(rec)
    return records, {"missing_field_counts": dict(missing), "episode_summary_records": len(records)}


def fixed_horizon_opportunity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in records:
        m = _MATCHED_FIXED_RE.match(r["arm"])
        if not m:
            continue
        if r.get("physical") is None:
            continue
        key = (int(r["case"]), int(m.group("seed")), int(m.group("H")))
        matched[key] = r

    by_case_seed: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for (case, seed, H), r in matched.items():
        by_case_seed[(case, seed)][H] = r

    cases_analyzed = 0
    with_h25 = 0
    short_phys_better = 0
    short_phys_near_equal_005 = 0
    short_total_better = 0
    short_measured_faster_2pct = 0
    short_near_equal_and_faster = 0
    h25_phys_best = 0
    best_phys_h = Counter()
    best_total_h = Counter()
    time_ratios_by_h: Dict[int, List[float]] = defaultdict(list)
    opportunity_examples: List[Dict[str, Any]] = []
    anti_examples: List[Dict[str, Any]] = []

    for (case, seed), hs in sorted(by_case_seed.items()):
        cases_analyzed += 1
        b = hs.get(25)
        if not b:
            continue
        with_h25 += 1
        bphys = b.get("physical")
        btot = b.get("total")
        btime = b.get("decision_mean_s_per_step")
        if bphys is None:
            continue
        successful = [r for r in hs.values() if r.get("success") is not False and r.get("physical") is not None]
        if successful:
            bp = min(successful, key=lambda r: (float(r.get("physical")), int(_MATCHED_FIXED_RE.match(r["arm"]).group("H"))))
            best_phys_h[int(_MATCHED_FIXED_RE.match(bp["arm"]).group("H"))] += 1
            if _MATCHED_FIXED_RE.match(bp["arm"]).group("H") == "25":
                h25_phys_best += 1
            totalable = [r for r in successful if r.get("total") is not None]
            if totalable:
                bt = min(totalable, key=lambda r: (float(r.get("total")), int(_MATCHED_FIXED_RE.match(r["arm"]).group("H"))))
                best_total_h[int(_MATCHED_FIXED_RE.match(bt["arm"]).group("H"))] += 1
        any_better = False
        any_near = False
        any_total = False
        any_fast = False
        any_near_fast = False
        best_short_row = None
        for H, r in hs.items():
            if H >= 25 or r.get("success") is False or r.get("physical") is None:
                continue
            dphys = float(r["physical"]) - float(bphys)
            dtot = (float(r["total"]) - float(btot)) if (r.get("total") is not None and btot is not None) else None
            ratio = (float(r["decision_mean_s_per_step"]) / float(btime)) if (r.get("decision_mean_s_per_step") is not None and btime and btime > 0) else None
            if ratio is not None:
                time_ratios_by_h[H].append(ratio)
            better = dphys < -1e-6
            near = dphys <= 0.05
            total_better = (dtot is not None and dtot < -1e-6)
            faster = (ratio is not None and ratio < 0.98)
            near_fast = near and faster
            any_better = any_better or better
            any_near = any_near or near
            any_total = any_total or total_better
            any_fast = any_fast or faster
            any_near_fast = any_near_fast or near_fast
            row = {"case": case, "seed": seed, "H": H, "physical_delta_vs_H25": dphys, "total_delta_vs_H25": dtot, "decision_ratio_vs_H25": ratio}
            if best_short_row is None or (row["physical_delta_vs_H25"], row["H"]) < (best_short_row["physical_delta_vs_H25"], best_short_row["H"]):
                best_short_row = row
        short_phys_better += int(any_better)
        short_phys_near_equal_005 += int(any_near)
        short_total_better += int(any_total)
        short_measured_faster_2pct += int(any_fast)
        short_near_equal_and_faster += int(any_near_fast)
        if any_near_fast and len(opportunity_examples) < 12:
            opportunity_examples.append(best_short_row)
        if (not any_near) and len(anti_examples) < 12:
            anti_examples.append({"case": case, "seed": seed, "best_short_by_physical": best_short_row})

    return {
        "matched_fixed_episode_records": len(matched),
        "case_seed_groups": cases_analyzed,
        "case_seed_groups_with_H25": with_h25,
        "short_H_any_physical_better_count": short_phys_better,
        "short_H_any_physical_near_equal_le_0p05_count": short_phys_near_equal_005,
        "short_H_any_total_better_count": short_total_better,
        "short_H_any_measured_faster_by_2pct_count": short_measured_faster_2pct,
        "short_H_any_near_equal_and_faster_count": short_near_equal_and_faster,
        "H25_physical_best_count": h25_phys_best,
        "best_physical_horizon_counts": dict(sorted(best_phys_h.items())),
        "best_total_horizon_counts": dict(sorted(best_total_h.items())),
        "fixed_short_decision_ratios_vs_H25_by_H": {
            str(H): {
                "n": len(vals),
                "mean": mean(vals) if vals else None,
                "median": median(vals) if vals else None,
                "lt_0p98_count": sum(1 for v in vals if v < 0.98),
                "gt_1p02_count": sum(1 for v in vals if v > 1.02),
            }
            for H, vals in sorted(time_ratios_by_h.items())
        },
        "opportunity_examples_near_equal_and_faster": opportunity_examples,
        "anti_examples_no_near_equal_short": anti_examples,
    }


def risk_alignment(records: List[Dict[str, Any]], opportunity: Dict[str, Any]) -> Dict[str, Any]:
    # Rebuild short opportunity at case/seed level, then compare whether risk policy actually used short horizons.
    fixed: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in records:
        m = _MATCHED_FIXED_RE.match(r["arm"])
        if m and r.get("physical") is not None:
            fixed[(int(r["case"]), int(m.group("seed")), int(m.group("H")))] = r
    short_near_fast = set()
    short_near = set()
    for (case, seed, H), r in fixed.items():
        if H >= 25 or r.get("success") is False:
            continue
        b = fixed.get((case, seed, 25))
        if not b or b.get("physical") is None:
            continue
        dphys = float(r["physical"]) - float(b["physical"])
        ratio = None
        if r.get("decision_mean_s_per_step") is not None and b.get("decision_mean_s_per_step"):
            ratio = float(r["decision_mean_s_per_step"]) / float(b["decision_mean_s_per_step"])
        if dphys <= 0.05:
            short_near.add((case, seed))
        if dphys <= 0.05 and ratio is not None and ratio < 0.98:
            short_near_fast.add((case, seed))
    risk_records = []
    for r in records:
        m = _RISK_RE.match(r["arm"])
        if not m:
            continue
        seed = int(m.group("seed")); case = int(r["case"])
        hcounts = Counter({int(k): int(v) for k, v in (r.get("horizon_counts") or {}).items() if str(k).lstrip("-").isdigit()})
        steps = sum(hcounts.values()) or (r.get("steps") or 0)
        short_steps = sum(v for h, v in hcounts.items() if h < 25)
        risk_records.append({"case": case, "seed": seed, "steps": steps, "short_steps": short_steps, "short_fraction": (short_steps / steps) if steps else None, "near_short_opportunity": (case, seed) in short_near, "near_and_faster_short_opportunity": (case, seed) in short_near_fast})
    agg = defaultdict(lambda: {"episodes": 0, "steps": 0, "short_steps": 0, "near_opp_episodes": 0, "near_fast_opp_episodes": 0, "short_in_near_opp": 0, "short_without_near_opp": 0})
    for r in risk_records:
        a = agg[r["seed"]]
        a["episodes"] += 1
        a["steps"] += int(r["steps"] or 0)
        a["short_steps"] += int(r["short_steps"] or 0)
        a["near_opp_episodes"] += int(bool(r["near_short_opportunity"]))
        a["near_fast_opp_episodes"] += int(bool(r["near_and_faster_short_opportunity"]))
        if r["short_steps"]:
            if r["near_short_opportunity"]:
                a["short_in_near_opp"] += 1
            else:
                a["short_without_near_opp"] += 1
    out = {}
    for seed, a in sorted(agg.items()):
        out[str(seed)] = dict(a)
        out[str(seed)]["short_fraction"] = (a["short_steps"] / a["steps"]) if a["steps"] else None
    return {"by_seed": out, "records": risk_records[:200]}


def format_float(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}g}"


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    with path.open("a", encoding="utf-8") as f:
        if old and not old.endswith("\n"):
            f.write("\n")
        f.write("\n" + text.strip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-shard", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    completed = []
    for p in sorted(ROOT.glob("shard*/completed.json")):
        m = re.search(r"shard(\d+)", str(p))
        if m and int(m.group(1)) <= args.max_shard:
            completed.append(int(m.group(1)))
    completed = sorted(set(completed))
    expected = list(range(args.max_shard + 1))
    missing_shards = [s for s in expected if s not in completed]
    if missing_shards:
        raise RuntimeError(f"Missing completed shards needed for partial diagnostic: {missing_shards}")

    parsed = parse_shard_summaries(completed)
    records, record_diag = collect_episode_records(completed)
    opportunity = fixed_horizon_opportunity(records)
    alignment = risk_alignment(records, opportunity)

    # Interpretation flags, deliberately conservative.
    risk_by_seed = parsed["risk_vs_h25_by_seed"]
    adaptive = parsed["adaptive_arm_counts"]
    findings: List[str] = []
    verified_causes: List[str] = []
    hypotheses: List[str] = []
    missing_evidence: List[str] = []

    # Runtime overhead from risk-vs-H25 shard summaries.
    slower_risk_seeds = []
    for seed_s, row in sorted(risk_by_seed.items(), key=lambda kv: int(kv[0])):
        if (row.get("decision_ratio_mean_unweighted_shards") or 0) > 1.0:
            slower_risk_seeds.append(seed_s)
    if slower_risk_seeds:
        verified_causes.append("Measured risk-policy decision time is not reliably lower than fixed H25 on completed shards; unweighted shard-mean ratios exceed 1.0 for seeds " + ", ".join(slower_risk_seeds) + ".")
    else:
        findings.append("Risk-policy decision-time ratios are below fixed H25 in all seeds on completed shards, but this remains partial development evidence.")

    # Sparse adaptation.
    for seed in [0, 1, 2]:
        arm = f"risk_reselected_v1_vehicle_s{seed}"
        row = adaptive.get(arm, {})
        frac = row.get("short_horizon_fraction")
        if frac is not None and frac < 0.20:
            verified_causes.append(f"Risk seed{seed} remains sparse-switching on completed shards: short-horizon fraction {frac:.3f} with horizons {row.get('horizons')}.")

    # Physical/total mismatch.
    for seed_s, row in sorted(risk_by_seed Dashboard if False else risk_by_seed.items(), key=lambda kv: int(kv[0])):  # noqa: E999 placeholder patched below
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
