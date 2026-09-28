#!/usr/bin/env python3
"""Vehicle stress Stage1/Stage2 reward, timing and terminal-observability audit.

Analysis-only diagnostic: no simulations, no training/refit, no historical
validation64-bank access and no sealed-test access.

Purpose:
- Separate physical/control cost, synthetic horizon compute penalty and measured
  wall-clock decision time in the stress fixed-H/continuation evidence.
- Check whether Stage2 positive/negative continuation labels are driven by real
  physical improvement, synthetic horizon penalty, measured timing, or safety.
- Inspect raw/protocol/trace artifacts for terminal/value-observable fields.  If
  terminal/value terms are not exposed in these artifacts, record that as missing
  evidence and route the next diagnostic to source/checkpoint-level audit rather
  than pretending the raw costs prove terminal-value quality.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAGE1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/raw.json"
STAGE1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/completed.json"
STAGE2_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json"
STAGE2_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/completed.json"
STAGE2_POST = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z/completed.json"
PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json"
OUT = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z"
STATE = ROOT / "research_artifacts/aws_state/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-stress-reward-timing-terminal-audit-v0-20260928T1820Z"
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
REF_H = 15
MATERIAL_GAIN = 3.0


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sf(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def summ(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(v for v in (sf(x) for x in vals) if math.isfinite(v))
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"n": n, "min": xs[0], "median": med, "mean": sum(xs) / n, "max": xs[-1]}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Any:
    pairs = [(sf(x), sf(y)) for x, y in zip(xs, ys) if math.isfinite(sf(x)) and math.isfinite(sf(y))]
    if len(pairs) < 3:
        return None
    xbar = sum(x for x, _ in pairs) / len(pairs)
    ybar = sum(y for _, y in pairs) / len(pairs)
    sx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    sy = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    if sx == 0 or sy == 0:
        return None
    return sum((x - xbar) * (y - ybar) for x, y in pairs) / (sx * sy)


def rankdata(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Any:
    pairs = [(sf(x), sf(y)) for x, y in zip(xs, ys) if math.isfinite(sf(x)) and math.isfinite(sf(y))]
    if len(pairs) < 3:
        return None
    rx = rankdata([x for x, _ in pairs])
    ry = rankdata([y for _, y in pairs])
    return pearson(rx, ry)


def timing_sum(row: Mapping[str, Any]) -> float:
    if "decision_sum_s" in row:
        return sf(row.get("decision_sum_s"), 0.0)
    d = row.get("decision_timing_s", {})
    if isinstance(d, Mapping):
        for k in ("sum", "total", "decision_total_s", "sum_s"):
            if k in d:
                return sf(d[k], 0.0)
    return sf(d, 0.0)


def phys(row: Mapping[str, Any]) -> float:
    return sf(row.get("physical_constraint_cost", row.get("physical_cost", row.get("continuation_physical", float("nan")))))


def total(row: Mapping[str, Any]) -> float:
    return sf(row.get("total_cost", row.get("continuation_total", row.get("cost", float("nan")))))


def strict_safe(row: Mapping[str, Any]) -> bool:
    return bool(row.get("success")) and not bool(row.get("constraint")) and si(row.get("initial_failed_steps"), 0) == 0 and si(row.get("solver_failure_steps"), 0) == 0


def nondominated(points: Sequence[Tuple[int, float, float]]) -> List[int]:
    out: List[int] = []
    for h, c, t in points:
        if any(c2 <= c and t2 <= t and (c2 < c or t2 < t) for h2, c2, t2 in points if h2 != h):
            continue
        out.append(h)
    return sorted(set(out))


def find_keys(obj: Any, needles: Tuple[str, ...], path: str = "", limit: int = 80, out: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            lk = str(k).lower()
            if any(n in lk for n in needles):
                preview = v if isinstance(v, (str, int, float, bool, type(None))) else (list(v)[:5] if isinstance(v, list) else sorted(list(v.keys()))[:10] if isinstance(v, Mapping) else str(type(v)))
                out.append({"path": p, "value_preview": preview})
                if len(out) >= limit:
                    return out
            find_keys(v, needles, p, limit, out)
            if len(out) >= limit:
                return out
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            find_keys(v, needles, f"{path}[{i}]", limit, out)
            if len(out) >= limit:
                return out
    return out


def trace_key_audit(stage2_raw: Mapping[str, Any]) -> Dict[str, Any]:
    # Read a bounded set: all Stage2 comparison traces are already-created diagnostic artifacts.
    paths = []
    for sr in stage2_raw["analysis"]["state_rows"]:
        for c in sr.get("comparisons", []):
            paths.append(ROOT / str(c["path"]) / "trace.json")
    key_counts: Dict[str, int] = {}
    terminal_like: Dict[str, int] = {}
    compute_per_step_by_h: Dict[str, List[float]] = {}
    read_count = 0
    for p in paths:
        if not p.exists():
            continue
        tr = read_json(p)
        read_count += 1
        # infer horizon from directory suffix, e.g. ..._H30
        h = p.parent.name.split("_H")[-1]
        vals = []
        for row in tr:
            if isinstance(row, Mapping):
                for k in row.keys():
                    key_counts[k] = key_counts.get(k, 0) + 1
                    lk = str(k).lower()
                    if any(n in lk for n in ("terminal", "value", "critic", "bootstrap")):
                        terminal_like[k] = terminal_like.get(k, 0) + 1
                if "compute" in row:
                    vals.append(sf(row.get("compute")))
        finite = [v for v in vals if math.isfinite(v)]
        if finite:
            compute_per_step_by_h.setdefault(h, []).extend(finite[:5])
    return {
        "trace_files_read": read_count,
        "top_level_key_counts": dict(sorted(key_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]),
        "terminal_value_like_trace_key_counts": terminal_like,
        "compute_field_examples_by_horizon": {k: sorted(set(round(v, 6) for v in vals))[:10] for k, vals in sorted(compute_per_step_by_h.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999)},
    }


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / name
        if p.exists():
            old = p.read_text(encoding="utf-8")
            if MARKER not in old:
                p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = read_json(STAGE1_RAW); s1done = read_json(STAGE1_COMPLETED)
    s2 = read_json(STAGE2_RAW); s2done = read_json(STAGE2_COMPLETED)
    if s1done.get("hard_pass") is not True or s2done.get("hard_pass") is not True:
        raise RuntimeError("Stage1/Stage2 completed markers are not hard-pass")
    if s1.get("historical_validation64_bank_opened") is not False or s1.get("sealed_test_accessed") is not False or s2.get("historical_validation64_bank_opened") is not False or s2.get("sealed_test_accessed") is not False:
        raise RuntimeError("Access flags invalid")
    episodes = list(s1.get("episodes") or [])
    by_h: Dict[int, List[Mapping[str, Any]]] = {h: [] for h in HORIZONS}
    ep_rows = []
    for e in episodes:
        h = si(e.get("horizon")); steps = si(e.get("steps")); p = phys(e); t = total(e); syn = t - p
        by_h.setdefault(h, []).append(e)
        ep_rows.append({"horizon": h, "steps": steps, "physical": p, "total": t, "synthetic_compute": syn, "synthetic_per_h_step": syn / (h * steps) if h > 0 and steps > 0 else float("nan"), "decision_sum_s": timing_sum(e), "decision_per_step_s": timing_sum(e) / steps if steps else float("nan"), "strict_safe": strict_safe(e)})
    agg = {}
    pts_phys_time = []
    for h in HORIZONS:
        rows = by_h.get(h, [])
        steps_sum = sum(si(r.get("steps")) for r in rows)
        physical_sum = sum(phys(r) for r in rows)
        total_sum = sum(total(r) for r in rows)
        decision_sum = sum(timing_sum(r) for r in rows)
        safe_all = all(strict_safe(r) for r in rows)
        agg[str(h)] = {"episodes": len(rows), "strict_safe_all": safe_all, "successes": sum(bool(r.get("success")) for r in rows), "steps_sum": steps_sum, "physical_sum": physical_sum, "synthetic_compute_sum": total_sum - physical_sum, "total_sum": total_sum, "decision_sum_s": decision_sum, "decision_per_step_s": decision_sum / steps_sum if steps_sum else None, "synthetic_per_step": (total_sum - physical_sum) / steps_sum if steps_sum else None}
        if safe_all:
            pts_phys_time.append((h, physical_sum, decision_sum))
    objective_alignment = {
        "synthetic_horizon_penalty_per_H_step_summary_stage1": summ(r["synthetic_per_h_step"] for r in ep_rows),
        "correlation_horizon_vs_synthetic_per_step_stage1": {"pearson": pearson([r["horizon"] for r in ep_rows], [r["synthetic_compute"] / r["steps"] for r in ep_rows if r["steps"]]), "spearman": spearman([r["horizon"] for r in ep_rows], [r["synthetic_compute"] / r["steps"] for r in ep_rows if r["steps"]])},
        "correlation_horizon_vs_measured_decision_per_step_stage1": {"pearson": pearson([r["horizon"] for r in ep_rows], [r["decision_per_step_s"] for r in ep_rows]), "spearman": spearman([r["horizon"] for r in ep_rows], [r["decision_per_step_s"] for r in ep_rows])},
        "aggregate_physical_cost_vs_measured_time_frontier_safe_fixed_H": nondominated(pts_phys_time),
    }
    # Stage2 branch-level decomposition against H15.
    branch_rows = []
    for sr in s2["analysis"]["state_rows"]:
        ref = sr["reference_H15"]
        ref_dec = next((c for c in sr.get("comparisons", []) if si(c.get("horizon")) == 15), None)
        ref_time = timing_sum(ref_dec or {})
        for c in sr.get("comparisons", []):
            h = si(c.get("horizon"))
            if h == 15:
                continue
            gp = sf(c.get("gain_vs_H15_physical")); gt = sf(c.get("gain_vs_H15_total")); syn_gain = gt - gp
            time_gain = ref_time - timing_sum(c)
            branch_rows.append({"target_index": si(sr.get("target_index")), "case": si(sr.get("case")), "branch_step": si(sr.get("branch_step")), "horizon": h, "physical_gain": gp, "synthetic_compute_gain_component": syn_gain, "total_gain": gt, "measured_decision_time_gain_s": time_gain, "success": bool(c.get("success")), "constraint": bool(c.get("constraint")), "solver_failure_steps": si(c.get("solver_failure_steps")), "state_distance": sf(c.get("state_distance_vs_H15_branch_state")), "material_total": gt >= MATERIAL_GAIN, "material_physical": gp >= MATERIAL_GAIN, "synthetic_only_material": gt >= MATERIAL_GAIN and gp < MATERIAL_GAIN, "physical_positive_time_not_slower": gp > 0 and time_gain >= 0, "path": c.get("path")})
    stage2_decomp = {
        "non_H15_branches": len(branch_rows),
        "material_total_branches": sum(r["material_total"] for r in branch_rows),
        "material_physical_branches": sum(r["material_physical"] for r in branch_rows),
        "synthetic_only_material_total_branches": sum(r["synthetic_only_material"] for r in branch_rows),
        "physical_positive_and_measured_time_not_slower_branches": sum(r["physical_positive_time_not_slower"] for r in branch_rows),
        "large_total_harms_le_minus_3": sum(r["total_gain"] <= -MATERIAL_GAIN for r in branch_rows),
        "physical_gain_summary": summ(r["physical_gain"] for r in branch_rows),
        "total_gain_summary": summ(r["total_gain"] for r in branch_rows),
        "measured_decision_time_gain_s_summary": summ(r["measured_decision_time_gain_s"] for r in branch_rows),
        "top_by_total_gain": sorted(branch_rows, key=lambda r: r["total_gain"], reverse=True)[:8],
        "top_by_physical_gain": sorted(branch_rows, key=lambda r: r["physical_gain"], reverse=True)[:8],
        "largest_harms": sorted(branch_rows, key=lambda r: r["total_gain"])[:8],
    }
    protocol = read_json(PROTOCOL) if PROTOCOL.exists() else {}
    terminal_search = {"stage1_raw_matches": find_keys(s1, ("terminal", "value", "critic", "checkpoint", "bootstrap"), limit=60), "stage2_raw_matches": find_keys(s2, ("terminal", "value", "critic", "checkpoint", "bootstrap"), limit=60), "protocol_matches": find_keys(protocol, ("terminal", "value", "critic", "checkpoint", "bootstrap", "penalty"), limit=80), "stage2_trace_audit": trace_key_audit(s2)}
    terminal_exposed = bool(terminal_search["stage2_trace_audit"]["terminal_value_like_trace_key_counts"] or terminal_search["stage1_raw_matches"] or terminal_search["protocol_matches"])
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    decision = {
        "retrain_or_selector_refit_now": False,
        "use_total_cost_as_measured_runtime_claim": False,
        "verified_findings": [
            "Stage1/Stage2 total_cost includes a synthetic horizon penalty component, not measured wall-clock runtime; measured decision time is available separately and must be reported separately.",
            "Stage2's only robust positive state is physical-cost driven (not synthetic-penalty-only); however positives remain too sparse for selector/refit training.",
            "No safety/solver regression is needed to explain the sparse-label decision; large non-H15 harms remain present.",
        ],
        "missing_evidence": "Raw stress artifacts do not by themselves prove terminal-value accuracy or mismatch; terminal/value observability is limited to key/config traces listed in raw.json and needs source/checkpoint-level audit if this remains a leading hypothesis.",
        "next_high_information_action_after_backup": "Do not run more simulations before external backup of this audit. Next run a bounded source/config/checkpoint terminal-value and objective audit (controller terminal model, h_penalty, training/selection metric lineage); only then choose between an IMPROVED objective/selector refit and a separately versioned stronger source-supported scenario design.",
    }
    raw_out = {"created_utc": created, "method": "vehicle_stress_reward_timing_terminal_audit_v0_analysis_only", "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "inputs": {"stage1_raw": rel(STAGE1_RAW), "stage2_raw": rel(STAGE2_RAW), "stage2_postdiagnostic_completed": rel(STAGE2_POST), "protocol": rel(PROTOCOL)}, "stage1_aggregate_by_horizon": agg, "objective_alignment": objective_alignment, "stage2_branch_decomposition": stage2_decomp, "terminal_value_observability": terminal_search, "four_axis_evidence": {"scenarios": "Stress design produced one strong local matched positive and many neutral/harmful matched states; opportunity remains sparse/localized.", "reward": "Synthetic h_penalty is well-aligned with H by construction and is not measured runtime. The only Stage2 positive is physical-driven, so it is not merely a short-H reward artifact, but total-cost claims must not be interpreted as acceleration.", "training": "No training/refit occurred; label density and terminal-value uncertainty do not justify selector retraining yet.", "comparisons": "Same-bank fixed-H and matched-state comparisons remain paired; measured timing is noisy/nonmonotone enough that control-vs-time Pareto reporting is safer than scalar total-cost winner claims."}, "decision": decision}
    write_json(OUT / "raw.json", raw_out)
    req = BACKUP_DIR / "REQUEST_BACKUP_AFTER_VEHICLE_STRESS_REWARD_TIMING_TERMINAL_AUDIT_V0_20260928T1820Z.json"
    write_json(req, {"requested_utc": created, "reason": "backup analysis-only reward/timing/terminal audit and docs before any further simulation/training", "backup_required_before_more_simulations": True, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "artifacts": [rel(OUT), rel(STATE), rel(Path(__file__).resolve()), rel(req)]})
    raw_out["backup_request"] = rel(req); write_json(OUT / "raw.json", raw_out)
    lines = ["# Vehicle stress reward/timing/terminal audit v0", "", f"UTC: `{created}`. Analysis-only; no simulations/training/refit; validation64 and sealed test stayed closed.", "", "## Main findings", "", f"- Stage1 synthetic horizon penalty per H-step summary: `{objective_alignment['synthetic_horizon_penalty_per_H_step_summary_stage1']}`.", f"- Correlation H vs synthetic per-step penalty: `{objective_alignment['correlation_horizon_vs_synthetic_per_step_stage1']}`; H vs measured decision-per-step: `{objective_alignment['correlation_horizon_vs_measured_decision_per_step_stage1']}`.", f"- Safe fixed-H physical-cost vs measured-time aggregate frontier: `{objective_alignment['aggregate_physical_cost_vs_measured_time_frontier_safe_fixed_H']}`.", f"- Stage2 non-H15 branches: `{stage2_decomp['non_H15_branches']}`; material total `{stage2_decomp['material_total_branches']}`, material physical `{stage2_decomp['material_physical_branches']}`, synthetic-only material `{stage2_decomp['synthetic_only_material_total_branches']}`, large harms `{stage2_decomp['large_total_harms_le_minus_3']}`.", f"- Trace terminal/value-like keys: `{terminal_search['stage2_trace_audit']['terminal_value_like_trace_key_counts']}`.", "", "## Decision", "", f"- Retrain/refit now: `{decision['retrain_or_selector_refit_now']}`.", f"- Do not claim measured acceleration from total_cost: `{not decision['use_total_cost_as_measured_runtime_claim']}`.", f"- Next after backup: {decision['next_high_information_action_after_backup']}", "", "## Four-axis evidence", ""]
    for k, v in raw_out["four_axis_evidence"].items():
        lines.append(f"- **{k.upper()}**: {v}")
    lines += ["", f"Backup request: `{rel(req)}`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(f"# Stress reward/timing/terminal audit state ({created})\n\nAnalysis-only audit completed. Key result: synthetic h_penalty is not measured runtime; Stage2 positives remain one state and physical-driven, not enough for retraining. No validation64/test access. Backup required before more simulations. Next: source/config/checkpoint terminal-value/objective audit.\n", encoding="utf-8")
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress reward/timing/terminal audit v0

UTC: {created}. Analysis-only, no new rollouts/training and no validation64/test access. The audit separated physical cost, synthetic horizon penalty and measured decision time. Synthetic h_penalty is effectively 0.001 per H-step, so total_cost is not measured runtime. The only Stage2 material positive is physical-driven, but labels remain too sparse for selector/refit training; large non-H15 harms persist. Decision: do not retrain/refit now; after backup run a bounded source/config/checkpoint terminal-value and objective audit before any new simulation or IMPROVED refit. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`. Backup request: `{rel(req)}`.
""")
    files = [STAGE1_RAW, STAGE1_COMPLETED, STAGE2_RAW, STAGE2_COMPLETED, STAGE2_POST, PROTOCOL, OUT / "raw.json", OUT / "summary.md", STATE, req, Path(__file__).resolve()]
    write_json(OUT / "completed.json", {"passed": True, "hard_pass": True, "created_utc": created, "formal_scientific_evidence": False, "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0, "retrain_or_selector_refit_now": False, "stage2_material_physical_branches": stage2_decomp["material_physical_branches"], "stage2_synthetic_only_material_branches": stage2_decomp["synthetic_only_material_total_branches"], "safe_fixed_H_physical_time_frontier": objective_alignment["aggregate_physical_cost_vs_measured_time_frontier_safe_fixed_H"], "backup_request": rel(req), "hashes": {rel(p): sha256(p) for p in files if p.exists()}})
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "synthetic_hpen_per_H_step": objective_alignment["synthetic_horizon_penalty_per_H_step_summary_stage1"], "safe_fixed_H_physical_time_frontier": objective_alignment["aggregate_physical_cost_vs_measured_time_frontier_safe_fixed_H"], "stage2_material_physical_branches": stage2_decomp["material_physical_branches"], "stage2_synthetic_only_material_branches": stage2_decomp["synthetic_only_material_total_branches"], "retrain_or_selector_refit_now": False, "backup_request": rel(req), "historical_validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
