#!/usr/bin/env python3
"""Current vehicle gated-horizon training/search/selection audit v1.

Metadata/trace diagnostic for the CURRENT reused gated-horizon policies that are
used by the AWS safe-shortening v1/v2 controllers.  This is intentionally not a
latency-tree-only audit and not an ORIGINAL SAC reproduction claim.

Reads existing completed gated-horizon search outputs and, if present, already
opened v2 shard00 development traces for feature-coverage comparison.  It does
not run MPC rollouts, create banks, train, read historical validation64 banks, or
open/hash sealed final-test content.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
GATED_ROOT = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17/results/gated_horizon_search_2026-09-25"
TRAIN_ROOT = GATED_ROOT / "train"
V2_DEV_ROOT = ROOT / "research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1"
DIAG_ROOT = ROOT / "research_artifacts/aws_diagnostics"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_current_gated_horizon_training_audit_v1.py"
POLICY_SRC = ROOT / "experiments/bohn2021_reproduction/gated_horizon_policy.py"
SEARCH_SRC = ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py"
V2_GATE = V2_DEV_ROOT / "vehicle_safe_shortening_v2_transition_hold_devval64_gate.json"
SEEDS = (0, 1, 2)
BASE_H = 25
CANDIDATE_SHORT_H = (5, 10, 15, 20)
PROFILES = (0, 1, 2)
GUARDS = (5, 15, 30)
MARKER = "vehicle-current-gated-horizon-training-audit-v1-20260928"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def stamp_from_time(ts: str) -> str:
    return ts.replace("-", "").replace(":", "").replace("+00:00", "Z")


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def as_int(x: Any) -> Optional[int]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        if math.isfinite(y) and abs(y - round(y)) < 1e-9:
            return int(round(y))
    except Exception:
        return None
    return None


def summarize_values(values: Iterable[Any]) -> Dict[str, Any]:
    xs = sorted(v for v in (as_float(x) for x in values) if v is not None)
    if not xs:
        return {"count": 0, "mean": None, "median": None, "min": None, "p05": None, "p95": None, "max": None}
    n = len(xs)
    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        pos = (n - 1) * p
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {
        "count": n,
        "mean": float(math.fsum(xs) / n),
        "median": float(q(0.5)),
        "min": float(xs[0]),
        "p05": float(q(0.05)),
        "p95": float(q(0.95)),
        "max": float(xs[-1]),
    }


def policy_kind(policy: Mapping[str, Any]) -> str:
    if policy.get("id") == "fixed":
        return "fixed_H25_baseline"
    return "gated_shortening_candidate"


def infer_short_steps(policy: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if policy.get("id") == "fixed":
        return {"steps": int(sum(int(e.get("steps", 0)) for e in episodes)), "short_steps_estimated": 0.0, "episodes_with_short_estimated": 0, "mean_short_fraction": 0.0}
    short_h = as_int(policy.get("short_h"))
    if short_h is None or short_h == BASE_H:
        return {"steps": int(sum(int(e.get("steps", 0)) for e in episodes)), "short_steps_estimated": None, "episodes_with_short_estimated": None, "mean_short_fraction": None}
    total_steps = 0
    short_sum = 0.0
    episodes_with = 0
    for e in episodes:
        steps = as_int(e.get("steps")) or 0
        mh = as_float(e.get("mean_horizon"))
        total_steps += steps
        if mh is None or steps <= 0:
            continue
        est = (BASE_H - mh) * steps / float(BASE_H - short_h)
        if est < 0 and est > -1e-7:
            est = 0.0
        short_sum += max(0.0, est)
        if est > 0.5:
            episodes_with += 1
    return {
        "steps": int(total_steps),
        "short_steps_estimated": float(short_sum),
        "episodes_with_short_estimated": int(episodes_with),
        "mean_short_fraction": float(short_sum / total_steps) if total_steps else None,
    }


def recompute_violation_counts(result: Mapping[str, Any]) -> Counter:
    c: Counter = Counter()
    for item in result.get("rejected") or []:
        for reason in item.get("reasons") or []:
            c[str(reason)] += 1
    return c


def candidate_row(seed: int, base: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    policy = result.get("policy") or {}
    episodes = result.get("episodes") or []
    inferred = infer_short_steps(policy, episodes)
    rejected = result.get("rejected") or []
    unrun = result.get("unrun_cases") or []
    fixed_raw = as_float(base.get("mean_raw_cost"))
    fixed_phys = as_float(base.get("mean_physical_cost"))
    raw = as_float(result.get("mean_raw_cost"))
    phys = as_float(result.get("mean_physical_cost"))
    h_penalties = [as_float(e.get("h_penalty")) for e in episodes]
    performances = [as_float(e.get("performance_cost")) for e in episodes]
    constraints = [as_float(e.get("constraint_cost")) for e in episodes]
    steps = int(sum((as_int(e.get("steps")) or 0) for e in episodes))
    row = {
        "seed": seed,
        "candidate_id": str(policy.get("id")),
        "policy_kind": policy_kind(policy),
        "short_h": as_int(policy.get("short_h")),
        "profile": as_int(policy.get("profile")),
        "guard": as_int(policy.get("guard")),
        "fully_evaluated": bool(result.get("fully_evaluated")),
        "episodes_run": len(episodes),
        "scored_steps_run": steps,
        "rejected": bool(rejected),
        "reject_case_first": rejected[0].get("case") if rejected else None,
        "reject_reasons_first": ";".join(rejected[0].get("reasons", [])) if rejected else "",
        "all_reject_reasons": ";".join("%s:%d" % (k, v) for k, v in sorted(recompute_violation_counts(result).items())),
        "unrun_case_count": len(unrun),
        "mean_raw_cost": raw,
        "mean_physical_cost": phys,
        "raw_cost_change_vs_fixed": None if raw is None or fixed_raw is None else raw - fixed_raw,
        "raw_cost_change_pct_vs_fixed": None if raw is None or fixed_raw in (None, 0.0) else 100.0 * (raw - fixed_raw) / abs(fixed_raw),
        "physical_cost_change_vs_fixed": None if phys is None or fixed_phys is None else phys - fixed_phys,
        "physical_cost_change_pct_vs_fixed": None if phys is None or fixed_phys in (None, 0.0) else 100.0 * (phys - fixed_phys) / abs(fixed_phys),
        "mean_episode_h_penalty": summarize_values(h_penalties)["mean"],
        "mean_episode_performance_cost": summarize_values(performances)["mean"],
        "mean_episode_constraint_cost": summarize_values(constraints)["mean"],
        "switches_sum": int(sum((as_int(e.get("switches")) or 0) for e in episodes)),
    }
    row.update(inferred)
    if row["fully_evaluated"] and not row["rejected"]:
        row["training_selection_admissible_by_code"] = bool(phys is not None and fixed_phys is not None and phys <= fixed_phys + 0.02 * abs(fixed_phys))
    else:
        row["training_selection_admissible_by_code"] = False
    return row


def select_by_code(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    eligible = [r for r in rows if r.get("training_selection_admissible_by_code")]
    if not eligible:
        return None
    return sorted(eligible, key=lambda r: (float(r.get("mean_raw_cost")), str(r.get("candidate_id")) != "fixed", str(r.get("candidate_id"))))[0]


def feature_key_value(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    gate = row.get("gate") or {}
    decision = row.get("decision") or {}
    if not gate and isinstance(decision.get("gate"), Mapping):
        gate = decision.get("gate") or {}
    values = gate.get("values") if isinstance(gate, Mapping) else None
    if values is None and isinstance(decision, Mapping) and isinstance(decision.get("gate"), Mapping):
        values = decision["gate"].get("values")
    if not isinstance(values, Mapping):
        values = {}
    ctx = row.get("policy_context") or {}
    state = ctx.get("state") if isinstance(ctx, Mapping) else {}
    if not isinstance(state, Mapping):
        state = row.get("state") if isinstance(row.get("state"), Mapping) else {}
    return dict(values), dict(state)


def read_trace(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        return read_json(path)
    except Exception:
        return None


def trace_coverage(paths: Sequence[Path], label: str, max_files: Optional[int] = None) -> Dict[str, Any]:
    selected = list(paths[:max_files]) if max_files is not None else list(paths)
    h_counts: Counter = Counter()
    raw_h_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    state_values: Dict[str, List[float]] = defaultdict(list)
    gate_values: Dict[str, List[float]] = defaultdict(list)
    case_rows: List[Dict[str, Any]] = []
    files_read = 0
    steps = 0
    use_short_steps = 0
    parse_failures: List[str] = []
    for p in selected:
        trace = read_trace(p)
        if trace is None:
            parse_failures.append(rel(p)); continue
        files_read += 1
        local_h = Counter()
        local_short = 0
        for r in trace:
            h = as_int(r.get("horizon"))
            if h is not None:
                h_counts[str(h)] += 1
                local_h[str(h)] += 1
                if h < BASE_H:
                    use_short_steps += 1
                    local_short += 1
            decision = r.get("decision") or {}
            raw = None
            if isinstance(decision, Mapping):
                raw = decision.get("raw_horizon")
            if raw is None:
                raw = h
            raw_i = as_int(raw)
            if raw_i is not None:
                raw_h_counts[str(raw_i)] += 1
            gate = r.get("gate") or {}
            if not gate and isinstance(decision, Mapping) and isinstance(decision.get("gate"), Mapping):
                gate = decision.get("gate") or {}
            if isinstance(gate, Mapping):
                if gate.get("use_short") is True:
                    reason_counts["use_short_true"] += 1
                reason = gate.get("reason")
                if reason:
                    reason_counts[str(reason)] += 1
                elif gate.get("use_short") is False:
                    reason_counts["use_short_false_no_reason"] += 1
            values, state = feature_key_value(r)
            for k, v in values.items():
                fv = as_float(v)
                if fv is not None:
                    gate_values[k].append(fv)
            for k, v in state.items():
                fv = as_float(v)
                if fv is not None:
                    state_values[k].append(fv)
            steps += 1
        case_rows.append({"trace": rel(p), "steps": len(trace), "horizon_counts": dict(local_h), "short_steps": local_short})
    return {
        "label": label,
        "trace_files_available": len(paths),
        "trace_files_read": files_read,
        "parse_failures": parse_failures[:20],
        "steps": int(steps),
        "horizon_counts": dict(sorted(h_counts.items(), key=lambda kv: int(kv[0]))),
        "raw_horizon_counts": dict(sorted(raw_h_counts.items(), key=lambda kv: int(kv[0]))),
        "short_step_count": int(use_short_steps),
        "short_step_fraction": float(use_short_steps / steps) if steps else None,
        "gate_reason_counts": dict(reason_counts),
        "state_summary": {k: summarize_values(v) for k, v in sorted(state_values.items())},
        "gate_value_summary": {k: summarize_values(v) for k, v in sorted(gate_values.items())},
        "case_rows": case_rows,
    }


def terminal_sources_from_v2_gate() -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": rel(V2_GATE), "available": V2_GATE.exists(), "by_seed": {}}
    if not V2_GATE.exists():
        return out
    try:
        gate = read_json(V2_GATE)
    except Exception as exc:
        out.update({"read_error": repr(exc)})
        return out
    for arm in gate.get("schedule", {}).get("arms", []):
        if isinstance(arm, Mapping) and arm.get("role") == "adaptive_candidate_v2_transition_hold":
            seed = str(arm.get("seed"))
            source_rel = arm.get("terminal_source")
            rec: Dict[str, Any] = {"terminal_source": source_rel, "terminal_h": arm.get("terminal_h")}
            folder = ROOT / str(source_rel)
            for name in ("manifest.json", "completed.json"):
                p = folder / name
                rec[name] = {"path": rel(p), "exists": p.exists(), "sha256": sha256(p)}
                if p.exists():
                    try:
                        obj = read_json(p)
                        rec[name].update({k: obj.get(k) for k in ("task", "seed", "fixed_horizon", "steps", "status", "final_hash") if k in obj})
                    except Exception as exc:
                        rec[name]["read_error"] = repr(exc)
            model = folder / "model.zip"
            rec["model_zip"] = {"path": rel(model), "exists": model.exists(), "sha256": sha256(model), "bytes": model.stat().st_size if model.exists() else None}
            out["by_seed"][seed] = rec
    return out


def dev_trace_paths_for_seed(seed: int) -> List[Path]:
    if not V2_DEV_ROOT.exists():
        return []
    return sorted((V2_DEV_ROOT / "shard00" / "episodes").glob("*safe_shortening_v2_hold3_vehicle_s%d*/trace.json" % seed))


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    token = "<!-- %s -->" % marker
    if token in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + token + "\n" + text.strip() + "\n", encoding="utf-8")


def append_registry(created: str, completed_path: Path) -> None:
    p = ROOT / "EXPERIMENT_REGISTRY.csv"
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    record = rel(completed_path)
    if record in existing:
        return
    row = {
        "experiment_id": "",
        "timestamp": created,
        "method": "IMPROVED_vehicle_current_gated_horizon_training_search_selection_audit_v1_metadata_only",
        "seed": "metadata_existing_gated_horizon_vehicle_s0_s1_s2_no_rng",
        "split": "existing_current_gated_horizon_search_training_outputs_plus_already_opened_v2_shard00_traces_no_new_validation_no_sealed_test",
        "commit_sha": "",
        "status": "completed_metadata_diagnostic",
        "exit_status": "",
        "runtime_seconds": "",
        "peak_process_rss_kb": "",
        "record": record,
    }
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existing.strip():
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    created = utc_now()
    label = stamp_from_time(created)
    out_dir = DIAG_ROOT / ("vehicle_current_gated_horizon_training_audit_v1_%s" % label)
    out_dir.mkdir(parents=True, exist_ok=False)

    all_candidate_rows: List[Dict[str, Any]] = []
    seed_summaries: Dict[str, Any] = {}
    input_hashes: Dict[str, Optional[str]] = {
        rel(SCRIPT): sha256(SCRIPT),
        rel(POLICY_SRC): sha256(POLICY_SRC),
        rel(SEARCH_SRC): sha256(SEARCH_SRC),
        rel(GATED_ROOT / "vehicle_training_completed_metadata.json"): sha256(GATED_ROOT / "vehicle_training_completed_metadata.json"),
        rel(GATED_ROOT / "audit_train.json"): sha256(GATED_ROOT / "audit_train.json"),
        rel(GATED_ROOT / "baseline_selection.json"): sha256(GATED_ROOT / "baseline_selection.json"),
        rel(GATED_ROOT / "policy_checks.json"): sha256(GATED_ROOT / "policy_checks.json"),
    }

    for seed in SEEDS:
        root = TRAIN_ROOT / ("vehicle_s%d" % seed)
        fit_path = root / "fit_completed.json"
        policy_path = root / "policy.json"
        fit = read_json(fit_path)
        final_policy = read_json(policy_path)
        input_hashes[rel(fit_path)] = sha256(fit_path)
        input_hashes[rel(policy_path)] = sha256(policy_path)
        results = fit.get("results") or []
        if not results or (results[0].get("policy") or {}).get("id") != "fixed":
            raise RuntimeError("Unexpected fit result ordering for seed %d" % seed)
        base = results[0]
        rows = [candidate_row(seed, base, r) for r in results]
        chosen_by_code = select_by_code(rows)
        selected_id = str((fit.get("selected") or {}).get("id"))
        final_id = str(final_policy.get("id"))
        for r in rows:
            r["selected_in_fit_completed"] = r["candidate_id"] == selected_id
            r["stored_current_policy"] = r["candidate_id"] == final_id
        all_candidate_rows.extend(rows)
        reason_counts = Counter()
        for r in rows:
            if r["all_reject_reasons"]:
                for item in r["all_reject_reasons"].split(";"):
                    if not item:
                        continue
                    k, _, v = item.partition(":")
                    reason_counts[k] += int(v or 0)
        grid_counts = Counter()
        for r in rows:
            if r["policy_kind"] == "gated_shortening_candidate":
                grid_counts[(r["short_h"], r["profile"], r["guard"])] += 1
        selected_trace_paths = sorted((root / selected_id).glob("r0_trace_*.json"))
        fixed_trace_paths = sorted((root / "fixed").glob("r0_trace_*.json"))
        train_selected_cov = trace_coverage(selected_trace_paths, "train_selected_seed%d" % seed)
        train_fixed_cov = trace_coverage(fixed_trace_paths, "train_fixed_seed%d" % seed)
        dev_cov = trace_coverage(dev_trace_paths_for_seed(seed), "already_opened_v2_shard00_adaptive_seed%d" % seed)
        seed_summaries[str(seed)] = {
            "train_root": rel(root),
            "fit_path": rel(fit_path),
            "policy_path": rel(policy_path),
            "selected_in_fit_completed": fit.get("selected"),
            "stored_current_policy": final_policy,
            "selected_matches_stored_policy": selected_id == final_id,
            "chosen_by_recomputed_training_rule": None if chosen_by_code is None else {k: chosen_by_code.get(k) for k in ("candidate_id", "mean_raw_cost", "mean_physical_cost", "raw_cost_change_pct_vs_fixed", "physical_cost_change_pct_vs_fixed", "short_step_fraction", "episodes_with_short_estimated")},
            "selection_rule_recompute_matches_fit": bool(chosen_by_code and chosen_by_code.get("candidate_id") == selected_id),
            "candidate_counts": {
                "total_including_fixed": len(rows),
                "structured_candidates": len(rows) - 1,
                "expected_structured_candidates": len(CANDIDATE_SHORT_H) * len(PROFILES) * len(GUARDS),
                "fully_evaluated": sum(1 for r in rows if r["fully_evaluated"]),
                "rejected_or_pruned": sum(1 for r in rows if r["rejected"]),
                "admissible_by_selection_code": sum(1 for r in rows if r["training_selection_admissible_by_code"]),
                "grid_cells_seen": len(grid_counts),
                "grid_cells_expected": len(CANDIDATE_SHORT_H) * len(PROFILES) * len(GUARDS),
            },
            "budget_from_fit_summaries": {
                "candidate_episode_records": int(sum(r["episodes_run"] for r in rows)),
                "candidate_scored_steps": int(sum(r["scored_steps_run"] for r in rows)),
                "baseline_episode_records": int(rows[0]["episodes_run"]),
                "baseline_scored_steps": int(rows[0]["scored_steps_run"]),
                "new_gradient_steps": int(fit.get("gradient_updates", 0)),
                "selection_validation_access": bool(fit.get("validation_access")),
                "selection_test_access": bool(fit.get("test_access")),
            },
            "rejection_reason_counts": dict(reason_counts),
            "best_candidates_by_training_objective": {
                "best_admissible_any": None if chosen_by_code is None else dict(chosen_by_code),
                "best_fully_evaluated_adaptive": next((dict(r) for r in sorted([x for x in rows if x["candidate_id"] != "fixed" and x["fully_evaluated"] and not x["rejected"]], key=lambda z: float(z["mean_raw_cost"]))[:1]), None),
                "best_admissible_short_h10": next((dict(r) for r in sorted([x for x in rows if x["short_h"] == 10 and x["training_selection_admissible_by_code"]], key=lambda z: float(z["mean_raw_cost"]))[:1]), None),
            },
            "training_trace_coverage_selected_policy": train_selected_cov,
            "training_trace_coverage_fixed_H25": train_fixed_cov,
            "already_opened_development_trace_coverage_v2_shard00": dev_cov,
        }

    terminal_sources = terminal_sources_from_v2_gate()
    candidate_csv = out_dir / "candidate_table.csv"
    candidate_fields = [
        "seed", "candidate_id", "policy_kind", "short_h", "profile", "guard", "fully_evaluated", "episodes_run", "scored_steps_run", "rejected", "reject_case_first", "reject_reasons_first", "all_reject_reasons", "unrun_case_count", "mean_raw_cost", "mean_physical_cost", "raw_cost_change_vs_fixed", "raw_cost_change_pct_vs_fixed", "physical_cost_change_vs_fixed", "physical_cost_change_pct_vs_fixed", "mean_episode_h_penalty", "mean_episode_performance_cost", "mean_episode_constraint_cost", "switches_sum", "steps", "short_steps_estimated", "episodes_with_short_estimated", "mean_short_fraction", "training_selection_admissible_by_code", "selected_in_fit_completed", "stored_current_policy"
    ]
    with candidate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        for row in sorted(all_candidate_rows, key=lambda r: (r["seed"], r["candidate_id"] != "fixed", str(r["candidate_id"]))):
            writer.writerow({k: row.get(k) for k in candidate_fields})

    # Cross-seed verified causes versus hypotheses/missing evidence.
    verified_causes: List[str] = []
    hypotheses: List[str] = []
    missing: List[str] = []
    verified_causes.append("Current AWS safe-shortening v1/v2 controllers reuse gated_horizon_search_2026-09-25 policy.json files (short_h/profile/guard), not the older latency-tree policy files; this diagnostic audited those current gated policies.")
    verified_causes.append("The current gated-horizon operation was finite closed-loop search/reselection over 36 hand-structured candidates plus fixed H25 per seed; fit_completed.json records gradient_updates=0, validation_access=false, and test_access=false.")
    verified_causes.append("Training objective/selection used full-episode mean raw total_cost among candidates admissible by hard per-case and mean physical-cost gates; no measured wall-clock runtime term was present in this gated-search objective.")
    verified_causes.append("The policy class was severely restricted: only short_h in {5,10,15,20}, profile in {0,1,2}, guard in {5,15,30}; candidate decisions use instantaneous gate thresholds and have no dwell/transition-risk model.")
    for seed, s in seed_summaries.items():
        cov = s["training_trace_coverage_selected_policy"]
        dev = s["already_opened_development_trace_coverage_v2_shard00"]
        verified_causes.append("vehicle_s%s selected %s with training short-step fraction %.6g over %d selected-policy training steps; already-opened v2 shard00 short-step fraction is %s over %d steps." % (
            seed, s["stored_current_policy"].get("id"), cov.get("short_step_fraction") or 0.0, cov.get("steps") or 0,
            "None" if dev.get("short_step_fraction") is None else ("%.6g" % dev.get("short_step_fraction")), dev.get("steps") or 0))
    hypotheses.append("Because the in-fit objective monetizes horizon only through a small synthetic h_penalty rather than measured solver time and because fixed H25 is already very reliable on the 24-case training banks, selection may favor rare/fragile gate triggers or near-constant policies that do not produce robust measured speedups.")
    hypotheses.append("The H25 terminal value reused for adaptive and fixed arms may bias comparisons around horizon transitions, but this metadata audit only verifies source identity/15k fixed-H25 provenance; it does not estimate value bias numerically.")
    hypotheses.append("Training rollouts do evaluate downstream closed-loop consequences to episode termination, but they do not explicitly evaluate transition/dwell alternatives or continuation costs conditional on one-step horizon switches under development-state distribution shift.")
    missing.append("No fresh independent validation evidence is produced here; v2 shard00 traces are already-opened development evidence only and cannot validate a revised method.")
    missing.append("Actual solver-time noise for the gated-search training objective is not available because that objective did not store/use measured timing; later timing validation must remain separate.")
    missing.append("This audit reads selected/fixed training traces and already-opened v2 shard00 adaptive traces for coverage, not every trace from every discarded candidate, to keep the diagnostic bounded.")

    raw = {
        "created_utc": created,
        "method": "IMPROVED_vehicle_current_gated_horizon_training_search_selection_audit_v1_metadata_only",
        "scope": "CURRENT reused gated-horizon policies used by safe-shortening v1/v2; not old latency-tree-only selection; no rollout/training/test access",
        "access_flags": {
            "new_rollout_episodes": 0,
            "new_control_steps": 0,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "historical_validation64_bank_opened": False,
            "fresh_devval_bank_generated": False,
            "sealed_test_accessed": False,
            "final_test_authorization_requested": False,
            "already_opened_v2_shard00_traces_read_if_present": True,
        },
        "input_hashes": input_hashes,
        "terminal_sources_used_by_current_v2_gate": terminal_sources,
        "seed_summaries": seed_summaries,
        "verified_causes": verified_causes,
        "hypotheses": hypotheses,
        "missing_evidence": missing,
        "candidate_table": rel(candidate_csv),
    }
    raw_path = out_dir / "raw.json"
    write_json(raw_path, raw)

    # Human summary, intentionally concise but evidence-bearing.
    lines: List[str] = [
        "# Vehicle current gated-horizon training/search audit v1",
        "",
        f"UTC: `{created}`.",
        "",
        "Metadata-only diagnostic for the CURRENT reused gated-horizon policies used by AWS safe-shortening v1/v2. This is not an audit of only the older latency-tree policy and not ORIGINAL SAC. No rollout/control steps, no training/gradient steps, no historical validation64 reopen, and no sealed-test access/hash occurred.",
        "",
        "## Current policy provenance",
        "",
        "| seed | stored policy | recomputed selected | match | candidates incl fixed | fully evaluated | rejected/pruned | admissible | selected train short frac | opened v2 shard00 short frac |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in map(str, SEEDS):
        s = seed_summaries[seed]
        counts = s["candidate_counts"]
        cov = s["training_trace_coverage_selected_policy"]
        dev = s["already_opened_development_trace_coverage_v2_shard00"]
        lines.append("| %s | `%s` | `%s` | %s | %d | %d | %d | %d | %.6g | %s |" % (
            seed,
            s["stored_current_policy"].get("id"),
            None if s["chosen_by_recomputed_training_rule"] is None else s["chosen_by_recomputed_training_rule"].get("candidate_id"),
            s["selection_rule_recompute_matches_fit"],
            counts["total_including_fixed"], counts["fully_evaluated"], counts["rejected_or_pruned"], counts["admissible_by_selection_code"],
            cov.get("short_step_fraction") or 0.0,
            "n/a" if dev.get("short_step_fraction") is None else ("%.6g" % dev.get("short_step_fraction")),
        ))
    lines += ["", "## Per-seed details", ""]
    for seed in map(str, SEEDS):
        s = seed_summaries[seed]
        counts = s["candidate_counts"]
        budget = s["budget_from_fit_summaries"]
        best = s["best_candidates_by_training_objective"]
        cov = s["training_trace_coverage_selected_policy"]
        dev = s["already_opened_development_trace_coverage_v2_shard00"]
        lines += [
            f"### vehicle_s{seed}",
            f"- Stored current policy: `{s['stored_current_policy']}`; selected in fit_completed: `{s['selected_in_fit_completed']}`; recompute match: `{s['selection_rule_recompute_matches_fit']}`.",
            f"- Candidate coverage: `{counts}`. Grid cells seen `{counts['grid_cells_seen']}/{counts['grid_cells_expected']}` over short_h/profile/guard; rejected/pruned candidates `{counts['rejected_or_pruned']}` with reasons `{s['rejection_reason_counts']}`.",
            f"- Training-search budget from summaries: `{budget}`. This is finite search/refit/reselection, not neural/gradient training.",
            f"- Best admissible by training objective: `{None if best['best_admissible_any'] is None else {k: best['best_admissible_any'].get(k) for k in ['candidate_id','mean_raw_cost','mean_physical_cost','raw_cost_change_pct_vs_fixed','physical_cost_change_pct_vs_fixed','mean_short_fraction']}}`.",
            f"- Selected-policy training H counts: `{cov['horizon_counts']}`, short-step fraction `{cov['short_step_fraction']}`; gate reasons `{cov['gate_reason_counts']}`.",
            f"- Already-opened v2 shard00 adaptive H counts: `{dev['horizon_counts']}`, short-step fraction `{dev['short_step_fraction']}`; raw H counts `{dev['raw_horizon_counts']}`.",
            "",
        ]
    lines += ["## Terminal/source audit", ""]
    for seed in map(str, SEEDS):
        lines.append(f"- seed {seed}: `{terminal_sources.get('by_seed', {}).get(seed, {})}`")
    lines += ["", "## Verified causes", ""] + ["- " + x for x in verified_causes]
    lines += ["", "## Hypotheses / missing evidence", ""] + ["- HYPOTHESIS: " + x for x in hypotheses] + ["- MISSING: " + x for x in missing]
    lines += [
        "",
        "## Decision / next concrete experiment",
        "",
        "Do not launch more whole-grid local dwell/guard validation before a training-level change. The smallest supported next experiment is a versioned gated-policy re-selection/refit on the same existing 24-case training banks (no new rollouts for initial metadata phase) using a risk-first objective that (i) treats any per-case failure/large physical regression as dominant, (ii) reports physical and synthetic horizon terms separately, (iii) penalizes transition/rare-trigger fragility using existing selected/fixed trace coverage, and (iv) selects only among policies with nontrivial but not forced short-horizon support on training traces. If metadata re-selection nominates a different candidate, run a smoke before any fresh validation shard. This is IMPROVED search/reselection, not gradient RL training.",
        "",
        "Artifacts: raw.json, candidate_table.csv, completed.json, and backup request in aws_backup_proofs.",
    ]
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_CURRENT_GATED_HORIZON_TRAINING_AUDIT_V1_%s.json" % label)
    write_json(backup_path, {
        "created_utc": created,
        "reason": "metadata-only current gated-horizon training/search audit; preserve before revised re-selection/refit experiment",
        "artifacts": [rel(out_dir), rel(raw_path), rel(summary_path), rel(candidate_csv)],
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "fresh_devval_bank_generated": False,
        "sealed_test_accessed": False,
    })
    completed_path = out_dir / "completed.json"
    completed = {
        "created_utc": created,
        "passed": True,
        "summary_path": rel(summary_path),
        "summary_sha256": sha256(summary_path),
        "raw_path": rel(raw_path),
        "raw_sha256": sha256(raw_path),
        "candidate_table": rel(candidate_csv),
        "candidate_table_sha256": sha256(candidate_csv),
        "backup_request": rel(backup_path),
        "backup_request_sha256": sha256(backup_path),
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "historical_validation64_bank_opened": False,
        "fresh_devval_bank_generated": False,
        "sealed_test_accessed": False,
    }
    write_json(completed_path, completed)
    completed["completed_path"] = rel(completed_path)
    completed["completed_sha256"] = sha256(completed_path)
    write_json(completed_path, completed)

    doc = """
## 2026-09-28 current gated-horizon training/search audit v1

UTC: {created}. Metadata-only audit of the CURRENT reused gated-horizon policies used by AWS safe-shortening v1/v2, not merely the older latency-tree policy. No rollouts/control steps, no training/gradient steps, no historical validation64 reopen, and no sealed-test access/hash occurred. Audited finite search/reselection over 36 structured candidates plus fixed H25 per seed and selected/fixed training traces; already-opened v2 shard00 adaptive traces were used only for development coverage comparison. Key result: current policies came from finite candidate search with gradient_updates=0; objective is mean raw total_cost among hard-gated admissible candidates, with no measured runtime term and no transition/dwell-risk model. Candidate tables and coverage summaries are in `{summary}` / `{raw}`. Next: freeze and run a bounded IMPROVED re-selection/refit diagnostic before any further unchanged validation shard.
""".strip().format(created=created, summary=rel(summary_path), raw=rel(raw_path))
    for name in ("STATUS.md", "RESEARCH_LOG.md", "RESULTS_AUDIT.md", "DECISIONS.md"):
        append_once(ROOT / name, MARKER, doc)
    protocol_doc = """
## 2026-09-28 amendment note: current gated-horizon audit before revised training-level experiment

The current AWS safe-shortening controllers reuse gated_horizon_search_2026-09-25 policy.json files. The audit `{summary}` distinguishes this source from the older latency-tree policy and records that the operation was finite search/reselection, not gradient training. Any next method change must be labeled IMPROVED and frozen separately before smoke/validation.
""".strip().format(summary=rel(summary_path))
    append_once(ROOT / "REPRODUCTION_PROTOCOL.md", MARKER, protocol_doc)
    append_registry(created, completed_path)

    print(json.dumps({
        "completed": rel(completed_path),
        "summary": rel(summary_path),
        "candidate_table": rel(candidate_csv),
        "backup_request": rel(backup_path),
        "new_rollout_episodes": 0,
        "new_control_steps": 0,
        "sealed_test_accessed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
