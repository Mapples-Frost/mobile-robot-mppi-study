#!/usr/bin/env python3
"""Offline robust-terminal H10/H15 label CV diagnostic.

Development-only. No new simulation, no rollout, no validation64 read, no sealed
final-test read, no training, no gradient updates and no selector/value refit.

Motivation: terminal-consistency audit v0 showed that terminal-agreement labels
retain measured compute value against fixed true H15, but transferring labels
across terminal profiles can be catastrophically unsafe. This script asks a
more deployable/value-calibration question before any rollout: if each branch
state is assigned one terminal-robust H10-vs-H15 target, can a bounded
state-observable model recover a safe measured compute tradeoff against fixed
true H15 under leave-state and leave-case splits?
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_robust_terminal_label_cv_v0"
STAMP = "20260929T0925Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V0B_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair.py"
STATECV_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded.py"
CONSIST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_consistency_audit_v0_20260929T0915Z/completed.json"
STATECV_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded_20260929T0905Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-robust-terminal-label-cv-v0-{STAMP}"
MIN_SAVE = 0.05
STRONG_SAVE = 0.10


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return rel(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    return obj


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sf(value: Any, default: float = 0.0) -> float:
    try:
        y = float(value)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def finite(values: Iterable[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        try:
            y = float(v)
        except Exception:
            continue
        if math.isfinite(y):
            xs.append(y)
    xs.sort()
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None, "sum": 0.0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * p
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "median": q(0.5), "mean": math.fsum(xs) / len(xs), "p95": q(0.95), "max": xs[-1], "sum": math.fsum(xs)}


def completed_ok(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ContractError(f"missing completed marker: {rel(path)}")
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise ContractError(f"completed marker did not pass: {rel(path)}")
    if obj.get("validation64_bank_opened") is True or obj.get("sealed_test_accessed") is True:
        raise ContractError(f"unexpected validation/test access flag in {rel(path)}")
    return obj


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import {rel(path)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def group_by_state(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["state_key"]), []).append(r)
    return out


def decision_saving(rows: Sequence[Mapping[str, Any]]) -> float:
    h10 = math.fsum(sf(r["h10"]["decision"]) for r in rows)
    h15 = math.fsum(sf(r["h15"]["decision"]) for r in rows)
    return (h15 - h10) / h15 if h15 > 0 else 0.0


def aggregate_label(rows: Sequence[Mapping[str, Any]]) -> int:
    # Treat the two terminal profiles as alternative terminal-value calibrations
    # for the same online state, and require aggregate physical/timing safety.
    h10_phys = math.fsum(sf(r["h10"]["physical"]) for r in rows)
    h15_phys = math.fsum(sf(r["h15"]["physical"]) for r in rows)
    tol = max(2.0 * len(rows), 0.05 * abs(h15_phys))
    if all(bool(r["h10"]["safe"]) for r in rows) and h10_phys - h15_phys <= tol and decision_saving(rows) >= MIN_SAVE:
        return 10
    return 15


def agreement_label(rows: Sequence[Mapping[str, Any]]) -> int:
    return 10 if rows and {int(r.get("label", 15)) for r in rows} == {10} else 15


def strict_per_profile_label(rows: Sequence[Mapping[str, Any]], cap: float = 2.0) -> int:
    if decision_saving(rows) < MIN_SAVE:
        return 15
    for r in rows:
        h15_phys = sf(r["h15"]["physical"])
        tol = min(max(2.0, 0.05 * abs(h15_phys)), cap)
        if not bool(r["h10"]["safe"]):
            return 15
        if sf(r["h10"]["physical"]) - h15_phys > tol:
            return 15
    return 10


def apply_strategy(rows: Sequence[Mapping[str, Any]], strategy: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    states = group_by_state(rows)
    labels: Dict[str, int] = {}
    state_debug = []
    for sk, state_rows in sorted(states.items()):
        if strategy == "terminal_agreement_only":
            lab = agreement_label(state_rows)
        elif strategy == "aggregate_terminal_robust":
            lab = aggregate_label(state_rows)
        elif strategy == "strict_per_profile_regret2":
            lab = strict_per_profile_label(state_rows, cap=2.0)
        else:
            raise ValueError(strategy)
        labels[sk] = lab
        state_debug.append({
            "state_key": sk,
            "source": state_rows[0].get("source"),
            "case": state_rows[0].get("case"),
            "labels_by_profile": {str(r.get("terminal_profile")): int(r.get("label", 15)) for r in state_rows},
            "robust_label": lab,
            "aggregate_h10_minus_h15_physical": math.fsum(sf(r["h10"]["physical"]) - sf(r["h15"]["physical"]) for r in state_rows),
            "aggregate_decision_saving": decision_saving(state_rows),
            "max_profile_h10_minus_h15_physical": max([sf(r["h10"]["physical"]) - sf(r["h15"]["physical"]) for r in state_rows] + [0.0]),
        })
    out: List[Dict[str, Any]] = []
    for r in rows:
        x = dict(r)
        x["original_label"] = int(r.get("label", 15))
        x["label"] = int(labels[str(r["state_key"])])
        x["robust_label_strategy"] = strategy
        out.append(x)
    return out, {"state_labels": state_debug, "label_counts_by_state": {"10": sum(1 for v in labels.values() if v == 10), "15": sum(1 for v in labels.values() if v == 15)}}


def bounded_configs() -> List[Dict[str, Any]]:
    # Smaller than the prior 36-config sweep: enough to test learnability of the
    # repaired target without another broad unchanged search.
    out: List[Dict[str, Any]] = []
    for mode in ("raw", "raw_abs", "raw_abs_l2"):
        for k in (1, 3):
            out.append({"family": "knn", "mode": mode, "k": k, "vote_threshold": 0.67})
        out.append({"family": "centroid", "mode": mode, "h10_distance_ratio": 1.0})
        out.append({"family": "centroid", "mode": mode, "h10_distance_ratio": 1.25})
    return out


def evaluate_strategy(v0b: Any, statecv: Any, rows: Sequence[Mapping[str, Any]], strategy: str, cand: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    repaired, dbg = apply_strategy(rows, strategy)
    datasets = {
        "all_profiles": repaired,
        "risk_anchor_source_all_profiles": [r for r in repaired if r.get("source") == "risk_anchor"],
        "matched_terminal_only": [r for r in repaired if r.get("terminal_profile") == "matched_terminal"],
        "shared_h15_terminal_only": [r for r in repaired if r.get("terminal_profile") == "shared_h15_terminal"],
    }
    results: Dict[str, Any] = {}
    for name, subset in datasets.items():
        if not subset:
            continue
        # analyze_dataset performs fixed-H10/H15/oracle baselines and nested
        # leave-state / leave-case CV using only initial branch observations.
        results[name] = statecv.analyze_dataset(v0b, name, subset, cand)
    passes = []
    for name, res in results.items():
        ls = (res.get("state_observable_nested_cv") or {}).get("leave_state_out") or {}
        lc = (res.get("state_observable_nested_cv") or {}).get("leave_case_out") or {}
        if bool(ls.get("core_pass_5pct") and lc.get("core_pass_5pct")):
            passes.append(name)
    strong = []
    for name, res in results.items():
        ls = (res.get("state_observable_nested_cv") or {}).get("leave_state_out") or {}
        lc = (res.get("state_observable_nested_cv") or {}).get("leave_case_out") or {}
        if bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct")):
            strong.append(name)
    return {"strategy": strategy, "debug": dbg, "results": results, "pass_datasets": passes, "strong_pass_datasets": strong}


def append_docs(block: str) -> None:
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if MARKER not in old:
            path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    done_path = OUT_DIR / "completed.json"
    if done_path.exists():
        done = read_json(done_path)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    consist_done = completed_ok(CONSIST_DONE)
    statecv_done = completed_ok(STATECV_DONE)
    v0b = import_module(V0B_SCRIPT, "h10h15_v0b_for_robust_terminal")
    statecv = import_module(STATECV_SCRIPT, "state_observable_cv_for_robust_terminal")
    samples, _aux = v0b.load_samples()
    if not samples:
        raise ContractError("no v0b samples loaded")
    cand = bounded_configs()
    strategies = ["terminal_agreement_only", "aggregate_terminal_robust", "strict_per_profile_regret2"]
    analyses = {s: evaluate_strategy(v0b, statecv, samples, s, cand) for s in strategies}
    pass_summary = {s: analyses[s]["pass_datasets"] for s in strategies}
    strong_summary = {s: analyses[s]["strong_pass_datasets"] for s in strategies}
    robust_oracle_value = {}
    for s, a in analyses.items():
        all_res = (a.get("results") or {}).get("all_profiles") or {}
        oracle = ((all_res.get("baselines") or {}).get("oracle_H10H15") or {})
        robust_oracle_value[s] = {k: oracle.get(k) for k in ("horizon_counts", "physical_delta_vs_H15", "physical_tolerance_vs_H15", "decision_relative_saving_vs_H15", "solver_relative_saving_vs_H15", "core_pass_5pct", "core_pass_10pct", "unsafe_chosen_groups")}
    any_all_pass = any("all_profiles" in v for v in pass_summary.values())
    any_risk_pass = any("risk_anchor_source_all_profiles" in v for v in pass_summary.values())
    if any_all_pass:
        decision = "robust terminal labels are learnable on pooled development states; after backup freeze a tiny selector-overhead smoke versus fixed true H15 before any validation"
    elif any_risk_pass:
        decision = "robust terminal labels are learnable only on risk-anchor subset; next collect/freeze source-independent confirmation or value-calibration smoke, not broad validation"
    else:
        decision = "robust terminal label repair preserves some oracle value but state-observable CV still fails; prioritize representation/value-calibration or additional controlled state coverage before selector rollout"
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_ROBUST_TERMINAL_LABEL_CV_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_robust_terminal_label_cv_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_configs_per_outer_fold": len(cand),
        "inputs": {"terminal_consistency_completed": rel(CONSIST_DONE), "state_observable_cv_completed": rel(STATECV_DONE), "v0b_script": rel(V0B_SCRIPT), "statecv_script": rel(STATECV_SCRIPT)},
        "prerequisite_headlines": {"terminal_consistency": consist_done.get("headline"), "state_observable_cv": statecv_done.get("headline")},
        "strategies": strategies,
        "pass_summary": pass_summary,
        "strong_pass_summary": strong_summary,
        "robust_oracle_value_all_profiles": robust_oracle_value,
        "decision": decision,
        "analyses": analyses,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle true-variable-H robust terminal-label CV v0",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Bounded state-observable configs per outer fold: `{len(cand)}`.",
        f"- Pass summary: `{pass_summary}`.",
        f"- Strong pass summary: `{strong_summary}`.",
        f"- All-profile robust-oracle value: `{robust_oracle_value}`.",
        f"- Decision: {decision}",
        "",
        "## Per-strategy CV against fixed true H15",
        "",
        "| strategy | dataset | robust label counts by state | oracle H counts | oracle physΔ/tol | oracle dec save | LS pass/save/physΔ | LC pass/save/physΔ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, a in analyses.items():
        counts = (a.get("debug") or {}).get("label_counts_by_state")
        for dname, res in (a.get("results") or {}).items():
            oracle = ((res.get("baselines") or {}).get("oracle_H10H15") or {})
            ls = ((res.get("state_observable_nested_cv") or {}).get("leave_state_out") or {})
            lc = ((res.get("state_observable_nested_cv") or {}).get("leave_case_out") or {})
            lines.append("| `%s` | `%s` | `%s` | `%s` | %.6g/%.6g | %.6g | `%s`/%.6g/%.6g | `%s`/%.6g/%.6g |" % (
                strategy, dname, counts, oracle.get("horizon_counts"), sf(oracle.get("physical_delta_vs_H15")), sf(oracle.get("physical_tolerance_vs_H15")), sf(oracle.get("decision_relative_saving_vs_H15")), bool(ls.get("core_pass_5pct")), sf(ls.get("decision_relative_saving_vs_H15")), sf(ls.get("physical_delta_vs_H15")), bool(lc.get("core_pass_5pct")), sf(lc.get("decision_relative_saving_vs_H15")), sf(lc.get("physical_delta_vs_H15")),
            ))
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A robust-label selector can only move to a closed-loop overhead smoke if a terminal-independent target passes leave-state and leave-case gates against fixed true H15 using online observations. Pooled-only/risk-only wins remain development diagnostics and do not open validation64 or test.",
        "",
        f"Backup request after this diagnostic: `{rel(req)}`.",
    ]
    summary = OUT_DIR / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(req, {
        "requested_utc": created.isoformat(),
        "reason": "backup robust-terminal-label CV before selector smoke, simulations, training or refit",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(req)],
    })
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H robust terminal-label CV v0

UTC: {created.isoformat()}. Offline/no-simulation robust-terminal label diagnostic; validation64 and sealed test stayed closed. Pass summary: {pass_summary}; strong pass summary: {strong_summary}. Decision: {decision}. Artifacts: `{rel(summary)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), V0B_SCRIPT, STATECV_SCRIPT, CONSIST_DONE, STATECV_DONE, STATE_PATH, req]
    done = {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "headline": {"pass_summary": pass_summary, "strong_pass_summary": strong_summary, "robust_oracle_value_all_profiles": robust_oracle_value, "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
