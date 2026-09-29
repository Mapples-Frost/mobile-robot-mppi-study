#!/usr/bin/env python3
"""Fast offline safety-gated robust-label selector CV diagnostic (v0b).

Development-only and no simulation/training/refit/validation/test access.

This is a bounded performance repair after v0 timed out at 600 s without outputs.
It avoids the expensive nested hyperparameter-selection loop and evaluates a
small predeclared set of high-precision H10-abstention rules by ordinary
leave-state and leave-case CV. Because configs are compared on development data,
any pass is diagnostic only; it can motivate a later optimized/nested or fresh
source-confirmation experiment, not deployment or validation64/test access.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_safety_gated_selector_cv_v0b_fast"
STAMP = "20260929T0950Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
BASE_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_safety_gated_selector_cv_v0.py"
FAILED_V0_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T091335_9e8e1e4e/registry.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-safety-gated-selector-cv-v0b-fast-{STAMP}"
MIN_SAVE = 0.05
STRONG_SAVE = 0.10


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clean(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Path):
        return rel(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, Mapping):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    return v


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


def sf(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def finite(vals: Iterable[Any]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in vals:
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


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {rel(path)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def cfg_name(cfg: Mapping[str, Any]) -> str:
    if cfg.get("family") == "all_h15":
        return "all_h15"
    return "guard_%s_k%s_q%s_m%s" % (cfg.get("mode"), cfg.get("k"), cfg.get("positive_radius_quantile"), cfg.get("negative_margin"))


def candidate_configs() -> List[Dict[str, Any]]:
    # Compact, predeclared high-precision subset of v0's 17 configs. The default
    # all-H15 abstention arm is included so the diagnostic can show no-switching
    # if positive support is too weak.
    return [
        {"family": "all_h15"},
        {"family": "positive_support_knn_guard", "mode": "raw", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
        {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
        {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 3, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
        {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.75, "negative_margin": 1.25},
    ]


def false_positive_rows(rows: Sequence[Mapping[str, Any]], preds: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r, p in zip(rows, preds):
        if int(p) == 10 and int(r.get("label", 15)) != 10:
            out.append({
                "sample_id": r.get("sample_id"),
                "state_key": r.get("state_key"),
                "case": r.get("case"),
                "source": r.get("source"),
                "terminal_profile": r.get("terminal_profile"),
                "h10_minus_h15_physical": sf(r.get("h10_minus_h15_physical")),
                "h10_safe": bool((r.get("h10") or {}).get("safe")),
            })
    return out


def cv_config(base: Any, v0b: Any, rows: Sequence[Mapping[str, Any]], fold_key: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    folds = sorted({str(r.get(fold_key)) for r in rows})
    if len(folds) < 3:
        return {"skipped": f"too_few_{fold_key}_folds", "fold_key": fold_key, "folds": len(folds), "n": len(rows), "config_name": cfg_name(cfg), "config": dict(cfg)}
    preds: Dict[str, int] = {}
    for fold in folds:
        train = [r for r in rows if str(r.get(fold_key)) != fold]
        test = [r for r in rows if str(r.get(fold_key)) == fold]
        model = base.fit_model(train, cfg)
        for r, p in zip(test, base.predict(model, test)):
            preds[str(r["sample_id"])] = int(p)
    ordered = [r for r in rows if str(r["sample_id"]) in preds]
    pred_seq = [preds[str(r["sample_id"])] for r in ordered]
    ev = base.eval_policy(v0b, ordered, pred_seq, require_nonconstant=True)
    fps = false_positive_rows(ordered, pred_seq)
    ev.update({
        "config_name": cfg_name(cfg),
        "config": dict(cfg),
        "fold_key": fold_key,
        "folds": len(folds),
        "h10_false_positive_count": len(fps),
        "h10_false_positive_physical_summary": finite(x["h10_minus_h15_physical"] for x in fps),
        "h10_false_positive_preview": fps[:8],
    })
    return ev


def admissible(ev: Mapping[str, Any]) -> bool:
    return bool(sf(ev.get("unsafe_chosen_groups"), 999.0) == 0 and sf(ev.get("physical_delta_vs_H15"), 1e9) <= sf(ev.get("physical_tolerance_vs_H15"), 0.0))


def combined_score(ls: Mapping[str, Any], lc: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float]:
    both_safe = admissible(ls) and admissible(lc)
    min_save = min(sf(ls.get("decision_relative_saving_vs_H15"), -1.0), sf(lc.get("decision_relative_saving_vs_H15"), -1.0))
    fp = sf(ls.get("h10_false_positive_count"), 999.0) + sf(lc.get("h10_false_positive_count"), 999.0)
    max_phys_ratio = max(sf(ls.get("physical_delta_vs_H15"), 1e9) / max(sf(ls.get("physical_tolerance_vs_H15"), 1.0), 1e-9), sf(lc.get("physical_delta_vs_H15"), 1e9) / max(sf(lc.get("physical_tolerance_vs_H15"), 1.0), 1e-9))
    return (1.0 if both_safe else 0.0, 1.0 if (ls.get("core_pass_5pct") and lc.get("core_pass_5pct")) else 0.0, min_save, -fp, -max_phys_ratio, 1.0 if (ls.get("nonconstant") and lc.get("nonconstant")) else 0.0)


def analyze_dataset(base: Any, v0b: Any, name: str, rows: Sequence[Mapping[str, Any]], cands: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(r) for r in rows if base.obs(r) is not None]
    out: Dict[str, Any] = {
        "dataset": name,
        "n": len(rows),
        "state_count": len({r["state_key"] for r in rows}),
        "case_count": len({r["case"] for r in rows}),
        "label_counts": {"10": sum(1 for r in rows if int(r["label"]) == 10), "15": sum(1 for r in rows if int(r["label"]) == 15)},
        "baselines": {
            "fixed_H15": base.eval_policy(v0b, rows, [15] * len(rows), require_nonconstant=False),
            "fixed_H10": base.eval_policy(v0b, rows, [10] * len(rows), require_nonconstant=False),
            "oracle_H10H15": base.eval_policy(v0b, rows, [int(r["label"]) for r in rows], require_nonconstant=False),
        },
    }
    per_config = []
    for cfg in cands:
        ls = cv_config(base, v0b, rows, "state_key", cfg)
        lc = cv_config(base, v0b, rows, "case", cfg)
        per_config.append({
            "config_name": cfg_name(cfg),
            "config": dict(cfg),
            "leave_state_out": ls,
            "leave_case_out": lc,
            "same_config_core5_both": bool(ls.get("core_pass_5pct") and lc.get("core_pass_5pct")),
            "same_config_core10_both": bool(ls.get("core_pass_10pct") and lc.get("core_pass_10pct")),
            "score": combined_score(ls, lc),
        })
    best = max(per_config, key=lambda x: tuple(x["score"])) if per_config else None
    out["per_config_cv"] = per_config
    out["same_config_core5_passes"] = [x["config_name"] for x in per_config if x.get("same_config_core5_both")]
    out["same_config_core10_passes"] = [x["config_name"] for x in per_config if x.get("same_config_core10_both")]
    if best:
        out["best_by_safety_score"] = {
            "config_name": best["config_name"],
            "score": best["score"],
            "leave_state": {k: best["leave_state_out"].get(k) for k in ("core_pass_5pct", "core_pass_10pct", "horizon_counts", "decision_relative_saving_vs_H15", "solver_relative_saving_vs_H15", "physical_delta_vs_H15", "physical_tolerance_vs_H15", "unsafe_chosen_groups", "h10_false_positive_count")},
            "leave_case": {k: best["leave_case_out"].get(k) for k in ("core_pass_5pct", "core_pass_10pct", "horizon_counts", "decision_relative_saving_vs_H15", "solver_relative_saving_vs_H15", "physical_delta_vs_H15", "physical_tolerance_vs_H15", "unsafe_chosen_groups", "h10_false_positive_count")},
        }
    return out


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
    if not BASE_SCRIPT.exists():
        raise RuntimeError(f"missing base script: {rel(BASE_SCRIPT)}")
    base = import_module(BASE_SCRIPT, "safety_v0_base_fast")
    robust_done = base.completed_ok(base.ROBUST_DONE)
    terminal_done = base.completed_ok(base.TERMINAL_CONSIST_DONE)
    v0b = base.import_module(base.V0B_SCRIPT, "h10h15_v0b_for_safety_fast")
    robust = base.import_module(base.ROBUST_SCRIPT, "robust_terminal_for_safety_fast")
    samples, _aux = v0b.load_samples()
    cands = candidate_configs()
    strategies = ["terminal_agreement_only", "aggregate_terminal_robust", "strict_per_profile_regret2"]
    analyses: Dict[str, Any] = {}
    for strategy in strategies:
        repaired, debug = robust.apply_strategy(samples, strategy)
        datasets = {
            "all_profiles": repaired,
            "risk_anchor_source_all_profiles": [r for r in repaired if r.get("source") == "risk_anchor"],
            "matched_terminal_only": [r for r in repaired if r.get("terminal_profile") == "matched_terminal"],
            "shared_h15_terminal_only": [r for r in repaired if r.get("terminal_profile") == "shared_h15_terminal"],
        }
        results = {d: analyze_dataset(base, v0b, d, rows, cands) for d, rows in datasets.items() if rows}
        analyses[strategy] = {
            "debug": debug,
            "results": results,
            "same_config_pass_datasets": [d for d, r in results.items() if r.get("same_config_core5_passes")],
            "same_config_strong_pass_datasets": [d for d, r in results.items() if r.get("same_config_core10_passes")],
        }
    pass_summary = {s: analyses[s]["same_config_pass_datasets"] for s in strategies}
    strong_summary = {s: analyses[s]["same_config_strong_pass_datasets"] for s in strategies}
    all_profile_pass = [s for s in strategies if "all_profiles" in pass_summary[s]]
    terminal_fixed_pass = [s for s in strategies for d in pass_summary[s] if d in ("matched_terminal_only", "shared_h15_terminal_only")]
    risk_only_pass = [s for s in strategies if "risk_anchor_source_all_profiles" in pass_summary[s] and "all_profiles" not in pass_summary[s]]
    if all_profile_pass or terminal_fixed_pass:
        decision = "diagnostic same-config safety-gated CV passes beyond risk-only; next run an optimized nested/fresh-source confirmation before any selector rollout or validation"
    elif risk_only_pass:
        decision = "diagnostic same-config safety-gated CV still passes only risk-anchor data; prioritize source-independent state coverage/value-calibration before rollout"
    else:
        decision = "even high-precision abstention cannot recover deployable compute savings from current observations; prioritize representation/value calibration and additional controlled source-independent states"
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_SAFETY_GATED_SELECTOR_CV_V0B_FAST_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_fast_safety_gated_selector_cv_not_validation_not_test",
        "replaces_failed_timeout_run": rel(FAILED_V0_REGISTRY),
        "timeout_repair": "remove nested hyperparameter selection; evaluate five predeclared configs directly under leave-state and leave-case CV",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_configs": cands,
        "candidate_configs_per_fold": len(cands),
        "inputs": {"base_safety_script": rel(BASE_SCRIPT), "robust_completed": rel(base.ROBUST_DONE), "terminal_consistency_completed": rel(base.TERMINAL_CONSIST_DONE), "failed_v0_registry": rel(FAILED_V0_REGISTRY)},
        "prerequisite_headlines": {"robust_terminal_label_cv": robust_done.get("headline"), "terminal_consistency": terminal_done.get("headline")},
        "pass_summary": pass_summary,
        "strong_pass_summary": strong_summary,
        "all_profile_pass_strategies": all_profile_pass,
        "terminal_fixed_pass_strategies": terminal_fixed_pass,
        "risk_only_pass_strategies": risk_only_pass,
        "decision": decision,
        "analyses": analyses,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle true-variable-H safety-gated selector CV v0b fast",
        "",
        f"UTC: `{created.isoformat()}`. Offline/no-simulation diagnostic; validation64 and sealed test remain closed.",
        "",
        "## Headline",
        "",
        f"- Repaired timeout from v0 run: `{rel(FAILED_V0_REGISTRY)}`.",
        f"- Predeclared configs per fold: `{len(cands)}`.",
        f"- Pass summary: `{pass_summary}`.",
        f"- Strong pass summary: `{strong_summary}`.",
        f"- All-profile pass strategies: `{all_profile_pass}`.",
        f"- Terminal-fixed pass strategies: `{terminal_fixed_pass}`.",
        f"- Decision: {decision}",
        "",
        "## Best same-config diagnostic CV by dataset",
        "",
        "| strategy | dataset | labels | oracle dec save | oracle physΔ/tol | best cfg | LS pass/save/physΔ/fp | LC pass/save/physΔ/fp |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for strategy, a in analyses.items():
        for dname, res in (a.get("results") or {}).items():
            oracle = ((res.get("baselines") or {}).get("oracle_H10H15") or {})
            best = res.get("best_by_safety_score") or {}
            ls = best.get("leave_state") or {}
            lc = best.get("leave_case") or {}
            lines.append("| `%s` | `%s` | `%s` | %.6g | %.6g/%.6g | `%s` | `%s`/%.6g/%.6g/%s | `%s`/%.6g/%.6g/%s |" % (
                strategy, dname, res.get("label_counts"), sf(oracle.get("decision_relative_saving_vs_H15")), sf(oracle.get("physical_delta_vs_H15")), sf(oracle.get("physical_tolerance_vs_H15")), best.get("config_name"),
                bool(ls.get("core_pass_5pct")), sf(ls.get("decision_relative_saving_vs_H15")), sf(ls.get("physical_delta_vs_H15")), ls.get("h10_false_positive_count"),
                bool(lc.get("core_pass_5pct")), sf(lc.get("decision_relative_saving_vs_H15")), sf(lc.get("physical_delta_vs_H15")), lc.get("h10_false_positive_count"),
            ))
    lines += [
        "",
        "## Interpretation rule",
        "",
        "This direct-CV diagnostic is not a deployable model-selection result. A pass only motivates an optimized/fresh confirmation; a fail supports representation/value-calibration or new controlled state coverage rather than another unchanged label-density sweep.",
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
        "reason": "backup fast safety-gated selector CV outputs and timeout repair provenance before simulations/training/refit",
        "backup_required_before_more_simulations": True,
        "backup_required_before_training_or_refit": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(req), rel(FAILED_V0_REGISTRY)],
    })
    append_docs(f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H safety-gated selector CV v0b fast

UTC: {created.isoformat()}. Offline/no-simulation fast safety-gated robust-label selector diagnostic; validation64 and sealed test stayed closed. This repairs the v0 timeout by removing nested hyperparameter selection and evaluating five predeclared configs directly. Pass summary: {pass_summary}; strong pass summary: {strong_summary}. Decision: {decision}. Artifacts: `{rel(summary)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), BASE_SCRIPT, FAILED_V0_REGISTRY, STATE_PATH, req]
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
        "headline": {"pass_summary": pass_summary, "strong_pass_summary": strong_summary, "all_profile_pass_strategies": all_profile_pass, "terminal_fixed_pass_strategies": terminal_fixed_pass, "risk_only_pass_strategies": risk_only_pass, "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
