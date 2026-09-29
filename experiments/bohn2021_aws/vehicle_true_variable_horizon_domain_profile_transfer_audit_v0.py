#!/usr/bin/env python3
"""Development-only source/terminal-profile transfer audit for true-variable-H selector.

Purpose
-------
The fast safety-gated selector diagnostic passed only the matched-terminal subset
and did not pass all-profiles. This audit tests whether that matched-terminal
signal transfers across development sources (oracle_bank <-> risk_anchor) and
terminal profiles without using validation64, sealed test, simulation, rollout,
training or refit.

Scientific question
-------------------
If a high-precision H10-abstention rule trained on one source/profile transfers
with no unsafe/physical regression and >=5% measured decision saving, the next
step can be an optimized/fresh-source confirmation. If it collapses to H15 or
causes false positives/physical loss, the bottleneck is state coverage / terminal
value calibration / objective representation rather than a deployable selector.
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
NAME = "vehicle_true_variable_horizon_domain_profile_transfer_audit_v0"
STAMP = "20260929T1000Z"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
BASE_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_safety_gated_selector_cv_v0.py"
ROBUST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_robust_terminal_label_cv_v0_20260929T0925Z/completed.json"
SAFETY_FAST_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_safety_gated_selector_cv_v0b_fast_20260929T0950Z/completed.json"
OUT_DIR = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE_PATH = ROOT / f"research_artifacts/aws_state/{NAME}_{STAMP}.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-domain-profile-transfer-audit-v0-{STAMP}"
MIN_SAVE = 0.05
STRONG_SAVE = 0.10


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {rel(path)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


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


def cfgs() -> List[Dict[str, Any]]:
    return [
        {"family": "all_h15"},
        {"family": "positive_support_knn_guard", "mode": "raw", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
        {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
        {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 3, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.50, "negative_margin": 1.25},
        {"family": "positive_support_knn_guard", "mode": "raw_abs_l2", "k": 1, "min_positive_fraction": 1.0, "min_positive_support": 2, "positive_radius_quantile": 0.75, "negative_margin": 1.25},
    ]


def cfg_name(cfg: Mapping[str, Any]) -> str:
    if cfg.get("family") == "all_h15":
        return "all_h15"
    return "guard_%s_k%s_q%s_m%s" % (cfg.get("mode"), cfg.get("k"), cfg.get("positive_radius_quantile"), cfg.get("negative_margin"))


def admissible(ev: Mapping[str, Any]) -> bool:
    return bool(sf(ev.get("unsafe_chosen_groups"), 999.0) == 0 and sf(ev.get("physical_delta_vs_H15"), 1e9) <= sf(ev.get("physical_tolerance_vs_H15"), 0.0))


def score(ev: Mapping[str, Any], confusion: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float]:
    saving = sf(ev.get("decision_relative_saving_vs_H15"), -1.0)
    tol = max(sf(ev.get("physical_tolerance_vs_H15"), 1.0), 1e-9)
    phys = sf(ev.get("physical_delta_vs_H15"), 1e9)
    return (
        1.0 if admissible(ev) else 0.0,
        1.0 if ev.get("core_pass_5pct") else 0.0,
        saving,
        -sf(confusion.get("false_positive_count"), 999.0),
        -max(0.0, phys / tol),
        sf(confusion.get("true_positive_count"), 0.0),
    )


def confusion(rows: Sequence[Mapping[str, Any]], preds: Sequence[int]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    fp_phys: List[float] = []
    fn_save: List[float] = []
    pred10_rows: List[Dict[str, Any]] = []
    for r, p in zip(rows, preds):
        lab = int(r.get("label", 15))
        if int(p) == 10 and lab == 10:
            tp += 1
        elif int(p) == 10 and lab != 10:
            fp += 1
            fp_phys.append(sf(r.get("h10_minus_h15_physical")))
        elif int(p) != 10 and lab == 10:
            fn += 1
            fn_save.append(sf(r.get("decision_relative_saving_h10_vs_h15"), 0.0))
        else:
            tn += 1
        if int(p) == 10:
            pred10_rows.append({
                "sample_id": r.get("sample_id"),
                "state_key": r.get("state_key"),
                "case": r.get("case"),
                "source": r.get("source"),
                "terminal_profile": r.get("terminal_profile"),
                "label": lab,
                "h10_minus_h15_physical": sf(r.get("h10_minus_h15_physical")),
                "h10_safe": bool((r.get("h10") or {}).get("safe")),
            })
    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "false_positive_h10_minus_h15_physical": finite(fp_phys),
        "false_negative_decision_saving_summary": finite(fn_save),
        "predicted_h10_preview": pred10_rows[:10],
    }


def support_preview(base: Any, train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], preds: Sequence[int], limit: int = 12) -> List[Dict[str, Any]]:
    if cfg.get("family") == "all_h15":
        return []
    model = base.fit_model(train, cfg)
    entries = list(model.get("entries") or [])
    out: List[Dict[str, Any]] = []
    for r, p in zip(test, preds):
        z = base.feat(r, model.get("std") or {})
        if z is None or not entries:
            dpos = dneg = None
        else:
            pos = [base.dist(z, e["z"]) for e in entries if int(e.get("label", 15)) == 10]
            neg = [base.dist(z, e["z"]) for e in entries if int(e.get("label", 15)) != 10]
            dpos = min(pos) if pos else None
            dneg = min(neg) if neg else None
        # Prioritize false positives and positive-label abstentions.
        lab = int(r.get("label", 15))
        include = (int(p) == 10 and lab != 10) or (int(p) != 10 and lab == 10)
        if include or len(out) < 3:
            out.append({
                "sample_id": r.get("sample_id"),
                "state_key": r.get("state_key"),
                "case": r.get("case"),
                "source": r.get("source"),
                "terminal_profile": r.get("terminal_profile"),
                "label": lab,
                "pred": int(p),
                "dpos": dpos,
                "dneg": dneg,
                "positive_radius": model.get("positive_radius"),
                "h10_minus_h15_physical": sf(r.get("h10_minus_h15_physical")),
            })
        if len(out) >= limit:
            break
    return out


def eval_transfer(base: Any, v0b: Any, name: str, train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train = [dict(r) for r in train if base.obs(r) is not None]
    test = [dict(r) for r in test if base.obs(r) is not None]
    out: Dict[str, Any] = {
        "name": name,
        "train_n": len(train),
        "test_n": len(test),
        "train_sources": sorted({str(r.get("source")) for r in train}),
        "test_sources": sorted({str(r.get("source")) for r in test}),
        "train_profiles": sorted({str(r.get("terminal_profile")) for r in train}),
        "test_profiles": sorted({str(r.get("terminal_profile")) for r in test}),
        "train_labels": {"10": sum(1 for r in train if int(r.get("label", 15)) == 10), "15": sum(1 for r in train if int(r.get("label", 15)) == 15)},
        "test_labels": {"10": sum(1 for r in test if int(r.get("label", 15)) == 10), "15": sum(1 for r in test if int(r.get("label", 15)) == 15)},
    }
    if not train or not test:
        out["skipped"] = "empty_train_or_test"
        return out
    out["baselines_on_test"] = {
        "fixed_H15": base.eval_policy(v0b, test, [15] * len(test), require_nonconstant=False),
        "fixed_H10": base.eval_policy(v0b, test, [10] * len(test), require_nonconstant=False),
        "oracle_H10H15": base.eval_policy(v0b, test, [int(r.get("label", 15)) for r in test], require_nonconstant=False),
    }
    per_cfg: List[Dict[str, Any]] = []
    for cfg in candidates:
        model = base.fit_model(train, cfg)
        preds = base.predict(model, test)
        ev = base.eval_policy(v0b, test, preds, require_nonconstant=True)
        conf = confusion(test, preds)
        per_cfg.append({
            "config_name": cfg_name(cfg),
            "config": dict(cfg),
            "eval": ev,
            "confusion": conf,
            "score": score(ev, conf),
            "support_preview": support_preview(base, train, test, cfg, preds),
        })
    best = max(per_cfg, key=lambda x: tuple(x["score"]))
    out["per_config"] = per_cfg
    out["passing_configs_5pct"] = [x["config_name"] for x in per_cfg if x["eval"].get("core_pass_5pct")]
    out["passing_configs_10pct"] = [x["config_name"] for x in per_cfg if x["eval"].get("core_pass_10pct")]
    out["best"] = {
        "config_name": best["config_name"],
        "score": best["score"],
        "eval": {k: best["eval"].get(k) for k in ("core_pass_5pct", "core_pass_10pct", "horizon_counts", "decision_relative_saving_vs_H15", "solver_relative_saving_vs_H15", "physical_delta_vs_H15", "physical_tolerance_vs_H15", "unsafe_chosen_groups", "nonconstant")},
        "confusion": best["confusion"],
        "support_preview": best["support_preview"],
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
    if not SAFETY_FAST_DONE.exists():
        raise RuntimeError(f"missing prerequisite safety fast completed marker: {rel(SAFETY_FAST_DONE)}")
    base = import_module(BASE_SCRIPT, "safety_selector_base_for_transfer")
    v0b = base.import_module(base.V0B_SCRIPT, "h10h15_v0b_for_transfer")
    robust = base.import_module(base.ROBUST_SCRIPT, "robust_terminal_for_transfer")
    samples, _aux = v0b.load_samples()
    candidates = cfgs()
    strategies = ["terminal_agreement_only", "aggregate_terminal_robust", "strict_per_profile_regret2"]
    analyses: Dict[str, Any] = {}
    for strategy in strategies:
        repaired, debug = robust.apply_strategy(samples, strategy)
        transfers: Dict[str, Any] = {}
        for profile_name, rows in {
            "all_profiles": repaired,
            "matched_terminal_only": [r for r in repaired if r.get("terminal_profile") == "matched_terminal"],
            "shared_h15_terminal_only": [r for r in repaired if r.get("terminal_profile") == "shared_h15_terminal"],
        }.items():
            oracle_rows = [r for r in rows if r.get("source") == "oracle_bank"]
            risk_rows = [r for r in rows if r.get("source") == "risk_anchor"]
            transfers[f"{profile_name}:train_oracle_test_risk"] = eval_transfer(base, v0b, f"{profile_name}:train_oracle_test_risk", oracle_rows, risk_rows, candidates)
            transfers[f"{profile_name}:train_risk_test_oracle"] = eval_transfer(base, v0b, f"{profile_name}:train_risk_test_oracle", risk_rows, oracle_rows, candidates)
        matched = [r for r in repaired if r.get("terminal_profile") == "matched_terminal"]
        shared = [r for r in repaired if r.get("terminal_profile") == "shared_h15_terminal"]
        transfers["profile_transfer:train_matched_test_shared"] = eval_transfer(base, v0b, "profile_transfer:train_matched_test_shared", matched, shared, candidates)
        transfers["profile_transfer:train_shared_test_matched"] = eval_transfer(base, v0b, "profile_transfer:train_shared_test_matched", shared, matched, candidates)
        analyses[strategy] = {"debug_label_counts": debug.get("label_counts_by_state"), "transfers": transfers}
    pass_summary: Dict[str, List[str]] = {}
    strong_summary: Dict[str, List[str]] = {}
    nontrivial_passes: List[Dict[str, Any]] = []
    for strategy, a in analyses.items():
        pass_summary[strategy] = []
        strong_summary[strategy] = []
        for tname, tr in a["transfers"].items():
            best = tr.get("best") or {}
            ev = best.get("eval") or {}
            if ev.get("core_pass_5pct"):
                pass_summary[strategy].append(tname)
                nontrivial_passes.append({"strategy": strategy, "transfer": tname, "best": best})
            if ev.get("core_pass_10pct"):
                strong_summary[strategy].append(tname)
    source_transfer_passes = [p for p in nontrivial_passes if "train_oracle_test_risk" in p["transfer"] or "train_risk_test_oracle" in p["transfer"]]
    profile_transfer_passes = [p for p in nontrivial_passes if p["transfer"].startswith("profile_transfer")]
    if source_transfer_passes:
        decision = "At least one source-held-out transfer retained a safe >=5% measured decision saving; after backup, run a small fresh-source confirmation or optimized nested source-held-out selector before any validation rollout."
    elif profile_transfer_passes:
        decision = "Only terminal-profile transfer retained safe compute saving; source transfer failed. Prioritize source-independent state acquisition/representation calibration before selector rollout."
    else:
        decision = "No source/profile transfer produced a safe >=5% nonconstant selector; matched-terminal direct-CV pass is not source-general. Prioritize controlled source-independent state coverage plus terminal/value calibration, not selector rollout or another unchanged label-density sweep."
    created = now_utc()
    req = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_DOMAIN_PROFILE_TRANSFER_AUDIT_V0_%s.json" % created.isoformat().replace("-", "").replace(":", "").replace("+00:00", "+0000"))
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_SUPERVISOR_EVENT).total_seconds(),
        "method": NAME,
        "classification": "development_offline_no_simulation_source_terminal_transfer_audit_not_validation_not_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_simulations": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "new_refit_steps": 0,
        "candidate_configs": candidates,
        "inputs": {"base_script": rel(BASE_SCRIPT), "robust_done": rel(ROBUST_DONE), "safety_fast_done": rel(SAFETY_FAST_DONE)},
        "pass_summary": pass_summary,
        "strong_pass_summary": strong_summary,
        "source_transfer_pass_count": len(source_transfer_passes),
        "profile_transfer_pass_count": len(profile_transfer_passes),
        "decision": decision,
        "analyses": analyses,
        "backup_request_after_diagnostic": rel(req),
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle true-variable-H domain/profile transfer audit v0",
        "",
        f"UTC: `{created.isoformat()}`. Development-only offline diagnostic; validation64 and sealed test stayed closed.",
        "",
        "## Headline",
        "",
        f"- Candidate configs: `{len(candidates)}` (same fast safety-gated subset).",
        f"- Pass summary: `{pass_summary}`.",
        f"- Strong pass summary: `{strong_summary}`.",
        f"- Source-transfer pass count: `{len(source_transfer_passes)}`.",
        f"- Profile-transfer pass count: `{len(profile_transfer_passes)}`.",
        f"- Decision: {decision}",
        "",
        "## Best transfer rows",
        "",
        "| strategy | transfer | train labels | test labels | oracle save | best cfg | pass/save/physΔ/tol/unsafe | confusion TP/FP/FN/TN |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for strategy, a in analyses.items():
        for tname, tr in a["transfers"].items():
            if tr.get("skipped"):
                continue
            oracle = ((tr.get("baselines_on_test") or {}).get("oracle_H10H15") or {})
            best = tr.get("best") or {}
            ev = best.get("eval") or {}
            conf = best.get("confusion") or {}
            lines.append("| `%s` | `%s` | `%s` | `%s` | %.6g | `%s` | `%s`/%.6g/%.6g/%.6g/%s | %s/%s/%s/%s |" % (
                strategy, tname, tr.get("train_labels"), tr.get("test_labels"), sf(oracle.get("decision_relative_saving_vs_H15")), best.get("config_name"),
                bool(ev.get("core_pass_5pct")), sf(ev.get("decision_relative_saving_vs_H15")), sf(ev.get("physical_delta_vs_H15")), sf(ev.get("physical_tolerance_vs_H15")), ev.get("unsafe_chosen_groups"),
                conf.get("true_positive_count"), conf.get("false_positive_count"), conf.get("false_negative_count"), conf.get("true_negative_count"),
            ))
    lines += [
        "",
        "## Interpretation",
        "",
        "This audit uses existing development samples only. It is not validation. Source-held-out failure means the current online-observable selector evidence is still too source/profile-specific for rollout; the next useful intervention is fresh controlled state coverage and terminal/value calibration rather than another unchanged selector CV.",
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
        "reason": "backup source/terminal transfer audit outputs and source before further simulation/training/refit",
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
## 2026-09-29 vehicle true-variable-H domain/profile transfer audit v0

UTC: {created.isoformat()}. Development-only offline source/profile transfer diagnostic; validation64 and sealed test stayed closed. Pass summary: {pass_summary}; strong pass summary: {strong_summary}; source-transfer passes: {len(source_transfer_passes)}; profile-transfer passes: {len(profile_transfer_passes)}. Decision: {decision}. Artifacts: `{rel(summary)}`, `{rel(OUT_DIR / 'raw.json')}`.
""")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [Path(__file__).resolve(), BASE_SCRIPT, ROBUST_DONE, SAFETY_FAST_DONE, STATE_PATH, req]
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
        "headline": {"pass_summary": pass_summary, "strong_pass_summary": strong_summary, "source_transfer_pass_count": len(source_transfer_passes), "profile_transfer_pass_count": len(profile_transfer_passes), "decision": decision},
        "backup_request": rel(req),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, done)
    print(json.dumps({"completed": rel(done_path), "summary": rel(summary), "headline": done["headline"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
