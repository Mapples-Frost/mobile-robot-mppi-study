#!/usr/bin/env python3
"""Fast runtime repair for v20 H12/H15 selector/refit.

v20 timed out under the 600 s bound after writing its pre-outcome protocol but
before raw/completed outputs.  This v20b script preserves the same scientific
hypothesis, inputs, strict splits, H12/H15 labels, uncertainty fallback, and full
1728-config grid.  It changes only computation: split-level model scores are
cached for the threshold-independent parts of each config, so threshold sweeps do
not recompute scalers/vectors/neighbours.

Development-only IMPROVED offline refit: no MPC simulation, no validation64, no
sealed test, and no gradient/RL training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_selector_refit_v20.py"
spec = importlib.util.spec_from_file_location("vehicle_true_variable_horizon_h12_h15_selector_refit_v20_mod", str(BASE_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import v20 base")
v20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v20)  # type: ignore[union-attr]

NAME = "vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast"
STAMP = "20260930T0100Z"
SOURCE = Path(__file__).resolve()
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260930T0100_after_h12_h15_selector_refit_v20b_fast.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_SELECTOR_REFIT_V20B_FAST_{STAMP}.json"
MARKER = f"vehicle-true-variable-H-h12-h15-selector-refit-v20b-fast-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
V20_TIMEOUT_REGISTRY = ROOT / "research_artifacts/aws_runs/20260930T002053_fd203a2f/registry.json"
V20_TIMEOUT_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_selector_refit_v20_preoutcome_frozen_20260930T0045Z.json"

SHORT_H = v20.SHORT_H
REF_H = v20.REF_H
MIN_DECISION_SAVING = v20.MIN_DECISION_SAVING
FAMILIES = v20.FAMILIES


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sf(x: Any, default: float = 0.0) -> float:
    return v20.sf(x, default)


def si(x: Any, default: int = 0) -> int:
    return v20.si(x, default)


def clean(x: Any) -> Any:
    return v20.clean(x)


def read_json(path: Path) -> Any:
    return v20.read_json(path)


def write_json(path: Path, value: Any) -> None:
    v20.write_json(path, value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def mean(xs: Sequence[float]) -> float:
    return math.fsum(float(x) for x in xs) / len(xs) if xs else 0.0


def summarize_timeout() -> Dict[str, Any]:
    out: Dict[str, Any] = {"observed": V20_TIMEOUT_REGISTRY.exists(), "registry": rel(V20_TIMEOUT_REGISTRY)}
    if V20_TIMEOUT_REGISTRY.exists():
        try:
            reg = read_json(V20_TIMEOUT_REGISTRY)
            out.update({"exit_status": reg.get("exit_status"), "runtime_seconds": reg.get("runtime_seconds"), "script_sha256": reg.get("script_sha256")})
        except Exception as exc:
            out["read_error"] = str(exc)
    if V20_TIMEOUT_PROTOCOL.exists():
        out["protocol"] = rel(V20_TIMEOUT_PROTOCOL)
        out["protocol_sha256"] = sha256(V20_TIMEOUT_PROTOCOL)
    return out


class ScoreCache:
    def __init__(self, rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]):
        self.rows = list(rows)
        self.features = features
        self.cache: Dict[Tuple[str, str, int, float, float, float, int], Dict[str, Any]] = {}
        self.fit_count = 0

    def scores(self, split_id: str, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
        family = str(cfg["family"])
        key = (
            split_id,
            family,
            si(cfg.get("k"), 1),
            sf(cfg.get("support_q"), 0.75),
            sf(cfg.get("support_mult"), 1.0),
            sf(cfg.get("risk_prior_alpha"), 0.5),
            si(cfg.get("min_cat_train_for_h12"), 1),
        )
        if key in self.cache:
            return self.cache[key]
        fdict = v20.feature_subset(self.features, family)
        names, means, stds = v20.train_scaler(train_rows, fdict)
        train_vecs = {v20.row_key(r): v20.vector(r, fdict, names, means, stds) for r in train_rows}
        pos_rows = [r for r in train_rows if bool(r["h12_beneficial_vs_h15"])]
        cat_rows = [r for r in train_rows if bool(r["h12_catastrophic_vs_h15"])]
        radius = v20.support_radius(pos_rows, train_vecs, cfg)
        min_cat = si(cfg.get("min_cat_train_for_h12"), 1)
        row_scores: Dict[str, Any] = {}
        for er in eval_rows:
            if not train_rows or not pos_rows or len(cat_rows) < min_cat:
                row_scores[v20.row_key(er)] = {
                    "fallback_h15": True,
                    "reason": "insufficient_positive_or_catastrophic_training_support",
                    "feature_count": len(names),
                    "train_cat_count": len(cat_rows),
                    "train_pos_count": len(pos_rows),
                    "risk_hat": 1.0,
                    "pred_gain_s": -1e9,
                    "pred_phys_delta": 1e9,
                    "dpos": 1e9,
                    "dcat": 1e9,
                    "support_radius": radius,
                    "support_ok": False,
                }
                continue
            ev = v20.vector(er, fdict, names, means, stds)
            neigh = sorted([(v20.dist(ev, train_vecs[v20.row_key(tr)]), tr) for tr in train_rows], key=lambda z: (z[0], v20.row_key(z[1])))
            pos_neigh = sorted([(v20.dist(ev, train_vecs[v20.row_key(tr)]), tr) for tr in pos_rows], key=lambda z: (z[0], v20.row_key(z[1])))
            cat_neigh = sorted([(v20.dist(ev, train_vecs[v20.row_key(tr)]), tr) for tr in cat_rows], key=lambda z: (z[0], v20.row_key(z[1])))
            kk = max(1, min(si(cfg.get("k"), 1), len(neigh)))
            near = neigh[:kk]
            weights = [1.0 / max(0.05, d) for d, _ in near]
            sw = math.fsum(weights) if weights else 1.0
            cat_w = math.fsum(w for w, (_, r) in zip(weights, near) if bool(r["h12_catastrophic_vs_h15"]))
            prior = sf(cfg.get("risk_prior_alpha"), 0.5)
            risk_hat = (cat_w + 0.5 * prior) / (sw + prior)
            kp = max(1, min(si(cfg.get("k"), 1), len(pos_neigh)))
            pnear = pos_neigh[:kp]
            pred_gain = mean([sf(r["decision_gain_h12_vs_h15_s"]) for _, r in pnear]) if pnear else -1e9
            pred_phys = mean([sf(r["phys_delta_h12_minus_h15"]) for _, r in pnear]) if pnear else 1e9
            dpos = pnear[-1][0] if pnear else 1e9
            dcat = cat_neigh[0][0] if cat_neigh else 1e9
            row_scores[v20.row_key(er)] = {
                "fallback_h15": False,
                "feature_count": len(names),
                "train_cat_count": len(cat_rows),
                "train_pos_count": len(pos_rows),
                "risk_hat": risk_hat,
                "pred_gain_s": pred_gain,
                "pred_phys_delta": pred_phys,
                "dpos": dpos,
                "dcat": dcat,
                "support_radius": radius,
                "support_ok": dpos <= radius,
            }
        obj = {"scores": row_scores, "feature_count": len(names), "train_rows": len(train_rows), "eval_rows": len(eval_rows)}
        self.cache[key] = obj
        self.fit_count += 1
        return obj


def choices_from_scores(scores: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, int]:
    choices: Dict[str, int] = {}
    for k, s in scores.items():
        if s.get("fallback_h15"):
            choices[k] = REF_H
            continue
        dpos = sf(s.get("dpos"), 1e9)
        dcat = sf(s.get("dcat"), 1e9)
        cat_guard_ok = dcat > sf(cfg.get("cat_guard_ratio"), 1.0) * max(dpos, 1e-9) + sf(cfg.get("cat_margin"), 0.0)
        choose = bool(
            s.get("support_ok")
            and cat_guard_ok
            and sf(s.get("risk_hat"), 1.0) <= sf(cfg.get("risk_max"), 0.5)
            and sf(s.get("pred_gain_s"), -1e9) >= sf(cfg.get("gain_min_s"), 0.0)
            and sf(s.get("pred_phys_delta"), 1e9) <= sf(cfg.get("phys_max"), 2.0)
        )
        choices[k] = SHORT_H if choose else REF_H
    return choices


def eval_split_cached(cache: ScoreCache, split_id: str, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    obj = cache.scores(split_id, train_rows, eval_rows, cfg)
    choices = choices_from_scores(obj["scores"], cfg)
    ev = v20.eval_choices(eval_rows, choices, v20.overhead_mode_for_family(str(cfg["family"])))
    ev["feature_count"] = obj["feature_count"]
    if include_scores:
        scored: Dict[str, Any] = {}
        for rk, s in obj["scores"].items():
            ss = dict(s)
            ss["cat_guard_ok"] = sf(s.get("dcat"), 1e9) > sf(cfg.get("cat_guard_ratio"), 1.0) * max(sf(s.get("dpos"), 1e9), 1e-9) + sf(cfg.get("cat_margin"), 0.0)
            ss["selected_h"] = choices[rk]
            scored[rk] = ss
        ev["prediction_scores"] = scored
    return ev


def eval_group_lobo_cached(cache: ScoreCache, rows: Sequence[Mapping[str, Any]], groups: Sequence[str], group_field: str, cfg: Mapping[str, Any], blocked_groups: Sequence[str] = (), include_scores: bool = False, split_prefix: str = "lobo") -> Dict[str, Any]:
    blocked = set(str(g) for g in blocked_groups)
    eval_groups = [g for g in groups if g not in blocked]
    holdouts: Dict[str, Any] = {}
    for hg in eval_groups:
        train = [r for r in rows if str(r[group_field]) not in blocked and str(r[group_field]) != hg]
        evrows = [r for r in rows if str(r[group_field]) == hg]
        sid = f"{split_prefix}|field={group_field}|blocked={','.join(sorted(blocked))}|holdout={hg}"
        holdouts[hg] = eval_split_cached(cache, sid, train, evrows, cfg, include_scores=include_scores)
    eval_rows = [r for r in rows if str(r[group_field]) in eval_groups]
    return {"holdouts": holdouts, "aggregate": v20.aggregate_from_details(eval_rows, holdouts)}


def nested_selection_cached(cache: ScoreCache, rows: Sequence[Mapping[str, Any]], cfgs: Sequence[Mapping[str, Any]], group_field: str) -> Tuple[Dict[str, Any], int]:
    groups = sorted({str(r[group_field]) for r in rows})
    outer: Dict[str, Any] = {}
    eval_count = 0
    for outer_group in groups:
        inner_rows = [r for r in rows if str(r[group_field]) != outer_group]
        inner_groups = sorted({str(r[group_field]) for r in inner_rows})
        candidates: List[Dict[str, Any]] = []
        for cfg in cfgs:
            lobo = eval_group_lobo_cached(cache, inner_rows, inner_groups, group_field, cfg, split_prefix=f"inner_outer={outer_group}")
            eval_count += len(inner_groups)
            candidates.append({"config_id": v20.cfg_id(cfg), "config": cfg, "summary": v20.summary(lobo)})
        selected = sorted(candidates, key=lambda x: v20.rank_summary(x["summary"]))[0]
        train_outer = inner_rows
        eval_outer = [r for r in rows if str(r[group_field]) == outer_group]
        outer_eval = eval_split_cached(cache, f"outer_eval|field={group_field}|holdout={outer_group}", train_outer, eval_outer, selected["config"], include_scores=True)
        eval_count += 1
        outer[outer_group] = {
            "selected_config_id": v20.cfg_id(selected["config"]),
            "selected_config": selected["config"],
            "inner_summary": selected["summary"],
            "outer_eval": outer_eval,
            "outer_train_cat_count": sum(1 for r in train_outer if r["h12_catastrophic_vs_h15"]),
            "outer_train_pos_count": sum(1 for r in train_outer if r["h12_beneficial_vs_h15"]),
        }
    # Aggregate mixed-family outer choices and already charged per-row overhead.
    fixed_phys = math.fsum(sf(r["h15_physical"]) for r in rows)
    fixed_dec = math.fsum(sf(r["h15_decision_sum_s"]) for r in rows)
    fixed_sol = math.fsum(sf(r["h15_solver_sum_s"]) for r in rows)
    tol_sum = math.fsum(sf(r["row_physical_tolerance"]) for r in rows)
    detail_by_id: Dict[str, Mapping[str, Any]] = {}
    for obj in outer.values():
        for d in obj["outer_eval"].get("details") or []:
            detail_by_id[str(d["candidate_id"])] = d
    pol_phys = pol_dec = pol_sol = extra_dec = extra_sol = 0.0
    counts: Counter[str] = Counter(); conf: Counter[str] = Counter(); bad: List[Dict[str, Any]] = []; details: List[Dict[str, Any]] = []
    for r in rows:
        d = detail_by_id.get(v20.row_key(r), {"selected_h": REF_H, "extra_probe_decision_s": 0.0, "extra_probe_solver_s": 0.0})
        h = int(d.get("selected_h", REF_H)); counts[str(h)] += 1
        pos = bool(r["h12_beneficial_vs_h15"]); cat = bool(r["h12_catastrophic_vs_h15"])
        ed = sf(d.get("extra_probe_decision_s")); es = sf(d.get("extra_probe_solver_s"))
        if h == SHORT_H:
            pol_phys += sf(r["h12_physical"]); pol_dec += sf(r["h12_decision_sum_s"]); pol_sol += sf(r["h12_solver_sum_s"])
            conf["TP" if pos else "FP"] += 1
            if cat:
                bad.append({"candidate_id": v20.row_key(r), "bank_id": r.get("bank_id"), "source_key": r.get("source_key"), "role": r.get("role"), "offset_from_center": r.get("offset_from_center"), "phys_delta": sf(r["phys_delta_h12_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h12_vs_h15_s"])})
        else:
            pol_phys += sf(r["h15_physical"]); pol_dec += sf(r["h15_decision_sum_s"]) + ed; pol_sol += sf(r["h15_solver_sum_s"]) + es
            extra_dec += ed; extra_sol += es
            conf["FN" if pos else "TN"] += 1
        details.append({"candidate_id": v20.row_key(r), "bank_id": r.get("bank_id"), "source_key": r.get("source_key"), "selected_h": h, "label_positive": pos, "catastrophic": cat, "extra_probe_decision_s": ed, "extra_probe_solver_s": es})
    phys_delta = pol_phys - fixed_phys
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    agg = {
        "rows": len(rows), "chosen_counts": dict(counts), "confusion": dict(conf),
        "policy_physical_sum": pol_phys, "fixed_H15_physical_sum": fixed_phys, "physical_delta_vs_fixed_H15": phys_delta, "physical_tolerance_sum": tol_sum, "physical_gate": phys_delta <= tol_sum,
        "policy_decision_sum_s": pol_dec, "fixed_H15_decision_sum_s": fixed_dec, "extra_probe_decision_sum_s": extra_dec, "decision_relative_saving_vs_fixed_H15": dec_save,
        "policy_solver_sum_s": pol_sol, "fixed_H15_solver_sum_s": fixed_sol, "extra_probe_solver_sum_s": extra_sol, "solver_relative_saving_vs_fixed_H15": sol_save,
        "catastrophic_false_positive_rows": bad, "pass_5pct_no_cat_fp": bool(dec_save >= MIN_DECISION_SAVING and phys_delta <= tol_sum and not bad), "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad), "details": details,
    }
    return {"group_field": group_field, "groups": groups, "outer": {g: {**o, "outer_eval": v20.compact_eval(o["outer_eval"])} for g, o in outer.items()}, "aggregate": agg}, eval_count


def run(args: argparse.Namespace) -> Dict[str, Any]:
    # Repoint v20 helpers that write summaries/docs to v20b paths.
    v20.NAME = NAME; v20.STAMP = STAMP; v20.SOURCE = SOURCE; v20.OUT = OUT; v20.PROTOCOL = PROTOCOL; v20.STATE = STATE; v20.BACKUP_REQUEST = BACKUP_REQUEST; v20.MARKER = MARKER
    created = now()
    rows, features, diag, hashes = v20.load_rows()
    hashes[rel(BASE_PATH)] = sha256(BASE_PATH)
    hashes[rel(SOURCE)] = sha256(SOURCE)
    timeout_diag = summarize_timeout()
    if V20_TIMEOUT_REGISTRY.exists(): hashes[rel(V20_TIMEOUT_REGISTRY)] = sha256(V20_TIMEOUT_REGISTRY)
    if V20_TIMEOUT_PROTOCOL.exists(): hashes[rel(V20_TIMEOUT_PROTOCOL)] = sha256(V20_TIMEOUT_PROTOCOL)
    cfgs = v20.cfg_grid()
    banks = sorted({str(r["bank_id"]) for r in rows})
    sources = sorted({str(r["source_key"]) for r in rows})
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_H12_H15_selector_refit_fast_runtime_repair_no_sim_no_validation_no_test",
        "runtime_repair_for": timeout_diag,
        "hypothesis": "Same as v20: a conservative true-H12/H15 selector may exploit v19 oracle opportunity while avoiding the localized fresh_v11 case05 high-cost H12 cluster under strict bank/source nested splits. v20b changes only computation/caching after v20 timeout.",
        "inputs": {"v19_raw": rel(v20.V19_RAW), "v19_completed": rel(v20.V19_DONE), "v19_protocol": rel(v20.V19_PROTOCOL)},
        "row_diagnostic_preoutcome": diag,
        "candidate_rows_are_repeats_averaged": True,
        "families": FAMILIES,
        "config_count": len(cfgs),
        "primary_safety_default": "min_cat_train_for_h12=1 for every config; if an outer training fold has zero H12-catastrophic examples, all rows in that outer fold are assigned H15 as an uncertainty veto",
        "strict_splits": ["leave_bank_nested", "leave_source_key_nested"],
        "decision_gate": {"both_strict_splits_pass5": True, "zero_catastrophic_H12_false_positives": True, "physical_gate": True, "minimum_measured_decision_saving_vs_fixed_H15": MIN_DECISION_SAVING},
        "overhead_semantics": "probe families add mean first-H12 decision/solver time when H12 is rejected and H15 is executed; static/history families have no measured online selector overhead yet",
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 250000, "cached_score_fits_cap": 10000, "offline_model_evaluations_cap": 250000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": args.backup_verified_commit,
        "input_hashes": hashes,
    }
    write_json(PROTOCOL, proto)

    cache = ScoreCache(rows, features)
    selector_evals = 0
    global_bank_items: List[Dict[str, Any]] = []
    global_source_items: List[Dict[str, Any]] = []
    for cfg in cfgs:
        lb = eval_group_lobo_cached(cache, rows, banks, "bank_id", cfg, split_prefix="global_bank")
        selector_evals += len(banks)
        ls = eval_group_lobo_cached(cache, rows, sources, "source_key", cfg, split_prefix="global_source")
        selector_evals += len(sources)
        global_bank_items.append({"config_id": v20.cfg_id(cfg), "config": cfg, "summary": v20.summary(lb), "aggregate": lb["aggregate"]})
        global_source_items.append({"config_id": v20.cfg_id(cfg), "config": cfg, "summary": v20.summary(ls), "aggregate": ls["aggregate"]})
    top_global_bank = sorted(global_bank_items, key=lambda x: v20.rank_summary(x["summary"]))[:20]
    top_global_source = sorted(global_source_items, key=lambda x: v20.rank_summary(x["summary"]))[:20]
    strict_bank, n_bank = nested_selection_cached(cache, rows, cfgs, "bank_id")
    strict_source, n_source = nested_selection_cached(cache, rows, cfgs, "source_key")
    selector_evals += n_bank + n_source

    hbank = v20.summary(strict_bank)
    hsource = v20.summary(strict_source)
    pass_both = bool(strict_bank["aggregate"]["pass_5pct_no_cat_fp"] and strict_source["aggregate"]["pass_5pct_no_cat_fp"])
    negative_source = diag.get("negative_sources", [])
    source_outer_notes: Dict[str, Any] = {}
    for s in negative_source:
        obj = strict_source["outer"].get(s)
        if obj:
            source_outer_notes[s] = {"outer_train_cat_count": obj.get("outer_train_cat_count"), "selected_config_id": obj.get("selected_config_id"), "outer_chosen_counts": obj.get("outer_eval", {}).get("chosen_counts"), "outer_bad": len(obj.get("outer_eval", {}).get("catastrophic_false_positive_rows") or []), "outer_save": obj.get("outer_eval", {}).get("decision_relative_saving_vs_fixed_H15")}
    headline = {
        "rows": len(rows), "banks": banks, "source_key_count": len(sources),
        "positive_rows": diag["positive_rows"], "catastrophic_rows": diag["catastrophic_rows"], "negative_sources": negative_source,
        "history_loaded_rows": diag["history_loaded_rows"], "feature_counts": v20.feature_counts(rows, features),
        "config_count": len(cfgs), "selector_refit_evaluations": selector_evals, "cached_score_fits": cache.fit_count, "cache_entries": len(cache.cache),
        "fixed_H12_save": diag["fixed_H12_decision_relative_saving_vs_H15"], "fixed_H12_solver_save": diag["fixed_H12_solver_relative_saving_vs_H15"], "fixed_H12_bad": diag["fixed_H12_bad_rows"],
        "strict_bank_save": hbank["aggregate_save"], "strict_bank_solver_save": hbank["aggregate_solver_save"], "strict_bank_bad": hbank["aggregate_bad"], "strict_bank_h12": hbank["aggregate_h12"], "strict_bank_physical_gate": hbank["aggregate_physical_gate"], "strict_bank_pass5": hbank["aggregate_pass5"],
        "strict_source_save": hsource["aggregate_save"], "strict_source_solver_save": hsource["aggregate_solver_save"], "strict_source_bad": hsource["aggregate_bad"], "strict_source_h12": hsource["aggregate_h12"], "strict_source_physical_gate": hsource["aggregate_physical_gate"], "strict_source_pass5": hsource["aggregate_pass5"],
        "global_bank_pass5_count": sum(1 for x in global_bank_items if x["summary"]["aggregate_pass5"]), "global_source_pass5_count": sum(1 for x in global_source_items if x["summary"]["aggregate_pass5"]), "pass_both_strict_splits": pass_both,
    }
    if pass_both:
        decision = "v20b H12/H15 conservative selector passes both strict opened bank and source nested >=5% zero-catastrophe gates, but the single held-out negative source is protected by a no-cat-training uncertainty fallback. Next do not validate yet: freeze source-independent H12-negative acquisition and an online overhead smoke to test whether this safety rule generalizes beyond one negative cluster."
    elif hbank["aggregate_pass5"] or hsource["aggregate_pass5"]:
        decision = "v20b partially passes one strict split but not both; H12/H15 opportunity is real but current selector evidence is not source/bank robust. Next inspect failing outer groups and acquire source-independent H12 negative/support data or richer terminal-risk features before validation."
    elif hbank["aggregate_bad"] == 0 and hsource["aggregate_bad"] == 0:
        decision = "v20b avoids catastrophic H12 selections but collapses below the 5% measured decision-saving gate; H12/H15 static/probe uncertainty is too conservative. Next prioritize source-independent H12 opportunity/negative acquisition or terminal-risk value training, not validation."
    else:
        decision = "v20b H12/H15 selector still makes catastrophic held-out choices; H12 improved the horizon grid but selector representation/data remain insufficient. Next acquire independent negative H12 boundary states and/or train richer risk/terminal-value models."
    raw = {
        "created_utc": now().isoformat(), "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(),
        "classification": proto["classification"], "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False, "sealed_test_accessed": False, "runtime_repair_for_v20_timeout": timeout_diag,
        "row_diag": diag, "headline": headline, "negative_source_outer_notes": source_outer_notes,
        "strict_bank_nested": strict_bank, "strict_source_nested": strict_source, "top_global_bank_lobo": top_global_bank, "top_global_source_lobo": top_global_source,
        "decision": decision,
        "budget_declared": proto["budget_declared"],
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations": selector_evals, "cached_score_fits": cache.fit_count, "offline_model_evaluations": selector_evals, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}, "input_hashes": hashes,
        "platform": {"python": sys.version, "platform": platform.platform()},
        "interpretation_limits": ["opened-development v19 boundary rows only", "candidate-level repeats are averaged; n=24 candidate states", "only one H12-catastrophic source exists, so source-independent negative generalization cannot be proven", "static/history families do not include measured online selector overhead", "probe-family overhead is approximated from first-H12 solve in saved v19 traces", "no validation64 or sealed final-test evidence", "v20b is a computational caching repair after v20 timeout"],
    }
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    v20.NAME = NAME; v20.STAMP = STAMP; v20.SOURCE = SOURCE; v20.OUT = OUT; v20.PROTOCOL = PROTOCOL; v20.STATE = STATE; v20.BACKUP_REQUEST = BACKUP_REQUEST; v20.MARKER = MARKER
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-h12-selector-v20b-fast", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_h12_selector_v20b_fast:
        raise v20.ContractError("requires --run and --i-accept-development-h12-selector-v20b-fast")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    v20.write_summary(raw)
    req = {"requested_utc": raw["created_utc"], "reason": "backup after v20b fast H12/H15 selector/refit runtime repair before any more unique diagnostics or simulations", "required_before_more_unique_science": True, "artifacts": [rel(SOURCE), rel(BASE_PATH), rel(PROTOCOL), rel(OUT), rel(STATE), rel(BACKUP_REQUEST), rel(V20_TIMEOUT_REGISTRY), rel(V20_TIMEOUT_PROTOCOL), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations": raw["budget_actual"]["selector_refit_evaluations"], "cached_score_fits": raw["budget_actual"]["cached_score_fits"], "offline_model_evaluations": raw["budget_actual"]["offline_model_evaluations"], "training_episodes": 0, "gradient_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False}
    write_json(BACKUP_REQUEST, req)
    done = {"passed": True, "status": "complete", "marker": MARKER, "classification": raw["classification"], "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "hashes": {rel(SOURCE): sha256(SOURCE), rel(BASE_PATH): sha256(BASE_PATH), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(BACKUP_REQUEST): sha256(BACKUP_REQUEST)}}
    write_json(OUT / "completed.json", done)
    v20.append_docs(raw)
    state = {"utc": raw["created_utc"], "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(BACKUP_REQUEST)}, "backup_status": "not verified after v20b; supervisor backup required before more unique science", "next_action": "After verified backup, follow v20b decision: if both strict splits passed, freeze source-independent H12 negative acquisition/online overhead smoke; otherwise inspect failures and revise risk/terminal-value data/model. No validation64 or sealed test."}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v20b fast H12/H15 selector/refit\n\n" + json.dumps(clean(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
