#!/usr/bin/env python3
"""v13b compact/cached history-feature risk/value refit.

Development-only IMPROVED repair after v13 timed out.  The scientific hypothesis is
unchanged from v13: short deployable H15-prefix history features may separate safe
beneficial H10 branch states from catastrophic H10 branch states under the v12
outcome-aligned labels.  The implementation change is purely computational: use a
smaller predeclared conservative-support grid and cache per-split feature scaling
and neighbor distances.

No MPC simulation, no validation64 access, no sealed-test access, and no gradient
training are performed.  Outputs are opened-development evidence only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_history_feature_refit_v13 as v13  # noqa:E402

NAME = "vehicle_true_variable_horizon_history_feature_refit_v13b_fast"
STAMP = "20260929T2150Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T2150_after_history_feature_refit_v13b.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-history-feature-refit-v13b-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def sf(x: Any, default: float = 0.0) -> float:
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


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if hasattr(x, "item"):
        return clean(x.item())
    return x


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def row_key(r: Mapping[str, Any]) -> str:
    return v13.row_key(r)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def compact_grid() -> List[Dict[str, Any]]:
    """Predeclared compact grid replacing v13's oversized exhaustive grid."""
    cfgs: List[Dict[str, Any]] = []
    for family in ["base_no_history", "history_no_risk", "history_with_risk"]:
        for k_pos in [1, 2, 3]:
            for cat_guard_ratio in [1.25, 1.75, 2.5]:
                for max_pos_quantile in [0.50, 0.75, 1.0]:
                    for max_pos_mult in [0.75, 1.0]:
                        for min_gain_s in [0.0, 0.25]:
                            for pred_phys_max in [0.0, 2.0, 20.0]:
                                cfgs.append({
                                    "family": family,
                                    "k_pos": k_pos,
                                    "cat_guard_ratio": cat_guard_ratio,
                                    "max_pos_quantile": max_pos_quantile,
                                    "max_pos_mult": max_pos_mult,
                                    "min_gain_s": min_gain_s,
                                    "pred_phys_max": pred_phys_max,
                                    "cat_margin": 0.0,
                                })
    return cfgs


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v13b_%s_k%s_cg%s_q%s_m%s_g%s_p%s" % (
        c["family"], c["k_pos"], c["cat_guard_ratio"], c["max_pos_quantile"],
        c["max_pos_mult"], c["min_gain_s"], c["pred_phys_max"])


def make_context(rows: Sequence[Mapping[str, Any]], fdict: Mapping[str, Mapping[str, float]], train_banks: Sequence[str], eval_bank: str) -> Dict[str, Any]:
    train_set = set(train_banks)
    train_rows = [r for r in rows if str(r["bank_id"]) in train_set]
    eval_rows = [r for r in rows if str(r["bank_id"]) == eval_bank]
    if not train_rows or not eval_rows:
        raise ContractError(f"empty context train={train_banks} eval={eval_bank}")
    names, means, stds = v13.train_scaler(train_rows, fdict)
    train_vecs = {row_key(r): v13.vec(r, fdict, names, means, stds) for r in train_rows}
    eval_vecs = {row_key(r): v13.vec(r, fdict, names, means, stds) for r in eval_rows}
    pos_train = [r for r in train_rows if r["h10_beneficial_vs_h15"]]
    cat_train = [r for r in train_rows if r["h10_catastrophic_vs_h15"]]
    pp: List[float] = []
    for i, r in enumerate(pos_train):
        ds = [dist(train_vecs[row_key(r)], train_vecs[row_key(q)]) for j, q in enumerate(pos_train) if j != i]
        if ds:
            pp.append(min(ds))
    pp.sort()
    neigh: Dict[str, Any] = {}
    for er in eval_rows:
        ex = eval_vecs[row_key(er)]
        pos_neigh = sorted([(dist(ex, train_vecs[row_key(r)]), r) for r in pos_train], key=lambda z: (z[0], row_key(z[1])))
        cat_neigh = sorted([(dist(ex, train_vecs[row_key(r)]), r) for r in cat_train], key=lambda z: (z[0], row_key(z[1])))
        neigh[row_key(er)] = {"pos": pos_neigh, "cat": cat_neigh}
    return {"train_banks": sorted(train_set), "eval_bank": eval_bank, "eval_rows": eval_rows, "feature_count": len(names), "pp": pp, "neighbors": neigh}


def max_pos_distance(ctx: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    pp = list(ctx["pp"])
    if not pp:
        return 1e9
    q = sf(cfg.get("max_pos_quantile"), 1.0)
    idx = max(0, min(len(pp) - 1, int(round(q * (len(pp) - 1)))))
    return pp[idx] * sf(cfg.get("max_pos_mult"), 1.0)


def eval_context(ctx: Mapping[str, Any], cfg: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    max_d = max_pos_distance(ctx, cfg)
    choices: Dict[str, int] = {}
    scores: Dict[str, Any] = {}
    for er in ctx["eval_rows"]:
        key = row_key(er)
        nb = ctx["neighbors"][key]
        pos_neigh = nb["pos"]
        cat_neigh = nb["cat"]
        if not pos_neigh:
            choices[key] = 15
            continue
        k = min(si(cfg.get("k_pos"), 1), len(pos_neigh))
        near = pos_neigh[:k]
        dpos = sf(near[-1][0])
        dcat = sf(cat_neigh[0][0], 1e9) if cat_neigh else 1e9
        pred_gain = mean([sf(r["decision_gain_h10_vs_h15_s"]) for _, r in near])
        pred_phys = mean([sf(r["phys_delta_h10_minus_h15"]) for _, r in near])
        cat_guard = dcat > sf(cfg.get("cat_guard_ratio"), 1.25) * max(dpos, 1e-9) + sf(cfg.get("cat_margin"), 0.0)
        choose_h10 = bool(dpos <= max_d and cat_guard and pred_gain >= sf(cfg.get("min_gain_s"), 0.0) and pred_phys <= sf(cfg.get("pred_phys_max"), 2.0))
        choices[key] = 10 if choose_h10 else 15
        if include_scores:
            scores[key] = {"dpos": dpos, "dcat": dcat, "max_pos_dist": max_d, "pred_gain_s": pred_gain, "pred_phys_delta": pred_phys, "cat_guard": cat_guard, "selected_h": choices[key]}
    ev = v13.v6.eval_choices(ctx["eval_rows"], choices)
    ev["feature_count"] = ctx["feature_count"]
    if include_scores:
        ev["prediction_scores"] = scores
    return ev


def summarize(hold: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    bad = sum(len(e.get("catastrophic_false_positive_rows") or []) for e in hold.values())
    phys_fail = sum(0 if e.get("physical_gate") else 1 for e in hold.values())
    saves = [sf(e.get("decision_relative_saving_vs_fixed_H15"), -9.0) for e in hold.values()]
    h10 = sum(int((e.get("chosen_counts") or {}).get("10", 0)) for e in hold.values())
    return {
        "all_pass10": all(bool(e.get("pass_10pct_no_cat_fp")) for e in hold.values()),
        "all_pass5": all(bool(e.get("pass_5pct_no_cat_fp")) for e in hold.values()),
        "total_bad": bad,
        "phys_fail": phys_fail,
        "min_save": min(saves) if saves else 0.0,
        "avg_save": mean(saves),
        "total_h10": h10,
    }


def rank(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (int(s["total_bad"]), int(s["phys_fail"]), not bool(s["all_pass10"]), not bool(s["all_pass5"]), -sf(s["min_save"]), -sf(s["avg_save"]), -int(s["total_h10"]))


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def build_all_contexts(rows: Sequence[Mapping[str, Any]], all_features: Mapping[str, Mapping[str, float]], banks: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    contexts: Dict[str, Any] = {}
    feature_counts: Dict[str, int] = {}
    for fam in ["base_no_history", "history_no_risk", "history_with_risk"]:
        fdict = v13.make_family_features(all_features, fam)
        feature_counts[fam] = len(v13.train_scaler(rows, fdict)[0])
        contexts[fam] = {"global": {}, "outer": {}, "inner": {}}
        for hb in banks:
            train = [b for b in banks if b != hb]
            contexts[fam]["global"][hb] = make_context(rows, fdict, train, hb)
            contexts[fam]["outer"][hb] = make_context(rows, fdict, train, hb)
        for outer in banks:
            for hb in banks:
                if hb == outer:
                    continue
                train = [b for b in banks if b not in (outer, hb)]
                contexts[fam]["inner"][(outer, hb)] = make_context(rows, fdict, train, hb)
    return contexts, feature_counts


def evaluate_global(cfg: Mapping[str, Any], banks: Sequence[str], contexts: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    fam = str(cfg["family"])
    return {hb: eval_context(contexts[fam]["global"][hb], cfg, include_scores=include_scores) for hb in banks}


def ambiguity(rows: Sequence[Mapping[str, Any]], all_features: Mapping[str, Mapping[str, float]], family: str) -> Dict[str, Any]:
    # Reuse v13's full ambiguity diagnostic once per family; not on the critical loop.
    return v13.ambiguity(rows, v13.make_family_features(all_features, family), family)


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v13b compact/cached history-feature refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED compact/cached H15-prefix history-feature conservative support refit; no MPC simulation, validation64, sealed test, or gradient training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` banks `{h['banks']}`; positives `{h['positive_rows']}`; catastrophic H10 rows `{h['catastrophic_rows']}`.",
        f"- H15-prefix trace history loaded for `{h['trace_loaded_rows']}/{h['rows']}` rows; feature counts `{h['feature_counts_by_family']}`.",
        f"- Compact configs `{h['config_count']}`; selector-evaluation calls `{raw['budget_actual']['selector_refit_evaluations']}`.",
        f"- Global pass10 `{h['global_pass10_count']}`, pass5 `{h['global_pass5_count']}`; history-family pass5 `{h['history_global_pass5_count']}`.",
        f"- Best global config `{h['best_global_config']}` min save `{pct(h['best_global_min_save'])}`, avg save `{pct(h['best_global_avg_save'])}`, bad `{h['best_global_bad']}`, H10 `{h['best_global_h10']}`.",
        f"- Nested aggregate save `{pct(h['nested_save'])}`, bad `{h['nested_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 `{h['nested_h10']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.",
        f"- Nested selected history configs `{h['nested_selected_history_configs']}/{len(h['banks'])}` outer banks.",
        f"- Decision: {raw['decision']}",
        "",
        "## Nested outer-bank selection",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer save |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b, x in raw["nested_outer"].items():
        ev = x["outer_eval"]
        lines.append(f"| `{b}` | `{x['selected_config_id']}` | `{x['inner_summary']['all_pass5']}` | {x['inner_summary']['total_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Top global LOBO configs", "", "| rank | config | pass10 | pass5 | bad | phys fails | min save | avg save | H10 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_lobo"][:15], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['all_pass10']}` | `{s['all_pass5']}` | {s['total_bad']} | {s['phys_fail']} | {pct(s['min_save'])} | {pct(s['avg_save'])} | {s['total_h10']} |")
    lines += ["", "## Feature-space ambiguity", ""]
    for fam, a in raw["ambiguity"].items():
        lines.append(f"- `{fam}`: features `{a['feature_count']}`, positive-with-closer-cat `{a['positive_with_cat_closer_than_pos']}`, catastrophic-with-closer-positive `{a['cat_with_pos_closer_than_cat']}`.")
    lines += ["", "## Interpretation", "", "This is not validation or final-test evidence. Passing supports only a next small unused-source overhead-aware confirmation. Failure means engineered deployable static/history support features remain insufficient on opened development banks and the next intervention should be explicit calibrated risk/terminal-value learning or targeted boundary acquisition, not another unchanged label-density or feature sweep."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v13b compact/cached history-feature refit

UTC: {raw['created_utc']}. Development-only IMPROVED compact/cached H15-prefix history-feature conservative support refit; no MPC simulation, no validation64, no sealed test, no gradient training. rows={h['rows']}; traces={h['trace_loaded_rows']}/{h['rows']}; global_pass5={h['global_pass5_count']}; nested_save={h['nested_save']:.4f}; nested_bad={h['nested_bad']}; nested_pass5={h['nested_pass5']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old_tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in old_tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_compact_cached_history_feature_support_refit,opened_banks_lobo_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows, input_hashes = v13.load_rows()
    all_features, hist_diag, hist_hashes = v13.enrich_history(rows)
    input_hashes.update(hist_hashes)
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    if banks != ["fresh_v0", "fresh_v1", "fresh_v11", "fresh_v2", "fresh_v8c"] or len(rows) != 68:
        raise ContractError(f"unexpected banks/rows {banks} {len(rows)}")
    cfgs = compact_grid()
    declared_eval_calls = len(cfgs) * len(banks) + len(banks) * (len(cfgs) * (len(banks) - 1) + 1)
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_compact_cached_history_feature_support_refit_no_simulation_no_validation_no_test",
        "hypothesis": "If H10 catastrophic states are distinguishable from safe/beneficial H10 states by short online-observable H15-prefix history features, a compact conservative support selector using v12 outcome-aligned labels should pass nested opened-bank gates faster than v13's oversized implementation.",
        "before_evidence": [
            "v12 aligned oracle over opened banks saves 30.8% decision time with physical gate, but deployable nested selectors remain unsafe.",
            "v13 froze the same hypothesis but timed out after 3600 s due to 6480 configs and repeated distance/scaler recomputation; no outcome evidence was produced."
        ],
        "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11; validation64 and sealed test forbidden",
        "feature_policy": "primary candidates use instantaneous deployable observation/pose/progress plus optional H15-prefix observation/pose history; no scenario group/window IDs",
        "compact_grid": {
            "family": ["base_no_history", "history_no_risk", "history_with_risk"],
            "k_pos": [1, 2, 3],
            "cat_guard_ratio": [1.25, 1.75, 2.5],
            "max_pos_quantile": [0.50, 0.75, 1.0],
            "max_pos_mult": [0.75, 1.0],
            "min_gain_s": [0.0, 0.25],
            "pred_phys_max": [0.0, 2.0, 20.0],
            "cat_margin": [0.0]
        },
        "config_count": len(cfgs),
        "decision_gate": "advance only if nested aggregate has zero catastrophic H10 false positives, physical gate, and >=5% measured branch decision saving; strong if >=10%",
        "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": declared_eval_calls, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "input_hashes": {**input_hashes, rel(SOURCE): sha256(SOURCE)},
        "history_availability_preoutcome": hist_diag,
    }
    write_json(PROTOCOL, protocol)

    contexts, feature_counts = build_all_contexts(rows, all_features, banks)
    selector_evals = 0
    global_rows: List[Dict[str, Any]] = []
    for cfg in cfgs:
        hold = evaluate_global(cfg, banks, contexts, include_scores=False)
        selector_evals += len(hold)
        global_rows.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summarize(hold)})
    ranked = sorted(global_rows, key=lambda x: rank(x["summary"]))
    # Recompute detailed holdouts for top candidates only to keep raw artifacts compact.
    top_global: List[Dict[str, Any]] = []
    for x in ranked[:50]:
        hold = evaluate_global(x["config"], banks, contexts, include_scores=True)
        top_global.append({"config_id": x["config_id"], "config": x["config"], "summary": x["summary"], "holdout_evaluations": hold})

    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    for outer in banks:
        candidates: List[Dict[str, Any]] = []
        for cfg in cfgs:
            fam = str(cfg["family"])
            hold = {hb: eval_context(contexts[fam]["inner"][(outer, hb)], cfg, include_scores=False) for hb in banks if hb != outer}
            selector_evals += len(hold)
            candidates.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summarize(hold)})
        selected = sorted(candidates, key=lambda x: rank(x["summary"]))[0]
        fam = str(selected["config"]["family"])
        outer_eval = eval_context(contexts[fam]["outer"][outer], selected["config"], include_scores=True)
        selector_evals += 1
        nested_outer[outer] = {"selected_config_id": selected["config_id"], "selected_config": selected["config"], "inner_summary": selected["summary"], "outer_eval": outer_eval}
        for d in outer_eval["details"]:
            nested_choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = int(d["selected_h"])
    nested_agg = v13.v6.eval_choices(rows, nested_choices)

    ambiguity_report = {fam: ambiguity(rows, all_features, fam) for fam in ["base_no_history", "history_no_risk", "history_with_risk"]}
    best = ranked[0]
    bs = best["summary"]
    history_global_pass5 = sum(1 for x in global_rows if x["summary"]["all_pass5"] and str(x["config"]["family"]).startswith("history"))
    headline = {
        "rows": len(rows),
        "banks": banks,
        "positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]),
        "catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]),
        "trace_loaded_rows": hist_diag["trace_loaded_rows"],
        "feature_counts_by_family": feature_counts,
        "config_count": len(cfgs),
        "global_pass10_count": sum(1 for x in global_rows if x["summary"]["all_pass10"]),
        "global_pass5_count": sum(1 for x in global_rows if x["summary"]["all_pass5"]),
        "history_global_pass5_count": history_global_pass5,
        "best_global_config": best["config_id"],
        "best_global_min_save": bs["min_save"],
        "best_global_avg_save": bs["avg_save"],
        "best_global_bad": bs["total_bad"],
        "best_global_h10": bs["total_h10"],
        "nested_save": nested_agg["decision_relative_saving_vs_fixed_H15"],
        "nested_bad": len(nested_agg["catastrophic_false_positive_rows"]),
        "nested_physical_gate": bool(nested_agg["physical_gate"]),
        "nested_h10": int(nested_agg["chosen_counts"].get("10", 0)),
        "nested_pass5": bool(nested_agg["pass_5pct_no_cat_fp"]),
        "nested_pass10": bool(nested_agg["pass_10pct_no_cat_fp"]),
        "nested_selected_history_configs": sum(1 for x in nested_outer.values() if str(x["selected_config"]["family"]).startswith("history")),
    }
    if headline["nested_pass10"]:
        decision = "v13b passes strong opened-bank gate; next freeze a tiny unused-source overhead-aware confirmation with actual selector overhead, still no validation64/sealed test."
    elif headline["nested_pass5"]:
        decision = "v13b passes weak opened-bank gate; next freeze a small unused-source overhead-aware confirmation requiring zero catastrophic H10 before any validation64."
    elif history_global_pass5 > 0 and headline["nested_bad"] > 0:
        decision = "History features can fit some opened splits but nested model selection still leaks catastrophic H10; next acquire targeted boundary labels or train calibrated risk/terminal value with stronger uncertainty, not validation rollout."
    elif headline["trace_loaded_rows"] < headline["rows"]:
        decision = "History-feature input is incomplete, so this refit is not decisive; next repair/extract missing deployable history features or acquire targeted risk data before rollout."
    else:
        decision = "H15-prefix history/static engineered features still do not yield a safe deployable selector; next prioritize explicit risk/terminal-value learning/refit with uncertainty or targeted boundary acquisition, not another static-feature sweep or validation rollout."

    return {
        "created_utc": now().isoformat(),
        "classification": protocol["classification"],
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": selector_evals, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "history_availability": hist_diag,
        "headline": headline,
        "top_global_lobo": top_global,
        "nested_outer": nested_outer,
        "nested_aggregate": nested_agg,
        "ambiguity": ambiguity_report,
        "input_hashes": {**input_hashes, rel(PROTOCOL): sha256(PROTOCOL), rel(SOURCE): sha256(SOURCE)},
        "decision": decision,
        "platform": {"python": sys.version, "platform": platform.platform()},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-history-refit-v13b", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_history_refit_v13b:
        raise ContractError("requires --run and explicit v13b development acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_HISTORY_FEATURE_REFIT_V13B_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v13b compact/cached history-feature refit before any further science", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "validation64_bank_opened": False, "sealed_test_accessed": False})
    done = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req), "headline": raw["headline"], "decision": raw["decision"], "validation64_bank_opened": False, "sealed_test_accessed": False, "new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": raw["budget_actual"]["selector_refit_evaluations"], "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(req): sha256(req)}}
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v13b compact/cached history-feature refit\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req)}, "current_backup_status": "not verified after v13b; run backup before further scientific simulation/refit", "next_action": "Follow v13b decision. If pass, freeze unused-source overhead-aware confirmation; otherwise move to explicit calibrated risk/terminal-value learning or targeted boundary acquisition. No validation64/sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
