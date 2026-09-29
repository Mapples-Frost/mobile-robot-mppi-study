#!/usr/bin/env python3
"""v10b fast bounded deployable risk/value refit after v10 timeout.

Development-only IMPROVED diagnostic.  The prior v10 kNN/RBF grid timed out
under the 4 h experiment bound before producing scientific results.  This
version freezes a smaller, optimized nested leave-opened-bank-out refit over the
same opened development rows (fresh_v0/v1/v2/v8c), with conservative H10 risk
support gates and no MPC simulation, validation64, sealed test, or RL gradient
training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_tree_representation_v6 as v6  # noqa:E402
import vehicle_true_variable_horizon_risk_refit_v9_opened_banks as v9  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_value_fast_refit_v10b"
STAMP = "20260929T1815Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1815_after_risk_value_fast_refit_v10b.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-risk-value-fast-refit-v10b-{STAMP}"
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
    return x


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def row_key(r: Mapping[str, Any]) -> str:
    return v6.row_key(r)


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    rows0, hashes, _ = v6.load_rows()
    rows8, h8 = v9.load_v8c_rows()
    rows = [dict(r) for r in rows0 + rows8]
    hashes.update(h8)
    hashes[rel(Path(v6.__file__))] = sha256(Path(v6.__file__))
    hashes[rel(Path(v9.__file__))] = sha256(Path(v9.__file__))
    hashes[rel(SOURCE)] = sha256(SOURCE)
    banks = sorted(set(str(r.get("bank_id")) for r in rows))
    if banks != ["fresh_v0", "fresh_v1", "fresh_v2", "fresh_v8c"] or len(rows) != 56:
        raise ContractError(f"unexpected opened-bank rows/banks: {len(rows)} {banks}")
    for r in rows:
        bid = str(r.get("bank_id", "")).lower()
        if "validation" in bid or "test" in bid:
            raise ContractError("forbidden validation/test-looking bank id: " + str(r.get("bank_id")))
    return rows, hashes


def feature_dict(row: Mapping[str, Any], family: str) -> Dict[str, float]:
    o = [sf(v) for v in (row.get("obs14") or [])[:14]]
    while len(o) < 14:
        o.append(0.0)
    th = sf(row.get("prev_theta"))
    d: Dict[str, float] = {
        "prev_x_30": sf(row.get("prev_x")) / 30.0,
        "prev_y_30": sf(row.get("prev_y")) / 30.0,
        "theta_sin": math.sin(th),
        "theta_cos": math.cos(th),
        "abs_theta_pi": abs(th) / math.pi,
        "branch_step_150": max(0.0, sf(row.get("branch_step"))) / 150.0,
        "slot": sf(row.get("branch_state_slot")),
        "trace_risk_20": sf(row.get("stage_a_trace_risk_score")) / 20.0,
        "obs01_norm": math.hypot(o[0], o[1]),
        "obs56_norm": math.hypot(o[5], o[6]),
        "obs89_norm": math.hypot(o[8], o[9]),
        "obs1112_norm": math.hypot(o[11], o[12]),
    }
    if family in ("compact", "poly"):
        for i in (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            d[f"obs_{i}"] = o[i]
            d[f"abs_obs_{i}"] = abs(o[i])
    if family == "poly":
        for a, b in [(0, 2), (1, 2), (0, 5), (1, 6), (5, 6), (8, 9), (11, 12), (7, 10), (10, 13)]:
            d[f"obs_{a}_x_obs_{b}"] = o[a] * o[b]
        for i in (0, 1, 2, 5, 6, 8, 9, 11, 12):
            d[f"obs_{i}_sq"] = o[i] * o[i]
        d["heading_times_risk"] = d["abs_theta_pi"] * d["trace_risk_20"]
        d["progress_times_risk"] = d["branch_step_150"] * d["trace_risk_20"]
    return d


def build_matrix(rows: Sequence[Mapping[str, Any]], family: str) -> Tuple[List[str], Dict[str, List[float]]]:
    names = sorted(set().union(*(feature_dict(r, family).keys() for r in rows)))
    raw = {row_key(r): feature_dict(r, family) for r in rows}
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for nm in names:
        xs = [sf(raw[row_key(r)].get(nm)) for r in rows]
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        means[nm] = m
        stds[nm] = math.sqrt(var) if var > 1e-18 else 1.0
    vecs = {row_key(r): [(sf(raw[row_key(r)].get(nm)) - means[nm]) / stds[nm] for nm in names] for r in rows}
    return names, vecs


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def weighted_mean(vals: Sequence[float], weights: Sequence[float], default: float = 0.0) -> float:
    sw = sum(weights)
    return default if sw <= 0 else sum(v * w for v, w in zip(vals, weights)) / sw


def predict_from_neighbors(train_rows: Sequence[Mapping[str, Any]], neigh: Sequence[Tuple[float, Mapping[str, Any]]], cfg: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
    k = min(int(cfg["k"]), len(neigh))
    near = list(neigh[:k])
    pwr = sf(cfg["weight_power"])
    ws = [1.0 if pwr <= 0 else 1.0 / ((d + 1e-6) ** pwr) for d, _ in near]
    cat_prior = sum(1 for r in train_rows if r["h10_catastrophic_vs_h15"]) / float(len(train_rows))
    pos_prior = sum(1 for r in train_rows if r["h10_beneficial_vs_h15"]) / float(len(train_rows))
    ps = sf(cfg["prior_strength"])
    sw = sum(ws)
    p_cat = (sum(w for w, (_, r) in zip(ws, near) if r["h10_catastrophic_vs_h15"]) + ps * cat_prior) / (sw + ps)
    p_pos = (sum(w for w, (_, r) in zip(ws, near) if r["h10_beneficial_vs_h15"]) + ps * pos_prior) / (sw + ps)
    pred_gain = weighted_mean([sf(r["decision_gain_h10_vs_h15_s"]) for _, r in near], ws)
    pred_phys = weighted_mean([sf(r["phys_delta_h10_minus_h15"]) for _, r in near], ws)
    cat_count = sum(1 for _, r in near if r["h10_catastrophic_vs_h15"])
    nearest_cat = min([d for d, r in neigh if r["h10_catastrophic_vs_h15"]] or [1e9])
    nearest_pos = min([d for d, r in neigh if r["h10_beneficial_vs_h15"]] or [1e9])
    ratio = sf(cfg["cat_guard_ratio"])
    guard_ok = True if ratio <= 0 else bool(nearest_cat > ratio * max(nearest_pos, 1e-9))
    choose = bool(cat_count == 0 and p_cat <= sf(cfg["risk_max"]) and p_pos >= sf(cfg["benefit_min"]) and pred_gain >= sf(cfg["min_gain_s"]) and pred_phys <= sf(cfg["pred_phys_max"]) and guard_ok)
    return (10 if choose else 15), {"p_cat": p_cat, "p_pos": p_pos, "pred_gain_s": pred_gain, "pred_phys_delta": pred_phys, "cat_count_k": cat_count, "nearest_cat_dist": nearest_cat, "nearest_pos_dist": nearest_pos, "cat_guard_ok": guard_ok, "k_eff": k}


def eval_config(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    family = str(cfg["family"])
    all_rows = list(train_rows) + list(eval_rows)
    _, vecs = build_matrix(train_rows, family)
    # Build eval vectors using train-set scaler by temporarily rebuilding from train stats.
    names = sorted(set().union(*(feature_dict(r, family).keys() for r in train_rows)))
    raw_train = {row_key(r): feature_dict(r, family) for r in train_rows}
    means = {}; stds = {}
    for nm in names:
        xs = [sf(raw_train[row_key(r)].get(nm)) for r in train_rows]
        m = sum(xs) / len(xs); var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        means[nm] = m; stds[nm] = math.sqrt(var) if var > 1e-18 else 1.0
    def vec(r: Mapping[str, Any]) -> List[float]:
        fd = feature_dict(r, family)
        return [(sf(fd.get(nm)) - means[nm]) / stds[nm] for nm in names]
    train_vecs = {row_key(r): vec(r) for r in train_rows}
    choices: Dict[str, int] = {}
    preds: Dict[str, Any] = {}
    for er in eval_rows:
        ex = vec(er)
        neigh = sorted([(dist(ex, train_vecs[row_key(tr)]), tr) for tr in train_rows], key=lambda z: (z[0], row_key(z[1])))
        h, info = predict_from_neighbors(train_rows, neigh, cfg)
        choices[row_key(er)] = h
        preds[row_key(er)] = info
    ev = v6.eval_choices(eval_rows, choices)
    ev["prediction_scores"] = preds
    return ev


def grid() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    for family in ["minimal", "compact", "poly"]:
        for k in [1, 3, 5]:
            for wp in [0.0, 1.0]:
                for risk_max in [0.0, 0.10]:
                    for benefit_min in [0.25, 0.50]:
                        for min_gain_s in [0.0, 0.4]:
                            for pred_phys_max in [0.0, 2.0]:
                                for cgr in [0.0, 1.25]:
                                    for ps in [0.0, 1.0]:
                                        cfgs.append({"family": family, "k": k, "weight_power": wp, "risk_max": risk_max, "benefit_min": benefit_min, "min_gain_s": min_gain_s, "pred_phys_max": pred_phys_max, "cat_guard_ratio": cgr, "prior_strength": ps})
    return cfgs


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v10b_%s_k%s_wp%s_rm%s_bm%s_g%s_p%s_cgr%s_ps%s" % (c["family"], c["k"], c["weight_power"], c["risk_max"], c["benefit_min"], c["min_gain_s"], c["pred_phys_max"], c["cat_guard_ratio"], c["prior_strength"])


def summarize(evals: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    bad = sum(len(e.get("catastrophic_false_positive_rows") or []) for e in evals.values())
    phys_fail = sum(0 if e.get("physical_gate") else 1 for e in evals.values())
    saves = [sf(e.get("decision_relative_saving_vs_fixed_H15"), -9.0) for e in evals.values()]
    h10 = sum(int((e.get("chosen_counts") or {}).get("10", 0)) for e in evals.values())
    return {"all_pass10": all(bool(e.get("pass_10pct_no_cat_fp")) for e in evals.values()), "all_pass5": all(bool(e.get("pass_5pct_no_cat_fp")) for e in evals.values()), "total_bad": bad, "phys_fail": phys_fail, "min_save": min(saves) if saves else 0.0, "avg_save": sum(saves) / len(saves) if saves else 0.0, "total_h10": h10}


def rank(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (int(s["total_bad"]), int(s["phys_fail"]), not bool(s["all_pass10"]), not bool(s["all_pass5"]), -sf(s["min_save"]), -sf(s["avg_save"]), -int(s["total_h10"]))


def eval_lobo(rows: Sequence[Mapping[str, Any]], banks: Sequence[str], cfg: Mapping[str, Any], blocked: Sequence[str] = ()) -> Dict[str, Any]:
    hold: Dict[str, Any] = {}
    blocked_set = set(blocked)
    for hb in banks:
        if hb in blocked_set:
            continue
        tr = [r for r in rows if str(r["bank_id"]) not in blocked_set and str(r["bank_id"]) != hb]
        ev = [r for r in rows if str(r["bank_id"]) == hb]
        hold[hb] = eval_config(tr, ev, cfg)
    return hold


def ambiguity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = {"family": "poly", "k": 1, "weight_power": 0.0, "risk_max": 0.0, "benefit_min": 0.0, "min_gain_s": 0.0, "pred_phys_max": 1e9, "cat_guard_ratio": 0.0, "prior_strength": 0.0}
    names, _ = build_matrix(rows, "poly")
    raw = {row_key(r): feature_dict(r, "poly") for r in rows}
    means = {}; stds = {}
    for nm in names:
        xs = [sf(raw[row_key(r)].get(nm)) for r in rows]
        m = sum(xs) / len(xs); var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        means[nm] = m; stds[nm] = math.sqrt(var) if var > 1e-18 else 1.0
    vecs = {row_key(r): [(sf(raw[row_key(r)].get(nm)) - means[nm]) / stds[nm] for nm in names] for r in rows}
    items = []
    for r in rows:
        k0 = row_key(r); others = [(dist(vecs[k0], vecs[row_key(q)]), q) for q in rows if row_key(q) != k0]
        cats = sorted([(d, q) for d, q in others if q["h10_catastrophic_vs_h15"]], key=lambda z: z[0])
        bens = sorted([(d, q) for d, q in others if q["h10_beneficial_vs_h15"]], key=lambda z: z[0])
        nc = cats[0] if cats else (1e9, {})
        nb = bens[0] if bens else (1e9, {})
        items.append({"id": k0, "bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "catastrophic": bool(r["h10_catastrophic_vs_h15"]), "beneficial": bool(r["h10_beneficial_vs_h15"]), "nearest_cat_dist": nc[0], "nearest_cat_id": row_key(nc[1]) if nc[1] else None, "nearest_beneficial_dist": nb[0], "nearest_beneficial_id": row_key(nb[1]) if nb[1] else None, "phys_delta": sf(r["phys_delta_h10_minus_h15"]), "decision_gain_s": sf(r["decision_gain_h10_vs_h15_s"])})
    return {"catastrophic_rows": sorted([x for x in items if x["catastrophic"]], key=lambda x: x["nearest_beneficial_dist"])[:8], "positive_rows_closest_to_cat": sorted([x for x in items if x["beneficial"]], key=lambda x: x["nearest_cat_dist"])[:8], "positive_with_cat_closer_than_beneficial": sum(1 for x in items if x["beneficial"] and x["nearest_cat_dist"] <= x["nearest_beneficial_dist"])}


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = ["# Vehicle true-variable-H v10b fast risk/value refit", "", f"UTC `{raw['created_utc']}`. Development-only opened-bank refit; no MPC simulation, validation64, sealed test, or gradient training.", "", "## Headline", "", f"- Rows `{h['rows']}` banks `{h['banks']}` positives `{h['positive_rows']}` catastrophic H10 rows `{h['catastrophic_rows']}`.", f"- Configs `{h['config_count']}`; global LOBO pass10 `{h['global_pass10_count']}`, pass5 `{h['global_pass5_count']}`.", f"- Best global config `{h['best_global_config']}` min save `{pct(h['best_global_min_save'])}`, avg save `{pct(h['best_global_avg_save'])}`, bad `{h['best_global_bad']}`, H10 total `{h['best_global_h10']}`.", f"- Nested aggregate save `{pct(h['nested_aggregate_save'])}`, bad `{h['nested_total_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 count `{h['nested_h10_count']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.", f"- Decision: {raw['decision']}", "", "## Nested outer-bank model selection", "", "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer save |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for b, x in raw["nested_outer"].items():
        ev = x["outer_eval"]
        lines.append(f"| `{b}` | `{x['selected_config_id']}` | `{x['inner_summary']['all_pass5']}` | {x['inner_summary']['total_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Top global LOBO configs", "", "| rank | config | pass10 | pass5 | bad | phys fails | min save | avg save | H10 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_lobo"][:12], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['all_pass10']}` | `{s['all_pass5']}` | {s['total_bad']} | {s['phys_fail']} | {pct(s['min_save'])} | {pct(s['avg_save'])} | {s['total_h10']} |")
    lines += ["", "## Ambiguity diagnostic", "", f"Positive rows with a catastrophic neighbor no farther than the nearest beneficial neighbor: `{raw['ambiguity_diagnostics']['positive_with_cat_closer_than_beneficial']}`.", "Closest positives to catastrophic rows:"]
    for x in raw["ambiguity_diagnostics"]["positive_rows_closest_to_cat"][:6]:
        lines.append(f"- `{x['id']}` nearest_cat `{x['nearest_cat_id']}` d={sf(x['nearest_cat_dist']):.4g}; physΔ={sf(x['phys_delta']):.4g}, gain={sf(x['decision_gain_s']):.4g}s")
    lines += ["", "## Interpretation", "", "This is development evidence only. Passing would justify a tiny unused-source overhead-aware confirmation. Failing means the next informative action is targeted risk-data acquisition or richer terminal/risk value training/refit, not validation64 rollout or another unchanged label-density sweep."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v10b fast risk/value refit

UTC: {raw['created_utc']}. Development-only opened-bank supervised/nonparametric refit after v10 timeout; no MPC simulation, no validation64, no sealed test. configs={h['config_count']}; global_pass5={h['global_pass5_count']}; nested_save={h['nested_aggregate_save']:.4f}; nested_bad={h['nested_total_bad']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old[-120000:]:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_fast_risk_value_refit,opened_banks_lobo_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-risk-value-v10b", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_risk_value_v10b:
        raise ContractError("requires --run and explicit v10b development acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hashes = load_rows()
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    cfgs = grid()
    protocol = {"protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}", "created_utc": now().isoformat(), "classification": "development_IMPROVED_fast_supervised_risk_value_refit_no_sim_no_validation_no_test", "hypothesis": "A bounded conservative deployable kNN risk/value support model may avoid v8c catastrophic H10 false positives while retaining >=5% measured branch decision-time savings; if nested LOBO fails, the blocker is risk representation/data or terminal value, not a validation rollout prerequisite.", "inputs": sorted(hashes), "backup_verified_commit_from_supervisor_context": args.backup_verified_commit, "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c", "feature_policy": "deployable online observations/pose/progress/risk scalar only; no scenario group/window labels for candidate", "config_count": len(cfgs), "decision_gate": "advance only if nested aggregate has zero catastrophic H10 false positives, physical gate, and >=5% measured decision saving; strong if >=10%; otherwise targeted risk-data/value training/refit", "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "rl_gradient_steps": 0, "selector_refit_evaluations_upper_bound": len(cfgs) * (len(banks) * (len(banks)-1) + len(banks)), "validation64_episodes": 0, "sealed_test_episodes": 0}, "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False}, "hashes": hashes}
    write_json(PROTOCOL, protocol)

    global_rows: List[Dict[str, Any]] = []
    selector_evals = 0
    for cfg in cfgs:
        hold = eval_lobo(rows, banks, cfg)
        selector_evals += len(hold)
        s = summarize(hold)
        global_rows.append({"config_id": cfg_id(cfg), "config": cfg, "summary": s, "holdout_evaluations": hold})
    global_ranked = sorted(global_rows, key=lambda x: rank(x["summary"]))

    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    for outer in banks:
        candidates = []
        inner_banks = [b for b in banks if b != outer]
        for cfg in cfgs:
            hold = eval_lobo(rows, banks, cfg, blocked=[outer])
            selector_evals += len(hold)
            candidates.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summarize(hold)})
        selected = sorted(candidates, key=lambda x: rank(x["summary"]))[0]
        train_outer = [r for r in rows if str(r["bank_id"]) != outer]
        eval_outer = [r for r in rows if str(r["bank_id"]) == outer]
        outer_eval = eval_config(train_outer, eval_outer, selected["config"])
        selector_evals += 1
        nested_outer[outer] = {"selected_config_id": selected["config_id"], "selected_config": selected["config"], "inner_summary": selected["summary"], "outer_eval": outer_eval}
        for d in outer_eval["details"]:
            nested_choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = int(d["selected_h"])
    nested_agg = v6.eval_choices(rows, nested_choices)
    best = global_ranked[0]
    bs = best["summary"]
    headline = {"rows": len(rows), "banks": banks, "positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]), "catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]), "config_count": len(cfgs), "global_pass10_count": sum(1 for r in global_rows if r["summary"]["all_pass10"]), "global_pass5_count": sum(1 for r in global_rows if r["summary"]["all_pass5"]), "best_global_config": best["config_id"], "best_global_min_save": bs["min_save"], "best_global_avg_save": bs["avg_save"], "best_global_bad": bs["total_bad"], "best_global_h10": bs["total_h10"], "nested_aggregate_save": nested_agg["decision_relative_saving_vs_fixed_H15"], "nested_total_bad": len(nested_agg["catastrophic_false_positive_rows"]), "nested_physical_gate": bool(nested_agg["physical_gate"]), "nested_h10_count": int(nested_agg["chosen_counts"].get("10", 0)), "nested_pass5": bool(nested_agg["pass_5pct_no_cat_fp"]), "nested_pass10": bool(nested_agg["pass_10pct_no_cat_fp"])}
    if headline["nested_pass10"]:
        decision = "Nested fast deployable risk/value refit passes strong opened-bank gate; next freeze tiny unused-source overhead-aware confirmation before any validation64."
    elif headline["nested_pass5"]:
        decision = "Nested fast deployable risk/value refit passes weak opened-bank gate; next freeze tiny unused-source overhead-aware confirmation with measured selector overhead before any validation64."
    elif headline["global_pass5_count"] > 0 and headline["nested_total_bad"] == 0:
        decision = "Post-hoc configs can pass but nested selection cannot retain saving; acquire targeted development risk rows before another refit."
    elif headline["nested_total_bad"] > 0 or headline["best_global_bad"] > 0:
        decision = "Deployable risk/value refit still permits catastrophic H10 false positives; next freeze targeted risk-data acquisition around ambiguous catastrophic/near-safe states or richer terminal/risk value training, not validation rollout."
    else:
        decision = "Conservative refit avoids catastrophes only by collapsing compute savings; next acquire boundary data or revise value/objective features before rollout."
    raw = {"created_utc": now().isoformat(), "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(), "classification": protocol["classification"], "validation64_bank_opened": False, "sealed_test_accessed": False, "protocol": rel(PROTOCOL), "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "rl_gradient_steps": 0, "selector_refit_evaluations": selector_evals, "validation64_episodes": 0, "sealed_test_episodes": 0}, "bank_label_summary": {b: {"rows": sum(1 for r in rows if r["bank_id"] == b), "positive": sum(1 for r in rows if r["bank_id"] == b and r["h10_beneficial_vs_h15"]), "catastrophic": sum(1 for r in rows if r["bank_id"] == b and r["h10_catastrophic_vs_h15"])} for b in banks}, "headline": headline, "nested_outer": nested_outer, "nested_aggregate": nested_agg, "top_global_lobo": global_ranked[:20], "ambiguity_diagnostics": ambiguity(rows), "decision": decision, "hashes": hashes, "platform": {"python": sys.version, "platform": platform.platform()}}
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_RISK_VALUE_FAST_REFIT_V10B_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v10b fast risk/value refit before any further simulation/training", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE)], "validation64_bank_opened": False, "sealed_test_accessed": False})
    completed = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "backup_request": rel(req), "headline": headline, "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_mpc_simulation_episodes": 0, "new_control_steps": 0, "rl_gradient_steps": 0, "selector_refit_evaluations": selector_evals, "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md")}}
    write_json(OUT / "completed.json", completed)
    STATE.write_text("# Continue state after v10b fast risk/value refit\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": headline, "decision": decision, "budgets": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "backup_request": rel(req)}, "next_action": "Verify external backup. If nested pass, run tiny unused-source overhead-aware confirmation; otherwise freeze targeted risk-data acquisition or richer terminal/risk value training. No validation64/sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_docs(raw)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": headline, "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"failed_utc": now().isoformat(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "validation64_bank_opened": False, "sealed_test_accessed": False})
        print(json.dumps({"failed": type(exc).__name__, "message": str(exc), "failed_artifact": rel(OUT / "failed.json"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        raise
