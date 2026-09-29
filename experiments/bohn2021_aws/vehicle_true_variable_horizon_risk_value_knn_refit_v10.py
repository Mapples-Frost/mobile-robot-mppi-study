#!/usr/bin/env python3
"""v10 deployable risk/value representation refit for true-variable H10/H15.

Development-only IMPROVED diagnostic after v9.  v9 found an opened-bank oracle
control/compute opportunity but no safe shallow-tree selector once v8c risk
anchors were included.  This script tests a different bounded supervised
risk/value representation: a conservative kNN/RBF risk-and-continuation-value
model over deployable online observations, selected by nested leave-bank-out
inner development folds and evaluated on the outer opened development bank.

No MPC simulation, no validation64, no sealed final test, and no RL/gradient
training are performed.  The result decides whether the next action should be an
overhead-aware fresh confirmation of this model class or targeted risk-data /
value-training acquisition.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_risk_tree_representation_v6 as v6  # noqa:E402
import vehicle_true_variable_horizon_risk_refit_v9_opened_banks as v9  # noqa:E402

NAME = "vehicle_true_variable_horizon_risk_value_knn_refit_v10"
STAMP = "20260929T1405Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1405_after_risk_value_knn_refit_v10.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-risk-value-knn-refit-v10-{STAMP}"
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


def load_all_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    source_rows, hashes, _ = v6.load_rows()
    v8c_rows, h8 = v9.load_v8c_rows()
    rows = [dict(r) for r in source_rows + v8c_rows]
    hashes.update(h8)
    hashes[rel(Path(v6.__file__))] = sha256(Path(v6.__file__))
    hashes[rel(Path(v9.__file__))] = sha256(Path(v9.__file__))
    hashes[rel(SOURCE)] = sha256(SOURCE)
    for r in rows:
        if "validation" in str(r.get("bank_id", "")).lower() or "test" in str(r.get("bank_id", "")).lower():
            raise ContractError("unexpected validation/test-looking bank id: " + str(r.get("bank_id")))
    return rows, hashes


def base_features(row: Mapping[str, Any], family: str) -> Dict[str, float]:
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
        "obs34_norm": math.hypot(o[3], o[4]),
        "obs56_norm": math.hypot(o[5], o[6]),
        "obs89_norm": math.hypot(o[8], o[9]),
        "obs1112_norm": math.hypot(o[11], o[12]),
    }
    if family in ("compact", "full", "poly_margin"):
        for i in (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            d[f"obs_{i}"] = o[i]
            d[f"abs_obs_{i}"] = abs(o[i])
    if family in ("full", "poly_margin"):
        for i in range(14):
            d[f"all_obs_{i}"] = o[i]
            d[f"all_abs_obs_{i}"] = abs(o[i])
    if family == "poly_margin":
        pairs = [(0, 2), (1, 2), (0, 5), (1, 6), (5, 6), (8, 9), (11, 12), (7, 10), (10, 13)]
        for a, b in pairs:
            d[f"obs_{a}_x_obs_{b}"] = o[a] * o[b]
        for i in (0, 1, 2, 5, 6, 8, 9, 11, 12):
            d[f"obs_{i}_sq"] = o[i] * o[i]
        d["heading_times_risk"] = d["abs_theta_pi"] * d["trace_risk_20"]
        d["progress_times_risk"] = d["branch_step_150"] * d["trace_risk_20"]
    return d


def feature_names(rows: Sequence[Mapping[str, Any]], family: str) -> List[str]:
    names = set()
    for r in rows:
        names.update(base_features(r, family).keys())
    return sorted(names)


def scaler(train_rows: Sequence[Mapping[str, Any]], names: Sequence[str], family: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    vals = [base_features(r, family) for r in train_rows]
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for nm in names:
        xs = [sf(v.get(nm)) for v in vals]
        m = sum(xs) / len(xs) if xs else 0.0
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        s = math.sqrt(var)
        means[nm] = m
        stds[nm] = s if s > 1e-9 else 1.0
    return means, stds


def vector(row: Mapping[str, Any], names: Sequence[str], family: str, means: Mapping[str, float], stds: Mapping[str, float]) -> List[float]:
    d = base_features(row, family)
    return [(sf(d.get(nm)) - sf(means.get(nm))) / sf(stds.get(nm), 1.0) for nm in names]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def row_key(r: Mapping[str, Any]) -> str:
    return v6.row_key(r)


def predict_one(train_rows: Sequence[Mapping[str, Any]], test_row: Mapping[str, Any], cfg: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
    fam = str(cfg["family"])
    names = feature_names(train_rows, fam)
    means, stds = scaler(train_rows, names, fam)
    tx = vector(test_row, names, fam, means, stds)
    neigh: List[Tuple[float, Mapping[str, Any]]] = []
    for r in train_rows:
        neigh.append((dist(tx, vector(r, names, fam, means, stds)), r))
    neigh.sort(key=lambda z: (z[0], str(z[1].get("bank_id")), str(z[1].get("base_state_id"))))
    k = min(int(cfg["k"]), len(neigh))
    near = neigh[:k]
    pwr = sf(cfg.get("weight_power"), 1.0)
    ws = []
    for d0, _ in near:
        ws.append(1.0 if pwr <= 0 else 1.0 / ((d0 + 1e-6) ** pwr))
    sw = sum(ws) if ws else 1.0
    cat_prior = sum(1 for r in train_rows if r["h10_catastrophic_vs_h15"]) / float(len(train_rows))
    pos_prior = sum(1 for r in train_rows if r["h10_beneficial_vs_h15"]) / float(len(train_rows))
    prior_strength = sf(cfg.get("prior_strength"), 0.0)
    cat_weight = sum(w for w, (_, r) in zip(ws, near) if r["h10_catastrophic_vs_h15"])
    pos_weight = sum(w for w, (_, r) in zip(ws, near) if r["h10_beneficial_vs_h15"])
    p_cat = (cat_weight + prior_strength * cat_prior) / (sw + prior_strength)
    p_pos = (pos_weight + prior_strength * pos_prior) / (sw + prior_strength)
    pred_gain = sum(w * sf(r["decision_gain_h10_vs_h15_s"]) for w, (_, r) in zip(ws, near)) / sw
    pred_phys = sum(w * sf(r["phys_delta_h10_minus_h15"]) for w, (_, r) in zip(ws, near)) / sw
    cat_count = sum(1 for _, r in near if r["h10_catastrophic_vs_h15"])
    nearest_cat = min([d0 for d0, r in neigh if r["h10_catastrophic_vs_h15"]] or [1e9])
    nearest_pos = min([d0 for d0, r in neigh if r["h10_beneficial_vs_h15"]] or [1e9])
    guard_ratio = sf(cfg.get("cat_guard_ratio"), 0.0)
    cat_guard_ok = True if guard_ratio <= 0 else bool(nearest_cat > guard_ratio * max(nearest_pos, 1e-9))
    choose = bool(
        cat_count <= int(cfg["max_cat_k"])
        and p_cat <= sf(cfg["risk_max"])
        and p_pos >= sf(cfg["benefit_min"])
        and pred_gain >= sf(cfg["min_gain_s"])
        and pred_phys <= sf(cfg["pred_phys_max"])
        and cat_guard_ok
    )
    return (10 if choose else 15), {
        "p_cat": p_cat,
        "p_pos": p_pos,
        "pred_gain_s": pred_gain,
        "pred_phys_delta": pred_phys,
        "cat_count_k": cat_count,
        "nearest_cat_dist": nearest_cat,
        "nearest_pos_dist": nearest_pos,
        "cat_guard_ok": cat_guard_ok,
        "k_eff": k,
    }


def eval_config(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    choices: Dict[str, int] = {}
    preds: Dict[str, Any] = {}
    for r in eval_rows:
        h, info = predict_one(train_rows, r, cfg)
        choices[row_key(r)] = h
        preds[row_key(r)] = info
    ev = v6.eval_choices(eval_rows, choices)
    ev["prediction_scores"] = preds
    return ev


def config_id(cfg: Mapping[str, Any]) -> str:
    return "v10_%s_k%s_wp%s_mc%s_rm%s_bm%s_g%s_p%s_cgr%s_ps%s" % (
        cfg["family"], cfg["k"], cfg["weight_power"], cfg["max_cat_k"], cfg["risk_max"], cfg["benefit_min"], cfg["min_gain_s"], cfg["pred_phys_max"], cfg["cat_guard_ratio"], cfg["prior_strength"])


def grid() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fam in ["compact", "full", "poly_margin"]:
        for k in [1, 3, 5, 7, 9]:
            for wp in [0.0, 1.0, 2.0]:
                for max_cat in [0, 1]:
                    for risk_max in [0.0, 0.10, 0.20]:
                        for benefit_min in [0.25, 0.50, 0.75]:
                            for min_gain in [0.0, 0.4, 0.8]:
                                for pred_phys_max in [0.0, 1.0, 2.0]:
                                    for cgr in [0.0, 1.0, 1.5]:
                                        for ps in [0.0, 1.0]:
                                            out.append({"family": fam, "k": k, "weight_power": wp, "max_cat_k": max_cat, "risk_max": risk_max, "benefit_min": benefit_min, "min_gain_s": min_gain, "pred_phys_max": pred_phys_max, "cat_guard_ratio": cgr, "prior_strength": ps})
    return out


def summarize_holdouts(evals: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    bad = sum(len(e.get("catastrophic_false_positive_rows") or []) for e in evals.values())
    phys_fail = sum(0 if e.get("physical_gate") else 1 for e in evals.values())
    saves = [sf(e.get("decision_relative_saving_vs_fixed_H15"), -9.0) for e in evals.values()]
    h10 = sum(int((e.get("chosen_counts") or {}).get("10", 0)) for e in evals.values())
    return {
        "all_pass10": all(bool(e.get("pass_10pct_no_cat_fp")) for e in evals.values()),
        "all_pass5": all(bool(e.get("pass_5pct_no_cat_fp")) for e in evals.values()),
        "total_bad": bad,
        "phys_fail": phys_fail,
        "min_save": min(saves) if saves else 0.0,
        "avg_save": sum(saves) / len(saves) if saves else 0.0,
        "total_h10": h10,
    }


def rank_summary(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (not bool(s["all_pass10"]), not bool(s["all_pass5"]), int(s["total_bad"]), int(s["phys_fail"]), -sf(s["min_save"]), -sf(s["avg_save"]), -int(s["total_h10"]))


def inner_select(rows: Sequence[Mapping[str, Any]], outer_bank: str, cfgs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_banks = sorted(set(str(r["bank_id"]) for r in rows if str(r["bank_id"]) != outer_bank))
    best = None
    for cfg in cfgs:
        hold: Dict[str, Any] = {}
        for hb in train_banks:
            tr = [r for r in rows if str(r["bank_id"]) not in (outer_bank, hb)]
            ev = [r for r in rows if str(r["bank_id"]) == hb]
            hold[hb] = eval_config(tr, ev, cfg)
        summ = summarize_holdouts(hold)
        cand = {"config": cfg, "config_id": config_id(cfg), "inner_holdouts": hold, "summary": summ}
        if best is None or rank_summary(cand["summary"]) < rank_summary(best["summary"]):
            best = cand
    assert best is not None
    return best


def ambiguity_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fam = "poly_margin"
    names = feature_names(rows, fam)
    means, stds = scaler(rows, names, fam)
    vecs = {row_key(r): vector(r, names, fam, means, stds) for r in rows}
    items = []
    for r in rows:
        k0 = row_key(r); x = vecs[k0]
        others = [(dist(x, vecs[row_key(q)]), q) for q in rows if row_key(q) != k0]
        cats = sorted([(d, q) for d, q in others if q["h10_catastrophic_vs_h15"]], key=lambda z: z[0])
        bens = sorted([(d, q) for d, q in others if q["h10_beneficial_vs_h15"]], key=lambda z: z[0])
        nearest_cat = cats[0] if cats else (1e9, {})
        nearest_ben = bens[0] if bens else (1e9, {})
        items.append({
            "bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "catastrophic": bool(r["h10_catastrophic_vs_h15"]), "beneficial": bool(r["h10_beneficial_vs_h15"]),
            "group": r.get("group"), "window": r.get("window"), "phys_delta": sf(r.get("phys_delta_h10_minus_h15")), "decision_gain_s": sf(r.get("decision_gain_h10_vs_h15_s")),
            "nearest_cat_dist": nearest_cat[0], "nearest_cat_id": row_key(nearest_cat[1]) if nearest_cat[1] else None,
            "nearest_beneficial_dist": nearest_ben[0], "nearest_beneficial_id": row_key(nearest_ben[1]) if nearest_ben[1] else None,
            "ambiguous_ratio_cat_over_ben": nearest_cat[0] / max(nearest_ben[0], 1e-9),
        })
    catastrophic_items = [x for x in items if x["catastrophic"]]
    positive_items = [x for x in items if x["beneficial"]]
    return {
        "family": fam,
        "catastrophic_rows": sorted(catastrophic_items, key=lambda x: x["nearest_beneficial_dist"])[:12],
        "positive_rows_closest_to_catastrophe": sorted(positive_items, key=lambda x: x["nearest_cat_dist"])[:12],
        "catastrophic_with_beneficial_closer_than_catastrophe_count": sum(1 for x in catastrophic_items if x["nearest_beneficial_dist"] <= x["nearest_cat_dist"]),
        "positive_with_catastrophe_closer_than_beneficial_count": sum(1 for x in positive_items if x["nearest_cat_dist"] <= x["nearest_beneficial_dist"]),
    }


def pct(x: Any) -> str:
    return "%.2f%%" % (100.0 * sf(x))


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v10 risk/value kNN refit",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only supervised/nonparametric risk-value refit over opened banks; no MPC simulation, validation64, sealed test, or RL gradient training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` across `{h['banks']}`; positives `{h['positive_rows']}`, catastrophic H10 rows `{h['catastrophic_rows']}`.",
        f"- Configs `{h['config_count']}`; LOBO post-hoc pass10 `{h['global_lobo_pass10_count']}`, pass5 `{h['global_lobo_pass5_count']}`.",
        f"- Best global LOBO deployable config `{h['best_global_config']}`: min save `{pct(h['best_global_min_save'])}`, bad `{h['best_global_bad']}`, H10 total `{h['best_global_h10']}`.",
        f"- Nested model-selection aggregate: decision save `{pct(h['nested_aggregate_save'])}`, bad `{h['nested_total_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 count `{h['nested_h10_count']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Nested outer-bank results",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer decision save |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b, r in raw["nested_outer"].items():
        ev = r["outer_eval"]
        lines.append(f"| `{b}` | `{r['selected_config_id']}` | `{r['inner_summary']['all_pass5']}` | {r['inner_summary']['total_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Top global LOBO configs", "", "| rank | config | pass10 | pass5 | bad | phys fails | min save | avg save | H10 total |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(raw["top_global_lobo"][:15], 1):
        s = r["summary"]
        lines.append(f"| {i} | `{r['config_id']}` | `{s['all_pass10']}` | `{s['all_pass5']}` | {s['total_bad']} | {s['phys_fail']} | {pct(s['min_save'])} | {pct(s['avg_save'])} | {s['total_h10']} |")
    amb = raw["ambiguity_diagnostics"]
    lines += ["", "## Representation/coverage diagnostic", "", f"- Catastrophic rows whose nearest beneficial row is at least as close as nearest catastrophic row: `{amb['catastrophic_with_beneficial_closer_than_catastrophe_count']}`.", f"- Positive rows whose nearest catastrophic row is at least as close as nearest beneficial row: `{amb['positive_with_catastrophe_closer_than_beneficial_count']}`.", "", "Closest positive rows to a catastrophic prototype (deployable features only):"]
    for x in amb["positive_rows_closest_to_catastrophe"][:6]:
        lines.append(f"- `{x['bank_id']}/{x['base_state_id']}` group `{x['group']}` window `{x['window']}` nearest_cat `{x['nearest_cat_id']}` d={sf(x['nearest_cat_dist']):.4g}; physΔ={sf(x['phys_delta']):.4g}, gain={sf(x['decision_gain_s']):.4g}s")
    lines += ["", "## Interpretation", "", "This is opened-development evidence only.  Passing would justify a small overhead-aware fresh confirmation; failing with ambiguity or nested false positives means the next informative experiment is targeted branch-data acquisition or richer value/risk training, not validation64 rollout or another unchanged threshold sweep."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v10 risk/value kNN refit

UTC: {raw['created_utc']}. Development-only bounded supervised/nonparametric risk-value refit over opened banks; no MPC simulation, no validation64, no sealed test. Configs={raw['headline']['config_count']}; global LOBO pass5={raw['headline']['global_lobo_pass5_count']}; pass10={raw['headline']['global_lobo_pass10_count']}; nested aggregate saving={raw['headline']['nested_aggregate_save']:.4f}; nested bad={raw['headline']['nested_total_bad']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    old = reg.read_text(encoding="utf-8", errors="replace") if reg.exists() else ""
    if MARKER not in old[-80000:]:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_supervised_risk_value_refit,opened_banks_lobo_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-risk-value-v10", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_risk_value_v10:
        raise ContractError("requires --run and explicit v10 development refit acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    created0 = now()
    rows, hashes = load_all_rows()
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    cfgs = grid()
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created0.isoformat(),
        "classification": "development_IMPROVED_supervised_risk_value_refit_opened_banks_no_sim_no_validation_no_test",
        "hypothesis": "A conservative deployable observation/history risk-value representation can avoid v8c catastrophic H10 false positives while retaining >=5-10% measured decision-time savings; if nested LOBO fails, evidence supports targeted risk-data/value training rather than validation rollout.",
        "inputs": sorted(hashes.keys()),
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c; no validation64/sealed test",
        "feature_policy": "deployable online observations, previous pose, branch progress, trace risk scalar and derived polynomial/margin terms only; no scenario group/window metadata for candidate policy",
        "config_count": len(cfgs),
        "decision_gate": "advance only if nested aggregate and per-bank outer evaluations have zero catastrophic H10 false positives, physical gate, and >=5% measured decision saving vs fixed true H15; otherwise acquire targeted risk data or train/refit richer value/risk representation",
        "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "rl_gradient_steps": 0, "selector_refit_evaluations_upper_bound": len(cfgs) * (len(banks) * (len(banks)-1) + len(banks)), "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "hashes": hashes,
    }
    write_json(PROTOCOL, protocol)

    # Post-hoc global LOBO screen (opened-development diagnostic, not selection evidence).
    global_rows = []
    for cfg in cfgs:
        hold: Dict[str, Any] = {}
        for hb in banks:
            tr = [r for r in rows if str(r["bank_id"]) != hb]
            ev = [r for r in rows if str(r["bank_id"]) == hb]
            hold[hb] = eval_config(tr, ev, cfg)
        summ = summarize_holdouts(hold)
        global_rows.append({"config_id": config_id(cfg), "config": cfg, "summary": summ, "holdout_evaluations": hold})
    global_ranked = sorted(global_rows, key=lambda r: rank_summary(r["summary"]))

    # Nested model selection: each outer bank chooses config using only other banks.
    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    selector_evals = len(cfgs) * len(banks)  # global LOBO holdouts
    for outer in banks:
        selected = inner_select(rows, outer, cfgs)
        selector_evals += len(cfgs) * (len(banks) - 1)
        train_outer = [r for r in rows if str(r["bank_id"]) != outer]
        eval_outer = [r for r in rows if str(r["bank_id"]) == outer]
        outer_eval = eval_config(train_outer, eval_outer, selected["config"])
        selector_evals += 1
        nested_outer[outer] = {"selected_config_id": selected["config_id"], "selected_config": selected["config"], "inner_summary": selected["summary"], "outer_eval": outer_eval}
        for d in outer_eval["details"]:
            nested_choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = int(d["selected_h"])
    nested_aggregate = v6.eval_choices(rows, nested_choices)
    best_global = global_ranked[0]
    bg = best_global["summary"]
    bank_summary = {b: {"rows": sum(1 for r in rows if r["bank_id"] == b), "positive": sum(1 for r in rows if r["bank_id"] == b and r["h10_beneficial_vs_h15"]), "catastrophic": sum(1 for r in rows if r["bank_id"] == b and r["h10_catastrophic_vs_h15"])} for b in banks}
    amb = ambiguity_diagnostics(rows)
    headline = {
        "rows": len(rows), "banks": banks,
        "positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]),
        "catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]),
        "config_count": len(cfgs),
        "global_lobo_pass10_count": sum(1 for r in global_rows if r["summary"]["all_pass10"]),
        "global_lobo_pass5_count": sum(1 for r in global_rows if r["summary"]["all_pass5"]),
        "best_global_config": best_global["config_id"],
        "best_global_min_save": bg["min_save"],
        "best_global_avg_save": bg["avg_save"],
        "best_global_bad": bg["total_bad"],
        "best_global_h10": bg["total_h10"],
        "nested_aggregate_save": nested_aggregate["decision_relative_saving_vs_fixed_H15"],
        "nested_total_bad": len(nested_aggregate["catastrophic_false_positive_rows"]),
        "nested_physical_gate": nested_aggregate["physical_gate"],
        "nested_h10_count": int(nested_aggregate["chosen_counts"].get("10", 0)),
        "nested_pass5": bool(nested_aggregate["pass_5pct_no_cat_fp"]),
        "nested_pass10": bool(nested_aggregate["pass_10pct_no_cat_fp"]),
    }
    if headline["nested_pass10"]:
        decision = "Nested deployable risk/value kNN passes strong opened-bank gate; next freeze a tiny unused fresh-source overhead-aware confirmation before any validation64."
    elif headline["nested_pass5"]:
        decision = "Nested deployable risk/value kNN passes weak opened-bank gate; next confirm with a tiny unused fresh-source overhead-aware branch/closed-loop smoke and measured selector overhead before validation64."
    elif headline["global_lobo_pass5_count"] > 0 and headline["nested_total_bad"] == 0:
        decision = "Some post-hoc opened-bank configs pass but nested selection cannot retain enough saving; data are too small/unstable for reliable model selection, so acquire targeted development risk rows before another refit."
    elif headline["nested_total_bad"] > 0 or headline["best_global_bad"] > 0:
        decision = "Deployable observation-only risk/value refit still leaves catastrophic false positives; next freeze targeted risk-data acquisition around ambiguous catastrophic/near-safe states or train a richer terminal/risk value model, not validation rollout."
    else:
        decision = "Conservative risk/value refit avoids catastrophes only by collapsing compute savings; next acquire targeted boundary data and/or revise objective/value features before rollout."
    created = now()
    raw = {
        "created_utc": created.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
        "classification": protocol["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "mobile_robot_mppi_resumed": False,
        "protocol": rel(PROTOCOL),
        "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "rl_gradient_steps": 0, "selector_refit_evaluations": selector_evals, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "bank_label_summary": bank_summary,
        "headline": headline,
        "nested_outer": nested_outer,
        "nested_aggregate": nested_aggregate,
        "top_global_lobo": global_ranked[:25],
        "ambiguity_diagnostics": amb,
        "decision": decision,
        "hashes": hashes,
        "platform": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_RISK_VALUE_KNN_REFIT_V10_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v10 supervised risk/value kNN refit before targeted data acquisition or further simulation", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE)], "validation64_bank_opened": False, "sealed_test_accessed": False})
    completed = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "backup_request": rel(req), "headline": headline, "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False, "new_mpc_simulation_episodes": 0, "new_control_steps": 0, "rl_gradient_steps": 0, "selector_refit_evaluations": selector_evals, "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md")}}
    write_json(OUT / "completed.json", completed)
    STATE.write_text("# Continue state after v10 risk/value kNN refit\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": headline, "decision": decision, "budgets": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "backup_request": rel(req)}, "next_action": "Verify external backup for v10 artifacts. If v10 passed weak/strong nested gate, freeze tiny unused fresh-source overhead confirmation; otherwise freeze targeted development-only risk-data acquisition around ambiguous catastrophic/near-safe states, then refit a richer risk/value model. No validation64 or sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_docs(raw)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": headline, "decision": decision, "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(req)}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "completed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
