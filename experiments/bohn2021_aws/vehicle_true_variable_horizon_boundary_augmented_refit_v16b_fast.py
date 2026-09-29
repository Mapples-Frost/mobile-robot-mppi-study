#!/usr/bin/env python3
"""Fast v16b boundary-augmented conservative risk/value refit.

This is a runtime repair of v16 after the first v16 attempt was SIGTERM'd by the
600 s experiment bound before writing raw/completed outputs.  It preserves the
v16 scientific hypothesis, augmented opened-development inputs, config grid and
strict held-out-bank decision gate, but caches split-level predictor scores so
threshold configurations are evaluated without recomputing scalers/distances for
every config.

Development-only IMPROVED diagnostic: no MPC simulation, no gradient/RL training,
no validation64 access, no sealed-test access.
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

import vehicle_true_variable_horizon_boundary_augmented_refit_v16 as v16  # noqa:E402

NAME = "vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast"
STAMP = "20260929T2320Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2320_after_boundary_augmented_refit_v16b_fast.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_BOUNDARY_AUGMENTED_REFIT_V16B_FAST_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-boundary-augmented-refit-v16b-fast-{STAMP}"
V16_TIMEOUT_REGISTRY = ROOT / "research_artifacts/aws_runs/20260929T230841_0364c24d/registry.json"
V16_TIMEOUT_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_boundary_augmented_refit_v16_preoutcome_frozen_20260929T2315Z.json"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sf(x: Any, default: float = 0.0) -> float:
    return v16.sf(x, default)


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


class ScoreCache:
    def __init__(self, rows: Sequence[Mapping[str, Any]], features: Mapping[str, Mapping[str, float]]):
        self.rows = list(rows)
        self.features = features
        self.cache: Dict[Tuple[str, str, int, float, float], Dict[str, Any]] = {}
        self.fit_count = 0

    def scores(self, split_id: str, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
        family = str(cfg["family"])
        k = int(cfg["k"])
        calib_q = float(cfg["calib_q"])
        support_mult = float(cfg["support_mult"])
        key = (split_id, family, k, calib_q, support_mult)
        if key in self.cache:
            return self.cache[key]
        fdict = v16.family_features(self.features, family)
        names, means, stds = v16.train_scaler(train_rows, fdict)
        train_vecs = {v16.row_key(r): v16.vec(r, fdict, names, means, stds) for r in train_rows}
        eval_vecs = {v16.row_key(r): v16.vec(r, fdict, names, means, stds) for r in eval_rows}
        cal = v16.calibration(train_rows, train_vecs, k, calib_q)
        radius = v16.support_radius(train_rows, train_vecs, support_mult)
        row_scores: Dict[str, Any] = {}
        for r in eval_rows:
            est = v16.base_estimate(eval_vecs[v16.row_key(r)], train_rows, train_vecs, k)
            risk_ucb = min(1.0, sf(est["risk_hat"], 1.0) + sf(cal["risk_q"]))
            phys_ucb = sf(est["pred_phys"], 1e9) + sf(cal["phys_q"])
            gain_lcb = sf(est["pred_gain"], -1e9) - sf(cal["gain_q"])
            row_scores[v16.row_key(r)] = {
                "risk_hat": est["risk_hat"],
                "risk_ucb": risk_ucb,
                "pred_phys": est["pred_phys"],
                "phys_ucb": phys_ucb,
                "pred_gain": est["pred_gain"],
                "gain_lcb": gain_lcb,
                "dpos": est["dpos"],
                "dcat": est["dcat"],
                "support_radius": radius,
                "support_ok": sf(est["dpos"], 1e9) <= radius,
                "calibration": cal,
                "feature_count": len(names),
            }
        obj = {"scores": row_scores, "feature_count": len(names), "train_rows": len(train_rows), "eval_rows": len(eval_rows)}
        self.cache[key] = obj
        self.fit_count += 1
        return obj


def choices_from_scores(scores: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, s in scores.items():
        choose = bool(
            s.get("support_ok")
            and sf(s.get("risk_ucb"), 1.0) <= sf(cfg["risk_ucb_max"])
            and sf(s.get("phys_ucb"), 1e9) <= sf(cfg["phys_ucb_max"])
            and sf(s.get("gain_lcb"), -1e9) >= sf(cfg["gain_lcb_min"])
        )
        out[k] = 10 if choose else 15
    return out


def eval_split_cached(cache: ScoreCache, split_id: str, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], include_scores: bool = False) -> Dict[str, Any]:
    obj = cache.scores(split_id, train_rows, eval_rows, cfg)
    choices = choices_from_scores(obj["scores"], cfg)
    ev = v16.eval_choices(eval_rows, choices)
    ev["feature_count"] = obj["feature_count"]
    if include_scores:
        scored = {}
        for rk, s in obj["scores"].items():
            ss = dict(s)
            ss["selected_h"] = choices[rk]
            scored[rk] = ss
        ev["prediction_scores"] = scored
    return ev


def eval_lobo_cached(cache: ScoreCache, rows: Sequence[Mapping[str, Any]], banks: Sequence[str], cfg: Mapping[str, Any], blocked_banks: Sequence[str] = (), include_scores: bool = False, split_prefix: str = "lobo") -> Dict[str, Any]:
    blocked = set(str(b) for b in blocked_banks)
    eval_banks = [b for b in banks if b not in blocked]
    holdouts: Dict[str, Any] = {}
    for hb in eval_banks:
        train = [r for r in rows if str(r["bank_id"]) not in blocked and str(r["bank_id"]) != hb]
        evrows = [r for r in rows if str(r["bank_id"]) == hb]
        sid = f"{split_prefix}|blocked={','.join(sorted(blocked))}|holdout={hb}"
        holdouts[hb] = eval_split_cached(cache, sid, train, evrows, cfg, include_scores=include_scores)
    eval_rows = [r for r in rows if str(r["bank_id"]) in eval_banks]
    return {"holdouts": holdouts, "aggregate": v16.aggregate_from_details(eval_rows, holdouts)}


def compact_eval(ev: Mapping[str, Any]) -> Dict[str, Any]:
    keep = dict(ev)
    if len(keep.get("details") or []) > 25:
        keep["details"] = list(keep["details"][:25])
        keep["details_truncated"] = True
    if "prediction_scores" in keep and len(keep["prediction_scores"]) > 20:
        keys = sorted(keep["prediction_scores"])[:20]
        keep["prediction_scores"] = {k: keep["prediction_scores"][k] for k in keys}
        keep["prediction_scores_truncated"] = True
    return keep


def summarize_timeout() -> Dict[str, Any]:
    out: Dict[str, Any] = {"observed": V16_TIMEOUT_REGISTRY.exists(), "registry": rel(V16_TIMEOUT_REGISTRY)}
    if V16_TIMEOUT_REGISTRY.exists():
        try:
            reg = read_json(V16_TIMEOUT_REGISTRY)
            out.update({
                "exit_status": reg.get("exit_status"),
                "runtime_seconds": reg.get("runtime_seconds"),
                "script_sha256": reg.get("script_sha256"),
            })
        except Exception as exc:
            out["read_error"] = str(exc)
    if V16_TIMEOUT_PROTOCOL.exists():
        out["protocol"] = rel(V16_TIMEOUT_PROTOCOL)
        out["protocol_sha256"] = sha256(V16_TIMEOUT_PROTOCOL)
    return out


def freeze_protocol(created: dt.datetime, rows_diag: Mapping[str, Any], cfgs: Sequence[Mapping[str, Any]], backup_commit: str, hashes: Mapping[str, str], timeout_diag: Mapping[str, Any]) -> None:
    proto = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_boundary_augmented_risk_value_refit_fast_runtime_repair_no_sim_no_validation_no_test",
        "runtime_repair_for": timeout_diag,
        "hypothesis": "Same as v16: v15 boundary labels may repair a conservative H10/H15 selector under strict held-out opened-bank evaluation. v16b changes only computation/caching after v16 timed out; it keeps the augmented rows, config grid, strict split and pass gate.",
        "inputs": {
            "old_opened_rows_source": "v13/v12 loader over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11",
            "v15_boundary_raw": rel(v16.V15_RAW),
            "v15_boundary_completed": rel(v16.V15_DONE),
            "v15_boundary_protocol": rel(v16.V15_PROTOCOL),
        },
        "rows_diag": rows_diag,
        "families": v16.FAMILIES,
        "config_count": len(cfgs),
        "strict_evaluation": "nested opened-bank holdout over augmented rows; for each outer bank, all rows from that bank, including v15 boundary rows from that bank/source, are excluded from training and used only for outer evaluation",
        "decision_gate_before_unused_source_confirmation": {"zero_catastrophic_h10_false_positives": True, "physical_gate": True, "minimum_measured_decision_saving_vs_fixed_H15": 0.05},
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 50000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False, "mobile_robot_mppi_resumed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": dict(hashes),
    }
    write_json(PROTOCOL, proto)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows, features, rows_diag, hashes = v16.load_augmented_rows()
    hashes[rel(SOURCE)] = sha256(SOURCE)
    if V16_TIMEOUT_REGISTRY.exists():
        hashes[rel(V16_TIMEOUT_REGISTRY)] = sha256(V16_TIMEOUT_REGISTRY)
    if V16_TIMEOUT_PROTOCOL.exists():
        hashes[rel(V16_TIMEOUT_PROTOCOL)] = sha256(V16_TIMEOUT_PROTOCOL)
    cfgs = v16.cfg_grid()
    timeout_diag = summarize_timeout()
    freeze_protocol(created, rows_diag, cfgs, args.backup_verified_commit, hashes, timeout_diag)

    banks = sorted(set(str(r["bank_id"]) for r in rows))
    cache = ScoreCache(rows, features)
    equivalent_selector_evals = 0

    global_items: List[Dict[str, Any]] = []
    for cfg in cfgs:
        lobo = eval_lobo_cached(cache, rows, banks, cfg, include_scores=False, split_prefix="global")
        equivalent_selector_evals += len(banks)
        global_items.append({"config_id": v16.cfg_id(cfg), "config": cfg, "summary": v16.summary_from_lobo(lobo), "lobo": lobo})
    global_ranked = sorted(global_items, key=lambda x: v16.rank_summary(x["summary"]))
    best_global = global_ranked[0]
    top_global: List[Dict[str, Any]] = []
    for x in global_ranked[:20]:
        top_global.append({"config_id": x["config_id"], "config": x["config"], "summary": x["summary"], "aggregate": x["lobo"]["aggregate"]})

    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    for outer_bank in banks:
        inner_items: List[Dict[str, Any]] = []
        for cfg in cfgs:
            lobo = eval_lobo_cached(cache, rows, banks, cfg, blocked_banks=[outer_bank], include_scores=False, split_prefix=f"inner_outer={outer_bank}")
            equivalent_selector_evals += len([b for b in banks if b != outer_bank])
            inner_items.append({"config_id": v16.cfg_id(cfg), "config": cfg, "summary": v16.summary_from_lobo(lobo)})
        selected = sorted(inner_items, key=lambda x: v16.rank_summary(x["summary"]))[0]
        train_outer = [r for r in rows if str(r["bank_id"]) != outer_bank]
        eval_outer = [r for r in rows if str(r["bank_id"]) == outer_bank]
        outer_eval = eval_split_cached(cache, f"outer_eval|holdout={outer_bank}", train_outer, eval_outer, selected["config"], include_scores=True)
        equivalent_selector_evals += 1
        for d in outer_eval.get("details") or []:
            nested_choices[str(d["unique_row_id"])] = int(d["selected_h"])
        nested_outer[outer_bank] = {"selected_config_id": selected["config_id"], "selected_config": selected["config"], "inner_summary": selected["summary"], "outer_eval": compact_eval(outer_eval)}
    nested_aggregate = v16.eval_choices(rows, nested_choices)

    in_eval = eval_split_cached(cache, "in_sample|all", rows, rows, best_global["config"], include_scores=True)
    equivalent_selector_evals += 1
    in_sample = compact_eval(in_eval)
    in_sample["self_neighbor_leakage"] = True

    h = {
        "rows": len(rows),
        "old_rows": rows_diag["old_row_count"],
        "v15_boundary_rows": rows_diag["v15_boundary_row_count"],
        "banks": banks,
        "source_key_count": rows_diag["source_key_count"],
        "positive_rows": rows_diag["label_counts"]["positive"],
        "catastrophic_rows": rows_diag["label_counts"]["catastrophic"],
        "v15_positive_rows": rows_diag["label_counts"]["v15_positive"],
        "v15_catastrophic_rows": rows_diag["label_counts"]["v15_catastrophic"],
        "config_count": len(cfgs),
        "equivalent_selector_refit_evaluations": equivalent_selector_evals,
        "cached_score_fits": cache.fit_count,
        "cache_entries": len(cache.cache),
        "global_pass5_count": sum(1 for x in global_items if x["summary"]["aggregate_pass5"]),
        "global_pass10_count": sum(1 for x in global_items if x["summary"]["aggregate_pass10"]),
        "best_global_config": best_global["config_id"],
        "best_global_bad": best_global["summary"]["aggregate_bad"],
        "best_global_save": best_global["summary"]["aggregate_save"],
        "best_global_solver_save": best_global["summary"]["aggregate_solver_save"],
        "best_global_h10": best_global["summary"]["aggregate_h10"],
        "nested_bad": len(nested_aggregate.get("catastrophic_false_positive_rows") or []),
        "nested_physical_gate": bool(nested_aggregate.get("physical_gate")),
        "nested_save": sf(nested_aggregate.get("decision_relative_saving_vs_fixed_H15")),
        "nested_solver_save": sf(nested_aggregate.get("solver_relative_saving_vs_fixed_H15")),
        "nested_h10": int((nested_aggregate.get("chosen_counts") or {}).get("10", 0)),
        "nested_pass5": bool(nested_aggregate.get("pass_5pct_no_cat_fp")),
        "nested_pass10": bool(nested_aggregate.get("pass_10pct_no_cat_fp")),
        "in_sample_bad": len(in_sample.get("catastrophic_false_positive_rows") or []),
        "in_sample_save": sf(in_sample.get("decision_relative_saving_vs_fixed_H15")),
        "in_sample_h10": int((in_sample.get("chosen_counts") or {}).get("10", 0)),
    }
    if h["nested_pass5"] and h["nested_bad"] == 0:
        decision = "v16b strict boundary-augmented refit meets the opened-development zero-catastrophe >=5% decision-saving gate; next freeze an unused-source development confirmation with actual selector overhead before validation64/test."
    elif h["global_pass5_count"] > 0 and not h["nested_pass5"]:
        decision = "v16b has at least one development global LOBO pass but fails strict nested model selection; treat as overfit/model-selection instability and do not confirm unchanged. Inspect nested failures, then pivot to richer risk/terminal-value learning or a better predeclared model-selection rule."
    elif h["in_sample_bad"] == 0 and h["in_sample_save"] >= 0.05:
        decision = "v16b boundary labels repair local/in-sample separability but not held-out generalization; boundary-risk information is relevant but static/refit features remain insufficient. Pivot to richer learned risk/terminal-value representation rather than another static-feature sweep."
    else:
        decision = "v16b boundary-augmented static/refit selector still fails to produce a safe useful held-out tradeoff; stop static/refit sweeps and prioritize explicit learned risk/terminal-value training or observability/scenario diagnostics."

    raw = {
        "created_utc": now().isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(),
        "classification": "development_IMPROVED_boundary_augmented_risk_value_refit_fast_runtime_repair_no_sim_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "runtime_repair_for_v16_timeout": timeout_diag,
        "rows_diag": rows_diag,
        "headline": h,
        "decision": decision,
        "top_global_lobo": top_global,
        "nested_outer": nested_outer,
        "nested_aggregate": nested_aggregate,
        "in_sample_repair_diagnostic": in_sample,
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 50000, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "equivalent_selector_refit_evaluations": equivalent_selector_evals, "cached_score_fits": cache.fit_count, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "platform": platform.platform()},
        "input_hashes": hashes,
        "interpretation_limits": ["opened development only", "v15 rows are deliberately mined boundary labels", "strict nested opened-bank result is decision-relevant", "global/in-sample repair diagnostics are not confirmation", "no online selector runtime overhead measured here", "v16b is a computational caching repair after v16 timeout, not a new scientific objective"],
    }
    return raw


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v16b-fast boundary-augmented refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED metadata/refit diagnostic; no MPC simulation, no validation64, no sealed test, no gradient/RL training.",
        f"Runtime repair: v16 registry `{raw['runtime_repair_for_v16_timeout'].get('registry')}` observed={raw['runtime_repair_for_v16_timeout'].get('observed')} exit={raw['runtime_repair_for_v16_timeout'].get('exit_status')} runtime={raw['runtime_repair_for_v16_timeout'].get('runtime_seconds')}. v16b changes computation/caching only.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` = old opened rows `{h['old_rows']}` + v15 boundary rows `{h['v15_boundary_rows']}` across banks `{h['banks']}` and source keys `{h['source_key_count']}`.",
        f"- Labels: positives `{h['positive_rows']}` (v15 `{h['v15_positive_rows']}`); catastrophic H10 `{h['catastrophic_rows']}` (v15 `{h['v15_catastrophic_rows']}`).",
        f"- Configs `{h['config_count']}`; equivalent selector-refit evaluations `{h['equivalent_selector_refit_evaluations']}`; cached score fits `{h['cached_score_fits']}`.",
        f"- Global LOBO pass5 `{h['global_pass5_count']}`, pass10 `{h['global_pass10_count']}`; best global `{h['best_global_config']}` save `{pct(h['best_global_save'])}`, solver save `{pct(h['best_global_solver_save'])}`, bad `{h['best_global_bad']}`, H10 `{h['best_global_h10']}`.",
        f"- Strict nested opened-bank aggregate: save `{pct(h['nested_save'])}`, solver save `{pct(h['nested_solver_save'])}`, bad `{h['nested_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 `{h['nested_h10']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.",
        f"- Local in-sample repair diagnostic: save `{pct(h['in_sample_save'])}`, bad `{h['in_sample_bad']}`, H10 `{h['in_sample_h10']}`; self-neighbour leakage makes this non-generalization evidence.",
        f"- Decision: {raw['decision']}",
        "",
        "## Strict nested outer-bank results",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer save | outer solver save |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, obj in raw["nested_outer"].items():
        ev = obj["outer_eval"]
        inn = obj["inner_summary"]
        lines.append(f"| `{b}` | `{obj['selected_config_id']}` | `{inn['aggregate_pass5']}` | {inn['aggregate_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} |")
    lines += ["", "## Top global LOBO configs", "", "| rank | config | pass10 | pass5 | bad | physical gate | save | solver save | H10 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_lobo"][:15], 1):
        s = x["summary"]
        lines.append(f"| {i} | `{x['config_id']}` | `{s['aggregate_pass10']}` | `{s['aggregate_pass5']}` | {s['aggregate_bad']} | `{s['aggregate_physical_gate']}` | {pct(s['aggregate_save'])} | {pct(s['aggregate_solver_save'])} | {s['aggregate_h10']} |")
    fps = raw.get("nested_aggregate", {}).get("catastrophic_false_positive_rows") or []
    lines += ["", "## Strict nested catastrophic H10 false positives", ""]
    if fps:
        lines.append("| bank | row | source | origin | role | off | phys delta | tol | decision gain s |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for r in fps[:50]:
            lines.append(f"| `{r.get('bank_id')}` | `{r.get('base_state_id')}` | `{r.get('source_key')}` | `{r.get('row_origin')}` | `{r.get('boundary_role')}` | {int(r.get('offset_from_center') or 0)} | {sf(r.get('phys_delta')):.6g} | {sf(r.get('tol')):.6g} | {sf(r.get('decision_gain_s')):.6g} |")
    else:
        lines.append("No strict nested catastrophic false positives.")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is opened-development refit evidence only. The strict nested bank holdout excludes the held-out bank's v15 boundary rows from training, but the data distribution is still development-informed and intentionally enriched around known failures. A pass would justify only an unused-source development confirmation with actual selector overhead. A failure means v15 boundary labels are informative but not sufficient for deployable generalization under this static/refit class.",
        "",
        f"Raw: `{rel(OUT / 'raw.json')}`; completed: `{rel(OUT / 'completed.json')}`; protocol: `{rel(PROTOCOL)}`; backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v16b-fast boundary-augmented refit

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED runtime repair after v16 timeout; no MPC simulation, no validation64/sealed-test access, no gradient training. rows={h['rows']}; configs={h['config_count']}; equivalent_selector_refit_evaluations={h['equivalent_selector_refit_evaluations']}; cached_score_fits={h['cached_score_fits']}; strict_nested_save={h['nested_save']:.6f}; strict_nested_bad={h['nested_bad']}; strict_nested_pass5={h['nested_pass5']}; global_pass5={h['global_pass5_count']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    reg = ROOT / "EXPERIMENT_REGISTRY.csv"
    tail = reg.read_text(encoding="utf-8", errors="replace")[-120000:] if reg.exists() else ""
    if MARKER not in tail:
        with reg.open("a", encoding="utf-8") as f:
            f.write(f"{STAMP},{NAME},development_boundary_augmented_refit_fast,opened_rows_plus_v15_boundary_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['equivalent_selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-boundary-augmented-refit", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_boundary_augmented_refit:
        raise RuntimeError("requires --run and explicit development boundary-augmented refit acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    if OUT.exists() and any(p.name != "run.lock" for p in OUT.iterdir()):
        raise RuntimeError("partial v16b output exists; inspect before rerun: " + rel(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    append_docs(raw)
    completed_utc = now()
    write_json(BACKUP_REQUEST, {
        "requested_utc": completed_utc.isoformat(),
        "reason": "backup v16 timeout record and v16b fast boundary-augmented refit source/protocol/raw/docs before any unused-source confirmation or training pivot",
        "backup_required_before_more_science": True,
        "development_mpc_simulation_episodes": 0,
        "development_control_steps": 0,
        "equivalent_selector_refit_evaluations": raw["budget_actual"]["equivalent_selector_refit_evaluations"],
        "cached_score_fits": raw["budget_actual"]["cached_score_fits"],
        "training_episodes": 0,
        "gradient_steps": 0,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(SOURCE), rel(PROTOCOL), rel(OUT), rel(STATE), rel(V16_TIMEOUT_REGISTRY), rel(V16_TIMEOUT_PROTOCOL), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
    })
    STATE.parent.mkdir(parents=True, exist_ok=True)
    h = raw["headline"]
    STATE.write_text(f"""# Continue state after v16b-fast boundary-augmented refit

UTC: {completed_utc.isoformat()}
Elapsed since first supervisor event: {(completed_utc - FIRST_EVENT).total_seconds()/3600.0:.2f} h.

Completed `{NAME}` metadata/refit diagnostic. This is a computational caching repair after v16 timed out at 600 s; it changed no scientific input/split/gate. No MPC simulation, no validation64, no sealed test, no gradient training.

Budget actual: {raw['budget_actual']}
Headline: {h}
Decision: {raw['decision']}
Summary: `{rel(OUT / 'summary.md')}`
Raw: `{rel(OUT / 'raw.json')}`
Completed: `{rel(OUT / 'completed.json')}`
Backup request: `{rel(BACKUP_REQUEST)}`

Next action: verify external backup for v16 timeout + v16b artifacts. If strict nested_pass5 is true with zero bad, freeze unused-source confirmation with actual selector overhead. Otherwise do not rerun static/refit sweeps unchanged; pivot to richer learned risk/terminal-value representation or observability/scenario diagnostics while preserving negative evidence.
""", encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "completed.json"] + [SOURCE, PROTOCOL, STATE, BACKUP_REQUEST, V16_TIMEOUT_REGISTRY, V16_TIMEOUT_PROTOCOL, v16.V15_RAW, v16.V15_DONE, v16.V15_PROTOCOL]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": completed_utc.isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (completed_utc - FIRST_EVENT).total_seconds(),
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "budget_actual": raw["budget_actual"],
        "headline": h,
        "decision": raw["decision"],
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(BACKUP_REQUEST),
        "runtime_repair_for_v16_timeout": raw["runtime_repair_for_v16_timeout"],
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(OUT / "completed.json", completed)
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": h, "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failure.json", {
            "failed_utc": now().isoformat(),
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "selector_refit_evaluations_cap": 50000, "training_episodes": 0, "gradient_steps": 0},
            "next_recovery_hint": "Preserve partial outputs; inspect failure before writing a versioned v16c or pivoting to learned risk/terminal-value diagnostics.",
        })
        raise
