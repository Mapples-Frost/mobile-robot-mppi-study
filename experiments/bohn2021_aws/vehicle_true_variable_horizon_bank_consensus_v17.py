#!/usr/bin/env python3
"""v17 bank-consensus uncertainty veto diagnostic for true-variable H10/H15.

Development-only IMPROVED offline/refit diagnostic after v16b.
No MPC simulation, no gradient/RL training, no validation64 access and no sealed
final-test access.

Hypothesis: v16b's held-out fresh_v11 catastrophic H10 false positives may be a
model-selection/generalization uncertainty failure rather than a lack of local
boundary labels.  A conservative cross-bank consensus veto should choose H10 only
when an aggregate model and every leave-one-training-bank submodel agree.  If
this removes catastrophic false positives but loses almost all timing benefit,
then current deployable features/data are too weak for a useful selector and the
next step should be terminal/risk-value learning or new source-independent state
coverage, not another static feature sweep.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_true_variable_horizon_boundary_augmented_refit_v16 as v16  # noqa:E402
import vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast as v16b  # noqa:E402

NAME = "vehicle_true_variable_horizon_bank_consensus_v17"
STAMP = "20260929T2345Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2345_after_bank_consensus_v17.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_BANK_CONSENSUS_V17_{STAMP}.json"
SOURCE = Path(__file__).resolve()
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
MARKER = f"vehicle-true-variable-H-bank-consensus-v17-{STAMP}"
V16B_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast_20260929T2320Z/raw.json"
V16B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast_20260929T2320Z/completed.json"
ASTRA_LATEST = ROOT / "docs/bohn2021_takeover/astra_reviews/LATEST.md"
ASTRA_REPORT = ROOT / "docs/bohn2021_takeover/astra_reviews/20260929T153837Z.md"
ASTRA_MANIFEST = ROOT / "docs/bohn2021_takeover/astra_reviews/20260929T153837Z.manifest.json"
ASTRA_RESPONSE = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"


class ContractError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for key in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(key) is True:
            raise ContractError(f"forbidden {key}=true in {label}")


def cfg_key(cfg: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (cfg.get("family"), int(cfg.get("k", 0)), float(cfg.get("calib_q", 0.0)), float(cfg.get("risk_ucb_max", 0.0)), float(cfg.get("phys_ucb_max", 0.0)), float(cfg.get("gain_lcb_min", 0.0)), float(cfg.get("support_mult", 0.0)))


def load_v16b_and_candidates() -> Tuple[Mapping[str, Any], List[Dict[str, Any]], Dict[str, str]]:
    if not V16B_DONE.exists() or not V16B_RAW.exists():
        raise ContractError("v16b completed/raw inputs missing")
    done = read_json(V16B_DONE)
    raw = read_json(V16B_RAW)
    assert_dev_only(done, "v16b completed")
    assert_dev_only(raw, "v16b raw")
    if done.get("passed") is not True and done.get("hard_pass") is not True:
        raise ContractError("v16b prerequisite did not pass")
    configs: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in list(raw.get("top_global_lobo") or [])[:20]:
        cfg = item.get("config")
        if isinstance(cfg, Mapping):
            configs[cfg_key(cfg)] = dict(cfg)
    for item in (raw.get("nested_outer") or {}).values():
        cfg = item.get("selected_config")
        if isinstance(cfg, Mapping):
            configs[cfg_key(cfg)] = dict(cfg)
    # Add four tightly scoped variants around the v16b best global threshold. This
    # is not a broad sweep: it tests whether the consensus veto can retain value
    # under slightly less/more conservative gain and support gates.
    best = (raw.get("headline") or {}).get("best_global_config")
    base = None
    for item in raw.get("top_global_lobo") or []:
        if item.get("config_id") == best and isinstance(item.get("config"), Mapping):
            base = dict(item["config"])
            break
    if base:
        for gain in [0.0, 0.25, 0.50]:
            for support in [0.75, 1.25]:
                c = dict(base)
                c["gain_lcb_min"] = gain
                c["support_mult"] = support
                configs[cfg_key(c)] = c
    hashes = {rel(V16B_DONE): sha256(V16B_DONE), rel(V16B_RAW): sha256(V16B_RAW), rel(SOURCE): sha256(SOURCE)}
    if ASTRA_REPORT.exists():
        hashes[rel(ASTRA_REPORT)] = sha256(ASTRA_REPORT)
    if ASTRA_MANIFEST.exists():
        hashes[rel(ASTRA_MANIFEST)] = sha256(ASTRA_MANIFEST)
    return raw, list(configs.values()), hashes


def rank_eval(ev: Mapping[str, Any]) -> Tuple[Any, ...]:
    bad = len(ev.get("catastrophic_false_positive_rows") or [])
    return (
        bad,
        0 if ev.get("physical_gate") else 1,
        not bool(ev.get("pass_10pct_no_cat_fp")),
        not bool(ev.get("pass_5pct_no_cat_fp")),
        -sf(ev.get("decision_relative_saving_vs_fixed_H15")),
        -int((ev.get("chosen_counts") or {}).get("10", 0)),
    )


def consensus_choices(cache: v16b.ScoreCache, split_id: str, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], mode: str = "unanimous") -> Tuple[Dict[str, int], Dict[str, Any]]:
    if not train_rows or not eval_rows:
        return {}, {"models": 0, "training_banks": [], "mode": mode}
    banks = sorted({str(r["bank_id"]) for r in train_rows})
    score_objs: List[Tuple[str, Dict[str, Any]]] = []
    agg = cache.scores(f"{split_id}|aggregate", train_rows, eval_rows, cfg)
    score_objs.append(("aggregate", agg))
    for b in banks:
        subtrain = [r for r in train_rows if str(r["bank_id"]) != b]
        if not subtrain:
            continue
        score_objs.append((f"minus_{b}", cache.scores(f"{split_id}|minus={b}", subtrain, eval_rows, cfg)))
    per_model_choices = [(name, v16b.choices_from_scores(obj["scores"], cfg)) for name, obj in score_objs]
    choices: Dict[str, int] = {}
    votes: Dict[str, Any] = {}
    for r in eval_rows:
        rk = v16.row_key(r)
        selected = [int(ch.get(rk, 15)) for _, ch in per_model_choices]
        h10_votes = sum(1 for h in selected if h == 10)
        required = len(selected) if mode == "unanimous" else max(1, len(selected) - 1)
        choices[rk] = 10 if h10_votes >= required else 15
        votes[rk] = {"h10_votes": h10_votes, "models": len(selected), "required": required, "model_choices": {name: int(ch.get(rk, 15)) for name, ch in per_model_choices}}
    return choices, {"models": len(per_model_choices), "training_banks": banks, "mode": mode, "votes_sample": {k: votes[k] for k in sorted(votes)[:20]}}


def eval_consensus(cache: v16b.ScoreCache, split_id: str, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], include_debug: bool = False) -> Dict[str, Any]:
    choices, debug = consensus_choices(cache, split_id, train_rows, eval_rows, cfg, mode="unanimous")
    ev = v16.eval_choices(eval_rows, choices)
    ev["consensus_debug"] = debug if include_debug else {k: v for k, v in debug.items() if k != "votes_sample"}
    ev["config_id"] = v16.cfg_id(cfg)
    return ev


def aggregate_holdouts(rows: Sequence[Mapping[str, Any]], holdouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    choices: Dict[str, int] = {}
    for ev in holdouts.values():
        for d in ev.get("details") or []:
            choices[str(d["unique_row_id"])] = int(d["selected_h"])
    return v16.eval_choices(rows, choices)


def select_inner(cache: v16b.ScoreCache, rows: Sequence[Mapping[str, Any]], banks: Sequence[str], cfgs: Sequence[Mapping[str, Any]], outer_bank: str) -> Tuple[Dict[str, Any], Mapping[str, Any], Mapping[str, Any], int]:
    train_outer = [r for r in rows if str(r["bank_id"]) != outer_bank]
    inner_banks = [b for b in banks if b != outer_bank]
    best_cfg: Optional[Dict[str, Any]] = None
    best_ev: Optional[Mapping[str, Any]] = None
    best_holdouts: Optional[Mapping[str, Any]] = None
    model_evals = 0
    for cfg in cfgs:
        holdouts: Dict[str, Any] = {}
        for hb in inner_banks:
            train = [r for r in train_outer if str(r["bank_id"]) != hb]
            evrows = [r for r in train_outer if str(r["bank_id"]) == hb]
            holdouts[hb] = eval_consensus(cache, f"inner_outer={outer_bank}|holdout={hb}|{v16.cfg_id(cfg)}", train, evrows, cfg)
            model_evals += 1
        agg = aggregate_holdouts(train_outer, holdouts)
        if best_ev is None or rank_eval(agg) < rank_eval(best_ev):
            best_cfg, best_ev, best_holdouts = dict(cfg), agg, holdouts
    assert best_cfg is not None and best_ev is not None and best_holdouts is not None
    return best_cfg, best_ev, best_holdouts, model_evals


def compact(ev: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(ev)
    if len(out.get("details") or []) > 20:
        out["details"] = list(out["details"][:20])
        out["details_truncated"] = True
    return out


def freeze_protocol(created: dt.datetime, cfgs: Sequence[Mapping[str, Any]], v16b_headline: Mapping[str, Any], hashes: Mapping[str, str], backup_commit: str) -> None:
    write_json(PROTOCOL, {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_bank_consensus_uncertainty_veto_no_sim_no_validation_no_test",
        "hypothesis": "A cross-bank consensus uncertainty veto over the v16b candidate configs will suppress held-out catastrophic H10 false positives if v16b failed mainly by model-selection/uncertainty, but will become too conservative if current deployable representation lacks bank-invariant risk information.",
        "inputs": {"v16b_raw": rel(V16B_RAW), "v16b_completed": rel(V16B_DONE), "v15_boundary_labels": rel(v16.V15_RAW)},
        "v16b_headline_before_v17": dict(v16b_headline),
        "candidate_config_count": len(cfgs),
        "candidate_config_ids": [v16.cfg_id(c) for c in cfgs],
        "split": "strict nested opened-bank: outer bank fully held out; inner model selection uses only remaining banks; H10 chosen on outer row only if aggregate and all leave-one-training-bank submodels agree",
        "decision_gate": {"zero_catastrophic_H10_false_positives": True, "physical_gate": True, "minimum_measured_decision_saving_vs_fixed_H15": 0.05},
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "candidate_configs": len(cfgs), "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "latest_verified_backup_before_run_from_supervisor_context": backup_commit,
        "input_hashes": dict(hashes),
    })


def write_astra_response_log(created: dt.datetime, hashes: Mapping[str, str], v16b_headline: Mapping[str, Any], v17_headline: Mapping[str, Any]) -> None:
    if not ASTRA_LATEST.exists() or not ASTRA_REPORT.exists():
        return
    rows = [
        ("A1_ORIGINAL_incomplete_not_final_success", "accepted", "SOURCE_MAP/run.py limitations and project status: ORIGINAL exact paper config/test files remain unavailable; no final-test success claimed.", "Continue reporting as core author-code reconstruction and/or IMPROVED only."),
        ("A2_current_vehicle_selector_not_original_SAC", "accepted", "Latency-tree/true-variable-H/v10-v17 scripts are finite search/refit diagnostics with no RL gradient training.", "All true-variable-H/risk selector results remain labeled IMPROVED."),
        ("A3_original_masked_AHMPC_does_not_reduce_NLP_dimension", "accepted", "Reviewer-cited controller mask behavior verified; case5 true-variable-H smoke showed opt_x H10 [182] < H15 [267] only for the IMPROVED true-H branch.", "Do not infer original AHMPC runtime reduction from action H; true-variable-H stays IMPROVED."),
        ("A4_offline_selector_savings_exclude_online_selector_overhead", "accepted", "v16b and v17 are offline branch-decision sums; no online selector overhead included.", "No speed claim until a blocked/randomized overhead smoke measures feature+selector+solver whole decision time."),
        ("A5_pendulum_three_seed_training_incomplete", "deferred", "Pendulum inventory still has s0 complete, s1 interrupted, s2 absent; vehicle remains prioritized.", "Do not claim pendulum completion; revisit after vehicle diagnostic line reaches a decision point."),
        ("A6_strong_fixed_H_and_terminal_opportunity_not_closed", "accepted/deferred", "Current v15-v17 compare mainly fixed true H15 in development; formal fixed-H grid/per-H terminal baselines remain required for final claims.", "Keep final claims gated on full strong baselines and independent validation/test."),
        ("A7_targeted_risk_banks_are_not_population_estimates", "accepted", "v15/v16b/v17 use mined opened boundary/development rows.", "Interpret only as mechanism/development diagnostics, not population estimates."),
        ("A8_zero_catastrophe_small_sample_model_selection_risk", "accepted", f"v16b in-sample bad=0/save={sf(v16b_headline.get('in_sample_save')):.4f} but strict nested bad={v16b_headline.get('nested_bad')}; v17 tests a consensus veto as follow-up.", "Do not use in-sample/global repair as confirmation."),
        ("A9_validation64_exposed_for_latency-tree_development", "accepted", "Existing validation64 was used for latency-tree development; current true-variable-H diagnostics avoid validation64 and sealed test.", "If method changes continue after exposure, require fresh independent confirmation before any final test request."),
        ("A10_runtime_semantic_modifications_affect_original_fidelity", "accepted", "Local runtime semantic fixes are documented in project protocol and reviewer report.", "Scope ORIGINAL claims to reconstructed core author code with local semantic fixes unless exact author config evidence is found."),
        ("A11_training_failure_modes_need_separation", "accepted/deferred", "Terminal/reward audits show terminal-profile sensitivity; v16b/v17 address representation/uncertainty without new gradients.", "If v17 fails or is too conservative, prioritize terminal/risk-value training/refit or controlled terminal-value ablation over more static sweeps."),
        ("A12_registry_backup_schema_contract", "accepted", "Schema false positives and backup requests are preserved; latest supervisor backup was verified before v17.", "Continue external backup gating after v17 artifacts."),
    ]
    lines = [
        "# Executor responses to independent reviews",
        "",
        f"Updated: {created.isoformat()} by GPT-5.5 executor after reading `{rel(ASTRA_LATEST)}` and `{rel(ASTRA_REPORT)}` at a safe boundary. This log records dispositions; it does not authorize sealed-test access or change acceptance criteria.",
        "",
        "## Report `20260929T153837Z`",
        "",
        f"Evidence hashes available this cycle include report `{hashes.get(rel(ASTRA_REPORT), 'unhashed')}`, manifest `{hashes.get(rel(ASTRA_MANIFEST), 'unhashed')}`, v16b raw `{hashes.get(rel(V16B_RAW), 'unhashed')}`. v17 follow-up headline: strict_nested_save={sf(v17_headline.get('nested_save')):.6f}, strict_nested_bad={v17_headline.get('nested_bad')}, pass5={v17_headline.get('nested_pass5')}.",
        "",
        "| ID | disposition | verified evidence | concrete action / result |",
        "|---|---|---|---|",
    ]
    for rid, disp, ev, action in rows:
        lines.append(f"| `{rid}` | {disp} | {ev} | {action} |")
    ASTRA_RESPONSE.parent.mkdir(parents=True, exist_ok=True)
    ASTRA_RESPONSE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v17 bank-consensus uncertainty veto

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline/refit diagnostic; no MPC simulation, no validation64/sealed-test access, no gradient training. candidate_configs={h['candidate_config_count']}; strict_nested_save={h['nested_save']:.6f}; strict_nested_solver_save={h['nested_solver_save']:.6f}; strict_nested_bad={h['nested_bad']}; strict_nested_h10={h['nested_h10']}; strict_nested_pass5={h['nested_pass5']}; cache_fits={h['cached_score_fits']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Astra response log updated at `{rel(ASTRA_RESPONSE)}` if latest review existed.
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
            f.write(f"{STAMP},{NAME},development_bank_consensus_uncertainty_veto,opened_rows_plus_v15_boundary_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['equivalent_consensus_model_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines = [
        "# Vehicle true-variable-H v17 bank-consensus uncertainty veto",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED offline/refit diagnostic; no MPC simulation, no validation64, no sealed test, no gradient/RL training.",
        "",
        "## Headline",
        "",
        f"- Candidate configs from v16b top/nested set plus tight variants: `{h['candidate_config_count']}`.",
        f"- v16b baseline strict nested: save `{pct(raw['v16b_headline'].get('nested_save'))}`, bad `{raw['v16b_headline'].get('nested_bad')}`, H10 `{raw['v16b_headline'].get('nested_h10')}`, pass5 `{raw['v16b_headline'].get('nested_pass5')}`.",
        f"- v17 consensus strict nested: save `{pct(h['nested_save'])}`, solver save `{pct(h['nested_solver_save'])}`, bad `{h['nested_bad']}`, H10 `{h['nested_h10']}`, physical gate `{h['nested_physical_gate']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.",
        f"- Cached score fits `{h['cached_score_fits']}`; equivalent consensus model evaluations `{h['equivalent_consensus_model_evaluations']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Strict nested outer-bank results",
        "",
        "| outer bank | selected config | inner save | inner bad | outer H counts | outer bad | outer save | outer solver save | outer physical gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, obj in raw["nested_outer"].items():
        ev = obj["outer_eval"]
        inn = obj["inner_eval"]
        lines.append(f"| `{b}` | `{obj['selected_config_id']}` | {pct(inn.get('decision_relative_saving_vs_fixed_H15'))} | {len(inn.get('catastrophic_false_positive_rows') or [])} | `{ev.get('chosen_counts')}` | {len(ev.get('catastrophic_false_positive_rows') or [])} | {pct(ev.get('decision_relative_saving_vs_fixed_H15'))} | {pct(ev.get('solver_relative_saving_vs_fixed_H15'))} | `{ev.get('physical_gate')}` |")
    fps = raw.get("nested_aggregate", {}).get("catastrophic_false_positive_rows") or []
    lines += ["", "## Strict nested catastrophic H10 false positives", ""]
    if fps:
        lines.append("| bank | row | source | origin | role | off | phys delta | decision gain s |")
        lines.append("|---|---|---|---|---|---:|---:|---:|")
        for r in fps[:40]:
            lines.append(f"| `{r.get('bank_id')}` | `{r.get('base_state_id')}` | `{r.get('source_key')}` | `{r.get('row_origin')}` | `{r.get('boundary_role')}` | {int(r.get('offset_from_center') or 0)} | {sf(r.get('phys_delta')):.6g} | {sf(r.get('decision_gain_s')):.6g} |")
    else:
        lines.append("No strict nested catastrophic false positives.")
    lines += [
        "",
        "## Interpretation",
        "",
        "This diagnostic tests one uncertainty/model-selection change, not a new validation claim. A pass would only justify an unused-source development confirmation with actual selector overhead. A zero-bad but <5% saving result means cross-bank uncertainty can avoid catastrophic H10 but current deployable representation/data cannot extract useful compute benefit. Any bad result means even consensus over the current representation is unsafe.",
        "",
        f"Protocol: `{rel(PROTOCOL)}`. Raw: `{rel(OUT / 'raw.json')}`. Completed: `{rel(OUT / 'completed.json')}`. Backup request: `{rel(BACKUP_REQUEST)}`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    v16b_raw, cfgs, hashes = load_v16b_and_candidates()
    rows, features, rows_diag, load_hashes = v16.load_augmented_rows()
    hashes.update(load_hashes)
    freeze_protocol(created, cfgs, v16b_raw.get("headline") or {}, hashes, args.backup_verified_commit)
    cache = v16b.ScoreCache(rows, features)
    banks = sorted({str(r["bank_id"]) for r in rows})
    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    equivalent = 0
    for outer in banks:
        selected_cfg, inner_eval, _inner_holdouts, evals = select_inner(cache, rows, banks, cfgs, outer)
        equivalent += evals
        train_outer = [r for r in rows if str(r["bank_id"]) != outer]
        eval_outer = [r for r in rows if str(r["bank_id"]) == outer]
        outer_eval = eval_consensus(cache, f"outer={outer}|{v16.cfg_id(selected_cfg)}", train_outer, eval_outer, selected_cfg, include_debug=True)
        equivalent += 1
        for d in outer_eval.get("details") or []:
            nested_choices[str(d["unique_row_id"])] = int(d["selected_h"])
        nested_outer[outer] = {"selected_config_id": v16.cfg_id(selected_cfg), "selected_config": selected_cfg, "inner_eval": compact(inner_eval), "outer_eval": compact(outer_eval)}
    nested_aggregate = v16.eval_choices(rows, nested_choices)
    h = {
        "candidate_config_count": len(cfgs),
        "banks": banks,
        "rows": len(rows),
        "old_rows": rows_diag.get("old_row_count"),
        "v15_boundary_rows": rows_diag.get("v15_boundary_row_count"),
        "nested_bad": len(nested_aggregate.get("catastrophic_false_positive_rows") or []),
        "nested_physical_gate": bool(nested_aggregate.get("physical_gate")),
        "nested_save": sf(nested_aggregate.get("decision_relative_saving_vs_fixed_H15")),
        "nested_solver_save": sf(nested_aggregate.get("solver_relative_saving_vs_fixed_H15")),
        "nested_h10": int((nested_aggregate.get("chosen_counts") or {}).get("10", 0)),
        "nested_pass5": bool(nested_aggregate.get("pass_5pct_no_cat_fp")),
        "nested_pass10": bool(nested_aggregate.get("pass_10pct_no_cat_fp")),
        "equivalent_consensus_model_evaluations": equivalent,
        "cached_score_fits": cache.fit_count,
    }
    if h["nested_pass5"] and h["nested_bad"] == 0:
        decision = "v17 bank-consensus veto meets the opened-development zero-catastrophe >=5% saving gate; after backup, freeze a small unused-source development confirmation with measured selector overhead before any validation64/test use."
    elif h["nested_bad"] == 0 and h["nested_physical_gate"]:
        decision = "v17 bank-consensus veto eliminates catastrophic H10 false positives but is too conservative for the >=5% gate; current deployable representation/data appear insufficient for useful safe compute savings. Pivot to terminal/risk-value learning or new source-independent state coverage, not another static selector sweep."
    else:
        decision = "v17 bank-consensus veto still permits catastrophic H10 or physical-gate failure; cross-bank uncertainty does not repair current representation. Prioritize terminal/risk-value training or observability/scenario diagnostics before any selector rollout."
    raw = {
        "created_utc": now().isoformat(),
        "elapsed_since_first_supervisor_event_seconds": (now() - FIRST_EVENT).total_seconds(),
        "classification": "development_IMPROVED_bank_consensus_uncertainty_veto_no_sim_no_validation_no_test",
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "v16b_headline": v16b_raw.get("headline") or {},
        "rows_diag": rows_diag,
        "headline": h,
        "decision": decision,
        "nested_outer": nested_outer,
        "nested_aggregate": nested_aggregate,
        "budget_declared": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "candidate_configs": len(cfgs), "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "budget_actual": {"development_mpc_simulation_episodes": 0, "development_control_steps": 0, "equivalent_consensus_model_evaluations": equivalent, "cached_score_fits": cache.fit_count, "training_episodes": 0, "gradient_steps": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "platform": {"python": sys.version, "platform": platform.platform()},
        "input_hashes": hashes,
        "interpretation_limits": ["opened development only", "candidate configs are outcome-informed by v16b development diagnostics", "no MPC simulation or online selector overhead measured", "not validation/test evidence", "IMPROVED, not ORIGINAL SAC"],
    }
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    args = ap.parse_args(argv)
    if not args.run:
        raise SystemExit("must pass --run")
    done_path = OUT / "completed.json"
    if done_path.exists():
        done = read_json(done_path)
        print(json.dumps({"already_completed": rel(done_path), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    write_json(BACKUP_REQUEST, {
        "requested_utc": raw["created_utc"],
        "reason": "backup v17 bank-consensus diagnostic, Astra response log and docs before further simulations/training/refit",
        "backup_required_before_more_science": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "artifacts": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(BACKUP_REQUEST), rel(SOURCE), rel(ASTRA_RESPONSE)],
        "budgets": raw["budget_actual"],
    })
    write_astra_response_log(dt.datetime.fromisoformat(raw["created_utc"]), raw["input_hashes"], raw["v16b_headline"], raw["headline"])
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text((OUT / "summary.md").read_text(encoding="utf-8") + "\nNext action: " + raw["decision"] + "\n", encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file()] + [PROTOCOL, STATE, BACKUP_REQUEST, SOURCE, ASTRA_RESPONSE]
    completed = {
        "passed": True,
        "hard_pass": True,
        "created_utc": raw["created_utc"],
        "elapsed_since_first_supervisor_event_seconds": raw["elapsed_since_first_supervisor_event_seconds"],
        "classification": raw["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "budget_actual": raw["budget_actual"],
        "headline": raw["headline"],
        "decision": raw["decision"],
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "protocol": rel(PROTOCOL),
        "state": rel(STATE),
        "backup_request": rel(BACKUP_REQUEST),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    }
    write_json(done_path, completed)
    print(json.dumps({"completed": rel(done_path), "summary": rel(OUT / "summary.md"), "headline": completed["headline"], "decision": completed["decision"], "validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": rel(BACKUP_REQUEST)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
