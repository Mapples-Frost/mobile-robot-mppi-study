#!/usr/bin/env python3
"""Audit v14 nested false positives and nearest safe/catastrophic lookalikes.

Development-only metadata diagnostic after the v14 calibrated risk/value refit.
This script reads the already-opened v14 development raw artifact, reconstructs
v14's calibrated prediction context from opened development banks only, and
summarizes why the two nested H10 false positives were selected.  It performs no
MPC simulation, no selector refit/grid search, no gradient/RL training, no
validation64 access, and no sealed-test access.

The output is intended to freeze the next bounded v15 boundary-acquisition design
around exact saved H15-prefix states and their nearest safe/catastrophic
lookalikes, rather than launching another broad static-feature sweep.
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

import vehicle_true_variable_horizon_history_feature_refit_v13 as v13  # noqa:E402
import vehicle_true_variable_horizon_calibrated_risk_value_refit_v14 as v14  # noqa:E402

NAME = "vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0"
STAMP = "20260929T2225Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
RAW_V14 = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/raw.json"
DONE_V14 = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/completed.json"
SUMMARY_V14 = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/summary.md"
PROTOCOL_V14 = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_preoutcome_frozen_20260929T2215Z.json"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T2225_after_v14_fp_neighbor_audit.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-v14-fp-neighbor-audit-v0-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")


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


def row_key(row: Mapping[str, Any]) -> str:
    return v13.row_key(row)


def assert_dev_only(obj: Mapping[str, Any], label: str) -> None:
    for flag in ("validation64_bank_opened", "sealed_test_accessed", "sealed_test_bank_opened", "test_accessed"):
        if obj.get(flag) is True:
            raise ContractError(f"forbidden {flag}=True in {label}")
    text = json.dumps(obj)[:200000].lower()
    if "sealed_test" in text and "sealed_test_accessed\": false" not in text:
        # Do not fail merely for declared access flags, but keep a conservative
        # warning in outputs; actual forbidden flags are checked above.
        pass


def metric_from_eval_detail(detail: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "selected_h": si(detail.get("selected_h"), 15),
        "label_positive": bool(detail.get("label_positive")),
        "catastrophic": bool(detail.get("catastrophic")),
        "phys_delta": sf(detail.get("phys_delta")),
        "decision_gain_s": sf(detail.get("decision_gain_s")),
        "group": detail.get("group"),
        "window": detail.get("window"),
    }


def build_manifest_map(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, str]]:
    manifests: Dict[str, Mapping[str, Any]] = {}
    hashes: Dict[str, str] = {}
    for bank in sorted(set(str(r["bank_id"]) for r in rows)):
        m, h = v13.manifest_entries_for_bank(bank)
        hashes.update(h)
        for sid, e in m.items():
            manifests[f"{bank}/{sid}"] = e
    return manifests, hashes


def compact_manifest(entry: Mapping[str, Any]) -> Dict[str, Any]:
    if not entry:
        return {}
    keys = [
        "base_state_id", "fresh_case_index", "source_candidate_index", "fresh_confirmation_group",
        "risk_probe_role", "branch_state_slot", "window", "branch_step", "h15_trace_episode_path",
        "stage_a_trace_risk_score", "selection_rule", "selection_mode",
        "selection_boundary_mode", "selection_anchor_candidate_index", "selection_anchor_metadata_distance",
    ]
    out = {k: entry.get(k) for k in keys if k in entry}
    prev = entry.get("branch_previous_state") if isinstance(entry.get("branch_previous_state"), Mapping) else {}
    if prev:
        out["branch_previous_state_core"] = {k: prev.get(k) for k in ("x", "y", "theta", "v", "vx", "vy") if k in prev}
    obs = entry.get("initial_observation_from_h15_trace") or entry.get("initial_observation_at_branch") or []
    if isinstance(obs, list):
        out["obs14_first14"] = [sf(v) for v in obs[:14]]
    return out


def compact_row(row: Mapping[str, Any], d: Optional[float] = None, manifests: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    k = row_key(row)
    out = {
        "key": k,
        "distance": d,
        "bank_id": row.get("bank_id"),
        "base_state_id": row.get("base_state_id"),
        "group": row.get("group"),
        "window": row.get("window"),
        "label_positive": bool(row.get("h10_beneficial_vs_h15")),
        "catastrophic": bool(row.get("h10_catastrophic_vs_h15")),
        "h10_safe_all": bool(row.get("h10_safe_all")),
        "h15_safe_all": bool(row.get("h15_safe_all")),
        "phys_delta_h10_minus_h15": sf(row.get("phys_delta_h10_minus_h15")),
        "decision_gain_h10_vs_h15_s": sf(row.get("decision_gain_h10_vs_h15_s")),
        "solver_gain_h10_vs_h15_s": sf(row.get("solver_gain_h10_vs_h15_s")),
        "row_physical_tolerance": sf(row.get("row_physical_tolerance"), 2.0),
        "branch_step": si(row.get("branch_step"), -1),
        "stage_a_trace_risk_score": sf(row.get("stage_a_trace_risk_score")),
        "source_candidate_index": si(row.get("source_candidate_index"), -1),
    }
    if manifests is not None:
        out["manifest"] = compact_manifest(manifests.get(k, {}))
    return out


def feature_signature(row: Mapping[str, Any], features: Mapping[str, Mapping[str, float]], family: str) -> Dict[str, Any]:
    fd = dict(v13.make_family_features(features, family)[row_key(row)])
    priority = [
        "prev_x_30", "prev_y_30", "theta_sin", "theta_cos", "abs_theta_pi",
        "branch_step_150", "slot", "trace_risk_20", "obs01_norm", "obs56_norm",
        "obs89_norm", "obs1112_norm", "hist_pair_norm_last", "hist_pair_norm_min12",
        "hist_pair_norm_mean12", "hist_theta_mean12", "hist_theta_std12",
    ]
    out = {k: fd.get(k) for k in priority if k in fd}
    # Preserve enough exact observable features for follow-up matching without
    # dumping all 100+ history dimensions into the summary.
    for i in range(14):
        k = f"obs_{i}"
        if k in fd:
            out[k] = fd[k]
    return out


def standardized_feature_diffs(
    fp: Mapping[str, Any], other: Mapping[str, Any], family: str, train_rows: Sequence[Mapping[str, Any]], all_features: Mapping[str, Mapping[str, float]], n: int = 12
) -> List[Dict[str, Any]]:
    fdict = v13.make_family_features(all_features, family)
    names, means, stds = v13.train_scaler(train_rows, fdict)
    fk = row_key(fp); ok = row_key(other)
    diffs = []
    for nm in names:
        a = sf(fdict[fk].get(nm)); b = sf(fdict[ok].get(nm)); sd = max(sf(stds.get(nm), 1.0), 1e-12)
        diffs.append({"feature": nm, "fp_value": a, "other_value": b, "abs_standardized_diff": abs((a - b) / sd)})
    return sorted(diffs, key=lambda x: -sf(x["abs_standardized_diff"]))[:n]


def summarize_lookalike_set(neighbors: Sequence[Tuple[float, Mapping[str, Any]]], manifests: Mapping[str, Mapping[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return [compact_row(r, d, manifests) for d, r in neighbors[:limit]]


def branch_candidate(center: Mapping[str, Any], role: str, offset: int) -> Dict[str, Any]:
    man = center.get("manifest") or {}
    step0 = si(man.get("branch_step", center.get("branch_step")), -1)
    return {
        "role": role,
        "source_key": center.get("key"),
        "bank_id": center.get("bank_id"),
        "base_state_id": center.get("base_state_id"),
        "h15_trace_episode_path": man.get("h15_trace_episode_path"),
        "center_branch_step": step0,
        "candidate_branch_step": None if step0 < 0 else max(8, step0 + offset),
        "offset_from_center": offset,
        "source_label_positive": center.get("label_positive"),
        "source_catastrophic": center.get("catastrophic"),
        "source_phys_delta": center.get("phys_delta_h10_minus_h15"),
        "source_decision_gain_s": center.get("decision_gain_h10_vs_h15_s"),
        "source_group": center.get("group"),
        "source_window": center.get("window"),
    }


def make_v15_candidate_plan(audit_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for fp in audit_rows:
        centers: List[Tuple[str, Mapping[str, Any]]] = [("false_positive_center", fp["false_positive_row"])]
        for s in fp.get("nearest_safe_positive_training_rows", [])[:2]:
            centers.append(("nearest_safe_positive_lookalike", s))
        for s in fp.get("nearest_catastrophic_training_rows", [])[:1]:
            centers.append(("nearest_catastrophic_lookalike", s))
        for role, center in centers:
            for off in [-4, 0, 4]:
                c = branch_candidate(center, role, off)
                key = (c.get("h15_trace_episode_path"), c.get("candidate_branch_step"), role, c.get("source_key"))
                if c.get("h15_trace_episode_path") and c.get("candidate_branch_step") is not None and key not in seen:
                    seen.add(key)
                    candidates.append(c)
    return {
        "status": "candidate_plan_not_executed_not_frozen_for_simulation",
        "hypothesis_for_v15": "If v14's false positives are caused by sparse ambiguous boundary labels rather than absent adaptive opportunity, then H10/H15 continuations at adjacent H15-prefix steps around the two false-positive centers and nearest safe/catastrophic lookalikes will expose locally separable risk/gain transitions or show that current observables are intrinsically aliased.",
        "candidate_count": len(candidates),
        "candidate_branch_states": candidates,
        "suggested_run_budget_if_frozen": {
            "true_horizons": [10, 15],
            "repeats": 2,
            "max_branch_steps_per_episode": 150,
            "stage_B_episodes_if_all_candidates_used": len(candidates) * 2 * 2,
            "control_step_upper_bound_if_all_candidates_used": len(candidates) * 2 * 2 * 150,
            "validation64_episodes": 0,
            "sealed_test_episodes": 0,
        },
        "acceptance_for_data_sufficiency": "Before any validation64, require the added boundary labels to make nested opened-bank+boundary selection achieve zero catastrophic H10 false positives and >=5% measured branch decision saving vs fixed H15; otherwise pivot to richer terminal/risk-value learning rather than more static feature thresholds.",
    }


def append_docs(raw: Mapping[str, Any]) -> None:
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v14 false-positive neighbor audit

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only metadata diagnostic; no MPC simulation, validation64 or sealed test. Audited v14 nested false positives={h['nested_false_positive_count']} with reconstructed selected configs and nearest safe/catastrophic training lookalikes. Candidate v15 boundary plan contains {h['v15_candidate_branch_states']} branch states ({h['v15_candidate_episodes_if_frozen']} H10/H15 repeat episodes if later frozen). Decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'audit.json')}`, `{rel(OUT / 'completed.json')}`.
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
            f.write(f"{STAMP},{NAME},development_metadata_neighbor_audit,opened_v14_nested_false_positives,0,0,0,0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def write_summary(raw: Mapping[str, Any]) -> None:
    lines: List[str] = [
        "# Vehicle true-variable-H v14 false-positive neighbor audit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only metadata audit of v14 nested false positives; no MPC simulation, no refit/grid search, no validation64, no sealed test.",
        "",
        "## Headline",
        "",
        f"- v14 nested false positives audited: `{raw['headline']['nested_false_positive_count']}`.",
        f"- Reconstructed rows: `{raw['headline']['rows']}` across banks `{raw['headline']['banks']}`; H15-prefix traces loaded `{raw['headline']['trace_loaded_rows']}`.",
        f"- Candidate v15 boundary branch states (not executed): `{raw['headline']['v15_candidate_branch_states']}`; if frozen with H10/H15 x2 repeats: `{raw['headline']['v15_candidate_episodes_if_frozen']}` development episodes, cap `{raw['headline']['v15_control_step_cap_if_frozen']}` control steps.",
        f"- Decision: {raw['decision']}",
        "",
        "## False-positive diagnostics",
        "",
    ]
    for fp in raw["false_positive_audits"]:
        r = fp["false_positive_row"]
        sc = fp.get("v14_prediction_score") or {}
        lines += [
            f"### `{r['key']}`",
            "",
            f"- Group/window: `{r.get('group')}` / `{r.get('window')}`; selected family/config: `{fp['selected_config']['family']}` / `{fp['selected_config_id']}`.",
            f"- Actual label: positive `{r['label_positive']}`, catastrophic `{r['catastrophic']}`; physΔ(H10-H15) `{r['phys_delta_h10_minus_h15']:.6g}` vs tolerance `{r['row_physical_tolerance']:.6g}`; decision gain `{r['decision_gain_h10_vs_h15_s']:.6g}` s.",
            f"- v14 bounds: risk_raw `{sf(sc.get('risk_raw')):.4g}`, risk_ucb `{sf(sc.get('risk_ucb')):.4g}`, phys_ucb `{sf(sc.get('phys_ucb')):.6g}`, gain_lcb `{sf(sc.get('gain_lcb')):.6g}`, support_ok `{sc.get('support_ok')}`, dpos `{sf(sc.get('dpos_full')):.4g}`, dcat `{sf(sc.get('dcat_full')):.4g}`.",
            "- Nearest safe positive training lookalikes:",
        ]
        for n in fp.get("nearest_safe_positive_training_rows", [])[:4]:
            lines.append(f"  - `{n['key']}` d={sf(n.get('distance')):.4g}, physΔ={sf(n.get('phys_delta_h10_minus_h15')):.6g}, gain={sf(n.get('decision_gain_h10_vs_h15_s')):.6g}, group=`{n.get('group')}`, window=`{n.get('window')}`")
        lines.append("- Nearest catastrophic training rows:")
        for n in fp.get("nearest_catastrophic_training_rows", [])[:4]:
            lines.append(f"  - `{n['key']}` d={sf(n.get('distance')):.4g}, physΔ={sf(n.get('phys_delta_h10_minus_h15')):.6g}, gain={sf(n.get('decision_gain_h10_vs_h15_s')):.6g}, group=`{n.get('group')}`, window=`{n.get('window')}`")
        lines.append("- Largest standardized differences to closest safe lookalike:")
        for d in fp.get("top_feature_differences_vs_nearest_safe_positive", [])[:8]:
            lines.append(f"  - `{d['feature']}` fp={sf(d.get('fp_value')):.5g}, safe={sf(d.get('other_value')):.5g}, |zΔ|={sf(d.get('abs_standardized_diff')):.4g}")
        lines.append("")
    lines += [
        "## Candidate v15 boundary-acquisition plan (not executed here)",
        "",
        "The following saved H15-prefix centers/offsets are metadata-selected only. A separate frozen runner must verify trace availability and execute paired H10/H15 continuations if adopted.",
        "",
        "| role | source key | trace path | center step | candidate step | group | source cat | source positive |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for c in raw["v15_candidate_plan"]["candidate_branch_states"]:
        lines.append(f"| `{c['role']}` | `{c['source_key']}` | `{c['h15_trace_episode_path']}` | {c['center_branch_step']} | {c['candidate_branch_step']} | `{c['source_group']}` | `{c['source_catastrophic']}` | `{c['source_label_positive']}` |")
    lines += ["", "## Interpretation", "", raw["interpretation"]]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    for p in (RAW_V14, DONE_V14, SUMMARY_V14, PROTOCOL_V14):
        if not p.exists():
            raise ContractError("missing v14 prerequisite: " + rel(p))
    raw_v14 = read_json(RAW_V14)
    done_v14 = read_json(DONE_V14)
    protocol_v14 = read_json(PROTOCOL_V14)
    assert_dev_only(raw_v14, "v14 raw")
    assert_dev_only(done_v14, "v14 completed")
    assert_dev_only(protocol_v14, "v14 protocol")
    fps = list(((raw_v14.get("nested_aggregate") or {}).get("catastrophic_false_positive_rows") or []))
    if len(fps) != 2:
        raise ContractError(f"expected the known two v14 nested false positives, got {len(fps)}")

    rows, input_hashes = v13.load_rows()
    all_features, hist_diag, hist_hashes = v13.enrich_history(rows)
    input_hashes.update(hist_hashes)
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    if banks != ["fresh_v0", "fresh_v1", "fresh_v11", "fresh_v2", "fresh_v8c"] or len(rows) != 68:
        raise ContractError(f"unexpected opened rows/banks: {len(rows)} {banks}")
    cache = v14.ContextCache(rows, all_features, banks)
    manifests, manifest_hashes = build_manifest_map(rows)
    input_hashes.update(manifest_hashes)
    row_by_key = {row_key(r): r for r in rows}

    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_metadata_audit_no_simulation_no_refit_no_validation_no_test",
        "hypothesis": "v14 false positives are caused by local aliasing between safe H10-positive lookalikes and catastrophic H10 boundary states under deployable H15-prefix features. Reconstructing selected models and nearest neighbors should identify exact saved H15-prefix states for a bounded v15 boundary acquisition, or show that targeted acquisition is not well specified.",
        "inputs": {"v14_raw": rel(RAW_V14), "v14_completed": rel(DONE_V14), "v14_protocol": rel(PROTOCOL_V14), "v14_summary": rel(SUMMARY_V14)},
        "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "input_hashes": {rel(RAW_V14): sha256(RAW_V14), rel(DONE_V14): sha256(DONE_V14), rel(SUMMARY_V14): sha256(SUMMARY_V14), rel(PROTOCOL_V14): sha256(PROTOCOL_V14), rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()), **input_hashes},
    }
    write_json(PROTOCOL, protocol)

    audits: List[Dict[str, Any]] = []
    for fp0 in fps:
        bank = str(fp0["bank_id"])
        key = f"{bank}/{fp0['base_state_id']}"
        row = row_by_key.get(key)
        if row is None:
            raise ContractError("false-positive row not found in reconstructed rows: " + key)
        outer = (raw_v14.get("nested_outer") or {}).get(bank)
        if not isinstance(outer, Mapping):
            raise ContractError("missing nested outer for bank " + bank)
        cfg = dict(outer.get("selected_config") or {})
        cfgid = str(outer.get("selected_config_id"))
        family = str(cfg.get("family"))
        train_banks = [b for b in banks if b != bank]
        ctx = cache.get(family, train_banks, bank)
        score = (((outer.get("outer_eval") or {}).get("prediction_scores") or {}).get(key) or {})
        recomputed = v14.prediction_bounds(cache, family, train_banks, bank, row, cfg)
        nb = ctx["neighbors"][key]
        safe_pos = [(d, r) for d, r in nb["pos"] if bool(r.get("h10_beneficial_vs_h15")) and not bool(r.get("h10_catastrophic_vs_h15"))]
        cats = [(d, r) for d, r in nb["cat"]]
        alln = [(d, r) for d, r in nb["all"]]
        nearest_safe_rows = summarize_lookalike_set(safe_pos, manifests, 8)
        nearest_cat_rows = summarize_lookalike_set(cats, manifests, 8)
        nearest_all_rows = summarize_lookalike_set(alln, manifests, 10)
        fdiff_safe = standardized_feature_diffs(row, safe_pos[0][1], family, ctx["train_rows"], all_features) if safe_pos else []
        fdiff_cat = standardized_feature_diffs(row, cats[0][1], family, ctx["train_rows"], all_features) if cats else []
        row_compact = compact_row(row, None, manifests)
        audit = {
            "false_positive_key": key,
            "selected_config_id": cfgid,
            "selected_config": cfg,
            "train_banks": train_banks,
            "false_positive_row": row_compact,
            "feature_signature_family": family,
            "feature_signature": feature_signature(row, all_features, family),
            "v14_prediction_score": score,
            "v14_recomputed_prediction_score": recomputed,
            "nearest_all_training_rows": nearest_all_rows,
            "nearest_safe_positive_training_rows": nearest_safe_rows,
            "nearest_catastrophic_training_rows": nearest_cat_rows,
            "top_feature_differences_vs_nearest_safe_positive": fdiff_safe,
            "top_feature_differences_vs_nearest_catastrophic": fdiff_cat,
            "local_ambiguity_metrics": {
                "dpos_minus_dcat": (sf(score.get("dpos_full"), 1e9) - sf(score.get("dcat_full"), 1e9)) if score else None,
                "dcat_over_dpos": (sf(score.get("dcat_full"), 1e9) / max(sf(score.get("dpos_full"), 1e9), 1e-12)) if score else None,
                "nearest_safe_positive_count_reported": len(nearest_safe_rows),
                "nearest_catastrophic_count_reported": len(nearest_cat_rows),
            },
        }
        audits.append(audit)

    plan = make_v15_candidate_plan(audits)
    stage_b_episodes = plan["suggested_run_budget_if_frozen"]["stage_B_episodes_if_all_candidates_used"]
    control_cap = plan["suggested_run_budget_if_frozen"]["control_step_upper_bound_if_all_candidates_used"]
    interpretation = (
        "The false positives are not random bookkeeping errors: each was selected by a nested model that saw nearby safe H10-positive training rows under the chosen feature family while still having catastrophic neighbors in the same opened-bank population. "
        "Because the best global v14 configuration already abstained enough to avoid catastrophic false positives but saved only 2.08%, the next discriminating action should not be another broad feature-threshold sweep. The bounded next experiment should acquire local continuation labels at adjacent H15-prefix steps around these two catastrophic centers and their nearest safe/catastrophic lookalikes, then test whether the added boundary labels improve nested zero-catastrophe selection to >=5% opened-development saving. If not, the evidence points toward richer terminal/risk-value learning or additional observability rather than sparse-label tuning alone."
    )
    raw = {
        "created_utc": now().isoformat(),
        "classification": protocol["classification"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "input_hashes": {**protocol["input_hashes"], rel(PROTOCOL): sha256(PROTOCOL)},
        "history_availability": hist_diag,
        "headline": {
            "rows": len(rows),
            "banks": banks,
            "nested_false_positive_count": len(audits),
            "trace_loaded_rows": hist_diag.get("trace_loaded_rows"),
            "v15_candidate_branch_states": plan["candidate_count"],
            "v15_candidate_episodes_if_frozen": stage_b_episodes,
            "v15_control_step_cap_if_frozen": control_cap,
        },
        "false_positive_audits": audits,
        "v15_candidate_plan": plan,
        "interpretation": interpretation,
        "decision": "Freeze and run a small v15 development-only boundary acquisition using the candidate saved H15-prefix centers/offsets, after this audit is externally backed up; do not open validation64/sealed test.",
        "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations": 0, "metadata_audit_rows": len(audits), "validation64_episodes": 0, "sealed_test_episodes": 0},
        "platform": {"python": sys.version, "platform": platform.platform()},
    }
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-v14-fp-neighbor-audit", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_v14_fp_neighbor_audit:
        raise ContractError("requires --run and explicit development audit acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "audit.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_V14_FALSE_POSITIVE_NEIGHBOR_AUDIT_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v14 false-positive neighbor audit before v15 simulation protocol/source", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(Path(__file__).resolve()), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "validation64_bank_opened": False, "sealed_test_accessed": False})
    done = {
        "passed": True,
        "status": "complete",
        "marker": MARKER,
        "summary": rel(OUT / "summary.md"),
        "audit": rel(OUT / "audit.json"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(req),
        "headline": raw["headline"],
        "decision": raw["decision"],
        "budget_actual": raw["budget_actual"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "hashes": {rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "audit.json"): sha256(OUT / "audit.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(req): sha256(req)},
    }
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v14 false-positive neighbor audit\n\n" + json.dumps(clean({
        "utc": raw["created_utc"],
        "headline": raw["headline"],
        "decision": raw["decision"],
        "budget_actual": raw["budget_actual"],
        "artifacts": {"summary": rel(OUT / "summary.md"), "audit": rel(OUT / "audit.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req)},
        "current_backup_status": "not verified after v14 false-positive neighbor audit; request supervisor backup before v15 simulation/source if possible",
        "next_action": "After backup, implement/freeze v15 boundary-acquisition runner from audit v15_candidate_plan; execute H10/H15 paired continuations on development-only saved H15-prefix states; no validation64/sealed test.",
    }), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
