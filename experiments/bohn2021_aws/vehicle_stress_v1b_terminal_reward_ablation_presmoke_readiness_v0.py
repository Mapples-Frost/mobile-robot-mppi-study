#!/usr/bin/env python3
"""No-simulation pre-smoke readiness diagnostic for vehicle stress-v1b terminal/reward ablation.

This intentionally does NOT repeat the prefix-artifact audit and does NOT run any
rollouts.  It uses the already-frozen terminal/reward ablation dry-run and the
compact corrected-prefix labels to answer the bounded question left by the prior
cycle: are the inputs sufficient to run the smoke now, and what exactly will the
smoke be able to discriminate?

It preserves a durable state handoff and a backup request.  It opens no
validation64 bank and no sealed test bank.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260928T2325Z"
DRYRUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z"
DRYRUN_RAW = DRYRUN_DIR / "raw.json"
DRYRUN_DONE = DRYRUN_DIR / "completed.json"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_20260928T2320Z.json"
PREFIX_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z/raw.json"
PREFIX_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
OUT = ROOT / f"research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_presmoke_readiness_v0_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_terminal_reward_presmoke_readiness.md"
REQUEST = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_PRESMOKE_READINESS_V0_{STAMP}.json"
DOCS = ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]
MARKER = f"vehicle-stress-v1b-terminal-reward-ablation-presmoke-readiness-v0-{STAMP}"
FIRST_SUPERVISOR = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def proof_time(obj: Mapping[str, Any]) -> Optional[dt.datetime]:
    for k in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp", "proof_created_utc", "reported_time_utc"):
        t = parse_time(obj.get(k))
        if t is not None:
            return t
    return None


def completed_ok(path: Path) -> Dict[str, Any]:
    obj = read_json(path)
    if obj.get("passed") is not True and obj.get("hard_pass") is not True:
        raise RuntimeError(f"completed marker not passed: {rel(path)}")
    if obj.get("sealed_test_accessed") is not False:
        raise RuntimeError(f"sealed-test flag not false in {rel(path)}")
    if obj.get("validation64_bank_opened") not in (False, None) or obj.get("historical_validation64_bank_opened") not in (False, None):
        raise RuntimeError(f"validation64 flag not false in {rel(path)}")
    return obj


def scan_backups(min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    rows = []
    adequate = []
    latest = None
    for p in sorted(BACKUP_DIR.glob("backup_proof_*.json")):
        try:
            obj = read_json(p)
        except Exception as exc:
            rows.append({"path": rel(p), "json_ok": False, "error": repr(exc)})
            continue
        t = proof_time(obj)
        packages = obj.get("packages_this_run") or []
        has_pkg = bool(obj.get("package_sha256") or obj.get("asset_sha256") or obj.get("release_asset_sha256") or packages)
        row = {
            "path": rel(p),
            "time": None if t is None else t.isoformat(),
            "status": obj.get("status"),
            "backup_verified": obj.get("backup_verified"),
            "remaining_changed_files": obj.get("remaining_changed_files"),
            "commit": obj.get("commit"),
            "has_package_sha_or_packages": has_pkg,
            "after_required_time": bool(t is not None and min_time is not None and t >= min_time),
            "sha256": sha256(p),
        }
        rows.append(row)
        if t is not None and (latest is None or t > latest[0]):
            latest = (t, row)
        if (obj.get("backup_verified") is True or obj.get("status") == "verified") and obj.get("remaining_changed_files") == 0 and has_pkg and min_time is not None and t is not None and t >= min_time:
            adequate.append(row)
    return {
        "required_not_before_utc": None if min_time is None else min_time.isoformat(),
        "local_proof_count": len(rows),
        "latest_local_proof": None if latest is None else latest[1],
        "adequate_local_proof_count": len(adequate),
        "adequate_local_proofs": adequate,
        "post_required_or_latest_tail": rows[-8:],
    }


def horizon_of(d: Any) -> Optional[int]:
    if not isinstance(d, Mapping):
        return None
    for k in ("horizon", "H", "h"):
        if k in d:
            try:
                return int(d[k])
            except Exception:
                pass
    return None


def numeric_fields(d: Any, names: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(d, Mapping):
        return out
    wanted = tuple(names)
    for k, v in d.items():
        lk = str(k).lower()
        if any(w in lk for w in wanted):
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                out[str(k)] = fv
    return out


def summarize_state_specs(protocol: Mapping[str, Any], prefix_raw: Mapping[str, Any]) -> Dict[str, Any]:
    corrected_rows = (prefix_raw.get("analysis") or {}).get("corrected_rows") or []
    row_by_target = {int(r.get("target_index")): r for r in corrected_rows if "target_index" in r}
    state_specs = protocol.get("state_specs") or []
    smoke_plan = protocol.get("smoke_plan") or []
    full_plan = protocol.get("full_plan") or []
    smoke_targets = sorted({int(x["target_index"]) for x in smoke_plan})
    full_targets = sorted({int(x["target_index"]) for x in full_plan})
    positive_targets = sorted(int(s["target_index"]) for s in state_specs if s.get("state_role") == "corrected_positive")
    control_targets = sorted(int(s["target_index"]) for s in state_specs if s.get("state_role") != "corrected_positive")
    rows = []
    for s in state_specs:
        target = int(s["target_index"])
        r = row_by_target.get(target, {})
        best_phys = r.get("best_non_H15_physical") or s.get("best_physical_from_v1b") or {}
        best_total = r.get("best_non_H15_total") or s.get("best_total_from_v1b") or {}
        hp, ht = horizon_of(best_phys), horizon_of(best_total)
        rows.append({
            "target_index": target,
            "case": int(s.get("case", -1)),
            "state_id": s.get("state_id"),
            "role": s.get("state_role"),
            "in_smoke": target in smoke_targets,
            "candidate_horizons": [int(h) for h in (s.get("candidate_horizons") or [])],
            "material_horizons_from_v1b": [int(h) for h in (s.get("material_horizons_from_v1b") or [])],
            "best_physical_horizon": hp,
            "best_total_horizon": ht,
            "best_physical_total_horizon_disagree": bool(hp is not None and ht is not None and hp != ht),
            "best_physical_numeric_fields": numeric_fields(best_phys, ("gain", "cost", "delta")),
            "best_total_numeric_fields": numeric_fields(best_total, ("gain", "cost", "delta")),
        })
    mode_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    pair_counts: Dict[str, int] = {}
    refs_by_state: Dict[str, int] = {}
    for item in smoke_plan:
        mode = str(item.get("terminal_mode"))
        role = str(item.get("role"))
        h = int(item.get("horizon"))
        sid = str(item.get("state_id"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        pair_counts[f"H{h}_{mode}"] = pair_counts.get(f"H{h}_{mode}", 0) + 1
        if h == 15 and mode == "per_h":
            refs_by_state[sid] = refs_by_state.get(sid, 0) + 1
    smoke_positive_targets = sorted(set(smoke_targets).intersection(positive_targets))
    smoke_control_targets = sorted(set(smoke_targets).intersection(control_targets))
    positive_cases_in_smoke = sorted({int(s["case"]) for s in state_specs if int(s["target_index"]) in smoke_positive_targets})
    positive_cases_all = sorted({int(s["case"]) for s in state_specs if int(s["target_index"]) in positive_targets})
    smoke_has_reference_for_each_state = all(refs_by_state.get(str(item.get("state_id")), 0) >= 1 for item in smoke_plan)
    # A priori risk tags based only on old labels/protocol, not on new outcomes.
    risk_tags = []
    if len(positive_targets) < 6 or len(positive_cases_all) < 3:
        risk_tags.append("sparse_development_mined_positive_labels")
    if any(r["best_physical_total_horizon_disagree"] for r in rows if r["role"] == "corrected_positive"):
        risk_tags.append("physical_vs_total_best_horizon_disagreement")
    if any(len(r["material_horizons_from_v1b"]) >= 4 for r in rows if r["role"] == "corrected_positive"):
        risk_tags.append("multi_horizon_material_labels_possible_terminal_or_path_sensitivity")
    if set(positive_cases_in_smoke) == set(positive_cases_all) and len(smoke_positive_targets) >= 2:
        smoke_case_coverage = "covers_each_positive_case_once_or_more"
    else:
        smoke_case_coverage = "incomplete_positive_case_coverage"
    return {
        "corrected_label_headline": (prefix_raw.get("analysis") or {}).get("corrected_label_headline"),
        "state_rows": rows,
        "positive_targets_all": positive_targets,
        "control_targets_all": control_targets,
        "smoke_targets": smoke_targets,
        "full_targets": full_targets,
        "positive_targets_in_smoke": smoke_positive_targets,
        "positive_targets_omitted_from_smoke": sorted(set(positive_targets) - set(smoke_positive_targets)),
        "control_targets_in_smoke": smoke_control_targets,
        "positive_cases_all": positive_cases_all,
        "positive_cases_in_smoke": positive_cases_in_smoke,
        "smoke_case_coverage": smoke_case_coverage,
        "smoke_plan_episode_count": len(smoke_plan),
        "full_plan_episode_count": len(full_plan),
        "smoke_terminal_mode_counts": mode_counts,
        "smoke_role_counts": role_counts,
        "smoke_horizon_terminal_pair_counts": pair_counts,
        "smoke_has_H15_per_h_reference_for_each_scheduled_state": smoke_has_reference_for_each_state,
        "a_priori_risk_tags": risk_tags,
        "smoke_interpretation_power": {
            "can_detect_terminal_source_artifact": "yes, for scheduled candidate horizons via per_h vs h15/h25/zero terminal rows on positive states",
            "can_detect_synthetic_reward_artifact": "yes, by comparing physical-gain materiality vs total-gain materiality and synthetic component",
            "can_establish_generalization": "no, development-mined and only 27 episodes over four states",
            "can_justify_training_alone": "no, only if labels robust and later density/fresh-confirmation gates pass",
        },
    }


def append_once(path: Path, marker: str, text: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    if old and not old.endswith("\n"):
        old += "\n"
    path.write_text(old + "\n" + text.strip() + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc)
    for p in (DRYRUN_RAW, DRYRUN_DONE, PROTOCOL_JSON, PREFIX_RAW, PREFIX_DONE):
        if not p.exists():
            raise RuntimeError(f"missing required input: {rel(p)}")
    dry_done = completed_ok(DRYRUN_DONE)
    prefix_done = completed_ok(PREFIX_DONE)
    dry_time = parse_time(dry_done.get("created_utc"))
    source_time = dt.datetime.fromtimestamp(Path(__file__).stat().st_mtime, dt.timezone.utc)
    required_backup_after = max([t for t in (dry_time, source_time) if t is not None])
    protocol = read_json(PROTOCOL_JSON)
    prefix_raw = read_json(PREFIX_RAW)
    plan_summary = summarize_state_specs(protocol, prefix_raw)
    backups = scan_backups(required_backup_after)
    smoke_inputs_suffice_now = backups["adequate_local_proof_count"] > 0
    next_action = "run_smoke_with_latest_adequate_backup" if smoke_inputs_suffice_now else "await_verified_external_backup_then_run_smoke"
    raw = {
        "created_utc": created.isoformat(),
        "classification": "development_no_simulation_presmoke_readiness_diagnostic_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "inputs": {
            "dryrun_completed": rel(DRYRUN_DONE),
            "dryrun_created_utc": dry_done.get("created_utc"),
            "dryrun_sha256": sha256(DRYRUN_DONE),
            "protocol_json": rel(PROTOCOL_JSON),
            "protocol_sha256": sha256(PROTOCOL_JSON),
            "prefix_completed": rel(PREFIX_DONE),
            "prefix_sha256": sha256(PREFIX_DONE),
            "script_source_mtime_utc": source_time.isoformat(),
        },
        "required_backup_after_utc": required_backup_after.isoformat(),
        "backup_readiness": backups,
        "smoke_inputs_suffice_now": smoke_inputs_suffice_now,
        "plan_summary": plan_summary,
        "decision": {
            "run_smoke_now": False,
            "reason": "No simulation is run by this diagnostic. If no adequate local proof postdates the dry-run and this readiness source, the predeclared smoke remains backup-blocked.",
            "next_action": next_action,
            "why_this_diagnostic": "Concrete no-simulation check because rollout smoke inputs require an external backup that is not locally adequate; avoids re-running the prefix audit and quantifies smoke discriminating power.",
        },
    }
    write_json(OUT / "raw.json", raw)
    lines = [
        "# Vehicle stress-v1b terminal/reward ablation pre-smoke readiness v0",
        "",
        f"UTC: `{created.isoformat()}`. No simulation, no training/refit, no validation64 bank, no sealed test.",
        "",
        "## Backup gate",
        "",
        f"- Required backup not before: `{required_backup_after.isoformat()}`.",
        f"- Adequate local backup proofs found: `{backups['adequate_local_proof_count']}`.",
        f"- Latest local proof: `{backups['latest_local_proof']}`.",
        f"- Smoke inputs suffice now: `{smoke_inputs_suffice_now}`.",
        "",
        "## Frozen smoke diagnostic coverage",
        "",
        f"- Smoke episodes: `{plan_summary['smoke_plan_episode_count']}`; full episodes: `{plan_summary['full_plan_episode_count']}`.",
        f"- Positive targets all/smoke/omitted: `{plan_summary['positive_targets_all']}` / `{plan_summary['positive_targets_in_smoke']}` / `{plan_summary['positive_targets_omitted_from_smoke']}`.",
        f"- Positive cases all/smoke: `{plan_summary['positive_cases_all']}` / `{plan_summary['positive_cases_in_smoke']}` (`{plan_summary['smoke_case_coverage']}`).",
        f"- Control targets in smoke: `{plan_summary['control_targets_in_smoke']}`.",
        f"- Smoke terminal-mode counts: `{plan_summary['smoke_terminal_mode_counts']}`.",
        f"- H15 per-H reference for every scheduled state: `{plan_summary['smoke_has_H15_per_h_reference_for_each_scheduled_state']}`.",
        f"- A-priori risk tags: `{plan_summary['a_priori_risk_tags']}`.",
        "",
        "## State-level pre-smoke rows",
        "",
        "| target | case | role | in smoke | candidates | material H | best phys H | best total H | disagree |",
        "|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for r in plan_summary["state_rows"]:
        lines.append("| %d | %d | `%s` | `%s` | `%s` | `%s` | %s | %s | `%s` |" % (
            r["target_index"], r["case"], r["role"], r["in_smoke"], r["candidate_horizons"], r["material_horizons_from_v1b"],
            "NA" if r["best_physical_horizon"] is None else str(r["best_physical_horizon"]),
            "NA" if r["best_total_horizon"] is None else str(r["best_total_horizon"]),
            r["best_physical_total_horizon_disagree"],
        ))
    lines += [
        "",
        "## Decision",
        "",
        f"Next action: `{next_action}`. Do not run smoke until a verified external backup postdates this readiness diagnostic and the dry-run/protocol/source outputs. Once backed up, run exactly the frozen 27-episode smoke; do not train/refit from these sparse labels alone.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REQUEST, {
        "requested_utc": created.isoformat(),
        "reason": "backup v1b terminal/reward ablation dry-run plus pre-smoke readiness diagnostic before the 27-episode smoke rollout",
        "backup_required_before_more_simulations": True,
        "must_postdate_utc": created.isoformat(),
        "artifacts": [rel(Path(__file__).resolve()), rel(OUT), rel(STATE), rel(REQUEST), rel(PROTOCOL_JSON), rel(DRYRUN_DIR), rel(PREFIX_DONE)],
        "planned_next_smoke_episodes": plan_summary["smoke_plan_episode_count"],
        "planned_next_smoke_control_step_upper_bound": int(plan_summary["smoke_plan_episode_count"]) * 150,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
    })
    elapsed = created - FIRST_SUPERVISOR
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        "# Continue state after terminal/reward pre-smoke readiness diagnostic\n\n"
        f"UTC: {created.isoformat()} (~{h}h{m:02d}m{s:02d}s since first supervisor event).\n\n"
        "No simulation/training/validation64/sealed-test access. The frozen ablation smoke is scientifically ready but remains storage-gated unless a verified backup postdates this diagnostic and the dry-run/protocol/source artifacts.\n\n"
        f"Adequate local backup proofs after required time: {backups['adequate_local_proof_count']}. Next action: {next_action}. Backup request: `{rel(REQUEST)}`.\n\n"
        "Four-axis memory: SCENARIOS sparse corrected opportunity (4 positives across cases 4/5); REWARD terminal/source and synthetic-total artifacts remain the immediate discriminator; TRAINING/refit deferred because label gate remains false; COMPARISONS remain development-only with no validation/test claim.\n",
        encoding="utf-8",
    )
    completed_files = [OUT / "raw.json", OUT / "summary.md", STATE, REQUEST, Path(__file__).resolve(), DRYRUN_DONE, PREFIX_DONE, PROTOCOL_JSON]
    write_json(OUT / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "created_utc": created.isoformat(),
        "classification": raw["classification"],
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "smoke_inputs_suffice_now": smoke_inputs_suffice_now,
        "adequate_local_backup_proof_count": backups["adequate_local_proof_count"],
        "next_action": next_action,
        "backup_request": rel(REQUEST),
        "state": rel(STATE),
        "hashes": {rel(p): sha256(p) for p in completed_files if p.exists()},
    })
    block = f"""<!-- {MARKER} -->
## 2026-09-28 vehicle stress-v1b terminal/reward ablation pre-smoke readiness v0

UTC: {created.isoformat()}. No-simulation readiness diagnostic completed; no validation64/test access and no training/refit. It did not repeat the prefix audit. The frozen smoke covers 27 episodes: positive targets {plan_summary['positive_targets_in_smoke']} from cases {plan_summary['positive_cases_in_smoke']} plus controls {plan_summary['control_targets_in_smoke']}; full protocol remains 59 episodes. A-priori risks are {plan_summary['a_priori_risk_tags']}. Adequate local backup proofs after `{required_backup_after.isoformat()}`: {backups['adequate_local_proof_count']}; therefore the next smoke remains backup-gated unless a newer external proof is supplied. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`. Backup request: `{rel(REQUEST)}`.
"""
    for doc in DOCS:
        append_once(ROOT / doc, MARKER, block)
    print(json.dumps({
        "completed": rel(OUT / "completed.json"),
        "summary": rel(OUT / "summary.md"),
        "state": rel(STATE),
        "smoke_inputs_suffice_now": smoke_inputs_suffice_now,
        "adequate_local_backup_proof_count": backups["adequate_local_proof_count"],
        "smoke_episodes": plan_summary["smoke_plan_episode_count"],
        "positive_targets_in_smoke": plan_summary["positive_targets_in_smoke"],
        "a_priori_risk_tags": plan_summary["a_priori_risk_tags"],
        "backup_request": rel(REQUEST),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
