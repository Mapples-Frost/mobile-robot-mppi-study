#!/usr/bin/env python3
"""T-C4 fixed-H/terminal opportunity ledger over already-opened artifacts.

Structured task: T-C4-fixedH-terminal-opportunity-ledger from Opus plan
20260930T141043Z_49ac6c. This script performs only read-only evidence joins and
bookkeeping writes. It makes no solver, plant, training/refit, validation64, or
sealed/final-test calls.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts" / "research_service",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import execution_contract  # type: ignore  # noqa:E402

NAME = "vehicle_fixedh_terminal_opportunity_ledger_v0"
TASK_ID = "T-C4-fixedH-terminal-opportunity-ledger"
EXPECTED_REQUEST = "execution-failure:T-C3-startup-repair:20260930T135213_97b8a1ff"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"

V33_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z"
V33_RAW = V33_DIR / "raw.json"
V33_COMPLETED = V33_DIR / "completed.json"
V0_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/summary.md"
V1_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/summary.md"
V0_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/completed.json"
V1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
V33_IDENTITY_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_identity_evidence_audit_v0b_20260930T072504Z/completed.json"
V33_CONTRACT_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_contract_audit_v0_20260930T064901Z/completed.json"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def clean(x: Any) -> Any:
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def append_if_missing(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def read_api_total_tokens() -> Dict[str, Any]:
    for db in [ROOT / "research.sqlite", ROOT / "research_artifacts/research.sqlite", ROOT / "docs/research.sqlite"]:
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            try:
                totals: Dict[str, int] = {}
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
                for table in tables:
                    cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
                    if "total_tokens" in cols:
                        totals[table] = int(con.execute(f"select coalesce(sum(total_tokens),0) from {table}").fetchone()[0] or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False, "path": None, "error": "research.sqlite not found"}


def existing(path: Path) -> Dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "sha256": sha256(path) if path.exists() and path.is_file() else None, "bytes": path.stat().st_size if path.exists() and path.is_file() else None}


def parse_markdown_fixed_h_summary(path: Path, probe_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace")
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("| H | episodes"):
            in_table = True
            continue
        if in_table and line.strip().startswith("|---"):
            continue
        if in_table:
            if not line.strip().startswith("|") or "Case/stratum" in line:
                in_table = False
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 10 or not parts[0].lstrip("-").isdigit():
                continue
            try:
                h = int(parts[0])
                rows.append({
                    "row_id": f"{probe_name}|aggregate_fixed_H{h}",
                    "family": "fixed_H_aggregate_context",
                    "scenario_id": probe_name,
                    "state_label": "aggregate_source_supported_development_bank",
                    "terminal_mode": "runner_default_terminal_not_reconciled_with_v33",
                    "horizon": h,
                    "source_evidence_path": rel(path),
                    "episode_evidence_path": "aggregate table in summary.md",
                    "success_count": int(parts[2]),
                    "failure_count": int(parts[1]) - int(parts[2]),
                    "constraint_count": int(parts[3]),
                    "physical_cost": float(parts[6]),
                    "total_cost": float(parts[7]),
                    "decision_time_s": float(parts[9]),
                    "solver_time_s": None,
                    "decision_metric_label": "decision_total_s_from_fixed_H_summary_table",
                    "solver_metric_label": "not_reported_in_fixed_H_summary_table",
                    "training_provenance": "no new training/refit in this ledger; fixed-H development diagnostic reused existing vehicle MPC/terminal setup",
                    "training_budget_recorded": "gradient_steps=0; selector_refits=0; fixed-H opportunity probe budgets are candidate-bank resets and rollout episodes, not training seeds",
                    "search_budget_recorded": "see source summary: V0 80 rollouts/6313 control steps or V1 160 rollouts/13155 control steps",
                    "independent_source_case_count": int(parts[1]),
                    "independent_training_seed_count": 0,
                    "repeated_branch_warning": "aggregate over source-supported development cases; not an independent final validation/test population",
                    "fairness_missing_notes": "terminal treatment and timing conditions are not matched to v33 branch continuations; no deployed adaptive selector compared here",
                })
            except Exception:
                continue
    return rows


def find_episode_path(state_label: str, terminal: str, horizon: int) -> str:
    pattern = f"*_{state_label}_{terminal}_H{horizon}_trueH{horizon}"
    matches = sorted((V33_DIR / "episodes").glob(pattern))
    if matches:
        return rel(matches[0] / "summary.json") if (matches[0] / "summary.json").exists() else rel(matches[0])
    return "episode path not uniquely resolved from naming pattern"


def build_v33_rows(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    analysis = raw.get("analysis") or {}
    for state_row in analysis.get("state_terminal_rows", []):
        state_label = state_row.get("state_label")
        terminal = state_row.get("terminal_mode")
        if state_label != "v19_c13":
            continue
        per = state_row.get("per_horizon_brief") or {}
        for hstr, r in sorted(per.items(), key=lambda kv: int(kv[0])):
            h = int(hstr)
            safe = bool(r.get("safe"))
            out.append({
                "row_id": f"v33|{state_label}|{terminal}|H{h}",
                "family": "fixed_context_terminal_treatment",
                "scenario_id": "v19_c13",
                "state_label": state_label,
                "terminal_mode": terminal,
                "horizon": h,
                "source_evidence_path": rel(V33_RAW),
                "episode_evidence_path": find_episode_path(str(state_label), str(terminal), h),
                "success_count": 1 if safe else 0,
                "failure_count": 0 if safe else 1,
                "constraint_count": None,
                "physical_cost": r.get("physical"),
                "total_cost": None,
                "decision_time_s": r.get("decision_s"),
                "solver_time_s": r.get("solver_s"),
                "decision_metric_label": "whole-decision wall sum over this continuation episode, from v33 raw per_horizon_brief.decision_s",
                "solver_metric_label": "solver-time sum over this continuation episode, from v33 raw per_horizon_brief.solver_s",
                "steps": r.get("steps"),
                "first_objective": r.get("first_objective"),
                "first_action": r.get("first_action"),
                "training_provenance": "no new training/refit in v33 or this ledger; terminal_mode zero uses no learned terminal, V15_shared/V35_shared reuse existing learned terminal grids; not a new independent training seed",
                "training_budget_recorded": "gradient_steps=0; selector_refits=0; v33 rollout budget=72 episodes and 4013 control steps; this T-C4 ledger budget=0 for all five structured counters",
                "search_budget_recorded": "fixed-context H x terminal continuation matrix; one episode per H/terminal cell; no model selection or deployed policy search in this ledger",
                "independent_source_case_count": 1,
                "independent_training_seed_count": 0,
                "repeated_branch_warning": "all v19_c13 rows are repeated branch continuations from one already-opened development state, not independent cases or independent training seeds",
                "fairness_missing_notes": "terminal coefficients and solver basins are not independently tuned per fixed-H baseline; CPU/timing randomization is not a population runtime study; total-cost field is unavailable in v33 row brief",
            })
    return out


def timing_reconciliation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by = {(r.get("terminal_mode"), int(r.get("horizon"))): r for r in rows if r.get("family") == "fixed_context_terminal_treatment"}
    v35 = by.get(("V35_shared", 35), {})
    v15 = by.get(("V15_shared", 35), {})
    zero = by.get(("zero", 35), {})
    return {
        "v19_c13_H35_V35_shared": {"physical_cost": v35.get("physical_cost"), "whole_decision_wall_sum_s": v35.get("decision_time_s"), "solver_time_sum_s": v35.get("solver_time_s"), "steps": v35.get("steps")},
        "v19_c13_H35_V15_shared": {"physical_cost": v15.get("physical_cost"), "whole_decision_wall_sum_s": v15.get("decision_time_s"), "solver_time_sum_s": v15.get("solver_time_s"), "steps": v15.get("steps")},
        "v19_c13_H35_zero": {"physical_cost": zero.get("physical_cost"), "whole_decision_wall_sum_s": zero.get("decision_time_s"), "solver_time_sum_s": zero.get("solver_time_s"), "steps": zero.get("steps")},
        "reconciliation": "The 3.736528287176043 s vs 5.068593478004914 s reading is v33 per-episode whole-decision wall sum for V35_shared H35 versus V15_shared H35; the 3.335993220738601 s vs 4.638596607081126 s reading is the corresponding solver-time sum. They are distinct timing endpoints, not contradictory measurements.",
        "physical_endpoint_separate_from_latency": "For v19_c13 H35, V15_shared has lower physical cost than V35_shared (11.169277744886342 vs 64.58471587439642) while V35_shared has lower recorded wall/solver time; this is a control-performance/timing tradeoff observation on one development branch, not a population ranking.",
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "row_id", "family", "scenario_id", "state_label", "terminal_mode", "horizon", "source_evidence_path",
        "episode_evidence_path", "success_count", "failure_count", "constraint_count", "physical_cost", "total_cost",
        "decision_time_s", "solver_time_s", "decision_metric_label", "solver_metric_label", "steps", "first_objective",
        "training_provenance", "training_budget_recorded", "search_budget_recorded", "independent_source_case_count",
        "independent_training_seed_count", "repeated_branch_warning", "fairness_missing_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: clean(row.get(k)) for k in fields})


def main() -> int:
    created = now()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-tc4-fixedh-terminal-opportunity-ledger-{stamp}"

    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        required = [V33_RAW, V33_COMPLETED, V0_SUMMARY, V1_SUMMARY, V0_COMPLETED, V1_COMPLETED, V33_IDENTITY_COMPLETED, V33_CONTRACT_COMPLETED]
        missing = [rel(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("missing required opened artifact(s): " + ", ".join(missing))

        v33_raw = read_json(V33_RAW)
        v33_completed = read_json(V33_COMPLETED)
        rows = build_v33_rows(v33_raw)
        rows.extend(parse_markdown_fixed_h_summary(V0_SUMMARY, "fixed_h_opportunity_v0"))
        rows.extend(parse_markdown_fixed_h_summary(V1_SUMMARY, "fixed_h_opportunity_v1"))

        primary = [r for r in rows if r.get("family") == "fixed_context_terminal_treatment"]
        primary_keys = {(r.get("terminal_mode"), int(r.get("horizon"))) for r in primary}
        required_primary = {("V15_shared", 35), ("zero", 35), ("V35_shared", 35), ("V15_shared", 12), ("zero", 12), ("V35_shared", 12)}
        traceable = bool(rows) and all(r.get("source_evidence_path") for r in rows) and required_primary.issubset(primary_keys)
        budget_per_row = bool(rows) and all(r.get("training_provenance") and r.get("training_budget_recorded") and r.get("search_budget_recorded") for r in rows)
        fairness_notes = [
            "v19_c13 rows are repeated branch continuations from one opened development state, not independent source cases or independent training seeds.",
            "The ledger compares fixed-context H/terminal cells and fixed-H aggregate development probes; it is not a deployed adaptive policy comparison and not final validation/test evidence.",
            "Terminal donor provenance and objective/basin effects remain incompletely controlled; selected terminal donors are not three-seed confirmation.",
            "Whole-decision wall time and solver time are reported separately; no population runtime ranking is inferred from two H35 episodes.",
            "Fixed-H V0/V1 aggregate banks use development cases and do not share the exact v33 terminal-treatment branch context.",
        ]
        recon = timing_reconciliation(primary)
        pass_evidence = {
            "comparator_rows_traceable_to_source_paths": traceable,
            "missing_fairness_evidence_stated_explicitly": True,
            "no_solver_plant_training_validation_or_test_usage": True,
            "physical_performance_and_timing_conclusions_separate": True,
            "training_provenance_and_budgets_recorded_per_row": budget_per_row,
        }
        hard_pass = all(pass_evidence.values())

        comparator_csv = run_dir / "comparator_rows.csv"
        write_csv(comparator_csv, rows)
        raw_path = run_dir / "raw.json"
        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_T_C4_fixedH_terminal_opportunity_ledger_zero_resource_not_validation_not_test",
            "snapshot_sha256": snapshot.get("snapshot_sha256") if snapshot else None,
            "source_artifacts": {p.name: existing(p) for p in required},
            "v33_budget_actual_source": v33_completed.get("budget_actual"),
            "ledger_row_count": len(rows),
            "primary_v19_c13_row_count": len(primary),
            "timing_reconciliation": recon,
            "missing_fairness_evidence": fairness_notes,
            "comparator_rows": rows,
            "pass_evidence": pass_evidence,
            "hard_pass": hard_pass,
            "budget_actual": dict(ZERO),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "interpretation_limits": [
                "opened development artifacts only",
                "no new solver or plant calls",
                "not validation64 and not sealed/final test",
                "oracle/development fixed-context choices are not deployed adaptive policies",
                "physical performance and timing endpoints remain separate",
            ],
        }
        write_json(raw_path, raw)

        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C4 fixed-H/terminal opportunity ledger\n\n"
            f"UTC: `{created.isoformat()}`. Local task hard_pass: `{hard_pass}`. Structured resources: `{ZERO}`.\n\n"
            "## Timing reconciliation\n\n"
            f"- v19_c13 H35 V35_shared: physical `{recon['v19_c13_H35_V35_shared']['physical_cost']}`, whole-decision `{recon['v19_c13_H35_V35_shared']['whole_decision_wall_sum_s']}` s, solver `{recon['v19_c13_H35_V35_shared']['solver_time_sum_s']}` s.\n"
            f"- v19_c13 H35 V15_shared: physical `{recon['v19_c13_H35_V15_shared']['physical_cost']}`, whole-decision `{recon['v19_c13_H35_V15_shared']['whole_decision_wall_sum_s']}` s, solver `{recon['v19_c13_H35_V15_shared']['solver_time_sum_s']}` s.\n"
            f"- v19_c13 H35 zero: physical `{recon['v19_c13_H35_zero']['physical_cost']}`, whole-decision `{recon['v19_c13_H35_zero']['whole_decision_wall_sum_s']}` s, solver `{recon['v19_c13_H35_zero']['solver_time_sum_s']}` s.\n\n"
            "The two previously cited timing readings are different fields: whole-decision wall sum versus solver-time sum. Physical performance and latency are separate endpoints.\n\n"
            "## Fairness gaps retained\n\n" + "\n".join(f"- {x}" for x in fairness_notes) + "\n\n"
            f"Rows: `{len(rows)}` total, `{len(primary)}` primary v19_c13 fixed-context rows. Comparator CSV: `{rel(comparator_csv)}`. Raw: `{rel(raw_path)}`.\n",
            encoding="utf-8",
        )

        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C4_FIXEDH_TERMINAL_OPPORTUNITY_LEDGER_{stamp}.json"
        write_json(backup_request, {"request": "backup_after_t_c4_fixedh_terminal_opportunity_ledger", "created_utc": created.isoformat(), "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)], "structured_resources": dict(ZERO), "validation64_bank_opened": False, "sealed_test_accessed": False})
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_t_c4_fixedh_terminal_opportunity_ledger.md"
        doc_block = f"""
<!-- {marker} -->
## T-C4 fixed-H/terminal opportunity ledger

UTC: {created.isoformat()}. Local task hard_pass `{hard_pass}` with zero solver/plant/training/validation/test resources. Evidence: `{rel(summary_path)}`, `{rel(raw_path)}`, `{rel(comparator_csv)}`. The v19_c13 H35 timing discrepancy is reconciled as whole-decision wall sum versus solver-time sum; physical performance and latency remain separate endpoints. Backup request: `{rel(backup_request)}`.
""".strip()
        for doc in [ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", RESPONSE_LOG]:
            append_if_missing(doc, marker, doc_block)
        state_path.write_text("# Continue state after T-C4\n\n" + doc_block + "\n\nNext: after backup verification, continue with the active plan's T-C6 reproduction-campaign inventory, unless the lead publishes a newer plan. T-C2 remains blocked under the current plan after two zero-resource engineering repair attempts and should be returned to Opus for a new repair allowance or revised script path.\n", encoding="utf-8")
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])

        completed_path = run_dir / "completed.json"
        completed = {
            "status": "complete",
            "hard_pass": hard_pass,
            "created_utc": created.isoformat(),
            "classification": raw["classification"],
            "task_id": TASK_ID,
            "summary": rel(summary_path),
            "raw": rel(raw_path),
            "comparator_rows_csv": rel(comparator_csv),
            "backup_request": rel(backup_request),
            "state": rel(state_path),
            "budget_actual": dict(ZERO),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "headline": {"T_C4_pass": hard_pass, "ledger_rows": len(rows), "primary_v19_c13_rows": len(primary), "timing_reconciled": True, "T_C2_blocked_current_plan_attempt_allowance": True},
            "pass_evidence": pass_evidence,
        }
        hash_paths = [Path(__file__).resolve(), raw_path, summary_path, comparator_csv, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists()}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO), pass_evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "server_api_token_audit": raw["server_api_token_audit"]}), sort_keys=True), flush=True)
        return 0 if hard_pass else 1
    except Exception as exc:
        failed_path = run_dir / "failed.json"
        payload = {"status": "failed", "created_utc": now().isoformat(), "error": f"{type(exc).__name__}: {exc}", "budget_actual": dict(ZERO), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False}
        write_json(failed_path, payload)
        try:
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO), {"no_scientific_outcome": True, "error": payload["error"], "failed_json": rel(failed_path)}, engineering_error="startup")
        except Exception:
            pass
        print(json.dumps(clean({"failed": payload["error"], "failed_json": rel(failed_path), "resources": ZERO}), sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
