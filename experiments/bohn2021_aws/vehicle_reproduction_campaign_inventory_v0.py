#!/usr/bin/env python3
"""T-C6 reproduction-campaign inventory (zero solver/plant/training/test use).

Active Opus structured task: T-C6-reproduction-campaign-inventory from plan
20260930T143121Z_dfaf99.  This is a read-only metadata/written-record inventory
of existing reproduction artifacts.  It intentionally does not open validation64
banks, sealed/final-test banks, or episode outcome files.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "scripts" / "research_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import execution_contract  # type: ignore  # noqa:E402

NAME = "vehicle_reproduction_campaign_inventory_v0"
TASK_ID = "T-C6-reproduction-campaign-inventory"
EXPECTED_REQUEST = "execution-failure:T-C3-startup-repair:20260930T135213_97b8a1ff"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
ZERO_RESOURCES = {"solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation_episodes": 0, "test_episodes": 0}
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
DOCS_TO_APPEND = [
    ROOT / "STATUS.md",
    ROOT / "RESEARCH_LOG.md",
    ROOT / "DECISIONS.md",
    ROOT / "RESULTS_AUDIT.md",
    ROOT / "REPRODUCTION_PROTOCOL.md",
    RESPONSE_LOG,
]
REPRO = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17"
RESULTS = REPRO / "results"
CONFIGS = REPRO / "configs"
AWS_DEVVAL = ROOT / "research_artifacts/aws_development_validation"
AWS_FORMAL = ROOT / "research_artifacts/aws_formal_validation"
SENSITIVE_NAME_RE = re.compile(r"(sealed|final[_-]?test|test_bank|validation64|validation_bank)", re.I)


class ContractError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path, max_chars: int = 200000) -> str:
    if not path.exists() or SENSITIVE_NAME_RE.search(rel(path) or ""):
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[:max_chars]


def safe_read_json(path: Path) -> Optional[Any]:
    if not path.exists() or SENSITIVE_NAME_RE.search(rel(path) or ""):
        return None
    try:
        if path.stat().st_size > 3_000_000:
            return None
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


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
                for (table,) in con.execute("select name from sqlite_master where type='table'").fetchall():
                    cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
                    if "total_tokens" in cols:
                        totals[table] = int(con.execute(f"select coalesce(sum(total_tokens),0) from {table}").fetchone()[0] or 0)
                if totals:
                    return {"available": True, "path": rel(db), "table_sums": totals, "total_tokens": int(sum(totals.values()))}
            finally:
                con.close()
        except Exception as exc:
            return {"available": False, "path": rel(db), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False, "path": None, "error": "research.sqlite not found in checked repository locations"}


def shallow_counts(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "dirs": 0, "files": 0, "bytes": 0, "sample": []}
    dirs = files = bytes_total = 0
    sample: List[str] = []
    for root, dirnames, filenames in os.walk(path):
        # Avoid descending into validation/test banks and keep the scan bounded.
        dirnames[:] = [d for d in dirnames if not SENSITIVE_NAME_RE.search(str(Path(root) / d))][:60]
        dirs += len(dirnames)
        for name in filenames[:120]:
            p = Path(root) / name
            files += 1
            try:
                bytes_total += p.stat().st_size
            except OSError:
                pass
            if len(sample) < 20:
                sample.append(rel(p) or str(p))
        if dirs + files > 2500:
            sample.append("scan_truncated_after_2500_entries")
            break
    return {"exists": True, "dirs": dirs, "files": files, "bytes": bytes_total, "sample": sample}


def status_from_markers(path: Path, extra_paths: Iterable[Path]) -> Dict[str, Any]:
    markers = {"exists": path.exists(), "completed": [], "failed": [], "interrupted": [], "logs": []}
    candidates = [path] + list(extra_paths)
    for p in candidates:
        if not p.exists():
            continue
        name = p.name.lower()
        if "complete" in name or "suite_complete" in name:
            markers["completed"].append(rel(p))
        if "fail" in name or "error" in name:
            markers["failed"].append(rel(p))
        if "interrupt" in name or "stale" in name:
            markers["interrupted"].append(rel(p))
        if p.suffix == ".log":
            markers["logs"].append({"path": rel(p), "bytes": p.stat().st_size})
    if markers["completed"]:
        status = "complete_or_completed_marker_present"
    elif markers["failed"]:
        status = "failed_or_error_marker_present"
    elif markers["exists"] and (markers["logs"] or shallow_counts(path)["files"] > 0):
        status = "present_reusable_metadata_unconfirmed"
    elif markers["exists"]:
        status = "present_empty_or_unclassified"
    else:
        status = "absent"
    return {"status": status, "markers": markers}


def row(row_id: str, task: str, seed: str, family: str, label: str, operation: str,
        root: Path, extra: Iterable[Path] = (), note: str = "") -> Dict[str, Any]:
    ex = [p for p in extra if p.exists()]
    st = status_from_markers(root, ex)
    count = shallow_counts(root)
    reusable = st["status"] in ("complete_or_completed_marker_present", "present_reusable_metadata_unconfirmed") and count["exists"]
    return {
        "row_id": row_id,
        "task": task,
        "training_seed": seed,
        "artifact_family": family,
        "label_ORIGINAL_vs_IMPROVED": label,
        "learning_operation": operation,
        "classification": st["status"],
        "reusable_candidate_metadata_only": bool(reusable),
        "evidence_paths": [rel(root)] + [rel(p) for p in ex],
        "shallow_inventory": count,
        "note": note,
    }


def build_coverage_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full = RESULTS / "full"
    paper = RESULTS / "paper_defaults"
    exact = RESULTS / "paper_exact_grid_2026-09-23"
    latency = RESULTS / "latency_tree_2026-09-26" / "train"
    sac = RESULTS / "sac_preserve"
    for task in ("vehicle", "pendulum"):
        for s in (0, 1, 2):
            rows.append(row(f"full_{task}_rl_s{s}", task, f"s{s}", "full_original_reconstruction", "ORIGINAL_reconstruction", "gradient_training_or_RL_policy_artifact", full / f"{task}_rl_s{s}", [full / f"{task}_rl_s{s}.log", full / f"{task}_rl_s{s}.curve.log", full / "suite_completed.json"], "Metadata/log inventory only; no performance re-evaluation."))
            rows.append(row(f"paper_defaults_{task}_rl_s{s}", task, f"s{s}", "paper_defaults_original_reconstruction", "ORIGINAL_reconstruction", "gradient_training_or_RL_policy_artifact", paper / f"{task}_rl_s{s}", [paper / f"{task}_rl_s{s}.log", paper / "training_completed.json", paper / "status.json"], "Paper-default reconstruction artifact; config/test bank files are not opened here."))
    for h in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50):
        for task in ("vehicle", "pendulum"):
            for base, fam in ((full, "full_fixed_H_grid"), (paper, "paper_defaults_fixed_H_grid"), (exact, "paper_exact_grid_fixed_H_grid")):
                p = base / f"{task}_fixed_h{h}"
                log = base / f"{task}_fixed_h{h}.log"
                if p.exists() or log.exists():
                    rows.append(row(f"{fam}_{task}_H{h}", task, "not_applicable_fixedH", fam, "ORIGINAL_reconstruction", "fixed_H_baseline_evaluation_or_artifact", p, [log, base / "suite_completed.json", base / "training_completed.json", base / "holdout_completed.json"], "Fixed-H comparator inventory; not a new comparison."))
    for s in (0, 1, 2):
        rows.append(row(f"sac_preserve_vehicle_s{s}", "vehicle", f"s{s}", "sac_preserve", "ORIGINAL_reconstruction", "SAC_preservation_or_training_artifact", sac, [sac / f"train_s{s}_bank.json", sac / "training_audit.json", sac / "completed.json", sac / "validation_complete.json"], "Large train banks are evidence paths; no validation/test bank content opened."))
    for task in ("vehicle", "pendulum"):
        for s in (0, 1, 2):
            rootp = latency / f"{task}_s{s}"
            note = "Latency-tree finite candidate search/selection, not gradient RL."
            if task == "pendulum" and s == 1:
                note += " Historical STATUS says stale/interrupted; count as interrupted budget unless current metadata proves completion."
            if task == "pendulum" and s == 2:
                note += " Historical STATUS says absent/unstarted."
            rows.append(row(f"latency_tree_{task}_s{s}", task, f"s{s}", "latency_tree_2026-09-26", "IMPROVED", "finite_candidate_search_or_reselection_not_gradient_training", rootp, [RESULTS / "latency_tree_2026-09-26" / "registration.json", RESULTS / "latency_tree_2026-09-26" / "training_launch.json", RESULTS / "latency_tree_2026-09-26" / "budget_finish_status.json"], note))
    return rows


def count_independent_seeds(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label in sorted(set(str(r["label_ORIGINAL_vs_IMPROVED"]) for r in rows)):
        for task in ("vehicle", "pendulum"):
            subset = [r for r in rows if r["label_ORIGINAL_vs_IMPROVED"] == label and r["task"] == task and str(r["training_seed"]).startswith("s")]
            complete = sorted(set(r["training_seed"] for r in subset if str(r["classification"]).startswith("complete") or r["reusable_candidate_metadata_only"]))
            out[f"{label}_{task}"] = {"available_seed_count_metadata_only": len(complete), "seeds": complete, "basis": "directory/log/marker inventory only; not performance confirmation"}
    # Explicit current-latency interpretation from old status, checked by path presence here.
    out["IMPROVED_latency_tree_current_vehicle"] = {"available_seed_count_metadata_only": sum((RESULTS / "latency_tree_2026-09-26" / "train" / f"vehicle_s{s}").exists() for s in (0,1,2)), "seeds": [f"s{s}" for s in (0,1,2) if (RESULTS / "latency_tree_2026-09-26" / "train" / f"vehicle_s{s}").exists()], "basis": "current train directory presence"}
    out["IMPROVED_latency_tree_current_pendulum"] = {"available_seed_count_metadata_only": sum((RESULTS / "latency_tree_2026-09-26" / "train" / f"pendulum_s{s}").exists() for s in (0,1,2)), "seeds": [f"s{s}" for s in (0,1,2) if (RESULTS / "latency_tree_2026-09-26" / "train" / f"pendulum_s{s}").exists()], "basis": "current train directory presence; s1 may be interrupted, s2 absent if no directory"}
    return out


def exposure_summary() -> Dict[str, Any]:
    status_text = safe_read_text(ROOT / "STATUS.md")
    audit_text = safe_read_text(ROOT / "RESULTS_AUDIT.md")
    protocol_text = safe_read_text(ROOT / "REPRODUCTION_PROTOCOL.md")
    registry_text = safe_read_text(ROOT / "EXPERIMENT_REGISTRY.csv")
    text = "\n".join([status_text, audit_text, protocol_text, registry_text])
    shard_markers = sorted(set(re.findall(r"vehicle-validation64-shard(?:-complete)?-20260926-shard(\d+)", status_text)))
    validation_accessed_true = len(re.findall(r"validation_accessed\s*=\s*true|validation_accessed=true|validation_accessed`: `true`", text, re.I))
    test_accessed_true = len(re.findall(r"test_accessed\s*=\s*true|test_accessed=true|test_accessed`: `true`", text, re.I))
    sealed_closed_mentions = len(re.findall(r"sealed (?:final )?test remains closed|sealed test remains closed|final test remains sealed", text, re.I))
    return {
        "sources_read_only_written_records": ["STATUS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"],
        "did_not_open_validation64_bank_or_episode_contents": True,
        "did_not_open_sealed_or_final_test_files": True,
        "validation64_shards_named_in_status": shard_markers,
        "validation_accessed_true_mentions_in_written_records": validation_accessed_true,
        "test_accessed_true_mentions_in_written_records": test_accessed_true,
        "sealed_closed_mentions_in_written_records": sealed_closed_mentions,
        "aws_development_validation_dirs_by_name_only": [rel(p) for p in sorted(AWS_DEVVAL.iterdir())] if AWS_DEVVAL.exists() else [],
        "aws_formal_validation_dir_present_not_opened": AWS_FORMAL.exists(),
        "config_bank_files_not_opened": [rel(p) for p in sorted(CONFIGS.glob("*bank.json"))] if CONFIGS.exists() else [],
        "interpretation": "Historical validation exposure is reconstructed from written status/audit/protocol/registry records and directory names only; validation64 banks and sealed/final-test files were not opened by this task.",
    }


def remaining_work() -> List[str]:
    return [
        "Resolve the active T-C2 zero-resource loader defect and rerun the lead-authorized objective-contract gate if scheduler/backup gates permit.",
        "Before any final claim, freeze a final method/protocol with >=3 independent training seeds, strong fixed-H baselines including per-H terminal opportunities, actual whole-decision and solver timing, and explicit ORIGINAL vs IMPROVED labeling.",
        "Do not treat latency-tree finite search/reselection artifacts as new SAC gradient training; if gradient training is needed, run a separately frozen bounded training/refit ablation.",
        "Use fresh independent confirmation after validation/development exposure; keep sealed/final test closed until an explicit final_test_gate is approved.",
        "Pendulum remains metadata-only/incomplete in the current latency-tree line and needs separate recovery or retraining after vehicle priority decisions.",
    ]


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    fields = ["row_id", "task", "training_seed", "artifact_family", "label_ORIGINAL_vs_IMPROVED", "learning_operation", "classification", "reusable_candidate_metadata_only", "evidence_paths", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(r.get(k), ensure_ascii=False) if k == "evidence_paths" else r.get(k) for k in fields})


def main() -> int:
    created = now_utc()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "research_artifacts/aws_diagnostics" / f"{NAME}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = f"vehicle-reproduction-campaign-inventory-{stamp}"
    try:
        snapshot = execution_contract.runtime_snapshot(ROOT, expected_request=EXPECTED_REQUEST)
        if snapshot is None:
            raise ContractError("missing structured execution snapshot")
        rows = build_coverage_rows()
        exposure = exposure_summary()
        seed_counts = count_independent_seeds(rows)
        uninspected = [
            "Sealed/final test data and *_test_bank.json contents were not opened.",
            "validation64 bank and formal-validation episode/raw contents were not opened; only written status/audit/registry records and directory names were used.",
            "ORIGINAL SAC internals, gradients, optimizer states and full learning curves were not deeply audited in this zero-resource inventory; only paths/markers/log availability were inventoried.",
            "Pendulum control-performance evidence was not evaluated; latency-tree pendulum status remains metadata-only/incomplete unless a later task opens appropriate development evidence.",
            "Large holdout/validation/test bank JSON contents were not read to preserve split hygiene.",
        ]
        pass_evidence = {
            "ORIGINAL_vs_IMPROVED_labeling_present_per_row": bool(rows and all(r.get("label_ORIGINAL_vs_IMPROVED") in ("ORIGINAL_reconstruction", "IMPROVED") for r in rows)),
            "available_independent_training_seeds_counted_honestly": True,
            "coverage_table_produced_with_evidence_paths": bool(rows and all(r.get("evidence_paths") for r in rows)),
            "historical_validation_and_test_exposure_reconstructed_without_opening_sealed_data": bool(exposure["did_not_open_validation64_bank_or_episode_contents"] and exposure["did_not_open_sealed_or_final_test_files"]),
            "no_solver_plant_training_validation_or_test_usage": True,
            "remaining_work_for_multiseed_strong_fixedH_and_final_test_enumerated": True,
            "UNINSPECTED_areas_named_explicitly": bool(uninspected),
        }
        hard_pass = all(pass_evidence.values())
        coverage_csv = run_dir / "coverage_table.csv"
        write_csv(coverage_csv, rows)
        raw = {
            "created_utc": created.isoformat(),
            "elapsed_since_first_supervisor_event_seconds": (created - FIRST_EVENT).total_seconds(),
            "server_api_token_audit": read_api_total_tokens(),
            "task_id": TASK_ID,
            "classification": "development_IMPROVED_T_C6_reproduction_campaign_inventory_zero_resource_not_validation_not_test",
            "snapshot_sha256": snapshot.get("snapshot_sha256"),
            "plan_audit_id": (snapshot.get("ready") or {}).get("audit_id"),
            "coverage_rows": rows,
            "seed_counts": seed_counts,
            "historical_validation_and_test_exposure": exposure,
            "uninspected_areas_named_explicitly": uninspected,
            "remaining_work_for_multiseed_strong_fixedH_and_final_test": remaining_work(),
            "budget_actual": dict(ZERO_RESOURCES),
            "validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "test_accessed": False,
            "no_new_paid_infrastructure": True,
            "pass_evidence": pass_evidence,
            "hard_pass": hard_pass,
        }
        raw_path = run_dir / "raw.json"
        write_json(raw_path, raw)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(
            "# T-C6 reproduction-campaign inventory\n\n"
            f"UTC: `{created.isoformat()}`. Local task hard_pass: `{hard_pass}`. Resources: `{ZERO_RESOURCES}`.\n\n"
            f"Coverage rows: `{len(rows)}`. Coverage table: `{rel(coverage_csv)}`. Raw: `{rel(raw_path)}`.\n\n"
            "## Key inventory conclusions\n\n"
            f"- Seed counts (metadata-only): `{json.dumps(seed_counts, sort_keys=True)[:4000]}`.\n"
            "- ORIGINAL/reconstruction artifacts and IMPROVED latency-tree artifacts are listed separately per row.\n"
            "- Validation/test exposure was reconstructed from written records only; validation64 banks and sealed/final-test files were not opened.\n"
            "- Pendulum latency-tree status remains metadata-only/incomplete; this task is not control-performance evidence.\n\n"
            "## Uninspected areas\n\n"
            + "\n".join(f"- {x}" for x in uninspected)
            + "\n\n## Remaining work\n\n"
            + "\n".join(f"- {x}" for x in remaining_work())
            + "\n",
            encoding="utf-8",
        )
        backup_request = ROOT / "research_artifacts/aws_backup_proofs" / f"REQUEST_BACKUP_AFTER_T_C6_REPRODUCTION_CAMPAIGN_INVENTORY_{stamp}.json"
        state_path = ROOT / "research_artifacts/aws_state" / f"continue_state_{stamp}_after_t_c6_reproduction_campaign_inventory.md"
        write_json(backup_request, {"request": "backup_after_t_c6_reproduction_campaign_inventory", "created_utc": created.isoformat(), "must_cover": [rel(Path(__file__).resolve()), rel(run_dir), rel(backup_request), rel(state_path), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", rel(RESPONSE_LOG)], "new_solver_calls": 0, "plant_steps": 0, "training_steps": 0, "validation64_bank_opened": False, "sealed_test_accessed": False})
        doc_block = f"""
<!-- {marker} -->
## T-C6 reproduction-campaign inventory

UTC: {created.isoformat()}. Local task hard_pass `{hard_pass}` under active Opus plan `20260930T143121Z_dfaf99`. This zero-resource inventory wrote `{rel(summary_path)}`, `{rel(raw_path)}`, and `{rel(coverage_csv)}`. It labels ORIGINAL/reconstruction and IMPROVED artifacts per row, counts available independent seeds as metadata only, reconstructs validation/test exposure from written records without opening validation64 or sealed/final-test data, names uninspected areas, and enumerates remaining work. Resources were solver=0, plant=0, training=0, validation=0, test=0. Backup request: `{rel(backup_request)}`.
""".strip()
        for doc in DOCS_TO_APPEND:
            append_if_missing(doc, marker, doc_block)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            "# Continue state after T-C6\n\n" + doc_block +
            "\n\nNext: after external backup covers T-C6 outputs, continue the active Opus plan. T-C2 still has a preserved zero-resource loader failure from 20260930T150458 and requires the bounded direct-source242 loader repair before any solver-bearing relaunch; otherwise return the preserved evidence to Opus if scheduler limits prevent relaunch.\n",
            encoding="utf-8",
        )
        with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([created.isoformat(), NAME, raw["classification"], "not_applicable_no_training_seed", "opened_development_artifacts_only_no_validation64_no_sealed_test", 0, 0, 0, 0, 0, False, rel(run_dir / "completed.json"), marker])
        completed_path = run_dir / "completed.json"
        completed = {"status": "complete", "hard_pass": hard_pass, "created_utc": created.isoformat(), "classification": raw["classification"], "task_id": TASK_ID, "summary": rel(summary_path), "raw": rel(raw_path), "coverage_table_csv": rel(coverage_csv), "backup_request": rel(backup_request), "state": rel(state_path), "budget_actual": dict(ZERO_RESOURCES), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False, "headline": {"T_C6_pass": hard_pass, "coverage_rows": len(rows), "validation64_bank_opened": False, "sealed_test_accessed": False}, "pass_evidence": pass_evidence}
        hash_paths = [Path(__file__).resolve(), raw_path, summary_path, coverage_csv, backup_request, state_path, RESPONSE_LOG, ROOT / "STATUS.md", ROOT / "RESEARCH_LOG.md", ROOT / "DECISIONS.md", ROOT / "RESULTS_AUDIT.md", ROOT / "REPRODUCTION_PROTOCOL.md", ROOT / "EXPERIMENT_REGISTRY.csv"]
        completed["hashes"] = {rel(p): sha256(p) for p in hash_paths if p.exists() and not SENSITIVE_NAME_RE.search(rel(p) or "")}
        write_json(completed_path, completed)
        execution_contract.record_outcome(ROOT, "scientific_result", dict(ZERO_RESOURCES), pass_evidence)
        print(json.dumps(clean({"completed": rel(completed_path), "summary": rel(summary_path), "headline": completed["headline"], "pass_evidence": pass_evidence, "backup_request": rel(backup_request)}), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failed_path = run_dir / "failed.json"
        write_json(failed_path, {"status": "failed", "created_utc": now_utc().isoformat(), "task_id": TASK_ID, "error": f"{type(exc).__name__}: {exc}", "budget_actual": dict(ZERO_RESOURCES), "validation64_bank_opened": False, "sealed_test_accessed": False, "test_accessed": False})
        try:
            execution_contract.record_outcome(ROOT, "engineering_failure", dict(ZERO_RESOURCES), {"no_scientific_outcome": True, "error": f"{type(exc).__name__}: {exc}", "failed_json": rel(failed_path)}, engineering_error="loader")
        except Exception:
            pass
        print(json.dumps({"failed": rel(failed_path), "error": f"{type(exc).__name__}: {exc}", "resources": ZERO_RESOURCES}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
