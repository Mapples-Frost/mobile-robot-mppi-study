#!/usr/bin/env python3
"""v18d execution wrapper for repaired H10 probe-telemetry diagnostic.

The v18 script was backed but not executed; v18b/v18c are archived pre-execution
repairs. v18d executes the repaired diagnostic while correcting the offline model
budget cap in the frozen protocol. It still uses only existing v15 development
branch traces: no new MPC simulation, no validation64 access, no sealed test, no
RL/gradient training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
V18C_PATH = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_probe_telemetry_v18c.py"
spec = importlib.util.spec_from_file_location("vehicle_true_variable_horizon_probe_telemetry_v18c_mod", str(V18C_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import v18c")
v18c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v18c)  # type: ignore[union-attr]

NAME = "vehicle_true_variable_horizon_probe_telemetry_v18d"
STAMP = "20260930T0035Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260930T0035_after_probe_telemetry_v18d.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_PROBE_TELEMETRY_V18D_{STAMP}.json"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-probe-telemetry-v18d-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
OFFLINE_EVAL_CAP = 500000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return v18c.rel(p)


def repoint() -> None:
    for mod in (v18c, v18c.base):
        mod.NAME = NAME
        mod.STAMP = STAMP
        mod.OUT = OUT
        mod.PROTOCOL = PROTOCOL
        mod.STATE = STATE
        mod.BACKUP_REQUEST = BACKUP_REQUEST
        mod.SOURCE = SOURCE
        mod.MARKER = MARKER
        mod.FIRST_EVENT = FIRST_EVENT


def freeze_protocol_v18d(created: dt.datetime, diag: Mapping[str, Any], cfgs: Sequence[Mapping[str, Any]], hashes: Mapping[str, str], backup_commit: str) -> None:
    # Call v18c's repaired freezer, then amend only the pre-outcome metadata fields.
    v18c.freeze_protocol_v18c(created, diag, cfgs, hashes, backup_commit)
    with PROTOCOL.open("r", encoding="utf-8") as f:
        proto = json.load(f)
    proto["protocol_id"] = f"{NAME}_preoutcome_frozen_{STAMP}"
    proto["source_repair_from_v18_v18b"]["v18d_wrapper"] = rel(SOURCE)
    proto["source_repair_from_v18_v18b"]["v18d_wrapper_sha256"] = sha256(SOURCE)
    proto["source_repair_from_v18_v18b"]["v18c_source"] = rel(V18C_PATH)
    proto["source_repair_from_v18_v18b"]["v18c_source_sha256"] = sha256(V18C_PATH)
    proto["source_repair_from_v18_v18b"]["reason"] += "; v18d additionally corrects the declared offline-evaluation cap before first execution"
    proto["budget_declared"]["offline_model_evaluations_cap"] = OFFLINE_EVAL_CAP
    v18c.base.write_json(PROTOCOL, proto)


def append_docs_v18d(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (v18c.base.now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-30 vehicle true-variable-H v18d H10 probe-telemetry diagnostic

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline diagnostic over existing v15 true-H branch traces; no MPC simulation, no validation64/sealed-test access, no gradient training. v18d executes the v18c repaired telemetry diagnostic and fixes the predeclared offline-evaluation cap. rows={h['rows']}; configs={h['config_count']}; offline_model_evaluations={h['offline_model_evaluations']}; leave_bank_save={h['leave_bank_save']:.6f}; leave_bank_bad={h['leave_bank_bad']}; leave_bank_h10={h['leave_bank_h10']}; leave_source_save={h['leave_source_save']:.6f}; leave_source_bad={h['leave_source_bad']}; leave_source_h10={h['leave_source_h10']}; best_global={h['best_global_config']} save={h['best_global_save']:.6f} bad={h['best_global_bad']} H10={h['best_global_h10']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT/'summary.md')}`, `{rel(OUT/'raw.json')}`, `{rel(OUT/'completed.json')}`. Canonical experiment registry row is written by run_experiment, not by this script.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    repoint()
    v18c.base.load_rows = v18c.load_rows_v18c
    v18c.base.evaluate = v18c.evaluate_v18c
    v18c.base.aggregate = v18c.aggregate_v18c
    v18c.base.nested = v18c.nested_v18c
    v18c.base.freeze_protocol = freeze_protocol_v18d
    v18c.base.append_docs = append_docs_v18d
    # Ensure helper globals used by the v18c patched functions also point here.
    v18c.NAME = NAME
    v18c.STAMP = STAMP
    v18c.OUT = OUT
    v18c.PROTOCOL = PROTOCOL
    v18c.STATE = STATE
    v18c.BACKUP_REQUEST = BACKUP_REQUEST
    v18c.SOURCE = SOURCE
    v18c.MARKER = MARKER
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    args = ap.parse_args(argv)
    if not args.run:
        print("pass --run", file=sys.stderr)
        return 2
    code = int(v18c.base.main(["--run", "--backup-verified-commit", args.backup_verified_commit]))
    if code == 0:
        # Add wrapper/dependency source hashes to the completed record and backup request.
        comp_p = OUT / "completed.json"
        req_p = BACKUP_REQUEST
        if comp_p.exists():
            comp = json.loads(comp_p.read_text(encoding="utf-8"))
            comp.setdefault("hashes", {})[rel(SOURCE)] = sha256(SOURCE)
            comp.setdefault("hashes", {})[rel(V18C_PATH)] = sha256(V18C_PATH)
            v18c.base.write_json(comp_p, comp)
        if req_p.exists():
            req = json.loads(req_p.read_text(encoding="utf-8"))
            arts = list(req.get("artifacts") or [])
            for p in (rel(SOURCE), rel(V18C_PATH), rel(v18c.BASE_PATH)):
                if p not in arts:
                    arts.insert(0, p)
            req["artifacts"] = arts
            req["required_before_more_unique_science"] = True
            v18c.base.write_json(req_p, req)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
