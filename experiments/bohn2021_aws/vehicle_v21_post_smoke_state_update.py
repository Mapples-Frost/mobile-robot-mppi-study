#!/usr/bin/env python3
"""Persist v21 source/CLI-smoke state and backup request.

Metadata/documentation only: no simulation, no training/refit, no validation64,
no sealed test.  This records that the v21 H12/H15 source-independent acquisition
runner has been written and CLI-smoked, and that further unique science is gated
on an external backup covering the new source, smoke registry, docs and request.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260930T0105Z"
MARKER = f"vehicle-v21-post-smoke-state-update-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21.py"
SMOKE_REGISTRY = ROOT / "research_artifacts/aws_runs/20260930T005800_e89debda/registry.json"
SMOKE_DIR = ROOT / "research_artifacts/aws_runs/20260930T005800_e89debda"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_v21_source_smoke.md"
BACKUP_REQUEST = ROOT / f"research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V21_SOURCE_AND_SMOKE_{STAMP}.json"
RESPONSE_LOG = ROOT / "docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md"
LATEST_BACKUP = {
    "time_utc": "2026-09-30T00:54:03.400169+00:00",
    "status": "verified",
    "remaining_changed_files": 0,
    "commit": "e56a8e58074f10630db2ba69b4ee6e2079900986",
    "package": "20260930T005400_3e34c015.tar.gz",
    "sha256": "c0908a8f5ee526eb72f25ad7801af933b338ea2d9636022d64dbe2514c8c01d1",
    "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
}


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old:
        path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> int:
    if not SCRIPT.exists():
        raise SystemExit(f"missing v21 script {rel(SCRIPT)}")
    if not SMOKE_REGISTRY.exists():
        raise SystemExit(f"missing smoke registry {rel(SMOKE_REGISTRY)}")
    reg = read_json(SMOKE_REGISTRY)
    if int(reg.get("exit_status", -1)) != 0:
        raise SystemExit("v21 CLI smoke did not exit 0")
    created = now()
    hashes = {
        rel(SCRIPT): sha256(SCRIPT),
        rel(SMOKE_REGISTRY): sha256(SMOKE_REGISTRY),
    }
    for name in ("stdout.log", "stderr.log", "cpu_samples.jsonl", "cloudwatch_snapshot.json"):
        p = SMOKE_DIR / name
        if p.exists():
            hashes[rel(p)] = sha256(p)
    request = {
        "requested_utc": created.isoformat(),
        "reason": "Backup v21 H12/H15 source-independent acquisition runner, CLI smoke registry, docs/state updates and this request before running the unique v21 simulation acquisition.",
        "backup_required_before_more_unique_science": True,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "simulations_in_this_update": 0,
        "training_episodes_in_this_update": 0,
        "gradient_steps_in_this_update": 0,
        "latest_verified_backup_before_new_v21_source_from_supervisor_context": LATEST_BACKUP,
        "new_artifacts_requiring_backup": [
            rel(SCRIPT),
            rel(SMOKE_DIR),
            rel(STATE),
            rel(BACKUP_REQUEST),
            "STATUS.md",
            "RESEARCH_LOG.md",
            "DECISIONS.md",
            "RESULTS_AUDIT.md",
            "REPRODUCTION_PROTOCOL.md",
            "EXPERIMENT_REGISTRY.csv",
            rel(RESPONSE_LOG),
        ],
        "hashes_observed_before_request": hashes,
    }
    write_json(BACKUP_REQUEST, request)

    block = f"""<!-- {MARKER} -->
## 2026-09-30 v21 source-independent H12/H15 acquisition runner frozen and CLI-smoked

UTC: {created.isoformat()}. Wrote `experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21.py` (sha256 `{hashes[rel(SCRIPT)]}`) to acquire development-only source-independent H12/H15 labels after the v20b H12/H15 selector passed opened strict splits but had only one H12-negative source. CLI smoke `research_artifacts/aws_runs/20260930T005800_e89debda/registry.json` exited 0 with no simulations, no validation64, no sealed test, no training/refit. The v21 design uses metadata-only fresh source selection from the stress-v1 pool, excludes prior source_candidate_index values, runs H15 Stage-A traces, freezes a selected-state manifest before any H12 branch outcome, then runs blocked true-H12/H15 shared-H15-terminal branches.

Operational gate: no further unique science should run until a verified external backup covers the v21 source, smoke run, this docs/state update and backup request `{rel(BACKUP_REQUEST)}`. After backup, run v21 with the legacy interpreter and `--run --backup-verified-commit <verified_commit> --i-accept-development-v21`.
"""
    for doc in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        append_once(ROOT / doc, MARKER, block)
    append_once(RESPONSE_LOG, MARKER, f"""<!-- {MARKER} -->
### v21 follow-up to Astra A4/A7/A8/A11

UTC: {created.isoformat()}. Disposition remains accepted/open. Concrete action prepared after v20b: v21 source-independent H12/H15 acquisition runner frozen and CLI-smoked (source sha256 `{hashes[rel(SCRIPT)]}`, smoke registry `{rel(SMOKE_REGISTRY)}`). No validation64 or sealed test accessed, no simulation/training/refit in the smoke. This directly follows A7/A8 by seeking source-independent H12 negative/support labels before validation, and A4 remains open because online selector overhead still requires a separate smoke after/alongside v21. Backup is required before executing v21 unique simulation.
""")
    reg_path = ROOT / "EXPERIMENT_REGISTRY.csv"
    if reg_path.exists():
        old = reg_path.read_text(encoding="utf-8", errors="replace")
        row = f"{created.isoformat()},vehicle_v21_source_and_cli_smoke_state_update,metadata_docs_backup_request_no_simulation,development_no_validation_no_test,0,0,0,0,0,False,{rel(STATE)}\n"
        if MARKER not in old[-200000:]:
            reg_path.write_text(old.rstrip() + "\n" + row, encoding="utf-8")
    state_text = f"""# Continue state after v21 source/CLI-smoke freeze

UTC: {created.isoformat()}

- v21 runner: `{rel(SCRIPT)}` sha256 `{hashes[rel(SCRIPT)]}`.
- CLI smoke registry: `{rel(SMOKE_REGISTRY)}` exit_status 0; no simulations, no training/refit, no validation64, no sealed test.
- Backup request: `{rel(BACKUP_REQUEST)}`.
- Latest verified backup before these new artifacts (from supervisor context): commit `{LATEST_BACKUP['commit']}`, package `{LATEST_BACKUP['package']}`, sha256 `{LATEST_BACKUP['sha256']}`.

Next action after verified backup only:
1. Run `experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21.py --run --backup-verified-commit <verified_commit> --i-accept-development-v21` with the legacy interpreter.
2. If fresh H12 catastrophic/high-cost rows appear, inspect features/telemetry and pivot to richer terminal-risk/value refit/training.
3. If zero bad and opportunity remains, run separate online selector-overhead smoke before any validation planning.

Sealed final test remains closed; validation64 is not authorized for v21.
"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(state_text, encoding="utf-8")
    # Hash state and request after writing for a concise completion marker.
    completed = {
        "passed": True,
        "created_utc": created.isoformat(),
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "simulations": 0,
        "training_episodes": 0,
        "gradient_steps": 0,
        "v21_source": rel(SCRIPT),
        "v21_source_sha256": hashes[rel(SCRIPT)],
        "smoke_registry": rel(SMOKE_REGISTRY),
        "backup_request": rel(BACKUP_REQUEST),
        "state": rel(STATE),
        "hashes": {**hashes, rel(BACKUP_REQUEST): sha256(BACKUP_REQUEST), rel(STATE): sha256(STATE)},
    }
    out = ROOT / f"research_artifacts/aws_diagnostics/vehicle_v21_post_smoke_state_update_{STAMP}/completed.json"
    write_json(out, completed)
    print(json.dumps({"completed": rel(out), "backup_request": rel(BACKUP_REQUEST), "v21_source_sha256": hashes[rel(SCRIPT)], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
