#!/usr/bin/env python3
"""Metadata-only postdiagnostic after V1 fixed-H opportunity preflight.

No simulations, no training/refit, no historical validation64-bank access and no
sealed-test access.  The purpose is to check whether the enlarged V1 opportunity
probe can be launched immediately, estimate its bounded runtime from the already
completed V0 diagnostic, and preserve a precise next-run state.  It must not
open scenario-bank contents from validation/test splits.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_postdiagnostic_20260928T1048Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_probe_v1_preflight_postdiagnostic_20260928T1048Z.md"
BACKUP_PATH = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_FIXED_H_OPPORTUNITY_PROBE_V1_PREFLIGHT_POSTDIAGNOSTIC_20260928T1048Z.json"
V1_PREFLIGHT_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/completed.json"
V1_PREFLIGHT_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/raw.json"
V1_PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.json"
V1_PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.md"
V1_RUNNER = ROOT / "experiments/bohn2021_aws/vehicle_fixed_h_opportunity_probe_v1_runner.py"
V1_PREFLIGHT_SCRIPT = ROOT / "experiments/bohn2021_aws/vehicle_fixed_h_opportunity_probe_v1_preflight.py"
V0_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/completed.json"
V0_RUN_REGISTRY = ROOT / "research_artifacts/aws_runs/20260928T100320_b6f64eb5/registry.json"
V0_POST_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/completed.json"
THREE_LAYER = ROOT / "research_artifacts/aws_state/vehicle_fixed_h_opportunity_three_layer_diagnosis_20260928T1035Z.md"
MARKER = "vehicle-fixed-h-opportunity-v1-preflight-postdiagnostic-20260928T1048Z"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def verify_completed(path: Path) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise RuntimeError("completed marker did not pass: %s" % rel(path))
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise RuntimeError("access flags are invalid in %s" % rel(path))
    for name, expected in done.get("hashes", {}).items():
        p = ROOT / name
        if not p.exists():
            raise RuntimeError("hash-listed file missing: %s" % name)
        actual = sha256(p)
        if actual != expected:
            raise RuntimeError("hash mismatch for %s in %s" % (name, rel(path)))
    return done


def backup_proof_candidates(after_time: Optional[dt.datetime]) -> Dict[str, Any]:
    hits = []
    for path in sorted((ROOT / "research_artifacts/aws_backup_proofs").glob("*.json")):
        if path.name.startswith("REQUEST_"):
            continue
        try:
            proof = read_json(path)
        except Exception:
            continue
        verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
        remaining_ok = int(proof.get("remaining_changed_files", -1)) == 0
        has_commit = bool(proof.get("commit"))
        has_package_hash = bool(proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("packages_this_run"))
        proof_time = None
        for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
            proof_time = parse_time(proof.get(key))
            if proof_time is not None:
                break
        after_ok = after_time is not None and proof_time is not None and proof_time >= after_time
        if verified and remaining_ok and has_commit and has_package_hash and after_ok:
            hits.append({"path": rel(path), "sha256": sha256(path), "time": proof_time.isoformat(), "commit": proof.get("commit")})
    return {"adequate_count": len(hits), "adequate_proofs_after_preflight": hits[:5]}


def append_docs(raw: Dict[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## Vehicle fixed-H opportunity probe V1 preflight postdiagnostic\n\n"
        f"UTC: {raw['created_utc']}. Metadata-only readiness diagnostic completed with no simulations/training/validation64/test access. "
        f"V1 preflight hard_pass={raw['v1_preflight']['hard_pass']}; runner hash={raw['v1_runner']['sha256']}; "
        f"estimated V1 runtime from V0={raw['v1_runtime_estimate_from_v0']['estimated_runtime_seconds']:.1f}s; "
        f"adequate post-preflight backup proofs found={raw['backup_gate']['adequate_count']}. "
        f"Next experiment is blocked until backup covers V1 preflight/protocol/runner and this postdiagnostic; then run the V1 legacy fixed-H opportunity probe. "
        f"Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    v1_done = verify_completed(V1_PREFLIGHT_COMPLETED)
    v0_done = verify_completed(V0_COMPLETED)
    v0_post_done = verify_completed(V0_POST_COMPLETED)
    protocol = read_json(V1_PROTOCOL_JSON)
    v1_raw = read_json(V1_PREFLIGHT_RAW)
    registry = read_json(V0_RUN_REGISTRY)

    if protocol.get("protocol_id") != "vehicle_fixed_h_opportunity_probe_v1_frozen_20260928":
        raise RuntimeError("unexpected V1 protocol id")
    if protocol.get("split") != "vehicle_fixed_h_opportunity_probe_v1_fresh64_select16_no_validation64_no_test":
        raise RuntimeError("unexpected V1 split")
    if protocol.get("budget", {}).get("rollout_episodes_exact") != 160:
        raise RuntimeError("unexpected V1 episode budget")
    if protocol.get("budget", {}).get("new_gradient_steps") != 0:
        raise RuntimeError("V1 protocol unexpectedly includes gradient steps")
    preflight_time = parse_time(v1_done.get("created_utc")) or parse_time(v1_raw.get("created_utc"))
    backup_gate = backup_proof_candidates(preflight_time)

    v0_runtime = float(registry.get("runtime_seconds", 0.0))
    v0_episodes = int(v0_done.get("episodes", 80))
    v0_steps = int(v0_done.get("control_steps", 6313))
    v1_declared_episodes = int(protocol["budget"]["rollout_episodes_exact"])
    v1_control_bound = int(protocol["budget"]["control_step_upper_bound"])
    # Conservative runtime estimate: episode-scaled from V0 with 20% overhead;
    # also record the hard 4h cap check for the runner.
    estimated_runtime = (v0_runtime / max(v0_episodes, 1)) * v1_declared_episodes * 1.20
    estimated_steps = int(round((v0_steps / max(v0_episodes, 1)) * v1_declared_episodes))
    under_four_hours = estimated_runtime < 4 * 3600 and v1_control_bound <= 24000

    decision = {
        "next_action": "request/await external backup, then run vehicle_fixed_h_opportunity_probe_v1_runner.py with legacy interpreter and the verified post-preflight backup proof",
        "next_run_args": ["--backup-proof", "<verified_post_v1_preflight_backup_proof.json>", "--i-accept-fresh-development-rollout"],
        "why_not_run_now": "no adequate backup proof after the V1 preflight is present in repository evidence",
        "why_not_retrain_now": "V0 oracle opportunity was below materiality and V1 was just frozen to distinguish weak scenario opportunity from training/selector failure before refit/retraining",
        "if_v1_fails_materiality": "freeze versioned scenario-redesign or controlled continuation/value diagnostic rather than launch unchanged adaptive validation",
        "if_v1_passes_materiality": "freeze one IMPROVED selector/refit smoke with fair fixed-H retuning before any 3-seed campaign",
    }
    raw = {
        "created_utc": created,
        "method": "vehicle_fixed_h_opportunity_probe_v1_preflight_postdiagnostic_metadata_only",
        "classification": "metadata_only_readiness_and_backup_gate_no_simulation_no_training",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "v1_preflight": {"path": rel(V1_PREFLIGHT_COMPLETED), "sha256": sha256(V1_PREFLIGHT_COMPLETED), "hard_pass": v1_done.get("hard_pass"), "created_utc": v1_done.get("created_utc")},
        "v1_protocol": {"path": rel(V1_PROTOCOL_JSON), "sha256": sha256(V1_PROTOCOL_JSON), "split": protocol.get("split"), "budget": protocol.get("budget"), "materiality_thresholds": protocol.get("materiality_thresholds")},
        "v1_runner": {"path": rel(V1_RUNNER), "sha256": sha256(V1_RUNNER)},
        "v1_preflight_script": {"path": rel(V1_PREFLIGHT_SCRIPT), "sha256": sha256(V1_PREFLIGHT_SCRIPT)},
        "v0_inputs": {"completed": {"path": rel(V0_COMPLETED), "sha256": sha256(V0_COMPLETED), "headline": v0_done.get("headline")}, "postdiagnostic": {"path": rel(V0_POST_COMPLETED), "sha256": sha256(V0_POST_COMPLETED), "headline": v0_post_done.get("headline")}},
        "v1_runtime_estimate_from_v0": {"v0_runtime_seconds": v0_runtime, "v0_episodes": v0_episodes, "v0_control_steps": v0_steps, "v1_declared_episodes": v1_declared_episodes, "estimated_control_steps": estimated_steps, "estimated_runtime_seconds": estimated_runtime, "under_4h_cap_estimate": under_four_hours},
        "backup_gate": backup_gate,
        "decision": decision,
        "backup_request": rel(BACKUP_PATH),
        "input_hashes": {rel(p): sha256(p) for p in [V1_PREFLIGHT_COMPLETED, V1_PREFLIGHT_RAW, V1_PROTOCOL_JSON, V1_PROTOCOL_MD, V1_RUNNER, V1_PREFLIGHT_SCRIPT, V0_COMPLETED, V0_RUN_REGISTRY, V0_POST_COMPLETED, THREE_LAYER, Path(__file__).resolve()] if p.exists()},
    }
    write_json(OUT_DIR / "raw.json", raw)
    lines = [
        "# Vehicle fixed-H opportunity probe V1 preflight postdiagnostic",
        "",
        f"UTC: `{created}`. Metadata-only; no simulations, no training/refit, no historical validation64-bank access, no sealed-test access.",
        "",
        "## Readiness findings",
        "",
        f"- V1 preflight hard_pass: `{v1_done.get('hard_pass')}`; protocol split: `{protocol.get('split')}`.",
        f"- V1 runner hash: `{sha256(V1_RUNNER)}`.",
        f"- Budget: `{v1_declared_episodes}` episodes, <=`{v1_control_bound}` control steps, 0 training episodes, 0 gradient steps.",
        f"- V0-based runtime estimate with 20% buffer: `{estimated_runtime:.1f}` s; under 4h cap estimate: `{under_four_hours}`.",
        f"- Adequate backup proofs after V1 preflight found: `{backup_gate['adequate_count']}`.",
        "",
        "## Decision",
        "",
        f"Do not run V1 rollout until a verified external backup covers V1 preflight/protocol/runner and this postdiagnostic. Reason: `{decision['why_not_run_now']}`.",
        "",
        "Next runnable command after backup: legacy `experiments/bohn2021_aws/vehicle_fixed_h_opportunity_probe_v1_runner.py --backup-proof <verified_post_v1_preflight_backup_proof.json> --i-accept-fresh-development-rollout`.",
        "",
        "Retraining/refit remains deferred because V0 failed materiality; V1 is the frozen discriminator. If V1 passes materiality and a predictor gate, freeze one IMPROVED selector/refit smoke; if V1 fails, move to versioned scenario-redesign/continuation diagnostics rather than another unchanged adaptive campaign.",
        "",
        f"Backup request: `{rel(BACKUP_PATH)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 opportunity readiness state ({created})\n\n"
        f"Metadata-only postdiagnostic complete. V1 preflight hard_pass={v1_done.get('hard_pass')}; adequate post-preflight backup proofs={backup_gate['adequate_count']}; estimated V1 runtime={estimated_runtime:.1f}s. "
        "Next action: obtain/verify external backup, then run the V1 legacy fixed-H opportunity probe. No validation64/test access and no training occurred.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_PATH, {
        "requested_utc": created,
        "reason": "backup V1 preflight postdiagnostic plus V1 protocol/runner before enlarged fixed-H opportunity rollout",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(BACKUP_PATH), rel(V1_PROTOCOL_MD), rel(V1_PROTOCOL_JSON), rel(V1_RUNNER), rel(V1_PREFLIGHT_SCRIPT), rel(Path(__file__).resolve())],
    })
    append_docs(raw)
    files = [OUT_DIR / "raw.json", OUT_DIR / "summary.md", STATE_PATH, BACKUP_PATH, V1_PROTOCOL_MD, V1_PROTOCOL_JSON, V1_RUNNER, V1_PREFLIGHT_SCRIPT, Path(__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "backup_required_before_more_simulations": True,
        "backup_gate_adequate_count": backup_gate["adequate_count"],
        "backup_request": rel(BACKUP_PATH),
        "next_action": decision["next_action"],
        "hashes": {rel(p): sha256(p) for p in files if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "hard_pass": True,
        "new_rollouts": 0,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "backup_gate_adequate_count": backup_gate["adequate_count"],
        "estimated_runtime_seconds": estimated_runtime,
        "next_action": decision["next_action"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
