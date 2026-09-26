#!/usr/bin/env python3
"""Write the 2026-09-26 latency-tree recovery/migration amendment.

This is a documentation-only action. It performs no simulations and does not read
validation or sealed-test result folders. It freezes the amended interpretation and
recovery rules before any resumed timing-sensitive latency-tree work.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
AMENDMENT = REPO / "docs" / "bohn2021_takeover" / "LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md"
MANIFEST = REPO / "docs" / "bohn2021_takeover" / "LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.manifest.json"
MARKER = "<!-- latency-tree-recovery-migration-amendment-20260926 -->"

EVIDENCE_PATHS = [
    "docs/protocols/bohn2021_latency_tree_2026-09-26.md",
    "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/registration.json",
    "research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/status.json",
    "docs/bohn2021_takeover/wsl/latest_completed_audit.json",
    "research_artifacts/aws_diagnostics/vehicle_selection_noise_diagnostic/summary.md",
    "research_artifacts/aws_diagnostics/vehicle_selection_noise_diagnostic/raw.json",
    "research_artifacts/aws_runs/20260926T114930_0c27a2b9/registry.json",
    "docs/bohn2021_takeover/TRAINING_FEASIBILITY.md",
    "research_artifacts/aws_diagnostics/training_feasibility_20260926/summary.json",
]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_once(path: Path, heading: str, body: str) -> bool:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER in text:
        return False
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + MARKER + "\n" + heading + "\n\n" + body.strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    evidence_hashes: Dict[str, str | None] = {p: sha256_file(REPO / p) for p in EVIDENCE_PATHS}

    amendment_text = f"""# Latency-tree recovery / migration amendment (2026-09-26)

Frozen UTC: {now}

Scope: Bøhn et al. 2021 vehicle and inverted pendulum reproduction work after the
move from the historical WSL checkout to the AWS t3a.medium host. This amendment
is documentation and governance only. It runs no simulations, reads no validation
outcomes, and reads no sealed test outcomes. It does not alter the historical
latency-tree preregistration at `docs/protocols/bohn2021_latency_tree_2026-09-26.md`;
that file remains immutable historical evidence. The depth-2 cross-entropy
latency tree remains an **IMPROVED** method, not the ORIGINAL SAC method.

## Evidence motivating the amendment

1. Source snapshots remain distinct: Windows prototype HEAD `a9aea0c`; WSL
author-based reproduction HEAD `cad9f76`. The migrated server evidence audit
reported 726 frozen files, 12,213 transitive files, and 0 mismatches.
2. Historical latency-tree status files are stale for liveness. The recorded WSL
PID `1694525` is no longer running, even though `status.json` still says
`active: true`.
3. Completed historical training artifacts currently verified at metadata level:
   vehicle seeds 0/1/2 and pendulum seed0 completed; pendulum seed1 started but
   partial; pendulum seed2 absent/unstarted pending exact inventory.
4. Vehicle selection diagnostics completed on AWS without simulations and without
validation/test reads (`20260926T114930_0c27a2b9`). The saved `policy.json` choices
match the reconstructed selection-race objective in all three vehicle seeds, so
the immediate issue is not a final-policy write/fallback mismatch. The objective
is nevertheless fragile: seed0 selected a behaviorally fixed H25 tree solely by a
small measured wall-time advantage over fixed H25; seed1 selected the fixed arm;
only seed2 selected a genuinely switching tree with aggregate training-selection
cost improvement, and its per-episode cost differences were heterogeneous.
5. A short AWS training feasibility probe is engineering evidence only. It showed
that both tasks can execute on the current host, but it did not establish formal
training throughput, CPU-credit sustainability, or scientific performance.
6. Supervisor backup context at 2026-09-26T12:08:22Z reports the GitHub release
backup `bohn-aws-evidence-20260926` as verified with `remaining_changed_files: 0`,
commit `1e93d44d9c0b10c1a1452eb12a88459c382b9e55`, and server-side SHA256
verification for the latest release assets. This satisfies the external
recoverability gate for resuming bounded development/formal work on this host,
subject to continuing normal backup practice for new artifacts.

## Recovery rules frozen by this amendment

### 1. No WSL/AWS timing mixing

Timing is host- and load-dependent. WSL decision-time measurements and AWS
decision-time measurements must not be pooled as if they were homogeneous samples.
A policy ranking, model-selection decision, or claim that uses actual wall time
must use a complete paired comparison block measured on the same host, in the
same run family, with the same construction/reset boundaries and randomized arm
order. If a candidate was generated on WSL and timing influences its selection on
AWS, all competing candidates and fixed comparators required for that selection
must be remeasured together on AWS before the selection is treated as formal.

Consequences:
- Existing WSL timing remains historical/development evidence.
- Existing completed policies may be reused as candidate policies, but not as
formal timing winners unless the relevant paired AWS timing block is rerun.
- Any resumed or fresh job whose objective includes timing must be AWS-only from
the beginning of the timing-sensitive block. Partial WSL timing cannot be
continued and combined with AWS timing for an objective value.

### 2. Behaviorally fixed trees are fixed-H comparators, not adaptive policies

A tree policy that emits one horizon for all evaluated control steps in a paired
block is classified as a fixed-H policy for that block, even if its JSON structure
contains tree nodes. Such a policy may be reported as a fixed-H comparator or as
evidence of adaptive-learning failure, but not as evidence for an adaptive
prediction-horizon effect. A reported adaptive policy must show at least two
actually used horizons and nonzero within-episode or within-block horizon changes
on the predeclared evaluation block. The horizon distribution is only a behavior
audit; it is not a speed or control-performance metric by itself.

### 3. Adaptive success requires behavioral adaptation and paired control benefit

For any adaptive-horizon claim, the policy must simultaneously satisfy all
inherited safety/non-inferiority gates and the following stricter interpretation
rules:
- non-fixed behavior under the predeclared evaluation block;
- paired raw total-cost improvement against the nominated fixed-H comparator, or
an explicitly separately reported timing improvement that survives AWS-only paired
remeasurement;
- no increase in constraint episodes, initial/final solver failure rates, or
complete-episode failure counts;
- actual decision and solver wall times reported as mean, median, and p95 with
reset/construction boundaries retained;
- all independently trained seeds reported, including failed/interrupted seeds;
- no best-seed-only conclusion.

A sub-percent wall-time difference between behaviorally identical policies is
classified as timing-noise-susceptible development evidence unless confirmed by a
fresh same-host paired timing design.

### 4. Historical artifacts retain their original interpretation

The historical latency-tree preregistration and all negative results remain
unchanged. This amendment cannot retrospectively turn failed or behaviorally fixed
historical policies into successes. Current vehicle seed interpretation is:
- seed0 `g3_c09`: behaviorally fixed H25 over the saved selection block; timing
noise susceptible; not adaptive evidence;
- seed1 `fixed`: fixed-H selection; not adaptive evidence;
- seed2 `g3_c08`: switching tree with aggregate training-selection cost
improvement but heterogeneous paired episodes; development evidence requiring
independent validation before any claim.

### 5. Incomplete pendulum recovery

The partial pendulum seed1 WSL run is preserved as an interrupted historical
artifact. Its timing-sensitive threshold-reference work must not be spliced into
an AWS timing objective. For any formal continuation of pendulum latency-tree
training, either:
- restart the affected timing-sensitive block on AWS under a fresh recovery run ID,
with the old partial run counted as failed/interrupted budget; or
- explicitly restrict reuse to non-timing metadata and perform any timing-based
selection in a new complete AWS-only paired block.

Pendulum seed2 is treated as unstarted unless an exact inventory audit finds
contrary evidence.

### 6. Validation/test access and model selection

The validation64 split remains unopened for post-amendment model selection until
code, candidate set, metrics, and selection rules for the validation pass are
frozen. The sealed test128 split remains unauthorized. To open final test, a
`final_test_gate.json` must first be published with frozen code/config/model
hashes and validation audit evidence, followed by a single explicit authorization
request through `update_state`.

Existing opened historical test results, if any in the migrated archive, are
classified only as development evidence for this recovery program and cannot be
used as the independent final test.

### 7. Fixed-H baselines remain strong and separately reported

The inherited fixed-H requirements are not weakened. Formal claims must include a
full reasonable H grid, independent-terminal and shared-terminal comparators when
required by inherited protocol, complete episode/control/solver/failure metrics,
paired scenario uncertainty, and all training/validation/simulation budgets.
Unequal search or training budget must be disclosed and cannot support an
equal-budget superiority claim.

## Immediate post-amendment queue

1. Run an exact metadata-only pendulum inventory audit: enumerate pendulum_s0 and
pendulum_s1 artifacts, completion markers, policies/checkpoints, selection
summaries, interruption/failure records, and confirm pendulum_s2 absence/unstarted
without reading validation or sealed test outcomes.
2. Use the verified external backup status as the gate allowing new bounded work,
while continuing to back up new artifacts.
3. Choose the next informative actual experiment under this amendment. Preferred
first formal-development action is an AWS-only paired remeasurement/validation
plan for already completed candidate policies and strong fixed-H comparators, or
an AWS-only restart of the missing pendulum timing-sensitive training block if the
inventory shows it is necessary. Any such experiment must be registered with zero
final-test access and with budgets counted separately from historical WSL work.

## Hashes of evidence files at amendment creation

```json
{json.dumps(evidence_hashes, indent=2, sort_keys=True)}
```
"""

    AMENDMENT.parent.mkdir(parents=True, exist_ok=True)
    if AMENDMENT.exists():
        existing = AMENDMENT.read_text(encoding="utf-8")
        if "# Latency-tree recovery / migration amendment (2026-09-26)" not in existing:
            raise RuntimeError(f"Refusing to overwrite unexpected existing amendment: {AMENDMENT}")
        wrote_amendment = False
    else:
        AMENDMENT.write_text(amendment_text, encoding="utf-8")
        wrote_amendment = True

    updates: Dict[str, bool] = {}
    updates["DECISIONS.md"] = append_once(
        REPO / "DECISIONS.md",
        "## 2026-09-26 latency-tree recovery/migration amendment",
        """
Decision: freeze a recovery/migration amendment before any resumed timing-sensitive latency-tree work. The historical preregistration is preserved unchanged. WSL and AWS wall-time measurements must not be mixed for ranking; any timing-influenced selection requires a whole same-host paired AWS block. Behaviorally fixed trees are fixed-H comparators, not adaptive policies. Adaptive claims now require actually used multiple horizons plus inherited safety/non-inferiority gates and paired cost or same-host timing benefit. The current vehicle training-selection evidence is development-only: seed0 is timing-noise-susceptible fixed-H25 behavior, seed1 is fixed, and only seed2 is a switching candidate requiring independent validation.
""",
    )
    updates["RESEARCH_LOG.md"] = append_once(
        REPO / "RESEARCH_LOG.md",
        "## 2026-09-26 latency-tree recovery/migration amendment",
        """
Wrote `docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md`. No simulations were run. No validation or sealed-test outcomes were read. The amendment records the vehicle selection-noise diagnosis, separates WSL and AWS timing evidence, classifies behaviorally fixed trees as fixed-H comparators, keeps final test sealed, and sets the next queue to a metadata-only pendulum inventory followed by AWS-only paired timing/validation or AWS-only pendulum recovery as appropriate. Supervisor context reports the GitHub release backup as verified with zero remaining changed files before new formal evidence is accumulated.
""",
    )
    updates["RESULTS_AUDIT.md"] = append_once(
        REPO / "RESULTS_AUDIT.md",
        "## 2026-09-26 amendment audit entry",
        """
Vehicle selection-noise diagnostics are recorded as development/training-selection evidence only. They support the finding that saved vehicle policies match the reconstructed selection objective but do not support a reproduction or adaptive-horizon success claim. Seed0 and seed1 provide fixed-H/nonadaptive evidence; seed2 is only a candidate. Any future timing claim must be based on AWS-only paired remeasurement. The amendment file and manifest record hashes of the preregistration, migration registration/status, and diagnostic outputs. Validation64 and sealed test128 remained unread during this action.
""",
    )
    updates["STATUS.md"] = append_once(
        REPO / "STATUS.md",
        "## 2026-09-26 recovery amendment status",
        """
Frozen recovery/migration amendment now governs further latency-tree work. Backup status from supervisor context is verified (`remaining_changed_files=0`, release `bohn-aws-evidence-20260926`, commit `1e93d44d9c0b10c1a1452eb12a88459c382b9e55`). Current scientific status remains: no reproduction success claim; final test unauthorized/sealed. Next action: exact metadata-only pendulum inventory, then decide an AWS-only paired remeasurement/validation or AWS-only pendulum recovery experiment under the amendment.
""",
    )
    updates["REPRODUCTION_PROTOCOL.md"] = append_once(
        REPO / "REPRODUCTION_PROTOCOL.md",
        "## 2026-09-26 amendment index",
        """
Additional governing document for migrated latency-tree work: `docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md`. It preserves the inherited preregistration unchanged but adds recovery rules for host-specific timing, incomplete pendulum runs, behaviorally fixed trees, validation access, and final-test gating. The amendment is stricter than the historical protocol where necessary and does not weaken any original success gate.
""",
    )

    manifest = {
        "created_utc": now,
        "action": "write_latency_tree_recovery_migration_amendment",
        "simulations_run": 0,
        "validation_accessed": False,
        "test_accessed": False,
        "amendment_path": str(AMENDMENT.relative_to(REPO)),
        "amendment_written": wrote_amendment,
        "amendment_sha256": sha256_file(AMENDMENT),
        "evidence_hashes": evidence_hashes,
        "doc_updates_appended": updates,
        "backup_context_recorded": {
            "status": "verified_from_supervisor_context",
            "remaining_changed_files": 0,
            "commit": "1e93d44d9c0b10c1a1452eb12a88459c382b9e55",
            "release": "https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926",
        },
        "next_queue": [
            "metadata-only pendulum inventory without validation/test reads",
            "then AWS-only paired remeasurement/validation plan or AWS-only pendulum recovery under amendment",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
