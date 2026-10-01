# Research-progress supervision and concrete failure diagnosis

The user asked why progress was slow and how supervision worked. Earlier supervision
checked service liveness and inspected state on user requests; it did not persistently
detect repeated zero-measurement cycles. That was insufficient.

## Evidence from the solo overnight run

As of 2026-10-01 01:50 UTC: 40 registered runs, 15 failures, 566.67 seconds of summed
registered run wall time, 867 GPT-5.5 calls, 17 measured solver calls, zero measured
plant transitions and zero training updates. Unknown/missing receipts remain unknown.
This is not new reproduction success, and it is not evidence of insufficient EC2 CPU.

## Verified backup cause and narrowed interface diagnosis

1. The original microcontinuation's `verify_pre_resource_backup_and_priors()` checks
   Git cleanliness of STATUS/RESEARCH_LOG/DECISIONS/registry and other live logs. It is
   repeatedly inherited through v0b...v0j; descendants add dated backup receipts, mtime
   requirements and more clean-tree checks. Research logs continue changing between
   backup and launch. Thus verified global recoverability can coexist with a rejecting
   wrapper gate. Stop adding another version-specific backup wrapper. Use a content
   proof for frozen code/config/plan/input/checkpoint files, preserve old failure
   evidence, and treat live logs as point-in-time records. `backup_contract.verify`
   checks verified release digests, pushed Git content and raw release-index hashes.
   It still rejects unbacked/modified immutable inputs and failed/partial backups.
   A dry content check verified the current v0j script, frozen v34z2 controller source,
   current solo report/plan and prior T-C2G raw against the pushed Git ref/verified
   release index. Live Git logs were clean at that particular check, so log dirtiness
   is a possible recurring rejection mechanism, not the only proven cause of the last
   failure. Another mismatch is that legacy gates scan named repo receipts while the
   newest verified receipt lives in STATE/backup_status.json. The supervisor now
   automatically materializes compatible verified receipts, retaining their real
   verification timestamp rather than fabricating a newer one.
2. The author `gym-horizon/gym_let_mpc/let_mpc.py:230` step method indexes `action[a_i]`
   at entry and expects a one-dimensional horizon action. Direct inspection confirms
   v0h line 302 and v0i line 285 ALREADY construct `np.asarray([float(h)], dtype=float)`.
   A scalar test fails at entry; one-element vectors reach a mock controller boundary
   with each correct H. Thus do not repeatedly "repair" the already-correct top-level
   action shape. The observed float-not-subscriptable exception can be deeper in the
   controller/TVP path; raw records only preserve repr(error), not the decisive stack.
   After fixing the backup gate, capture a full sanitized traceback in the smallest
   bounded source242 step diagnostic, before choosing another interface patch. Preserve
   horizons [12,15,35], goal61, V15_shared, existing raw-TVP repair and budget. The mock
   author entry-point test consumes no real solver calls or plant steps and does not
   prove the full environment works.

## Required immediate work in the existing scope

Publish one justified successor plan in solo mode. Consolidate the historical wrapper
chain at the shared backup gate and inspect the exact remaining environment traceback; preserve scientific/source
identity checks and failure receipts, but do not require mutable logs to stay Git-clean.
Check immutable inputs through the central recoverability verifier and satisfy any real
missing backup once. Then execute the already-approved saved-development microcontinuation
within the existing <=60 solver/<=30 plant budget. Record actual transitions and raw costs,
constraints, solver status and whole-decision/solver timing. A zero-transition result
cannot pass the closed-loop scientific gate. Do not declare improvement from a source
check or metadata receipt. No validation64 generation/opening, no sealed/final test.

## Persistent supervision

`bohn-progress-watchdog` samples registry/receipts each minute without model/API calls.
It distinguishes process completion, measured solves and plant/training transitions.
Repeated metadata-only cycles or repeated solves without any transition generate a
persisted stagnation alert supplied to every GPT-5.5 call. A bounded active experiment
is allowed to finish. The alarm does not pause the project or change scientific gates.
Up to three registered runs can proceed sequentially inside one bounded solo cycle;
no parallel training is introduced. Pre-launch failures do not consume a run slot.

Progress is assessed from raw measurement and protocol-valid comparisons, not uptime,
model-call/token counts or report counts. Opus and Astra remain paused per the user's
instruction. Model, reasoning effort, existing infrastructure and test isolation stay
unchanged. The actual watchdog output is `/data/openai-agent/state/progress_watchdog.json`.
