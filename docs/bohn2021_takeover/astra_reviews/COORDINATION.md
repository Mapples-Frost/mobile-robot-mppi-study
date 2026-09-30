# Reviewer-executor coordination contract

User reinforced on 2026-09-30: the agents must cooperate through evidence and outcomes, not operate as disconnected report writers.

1. Astra starts each follow-up review with LATEST.md and RESPONSE_LOG.md, then reads the cited new code, raw results and registry. Do not repeat a full old audit while ignoring subsequent experiments.
2. Preserve stable recommendation IDs across reviews. For each prior issue mark verified-resolved, still-open, contradicted, or not-yet-verifiable, with evidence. A claimed fix is not resolved until its result is checked.
3. GPT-5.5 records accepted/rejected/deferred, scientific rationale, precise next action, experiment ID, outcome and evidence paths for each actionable recommendation. A justified rejection is allowed; uncertainty must remain explicit.
4. At each safe cycle boundary prioritize the most informative unresolved issue. Run concrete controlled work when justified; avoid endless reporting or duplicating expensive completed experiments.
5. After important experiment results update RESPONSE_LOG.md with actual findings and new questions for Astra. Retain unresolved issues rather than silently dropping them.
6. Astra prioritizes up to three next actions, distinguishes new findings from repeated issues, and verifies whether the previous action changed the diagnosis. Its suggestions must account for already completed work and elapsed review time.
7. Only GPT-5.5 modifies scientific code or launches experiments. Astra remains read-only except for its reports. Do not interrupt a frozen experiment, open sealed tests, weaken baseline fairness or change the research goal through this handoff.
8. Neither agent waits idly for the other when independent useful work is available. Reviewer cycles remain bounded and periodic; temporary reviewer/API failure does not halt the executor.

## Role correction 2026-09-30

Astra is the primary scientific analyst and direction selector. GPT-5.5 executes the scientific plan and performs necessary numerical/implementation/operational checks. Meaningful new results trigger focused Astra analysis immediately, bypassing the six-hour broad-audit interval. NEXT_REVIEW_REQUEST.json and ANALYSIS_READY.json explicitly connect the requested evidence to its returned report. Frozen experiments finish normally; useful work within an approved plan continues while analysis is pending.

## Event wait reliability update 2026-09-30

Executor automatically creates an evidence-linked analysis request after substantive experiment outcomes, including failures. During a pending request the supervisor maintains heartbeat/backups and waits for a matching/superseding Astra report, instead of generating repetitive metadata-only API cycles. Resumption checks report SHA-256 and primary analyst identity. Successful reviewer API turns reset the consecutive-error backoff count; updated requests are adopted within the running focused analysis. These are scheduling/recovery controls, not API/token budgets.

## Thorough investigation reinforcement 2026-09-30

Apply SCIENTIFIC_LEAD_AUTHORIZATION_20260930.md. Astra issues precise tasks; GPT-5.5 executes and returns complete evidence. Approved dependent tasks need no duplicate scientific-direction approval when the stated gates pass.

## Executor timeout recovery 2026-09-30

Successful tool reads and writes now have persistent receipts and bounded-cycle contexts are saved after each tool result. API timeouts resume the same cycle with prior evidence. Registered experiment call IDs prevent replaying a completed/interrupted experiment solely because a model-facing tool response was lost. The executor request timeout is 900 seconds to allow long code-generation/reasoning responses; GPT-5.5/xhigh and the scientific plan are unchanged. Verified with simulated timeout/resumption and registered-experiment replay tests.
