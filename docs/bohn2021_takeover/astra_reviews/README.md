# Independent review / execution workflow

User authorized GPT-6 Astra at max on 2026-09-29 for comprehensive research review. GPT-5.5 at xhigh remains the executor. Astra has only bounded read/list/search/state tools; no command, write, experiment, or sealed-test tools.

Astra stores reports, inspected-source hashes and transcripts here. LATEST.md is published atomically only after a substantive report. GPT-5.5 records recommendation dispositions and experimental follow-up in RESPONSE_LOG.md. Reviews are advisory, not evidence of successful reproduction.

The reviewer runs independently as bohn-astra-reviewer.service, resumes persisted turns after a restart, performs at most 24 API turns per review, and reviews again no sooner than six hours after completion when research state has changed. No daily token cap or automatic model fallback. Calls and token usage join state/research.sqlite with the distinct model name. API credentials live outside Git in mode-600 .secrets/reviewer.env. Existing executor credentials are unchanged.

Service status: systemctl status bohn-astra-reviewer.service
Worker progress: /data/openai-agent/state/astra_reviewer/status.json
Evidence coverage: /data/openai-agent/state/astra_reviewer/checkpoint.json
Events: /data/openai-agent/state/astra_reviewer/events.jsonl

Report limitations include concurrent-source changes, development-data reuse and inaccessible sealed evidence. Each report must distinguish verified defects from hypotheses. Existing experiment budgets, data isolation, fair baselines and original/improved labeling remain mandatory.

## Provider compatibility evidence

2026-09-29: initial smoke echoed max; subsequent Responses requests, including ordinary text and tool calls, echoed xhigh despite max in request. Worker continues requesting max, accepts only verified max/xhigh, records returned effort per call and in each report manifest, and never claims unverified max execution. No model substitution. Chat Completions probe omitted effort metadata and was not selected. This limitation was disclosed to user.

## Scientific lead and executor (2026-09-30 supersedes previous periodic-only role)

Astra owns all major scientific audits, causal interpretation and direction decisions; GPT-5.5 implements, trains, runs experiments, extracts metrics and repairs implementation/infrastructure bugs. Substantive new completed/failed experiment results or explicit NEXT_REVIEW_REQUEST.json trigger focused Astra analysis immediately, bypassing the six-hour broad-review timer. ANALYSIS_READY.json links each completed analysis to its request and evidence. No daily token/call quota. Bounded cycles and backoff remain recovery controls. Current pending correction request covers v29 and its feature-separability follow-up.
