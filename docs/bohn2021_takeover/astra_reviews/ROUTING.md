# Astra endpoint routing

The user authorized a backup endpoint on 2026-09-30. Primary remains https://api.sharesai.xyz/v1; backup is https://mdkj.lol/v1. Credentials live only in /data/openai-agent/.secrets/reviewer.env and astra_backup.env (0600); never include them in source, reports, usage records or Git.

Both routes request the same gpt-6-astra model, reasoning.effort=max, stateless Responses input, tools and output budget. Existing provider mapping max -> xhigh is accepted and recorded; a different model or lower/unknown effort is rejected. Returned model labels are provider claims, not independent model-identity attestation. No fallback to another model is permitted.

- Prefer primary. On retryable transport/auth/availability errors (including HTTP 429 and 5xx), replay the unchanged request to backup. At most two endpoint attempts per logical review call; payload HTTP 400 does not reroute.
- While backup is preferred, send a real gpt-6-astra/max health probe to primary every 120 seconds after the preceding probe/failure completes, respecting a longer Retry-After (bounded at one hour). No daily token quota.
- When primary is idle, check it every 300 seconds. Real successful review calls count as health evidence. Checks defer if a local primary request is still running; local lock contention does not mark the provider down.
- A successful primary probe restores primary preference for the next model call; an in-flight backup response finishes normally. Probe heartbeat and endpoint health persist in state/astra_router/status.json.
- Serialize requests within each endpoint. If both fail, retain review checkpoint and use persisted bounded exponential backoff; primary recovery wakes that cooldown. systemd restores the worker/monitor after process failure or reboot.
- Read the credential files afresh per request so a future authorized rotation does not require placing credentials in context. Exact API attempts, endpoint, call ID, returned effort, usage and timing are recorded. Unknown usage after interruption is explicitly unknown.

Only the independent reviewer uses this router. Opus remains scientific lead and GPT-5.5 remains executor. Review session, inspected file hashes and tools remain checkpointed; no scientific experiment or sealed test is rerun as part of routing.

Verification: authenticated backup model catalog and response smoke passed; native tool call plus stateless tool-result continuation passed with max; 15 isolated offline tests cover primary preference, failover, recovery, failed probes, unchanged body, model/effort pins, bounded attempts, accounting, secret redaction, and recovery races, incomplete HTTP bodies, invalid JSON response shapes, and stale observation ordering.

Run offline checks: `python3 -m unittest discover -s scripts/research_service -p test_astra_routing.py -v`.
Inspect routing: `cat /data/openai-agent/state/astra_router/status.json` and the nonsecret events.jsonl in that directory. Do not cat the credential files.


Full-context verification on 2026-09-30T11:22:09.600747+00:00: backup accepted the current read-only review context (468577 bytes; provider reported 141451 input tokens), returned completed gpt-6-astra/max with native read-tool calls, in 33.57 seconds. This shadow compatibility output was not executed and did not alter scientific evidence or the production review session. Detailed nonsecret record: state/astra_backup_full_context_smoke.json.


## 20260930T120717Z Long-review transport repair

The backup accepted short calls and full-context read-tool calls, but non-streaming final reports repeatedly hit Cloudflare HTTP 524 at the 120-second read window; primary long calls also timed out. Updated the shared transport to request Responses SSE (`stream=true`) following official OpenAI documentation (https://developers.openai.com/api/docs/guides/streaming-responses). The same model, max effort, full context, tool definitions and output budget are preserved. Only a validated full response.completed payload can be published; incomplete streams and partial text fail closed. Per-call stream progress/usage metadata is checkpointed, with existing bounded failover and primary recovery preserved. JSON fallback is accepted only if the endpoint ignores streaming and returns a validated completed object.

21 isolated tests passed, including partial-stream rejection and exact tool-payload/accounting preservation. A real backup SSE smoke returned gpt-6-astra/max completed successfully. Long production-report recovery is being verified separately.


Production recovery verified at 2026-09-30T12:14:51.913468+00:00: the preserved in-progress review completed through primary SSE in 312.0 seconds at gpt-6-astra/max, producing docs/bohn2021_takeover/astra_reviews/20260930T094912Z.md plus its manifest/transcript. The worker reset its failure streak and automatically started the next event-triggered review; scientific lead/executor continued independently. This verifies the long-call transport repair on a real review, not just a short smoke.
