# Local Opus 5.5 takeover

Read `GPT55_HANDOFF_REPORT.md`, `OPUS_RESUME_MEMORY.json`, `GPT55_HANDOFF_READY.json`, then `DELIVERY_VERIFIED.json` and `COVERAGE_VERIFICATION.json`. The server GPT-5.5 authored the scientific report from actual source, raw results and registry evidence. This handoff does not claim reproduction success.

Repository: https://github.com/Mapples-Frost/mobile-robot-mppi-study

Use branch **codex/bohn-aws-20260926**, not the default branch.

Handoff release: https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-local-opus-handoff-20261001

Clone into a new directory to preserve the user's existing dirty workspaces:

```sh
git clone --branch codex/bohn-aws-20260926 https://github.com/Mapples-Frost/mobile-robot-mppi-study.git mobile-robot-mppi-study
```

Raw results, checkpoints, source snapshots, non-secret server state and runtimes are stored in immutable GitHub release archives. The final HANDOFF_INDEX.json.gz maps every evidence file to its final verified archive. Downloading only the newest incremental archive is not a complete restore.

For complete recovery, use Python >=3.8 and a new transfer base with sufficient free space. Clone source into its `mobile-robot-mppi-study` child directory before restoring. Plan-only reports exact storage and download requirements:

```sh
python3 scripts/research_service/restore_handoff.py --index https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-local-opus-handoff-20261001/HANDOFF_INDEX.json.gz --destination /path/to/new/transfer-base --plan-only
python3 scripts/research_service/restore_handoff.py --index https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/download/bohn-local-opus-handoff-20261001/HANDOFF_INDEX.json.gz --destination /path/to/new/transfer-base
```

The restorer verifies each archive and selected file by SHA256, resumes via a checkpoint, and keeps only one download at a time. It does not start research or call a model. Server upload verification is distinct from a completed full local restore.

The existing EC2 backend remains available at `ubuntu@18.236.70.13`, with Windows SSH key `D:\chorme\openai-agent-key.pem`. A local Opus coordinator can use it without downloading approximately 60 GB locally. Server source root: `/data/openai-agent/mobile-robot-mppi-study`. The old research, watchdog, Opus and Astra services are disabled for takeover; local Opus is the sole coordinator.

Exact environment locks, system package versions, service definitions and symlink inventory are in `environment/`. Legacy scientific Python is 3.7.16; modern supervision Python is 3.12.3; server OS is Ubuntu 24.04. Archived Linux runtimes are not native Windows programs. Scientific scripts contain absolute server/legacy paths and require explicit mapping for a new local Linux workspace. Preserve existing workspace paths and environments until their use is audited.

Credentials are intentionally excluded from GitHub. Permission-restricted server secret files remain available to the authorized local coordinator. AWS infrastructure and scheduled expiration are unchanged. Current instance remains 2 vCPU/4 GB; termination remains 2026-10-25 18:30 Asia/Shanghai.

Keep ORIGINAL versus IMPROVED clear; audit prior test/validation exposure; preserve failed seeds and negative results. No reproduction-success claim is justified yet. Strong fixed-H search, fair budgets, at least three independent training seeds, frozen independent final tests, full provenance, and real wall/solver timing remain necessary. The server GPT report and LOCAL_OPUS_PROMPT.md identify immediate debugging priorities.
