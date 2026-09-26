# Unattended Bohn research service

Runtime location: /data/openai-agent/mobile-robot-mppi-study. Historical path is a persistent bind mount at /home/mapples/projects/mobile-robot-mppi-study; scientific scripts launch through it so frozen absolute-path records remain valid. No original registered source is rewritten for relocation.

Commands:

    sudo systemctl status bohn-research
    journalctl -u bohn-research --since '1 hour ago'
    cat /data/openai-agent/state/heartbeat.json
    cat /data/openai-agent/state/research_state.json
    cat /data/openai-agent/state/backup_status.json
    sudo systemctl stop bohn-research
    sudo systemctl start bohn-research

The service restarts after failure and reboot; it runs independently of SSH/Windows/Codex. Its job is ongoing scientific work, not a claim that success is guaranteed. Check heartbeat and evidence, not stale active flags. One Python experiment at a time; 4h experiment timeout; systemd 3100M cap, 180% CPU cap; single-thread numeric libraries; 12 GPT-5.5/xhigh calls per bounded iteration, with no daily call/token/API-cost limits and no local per-request output-token cap. Failure backoff is capped. State/SQLite and JSONL survive restarts. Iterations checkpoint and continue after5 seconds; only provider errors/rate limits trigger retry backoff.

Credentials are outside the repository under /data/openai-agent/.secrets, mode600. Never include that directory in an archive. Child experiments receive a cleaned environment, not API keys. The remote API smoke metadata is state/api_smoke.json. Only the verified model/effort is accepted.

Backup remote: https://github.com/Mapples-Frost/mobile-robot-mppi-study, branch codex/bohn-aws-20260926, release tag bohn-aws-evidence-20260926. Release is a research evidence archive, not a success publication. Code is committed/pushed before raw incremental packages. Each archive and accompanying file manifest has a SHA256 verified against GitHub's server digest or a full download. No formal results are deleted. Temporary archive files are removed only after verified upload. Existing Windows/WSL artifacts remain additional copies.

Restore: clone the branch; download the release assets; for every *.manifest.json verify its named archive SHA256; apply tar.gz packages in filename chronological order to a clean /data/openai-agent directory. Member paths are relative to that base and contain no credentials. Later packages replace earlier versions; deleted files are not pruned. The manifests identify exact intended versions. Recreate secrets separately, rebuild Python from recorded locks or restore runtime archives if present, install the mount/service units, then verify scientific hashes before running. Historical trained artifacts must not be overwritten to force a mismatched hash to pass.

Expiry: EC2 termination 2026-10-25 10:30 UTC. Research is switched to archival-only at08:00 UTC that day; target final recoverable delivery2026-10-24. No AWS termination/IAM/network/disk policy is modified.
