# Cancelled VPN deployment and timing provenance

User cancelled the VPN deployment. Staged Hysteria service was started briefly for setup, then stopped and disabled at 2026-09-27T14:08:16.405097+00:00. No cloud firewall was changed; no public client traffic test was run; staged files retained outside research repository.

ExecMainStartTimestamp=
ExecMainExitTimestamp=
ActiveState=inactive
UnitFileState=disabled

Downloads/setup and idle service overlapped safe-shortening devval shard05. Treat this overlap as a potential timing confound; do not silently exclude outcomes. Inspect resource samples and, if timing claims depend on affected blocks, register balanced timing remeasurement before drawing speed conclusions.
