# L278 Gate Status

- Status: complete, negative retention Gate.
- Decision: `recovery_retention_failed_anchor_probe_authorized`.
- Formal output: `results/research_platform/rl/l278_recovery_retention_diagnosis`.
- Summary SHA256: `674b7a91a0e96b82dc27d9a9e44ce415a4d414bd947c05bf50da50063932bb4a`.

The recovery Actor was intact at step 3,000, when it had received only its
first online SAC Actor update.  By step 6,000 the paired test teacher-action
RMSE had increased by 46.6%, 65.5%, and 140.2% across the three seeds.  Return,
reentry, and collision/boundary checks largely remained acceptable, so the
frozen interpretation is gradual recovery-policy forgetting under the later
unanchored Actor updates, rather than a numerical or immediate initialization
failure.

L278 authorizes only the preregistered L279 paired recovery-retention anchor
probe.  It does not authorize larger training or final-map evaluation.

