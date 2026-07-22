# L276 Gate Status

Status: **PASS** (`recovery_initialization_gate_pass`).

- Three of three paired seeds improved held-out test teacher-action RMSE.
- Median relative test RMSE improvement: `0.7394`.
- Median validation/test discounted-return gains: `+7.1086 / +7.4738`.
- Median test corridor-reentry gain: `+0.4444`.
- Test return improved on five of six scenes.
- Maximum source-replay action drift: `0.0295` (limit `0.10`).
- Collision/boundary failures decreased from 30 to 10 paired evaluations.
- Critics, target critics, alpha, and normalizer remained hash-identical; all
  outputs were finite.

The result supports the L275 diagnosis that the Actor lacked a sustained
recovery continuation. It authorizes only the preregistered L277 small-budget
SAC probe, not final held-out-map evaluation.

