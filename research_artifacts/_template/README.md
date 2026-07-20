# Research artifact template

Copy this directory for each new experiment. Replace every `TBD` before a
formal run. Formal data are valid only when the preregistration is frozen,
`integrity_audit.json` passes, and all per-run provenance files bind the exact
Git/config/checkpoint/seed-registry hashes.

The `runs/` hierarchy is:

`runs/<method>/<scene>/<domain>/<seed>/`

Each leaf contains `config_resolved.yaml`, `trajectory.csv`, `metrics.json`
and `provenance.json`.

