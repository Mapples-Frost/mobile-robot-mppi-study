# L223 six-scene MuJoCo confirmation artifact

This artifact freezes the complete development-seed evidence for the L223
six-scene feasibility confirmation. All six runs used MuJoCo 3.2.3,
ICODE-MPPI, seed 91001, `K=30`, one MPPI iteration, and Git commit
`b77ec38489995e8114838de1483a52c71b3c01e5`.

Outcome: **6/6 goal reached, 0/6 collisions**.

Contents:

- `summary/`: machine-readable cross-scene audit;
- `figures/`: vector PDF and 300 dpi PNG figures;
- `configs/`: frozen manifest and six scene configurations;
- `protocols/`: protocol, earlier three-scene report, and final Chinese report;
- `raw_results/`: complete per-run and per-episode metrics, trajectories,
  schedules, resolved configs, and provenance;
- `logs/`: captured stdout/stderr for each run.

This is a development result, not a sealed multi-seed claim. RL was disabled,
so the artifact validates the shared MuJoCo/ICODE-MPPI platform rather than
claiming superiority of the full RL-coupled method.
