# Table 01: L31 dynamic-variant development gate

- **Provenance**: ai-executed
- **Design**: 3 independent model blocks × 5 fresh episode seeds × 4 dynamic variants × 2 physics domains × 5 methods = 600 MuJoCo episodes.
- **Integrity**: 600/600 unique expected keys; 0 missing, duplicate, extra or invalid rows; 0 protected/sealed seeds used.
- **Primary method**: `temporal_gated_lcb_icode`.
- **Primary outcome**: 64/120 successes and 56/120 collisions.
- **Gate**: failed; hard zero-collision check failed and 1/5 efficacy checks passed.
- **Paired result versus spatial gate + ICODE**: +28 net successes, -28 net collisions, +0.639 m mean final-distance improvement (95% two-stage bootstrap CI 0.448--0.806 m).
- **Diagnostic ICODE effect versus temporal gate + nominal**: -5 net successes, +5 net collisions, -0.067 m mean final-distance improvement (95% CI -0.265--0.144 m).
- **Sources**: `results/research_platform/rl/l31_dynamic_variant_development_multiblock_20260715_v1/development_gate.json`, `paired_summary.csv`, `condition_summary.csv`.
- **Interpretation boundary**: Development evidence only; confirmation seeds `20283106--20283120` remained sealed.
