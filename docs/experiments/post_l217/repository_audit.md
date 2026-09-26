# Post-L217 / L223 repository audit

Audit date: 2026-07-20

## Execution baseline

- Working branch: `codex/post-l233-five-direction-experiments`
- Parent commit: `a9be4e933c3064707c04b0a61cf9930339de8d5e`
- Parent subject: `research:L223-six-scene-MuJoCo-confirmation`
- L217 preregistration/run commit: `9ec885007a56b3c3689422a45e450262fb9a4dc5`
- L217 archive commit: `43034be`
- L223 execution provenance commit: `b77ec38`
- L223 archive commit and current execution baseline: `a9be4e9`

The five-direction programme therefore starts from the latest L223 code and
does **not** reset the repository to L217. L217 remains the frozen evidence
baseline; L223 contributes the retained MuJoCo scenes, bounded simulator
scan-guard behaviour, path-preview/task representation and safety-preserving
engineering changes.

## Working-tree protection

The following pre-existing untracked files are user-owned and are outside this
programme. They must not be edited, staged or deleted:

- `ara/evidence/tables/table18_l214_final_point_goal.md`
- `docs/rl/214_final_paper_point_goal_results_2026-07-19.md`

No reset, clean or path checkout is authorised. Git inspection must be run by
the WSL Git executable because Windows Git reports false deletions for several
historical artifact paths whose names exceed the Windows legacy path limit.

## Frozen L217 evidence

- Raw results: `results/research_platform/rl/complex_navigation_sealed_l217/`
- Curated artifact: `research_artifacts/l217_complex_navigation_sealed_2026-07-20/`
- Frozen configuration: `configs/research/complex_navigation_sealed_l217.yaml`
- Preregistration: `docs/rl/217_complex_navigation_sealed_preregistration_2026-07-20.md`
- Result report: `docs/rl/218_l217_complex_navigation_sealed_results_2026-07-20.md`
- Data guide: `docs/rl/219_l217_data_quality_and_raw_data_guide_2026-07-20.md`
- Delivery guide: `docs/rl/220_l217_delivery_readme_2026-07-20.md`
- Validation log: `docs/rl/221_l217_validation_log_2026-07-20.md`

The integrity audit records 420/420 unique formal episodes: seven methods,
three scenes, two physics domains and ten sealed seeds, with 60 randomized
complete blocks and no qualification rows.

## Current method stack

The retained implementation contains:

- ordinary and value-aligned continuous-time control-affine ICODE residuals;
- nominal/learned combined rollout dynamics;
- residual-conditioned Actor and terminal Critic;
- role-aware reliability-weighted HSS and counterfactual proposal authority;
- MuJoCo 3.2.3 differential-drive plant and physics-domain overrides;
- LaserScan-derived local obstacle input, scan guard and final safety arbiter;
- variable point, waypoint and polyline task references;
- L223 expanded MuJoCo scenes and bounded simulator-only scan-guard settings.

The L217 formal matrix confirms the role-aware HSS efficiency contribution in
its tested complex-navigation scope. It does not independently establish the
value-alignment contribution, and the older formal rows do not demonstrate
full activation of every later residual-context or terminal-value mechanism.
L223's six-scene 6/6 record is a platform qualification, not a randomized
algorithm comparison.

## Baseline test status

The first full-suite audit produced `774 passed, 14 failed`. All 14 failures
shared one serialization cause: `ObservationEncoderConfig.to_dict()` returned
`path_preview_distances` as a tuple, while JSON reload returned a list, making
strict canonical manifest comparison fail. The compatibility fix serializes
that value as a list. The focused observation/demonstration/BC regression set
then passed `65/65`. The full suite is rerun after Tracking implementation.

## Protected boundaries

This programme must not weaken or bypass:

- the ROS hardware bridge and its Python-2 boundary;
- `/cmd_vel`, odometry or LaserScan contracts;
- `scan_guard`, local obstacle construction or safety arbitration;
- the rule that planner obstacles originate from perceived scans rather than
  global obstacle truth;
- the distinction between planner prediction dynamics and the MuJoCo plant.

L217 raw results, configurations and reports are read-only. Negative
development results remain preserved. Sealed seeds are never used for tuning,
smoke tests or geometry selection.

## Approved first implementation tranche

1. Freeze and checksum L217 evidence.
2. Add a unified experiment-artifact template.
3. Register disjoint development, sealed Tracking and sealed Dynamic seeds.
4. Add three constrained polyline Tracking geometries.
5. Add footprint-aware boundary and event telemetry.
6. Add the six specified Tracking unit-test files.
7. Run a four-arm by three-path, one-domain, one-development-seed smoke matrix.
8. Stop for human review before the 72-episode Tracking development matrix.
