# L260 Corrected Tracking Small-Budget Protocol

Status: development preregistration. This protocol is not sealed evidence.

## Motivation and invalid predecessor

L258 is frozen as `engineering-invalid`: its generated path files inherited the
fixed `initial_state` and four static walls from
`mujoco_l218_serpentine_polyline.yaml`.  L258 raw checkpoints, replay, and Gate
artifacts remain immutable, but no L258 checkpoint or outcome may initialize,
select, tune, or justify L260.

## Frozen question

Does the existing Direct SAC Actor improve when trained on internally
consistent Tracking environments, without adding behavior cloning or changing
the algorithm?  L260 changes only the training/validation geometry package.
Value-Consistent ICODE, the Path/Residual-Conditioned RL Prior, role-aware
Reliability-Weighted Value/HSS, MPPI, the Actor/Traditional fusion, network,
reward, planner cost, and safety chain remain frozen.

The inherited L257 checkpoint retains its historical BC-anchor provenance,
but `rl.training.bc_anchor.enabled=false` is frozen for L260.  Thus L260 adds
online SAC experience only and performs no new supervised update.  This is an
explicit treatment choice, not an attempt to relabel historical training.

## Initialization whitelist

All runs initialize actor and normalizer only from:

`results/research_platform/rl/l257_path_preview_bc_anchor_seed20262333_120k_v1/checkpoints/step_000030000.pt`

SHA256:
`fc9166f5c3010156a7c9fad4cb9d155ac222447506ff2ebdfe6c2205250a1547`

This checkpoint was selected by the frozen L257 validation dictionary before
L260.  Every path below `l258_tracking_curriculum_*` is forbidden.

## Fixed map design

- Six immutable training maps and three immutable validation maps.
- Every YAML explicitly owns `experiment.initial_state`, `task.points`,
  `scene.field_size`, and `scene.obstacles`.
- Initial XY equals the first reference point and yaw equals the first segment
  heading.
- The entire reference must retain at least `0.08 m` clearance after inflating
  obstacles by the `0.25 m` robot collision radius.
- Map geometry is fixed.  The small-budget screen uses nominal physics only,
  isolating the geometry/data correction before any domain-randomized scale-up.
- Two deterministic six-map exposure blocks cover every training map once
  before 10k and twice before 20k.
- A training-only initial-state curriculum uses audited start, 25%, 50%, and
  75% path anchors.  Validation always starts from the configured start; the
  Actor never observes anchor identity or privileged simulator geometry.
- Exact Hairpin, S-Chicane, and Infinity geometry, trajectories, outcomes, and
  sealed seeds are forbidden from training and checkpoint selection.

## Mandatory preflight

Training is prohibited until all of the following pass:

1. deterministic generation and unique geometry fingerprints;
2. explicit-field/source-versus-resolved configuration equality;
3. footprint path clearance, free start/goal, and connectivity;
4. field-boundary reserve and exact-heldout-point-list rejection;
5. train/validation seed and geometry isolation;
6. one short Windows-native MuJoCo smoke per map;
7. finite observation, action, reward, clearance, and progress;
8. checkpoint initialization SHA and forbidden-path checks;
9. checkpoint/replay exact-resume smoke.

The nine map top views and a machine-readable preflight manifest are reviewed
before any 10k run starts.

## Budget and comparison

- Three independent Windows CUDA seeds: `20262501`, `20262502`, `20262503`.
- Validation seed bases: `20263501`, `20263502`, `20263503`.
- Exactly `20000` environment steps per run, with immutable checkpoints at
  `10000` and `20000`.
- Compare the frozen L257 initialization, 10k, and 20k checkpoints using only
  the three L260 validation maps and their frozen seed/physics blocks.
- No final Tracking outcome may select a checkpoint or alter training.

## Decision rule

Before result access, the validation analysis must report collision, boundary,
completion/progress, cross-track error, Actor/Gaussian opportunity and elite
yield, MPPI cost gap, proposal authority, safety burden, and planner time.
Only a safety-noninferior reduction in Actor-versus-Gaussian cost gap with
nonzero Actor elite yield permits a later small final-map development Gate.
If neither 10k nor 20k improves these prespecified diagnostics, stop RL scaling
and audit reward/action/MPPI alignment before considering supervised learning.

Negative results retain raw artifacts and a short Gate status only.
