# L258 Tracking Curriculum Protocol

Status: development preregistration. This protocol is not a sealed result.

## Frozen objective

Train a path/residual-conditioned SAC prior on procedural tracking families so
that the learner sees geometric variation without seeing the held-out final
tracking routes. The Value-Consistent ICODE + Path/Residual-Conditioned RL
Prior + role-aware Reliability-Weighted Value/HSS + MPPI method, actor/traditional
prior fusion, network, reward, planner cost, maps, and safety chain remain
unchanged.

## Budget and seeds

- Three independent Windows CUDA runs: seeds `20262441`, `20262442`, `20262443`.
- Exactly `60000` learner steps per seed, frozen before any L258 result is read.
- Validation seed bases: `20263441`, `20263442`, `20263443`.
- L234/L247-L257 development seeds and all sealed seeds are excluded.
- Runs are sequential on this 8 GB GPU host if the committed-memory audit
  requires it; batch, network, and the 60000-step budget are not reduced.

## Procedural training domain

The generator creates only the following training-only path families:

1. randomized S curves;
2. randomized hairpin-like turns (no held-out Hairpin points are copied);
3. double loops and near self-intersections;
4. monotone curvature ramps;
5. obstacle-corridor polylines.

Scale, direction, start pose, curvature, corridor width, and physics-domain
role are randomized by the generator seed. A blocked manifest records family
and physics-role coverage for every generated route.

## Leakage and integrity rules

- No exact held-out route files, point lists, names, outcomes, or checkpoints
  may be read by the generator or training configs.
- Generated files live below `configs/research/l258_curriculum/` and carry a
  `l258_generated` marker plus their generator seed.
- The leakage test rejects held-out route identifiers and direct references to
  L234-L257 result directories.
- Every run must retain raw checkpoints, replay, logs, config snapshot, and
  provenance. Failures are retained and reported only as a short Gate status.

## Execution order

1. Run generator and leakage/determinism/config-contract tests.
2. Commit and push the protocol, generator, tests, generated route configs, and
   three run configs.
3. Select the L257 checkpoint using the frozen validation dictionary order,
   never by Tracking outcome.
4. Launch the three 60000-step Windows CUDA runs sequentially when required by
   the memory audit.
5. During training inspect engineering progress only. Read method outcomes only
   after all three runs are complete.
