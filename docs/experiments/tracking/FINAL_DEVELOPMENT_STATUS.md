# Tracking Final Development Status

Status: **archived development line**  
Decision date: 2026-07-23  
Reason: the advisor has paused Tracking so the project can focus on stochastic
dynamic-obstacle prediction and risk-aware planning.

## What was established

- Path-conditioned Actors can learn sustained recovery sequences.
- L276 passed its recovery-initialization Gate: all three seeds improved
  teacher-action RMSE, median test RMSE improved by 73.94%, test return improved
  in five of six scenes, and collision/boundary failures decreased.
- Recovery is not a one-step action-selection problem. L275 showed that many
  recovery states become beneficial only after at least five committed steps.
- Linear and angular commands are coupled across the recovery sequence.
  L282--L283 found that replacing velocity or steering alone could not explain
  or recover the teacher gap.
- Fixed-checkpoint and gradient diagnostics were reproducible and retained
  finite, hash-audited artifacts.

## What did not pass

- Ordinary SAC continuation after the L276 initialization did not yield stable
  cross-seed closed-loop improvement (L277).
- The recovery behavior was gradually forgotten during later unanchored Actor
  updates (L278).
- A recovery-retention anchor reduced forgetting but interfered with, or did
  not preserve, useful online adaptation (L279).
- Component-separated and closed-loop roll-in anchors improved some recovery
  or validation metrics but failed their combined preregistered Gates
  (L281 and L284).
- Critic capacity, raw action representation, static privileged path features,
  and per-scene Critics were not sufficient explanations or remedies
  (L272--L274).
- No Tracking result currently supports stable full closed-loop success across
  independent seeds and final held-out maps.

## Interpretation boundary

The evidence supports three mechanism statements:

1. recovery requires temporally coordinated velocity and steering;
2. ordinary SAC updates can forget a recovery-capable initialization; and
3. sparse recovery replay and continuation-policy mismatch contribute to
   Critic/Actor ranking failures.

It does not establish that any current training intervention solves Tracking
generally. Long-horizon credit assignment, observation aliasing, and
multi-scene interference remain plausible contributors.

## Final experiment status

L285 was amended into a resource-limited, single-seed direction screen. Its
current artifact has 9/15 scheduled rows and is incomplete. It is descriptive
only and must not be represented as the original 45-episode confirmatory Gate.
No attempt will be made to finish or reinterpret it under the new program.

## Preserved evidence

- Protocols and Gate reports: `docs/experiments/post_l217/`
- Raw Tracking and RL outputs: `results/research_platform/rl/`
- Packaged research artifacts: `research_artifacts/`
- Frozen seed registries: `configs/seeds/`
- Advisor-figure generator: `figures/gen_fig_tracking_advisor_summary.py`
- Tracking tests: `tests/tracking/`, `tests/rl/`, and `tests/experiments/`

No file, result, seed, failed Gate, or incomplete run is deleted. Tracking code
remains available for future work, but the active research line now moves to
dynamic-obstacle uncertainty.

