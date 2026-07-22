# L258 Curriculum Actor Held-out Tracking Development Gate Protocol

Date: 2026-07-22  
Status: preregistered development protocol; no Gate outcome read

## Question and scope

Does the validation-selected L258 curriculum Actor complete the three frozen
Tracking routes safely across multiple new development seeds when inserted into
the unchanged Value-Consistent ICODE + Path/Residual-Conditioned RL Prior +
role-aware Reliability-Weighted Value/HSS + MPPI stack?

“Held-out” here means excluded from L258 curriculum generation, training, and
checkpoint selection. The geometries have historical development results, so
this Gate is not a sealed or formal paper confirmation.

## Frozen model selection

- Selector: `experiments/rl/select_actor_checkpoint_lexicographic.py`.
- Selection input: the three valid L258 runs only. The chained seed42 run is
  engineering-invalid raw and is excluded without deletion.
- Selected checkpoint: L258 seed `20262441`, step `60000`.
- Checkpoint SHA256:
  `a8ea2ae001493db1bde2eea6aeb38ffb712ec39977794e3c475199fd13e75429`.
- Tracking outcomes are forbidden for checkpoint selection or retraining.

## Frozen Gate design

- Status: development qualification (`qualification=1`).
- Scenes: the L239 `W=4.0D` Hairpin, S-Chicane, and Infinity routes.
- Development seeds: `923301011`, `923301012`, `923301013`.
- Physics: `nominal_seen`; MuJoCo `3.2.3`.
- Arms: `icode_mppi` and `full_proposed`.
- Equal sampling budget: 100 rollouts per decision. Full uses 50 candidates x
  2 iterations; ICODE uses 100 candidates x 1 iteration.
- Max steps: Hairpin `2210`, S-Chicane `1405`, Infinity `2030`.
- Network, Actor/traditional fusion, ICODE/value checkpoints, reliability/HSS,
  reward, MPPI cost, maps, boundary handling, LaserScan, scan_guard, and safety
  arbitration remain inherited and unchanged from L247/L244.

The complete design is 3 scenes x 3 seeds x 2 arms = 18 unique episodes.

## Integrity Gate

Before method effects are read, require:

1. all 18 unique scene x seed x arm keys;
2. qualification `1`, `nominal_seen`, MuJoCo `3.2.3`, and the frozen step budgets;
3. selected Actor path and SHA, manifest/config/Git provenance, and per-episode
   `config_resolved.yaml`, `trajectory.csv`, `metrics.json`, and `provenance.json`;
4. no duplicate keys, Traceback, Exception, OOM, NaN, or Inf;
5. all collisions, boundary violations, stalls, max-step failures, and fallbacks
   retained without rerunning or filtering completed episodes.

## Pre-registered assessment

Report per scene and seed: success, completion, termination reason, collision,
boundary violation, minimum footprint margin, cross-track error, safety burden,
RL proposal/authority diagnostics, no-feasible/fallback diagnostics, and planner
time.

The curriculum Actor passes this development Gate only if:

1. Full has zero collisions and zero boundary violations across all 9 episodes;
2. Full completes all three routes in at least two of the three development
   seeds (at least 6/9 successes), with every scene successful in at least one
   seed;
3. Full mean completion is no lower than ICODE and no scene-level mean regresses
   by more than `0.02`;
4. Full proposal authority is finite and non-trivial in every scene;
5. no safety, fallback, or planner-time diagnostic is missing or non-finite.

Passing is development evidence only. It does not authorize sealed evaluation
or formal paper claims without a separately frozen confirmation protocol.

## Reporting rule

During execution, inspect only progress, uniqueness, artifacts, resources, and
exceptions. Read method outcomes only after all 18 episodes finish. A negative
Gate retains raw data plus a short status only; no plot or detailed report. A
positive Gate receives the detailed Chinese report, figures, and unified
desktop delivery.
