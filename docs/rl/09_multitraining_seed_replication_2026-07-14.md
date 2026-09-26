# RL Gate L12 replication across independent training seeds

Date: 2026-07-14

Status: **L12 outcome-balanced replay efficacy gate failed**.  The code remains
available as a configurable ablation; it is not the selected paper method.

## 1. Why this gate was mandatory

Training seed 20260718 produced a compelling controlled-localization result:
L12 succeeded in 10/10 held-out U-trap episodes while uniform replay succeeded
in 5/10.  A single RL training seed is not sufficient evidence because neural
initialization, stochastic actions, episode resets and replay composition all
change the optimized policy.

The replication therefore added two independent paired training seeds.  Each
pair changed only replay sampling:

- `l12`: `outcome_balanced`, 25% transitions from completed successful
  episodes after such data became available;
- `uniform`: ordinary uniform replay;
- identical SAC architecture, reward, reverse-reset curriculum, MuJoCo plant,
  MPPI settings, near-goal handoff and safety stack;
- ICODE and Memory disabled;
- controlled ground-truth/mocap-like localization;
- 40k environment steps per model;
- each method's best checkpoint evaluated on the same held-out seeds 101--110.

The four new training jobs ran concurrently to reduce wall-clock time.  All
held-out evaluations ran serially, so planner timing was not measured under
training contention.

## 2. Reproducible seed control

`experiments/rl/train_rl_sampling_prior.py` now accepts:

```text
--seed N
```

The override is written to both `experiment.seed` and `rl.training.seed`, and
therefore appears in checkpoint/run metadata.  This avoids copying config files
and makes the training seed an explicit experiment factor.

Future configurations additionally use:

```yaml
rl:
  training:
    validation_seed_base: 20270718
```

This separates optimization randomness from model-selection randomness.  A
missing value preserves the legacy `training_seed + 100000` schedule so old
checkpoints remain reproducible.  The retrospective results below remain valid
because final comparison uses the same independent held-out seeds for every
trained model; the fixed model-selection seed applies to future runs.

## 3. Training validation

All validation collisions were zero.

| training seed | method | best validation success | best mean distance | 40k success |
|---:|---|---:|---:|---:|
| 20260718 | L12 | 5/5 | 0.296 m | 5/5 |
| 20260718 | uniform | 2/5 | 0.543 m | 0/5 |
| 20260719 | L12 | 0/5 | 1.664 m | 0/5 |
| 20260719 | uniform | 1/5 | 0.999 m | 0/5 |
| 20260720 | L12 | 0/5 | 2.390 m | 0/5 |
| 20260720 | uniform | 5/5 | 0.290 m | 5/5 |

Seed 20260720 demonstrates that uniform replay can also produce the strongest
possible training-validation result.  The original L12 win was therefore not
evidence that outcome balancing was necessary.

## 4. Common held-out evaluation

Each row below is one independently trained best checkpoint evaluated over the
same ten environment seeds.

| training seed | L12 success | uniform success | L12 distance | uniform distance |
|---:|---:|---:|---:|---:|
| 20260718 | 10/10 | 5/10 | 0.291 m | 0.515 m |
| 20260719 | 0/10 | 1/10 | 1.591 m | 1.210 m |
| 20260720 | 0/10 | 10/10 | 2.479 m | 0.293 m |

Pooled and training-seed-level aggregate:

| metric | L12 | uniform replay |
|---|---:|---:|
| pooled success | **10/30 (33.3%)** | **16/30 (53.3%)** |
| training-seed success mean | 0.333 | 0.533 |
| training-seed success std | 0.577 | 0.451 |
| final-distance seed mean | 1.454 m | 0.673 m |
| final-distance seed std | 1.100 m | 0.478 m |
| safety interventions seed mean | 56.4 | 29.4 |
| collision | 0/30 | 0/30 |
| mean planner compute | 4.74 ms | 4.80 ms |

Result artifact:

```text
results/research_platform/rl/multitraining_seed_ablation_20260714/
  per_training_seed.csv
  aggregate.json
```

The evidence rejects L12 as a robust improvement.  It is worse than uniform
replay in pooled success, final distance and safety interventions, despite one
excellent seed.

## 5. Why outcome balancing was unstable

Post-hoc episode audit found no stochastic training episode that succeeded from
`original_start` for any of the six models.  Successful replay data came from
shorter curriculum segments:

- `terminal_approach`;
- `trap_exit`;
- occasionally `left_upper`.

Thus “successful episode” was too coarse a label.  A fixed 25% sampler could
over-represent locally successful route fragments without teaching how to
connect the original start to those fragments.  The policy sometimes learned
the connection by generalization (L12 seed18 or uniform seed20), but this was
not stable across optimization seeds.

This is stronger evidence than simply saying that the network needs more
epochs: all models already trained for 40k steps, and the winner changed with
the seed.

## 6. Multi-seed aggregation tool

`experiments/rl/summarize_training_seed_ablation.py` requires explicit
`METHOD=TRAINING_SEED=RESULT_DIR` inputs.  It checks that every model of a method
uses the same held-out evaluation seeds and writes:

- per-training-seed metrics CSV;
- pooled success/collision rates;
- mean and sample standard deviation across training seeds;
- provenance paths for every result directory.

This prevents a future report from selecting only the most favorable trained
policy.

## 7. Decision and next gate

1. Keep `outcome_balanced` replay as a documented ablation, default off.
2. Do not expand L12 to additional scenes or combine it with ICODE.
3. Use uniform replay as the current RL reference, while explicitly reporting
   its own large training-seed variance.
4. Replace whole-episode success balancing with a mechanism that connects route
   segments.  Candidate next gate: demonstration/behavior-cloning bootstrap
   from the already verified scripted local-subgoal route, followed by SAC
   fine-tuning and BC-only/SAC-only/BC+SAC ablation.
5. Use fixed validation seeds for all future multi-training-seed runs.
6. Require at least three training seeds at smoke/research-gate stage and five
   for the paper table.
7. Only after one method passes this gate, test new layouts, dynamic obstacles,
   localization robustness, and finally nominal-versus-ICODE rollout dynamics.

## 8. Verification

The new seed override, validation-seed separation and aggregation code have
unit tests covering range validation, legacy behavior, fixed seed schedules,
inconsistent held-out seed rejection, pooled metrics and training-seed sample
standard deviation.
