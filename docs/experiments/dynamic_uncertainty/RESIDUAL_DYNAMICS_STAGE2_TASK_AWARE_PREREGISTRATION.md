# Residual Dynamics Stage 2: Task-Aware Retraining Preregistration

Status: frozen before training or evaluating any revised checkpoint.

## Motivation

Stage 1 showed that the three L57 checkpoints improve held-out H36 dynamics
prediction but degrade closed-loop collision safety.  Amendments 5 and 6 also
showed that a dual-model safety shield and speed-only residual authority do not
remove the failure.  No further threshold or wrapper tuning is authorized for
the L57 checkpoints.

The primary Stage 2 hypothesis is that L57 lacks dynamic-obstacle operating
states: its frozen training set contains generic high-dynamic paths but no
near-obstacle trajectories from the current task.

## Fixed intervention

1. Keep the L57 ICODE architecture, normalization, residual output mask
   `[0, 0, 0, 1, 1]`, RK4 integrator, and H36 objective.
2. Initialize each Stage 2 model from its matching L57 seed checkpoint.
3. Append complete collision-free nominal-controller episodes:
   - train: obstacle seed `730100001`
   - validation: obstacle seed `730100003`
   - test: obstacle seed `730100005`
4. Never split an episode across partitions.
5. Retain all original L56 rows with unit sample weight.  Give task rows a
   positive weight
   `4 + 4 exp(-max(clearance, 0) / 0.5) + 2 p_max`.
6. Apply an output-anchor penalty on original
   `nominal_mppi_onpolicy` rows so the revised model remains close to the
   corresponding L57 function away from the new task data.
7. Use learning rate `5e-5`, at most 60 epochs, and validation patience 12.
8. Keep RL disabled throughout Stage 2 residual qualification.

The sample weight is an importance coefficient, not a collision probability:
it says that an error close to the moving obstacle matters more to the training
objective than an equally sized error on an ordinary path.

## Leakage and interpretation rules

- Seed `730100003` is model-selection data and may not be used as a Stage 2
  closed-loop claim.
- Seed `730100005` is offline test data and may not be used for checkpoint
  selection.
- The original L56 test and unseen partitions remain evaluation-only.
- A passing offline gate permits only fresh development-seed closed-loop
  evaluation.  It is not evidence of collision safety.
- Sealed confirmation seeds remain unopened.

## Offline gate

Evaluate all contiguous H1/H5/H10/H20/H36 windows.

For each of three independently fine-tuned model blocks:

1. task-test active derivative RMSE must beat nominal;
2. task-test H36 endpoint position RMSE must not exceed its matching L57
   checkpoint;
3. original L56 test and unseen H36 endpoint position RMSE may degrade by at
   most 2% relative to the matching L57 checkpoint;
4. all metrics must be finite and artifact provenance must match the frozen
   dataset hashes.

The block gate requires at least 2/3 blocks to satisfy all four conditions.
If it fails, retain the negative result and do not launch Stage 2 closed-loop
experiments.

## Closed-loop gate after offline qualification

Use newly registered development obstacle seeds `730100006`, `730100008`, and
`730100010`, nominal and revised-residual paired conditions, and the existing
collision-first automatic gate.  Do not reuse the three task-data seeds for a
success claim.  Before the full matrix, run the fixed pilot pair on
`730100006`: nominal followed by revised block seed `20261201`.

The Stage 2 deployment stack uses the revised model directly (no x/y
canonicalization, reliability gate, stall latch, or speed-only authority) and
retains the full-action dual-model noninterference shield.  This avoids mixing
the retraining intervention with Stage 1 mechanisms that already failed while
still retaining exact nominal fallback.

The residual condition must have zero collisions and must not be worse than
nominal in collision count.  Runtime remains a separate engineering gate.
