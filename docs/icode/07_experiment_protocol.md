# Research Experiment Protocol

## Claims and gates

Smoke results validate interfaces, determinism, artifact production, and
failure handling only.  A result is formal only after the full dataset,
training, fixed checkpoint, preregistered configs, and all seeds are run.
Do not tune a method on test/unseen results.

## Model prediction baselines

Compare nominal, oracle, MLP, and ICODE on test and unseen-disturbance splits.
Report residual derivative RMSE, one-step error, H-step state/position/heading
RMSE for `H=[1,5,10,20]`, endpoint errors, parameter count, and inference/
rollout time.  The unseen delay regime is explicitly non-Markov for the current
input and must not be interpreted as an exact-oracle case.

## Control baselines

```text
A nominal MPPI, no mismatch
B nominal MPPI, configured mismatch
C oracle residual MPPI
D MLP residual MPPI
E ICODE residual MPPI
```

The clean benchmark reports success, final distance, path length, bounds/
collision rate, mean absolute yaw rate, control jerk, stuck/spin steps, and
mean/p95/max planner time.  Minimum clearance is `not_applicable` in the
obstacle-free scene, not fabricated.  Formal obstacle clearance must come from
the protected LaserScan/live-simulation chain.

Smoke uses five fixed seeds.  Formal config uses ten seeds and can be extended
to twenty without code changes.  Sample-efficiency counts are
`[50,100,200,400]`; `--sample-sweep` activates the formal sweep.  Keep the
checkpoint fixed across the MPPI sample sweep.

## Required artifacts

Every runner writes resolved config, seed(s), Git SHA, run type, metrics JSON,
summary/trajectory CSV, figures where useful, checkpoint paths and hashes,
and compute timing.  Figures include method, split/scene, seed count, sample
count, and smoke/research status.

## Commands

```bash
# Model benchmark
.venv/bin/python experiments/icode/evaluate_residual_model.py \
  --dataset-dir results/icode/datasets/research_protocol \
  --mlp-checkpoint MLP.pt --icode-checkpoint ICODE.pt \
  --output-dir results/icode/model_metrics/formal

# Five-method smoke
.venv/bin/python experiments/icode/run_residual_mppi_benchmark.py \
  --mlp-checkpoint MLP.pt --icode-checkpoint ICODE.pt --smoke --headless

# Formal sample-efficiency sweep
.venv/bin/python experiments/icode/run_residual_mppi_benchmark.py \
  --mlp-checkpoint MLP.pt --icode-checkpoint ICODE.pt \
  --sample-sweep --headless
```

## Current smoke observations

The tiny 3-epoch/48-transition MLP and ICODE checkpoints did not improve clean
control and are slow in scalar Torch rollout.  These observations are retained
as honest framework diagnostics.  A separate two-seed Oracle smoke reached the
goal while nominal did not, validating the residual integration seam.  Neither
set is a paper result.
