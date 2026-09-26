# Strong MPPI + MuJoCo Baseline Contract

## Purpose

This baseline is the non-learning control reference for all later residual and
RL ablations. It deliberately keeps both Memory and RL disabled and uses the
nominal five-state dynamic-unicycle model inside MPPI while the commanded robot
is simulated by an actuated differential-drive MuJoCo plant.

The canonical configuration is:

```text
configs/research/mujoco_strong_mppi_baseline.yaml
```

The clean single-obstacle scene requires a real detour but does not require a
separate global planner. The older `mujoco_diff_drive.yaml` labyrinth remains
available as a harder local-planning stress test; failure there must not be
silently relabeled as a dynamics-learning failure.

## Closed-loop contract

```text
LaserScan -> scan_guard -> local_obstacle_layer -> MPPI proposal
          -> safety arbitration -> issued command -> MuJoCo true plant
                                  -> feedback to next MPPI iteration
```

The planner never receives MuJoCo obstacle truth. Obstacles used in rollout
continue to come from LaserScan through `local_obstacle_layer`. `scan_guard`
always remains downstream of MPPI and has final authority.

When `scan_guard` blocks forward motion, the actual issued command is fed back
to MPPI. The next warm start clears translation for a configurable prefix but
preserves angular control. This allows the robot to rotate in place instead of
repeatedly proposing a forward command that safety must reject.

## MPPI update

Rollouts are sampled as

\[
U^{(k)} = \bar U + \epsilon^{(k)},
\]

weighted by

\[
w_k = \frac{\exp(-(S_k-\min_j S_j)/\lambda)}
           {\sum_j \exp(-(S_j-\min_j S_j)/\lambda)},
\]

and updated in perturbation form:

\[
U_{\mathrm{new}} = \bar U + \sum_k w_k\epsilon^{(k)}.
\]

With a non-zero Gaussian proposal, the optional likelihood-ratio correction is

\[
S_{\mathrm{IS}}^{(k)} =
\lambda R \sum_t \bar u_t^\mathsf{T}\Sigma^{-1}\epsilon_t^{(k)}.
\]

Here `temperature` is \(\lambda\), `control_weight` is the scalar \(R\), and
the configured standard deviations define diagonal \(\Sigma\). Archived YAML
files that omit `importance_sampling_correction` preserve the earlier weighted
sampling behavior.

The cost also contains terminal linear- and yaw-rate penalties. They prevent a
finite-horizon controller from repeatedly passing through or orbiting the point
goal with non-zero terminal velocity.

## MuJoCo timing correction

The 40 ms command delay is scheduled on MuJoCo's 2 ms physics clock. It is no
longer rounded against the 100 ms MPPI control period. A regression test checks
that exactly 20 of 50 physics substeps retain the old command for a 40 ms delay.

## Recorded metrics

Every episode records proposed, safety-issued, and plant-applied controls plus:

- collision, clearance, path length, goal distance, slip and safety reasons;
- control jerk, stuck steps and spin steps;
- cost, effective sample size and sample saturation;
- mean, p50, p95, p99 and maximum planner time;
- 100 ms deadline misses and termination reason.

## Commands

Interactive run:

```bash
.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/mujoco_strong_mppi_baseline.yaml --viewer
```

Five-seed frozen benchmark:

```bash
.venv/bin/python experiments/run_mujoco_baseline_benchmark.py \
  --output-dir results/research_platform/strong_mppi_baseline_v1 \
  --seeds 0 1 2 3 4
```

The benchmark refuses to mix residual prediction, Memory or RL into this
baseline, refuses to overwrite a non-empty output directory, and writes each
episode's full artifacts plus `runs.csv` and `benchmark_summary.json`.

## Gate smoke result (2026-07-12)

The command above was actually executed for seeds 0--4. This was a framework
acceptance smoke run, not a paper result:

```text
success:                 5 / 5
collision:               0 / 5
final goal distance:     0.2939 m mean
minimum clearance:       0.2005 m mean across episode minima
time to goal:            21.36 s mean, 34.90 s maximum
planner compute:          7.23 ms mean
planner compute p95:     10.15 ms mean across seeds
planner deadline misses: 0
```

Seed 2 is substantially slower than the others. The baseline is therefore safe
and reproducible enough to freeze as Gate 1, but speed/robustness remains an
explicit metric rather than a solved claim.

## Interpretation boundary

This gate validates a reproducible, safety-preserving traditional controller
and a physical simulation loop. It is not evidence that ICODE or RL improves
performance. Those claims require later matched-seed ablations using this exact
baseline contract.
