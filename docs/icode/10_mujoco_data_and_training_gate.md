# MuJoCo Residual Data and ICODE Training Gate

## Scope

This gate verifies that the actuated MuJoCo plant can produce traceable
five-state residual-learning data, that MLP and control-affine ICODE models can
train on episode-disjoint splits, and that the resulting checkpoint can execute
inside MPPI before the 100 ms control deadline.

It is a pipeline smoke test, not a paper experiment. It does not establish OOD
generalization, control improvement, stability, contraction or convergence.

## Transition semantics

For each 100 ms interval the dataset stores:

| Field | Meaning |
| --- | --- |
| `state_t` | MuJoCo ground-truth `[x, y, theta, v, omega]` at interval start |
| `control_t` | safety-issued `[v_cmd, omega_cmd]` available to the prediction model |
| `applied_control_t` | physics-substep average command after actuator delay |
| `state_t_plus_1` | MuJoCo ground truth at interval end |
| `nominal_derivative` | five-state dynamic-unicycle derivative at `(state_t, control_t)` |
| `observed_derivative` | wrapped finite difference `(state_t_plus_1-state_t)/dt` |
| `residual_target` | `observed_derivative-nominal_derivative` |

`control_t` is used as the learning input because it is also what future MPPI
knows. `applied_control_t` is recorded for diagnostics, not leaked into ordinary
planner inference. Delay, wheel contact and actuator states that are not present
in the five-state input therefore remain part of the model mismatch.

Random exploration uses temporally smoothed, held commands and still passes
through LaserScan and `scan_guard`. Task-specific collection uses nominal MPPI
and feeds the safety-issued command back to its next planning iteration.

## Mandatory quality gates

Collection aborts before training if either identity fails:

\[
r_t = \dot{x}_{\mathrm{obs},t}-f_{\mathrm{nom}}(x_t,u_t),
\]

\[
x_{t+1} = x_t + \Delta t\,\dot{x}_{\mathrm{obs},t},
\]

with wrapped heading error. It also rejects episode overlap between train,
validation, test and unseen splits.

## Control-affine ICODE implementation

The online residual is

\[
f_{\mathrm{res}}(x,u)=f_\theta(x)+G_\theta(x)u.
\]

State input uses sine/cosine heading encoding. Normalization does not destroy
control affinity: the implementation explicitly exposes physical-unit drift,
physical gain matrix and physical control contribution, and a regression test
checks affinity after non-zero control mean/scale normalization.

The real-time default is two 64-unit Softplus hidden layers. A 256-unit network
was measured above the CPU deadline and is therefore an ablation, not the
default online model.

## Executed smoke test (2026-07-12)

Data:

```text
episodes:                         8
transitions:                    240
train / validation / test:     6 / 1 / 1 episodes
residual identity max error:     0
transition closure max error:    3.32e-17
command/applied mismatch rows: 100%
command/applied RMSE:             0.03094
```

Five-epoch test-split prediction results:

| Method | Parameters | Residual RMSE | H=1 rollout | H=5 | H=10 | H=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal | 0 | 0.31389 | 0.02416 | 0.02827 | 0.02932 | 0.03581 |
| MLP residual | 5,061 | 0.29086 | 0.02288 | 0.02554 | 0.02582 | 0.02989 |
| ICODE residual | 10,191 | 0.26014 | 0.02126 | 0.02182 | 0.02160 | 0.02247 |

For ICODE, H=20 endpoint position RMSE changed from 0.02739 m to
0.01360 m and endpoint heading RMSE from 0.06555 rad to 0.03236 rad.

A 30-step ICODE-MPPI integration smoke with 64 samples measured:

```text
planner compute mean:        45.49 ms
planner compute p95:         60.76 ms
planner compute maximum:     83.94 ms
100 ms deadline misses:          0
collision:                    false
```

The run was intentionally too short to assess goal success. It proves checkpoint
loading, combined dynamics rollout and CPU timing only.

## Reproduction commands

```bash
.venv/bin/python experiments/collect_mujoco_residual_data.py \
  --config configs/research/mujoco_clean_dynamics.yaml \
  --output-dir results/research_platform/datasets/icode_smoke \
  --episodes 8 --steps 30 --source mixed

.venv/bin/python experiments/train_platform_residual.py \
  --config configs/research/icode_dynamic5.yaml \
  --dataset-dir results/research_platform/datasets/icode_smoke \
  --output-dir results/research_platform/checkpoints/icode_smoke --epochs 5

.venv/bin/python experiments/evaluate_platform_residual.py \
  --checkpoint results/research_platform/checkpoints/icode_smoke/best.pt \
  --dataset-dir results/research_platform/datasets/icode_smoke \
  --output results/research_platform/model_metrics/icode_smoke.json \
  --horizons 1,5,10,20
```

## Remaining gates before a scientific claim

- collect substantially more trajectories across matched training seeds;
- add explicit held-out friction, mass, delay and actuator configurations;
- report mean and confidence intervals across dataset/training seeds;
- compare nominal, MLP and ICODE under identical parameter/data budgets;
- demonstrate prediction gains translate to matched-seed MPPI control gains;
- retain failures and deadline misses rather than selecting successful runs.

The current code reproduces the public control-affine residual structure used
by ICODE-MPPI. It does not claim all theoretical guarantees of an original ICODE
formulation.
