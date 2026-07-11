# Incremental Learning Roadmap

The code is complete enough to learn in stages; do not try to absorb all
theory at once.

## Lesson 1 - nominal, plant, mismatch, residual

Read in order:

1. `src/dynamics/interfaces.py`
2. `src/dynamics/nominal_unicycle.py`
3. `src/dynamics/disturbed_unicycle.py`
4. `src/dynamics/residual/oracle_residual.py`
5. `src/dynamics/integrators.py`

Goal: explain why planner model and plant are separate, and why
`f_nom + f_res_oracle = f_true`.

## Lesson 2 - transitions and periodic state

Read `state_encoding.py`, `residual_dataset.py`, `normalization.py`, and the
collector.  Goal: derive the wrapped finite-difference label and explain why
splits are by episode.

## Lesson 3 - MLP versus ICODE

Read `mlp_residual.py` and `icode_residual.py`.  Goal: identify drift, gain,
and control contribution shapes and understand the control-affine prior.

## Lesson 4 - Neural ODE integration and loss

Read `residual_losses.py`.  Goal: manually trace one RK4 step and then a
multi-step autoregressive rollout with wrapped heading error.

## Lesson 5 - trainer and reproducibility

Read `checkpointing.py`, `residual_trainer.py`, and the training CLI.  Goal:
explain train-only normalizers, validation rollout model selection, resume,
and why a bare `state_dict` is insufficient.

## Lesson 6 - MPPI integration

Read `mppi_dynamics_adapter.py`, the small diff in the existing planner, and
the regression test.  Goal: distinguish candidate prediction, true execution,
oracle leakage, proposed control, and final safety-arbitrated control.

## Lesson 7 - experiments and interpretation

Run the smoke chain, inspect its manifests, then design a full run without
changing test/unseen data.  Goal: separate wiring evidence from a scientific
performance claim.

## Lesson 8 - future RL prior

Read `sampling_prior.py`.  Design an `RLPolicyPrior` that supplies only a mean
control sequence.  Do not let it bypass MPPI cost, dynamics, or safety.  RL
training is intentionally outside the current implementation.
