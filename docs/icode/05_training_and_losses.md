# Training, RK4, and Losses

## Combined integration

Training never integrates a residual alone.  Euler or RK4 integrates
`f_nom + f_res` under zero-order-held control.  RK4 evaluates at
`t, t+dt/2, t+dt/2, t+dt`, then wraps heading after the complete step.

## Loss terms

The configurable objective is

```text
L = lambda_res L_res + lambda_1 L_one
    + lambda_H L_rollout + lambda_reg L_reg
```

- `L_res`: residual-derivative MSE against the finite-difference label.
- `L_one`: one-step combined state-prediction MSE.
- `L_rollout`: weighted H-step autoregressive rollout MSE.
- `L_reg`: mean squared parameter regularizer.

All state errors wrap theta.  Horizon/state/residual weights are configurable.

ICODE-MPPI Eq. (12) explicitly displays the one-step RK4 state loss.  Although
its prose says multi-step prediction, the shown equation is `t -> t+1`.
`L_rollout` is therefore documented as a necessary research extension for the
MPPI horizon, not as a verbatim reproduction of Eq. (12).

## Trainer

`ResidualTrainer` provides Adam, plateau scheduling, gradient clipping,
deterministic seeds, early stopping, CSV epoch logs, best/last checkpoints, and
resume.  Model selection uses validation multi-step rollout RMSE, falling back
to total validation loss only if no valid rollout window exists.

Checkpoints contain model/optimizer/scheduler states, normalizers, resolved
config, epoch, best metric, Git SHA, model class/config, dimensions, and RNG
state.  Dynamic imports are disabled by default during model restoration;
runtime adapters use an explicit MLP/ICODE registry.

## Commands

```bash
.venv/bin/python experiments/icode/train_residual_model.py \
  --config configs/icode/mlp_residual.yaml --smoke

.venv/bin/python experiments/icode/train_residual_model.py \
  --config configs/icode/icode_residual.yaml --smoke

.venv/bin/python experiments/icode/inspect_residual_checkpoint.py CHECKPOINT
```

Smoke training only validates the pipeline.  It is not evidence that a model
outperforms nominal dynamics.
