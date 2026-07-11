# Control-Affine ICODE-Style Residual Architecture

## Equation

The implemented structural prior follows ICODE-MPPI Eq. (10):

```text
f_res(x,u;theta) = f_theta(x) + G_theta(x) u
```

- `f_theta(x)` is the residual drift: state-dependent effects present even at
  zero control, such as drag, bias, or external forcing.
- `G_theta(x)` is an `nx x nu` state-dependent control gain matrix.
- `G_theta(x)u` is the control-dependent residual, such as actuator
  effectiveness that changes with state.

With a batch:

```text
state features          [B, nz]
control                 [B, nu]
drift                   [B, nx]
gain                    [B, nx, nu]
control contribution    [B, nx]
residual                [B, nx]
```

The multiplication is `torch.bmm(gain, u.unsqueeze(-1)).squeeze(-1)`.

## Normalized coordinates

The network receives train-normalized encoded state and control and predicts a
normalized residual.  These are affine transformations, so the model remains
affine in the original control.  The runtime adapter converts debug components
back to physical coordinates using

```text
G_phys     = diag(residual_scale) G_norm diag(1 / control_scale)
drift_phys = residual_mean + residual_scale * drift_norm
             - G_phys control_mean
control_phys = G_phys control
```

and verifies `drift_phys + control_phys == residual_phys`.

## MLP baseline

`MLPResidual` instead learns `MLP([state_features,control])`.  It validates the
labels, normalization, trainer, and whether the control-affine prior helps.  It
is an ablation baseline, not ICODE.

## Configuration and diagnostics

Hidden widths, activation, dimensions, and output initialization scale are
configurable.  Softplus and `[256,256]` are the paper-style defaults; smoke
runs replace them with a small network.  Models support CPU/CUDA, batch and
single samples, finite-value checks, JSON-safe configs, parameter counts, and
separate ICODE drift/gain/control outputs.

## Claim boundary

The original ICODE work gives contraction sufficient conditions involving a
uniformly positive-definite metric and a variational-dynamics inequality.  No
such metric/Jacobian constraint is constructed or verified here.  Therefore
this is a control-affine ICODE-style residual implementation; it has no claimed
contraction, stability, convergence, closed-loop, or safety guarantee.
