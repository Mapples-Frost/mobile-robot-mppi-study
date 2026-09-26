# L64 residual-structure cross-plant development preregistration

Date: 2026-07-16

## Question

Do parameter-matched MLP and control-affine ICODE residual predictors trained
on one fixed MuJoCo plant improve closed-loop MPPI after mass, contact,
actuator, and delay parameters change? Does ICODE retain an advantage over the
unstructured MLP across those frozen shifts?

## Separation from calibration

L63 selected physical domains using only traditional nominal MPPI. L64 freezes
the passing L63-v3 selection before loading any learned-model outcome:
`train_anchor`, `mass_light`, `friction_high`, `torque_weak_v2`, `delay_long`,
and `combined_moderate_b`.

The planner is told the command delay for every condition. It is not told the
domain name, mass, friction, or actuator parameters. Both learned predictors
receive only the state/control representation supported by their checkpoints.

## Factorial design

- Models: nominal, parameter-matched MLP, parameter-matched ICODE.
- Independent training blocks: three matched MLP/ICODE seed pairs.
- Paths: training-family chicane and held-out reverse-S.
- Plants: anchor plus five frozen shifts.
- Development execution seeds: three.
- Total: 324 episodes.
- Memory and RL are disabled.

## Decision rule

Both learned models must retain positive shifted-plant improvement over nominal
in all three model blocks with a hierarchical 95% interval above zero. For the
primary ICODE-versus-MLP contrast, improvement must be positive in at least two
model blocks, at least four of five shifted domains, at least two unseen-path
blocks, and at least two combined-domain blocks. Shifted, unseen-shifted, and
combined-domain hierarchical 95% interval lower bounds must exceed zero.
Inherited safety, completion, and compute-time gates remain active.

Failure stops the sealed confirmation. Passing unlocks only a separately
registered confirmation using untouched seeds 21860831--21860835.

## Claim boundary

This experiment concerns residual-model transfer across five bounded MuJoCo
parameter changes. It does not establish arbitrary OOD robustness, mechanism,
real-robot transfer, or any RL claim.
