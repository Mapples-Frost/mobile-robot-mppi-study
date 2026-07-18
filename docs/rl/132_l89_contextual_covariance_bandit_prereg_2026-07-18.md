# L89 contextual covariance bandit preregistration

Date: 2026-07-18

## Motivation and method change

Continuous per-step SAC covariance residuals either violated precision (L85)
or collapsed to the global anchor and produced no time advantage (L86).  L87
shows that the useful decision is coarse and geometry-dependent.  L89
therefore moves RL one level upward: a contextual bandit chooses one audited
MPPI exploration option, while ICODE remains the prediction model and MPPI
still computes the control sequence.

The two actions are:

- `narrow = [0.5, 0.5]`;
- `speed = [1.75, 0.75]`.

The state contains only rigid-transform-invariant features of the supplied
polyline reference: absolute turn, route indirectness and peak curvature.
Scene identifiers, MuJoCo truth, physics-domain labels and future outcomes are
forbidden inputs.  LinUCB supplies an explicit exploration bonus for later
OOD/online study; deployment in this experiment is greedy.

## Separation of data and evaluation

Bandit fitting uses only selection-seed episodes from four support geometries:
straight, sweep, accel-turn and chicane.  Ridge strength is selected only on
new seeds from those same support geometries.  Hairpin and reverse-S provide
no fitting transition.  The final confirmation reruns both held-out
geometries using five new seeds in each of four physics domains.

The logged scalar training reward is

\[
r=-\frac{T + 1500 e_{\mathrm{rmse}} + 500 I_{\mathrm{fail}}
 + 500 I_{\mathrm{collision}}}{100}.
\]

This reward trains the critic only.  It is not the success criterion.

## Primary comparator and gate

The comparator is the strongest L87 global fixed covariance,
`speed = [1.75, 0.75]`.  The primary gate requires all of:

1. no loss of success and no increase in collision rate;
2. the upper hierarchical-bootstrap 95% confidence limit of paired
   cross-track RMSE change is at most +2 mm;
3. the upper 95% confidence limit of paired time-to-goal change is below zero.

The hierarchy resamples geometry/physics context first and seed second.  A
positive scalar reward or correct action on the already inspected L87 rows is
insufficient; only the independent closed-loop confirmation may pass L89.

## Claims allowed if passed

L89 may support the claim that a reward-trained, route-observable high-level
policy can generalize its MPPI exploration choice to held-out path geometries
and outperform the strongest fixed covariance without sacrificing safety or
tracking precision.  It does not establish general dynamic-obstacle or real-
robot performance, and it does not imply a theoretical optimality guarantee.

