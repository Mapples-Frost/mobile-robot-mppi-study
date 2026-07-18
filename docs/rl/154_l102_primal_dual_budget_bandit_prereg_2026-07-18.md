# L102 constrained contextual budget bandit: preregistration

Date: 2026-07-18

Status: frozen before L102 branch collection

## Motivation

L101 demonstrated that first-batch diagnostics predict selective K100 benefit
under an explicit mean-compute constraint. L102 asks the narrower RL question:

> Can a contextual bandit learn STOP/ADD50 from rewards observed only on its
> selected ADD actions while satisfying a long-run rollout-budget constraint?

The action is binary and does not command the robot. ICODE remains the rollout
dynamics, MPPI still computes control, and safety arbitration remains downstream.

## Algorithm

The ADD50 advantage model is ridge-linear in the frozen L100 feature schema.
During discovery the policy selects actions with a linear confidence bonus and
20% epsilon exploration. Only selected ADD50 actions update the advantage
model. A primal-dual compute price is updated after every action:

```text
lambda <- clip(lambda + 0.005 * (ADD - 0.50), 0, 0.25)
ADD when predicted_advantage + 0.1 * confidence - lambda > 0
```

Deployment on evaluation episodes is deterministic: exploration bonuses and
epsilon random actions are disabled, while the learned model and final dual
price are frozen. Hyperparameters were chosen during exploratory analysis of
L101 and are not changed after opening the disjoint L102 seeds.

## Data separation

- discovery: `20271401--20271403`;
- evaluation: `20271411--20271413`;
- two routes x four physics domains x six seeds = 48 episodes;
- eight nested K50/K100 anchors per episode = 384 records.

No L99, L100 or L101 evaluation episode is used to score L102. The common-
random-number, snapshot, no-obstacle and episode-level bootstrap contracts are
identical to L100/L101.

## Primary Gate

All clauses must pass on L102 evaluation episodes:

1. ADD50 fraction in `[0.10, 0.75]`;
2. paired true branch cost versus fixed K50 has 95% CI upper `< 0`;
3. mean sample budget no greater than 80;
4. at least 30% of the hindsight Oracle relative gain is retained.

Fixed K100, matched random budget and Oracle are reported. Bandit superiority
over matched random is secondary in L102 because one random realization is not
a stable scientific baseline; the later closed-loop factorial will use multiple
matched random seeds.

## Claim boundary

A pass supports a constrained contextual-bandit decision at sampled physical
states. It does not yet prove whole-episode improvement, temporal credit
assignment, dynamic-obstacle benefit or ICODE-specific interaction. Those are
tested only after an incremental online MPPI implementation passes regression.
