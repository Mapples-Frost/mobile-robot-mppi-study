# L102 failure record and L103 calibrated-bandit preregistration

Date: 2026-07-18

## L102 frozen result

L102 completed 384 nested counterfactual anchors from 48 MuJoCo episodes.
All integrity checks passed: no counterfactual collisions, finite diagnostics,
and byte-identical K50 prefixes inside K100.

The frozen contextual bandit improved true branch cost over fixed K50 at a mean
budget of 58.85 rollouts. The episode-level hierarchical-bootstrap 95% interval
for its cost delta was `[-0.2197, -0.0211]`. It nevertheless failed the complete
primary Gate: its 0.969% relative gain retained only 22.29% of the 4.346% Oracle
gain, below the frozen 30% requirement. The L102 result remains a failure.

The diagnostic cause was a deployment ADD fraction of 17.71%, despite a 56.25%
ADD fraction during discovery. A terminal online dual price therefore did not
transport the declared compute envelope to held-out contexts.

## L103 method change

L103 separates two roles that L102 conflated:

1. online primal-dual exploration controls which ADD rewards are observed;
2. after training, a deployment dual price is calibrated from the quantile of
   predicted advantages on discovery contexts only.

The target deployment ADD fraction is frozen at 0.50. No evaluation reward,
Oracle label or L103 context enters this calibration. The reward model is still
learned only from ADD actions selected by the contextual bandit; this is not a
conversion to full-information supervised learning.

All other algorithm hyperparameters remain equal to L102. L103 uses disjoint
seeds `20271501--20271503` for discovery and `20271511--20271513` for evaluation.

## L103 primary Gate

All clauses must pass on the new evaluation episodes:

1. ADD fraction in `[0.10, 0.75]`;
2. true-cost delta versus fixed K50 has 95% CI upper `< 0`;
3. mean rollout budget is no greater than 80;
4. at least 30% of hindsight-Oracle relative gain is retained;
5. all common-random-number and counterfactual-integrity checks pass.

Fixed K50, fixed K100, matched-random budget and Oracle remain mandatory
comparators. A pass is evidence at sampled physical states, not yet a claim of
closed-loop episode improvement or ICODE-specific interaction.
