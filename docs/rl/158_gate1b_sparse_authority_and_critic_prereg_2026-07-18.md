# Gate 1b preregistration: sparse policy authority and critic attribution

Date frozen: 2026-07-18

Gate 1 showed that the new equal-budget elite optimizer is substantially
stronger than standard MPPI, but a 30% SAC proposal allocation is not better
than the same optimizer without SAC. This result is retained. Gate 1b asks the
narrower, falsifiable question suggested by the mechanism logs: can the policy
be treated as a sparse exploration source while the critic supplies useful
long-horizon ranking?

No Gate-1 episode seed is reused. Development seeds are 22410201--22410203.
Confirmation seeds 22410211--22410215 remain sealed.

Frozen arms, all with ICODE, K=100, two equal-budget iterations, memory off and
the unchanged LaserScan/safety chain:

1. hybrid-no-RL: 50% shifted and 50% conventional proposals, no critic;
2. critic-only: identical proposal allocation, expected target twin-Q terminal
   cost with weight 0.02;
3. sparse-policy: 5% SAC, 47.5% shifted, 47.5% conventional, no critic;
4. sparse-full: the sparse policy allocation plus the same critic term.

The static single-obstacle high-dynamic scene and combined-unseen physics are
fixed. Three independently trained RL/ICODE checkpoint pairs are blocks.

Primary Gate for sparse-full versus hybrid-no-RL:

- no collision regression and no success loss;
- paired aggregate task-cost improvement at least 1%;
- positive mean improvement in at least two of three model blocks.

Mechanism attribution is reported regardless of the primary outcome:
critic-only versus hybrid-no-RL isolates terminal value, sparse-policy versus
hybrid-no-RL isolates policy candidates, and sparse-full versus both components
tests whether their combination is complementary. Controller steps are not
independent replicates. Failure prevents opening confirmation seeds.
