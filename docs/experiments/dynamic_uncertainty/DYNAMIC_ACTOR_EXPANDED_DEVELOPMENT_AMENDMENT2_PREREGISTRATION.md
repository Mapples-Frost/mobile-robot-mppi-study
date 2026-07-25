# Dynamic Actor Expanded Development Amendment 2 Preregistration

Date frozen: 2026-07-24 23:39 CST

## Motivation and admissible change

Amendment 1 improved success from 20/24 to 21/24, reduced collisions from 4/24 to 3/24, introduced no new Candidate collision, lost no Source success, improved mean final distance by 8.87%, and kept Candidate maximum planner P95 below 100 ms. Its automatic Gate nevertheless failed because raw episode steps were summed even when collision terminated an episode early. Seed 730100208 illustrates the censoring defect: Source collision ended at 266 steps, whereas Candidate avoided collision and reached the goal at 367 steps, so the successful rescue was counted as a 101-step efficiency regression.

A behavioral attempt to force earlier forward passage under a small absolute probability ceiling was tested only on diagnostic seeds and rejected before this protocol was frozen: it caused collisions on seeds 730100208 and 730100212. Those negative artifacts are preserved. No such risk-ceiling option is enabled here. The retained Pareto repair and controller behavior are unchanged from Amendment 1.

The only inferential amendment is an outcome-aware efficiency estimand. For each arm and episode, efficiency steps are defined before execution as:

    episode steps, if the goal is reached;
    max(episode steps, 400), otherwise.

This assigns every unsuccessful outcome the registered episode horizon instead of rewarding early collision. Pooled efficiency noninferiority permits a 1% relative margin, equivalent to approximately 0.33 seconds per paired episode at the 0.1-second control period. Batch nonregression uses the same outcome-aware steps with no margin. Safety, completion, distance, real-time, rollout-budget and integrity gates are unchanged.

## Frozen design

- Independent unit: complete paired MuJoCo episode seed.
- Fresh development seeds: 730100228--730100251.
- Three batches of eight pairs.
- Four Source-first and four Candidate-first pairs per batch.
- Source checkpoint: V3 roll-in Actor.
- Candidate checkpoint: V5 Amendment 6 update 250.
- Proposal-only Actor integration; old Critic disabled.
- Same-cycle guided-cost filter in both arms.
- Exactly 600 rollouts per controller decision.
- One dynamic obstacle; no obstacle future truth.
- Causal lidar history and probabilistic forecast only.
- No tuning, seed replacement, restart of healthy runs or artifact deletion during the matrix.
- Sealed seeds remain unopened.

## Gate

All checks must pass:

1. 24/24 pairs complete.
2. Zero Candidate-only collisions.
3. Zero lost Source successes.
4. Candidate success is noninferior.
5. Outcome-aware pooled efficiency steps are no more than 1% above Source.
6. Candidate mean final distance is noninferior.
7. At least one strict pooled improvement in success, outcome-aware efficiency or distance.
8. At least two of three batches are nonregressive using strict outcome-aware steps.
9. Same-cycle filtering is enabled and exercised in both arms.
10. All frozen bindings, raw files, seeds, order and resolved settings pass integrity audit.
11. Candidate maximum planner P95 remains below the 100 ms control period; this is reported as a separate real-time qualification check even though the historical runner Gate does not fold it into the Boolean result.

The protocol, schedule, checkpoints, controller code, runner and binding hashes are frozen in configs/research/dynamic_actor_v5a6_samecycle_expanded_development_amendment2.yaml before execution.
