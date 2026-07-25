# Single-obstacle v5 global review after A5

## Trigger for review

The review was mandatory after A3-A5 failed the frozen development gate in the same post-conflict recovery family. No A6 duration, threshold, or re-entry-count tuning is authorized.

## Evidence

- A4 showed that an unconditional 8 s forward commitment can convert safe non-completions, but it introduced a new OOD collision by changing encounter timing.
- A5 restored the 0.8 s commitment and added a 0.5 s progress watcher. It reduced collisions from 2/8 to 1/8 and improved mean clearance by 0.065 m, but reached only 5/8 successes.
- A5's progress watcher was exposed for 60-167 steps per candidate episode yet triggered zero times in all eight episodes.
- The two relevant A5 safe timeouts ended 0.319 m and 0.384 m from the goal while still moving forward at 0.236 m/s and 0.239 m/s. They were not terminal standstills.
- The remaining A5 candidate collision on seed 740200191 was inherited from frozen V4; A5 introduced no new paired collision.

## State-machine diagnosis

The old mechanism asks only whether progress over a short window is almost zero. It cannot distinguish:

1. motion that is fast enough to finish;
2. motion that is positive but too slow for the fixed deadline;
3. motion that is unsafe and must yield to the existing escape logic.

Changing a fixed commitment duration trades the second failure against the third. A larger stall threshold would merely encode the same trade-off in another parameter.

## New mechanism class

B1 uses an online deadline-feasibility supervisor. After a real dynamic escape has occurred, it computes:

    required average speed =
        max(0, goal distance - position tolerance) / safe remaining time

It may raise only the forward-speed floor, and only after all of the following are true:

- the existing front guard is clear for five consecutive cycles;
- the existing TTC/closing guard is clear;
- the selected predicted collision probability is below the existing recovery-entry threshold;
- no probabilistic hard violation is active;
- target-bearing error is within the existing 0.2 rad gate;
- the required speed exceeds both 0.2 m/s and the planner proposal by 0.02 m/s.

Dynamic escape and temporal emergency retain higher priority. The mechanism changes no steering command, prediction model, collision threshold, safety margin, obstacle generator, learned checkpoint, HSS logic, rollout budget, horizon, control period, episode length, or success tolerance.

## Prospective B1 gate

The old absolute-zero-candidate-collision check is not reused. It incorrectly required the recovery package to eliminate every inherited V4 collision. B1 prospectively requires:

- at least two eligible challenge conversions;
- zero newly introduced paired collisions;
- aggregate candidate collisions no worse than V4;
- zero lost V4 successes;
- no mean stuck-step increase;
- no more than 10% zero-speed-risk increase;
- mean clearance loss no greater than 0.03 m;
- mean final goal distance reduced by at least 0.10 m;
- no increase in direction switches or three-phase oscillations;
- exactly 600 rollouts at every decision.

This remains an outcome-informed engineering gate. Passing it can authorize only a new, disjoint, unopened held-out qualification.
