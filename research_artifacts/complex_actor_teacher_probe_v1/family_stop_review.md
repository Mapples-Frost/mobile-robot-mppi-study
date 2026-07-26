# Complex Actor teacher family stop review

## Question

Can the existing Actor architecture be retrained for all three complex maps
using a causal, higher-sample version of the same risk-aware MPPI stack as a
teacher, without changing the deployed algorithm?

## Frozen evidence

| Map | Teacher samples | Safety | Motion result |
|---|---:|---|---|
| Chapter 1, seed 790201009 | 1,200 | no collision | path progress `0.467 m`, fail |
| Chapter 2, seed 790202007 | 1,200 | no collision | path progress `4.645 m`, pass |
| Chapter 3, seed 790203007 | 1,200 | no collision | net displacement `4.989 m`, pass |
| Chapter 1, seed 790201011 | 2,400 | collision | path progress `0.404 m`, fail |

The 60-step preflight failure is separately retained and is not counted as an
episode outcome.

## Cross-layer conclusion

The original learned Actor is genuinely unusable at the Chapter-1 entrance:
its initial candidates are static-feasible but all violate the dynamic-risk
threshold. However, simply replacing it with a higher-sample causal teacher is
not a valid repair. The same teacher works on Chapters 2 and 3, while Chapter 1
remains trapped by a fast, near-start dynamic conflict. At 2,400 samples the
planner correctly reports that stopping and the selected trajectory are both
hard-risk, but the existing active-escape transaction still collides.

This is no longer an Actor-only problem. It is a coupled capability boundary
between the 3.6-second probabilistic horizon, active-escape candidate
generation, and final Safety execution at the Chapter-1 entrance.

## Scope decision

The following actions are ruled out:

- further sample-count escalation;
- weakening collision-probability thresholds;
- forcing HSS/Actor authority over infeasible candidates;
- training the Actor only on the two successful maps;
- changing or removing the failed seed;
- adding a new high-level planner or map-specific state machine;
- silently moving the Chapter-1 dynamic obstacle.

Fixing Chapter 1 now would require either a new active-escape mechanism,
longer-horizon prediction, or a scene-design amendment. Each changes the
scientific scope and requires a new preregistered mechanism family rather than
another unattended local tweak.

The responsible stopping point is therefore:

1. preserve Chapters 2 and 3 as demonstrated upper-bound successes;
2. retain Chapter 1 as a documented failure boundary;
3. do not train a replacement Actor or start baseline/ablation experiments;
4. wait for an explicit author decision on whether to narrow the complex-map
   claim, amend Chapter 1, or authorize a new active-escape mechanism.
