# Table 24. Temporal bidirectional dynamic Actor development evidence

## Candidate history and independent replication

| Development evaluation | Source success | Candidate success | Source collision | Candidate collision | Interpretation |
|---|---:|---:|---:|---:|---|
| V5 Amendment 2 early screen, seeds 730100157--164 | 6/8 | 7/8 | 0/8 | 0/8 | Initial positive screen: total steps -6.02%, mean final distance -12.35% |
| V5 Amendment 2 fresh check, seeds 730100172--179 | -- | -- | -- | candidate-only collisions on 176, 177 and 179 | Independent safety regression; Amendment 2 not retained |
| V5 Amendment 3 corrected-source check, seeds 730100172--179 | 3/8 | 4/8 | 2/8 | 3/8 | One extra success but seed 179 became a new collision |
| V5 Amendment 4 fresh check, seeds 730100180--187 | 6/8 | 5/8 | 1/8 | 2/8 | Lost one success and introduced a seed-182 collision |
| V5 Amendment 6 update 250 + same-cycle filter, fresh seeds 730100188--195 | 6/8 | 7/8 | 1/8 | 1/8 | Positive batch with no new collision |
| V5 Amendment 6 update 250 + same-cycle filter, replication seeds 730100196--203 | 7/8 | 7/8 | 1/8 | 1/8 | Independent directionally positive replication with no new collision |
| Same frozen checkpoint, expanded seeds 730100204--227 | 18/24 | 20/24 | 2/24 | 3/24 | Completion gain but one candidate-only collision, one lost source success and worse mean final distance; gate failed |

## Initial two-batch pooled screen

| Quantity | V3 source Actor | V5 Amendment 6 update 250 + same-cycle filter | Paired change |
|---|---:|---:|---:|
| Previously unused development seeds | 16 | 16 | same 730100188--203 |
| Goal success | 13/16 | 14/16 | +1 episode |
| Collision | 2/16 | 2/16 | 0; no new candidate collision |
| Lost source successes | -- | 0 | none |
| Rescued source failures | -- | 1 | seed 730100194 |
| Total steps | 4994 | 4873 | -121 (-2.42%) |
| Mean final goal distance | 0.75915 m | 0.68454 m | -9.83% |

This initial screen motivated but did not replace expanded qualification.

## Frozen 24-pair expanded result

| Quantity | V3 source Actor | V5 Amendment 6 update 250 | Paired change |
|---|---:|---:|---:|
| Previously unused development seeds | 24 | 24 | same 730100204--227 |
| Goal success | 18/24 | 20/24 | +2 episodes |
| Collision | 2/24 | 3/24 | +1 candidate-only collision |
| Lost source successes | -- | 1 | seed 730100206 |
| Rescued source failures | -- | 3 | seeds 730100218, 730100220, 730100221 |
| Total steps | 7845 | 7611 | -234 (-2.98%) |
| Mean final goal distance | 0.77807 m | 0.81760 m | +5.08% (worse) |
| Step-better pairs | -- | 16/24 | median delta -6.5 steps |
| Distance-better pairs | -- | 13/24 | median delta -0.00164 m |
| Maximum planner P95 | 95.71 ms | 100.36 ms | candidate exceeds 100 ms by 0.36 ms |

Both arms used proposal-only integration, the same same-cycle guided-cost filter, the unchanged 600-rollout budget and the same safety stack; the intended treatment difference was the Actor checkpoint. The 84-dimensional Actor observation preserves the frozen 48-dimensional prefix and adds 36 causal lidar-sector deltas. The old Critic remained disabled. Execution order was balanced within each eight-pair batch. The filter was enabled for every arm and exercised for 9,437 source versus 7,600 candidate iterations, so the seed-730100208 candidate-only collision demonstrates that current-cycle best-cost filtering is not by itself a sufficient trajectory-level safety guarantee.

The candidate checkpoint is `research_artifacts/dynamic_actor_temporal_bidirectional_correction_v5_amendment6/checkpoints/improved_update_000250.pt`, SHA-256 `714694f8d179f02b90c14e56c9e3e4f29a0101b70ad53f3cc795eb72dea7a1b5`. The expanded gate failed `zero_new_candidate_collisions`, `zero_lost_source_successes` and `mean_final_distance_noninferior`; it passed completion, total-step, protocol-integrity and filter-mechanism checks. The checkpoint is retained only as diagnostic evidence and is not qualified for sealed evaluation.

Primary evidence: `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/{summary.json,gate.json,paired_analysis.json,integrity_audit.json}` and the 24 paired result/run directories. Earlier screen evidence remains in `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_fresh_replication_summary.json`. Negative routes are retained in the V5 Amendment 2--4 artifacts. The similarly named V5 Amendment 3 artifacts without `fresh_corrected` accidentally used the old L217 source because `--source` was omitted and are not treated as V3 comparisons.
