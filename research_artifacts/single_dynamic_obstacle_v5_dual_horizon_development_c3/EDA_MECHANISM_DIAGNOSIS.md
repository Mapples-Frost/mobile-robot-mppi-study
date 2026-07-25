# C3 mechanism diagnosis

## Integrity and gate

- C3 is repeated, outcome-informed development and is not confirmatory evidence.
- All 16 paired jobs completed, stderr is empty, and an independent recomputation matched all 91 manifest file hashes and the canonical bundle hash.
- Freeze commit: `1e91323800b7b37ec10b9520cc040e422e7dc593`.
- Bundle SHA-256: `a277f21b2ca344a56c792939c73fb88904ea19155368156761b13da32d35ead2`.
- All 11 frozen development gates passed, including exactly two challenge conversions, no new paired collision, no lost V4 success, preserved clearance, bounded switching and oscillation, and exactly 600 rollouts per decision.

## Quantitative result

The frozen C3 amendment changed only the urgency reserve from 3 to 5 steps while leaving the authority reserve at 0. Success increased from 4/8 under `V4_full_frozen` to 6/8 under `V5_dual_horizon_full`, with collisions unchanged at 2/8. The two safe-noncompletion conversions were ID seed `740200006` and OOD seed `740300251`.

Across the eight paired blocks, mean final goal distance improved by 0.119417 m and mean minimum clearance improved by 0.025071 m. Mean stuck steps fell from 23.5 to 20.5, mean release-delay maximum fell from 1.325 s to 0.463 s, and mean zero-speed-under-risk steps fell from 19.75 to 18.5. Direction-switch and three-phase-oscillation counts did not increase.

On the boundary seed `740200006`, the candidate finished at 0.291216 m from the goal, crossing the pre-frozen 0.300 m success tolerance without collision or clearance loss. The deadline supervisor was active for 56 cycles and remained capped at 0.35 m/s; the reported required speed above that cap is diagnostic only and did not relax actuator or safety limits.

## Interpretation and next action

C3 supplies the planned two-step robustness margin and resolves the B3/C1/C2 endpoint failure without changing the safety hierarchy, steering policy, predictor, risk threshold, rollout budget, or learned checkpoints. This closes development of the dual-horizon family successfully; no further parameter tuning on these development seeds is permitted.

The candidate must now be evaluated once on a pre-frozen, disjoint, unopened held-out qualification cohort. That qualification remains an engineering gate rather than formal effect estimation. Its registry, arm order, thresholds, analyzer, protocol hashes, and seed-exclusion audit must be sealed before any held-out controller outcomes are generated.
