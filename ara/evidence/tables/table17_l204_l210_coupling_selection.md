# Table 17: L204-L210 coupling-mechanism selection

| Candidate | Independent scope | Main outcome | Decision |
|---|---|---|---|
| Full 61D residual-conditioned Actor (L204) | 60k SAC steps | Step 0 remained best; later validation degraded | Reject catastrophic-forgetting route |
| Frozen-base bounded correction (L205) | 40k SAC steps | 100% success, zero collision; direct-Actor RMSE improved only about 0.2% | Advance only to closed-loop screen |
| Mean-only correction (L207) | 3 paired seeds x 2 domains | nominal RMSE +16.15%; unseen RMSE -5.56% | Reject always-on correction |
| Innovation-gated correction (L209) | 3 fresh paired seeds x 2 domains | nominal RMSE +10.61%; unseen RMSE -1.22%; unseen jerk +5.69% | Reject paper mechanism |
| Pairwise value-ranked ICODE (L210) | episode-bootstrap member; route-disjoint validation/test/unseen | test/unseen rank change -0.0100/-0.0019; test rollout +7.88% | Reject before ensemble/closed-loop |
| Competence-gated value-aligned ICODE | sealed 2x2 factorial, 5 seed clusters x 3 physics domains | success +0.3333; final distance +0.17102 m, both CIs strictly favorable | Freeze as RL -> ICODE mechanism |
| Reliability-adaptive HSS | same sealed factorial, fixed K=100 | success +0.2667; final distance +0.15997 m; planner time -2.83 ms | Freeze as ICODE -> RL/MPPI mechanism |

Positive effects use favorable direction. The retained factors have reproducible
main effects but an adverse/sub-additive success interaction; no super-additive
synergy claim is supported.
