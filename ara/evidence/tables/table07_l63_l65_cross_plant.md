# Table 07: L63--L65 cross-plant residual-structure evidence

| Evidence unit | Result |
|---|---|
| L63 selection method | Traditional nominal MPPI only; MLP/ICODE outcomes unseen |
| Frozen plants | Anchor + mass, friction, actuator, delay and combined shifts |
| L64 development | 324/324 episodes; shifted ICODE--MLP +0.004754 m; 95% lower bound 0.004000 m |
| L65 sealed confirmation | 540/540 episodes; 180 paired comparisons per contrast |
| ICODE vs nominal | 26.88% lower cross-track RMSE; +0.017636 m |
| MLP vs nominal | 19.17% lower cross-track RMSE; +0.012579 m |
| ICODE vs MLP, shifted plants | 9.23% lower cross-track RMSE; +0.004956 m; 95% CI [0.003993, 0.005885] m |
| ICODE vs MLP, unseen shifted | +0.005872 m; 95% lower bound 0.004794 m |
| ICODE vs MLP, combined plant | +0.004889 m; 95% lower bound 0.003399 m |
| Per-domain direction | 6/6 positive, including anchor |
| Safety/task | Every method 180/180 successful; zero collisions |
| Mean planner time | Nominal 4.90 ms; MLP 29.45 ms; ICODE 34.78 ms |

Sources: `results/research_platform/rl/l65_residual_structure_cross_plant_confirmation_20260716_v1/residual_structure_cross_plant_summary.json`
and `docs/rl/95_l63_l65_cross_plant_residual_structure_results_2026-07-16.md`.

Interpretation boundary: this is bounded MuJoCo parameter-shift evidence with
clean state feedback and RL disabled, not arbitrary OOD, obstacle-rich,
real-robot, theoretical or RL-composition evidence.
