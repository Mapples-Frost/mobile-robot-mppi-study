# Table 08: L66--L68 observation-domain residual-structure evidence

| Evidence unit | Result |
|---|---|
| L66 selection method | Traditional nominal MPPI only; MLP/ICODE outcomes unseen |
| Primary domains | Clean ground-truth state, 100 ms latency, moderate noise + 100 ms latency |
| Pure-noise decision | Maximum nominal effect 1.61%, below 3% resolution; excluded as standalone primary factor |
| L67 development | 216/216 episodes; primary ICODE--MLP +0.005226 m (9.50%); 95% lower bound 0.004038 m |
| L68 sealed confirmation | 360/360 artifacts; 90 paired primary comparisons per contrast |
| ICODE vs nominal, primary | 30.17% lower cross-track RMSE; +0.020896 m; 95% CI [0.019502, 0.022256] m |
| MLP vs nominal, primary | 22.25% lower cross-track RMSE; +0.015408 m; 95% CI [0.014247, 0.016587] m |
| ICODE vs MLP, primary | 10.15% lower cross-track RMSE; +0.005487 m; 95% CI [0.004243, 0.006799] m |
| Per-domain ICODE--MLP | Clean +10.91%; latency +9.44%; combined +10.10% |
| Primary safety/task | Every method 90/90 successful; zero collisions |
| Raw wheel-odometry stress | Development successes: nominal 3/18, MLP 5/18, ICODE 10/18; sealed: nominal 18/30, MLP 13/30, ICODE 12/30 |
| Mean planner time, all domains | Nominal 5.20 ms; MLP 31.25 ms; ICODE 36.04 ms |

Sources: `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/residual_structure_observation_summary.json`
and `docs/rl/99_l66_l68_observation_robustness_results_2026-07-16.md`.

Interpretation boundary: this supports bounded latency/noise robustness with
ground-truth-derived state feedback, not raw-odometry correction, arbitrary OOD,
obstacle-rich, real-robot, theoretical or RL-composition claims.
