# L41 H=36 ICODE residual-only closed-loop eligibility — preregistration

## Question

Do the three L40 ICODE checkpoints, which passed held-out H=36 prediction screening, improve traditional MPPI closed-loop behavior without RL, memory or safety changes?

## Frozen design

- Conditions: traditional nominal MPPI and traditional ICODE MPPI.
- Model blocks: three independent L40 initialization seeds.
- Scenes: L36 calibrated easy, moderate and hard dynamic scenes.
- Physics: matched 40 ms and long 100 ms command-delay strata with all other parameters identical.
- Seeds: five new development seeds (`20860731`–`20860735`).
- Shared controls: 100 MPPI samples, 36-step horizon, GoalWarmStart, robust temporal scan safety and scan_guard.
- Experimental unit: model block × scene × physics × episode seed; conditions are paired repeated measurements.
- Total: 180 episodes in a seeded shuffled schedule.

## Frozen eligibility gate

All conditions must hold:

1. 180 complete finite episodes, no protected or sealed seed use;
2. positive net success in at least two of three model blocks;
3. collision noninferiority in all three blocks;
4. pooled net success gain at least three episodes;
5. pooled net collision increase no greater than zero;
6. mean final-goal-distance improvement at least 0.10 m;
7. ICODE mean planner time no greater than 50 ms.

Paired effects are summarized at the experimental-unit level. Hierarchical bootstrap intervals resample model blocks and then units within block. Passing L41 permits an independent residual-only confirmation; it does not yet support RL×ICODE claims.

Seeds `20860736`–`20860745` remain sealed until every gate item passes.
