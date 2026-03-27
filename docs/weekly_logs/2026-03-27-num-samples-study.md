## Experiment A
- num_samples = 100
- reached goal: near goal region
- reached step: executed trajectory length = 62
- final executed cost: 65.211
- final collision: False
- final state: (5.815656334080914, 1.9369187283264941, 2.672945769444326)
- observation: the controller remained collision-free and reached a state close to the goal

## Experiment B
- num_samples = 250
- reached goal: near goal region
- reached step: executed trajectory length = 62
- final executed cost: 67.511
- final collision: False
- final state: (5.803932722590211, 1.9677625895405662, 2.9609685252017415)
- observation: still collision-free, but final cost was the highest among the three runs

## Experiment C
- num_samples = 400
- reached goal: near goal region
- reached step: executed trajectory length = 62
- final executed cost: 64.648
- final collision: False
- final state: (5.820191405423816, 1.9315722412663652, 2.9516443457713324)
- observation: achieved the lowest final cost in this single-run comparison

## Preliminary conclusion
- In the current 2D setup, all three sample sizes produced collision-free trajectories and reached states close to the goal
- The run with 400 samples gave the lowest final executed cost in this comparison
- However, the difference is not dramatic, and a single run per setting is not enough to claim a stable ranking
- The current baseline appears reasonably robust to moderate changes in num_samples