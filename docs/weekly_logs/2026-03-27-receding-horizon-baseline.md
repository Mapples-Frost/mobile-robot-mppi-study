# 2026-03-27 Receding Horizon Baseline

## What I completed
- Added `mppi_receding_horizon_demo.py`
- Ran the minimal receding horizon MPPI baseline successfully in PyCharm
- Verified that the robot can move toward the goal and avoid obstacles in the current 2D setup

## Key observations
- The controller replans at every step
- Only the first control of the updated sequence is executed
- The remaining sequence is shifted forward and reused as the next nominal sequence
- The executed trajectory reached the goal tolerance region

## Run result summary
- Reached goal tolerance around step 49
- Final collision: False
- Final executed cost: 64.093
- Final state: (5.241268828145067, 1.7643281077990703, 0.046977815424427026)

## Current understanding
- I now understand the basic MPPI loop:
  sampling -> rollout -> cost -> weights -> weighted update -> execute first control -> shift -> replan

## Next step
- Record this baseline to Git
- Run a first small parameter study on horizon / num_samples / noise scale