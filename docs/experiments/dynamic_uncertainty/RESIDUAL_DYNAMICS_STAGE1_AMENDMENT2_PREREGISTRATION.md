# Residual-Dynamics Stage 1 Amendment 2 Preregistration

Status: **frozen before Amendment 2 closed-loop outcomes**  
Date: 2026-07-23  
Scope: causal residual-authority safety repair  
RL status: **disabled**

## Failure mechanism

In raw Stage 1 seed `730100003`, all three residual checkpoint blocks entered a
low-risk zero-motion state after the second obstacle interaction. A causal
replay of the saved online signals identifies this state at steps 170--173,
which is 53--56 steps before collision. The nominal controller instead invokes
its existing recovery behavior and resumes progress.

## Single intervention

Wrap the residual in a latched causal stall guard. Residual authority changes
from one to zero for the remainder of an episode after all conditions hold for
10 consecutive control steps:

1. measured `abs(v) <= 0.02 m/s`;
2. the previous completed plan's maximum collision probability is `<= 0.05`;
3. target distance is greater than `0.45 m`.

After latching, MPPI uses the unchanged nominal dynamic-unicycle prediction.
The obstacle predictor, probability calculation, MPPI costs, safety arbiter,
environment, checkpoint weights and nominal dynamics remain unchanged.

The constants are fixed from existing contracts rather than fitted to the
collision:

- `0.02 m/s`: repository `stuck_steps` definition;
- `0.05`: frozen Amendment 17 minimum meaningful risk-improvement scale and
  one quarter of the `0.20` hard probability threshold;
- `0.45 m`: frozen recovery goal-release distance;
- 10 steps: one physical second at the frozen `dt=0.1 s`.

The latch is causal. It sees measured state and the previous plan's diagnostics,
not simulator truth, future obstacle motion or the current solve's outcome.

## Design

- Shared nominal reference: seed `730100003`.
- Treatment: three frozen L57 checkpoint blocks with the identical stall guard.
- Randomized run order with seed `730199991`.
- Independent unit: complete episode.
- Model-training seed is the replication block.
- Time steps are repeated measurements, not independent replicates.
- No sealed seed is authorized.

## Outcomes and gate

Primary safety gate:

- zero collision in every residual checkpoint block.

Secondary gates:

- the guard must latch causally in every residual block;
- median completion delta at least `-0.02`;
- median clearance delta at least `-0.02 m`;
- horizon-36 velocity/yaw-rate prediction remains better than nominal in all
  three blocks before authority gating;
- maximum residual planner p95 at most `150 ms`;
- forecast and artifact contracts pass.

A collision stops the screen immediately. A compute miss is recorded but does
not stop the remaining safety blocks, because wall-clock compute is not injected
as simulated control delay and must be repaired as a separate implementation
factor.

If the safety gate fails, stop residual integration. If safety passes but
compute fails, do not open additional seeds; optimize inference under an
unchanged trajectory contract before repeating this same probe.
