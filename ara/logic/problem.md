# Problem

The project studies whether learned residual dynamics and a policy-guided sampling prior can improve MPPI under model mismatch and difficult geometry without bypassing LaserScan-based perception, safety arbitration or the existing MuJoCo/robot control chain.

The current subproblem is narrower: determine whether SAC can safely improve a frozen behavior-cloned MPPI sampling prior. A valid method must preserve paired successes and collision safety before any OOD, ICODE, memory or dynamic-obstacle extensions are introduced.

