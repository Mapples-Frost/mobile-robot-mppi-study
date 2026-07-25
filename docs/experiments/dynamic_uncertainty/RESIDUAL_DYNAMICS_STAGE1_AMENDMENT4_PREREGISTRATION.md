# Residual-Dynamics Stage 1 Amendment 4 Preregistration

Status: **frozen before Amendment 4 closed-loop outcomes**  
Date: 2026-07-23  
Scope: structure-preserving residual dynamics  
RL status: **disabled**

## Intervention

Amendment 4 adds one structural mask to Amendment 3:

```text
[x_dot, y_dot, theta_dot, v_dot, omega_dot]
mask = [0, 0, 0, 1, 1]
```

The known dynamic-unicycle pose kinematics remain exact. Learned residuals may
correct only acceleration and yaw-acceleration dynamics. This is not selected
from the Amendment 3 collision: it is the frozen design used by the earlier L44
structure-preserving protocol and subsequent delay-aware studies.

The causal innovation reliability gate and downstream stall latch remain
unchanged. The environment, predictor, probability model, MPPI cost, safety
controller, checkpoints and nominal model remain frozen.

## Development screen

- observed development seed `730100003`;
- three checkpoint blocks plus one shared nominal;
- randomized schedule seed `730199993`;
- collision stops the probe;
- no sealed seed is authorized.

The same safety, completion, clearance, raw H36 prediction, reliability-use,
compute and artifact gates from Amendment 3 apply. Passing is a development
result only.

