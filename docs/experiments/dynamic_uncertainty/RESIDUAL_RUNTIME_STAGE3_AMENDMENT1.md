# Residual Runtime Stage 3 Amendment 1

Date: 2026-07-23  
Status: preregistered before implementation and measurement.

## Trigger

The frozen Stage 3A screen found CUDA numerically equivalent to the CPU
baseline and approximately 45.8% faster in worst-block P95, but the CUDA
full-horizon residual rollout still required `163.58 ms`. It therefore failed
the `100 ms` deployment target before shield and cost evaluation were added.

The measured implementation performs 144 residual evaluations per H36 RK4
rollout. Each evaluation currently converts NumPy inputs to Torch and copies
the prediction back to CPU. This forces repeated device synchronization.

## Single authorized change

Implement an opt-in device-resident combined rollout:

- transfer the initial state and `[600,36,2]` controls to Torch once;
- retain integration state on the selected Torch device for all 36 RK4 steps;
- evaluate the unchanged TorchScript checkpoint 144 times;
- retain float64 nominal kinematics and RK4 accumulation;
- cast only neural-network inputs to float32, matching the existing adapter;
- apply the same `[0,0,0,1,1]` residual mask;
- wrap periodic state channels after every integration step;
- copy the completed trajectory back to NumPy once.

The default remains the legacy path. Activation requires
`planner.residual_device_rollout_enabled: true`.

## Frozen comparison

CUDA legacy-transfer and CUDA device-resident arms receive identical control
batches in all three checkpoint blocks. Execution order is randomized with
seed `730199909`; every cell has five warm-ups and thirty measured rollouts.

Amendment 1 passes only if:

- all trajectories are finite;
- maximum absolute trajectory difference is at most `1e-5`;
- worst-block P95 improves by at least 30%;
- selected worst-block P95 is no more than `100 ms`.

Passing this screen authorizes a complete shield-level development replay. It
does not itself establish closed-loop equivalence or real-time deployment.
