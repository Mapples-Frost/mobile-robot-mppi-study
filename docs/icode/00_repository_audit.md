# ICODE Research Framework - Repository Audit

Audit date: 2026-07-11  
Workspace: `/home/mapples/projects/mobile-robot-mppi-study`

## 1. Audit commands and observed state

The following commands were executed before any tracked file was changed:

```bash
git status --short
git branch --show-current
git diff
git log --oneline -15
```

Observed state:

- Current branch: `codex/icode-step1-model-mismatch`.
- Current `HEAD`: `90039f1cd16c99269b3ff9ed62dc8a28d6d37d3c`.
- Target research branch: `mujoco-memory-bridge-ablation` at the same commit.
- Ahead/behind between the two branches: `0/0`.
- Tracked-file diff: empty.
- Pre-existing untracked file: `experiments/icode_step1_model_mismatch.py`.
- The current branch has no upstream. The locally stored target branch and
  `origin/mujoco-memory-bridge-ablation` both point to `90039f1`; no `fetch`
  was performed during this audit.
- A user stash exists and was not inspected, applied, dropped, or modified.

The worktree was not clean, so the branch was **not switched**. Work continues
on `codex/icode-step1-model-mismatch`, which is an exact content-equivalent
work branch based on the requested research branch. No `reset`, `checkout --`,
`clean`, `add`, `commit`, or `push` operation was used.

Recent history recorded by the audit:

```text
90039f1 feat: complete memory-augmented MPPI implementation
8cb6b11 chore: save MuJoCo memory MPPI progress
26e57f6 Update README for current MPPI study status
1d22984 Add memory-augmented MPPI bridge
ebd5589 Stabilize MPPI hardware bridge avoidance and goal latch
122d1df hardware bridge: add safe ROS adapter utilities and local scan obstacle wiring
08daf06 Add scan guard for front obstacle safety
492a541 Add control adapter for safe E1E2 commands
db5b509 Add experiment frame transform for E1E2 hardware bridge
43c39fb Add experiment frame transform for E1E2 hardware bridge
105e99b Add runtime scenario config loader for E1E2 hardware bridge
885703b Ignore local IDE files results and backup files
7825502 Freeze trusted MuJoCo MPPI planner before E1E2 hardware bridge
2e4cfeb summary
4b633b4 fix the important bug which affcted the path
```

## 2. Protected untracked work

`experiments/icode_step1_model_mismatch.py` is a 569-line Stage-1 teaching
prototype owned by the user. It already demonstrates:

- nominal three-state unicycle dynamics;
- velocity gain, yaw gain, and yaw-bias mismatch;
- exact `f_res = f_true - f_nom` decomposition;
- Euler integration and wrapped heading;
- deterministic excitation, sanity checks, and a comparison figure.

It intentionally has no MPPI, MuJoCo, ROS, PyTorch, dataset, RK4, checkpoint,
or experiment-config dependency. It will remain untouched as a legacy teaching
artifact. The research implementation is placed in new package directories, so
there is no path collision.

## 3. Existing research mainline

The repository is no longer an early 2-D MPPI demo. Its protected system path
is:

```text
MuJoCo / Real Robot
    -> LaserScan / Synthetic LaserScan
    -> scan_guard
    -> local_obstacle_layer
    -> planner obstacles
    -> MPPI rollout and trajectory cost
    -> proposed control
    -> safety arbitration
    -> final control
    -> MuJoCo qvel / /cmd_vel
```

Primary assets inspected:

- `src/planners/mppi_mujoco_receding_horizon_experiment.py`
  - authoritative MPPI helper functions: initialization, sampling, rollout,
    cost, weighting, update, and sequence shift;
  - current nominal rollout is three-state unicycle Euler integration with
    heading wrap on every step.
- `experiments/mujoco_memory_mppi_live_sim.py`
  - the current MuJoCo live mainline;
  - fixed `GOAL = (3.0, 3.0)` and four protected scenes;
  - synthetic LaserScan, scan guard, local obstacle conversion, MPPI, Memory,
    execution arbitration, and viewer diagnostics.
- `experiments/mujoco_memory_mppi_ablation.py`
  - memory on/off, multi-seed, CSV, PNG, and GIF tooling;
  - contains an explicitly offline/global-obstacle fallback and therefore must
    not be presented as the formal perception-chain benchmark.
- `mppi_hardware_bridge/scripts/`
  - odometry and LaserScan ingestion, `/cmd_vel`, fail-safe checks,
    `scan_guard`, local-obstacle extraction, planner bridge, Memory, smoothing,
    goal tracking, and final arbitration.
- `docs/project_recovery_sync_2026-05-28.md` and
  `docs/memory_augmented_mppi_implementation_notes.md`
  - define the protected system narrative and safety priority;
  - some status statements are historical snapshots and predate commit
    `90039f1`, so current Git state takes precedence.

## 4. Paper audit and replication boundary

The local paper `2605.03260v1(1).pdf` was checked visually and by text
extraction. The reviewed material includes the Abstract, Sections II-B, II-C,
III-A, III-B, III-C, Equations (10)-(12), and Figures 1-2. The original ICODE
paper was also located:

- ICODE-MPPI: <https://arxiv.org/abs/2605.03260>
- Original ICODE: <https://arxiv.org/abs/2411.13914>

Implementation interpretation:

- Paper Eq. (10) is the control-affine residual
  `f_res(x,u) = f_theta(x) + G_theta(x) u`.
- Paper Eq. (11) combines nominal and residual derivatives before rollout.
- Paper Eq. (12) displays a **one-step** RK4 state-prediction loss, although
  the adjacent prose calls it multi-step prediction. This framework implements
  Eq. (12) and adds a separately identified H-step rollout loss as a research
  extension.
- Figure 1 places the augmented model only in MPPI forward prediction. The
  repository's perception and safety layers remain outside and above that
  replacement point.
- Figure 2 motivates random exploration plus task/on-policy aggregation and
  model-versioned iterative retraining.
- The paper nominal model is a five-state bicycle model with controls
  `(acceleration, steering_rate)`. This repository deliberately retains its
  existing three-state unicycle with controls `(v, omega)` for compatibility.

The original ICODE paper provides sufficient contraction conditions involving
a uniformly positive-definite metric and a transformed variational-dynamics
inequality. A normal unconstrained drift/gain MLP does not satisfy those
conditions by construction. This implementation therefore reproduces the
public control-affine residual structure and continuous-time integration, but
does **not** claim contraction, stability, convergence, MPPI closed-loop
stability, collision avoidance, or real-robot safety guarantees.

## 5. Planned modification surface

New research-owned modules:

```text
src/dynamics/**
src/learning/**
src/planners/mppi_dynamics_adapter.py
src/planners/sampling_prior.py
experiments/icode/**
configs/icode/**
tests/dynamics/**
tests/learning/**
tests/planners/**
docs/icode/**
```

Minimal compatible changes:

- `src/planners/mppi_mujoco_receding_horizon_experiment.py`
  - append an optional dynamics argument to `rollout_control_sequence`;
  - preserve the original three-argument path exactly;
  - pass the optional model through nominal-rollout sampling only where needed.
- `.gitignore`
  - append generated ICODE result/checkpoint/log rules without removing any
    existing rule.
- `README.md`
  - later add a concise research-framework entry without rewriting the
    historical project narrative.

Files treated as frozen in this phase:

- `mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py`;
- `mppi_hardware_bridge/scripts/scan_guard.py`;
- `mppi_hardware_bridge/scripts/local_obstacle_layer.py`;
- `mppi_hardware_bridge/scripts/control_adapter.py`;
- hardware runtime YAML safety and goal settings;
- existing MuJoCo XML assets;
- Memory implementation and its safety priority.

The planner bridge and Memory helper are also left unchanged unless a later
regression proves a narrowly scoped compatibility change is necessary.

## 6. Compatibility and research risks

| Risk | Control |
|---|---|
| RK4 silently changes the old nominal MPPI | `dynamics_model=None` keeps the old Euler `step` path; exact regression test uses a pre-change golden trajectory. |
| Planner observes the true plant | Planner and execution models are separate objects; only an explicitly named oracle ablation may use exact residuals. |
| Memory confounds residual ablation | ICODE experiment configs default Memory to off. |
| Formal planner sees global obstacle truth | Formal MuJoCo scenarios retain LaserScan -> local obstacle layer; clean dynamics benchmark is explicitly perception-free. |
| Angle discontinuity corrupts targets/loss | Wrapped finite differences and wrapped heading errors; optional sin/cos input encoding. |
| Episode leakage inflates metrics | Split by episode/seed/disturbance group; train-only normalization. |
| Stateful delay is treated as Markov | Control delay is implemented at plant-step level and recorded in metadata; its hidden-history limitation is documented. |
| Per-sample Torch inference is too slow | Correct scalar adapter first, inference timing reported, batch/vectorized rollout left as an explicit optimization path. |
| Existing ignored bytecode shadows new source | Add source `src/__init__.py`; never rely on legacy Python bytecode. |
| Formal results are confused with smoke output | Every run records run type, config, seed, Git SHA, checkpoint, and provenance. |

## 7. Python 2 / Python 3 boundary

- ROS Kinetic hardware scripts retain Python-2-compatible style and do not
  import PyTorch, checkpoints, trainers, or Python-3-only annotations.
- ICODE training/evaluation is Python 3. The host is Ubuntu 20.04 with Python
  3.8.10, so code uses `Optional[T]` instead of Python-3.10-only `T | None`.
- At audit time, neither the system interpreter nor the repository `.venv`
  contained `pytest` or `torch`. Dependencies must be declared and installed in
  the Python 3 research environment before Gates 3-5 can be validated.
- The existing hardware planner import path already has historical Python 2/3
  tension; this work does not expand it.
- A future real-robot inference path should use an independent Python 3 node,
  TorchScript/ONNX, IPC, or reviewed NumPy export. Direct learned-control
  deployment is outside this phase.

## 8. Gate 0 decision

Gate 0 is acceptable when this audit and `01_architecture.md` are present and
the repository still reports only the protected user file plus the new ICODE
work. Gate 1 may then implement dynamics in isolated modules before any MPPI
or hardware integration is attempted.
