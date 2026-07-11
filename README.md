# Mobile Robot MPPI Research Platform

本仓库是一套面向科研实验的移动机器人控制平台，主线包括：

- 可变目标、路径与轨迹参考；
- 维度可配置的 MPPI；
- MuJoCo 差速驱动物理仿真；
- LaserScan → `scan_guard` → `local_obstacle_layer` 感知安全链；
- Nominal、Oracle、MLP、ICODE 残差预测模型；
- Memory-Augmented MPPI；
- 为 RL-driven sampling 保留的稳定接口；
- 统一的数据、指标、配置与多随机种子 benchmark。

当前研究代码位于标准 `src` package：

```text
src/mobile_robot_mppi/
```

原有 planner、MuJoCo live sim 和 ROS Kinetic 实车脚本继续保留，作为兼容基线与实车安全资产，不会被训练依赖或 PyTorch 污染。

## 核心边界

```text
Task / Reference
       ↓
RobotObservation ← Odom / Encoder / IMU / LaserScan
       ↓
scan_guard → local_obstacle_layer
       ↓
MPPI rollout using PredictionDynamics
       ↓
proposed control
       ↓
safety arbitration
       ↓
executed control
       ↓
MuJoCo true plant / real robot
```

必须区分：

```text
PredictionDynamics ≠ environment true plant
RobotObservation ≠ MuJoCo ground truth
Proposed control ≠ executed control
Viewer ≠ simulation environment
```

MuJoCo ground truth 只能进入 logger、metrics 和 residual dataset，不得直接提供给普通 MPPI。

## MuJoCo 物理主线

新 backend：`mujoco_diff_drive`。

```text
6-DoF free chassis
+ left/right wheel hinge
+ caster contacts
+ two wheel actuators
+ wheel-ground friction/contact
+ encoder/actuator/IMU sensors
```

控制链：

```text
(v_cmd, omega_cmd)
→ left/right wheel targets
→ velocity servo or torque PI
→ MuJoCo contact dynamics
→ odometry/LaserScan observation
```

旧的 `legacy_kinematic` backend 继续存在，用于数值回归和历史结果复现。

## 可变 Goal 与扩展状态

Goal 不再是核心代码中的全局常量，支持 `PointGoal`、`PoseGoal`、`WaypointReference` 和 `TimeTrajectoryReference`。旧实验默认仍可配置为 `(3.0, 3.0)`，但可以通过 YAML 或 CLI 修改：

```bash
python experiments/run_research_simulation.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --goal 2.0 1.5 --headless
```

支持的规划状态：

```text
unicycle_3:         [x, y, theta]
dynamic_unicycle_5: [x, y, theta, v, omega]
wheel_augmented_7:  接口已定义，需显式七状态预测模型
```

高层动作默认保持 `[v_cmd, omega_cmd]`。`StateSpec` 与 `ActionSpec` 允许后续增加状态和控制维度，不需要重写 MPPI。

## ICODE

```text
f_res(x, u) = f_theta(x) + G_theta(x) u
f_pred(x, u) = f_nominal(x, u) + f_res(x, u)
```

当前保留两条研究链：

1. `src/dynamics`、`src/learning`：已验证的三状态 ICODE 兼容框架；
2. `src/mobile_robot_mppi/learning`：面向 MuJoCo 五状态数据的新训练入口。

当前实现复现公开的 control-affine residual structure，但没有实现或声称原始 ICODE 的 contraction、稳定性或收敛保证。

## RL 扩展

推荐的第一接入点：

```text
RLPolicyPrior → MPPI mean/covariance → MPPI optimization → safety arbitration
```

核心 planner 不依赖 Torch policy。可选 Gymnasium 风格 adapter 会分别记录 `proposed_action`、`executed_action` 和 `safety_override`。RL 永远不能绕过 `scan_guard`。

## 安装

```bash
cd /home/mapples/projects/mobile-robot-mppi-study
python3 -m venv .venv
.venv/bin/pip install -e '.[simulation,learning,dev]'
```

当前物理基线固定 MuJoCo `3.2.3`，避免重构与依赖升级同时改变实验行为。

## 快速运行

Legacy 回归：

```bash
.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/legacy_kinematic.yaml --headless
```

差速物理 MuJoCo：

```bash
.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/mujoco_diff_drive.yaml --headless
```

开启 viewer 时将 `--headless` 改为 `--viewer`。

## MuJoCo → ICODE

```bash
.venv/bin/python experiments/collect_mujoco_residual_data.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --output-dir results/research_platform/datasets/mujoco_v1 \
  --episodes 40 --steps 120 --source mixed

.venv/bin/python experiments/train_platform_residual.py \
  --config configs/research/icode_dynamic5.yaml \
  --dataset-dir results/research_platform/datasets/mujoco_v1 \
  --output-dir results/research_platform/checkpoints/icode_v1

.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --prediction-mode icode_residual \
  --checkpoint results/research_platform/checkpoints/icode_v1/best.pt \
  --headless
```

将训练配置替换为 `configs/research/mlp_dynamic5.yaml` 即可训练 MLP baseline。

## Multi-seed benchmark

```bash
.venv/bin/python experiments/run_research_benchmark.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --output-dir results/research_platform/benchmark_v1 \
  --seeds 11,12,13,14,15 \
  --samples 50,100,200,400 \
  --mlp-checkpoint results/research_platform/checkpoints/mlp_v1/best.pt \
  --icode-checkpoint results/research_platform/checkpoints/icode_v1/best.pt
```

每次运行输出 resolved config、provenance、trajectory CSV 和 metrics JSON；benchmark 额外输出 episode CSV、summary CSV 和聚合 JSON。

## 测试

```bash
.venv/bin/python -m pytest tests/dynamics -q
.venv/bin/python -m pytest tests/learning -q
.venv/bin/python -m pytest tests/planners -q
.venv/bin/python -m pytest tests/platform -q
```

## 实车边界

`mppi_hardware_bridge/scripts/` 仍是 ROS Kinetic/Python 2 兼容区域：

- 不直接 import Torch；
- 不把训练依赖变成启动依赖；
- `/cmd_vel`、odom、LaserScan、scan_guard 和控制仲裁保持不变；
- learned residual 必须先经过仿真、offline 和 shadow mode。

详细说明见 [docs/refactor/00_master_plan.md](docs/refactor/00_master_plan.md)。
