# Mobile Robot MPPI Study

本仓库用于研究移动机器人导航与动态避障中的 MPPI（Model Predictive Path Integral）控制方法，并逐步从二维 Python 原型推进到 MuJoCo 仿真和 ROS 实车硬件桥接。

当前项目重点已经从最初的二维算法学习，推进到：

- vanilla MPPI / receding horizon 控制
- MuJoCo 点机器人仿真
- ROS Kinetic 实车桥接
- LaserScan 局部障碍物建模
- 长条障碍物的几何表达
- 安全状态机与控制仲裁
- Memory-Augmented Potential Field / MA-MPPI 方向探索

---

## 当前阶段

### 已完成

- [x] 二维环境与圆形障碍物实验台
- [x] unicycle / point robot 运动学 rollout
- [x] vanilla MPPI 采样、rollout、trajectory cost、weighted update
- [x] receding horizon MPPI 主循环
- [x] MuJoCo 点机器人 MPPI 初步实验
- [x] ROS 硬件桥接层原型
- [x] LaserScan scan_guard 安全检测
- [x] LaserScan 局部障碍物提取
- [x] 长条障碍 line surface + representative circles 表达
- [x] MPPI 实车动态障碍物 cost 接入
- [x] CLEAR / APPROACH_SLOW / CREEP_ESCAPE / HARD_STOP_RECOVERY / GOAL_REACQUIRE 避障状态机
- [x] MPPI / safety / goal tracking / smoothing 控制仲裁
- [x] Memory-Augmented Potential Field 轻量实现
- [x] 实车日志记录、行为测试与 runtime diagnostics

### 正在做

- [ ] 将 Memory-Augmented MPPI 接入 MuJoCo 仿真平台
- [ ] 做 memory on/off 消融实验
- [ ] 对比普通 MPPI 与 Memory-Augmented MPPI 的局部最优逃逸能力
- [ ] 统计不同 samples / horizon / realtime profile 下的计算耗时
- [ ] 整理组会汇报图和实验表格

---

## 当前系统架构

当前实车控制链路如下：

```text
Odometry + LaserScan
        ↓
scan_guard / local_obstacle_layer
        ↓
front range / side range / line surface / representative circles
        ↓
MppiPlannerBridge
        ↓
MPPI sampling + rollout + trajectory cost
        ↓
goal cost + obstacle cost + control cost + spin cost + memory cost
        ↓
proposed_control = (v, omega)
        ↓
ROS Adapter Arbitration
        ↓
safety clamp / creep escape / memory escape / goal tracking / smoothing
        ↓
final_control
        ↓
/cmd_vel
```

---

## ICODE 残差动力学研究框架

仓库现已包含兼容式的 Python 3 residual-dynamics 研究层：

- 三状态 unicycle nominal model 与可控 disturbed true plant；
- Euler / RK4、角度周期编码、Oracle residual；
- 可追踪 NPZ + JSON + CSV 数据集与 episode-level split；
- MLP baseline 与 control-affine ICODE-style residual；
- derivative / one-step / multi-step rollout loss、完整 checkpoint、resume；
- 原 MPPI rollout 的可选 dynamics adapter，默认 nominal 行为保持不变；
- clean dynamics model/control benchmark 与五方法多 seed runner。

ICODE 代码不会进入 `mppi_hardware_bridge/scripts/`，不会替换
LaserScan、`scan_guard`、`local_obstacle_layer` 或控制安全仲裁。当前实现复现
公开的 `f_theta(x) + G_theta(x)u` residual structure，但没有实现或声称原始
ICODE contraction、稳定性或收敛保证。

从零开始的入口与边界说明见 [`docs/icode/01_architecture.md`](docs/icode/01_architecture.md)
和 [`docs/icode/07_experiment_protocol.md`](docs/icode/07_experiment_protocol.md)。
