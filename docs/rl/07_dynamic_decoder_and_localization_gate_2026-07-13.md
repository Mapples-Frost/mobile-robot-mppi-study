# L8：动态局部目标翻译器与定位混杂因素 Gate（2026-07-13）

## 1. 本阶段问题

L7 表明：即使离线无碰撞路线已经给出，二维局部目标仍不能在原 U-trap
设置中稳定引导 MPPI。L8 先检验以下假设：

1. 旧翻译器把命令速度直接当作实际速度，忽略当前 `v/omega`、一阶执行器响应与命令延迟；
2. 离线诊断路线可能贴着 `scan_guard` 停止边界，导致“正确路线”在闭环中不可执行；
3. 长路线上的轮速里程计漂移可能被误判为 RL、MPPI 或残差动力学失败。

本阶段始终保持：

- ICODE 关闭；
- memory 关闭；
- RL 网络不参与，局部目标来自特权离线路线；
- planner 障碍物仍只来自 `LaserScan -> local_obstacle_layer`；
- `scan_guard` 与安全仲裁保持开启；
- 目标保持 `(3.0, 3.0)`；
- 特权路线仅用于接口诊断，不是正式 planner，也不是性能上界的数学证明。

## 2. 代码变化

### 2.1 可切换翻译器

`PriorParameterizationConfig.subgoal_decoder` 支持：

- `kinematic`：原有三状态、命令立即生效的翻译器；
- `dynamic_first_order`：以测得的 `v, omega` 初始化内部五状态，并使用
  速度/角速度时间常数和命令延迟生成完整 MPPI 均值序列。

策略动作维度没有变化，仍为：

```text
action = [normalized_subgoal_distance, normalized_subgoal_bearing]
```

动态翻译器的内部预测为：

```text
x_dot     = v cos(theta)
y_dot     = v sin(theta)
theta_dot = omega
v_dot     = (v_cmd - v) / tau_v
omega_dot = (omega_cmd - omega) / tau_omega
```

`tau_v`、`tau_omega` 和 `command_delay` 默认从同一次实验的 plant 配置解析，
解析后的数值写入配置快照与 checkpoint。旧配置和旧 checkpoint 缺少
`subgoal_decoder` 时仍使用 `kinematic`，不会静默改变历史实验。

### 2.2 已执行控制闭环

训练 prior 现在与推理 prior 一样接收安全仲裁后的上一条执行控制。这样命令延迟模型使用的是
实际发送给 plant 的控制，而不是被 `scan_guard` 拦截前的提议控制。

### 2.3 分段净空诊断路线

原路线全程只增加 `0.10 m` 几何余量，接近 `scan_guard` 的停止边界；
而 U-trap 目标点受邻近障碍限制，最大额外几何净空只有约 `0.1209 m`，
不能简单把全局 margin 增大到 `0.20 m`。

新诊断路线采用：

```text
主体路线 margin = 0.20 m
终点后缀 margin = 0.10 m
```

高净空前缀与目标附近低净空后缀都由离线几何审计生成，并记录切换位置。
它只产生二维局部目标，不向 MPPI 提供障碍物真值。

### 2.4 显式定位来源

`SimulatedSensorSuite` 新增两个向后兼容开关：

```yaml
sensors:
  pose_source: wheel_odometry   # 默认，兼容旧实验
  twist_source: wheel_odometry
```

受控诊断可配置为：

```yaml
sensors:
  pose_source: ground_truth
  twist_source: ground_truth
```

后者等价于仿真中的外部定位/mocap，并继续叠加原传感器噪声。它只提供机器人状态，
不提供障碍物真值；LaserScan、安全链和局部障碍层保持不变。两种来源必须分开报告，
不能把仿真真值定位结果冒充纯轮速里程计结果。

## 3. 根因审计结果

在动态翻译器、前视距离 `0.70 m`、同一分段净空路线下：

| 位姿来源 | U-trap 成功 | 最终目标距离 | 定位位置 RMSE | 最大定位误差 | 碰撞 |
|---|---:|---:|---:|---:|---:|
| wheel odometry | 0/1 | 0.941 m | 0.642 m | 1.036 m | 0 |
| noisy simulated localization | 1/1 | 0.297 m | 0.00073 m | 0.00170 m | 0 |

轮速里程计实验末期，真实小车仍距目标约 `0.94 m`，但累计轮滑使里程计认为机器人
已经到达目标附近，二维局部目标距离因此被截到最小值。这个误差属于状态估计/定位层，
不能归因于 ICODE 动力学残差，也不能通过继续训练 SAC 诚实解决。

## 4. 开发种子选择

在 noisy simulated localization、U-trap、seed `0`、`K=100` 下：

| 前视距离 | 成功 | 最终目标距离 | 最小净空 | 安全干预 | 碰撞 |
|---:|---:|---:|---:|---:|---:|
| 0.45 m | 1/1 | 0.288 m | 0.310 m | 28 | 0 |
| 0.70 m | 1/1 | 0.297 m | 0.343 m | 16 | 0 |
| 0.95 m | 1/1 | 0.300 m | 0.363 m | 16 | 0 |

按预先采用的折中标准（成功、低安全干预、终点不过度贴成功阈值），选择 `0.70 m`，
随后不再用 held-out seeds 调参。

## 5. held-out seeds 41–45

统一条件：`K=100`、horizon `36`、前视 `0.70 m`、noisy simulated localization、
分段净空路线、ICODE off、memory off。

| 翻译器 | 场景 | 成功率 | 碰撞率 | 最终距离均值 | 最小净空均值 | 安全干预均值 | 规划均耗时 |
|---|---|---:|---:|---:|---:|---:|---:|
| dynamic_first_order | clean | 5/5 | 0/5 | 0.285 m | 0.483 m | 0.0 | 4.48 ms |
| dynamic_first_order | U-trap | 5/5 | 0/5 | 0.296 m | 0.341 m | 15.8 | 4.87 ms |
| kinematic | clean | 5/5 | 0/5 | 0.289 m | 0.483 m | 0.0 | 3.78 ms |
| kinematic | U-trap | 5/5 | 0/5 | 0.296 m | 0.352 m | 15.6 | 4.07 ms |

## 6. 可证伪结论与决策

### 已支持

1. 二维局部目标接口在定位受控、路线净空公平时可以稳定引导低样本 MPPI；
2. seed `0` 选择后，在独立 seeds `41–45` 上 clean 与 U-trap 均达到 5/5、零碰撞；
3. 先前失败包含显著定位混杂因素，不能全部归因于 SAC 或翻译器。

### 未支持

1. 当前数据不支持“动态翻译器显著优于运动学翻译器”；两者成功率相同，动态版耗时更高；
2. 当前结果不证明 RL 已学会 U-trap，因为本 Gate 使用特权离线路线；
3. 当前结果不证明轮速里程计条件下系统已可部署；
4. 当前结果与 ICODE 无因果关系，因为 ICODE 在整个 Gate 中关闭。

### 下一阶段决策

- 保留 `kinematic` 与 `dynamic_first_order` 两个开关；
- SAC 主可学习性实验先采用更简单、计算更低的 `kinematic` 版本；
- 使用 noisy simulated localization 隔离 RL sampling-prior 学习问题；
- 将 wheel-odometry robustness 作为独立实验轴，不与主方法训练混在一起；
- 现在满足进入 scene-balanced replay / SAC 训练的前置 Gate；
- 在 RL 单模块通过前，不接入 ICODE+RL、OOD gate 或动态障碍物。

## 7. 复现实验

动态翻译器 held-out Gate：

```bash
.venv/bin/python experiments/rl/run_scripted_subgoal_upper_bound.py \
  --rl-config configs/rl/sac_mppi_subgoal_dynamic_l8.yaml \
  --configs configs/research/mujoco_strong_mppi_baseline.yaml \
            configs/research/mujoco_u_trap_long_board.yaml \
  --output-dir results/research_platform/rl/l8_dynamic_subgoal_gate_heldout_20260713 \
  --seeds 41,42,43,44,45 --lookaheads 0.70 \
  --num-samples 100 --max-steps 360 \
  --route-margin 0.20 --endpoint-route-margin 0.10 --route-resolution 0.02
```

运动学翻译器同条件对照：

```bash
.venv/bin/python experiments/rl/run_scripted_subgoal_upper_bound.py \
  --rl-config configs/rl/sac_mppi_subgoal_kinematic_localized_l8.yaml \
  --configs configs/research/mujoco_strong_mppi_baseline.yaml \
            configs/research/mujoco_u_trap_long_board.yaml \
  --output-dir results/research_platform/rl/l8_kinematic_subgoal_control_heldout_20260713 \
  --seeds 41,42,43,44,45 --lookaheads 0.70 \
  --num-samples 100 --max-steps 360 \
  --route-margin 0.20 --endpoint-route-margin 0.10 --route-resolution 0.02
```

主要产物：

- `results/research_platform/rl/l8_dynamic_subgoal_gate_heldout_20260713/`
- `results/research_platform/rl/l8_kinematic_subgoal_control_heldout_20260713/`
- `results/research_platform/rl/l8_wheel_odom_confound_seed0_20260713/`
