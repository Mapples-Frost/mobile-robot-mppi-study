# mobile-robot-mppi-study 项目恢复同步文档

生成日期：2026-05-28
用途：账号 / Project 上下文丢失后，在新的 ChatGPT Project 中恢复科研上下文、代码进度、实验方向和后续协作约束。
本地仓库：`/home/mapples/projects/mobile-robot-mppi-study`
GitHub 仓库：`https://github.com/Mapples-Frost/mobile-robot-mppi-study`
当前本地分支：`mujoco-memory-bridge-ablation`

> 这份文档不是论文初稿，而是“科研项目状态快照”。新 Project 读完后，应能知道这个项目为什么做、做到哪一步、哪些代码是主线、哪些东西不能乱改、下一步应该怎么继续带着你推进。

---

## 0. 给新 ChatGPT Project 的使用方式

如果新 Project 只能读 GitHub 链接，它可能看不到当前本地未提交的新脚本。当前本地工作区还有这些重要文件处于未提交状态：

- `experiments/mujoco_memory_mppi_ablation.py`
- `experiments/mujoco_memory_mppi_live_sim.py`
- `mppi_hardware_bridge/scripts/mppi_memory_field.py` 有小改
- `src/planners/mppi_mujoco_receding_horizon_experiment.py` 有小改

因此，新的 ChatGPT Project 启动时，建议把这份文档全文放进去，并明确告诉它：

1. 先读这份同步文档。
2. 再读 GitHub 仓库。
3. 如果它不能看到本地未提交文件，要以这份文档描述的本地最新进展为准。
4. 不要默认重写系统，不要回到早期二维 MPPI demo。

---

## 1. 项目一句话定位

本项目研究移动机器人在局部障碍环境中的 MPPI（Model Predictive Path Integral）避障控制方法，并从早期二维 Python 原型逐步推进到：

- MuJoCo 实时物理仿真；
- synthetic LaserScan 感知闭环；
- scan_guard 安全检测；
- local_obstacle_layer 局部障碍建模；
- ROS Kinetic 实车硬件 bridge；
- Memory-Augmented Potential Field / Memory-Augmented MPPI；
- 后续 memory on/off 消融实验与实车对比。

当前项目已经不是“二维 MPPI 能不能跑”的阶段，而是在做：

> 使用 MuJoCo 复现实车感知-规划-执行链路，并测试 Memory-Augmented MPPI 是否能减少局部低进展、重复转圈和近障陷入。

---

## 2. 当前最重要结论

截至当前状态，项目的主线已经从单纯 planner 代码推进到系统闭环：

```text
MuJoCo World / Real Robot
        -> LaserScan / Synthetic LaserScan
        -> scan_guard
        -> local_obstacle_layer
        -> MPPI rollout cost
        -> Memory field soft cost
        -> proposed control
        -> safety arbitration
        -> final cmd_vel / MuJoCo qvel
        -> robot motion
```

这条链路的关键意义是：MPPI 现在不再直接“偷看全局障碍列表”，而是尽量通过 LaserScan-like 输入生成局部 planner obstacles。这样 MuJoCo 仿真与真实小车的输入形式更接近。

当前最重要的研究问题不是“MPPI 是否能从 A 到 B”，而是：

1. 在局部可见障碍、长条障碍、窄通道、U 型陷阱中，MPPI 是否稳定？
2. 当 MPPI 出现原地转圈、低进展、近障徘徊时，memory field 是否能记录失败区域？
3. memory cost 是否能改变 rollout cost landscape，从而减少重复进入相同局部失败区域？
4. memory 不应成为硬控制器，它应该通过 cost 和 sampling bias 影响 MPPI，而不是直接覆盖 final control。

---

## 3. 当前 Git / 工作区状态

最后一次检查时：

```text
branch: mujoco-memory-bridge-ablation

git status --short:
 M mppi_hardware_bridge/scripts/mppi_memory_field.py
 M src/planners/mppi_mujoco_receding_horizon_experiment.py
?? experiments/mujoco_memory_mppi_ablation.py
?? experiments/mujoco_memory_mppi_live_sim.py
```

远程仓库：

```text
origin  git@github.com:Mapples-Frost/mobile-robot-mppi-study.git
```

最近已推送的分支为：

```text
mujoco-memory-bridge-ablation -> origin/mujoco-memory-bridge-ablation
```

注意：

- 本地新脚本目前未提交，因此新 Project 只读 GitHub 可能看不到。
- 不要把 `experiments/results/` 里的 CSV、PNG、GIF 当作必须提交的代码资产。
- 后续若要把当前进展同步到 GitHub，需要先人工确认 diff，再单独 `git add` 指定文件，不要使用 `git add .`。

---

## 4. 本次组会 PPT 内容摘要

本次组会 PPT 路径：

```text
C:/Users/lenovo/OneDrive/文档/第二次组会汇报.pptx
```

PPT 标题：

```text
基于 Memory-Augmented MPPI 的移动机器人自主避障系统进展汇报
MuJoCo Live Simulation + Synthetic LaserScan + ROS Hardware Bridge
```

PPT 的主线可以概括为：

1. 项目从二维 MPPI 基线推进到真实闭环。
2. 当前三条线同时推进：
   - MuJoCo live simulation；
   - ROS 实车硬件 bridge；
   - Memory-Augmented MPPI。
3. MuJoCo 与实车共用一条感知-规划-执行逻辑：
   - MuJoCo ray casting 生成 synthetic LaserScan；
   - scan_guard 负责硬安全；
   - local_obstacle_layer 把 LaserScan 转成局部障碍；
   - MPPI + Memory 做采样规划；
   - MuJoCo / 实车执行速度命令。
4. Viewer 已经能显示：
   - robot；
   - MuJoCo obstacles；
   - LaserScan rays；
   - hit points；
   - planner obstacles；
   - MPPI predicted trajectory；
   - executed trajectory；
   - memory features。
5. 当前问题：
   - 青色预测轨迹仍可能在紫色 memory feature 附近回绕；
   - 黄色 planner obstacles 过密会让 MPPI 过于保守；
   - 需要区分 proposed_control 与 final_control；
   - 下一步加强 memory cost、escape bias 和 obstacle filtering。
6. 实车调试中发现控制权和驱动启动链耦合问题：
   - 原厂控制程序权限过高，可能干扰外部 MPPI 指令；
   - 注释 WebServer.sh 后，roscore / Lidar / Odom 链路可能被连带破坏；
   - 推测基础驱动层与上层演示逻辑没有解耦；
   - 预案包括手动拆解 launch、使用 cmd_vel_mux 优先级覆盖、物理端口重定向。

---

## 5. 项目阶段时间线

### 5.1 早期二维 MPPI baseline

早期阶段重点是搭建最小 MPPI 学习实验台：

- 二维平面环境；
- 起点、目标点、圆形障碍；
- unicycle / point robot 运动学；
- 控制序列采样；
- rollout；
- trajectory cost；
- exponential weights；
- weighted update；
- receding horizon 执行。

这个阶段形成了对 MPPI 主循环的基本理解：

```text
sample control sequences
    -> rollout trajectories
    -> compute trajectory costs
    -> compute weights
    -> weighted update sequence
    -> execute first control
    -> shift sequence
```

### 5.2 baseline 实验框架整理

随后把 demo 推进到可重复实验：

- 固定场景配置；
- sparse / dense / narrow 等场景；
- num_samples 影响；
- horizon 影响；
- warm start 影响；
- top-k predicted trajectories 可视化。

这一阶段的关键认识：

- horizon 影响机器人能否看到绕行路线；
- warm start 主要改善控制连续性和初始采样分布；
- dense 场景中的失败不只是 terminal cost 不够，而是采样分布、局部可行空间和安全边界共同导致。

### 5.3 MuJoCo 初步仿真

项目从二维绘图推进到 MuJoCo viewer：

- 使用 MuJoCo 点机器人；
- 在 viewer 中显示机器人、障碍、目标；
- 加入 predicted trajectory 在线可视化；
- 调试边界内推和采样偏置；
- 发现并修复过一次 anisotropic sampling / boundary bias 被旧逻辑覆盖的 bug。

这个阶段留下一个重要工程经验：

> 显示层和算法层要分开改。MuJoCo 外观 / overlay 调整不能随便混入 planner 行为修改，否则很难定位问题。

### 5.4 ROS 实车 bridge

项目从仿真继续推进到 E1/E2 小车：

- odom 输入；
- LaserScan 输入；
- MPPI planner bridge；
- control adapter；
- safety clamp；
- scan_guard；
- local obstacle layer；
- `/cmd_vel` 输出；
- runtime config；
- 实车调试 runbook；
- 回放 / diagnostics。

实车链路不是最终论文的唯一目标，但它证明当前算法不是只存在于离线 demo 中。

### 5.5 Memory-Augmented MPPI

当前新增方向是 Memory-Augmented Potential Field / Memory-Augmented MPPI。

核心思想：

```text
如果机器人在某些区域反复低进展、近障慢爬、原地旋转，
就把这些区域记录为 memory features。
之后 MPPI rollout 如果再次经过这些区域，
trajectory cost 会增加，
从而降低重复进入局部失败区域的概率。
```

注意：

- Memory 不是硬控制器。
- Memory 不应直接覆盖 final control。
- Memory 应通过 rollout cost 和轻量 sampling / nominal bias 起作用。
- scan_guard 的 hard safety 优先级永远高于 memory escape。

### 5.6 当前最新：MuJoCo live sim 主线

当前最新主脚本是：

```text
experiments/mujoco_memory_mppi_live_sim.py
```

它不再只是 offline ablation，而是一个真正的 MuJoCo live simulation 主入口。

---

## 6. 代码地图

### 6.1 MPPI planner 核心

```text
src/planners/mppi_mujoco_receding_horizon_experiment.py
```

关键 helper：

- `initialize_control_sequence`
- `sample_control_sequences`
- `rollout_control_sequence`
- `trajectory_cost`
- `compute_weights`
- `weighted_update_sequence`
- `shift_sequence`

当前新脚本会优先复用这些 helper，不重新实现 MPPI 框架。

已做小改：

- 将 MuJoCo 环境 import 改成 optional；
- 目的是让 helper 在没有 MuJoCo Python 包的环境中也可以被 ablation / bridge 复用；
- 不改变 helper 函数签名。

### 6.2 MuJoCo 环境

早期环境文件：

```text
src/envs/mujoco_point_env.py
```

当前 live sim 新入口中又内置了一个专用类：

```text
MujocoLiveMppiEnv
```

它在新脚本中负责：

- 用 XML string 创建 MuJoCo model；
- 维护 robot slide / yaw joint；
- 每步设置 qvel；
- 调用 `mujoco.mj_step`；
- 从 MuJoCo qpos 读取 state；
- 用 `mujoco.mj_ray` 做 synthetic LaserScan；
- 启动 viewer；
- 在 viewer.user_scn 中画 debug overlay。

### 6.3 ROS / 实车 bridge

主要目录：

```text
mppi_hardware_bridge/
```

重要文件：

- `mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py`
  - ROS 节点主骨架；
  - 订阅 odom / scan；
  - 输出 `/cmd_vel`；
  - 包含安全仲裁、目标跟踪、smoothing 等实车主逻辑；
  - 当前任务中不要大改。

- `mppi_hardware_bridge/scripts/mppi_planner_bridge.py`
  - 把 planner helper 封装给硬件 bridge；
  - 做配置解析、obstacle cost、memory cost 相关接入；
  - 连接 MPPI helper 和实车 adapter。

- `mppi_hardware_bridge/scripts/control_adapter.py`
  - 控制限幅；
  - emergency stop；
  - Twist dict 转换；
  - safe command 准备。

- `mppi_hardware_bridge/scripts/scan_guard.py`
  - 前方安全检测；
  - hard stop / slow down；
  - front sector / side / near body 点分类；
  - 当前不能关闭。

- `mppi_hardware_bridge/scripts/local_obstacle_layer.py`
  - LaserScan 点转换；
  - local points；
  - geometry features；
  - line segment / circle fitting；
  - long obstacle compression；
  - representative circles；
  - 转成 MPPI planner obstacles。

- `mppi_hardware_bridge/scripts/mppi_memory_field.py`
  - Memory feature 存储；
  - stuck / spin / low-progress 记录；
  - memory cost；
  - temperature scale；
  - 当前有小范围兼容 helper 新增。

- `mppi_hardware_bridge/config/lab_runtime.yaml`
  - 实车 runtime 参数；
  - 不要随意改 goal / scan_guard。

- `mppi_hardware_bridge/config/lab_runtime_goal_3_3_safe.yaml`
  - goal=(3,3) 安全配置；
  - 不要为了仿真实验乱改实车配置。

### 6.4 新增实验脚本

当前本地新增但未提交：

```text
experiments/mujoco_memory_mppi_ablation.py
experiments/mujoco_memory_mppi_live_sim.py
```

二者定位不同：

1. `mujoco_memory_mppi_ablation.py`
   - offline memory on/off 消融；
   - 支持 fallback kinematic simulator；
   - 输出 CSV / PNG / GIF；
   - 适合做统计表格和组会静态图；
   - 不是当前 live sim 主入口。

2. `mujoco_memory_mppi_live_sim.py`
   - 当前主线；
   - MuJoCo 是主平台；
   - viewer 实时显示；
   - synthetic LaserScan；
   - scan_guard；
   - local_obstacle_layer；
   - MPPI + Memory；
   - 多场景切换；
   - anti-spin / stuck escape。

---

## 7. 当前 MuJoCo live sim 设计

### 7.1 主脚本

```text
experiments/mujoco_memory_mppi_live_sim.py
```

默认目标点固定：

```text
GOAL = (3.0, 3.0)
START_STATE = (0.0, 0.0, 0.0)
```

默认 bounds：

```text
x in [-0.5, 3.6]
y in [-0.5, 3.6]
```

默认 scene：

```text
lab_complex
```

注意：代码当前 `parse_args()` 中 `--max-steps` 默认是 `500`。如果后续需要与早先需求文档完全一致，可以改回 `300`，但这不是当前文档生成任务必须处理的问题。

### 7.2 MuJoCo robot

live sim 中 robot 不是纯 Python 状态，而是 MuJoCo model 里的 body。

设计思想：

- robot body 是圆柱 / 圆盘；
- 使用 planar joints：
  - x slide；
  - y slide；
  - yaw hinge；
- 每步从 MuJoCo qpos 读取：
  - x；
  - y；
  - theta；
- 执行 control `(v, omega)` 时转换：
  - `xdot = v * cos(theta)`；
  - `ydot = v * sin(theta)`；
  - `yawdot = omega`；
- 写入对应 qvel；
- 调用 `mujoco.mj_step`。

### 7.3 MuJoCo obstacles

当前场景不是旧 XML 文件，而是在新脚本中用 XML string 生成。

优点：

- 不修改旧 MuJoCo XML；
- scene primitives 同时用于：
  - 生成 MuJoCo geoms；
  - raycast 命中；
  - debug / collision 参考；
  - viewer 中真实障碍物显示。

场景数据结构：

```text
SCENE_CONFIGS = {
    "simple": {...},
    "lab_complex": {...},
    "narrow_corridor": {...},
    "u_trap_long_board": {...},
}
```

### 7.4 当前四个 scene

#### simple

作用：

- 保留早期紧凑场景；
- 用作回归测试；
- 包含长板、U 型墙和圆柱。

#### lab_complex

作用：

- 当前默认场景；
- 设计成分散绕行，而不是中间障碍堆；
- 用于观察 MPPI + LaserScan + Memory 的综合行为。

当前 lab_complex 障碍：

- `low_board`
  - center=(0.85, 0.65)
  - half_size=(0.08, 0.45)
  - yaw=15 deg
- `mid_bar`
  - center=(1.55, 1.35)
  - half_size=(0.45, 0.07)
- `upper_gate_wall`
  - center=(1.65, 2.25)
  - half_size=(0.07, 0.42)
- `lower_right_pillar`
  - center=(2.25, 1.05)
  - radius=0.16
- `mid_right_pillar`
  - center=(2.55, 1.75)
  - radius=0.15
- `near_goal_1`
  - center=(2.45, 2.55)
  - radius=0.13
- `near_goal_2`
  - center=(2.85, 2.35)
  - radius=0.12
- `left_upper_pillar`
  - center=(0.65, 2.25)
  - radius=0.12

目标路线是：

```text
start=(0,0)
    -> 绕开 low_board
    -> 穿过 mid_bar 和 upper_gate_wall 形成的错位通道
    -> 避开右侧稀疏圆柱
    -> goal=(3,3)
```

#### narrow_corridor

作用：

- 测试长条墙和窄通道；
- 观察 LaserScan 扫到长条墙面后，local_obstacle_layer 是否能压缩成代表障碍，而不是产生过多分散点；
- 通道略宽于机器人直径。

#### u_trap_long_board

作用：

- 测试 memory；
- U 型陷阱 + 长板；
- 更容易诱发局部低进展和重复转向；
- 用来观察紫色 memory features 是否生成，以及青色 MPPI predicted trajectory 是否逐渐避开失败区域。

### 7.5 Synthetic LaserScan

每个 planning step：

1. 从 MuJoCo 读取 robot state；
2. 以 robot yaw 为中心生成 360 度扫描角；
3. 对每条 beam 调用 MuJoCo ray casting；
4. 排除机器人自身 geom；
5. 得到：
   - `ranges`
   - `angle_min`
   - `angle_increment`
   - `range_min`
   - `range_max`
   - `hit_points_world`
   - `ray_segments_world`

这一步非常关键，因为它避免了 planner 直接偷看全局 obstacle primitive。

### 7.6 scan_guard 接入

live sim 调用：

```text
mppi_hardware_bridge/scripts/scan_guard.py
```

核心函数：

```text
analyze_scan_front_sector(...)
```

输出包括：

- emergency_stop；
- should_slow_down；
- min_front_range；
- front_stop_mode；
- reason；
- front_points；
- side_points；
- near_body_points。

控制原则：

- 如果 emergency_stop，hard safety 优先；
- 如果 should_slow_down，缩小 v；
- 如果 front_soft_block，限制前进速度；
- memory escape 不允许突破 hard safety。

### 7.7 local_obstacle_layer 接入

live sim 调用：

```text
mppi_hardware_bridge/scripts/local_obstacle_layer.py
```

主路径：

```text
MuJoCo geoms
    -> mj_ray synthetic scan
    -> scan_guard
    -> local_obstacle_layer
    -> planner_obstacles
    -> MPPI trajectory_cost
```

当前不应把全局 scene primitives 直接塞给 MPPI 作为主规划障碍。

### 7.8 planner obstacle filtering

当前新增了轻量过滤：

```text
MAX_PLANNER_OBSTACLES = 35
```

过滤原则：

- 保留距离机器人近的障碍；
- 保留前方 / 侧前方障碍；
- 保留半径较大的 representative circles；
- 丢弃较远且在机器人后方的障碍；
- 丢弃远处小半径点。

目的：

- viewer 中 scan hit points 仍然可见；
- 但黄色 planner obstacles 不要过密；
- 减少 MPPI 因局部障碍点过密而保守或原地转向。

### 7.9 MPPI 每步流程

当前 live sim 的每步流程大致是：

```text
state = env.get_state()
scan = env.raycast_laserscan(state)
scan_result = run_scan_guard(scan)
planner_obstacles = run_local_obstacle_layer(scan, state)
planner_obstacles = filter_planner_obstacles(planner_obstacles, state)

if repeated_spin / stuck_low_progress:
    nominal_sequence = apply_escape_bias_to_nominal_sequence(...)

sampled_sequences = sample_control_sequences(...)
for each sequence:
    trajectory = rollout_control_sequence(...)
    base_cost = trajectory_cost(..., obstacles=planner_obstacles)
    memory_cost = memory_cost_for_trajectory(...)
    total_cost = base_cost + memory_cost

weights = compute_weights(total_costs, effective_temperature)
updated_sequence = weighted_update_sequence(sampled_sequences, weights)
proposed_control = updated_sequence[0]

final_control = apply_scan_guard_control(proposed_control, scan_result)
final_control = apply_anti_spin_final_control(final_control, repeated_spin, scan_result)

env.step(final_control, dt)
memory.update(...)
env.update_debug_visuals(...)
env.render(...)
nominal_sequence = shift_sequence(updated_sequence)
```

### 7.10 Viewer overlay

MuJoCo viewer 当前应能显示：

- robot body；
- static obstacles / walls / goal marker；
- executed trajectory；
- LaserScan rays；
- LaserScan hit points；
- planner obstacles；
- best predicted MPPI trajectory；
- top-k auxiliary predicted trajectories；
- memory feature positions；
- escape direction arrow（如果 escape_active）。

颜色约定：

- robot：蓝色；
- goal：绿色；
- boundary walls：灰色；
- long boards：棕 / 橙；
- U trap / middle wall：深灰；
- cylinders：红 / 深红；
- scan rays：浅蓝 / 灰；
- hit points：红 / 橙；
- planner obstacles：黄色；
- predicted trajectory：青色；
- executed trajectory：橙色；
- memory features：紫色。

---

## 8. Memory-Augmented MPPI 设计

### 8.1 Memory 的定位

Memory 不是一个单独控制器。

它不应该直接说：

```text
final_control = escape_control
```

而应当：

1. 从执行历史中识别失败区域；
2. 把这些区域记录成 MemoryFeature；
3. 在 MPPI rollout cost 中惩罚再次进入；
4. 在持续 stuck / spin 时，轻微改变 nominal_sequence 的前几步方向；
5. 最终仍由 MPPI sampling + weighted update 产生 proposed control；
6. scan_guard hard safety 仍然拥有最高优先级。

### 8.2 MppiMemoryField

文件：

```text
mppi_hardware_bridge/scripts/mppi_memory_field.py
```

主类：

```text
MppiMemoryField
```

已有 / 新增 helper：

- `update(...)`
- `debug_snapshot(...)`
- `memory_cost_for_state(state)`
- `memory_cost_for_trajectory(trajectory, stride=3)`
- `temperature_scale_for_state(state)`

这些 helper 是向后兼容新增，不应破坏实车 bridge。

### 8.3 Memory update 输入

每个真实执行 step 后更新 memory：

- state=(x,y,theta)
- goal_distance
- control=(v,omega)
- min_front_range
- avoidance_state
- now

avoidance_state 简化规则：

```text
if scan_guard emergency_stop:
    HARD_STOP_RECOVERY
elif front_stop_mode in front_soft_block / front_obstacle_slow / side_obstacle_soft:
    CREEP_ESCAPE
else:
    CLEAR
```

### 8.4 stuck / spin 判定

当前 live sim 维护窗口：

- recent_states；
- recent_controls；
- recent_goal_distances。

repeated_spin 典型判定：

- 最近 15~20 步 mean_abs_omega 较大；
- mean_v 很小；
- goal_distance improvement 很小；
- 或连续若干步 `abs(omega)>0.8` 且 `v<0.08`。

stuck_low_progress 判定：

- 最近约 20 步 position span 很小；
- goal distance 改善不足。

### 8.5 escape bias

当：

```text
(repeated_spin or stuck_low_progress) and memory_feature_count > 0
```

触发轻量 escape bias。

逻辑：

1. 找 nearest memory feature；
2. 计算从 memory feature 指向当前机器人位置的方向；
3. 与 goal direction 混合；
4. 得到 escape_dir；
5. 将 nominal_sequence 前几步轻微改向 escape_dir；
6. v 保持在较低前进速度；
7. omega 按 desired_heading-theta 计算并限幅；
8. 如果 scan_guard hard stop，不强行前进。

这解决的是：

> Memory feature 已经生成，但 MPPI 的采样云仍然围绕坏区域打转。

escape bias 的目标不是强行接管控制，而是给 sample cloud 一个离开坏区域的初始倾向。

### 8.6 final control anti-spin

当前还加了非常轻的 final control 限制：

```text
if repeated_spin and front is clear and not emergency_stop:
    if abs(final_omega) > 0.7 and final_v < 0.05:
        final_v = 0.08
        final_omega = clamp(final_omega, -0.45, 0.45)
```

目的：

- 当前方安全时，减少连续很多步 `v≈0` 但 `omega` 很大地原地转；
- 不在 hard_stop / front_soft_block 时强行前进；
- 不破坏 scan_guard。

---

## 9. Offline memory ablation 脚本

文件：

```text
experiments/mujoco_memory_mppi_ablation.py
```

定位：

- 组会 / 表格 / 静态图使用；
- 不是 live sim 主入口；
- 支持 memory off / memory on / run both；
- 支持 CSV；
- 支持 trajectory CSV；
- 支持 PNG；
- 支持 fallback GIF；
- 支持 MuJoCo 可用时尝试使用 viewer，但 fallback 是重要设计。

核心参数：

- `--memory-disable`
- `--memory-enable`
- `--run-both`
- `--episodes`
- `--max-steps`
- `--horizon`
- `--num-samples`
- `--temperature`
- `--dt`
- `--seed`
- `--output-dir`
- `--no-viewer`
- `--use-mujoco`
- `--viewer`
- `--save-gif`
- `--no-gif`

典型命令：

```bash
python3 experiments/mujoco_memory_mppi_ablation.py \
  --run-both \
  --episodes 1 \
  --max-steps 60 \
  --no-viewer \
  --save-gif
```

输出路径：

```text
experiments/results/mujoco_memory_mppi_ablation.csv
experiments/results/mujoco_memory_mppi_trajectory_memory_off.csv
experiments/results/mujoco_memory_mppi_trajectory_memory_on.csv
experiments/results/mujoco_memory_mppi_ablation.png
experiments/results/mujoco_memory_mppi_ablation_memory_off.gif
experiments/results/mujoco_memory_mppi_ablation_memory_on.gif
```

---

## 10. 实车硬件 bridge 当前状态

### 10.1 已完成内容

当前实车链路已经完成过阶段性接入：

- ROS Kinetic 环境；
- E1/E2 小车；
- `/odom`；
- `/scan`；
- `/cmd_vel`；
- MPPI planner bridge；
- scan_guard；
- local_obstacle_layer；
- line surface + representative circles；
- safety arbitration；
- goal tracking；
- smoothing；
- Memory field 接入；
- runtime diagnostics；
- 实车调试流程文档。

### 10.2 实车链路的控制结构

大致链路：

```text
Odometry + LaserScan
        -> scan_guard / local_obstacle_layer
        -> front range / side range / line surface / representative circles
        -> MppiPlannerBridge
        -> MPPI sampling + rollout + trajectory cost
        -> goal cost + obstacle cost + control cost + spin cost + memory cost
        -> proposed_control = (v, omega)
        -> ROS Adapter Arbitration
        -> safety clamp / creep escape / memory escape / goal tracking / smoothing
        -> final_control
        -> /cmd_vel
```

### 10.3 当前不要做的事

当前不是继续随便调实车参数的阶段。

不要：

- 改实车 goal 配置；
- 关闭 scan_guard；
- 大改 `mppi_ros_adapter_skeleton.py`；
- 破坏实车 bridge 主逻辑；
- 为了仿真方便改掉实车安全限制。

### 10.4 当前实车问题：原厂控制权与驱动耦合

PPT 后半部分记录了一个实车系统问题：

1. 小车自带电脑运行的原厂控制程序权限较高，可能干扰外部 MPPI `/cmd_vel`。
2. 曾尝试按手册通过 SSH 修改 `WebServer.sh` 自启动项。
3. 但关闭该脚本后，可能连带导致 `roscore`、Lidar、Odom 不再自动发布。
4. 推测该脚本把基础驱动层和上层演示逻辑耦合在一起。
5. 可能涉及软连接、udev、串口权限、launch 启动链。

预案：

- 从 WebServer.sh 中拆出核心 ROS launch，手动运行基础驱动；
- 使用 `cmd_vel_mux` 或高优先级话题覆盖低优先级原厂控制；
- 直接锁定真实硬件端口，例如将 `/dev/dashgo` 映射问题改为实际 `/dev/ttyUSB0`；
- 任何实车改动前先保留可回退状态。

---

## 11. 当前环境和测试状态

### 11.1 本地 Python / MuJoCo

当前 Codex / WSL 环境中曾出现：

```text
No module named 'mujoco'
```

这意味着：

- `py_compile` 可以通过；
- 但 live sim 运行需要 Python 环境安装 `mujoco`；
- 如果 PyCharm 使用的是另一个解释器，可能能正常打开 viewer；
- 如果在 WSL headless 环境中运行 viewer，还可能遇到 OpenGL / display 问题。

### 11.2 已知测试习惯

建议每次重要修改后至少跑：

```bash
python3 -m py_compile experiments/mujoco_memory_mppi_live_sim.py
```

如果 MuJoCo 可用：

```bash
python3 experiments/mujoco_memory_mppi_live_sim.py --scene lab_complex --max-steps 60 --no-viewer
```

如果 viewer 可用：

```bash
python3 experiments/mujoco_memory_mppi_live_sim.py --scene lab_complex --max-steps 150
```

离线消融：

```bash
python3 experiments/mujoco_memory_mppi_ablation.py \
  --run-both \
  --episodes 1 \
  --max-steps 60 \
  --no-viewer \
  --save-gif
```

### 11.3 PyCharm 运行建议

live sim：

```text
Script path:
/home/mapples/projects/mobile-robot-mppi-study/experiments/mujoco_memory_mppi_live_sim.py

Working directory:
/home/mapples/projects/mobile-robot-mppi-study

Parameters:
默认可以为空
```

切换场景：

```text
--scene simple
--scene lab_complex
--scene narrow_corridor
--scene u_trap_long_board
```

headless：

```text
--no-viewer --max-steps 60
```

关闭 memory：

```text
--memory-disable
```

保存 CSV：

```text
--save-csv
```

---

## 12. 当前研究问题和下一步优先级

### 12.1 短期第一优先级：稳定 live sim 行为

目标：

- lab_complex 中机器人能稳定接近 goal；
- 不在同一个 memory feature 附近无限转圈；
- planner obstacles 数量不过密；
- proposed_control 与 final_control 的差异可解释。

重点看：

- 青色 best predicted trajectory；
- 黄色 planner obstacles；
- 紫色 memory features；
- 绿色 / 紫色 escape arrow；
- terminal log 中：
  - `repeated_spin`
  - `stuck_low_progress`
  - `escape_active`
  - `proposed_v`
  - `proposed_omega`
  - `final_v`
  - `final_omega`
  - `front_stop_mode`
  - `scan_reason`
  - `planner_obstacle_count`
  - `nearest_memory_distance`
  - `effective_temperature`

### 12.2 短期第二优先级：memory cost 调参

当前 memory 有效果但可能不够强。

可观察现象：

- memory feature 已出现，但 predicted trajectory 仍绕着它转；
- escape bias 触发后是否能离开；
- memory_cost 是否足够改变 weights；
- temperature_scale 是否导致采样更分散。

调参建议：

- 不要无限增加 horizon / samples；
- 先保持 horizon=15、num_samples=200 量级；
- 逐步调：
  - feature radius；
  - cost weights；
  - merge distance；
  - decay；
  - escape prefix length；
  - final anti-spin threshold。

### 12.3 中期：固定 benchmark

至少固定 3-4 个 scene：

- `simple`
- `lab_complex`
- `narrow_corridor`
- `u_trap_long_board`

每个 scene 做：

- memory off；
- memory on；
- same seed；
- same horizon；
- same samples；
- success；
- final_goal_distance；
- stuck_steps；
- spin_steps；
- mean_v；
- mean_abs_omega；
- average_planner_compute_ms；
- collision；
- trajectory plot。

### 12.4 中期：MuJoCo 与实车对比

目标：

- MuJoCo synthetic LaserScan 与真实 `/scan` 的差异；
- local_obstacle_layer 在真实长条障碍上的输出；
- scan_guard hard safety 是否一致；
- memory feature 是否在真实车上也能捕捉低进展区域。

### 12.5 长期：论文 / 报告方向

当前项目可能形成的主线：

> 面向局部可见障碍和移动机器人实车部署的 Memory-Augmented MPPI：利用历史低进展与旋转陷阱构建软记忆场，在不破坏硬安全仲裁的前提下减少局部失败重复进入。

可能贡献点：

1. 从 LaserScan 到 local obstacle representation 的 MPPI 实车闭环；
2. 长条障碍 line surface + representative circles；
3. scan_guard 与 MPPI planner 的安全仲裁结构；
4. Memory-Augmented Potential Field 作为 rollout cost；
5. MuJoCo synthetic scan live sim 复现实车感知链路；
6. memory on/off 消融证明局部失败减少。

---

## 13. 新 Project 必须遵守的硬边界

以后让 ChatGPT 继续改代码时，一定要重复这些边界：

1. 不要把项目退回早期二维 MPPI demo。
2. 不要重写整个 MPPI 框架。
3. 优先复用已有 helper：
   - `sample_control_sequences`
   - `rollout_control_sequence`
   - `trajectory_cost`
   - `compute_weights`
   - `weighted_update_sequence`
   - `shift_sequence`
   - `initialize_control_sequence`
4. 不要改 goal：
   - 仿真 goal 固定为 `(3.0, 3.0)`。
5. 不要关闭 scan_guard。
6. 不要改实车 YAML goal / scan_guard。
7. 不要大改 `mppi_ros_adapter_skeleton.py`。
8. 不要破坏 `mppi_hardware_bridge/scripts/` 里的实车 bridge 主逻辑。
9. 不要修改旧 MuJoCo XML，除非明确新增专用 XML。
10. 不要为了避障效果盲目增加 horizon / num_samples。
11. 不要让 MPPI 直接偷看全局 obstacle primitives 作为主规划输入。
12. MuJoCo live sim 的 planner obstacles 主路径应来自：
    - MuJoCo geoms；
    - raycast_laserscan；
    - scan_guard；
    - local_obstacle_layer。
13. 不要 git add / commit / push，除非用户明确要求。
14. 如果要 stage，必须显式列文件，不要 `git add .`。

---

## 14. 给新 ChatGPT Project 的启动提示词

下面这段可以直接复制到新 Project 作为第一条消息：

```text
你现在接手一个科研项目：mobile-robot-mppi-study。

项目目标：
研究移动机器人局部避障中的 MPPI 控制，并从二维 baseline 推进到 MuJoCo live simulation、synthetic LaserScan、scan_guard、local_obstacle_layer、ROS 实车 bridge 和 Memory-Augmented MPPI。

当前阶段不是早期二维 demo。
当前阶段也不是重写 MPPI。
当前阶段是：
1. 稳定 MuJoCo live simulation；
2. 让 MuJoCo world -> raycast LaserScan -> scan_guard -> local_obstacle_layer -> MPPI + memory -> velocity command -> MuJoCo motion 形成真实闭环；
3. 用 memory on/off 消融验证 memory field 是否减少 stuck / spin / local trap。

重要文件：
- src/planners/mppi_mujoco_receding_horizon_experiment.py
- mppi_hardware_bridge/scripts/scan_guard.py
- mppi_hardware_bridge/scripts/local_obstacle_layer.py
- mppi_hardware_bridge/scripts/mppi_memory_field.py
- mppi_hardware_bridge/scripts/mppi_planner_bridge.py
- mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py
- experiments/mujoco_memory_mppi_live_sim.py
- experiments/mujoco_memory_mppi_ablation.py

当前本地最新分支：
mujoco-memory-bridge-ablation

注意：
GitHub 上可能还看不到本地未提交的新脚本：
- experiments/mujoco_memory_mppi_live_sim.py
- experiments/mujoco_memory_mppi_ablation.py
请以我给你的同步文档为当前最新状态。

硬边界：
1. 不要改 goal，保持 GOAL=(3.0,3.0)。
2. 不要关闭 scan_guard。
3. 不要大改 mppi_ros_adapter_skeleton.py。
4. 不要破坏实车 bridge 主逻辑。
5. 不要重写 MPPI 框架。
6. 优先复用已有 MPPI helper。
7. 不要让 planner 直接偷看全局障碍，planner_obstacles 应来自 LaserScan -> local_obstacle_layer。
8. 不要随便 git add / commit / push。

请先阅读同步文档和代码地图，然后帮我继续：
1. 稳定 lab_complex 场景；
2. 调整 planner obstacle filtering；
3. 调整 memory cost 和 escape bias；
4. 形成 memory on/off benchmark；
5. 最后再整理组会图表和实验结论。
```

---

## 15. 后续常用提示词模板

### 15.1 让 ChatGPT 做代码修改前

```text
先运行：
git branch --show-current
git status --short

确认当前分支是 mujoco-memory-bridge-ablation。
不要 git add。
不要 git commit。
不要 git push。

这次只修改 experiments/mujoco_memory_mppi_live_sim.py。
不要改 goal。
不要改实车 bridge。
不要改 mppi_ros_adapter_skeleton.py。
不要重写 MPPI。
```

### 15.2 调 MuJoCo live sim

```text
当前主脚本是 experiments/mujoco_memory_mppi_live_sim.py。
请保持 MuJoCo live simulation 主线：
MuJoCo geoms -> raycast_laserscan -> scan_guard -> local_obstacle_layer -> MPPI + memory -> env.step -> viewer overlay。

问题现象是：
小车在 lab_complex 中重复转圈 / 黄色 planner obstacles 过密 / memory feature 生成但没有明显逃逸。

请只做小范围修改：
1. filtering；
2. memory cost；
3. escape bias；
4. debug logging。
```

### 15.3 做 ablation

```text
请使用 experiments/mujoco_memory_mppi_ablation.py 做 memory on/off 消融。
默认 horizon=15、num_samples=200，不要靠无限加样本解决。
输出 summary CSV、trajectory CSV、PNG/GIF。
重点比较：
success、final_goal_distance、stuck_steps、spin_steps、mean_abs_omega、mean_v、average_planner_compute_ms。
不要夸大 memory 效果，以实际输出为准。
```

### 15.4 做组会汇报

```text
请基于当前代码和实验结果整理组会汇报材料。
主线是：
二维 MPPI baseline -> MuJoCo live sim -> synthetic LaserScan -> scan_guard/local_obstacle_layer -> MPPI + Memory -> 实车 bridge。

不要只讲公式。
要展示系统闭环、viewer 可视化、实车链路、当前问题和下一步计划。
```

---

## 16. 当前最值得保留的科研叙事

如果后面写报告 / PPT，可以这样讲：

1. 早期实现了 vanilla MPPI，掌握 sampling-based MPC 主循环。
2. 发现 dense / narrow 场景下，单纯调 cost 或 horizon 不能完全解决局部失败。
3. 引入 MuJoCo，让机器人运动和场景几何可视化。
4. 进一步引入 synthetic LaserScan，使仿真输入接近实车 `/scan`。
5. 加入 scan_guard，保证 hard safety。
6. 加入 local_obstacle_layer，使 MPPI 看到的是局部障碍表达，而不是全局地图。
7. 实车 bridge 已经跑通过 odom + scan + MPPI + safety + cmd_vel 链路。
8. 当前进一步引入 memory field，记录历史 stuck / spin / low-progress 区域。
9. Memory 通过 cost landscape 和轻量 escape bias 影响 MPPI，而不是硬接管控制。
10. 下一步用固定 MuJoCo benchmark + 实车复现验证 memory 是否减少重复局部失败。

---

## 17. 当前风险点

### 17.1 环境风险

- 新 Project 如果只看 GitHub，会漏掉本地未提交脚本。
- 当前 WSL / Codex 环境可能没有 `mujoco` 包。
- Windows Git 可能和 WSL Git 使用不同 SSH key。
- PyCharm interpreter 与命令行 Python 可能不同。

### 17.2 算法风险

- Memory feature 生成不等于一定有效。
- 如果 memory cost 太弱，trajectory 仍会绕着紫色点打转。
- 如果 memory cost 太强，可能误伤可行路径。
- planner obstacles 太密会让 MPPI 过保守。
- scan_guard 太强会压掉所有前进速度。
- final control 与 proposed control 差异大时，不能只看 MPPI predicted trajectory。

### 17.3 工程风险

- 不要把实车 bridge 和 MuJoCo live sim 混成一坨。
- 不要为了可视化美观改变算法行为。
- 不要修改旧 XML 破坏早期实验。
- 不要把结果文件、缓存、PPT 临时解包内容提交进 git。

---

## 18. 下一步推荐执行顺序

### Step 1：保存当前本地进度

人工检查 diff：

```bash
git status --short
git diff -- mppi_hardware_bridge/scripts/mppi_memory_field.py
git diff -- src/planners/mppi_mujoco_receding_horizon_experiment.py
```

再检查新增脚本内容。

如果确认要同步到 GitHub，建议只 stage 指定文件：

```bash
git add experiments/mujoco_memory_mppi_live_sim.py
git add experiments/mujoco_memory_mppi_ablation.py
git add mppi_hardware_bridge/scripts/mppi_memory_field.py
git add src/planners/mppi_mujoco_receding_horizon_experiment.py
git add docs/project_recovery_sync_2026-05-28.md
```

不要使用：

```bash
git add .
```

### Step 2：确认 MuJoCo 环境

```bash
python3 -c "import mujoco; print(mujoco.__version__)"
```

如果失败，先安装 / 修复解释器环境。

### Step 3：跑 no-viewer smoke

```bash
python3 experiments/mujoco_memory_mppi_live_sim.py \
  --scene lab_complex \
  --max-steps 60 \
  --no-viewer
```

### Step 4：跑 viewer

```bash
python3 experiments/mujoco_memory_mppi_live_sim.py \
  --scene lab_complex \
  --max-steps 150
```

观察：

- 小车是否到 goal；
- 是否还在紫色 memory 点附近原地转；
- 黄色 planner obstacles 是否过密；
- scan_guard 是否频繁 soft block；
- escape_active 是否触发；
- final_v 是否长期被压到 0。

### Step 5：做固定 benchmark

每个 scene 跑：

```bash
--memory-disable
--memory-enable
```

记录：

- success；
- final distance；
- stuck / spin；
- compute time；
- trajectory。

### Step 6：整理组会材料

重点展示：

- viewer 截图；
- scan rays / hit points / planner obstacles；
- predicted trajectory；
- memory features；
- memory on/off 对比表；
- 实车 bridge 链路图；
- 当前失败案例和下一步计划。

---

## 19. 最后一句状态总结

这个项目当前已经完成了从“二维 MPPI 算法原型”到“MuJoCo + synthetic LaserScan + scan_guard + local_obstacle_layer + MPPI + memory + viewer overlay”的闭环迁移，并且实车 ROS bridge 也已经具备 odom/scan/cmd_vel 和安全仲裁基础。后续工作的重点不应再是重写 MPPI，而是稳定 MuJoCo live benchmark、验证 Memory-Augmented MPPI 对局部失败的改善，并把同一套感知-规划-执行逻辑迁移回实车做对比。

---

## 20. 扩展交接：新 Project 应如何理解这个项目

如果新的 ChatGPT Project 要真正接上这个项目，不能只把它理解成“一个 MPPI 算法仓库”。更准确地说，这个仓库记录的是一个从学习、复现、仿真、实车桥接，到提出轻量改进方向的连续科研过程。它的价值不只在某一个脚本能跑，而在每一层系统逐步叠起来之后，已经形成了一个比较完整的移动机器人局部导航实验平台。

早期二维代码的意义是帮助你理解 MPPI 的基本循环：采样控制序列、rollout、计算轨迹代价、用指数权重更新控制序列、只执行第一步、再进行 receding horizon。这个阶段的代码不应该被贬低，因为它是后续所有仿真和实车桥接的算法底座。但是新 Project 不能停在这个阶段，否则它会反复建议“先搭一个二维 demo”，这会浪费时间。

MuJoCo 阶段的意义是把 planner 从静态画图推进到可视化执行平台。机器人在 viewer 中运动，障碍物有几何体，预测轨迹可以实时显示。这个阶段开始暴露真实闭环里的问题，例如控制序列看起来能绕，但执行第一步后又会进入新的局部不稳定状态；又例如局部障碍太密时，MPPI 会变得非常保守；再例如边界、近障、安全限制会把理论上可行的采样轨迹在执行层压掉。

ROS hardware bridge 阶段的意义是把 MPPI 放进真实小车控制链路。这里最重要的不是让小车“偶尔动起来”，而是把 odom、scan、planner、safety、cmd_vel 串起来，并建立一套不会因为 planner 发出危险速度而撞车的仲裁系统。因此 scan_guard、control_adapter、mppi_ros_adapter_skeleton、mppi_planner_bridge 这些文件是项目里非常关键的工程资产。新 Project 千万不能把它们当作“杂乱脚本”随便重写。

Memory-Augmented MPPI 阶段是当前科研方向。它不是单纯为了加一个 fancy 模块，而是针对实际调试中反复出现的失败模式：机器人在局部障碍附近低进展、原地转圈、重复尝试相同绕行方向、被安全层压住速度之后又重新进入同一个坏区域。memory field 的研究问题是：能否把这些失败经验以软代价的形式写入 MPPI rollout cost，使得后续采样轨迹更不愿意重新进入这些区域。

新 Project 读这个项目时，应该按如下层次理解：

```text
第一层：MPPI 数学和代码骨架
第二层：二维 / MuJoCo 运动学 rollout
第三层：障碍代价、边界代价、控制代价、目标代价
第四层：LaserScan 局部感知和 scan_guard 硬安全
第五层：local_obstacle_layer 将 scan 转为 planner obstacles
第六层：ROS 实车 bridge 和控制仲裁
第七层：Memory field 作为历史失败区域软代价
第八层：MuJoCo live sim 复现实车感知-规划-执行链路
第九层：消融实验、指标、组会汇报、后续论文叙事
```

如果缺少前几层，新 Project 可能会提出错误建议。例如只看 memory field，就可能建议“直接让 memory 控制机器人远离坏点”，这违反当前设计；只看 MuJoCo，就可能建议“把全局障碍直接塞给 MPPI”，这会绕开 LaserScan/local obstacle layer 主线；只看实车，就可能建议“关闭 scan_guard 先跑起来”，这会破坏硬安全边界。

---

## 21. 研究动机的细化表达

这个项目可以从一个更清楚的科研动机来解释：采样式 MPC / MPPI 很适合处理非线性控制和复杂代价函数，但是在移动机器人局部避障任务中，普通 MPPI 常见几个问题。

第一，局部感知下的可行空间是不完整的。机器人并不知道全局地图，只能看到 LaserScan 当前扫到的局部障碍。即使仿真世界里所有障碍都已知，实车运行时 planner 也不应该直接使用全局真值。这样一来，MPPI 在 rollout 时面对的是局部、近似、可能过密或过稀的障碍表达。

第二，采样分布有限。真实系统为了实时性，不能无限增加 sample 数，也不能无限增加 horizon。当前默认量级是 horizon=15、num_samples=200，这在实时控制里比较轻量，但也意味着如果采样云没有覆盖到正确绕行方向，MPPI 会倾向于选择局部看起来代价最低但长期进展差的控制。

第三，安全层会改变 planner 的实际执行结果。MPPI proposed_control 可能认为可以前进或转弯，但 scan_guard、front_soft_block、hard_stop、速度限幅、smoothing 等执行层逻辑会改变 final_control。如果只看 MPPI predicted trajectory，而不看 final control，就会误判失败原因。

第四，局部失败有历史结构。小车不是每一步独立失败，而是会在某些空间区域反复出现类似失败：靠近长条障碍的端点、U 型陷阱入口、窄通道转角、障碍密集区前沿、scan_guard 刚好持续 soft block 的位置。这些区域不是单帧 scan 就能完全表达的，它们需要从执行历史中总结。

第五，实车和仿真之间需要一致的信息流。如果 MuJoCo 中直接把障碍真值给 planner，而实车只能给 LaserScan，那么仿真调出来的行为可能不能迁移到真实车。因此当前 MuJoCo live sim 要坚持 synthetic LaserScan -> scan_guard -> local_obstacle_layer -> planner_obstacles 的路径。

在这个动机下，Memory-Augmented MPPI 可以表述为：在不改变硬安全层、不直接接管控制、不要求全局地图的前提下，利用执行历史识别局部低进展区域，将其作为软记忆场加入 MPPI rollout cost，并在检测到重复 spin/stuck 时对 nominal sequence 做轻量逃逸偏置，从而提高局部陷阱环境中的稳定性。

---

## 22. MPPI 算法层的详细交接

MPPI 的核心循环可以被理解成一个“以控制序列为随机变量”的优化过程。它不是直接对单步速度做贪心搜索，而是每次采样一批未来控制序列，预测它们在 horizon 内的轨迹，再根据轨迹总代价计算权重，最后用加权平均得到更新后的控制序列。

本项目中典型控制为：

```text
u = (v, omega)
```

其中 v 是线速度，omega 是角速度。默认实车 / 仿真中不允许倒车，因此 v_min=0。这样做的原因是 E1/E2 小车和室内避障任务中，倒车会带来更复杂的安全问题；同时 LaserScan 前方安全检测和局部障碍建模主要围绕前进方向展开。允许倒车不是绝对不能做，但当前阶段不要引入，否则会改变很多安全假设。

每条采样控制序列形如：

```text
U_i = [u_0, u_1, ..., u_{H-1}]
```

rollout 时从当前 state 开始，用运动学模型展开：

```text
x_{t+1} = x_t + v_t * cos(theta_t) * dt
y_{t+1} = y_t + v_t * sin(theta_t) * dt
theta_{t+1} = wrap(theta_t + omega_t * dt)
```

MuJoCo live sim 中真实执行由 MuJoCo qvel/mj_step 完成，但 MPPI 的预测 rollout 仍然使用轻量平面运动学。这是合理的，因为当前任务不是研究复杂动力学，而是研究局部避障控制与感知闭环。不要把问题升级成全动力学 MPC，否则会偏离当前科研阶段。

trajectory_cost 通常包括：

- goal distance cost；
- obstacle proximity cost；
- collision penalty；
- control effort cost；
- angular / spin cost；
- boundary cost；
- terminal cost；
- memory cost（在当前扩展中加入）。

MPPI 权重通常形如：

```text
w_i = exp(-(S_i - min(S)) / temperature)
```

temperature 控制 cost 差异对权重的影响。temperature 太小会导致少数最低 cost trajectory 主导更新，可能行为激进或抖动；temperature 太大则会让高低 cost 区分不明显，更新变钝。当前默认 temperature=8.0，memory field 可以根据附近记忆点做有限 scale，但必须 clamp，避免无限放大。

weighted_update_sequence 的意义是得到一条“在当前采样分布下更优”的 nominal sequence。下一步只执行 updated_sequence[0]，然后把序列 shift，作为下一轮 warm start。这就是 receding horizon。

新 Project 在改 MPPI 时要注意：不要轻易改 helper 函数签名，因为实车 bridge、ablation、live sim 都可能在复用它们。可以在调用侧写 adapter，但不要为了一个实验入口破坏全局 API。

---

## 23. cost landscape 与局部失败的解释

项目中很多行为问题都可以从 cost landscape 的角度理解。MPPI 不是“知道正确路径再执行”，它只是从采样轨迹中选择代价相对低的一批。如果障碍表达、采样分布和安全限制共同制造了一个局部低代价盆地，机器人就可能陷进去。

例如在 U 型陷阱前，短 horizon 内进入 U 型内部可能看起来离 goal 更近，障碍 cost 也暂时不高。但进入后，前方被墙挡住，scan_guard 开始 soft block 或 hard stop，MPPI 又采样到原地转向、慢速爬行等局部动作。这些动作单步看起来安全，却不能让 goal distance 明显下降。

再例如在长条障碍附近，LaserScan hit points 可能沿墙面产生许多点。如果 local_obstacle_layer 没有很好地压缩这些点，planner obstacles 会过密，MPPI 的 obstacle cost 会把大片空间都视为高代价区。此时机器人可能选择原地转向，因为前进方向的 rollout 都被高 obstacle cost 惩罚。

Memory field 试图补充的是“历史失败区域”这层信息。普通 obstacle cost 只告诉 MPPI 哪里有障碍；memory cost 还告诉 MPPI：某些没有直接碰撞的区域，过去执行后曾经导致低进展、打转或 recovery。这个信息不是几何障碍本身，而是行为经验。

但是 memory cost 也有风险。如果 memory feature 半径太大、权重太高，会把可行通路也惩罚掉；如果衰减太慢，早期调试中偶然形成的 feature 会长期影响行为；如果 feature 合并距离太小，可能堆出一堆紫色点，反而让 cost landscape 更混乱。因此 memory 参数必须和场景、planner obstacle density、scan_guard 阈值一起看。

当前设计中把 memory 作为 soft cost，而不是 hard forbidden zone，是因为真实机器人局部导航中很多失败区域不是绝对不能进入，而是“不应该反复以同样方式进入”。在某些情况下，机器人可能必须短暂经过 memory feature 附近才能绕出去。如果硬禁止，反而可能导致无路可走。

---

## 24. scan_guard 的详细角色

scan_guard 是实车安全层的基础，也是 MuJoCo live sim 必须复用的模块。它的任务不是规划，而是根据 LaserScan 判断前方是否存在需要减速或停车的风险。

scan_guard 通常会做：

- 将 ranges 转换成 base frame 下的点；
- 按角度筛选 front cone；
- 判断前方最近障碍距离；
- 判断 near body 是否有危险点；
- 根据阈值输出 emergency_stop；
- 根据距离输出 should_slow_down；
- 给出 front_stop_mode 和 reason；
- 提供 front_points、side_points、near_body_points 等 debug 信息。

在实车中，scan_guard 是不能关闭的。原因很简单：MPPI 是 sampling-based planner，它可以因为采样不足、cost 参数不佳、感知错误而给出不安全控制。scan_guard 是兜底的硬安全层。如果为了让实验“看起来跑得起来”而关闭 scan_guard，那么得到的结果没有实车意义。

在 MuJoCo live sim 中接入 scan_guard 的意义有两层。第一，让仿真控制链路更接近实车；第二，让 viewer 中的失败更可解释。如果 final_v 被压到 0，可以看 scan_reason 和 front_stop_mode 判断是 planner 自己不想走，还是 safety 把它拦住了。

新 Project 后续调试时要特别注意 proposed_control 与 final_control 的区别：

```text
proposed_control: MPPI 认为应该执行的速度
final_control: scan_guard / safety / anti-spin 之后真正执行的速度
```

如果 proposed_v 较大但 final_v 长期为 0，说明 safety 或 arbitration 在压制前进。如果 proposed_v 本身就很小，说明 MPPI cost landscape 或采样分布导致 planner 不敢走。两个问题的解法完全不同。

scan_guard 不能被 memory 绕过。即使 escape_active=True，如果 emergency_stop=True，也不能强行前进。memory escape 只能在安全层允许的范围内给 MPPI 一个离开坏区域的倾向。

---

## 25. local_obstacle_layer 的详细角色

local_obstacle_layer 是连接 LaserScan 和 MPPI obstacle cost 的关键桥梁。真实 LaserScan 给的是一组极坐标 ranges，而 MPPI trajectory_cost 更容易使用世界坐标下的圆形障碍或代表障碍。因此需要一个中间层，把 scan 点转成 planner 可用的局部障碍表达。

早期最简单的做法是把每个 scan hit point 都变成一个小圆形障碍。这很直观，但问题是点太多，尤其是长条墙面会产生大量密集点。如果每个点都进入 MPPI cost，planner 会看到一堵“由很多圆重叠组成的厚墙”，局部可行空间会被过度压缩。

因此 local_obstacle_layer 中加入了几何特征处理：

- scan ranges smoothing；
- ordered local points；
- edge point detection；
- neighbor distance segmentation；
- line fitting；
- circle fitting；
- line surface segment info；
- representative circles；
- long obstacle compression；
- local to experiment/world frame conversion。

对长条障碍来说，理想输出不是几十个点，而是少量能代表墙面/端点/安全距离的圆形障碍。这样 MPPI cost 既能知道那里有墙，又不会被过密点云压死。

MuJoCo live sim 中必须保持主路径：

```text
MuJoCo geoms -> mj_ray -> LaserScan-like scan -> local_obstacle_layer -> planner_obstacles
```

scene primitives 可以用于构建 MuJoCo model、collision reference 和 debug，但不应该成为 MPPI 的主 obstacle input。否则仿真会变成“全局地图规划”，不再对应实车 `/scan`。

当前新增的 planner obstacle filtering 是在 local_obstacle_layer 之后做的轻量保护。它不是替代 local_obstacle_layer，而是在局部输出过密时，优先保留近处、前方、较大 representative circle，丢弃后方较远小点。这样既保留安全相关障碍，又减少 MPPI 过度保守。

后续如果继续优化 local_obstacle_layer，应优先观察 viewer 中：

- 红/橙 hit points 是否合理；
- 黄色 planner obstacles 是否明显少于 hit points；
- 长条障碍是否被压缩成少量代表点；
- 机器人后方障碍是否还在影响前进；
- 窄通道两侧是否被表示得过宽；
- U 型陷阱入口是否产生误导性封堵。

---

## 26. MuJoCo live sim 的详细运行逻辑

当前 live sim 的价值在于它不是简单地“把离线轨迹画成 GIF”。它是实时闭环：机器人状态来自 MuJoCo，扫描来自 MuJoCo ray casting，控制执行通过 MuJoCo physics step，viewer 中显示的是当前闭环状态。

一个典型 step 可以拆得更细：

1. 从 MuJoCo qpos 读取机器人位姿。
2. 根据 robot pose 发出 181 条或类似数量的射线。
3. 对每条射线调用 MuJoCo ray casting。
4. 排除 robot 自身 geom，得到最近命中距离。
5. 组装 LaserScan-like dict。
6. 把 scan 送入 scan_guard。
7. 把 scan 送入 local_obstacle_layer。
8. 将 local obstacle 输出转换成 MPPI planner obstacles。
9. 对 planner obstacles 做数量和优先级过滤。
10. 根据 recent history 判断 repeated_spin / stuck_low_progress。
11. 如果需要，基于 nearest memory feature 对 nominal sequence 做 escape bias。
12. 调用 sample_control_sequences。
13. 对每条 sequence rollout。
14. 计算 base trajectory cost。
15. 叠加 memory_cost_for_trajectory。
16. 用 effective temperature 算 weights。
17. 生成 updated_sequence。
18. 取第一步作为 proposed_control。
19. 经过 scan_guard safety 得到 final_control。
20. 经过 anti-spin final control 修正。
21. 将 final_control 转成 qvel。
22. 调用 mj_step 若干 substeps。
23. 用新 state 更新 memory field。
24. 更新 executed trajectory。
25. 更新 best predicted trajectory。
26. 更新 viewer overlay。
27. render。
28. shift nominal sequence。

这个流程比 offline ablation 更接近真实系统，因为每一步都受到感知、规划、执行和安全层的影响。

如果 live sim 表现不好，不应该马上修改 MPPI 核心。应该先按以下顺序定位：

1. 机器人 state 是否从 MuJoCo 正确读取；
2. raycast 是否命中正确 geoms；
3. scan hit points 是否在 viewer 中与障碍物重合；
4. scan_guard 是否把前方误判为危险；
5. local_obstacle_layer 是否输出过多/过少 planner obstacles；
6. trajectory_cost 是否因为障碍 cost 过强导致所有前进轨迹高代价；
7. memory_cost 是否过强或过弱；
8. proposed_control 是否合理；
9. final_control 是否被 safety 压掉；
10. qvel 写入是否按 yaw 正确转换。

---

## 27. 多场景系统的详细意图

多场景不是为了“地图更好看”，而是为了把不同 failure mode 拆开测试。一个场景里同时存在太多问题时，无法判断失败原因。

`simple` 场景用于回归。它保留较早的长条障碍 + U 型结构 + 圆柱组合。如果一个改动导致 simple 都完全跑不动，说明改动可能破坏了基本闭环。

`lab_complex` 是默认综合场景。它不应该是迷宫，也不应该完全堵死路线。它的目的是模拟实验室里分散摆放的纸板、墙边、桌腿、柱子等局部障碍。小车需要绕行，但应该存在明显可行路径。这个场景用于日常观察 MPPI + memory 是否稳定。

`narrow_corridor` 专门测试长条障碍和通道。这个场景重点不是 memory，而是 LaserScan 局部障碍表达。理想情况下，长条墙面不应该变成大量黄色点把通道堵死，而应该被压缩成合理数量的 representative circles。

`u_trap_long_board` 专门测试 memory。它故意制造一个容易进入但不容易出来的局部区域。目标是观察：没有 memory 时是否反复低进展；有 memory 后是否生成紫色 feature；feature 生成后 predicted trajectory 是否逐渐避开；escape bias 是否在连续 spin/stuck 时触发。

后续可以新增场景，但不要无限增加。科研上更重要的是固定少数 benchmark，反复跑同一套指标。每次改 memory 参数后都换地图，会让结果不可比较。

场景设计原则：

- goal 保持 `(3.0, 3.0)`；
- start 保持 `(0.0, 0.0, 0.0)`；
- bounds 保持统一；
- 不要把障碍物密集堆在 goal 附近；
- 不要让机器人起点附近立即 hard stop；
- 要保留至少一条可通过路径；
- 长条障碍要能测试 line/circle compression；
- U 型陷阱要能诱发局部失败，但不能完全封死；
- 圆柱簇要稀疏，不要形成不可解释的障碍云。

---

## 28. 当前调试日志应该如何读

live sim 的日志不是装饰，它是定位问题的主要工具。每 10 步打印的信息应该能回答下面这些问题：

机器人是否在接近目标：

```text
goal_distance 是否下降
```

前方是否安全：

```text
min_front_range
front_stop_mode
scan_reason
```

local obstacle 是否过密：

```text
planner_obstacle_count
```

memory 是否参与：

```text
memory_feature_count
memory_cost
nearest_memory_distance
effective_temperature
```

是否检测到局部失败：

```text
repeated_spin
stuck_low_progress
escape_active
```

MPPI 和最终执行是否一致：

```text
proposed_v / proposed_omega
final_v / final_omega
```

如果 `proposed_v > 0.15` 但 `final_v = 0`，优先看 scan_guard。如果 `proposed_v` 本来就接近 0，优先看 trajectory_cost、planner_obstacles、memory_cost 和 sampling。如果 `planner_obstacle_count` 很高，优先看 local_obstacle_layer 和 filter。如果 `memory_feature_count` 增长很快，说明 stuck/spin 检测太敏感或场景本身太卡。如果 `escape_active=True` 但机器人仍不走，要看 min_front_range 是否允许前进，以及 final control 是否仍被压掉。

如果 viewer 中青色 predicted trajectory 很漂亮，但橙色 executed trajectory 不跟随，说明执行层改变了控制或者 MuJoCo qvel/state 有问题。如果青色 trajectory 本身绕圈，说明 planner 层已经陷入局部代价盆地。如果黄色 planner obstacles 明显偏离真实障碍，说明 scan/local layer 转换有问题。

---

## 29. 实车 bridge 的工程细节思路

实车 bridge 是项目中最容易被新 Project 误改坏的部分。它不只是“把 MPPI 输出转成 cmd_vel”。它承担了多个实际机器人运行中必须有的功能。

第一，坐标转换。真实 `/odom` 的坐标系、实验中设定的 goal、LaserScan base frame、MPPI rollout 使用的 experiment frame 之间必须一致。`frame_transform.py` 和 adapter 中的状态转换逻辑是为了解决这个问题。后续如果机器人走反方向或 goal tracking 异常，不要直接改 planner，先查坐标。

第二，scan 安全。真实小车对近障、前方障碍、侧向障碍的容错很低。scan_guard 负责把 scan 转成安全状态。它输出的不只是 emergency_stop，还包括 slow scale、front/side/near body points、reason。实车调试中应该把这些信息打印出来，否则不知道车停下是因为 planner 还是 safety。

第三，local obstacle。实车 LaserScan 点云会比 MuJoCo 更脏，可能有噪声、缺失、反光、边缘跳变。local_obstacle_layer 的几何处理是为了让 MPPI 不直接面对原始点云。后续如果实车长条纸板效果不好，优先调 local_obstacle_layer 的长条压缩，而不是简单增大 obstacle radius。

第四，控制仲裁。`mppi_ros_adapter_skeleton.py` 中的状态机和 smoothing 逻辑是多轮实车调试积累出来的。它处理 CLEAR、APPROACH_SLOW、CREEP_ESCAPE、HARD_STOP_RECOVERY、GOAL_REACQUIRE 等状态。新 Project 不应该为了“代码简洁”删掉这些状态。

第五，实时性。MPPI 计算如果偶发超时，实车不能继续盲目执行旧速度。已有代码中有 profile degrade、realtime diagnostics 等思路。后续做 memory 或更多 samples 时，一定要看 compute time。

第六，原厂控制权。E1/E2 小车可能有自带 web/server/control 逻辑。关闭它可能会连带关闭 roscore、Lidar、Odom。抢控制权要谨慎，最好先理解 launch 链和驱动层，不要直接删启动脚本。

---

## 30. 实车问题的排查路线

针对 PPT 中提到的 WebServer.sh / 原厂控制冲突，后续建议按保守路线排查。

第一步，只观察，不修改：

```bash
ps aux | grep ros
rostopic list
rostopic echo /odom
rostopic echo /scan
rostopic echo /cmd_vel
rosnode list
```

目标是弄清楚：当前谁在发布 `/cmd_vel`，谁在发布 `/odom`，谁在发布 `/scan`，roscore 是谁拉起来的。

第二步，记录原厂启动脚本：

```bash
cat WebServer.sh
ls -l /dev
ls -l /dev/dashgo
```

不要一上来修改。先复制内容，搞清楚里面是否包含 roslaunch、udev、串口权限、lidar driver、base driver。

第三步，手动启动基础驱动。目标不是关掉所有原厂东西，而是找到最小基础链路：

```text
roscore
base driver
lidar driver
odom publisher
scan publisher
```

只有当 `/odom` 和 `/scan` 稳定存在后，才运行 MPPI bridge。

第四步，处理 cmd_vel 冲突。可以用：

- 查所有发布 `/cmd_vel` 的节点；
- 改 MPPI 发布到高优先级 mux input；
- 或关闭原厂上层演示控制节点但保留底层驱动；
- 不要关闭整个启动链导致传感器也没了。

第五步，安全验证。先不用 MPPI，跑 `test_cmd_vel.py` 这种简单脚本确认：

- 速度方向；
- 角速度方向；
- 零速度是否能停；
- 小速度是否平滑；
- emergency stop 是否生效。

第六步，再接 MPPI。初始参数保守：

- 低 v_max；
- 小 omega_max；
- goal 近一点；
- scan_guard 开启；
- 有人手动准备急停；
- 先空场，再单障碍，再长条障碍，再复杂场景。

---

## 31. Offline ablation 应该怎样做才有说服力

memory on/off 消融不能只看“某一次跑得更好”。采样式方法有随机性，场景也会影响结果。要让组会或论文讨论更可信，至少要有固定协议。

固定项：

- scene；
- start；
- goal；
- horizon；
- num_samples；
- dt；
- temperature；
- control limits；
- random seed；
- max_steps；
- obstacle representation；
- memory 参数。

对照项：

- memory disabled；
- memory enabled；
- 其他参数尽量相同。

核心指标：

- success；
- final_goal_distance；
- total_steps；
- collision；
- min_obstacle_distance；
- stuck_steps；
- spin_steps；
- mean_v；
- mean_abs_omega；
- average_planner_compute_ms；
- memory_feature_count_final。

轨迹指标：

- 是否进入 U trap；
- 是否在同一区域重复回绕；
- 是否在 memory feature 生成后改变路径；
- 是否出现长时间原地转；
- 是否被 scan_guard 频繁 hard/soft block。

图像输出：

- obstacles；
- executed trajectory；
- start / goal；
- memory features；
- 如果可能，加 predicted trajectory snapshots。

注意不要夸大 memory 效果。如果 memory on 只是减少 spin 但 final distance 没明显改善，就如实说：“当前 memory 在该场景降低了原地旋转次数，但尚未稳定提升成功率。”这比硬说“显著提升”更可靠。

如果 memory on 反而更差，也有科研价值。可能原因包括：

- memory cost 过强；
- feature 半径太大；
- feature 生成太早；
- 误把可行通路标成失败区域；
- escape bias 方向和 goal direction 冲突；
- scan_guard 已经把前进压掉，memory 无法生效。

---

## 32. 组会汇报叙事建议

组会汇报不要从代码细节开始，而要从系统推进主线开始。一个清晰结构是：

第一部分：上次到这次发生了什么变化。强调从二维 MPPI 推进到 MuJoCo live sim 和实车 bridge，不再只是算法 demo。

第二部分：系统架构。用一张图展示：

```text
MuJoCo / Real Robot
  -> LaserScan
  -> scan_guard
  -> local_obstacle_layer
  -> MPPI + Memory
  -> safety arbitration
  -> command execution
```

第三部分：MuJoCo viewer。展示截图并解释颜色：蓝色机器人、绿色目标、红色障碍、黄色 planner obstacles、青色 predicted trajectory、橙色 executed trajectory、紫色 memory features。

第四部分：感知闭环。强调 planner obstacles 来自 synthetic LaserScan，不是直接使用全局地图。这是仿真向实车迁移的关键。

第五部分：Memory-Augmented MPPI。讲清楚 memory 不是硬控制，而是把历史失败区域加入 cost landscape。

第六部分：当前问题。诚实说明仍存在重复转圈、planner obstacles 过密、proposed/final control 差异等问题。

第七部分：已采取措施。说明 lab_complex 分散化、planner obstacle filtering、anti-spin/stuck escape bias。

第八部分：下一步。固定 benchmark、memory on/off 消融、compute time、实车复现。

不要把汇报做成“所有功能都完成了”。更好的表达是：系统闭环已经成立，下一步进入稳定性和指标化验证。

---

## 33. 新 Project 做代码 review 时要关注什么

如果让新 Project review 当前代码，不要让它只挑格式问题。应该让它优先看行为风险。

重点 review：

1. `mujoco_memory_mppi_live_sim.py` 中 planner 是否直接使用 scene primitives。若是，这是违背主线的。
2. raycast 是否排除了 robot 自身 geom。
3. scan angle / world angle / robot local frame 是否一致。
4. local_obstacle_layer 输出转 world frame 是否正确。
5. planner_obstacles filter 是否过度删除前方障碍。
6. scan_guard hard stop 是否永远优先。
7. escape bias 是否可能在 emergency_stop 时强行给 v。
8. memory_cost 是否对 disabled memory 返回 0。
9. effective_temperature 是否 clamp。
10. nominal_sequence shift 是否保持 horizon 长度。
11. final_control clamp 是否保持 v>=0。
12. qvel 地址是否通过 joint name 查询，而不是硬编码错。
13. viewer overlay 是否只影响显示，不改变算法。
14. CSV / results 是否不会被自动 git add。
15. 实车 bridge 文件是否被不必要改动。

不要过度关注：

- 变量名是否还能更漂亮；
- 是否能抽象更多类；
- 是否能把脚本拆成多个模块。

当前科研阶段优先级是行为正确、可视化可解释、实验可复现，而不是架构洁癖。

---

## 34. 常见失败现象与对应解释

### 34.1 小车原地转圈

可能原因：

- goal direction 和 obstacle cost 冲突；
- planner obstacles 太密；
- scan_guard soft block 压低 v；
- MPPI sampling 中低 v 高 omega 轨迹代价较低；
- memory feature 生成但 cost 太弱；
- final anti-spin 未触发；
- escape bias 方向被障碍挡住。

排查：

- 看 proposed_v 是否小；
- 看 final_v 是否被压；
- 看 planner_obstacle_count；
- 看 repeated_spin；
- 看 min_front_range；
- 看 memory_cost。

### 34.2 小车不动

可能原因：

- MuJoCo qvel 没写对；
- scan_guard emergency_stop；
- 起点离障碍太近；
- all sampled trajectories collision；
- final_control clamp 后为 0；
- viewer 运行但 mj_step 没推进。

排查：

- no-viewer 跑 5 步；
- 打印 state 是否变化；
- 打印 final_control；
- 检查 qvel address；
- 检查 robot contact。

### 34.3 黄色 planner obstacles 一大片

可能原因：

- local_obstacle_layer 点模式 fallback；
- line compression 未触发；
- scan downsample 太小；
- obstacle radius 太大；
- filter max_count 太高；
- 后方障碍未过滤。

排查：

- 看 layer debug mode；
- 看 hit points 和 planner obstacles 数量比；
- 切 narrow_corridor；
- 调 MAX_PLANNER_OBSTACLES；
- 调后方过滤条件。

### 34.4 memory feature 很多但没用

可能原因：

- memory cost 权重太低；
- feature radius 太小；
- feature 与 trajectory stride 不匹配；
- temperature scale 不明显；
- feature 位置不在实际失败中心；
- escape bias 没触发；
- nearest_memory_distance 太远；
- final control 被 safety 主导。

排查：

- 打印 memory_cost；
- 打印 feature_count；
- 打印 nearest_memory_distance；
- 看 predicted trajectory 是否穿过紫色点；
- 手动提高 memory weight 做极端测试。

### 34.5 memory 导致更差

可能原因：

- feature radius 太大；
- 在唯一通路上生成 feature；
- low-progress 检测太敏感；
- feature decay 太慢；
- escape direction 与可行通道相反；
- memory cost 与 obstacle cost 叠加过强。

处理：

- 降低 feature strength；
- 增大生成阈值；
- 缩小 radius；
- 调整 escape_dir goal 混合比例；
- 只在 repeated_spin 更明确时触发。

---

## 35. 代码修改策略：小步、可验证、可回退

这个项目已经跨越了算法、仿真、ROS、实车安全和汇报材料多个层面。后续改动一定要小步进行。

推荐一次只改一种东西：

- 只改场景；
- 只改 filtering；
- 只改 memory cost；
- 只改 escape bias；
- 只改日志；
- 只改 viewer overlay；
- 只改 CSV 输出。

不要一次同时改：

- 场景；
- cost；
- scan_guard；
- local_obstacle_layer；
- MPPI sampling；
- memory；
- qvel 执行。

否则一旦行为变差，无法知道原因。

每次修改后至少做：

```bash
python3 -m py_compile experiments/mujoco_memory_mppi_live_sim.py
python3 experiments/mujoco_memory_mppi_live_sim.py --scene lab_complex --max-steps 60 --no-viewer
```

如果 MuJoCo 环境不可用，也要至少保证 py_compile 通过，并在最终回复中明确说明运行失败是依赖问题不是语法问题。

如果改到了 memory helper：

```bash
python3 -m py_compile mppi_hardware_bridge/scripts/mppi_memory_field.py
```

如果改到了 planner helper：

```bash
python3 -m py_compile src/planners/mppi_mujoco_receding_horizon_experiment.py
```

如果改到了实车 bridge：

```bash
python3 -m py_compile mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py
```

但当前阶段尽量不要改实车 bridge 主逻辑。

---

## 36. 推荐的四周推进计划

### 第 1 周：稳定 MuJoCo live sim

目标：

- lab_complex 能稳定运行；
- viewer 中障碍、scan、planner obstacles、trajectory、memory 都清楚；
- repeated_spin / stuck_low_progress / escape_active 日志可信；
- no-viewer 和 viewer 都能跑。

任务：

- 检查 MuJoCo 环境；
- 录制或截图 viewer；
- 调 planner obstacle filtering；
- 调 lab_complex；
- 确认 `--scene` 切换正常。

### 第 2 周：Memory 参数和 failure case

目标：

- 在 u_trap_long_board 中稳定触发 memory feature；
- memory on 与 off 有可观察差异；
- 不追求百分百成功，先追求机制可解释。

任务：

- 调 feature_radius；
- 调 cost weights；
- 调 escape bias；
- 记录典型失败/成功轨迹；
- 做 3 个 seed 的小对比。

### 第 3 周：正式消融

目标：

- 固定 3-4 个场景；
- 每个场景 memory on/off；
- 输出 CSV / PNG / GIF；
- 形成表格。

任务：

- 统一参数；
- 跑多 episode；
- 统计指标；
- 整理图；
- 如实描述 memory 效果。

### 第 4 周：实车复现准备

目标：

- 用仿真中稳定的参数回到实车；
- 确认 scan_guard 和 local_obstacle_layer 在真实 `/scan` 上表现；
- 小范围、安全地测试 memory。

任务：

- 检查 E1/E2 启动链；
- 确认 `/odom`、`/scan`、`/cmd_vel`；
- 空场测试；
- 单障碍测试；
- 长条障碍测试；
- 记录日志和视频。

---

## 37. 写论文或阶段报告时的技术表述

可以把系统方法写成几个模块。

### 37.1 Baseline MPPI

本项目采用 receding-horizon MPPI 作为局部控制器。每个控制周期中，从当前状态采样多条未来控制序列，基于平面运动学模型进行 rollout，并根据目标距离、障碍距离、碰撞、控制平滑和边界等代价计算每条轨迹的总代价。随后使用指数权重对控制序列进行加权更新，并执行更新序列的第一步控制。

### 37.2 LaserScan-based local obstacle representation

为了使仿真与真实机器人输入一致，系统不直接将全局障碍真值输入 MPPI，而是使用 LaserScan 或 synthetic LaserScan 作为感知输入。扫描点经过前方安全检测和局部障碍层处理，被转换为世界坐标下的代表性圆形障碍，用于 MPPI 的 obstacle cost。对于长条障碍，局部障碍层尝试通过线段拟合和 representative circles 减少点云过密问题。

### 37.3 Safety arbitration

MPPI 输出的 proposed control 并不直接发送给机器人。系统通过 scan_guard 对前方近障和近体危险进行硬安全检测，并根据安全状态对速度进行停车、减速或限幅。最终控制命令由 MPPI proposal、安全检测、避障状态机、目标重获取和速度平滑共同决定。

### 37.4 Memory-Augmented MPPI

为减少局部低进展和重复旋转，本项目引入 memory field。系统根据执行历史检测 stuck、spin、近障慢爬和 recovery 状态，在相应位置生成 memory features。MPPI rollout 经过 memory feature 附近时，会额外产生 memory cost，从而降低重复进入历史失败区域的权重。在持续 spin/stuck 时，系统还会对 nominal control sequence 的前缀施加轻量 escape bias，使采样分布倾向于远离最近 memory feature 并朝目标方向恢复。

### 37.5 MuJoCo live validation

为了在实车实验前调试完整闭环，项目构建了 MuJoCo live simulation。该仿真环境通过 XML string 生成机器人、边界、长条障碍、圆柱障碍和目标点。机器人状态由 MuJoCo qpos/qvel 管理，synthetic LaserScan 由 MuJoCo ray casting 生成。viewer 中实时显示机器人、障碍、扫描射线、命中点、planner obstacles、MPPI 预测轨迹、已执行轨迹和 memory features，使系统行为可解释。

---

## 38. 新 Project 可能会问的问题与标准回答

### Q1：为什么不直接用全局障碍物列表做 MPPI？

因为实车没有全局障碍物真值，只有 LaserScan。当前研究重点是让 MuJoCo 仿真复现实车感知链路，所以 planner obstacles 应来自 synthetic LaserScan 和 local_obstacle_layer。全局 scene primitives 可以用于生成 MuJoCo geoms、debug 和碰撞参考，但不应成为 MPPI 主输入。

### Q2：为什么 memory 不直接控制机器人逃离？

因为那会绕开 MPPI 和安全仲裁，使系统变成另一个硬编码 controller。当前设计是让 memory 作为软代价影响 rollout，并在 stuck/spin 时轻量偏置 nominal sequence。最终控制仍由 MPPI proposal 和 scan_guard safety 共同决定。

### Q3：为什么不把 samples 增大到 2000？

因为项目面向实时机器人控制。无限增加 samples 会掩盖算法问题，也会让实车 compute time 不可接受。当前默认 200 samples、15 horizon 是轻量实时设置。可以做参数 sweep，但不能把“更大算力”当作主要解决方案。

### Q4：为什么 scan_guard 不能关？

scan_guard 是硬安全层。MPPI 是采样算法，可能因为采样不足或 cost 问题给出危险速度。关闭 scan_guard 得到的仿真结果不能迁移到实车，也不符合当前项目边界。

### Q5：为什么有 offline ablation 还要 live sim？

offline ablation 适合输出 CSV、PNG、统计指标；live sim 适合观察真实闭环中的感知、规划、执行和安全仲裁。两者互补。当前主线是 live sim，因为它更接近实车。

### Q6：为什么小车会在 memory feature 附近继续转？

可能是 memory cost 太弱，也可能是 scan_guard 把前进速度压掉，或者 planner obstacles 过密导致所有远离轨迹代价更高。要同时看 memory_cost、planner_obstacle_count、front_stop_mode、proposed/final control，不能只看紫色点。

### Q7：为什么要保留 simple scene？

simple 是回归测试。复杂场景跑不好可能是算法问题，也可能是场景太难。simple 能帮助确认基本闭环是否被破坏。

### Q8：如果实车和 MuJoCo 表现不一样怎么办？

先检查输入差异：真实 LaserScan 噪声、视场、range、安装位置、坐标系、时间延迟、里程计漂移、cmd_vel 执行延迟。不要马上认为 MPPI 算法错了。

---

## 39. 文件级别的阅读顺序

新 Project 或新协作者建议按这个顺序读代码。

第一组：项目总览

```text
README.md
docs/project_recovery_sync_2026-05-28.md
docs/weekly_logs/
```

第二组：MPPI 核心

```text
src/planners/mppi_sequence_demo.py
src/planners/mppi_receding_horizon_experiment.py
src/planners/mppi_mujoco_receding_horizon_experiment.py
```

第三组：MuJoCo

```text
src/envs/mujoco_point_env.py
experiments/mujoco_memory_mppi_live_sim.py
experiments/mujoco_memory_mppi_ablation.py
```

第四组：实车 bridge

```text
mppi_hardware_bridge/scripts/scenario_config.py
mppi_hardware_bridge/scripts/frame_transform.py
mppi_hardware_bridge/scripts/control_adapter.py
mppi_hardware_bridge/scripts/scan_guard.py
mppi_hardware_bridge/scripts/local_obstacle_layer.py
mppi_hardware_bridge/scripts/local_obstacle_cost.py
mppi_hardware_bridge/scripts/mppi_memory_field.py
mppi_hardware_bridge/scripts/mppi_planner_bridge.py
mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py
```

第五组：测试和检查

```text
mppi_hardware_bridge/scripts/check_avoidance_arbitration.py
mppi_hardware_bridge/scripts/check_realtime_and_soft_avoid.py
mppi_hardware_bridge/scripts/test_scan_guard_ros.py
mppi_hardware_bridge/scripts/test_cmd_vel.py
```

第六组：配置和 runbook

```text
mppi_hardware_bridge/config/lab_runtime.yaml
mppi_hardware_bridge/config/lab_runtime_goal_3_3_safe.yaml
mppi_hardware_bridge/docs/e1_e2_lab_runbook.md
```

---

## 40. 更细的实验记录模板

后续每次跑实验都建议按固定模板记录，避免组会前找不到信息。

```text
日期：
分支：
commit / 本地改动：
脚本：
命令：
scene：
memory on/off：
seed：
horizon：
num_samples：
temperature：
dt：
max_steps：
MuJoCo viewer / no-viewer：

环境：
Python：
MuJoCo：
是否 WSL：
是否 PyCharm：

结果：
success：
final_goal_distance：
total_steps：
collision：
stuck_steps：
spin_steps：
mean_v：
mean_abs_omega：
average_planner_compute_ms：
memory_feature_count_final：

观察：
1. predicted trajectory 行为：
2. executed trajectory 行为：
3. planner obstacles 是否过密：
4. scan_guard 是否频繁触发：
5. memory feature 是否生成：
6. escape_active 是否触发：
7. proposed/final control 差异：

问题：
1.
2.
3.

下一步：
1.
2.
3.
```

这个模板看起来啰嗦，但能避免“我记得昨天好像跑得更好”这种不可复现状态。

---

## 41. 当前项目的核心资产

这个项目最值得保护的资产不是某一个最终结果，而是以下这些已经形成的东西：

1. MPPI helper 函数和 receding horizon 实验框架。
2. MuJoCo viewer 中 predicted trajectory 可视化。
3. ROS hardware bridge 的安全仲裁结构。
4. scan_guard 前方硬安全检测。
5. local_obstacle_layer 对 LaserScan 长条障碍的表达。
6. Memory field 对 stuck/spin/low-progress 的记录和 cost。
7. MuJoCo live sim 对实车感知链路的复现。
8. 多场景 benchmark 雏形。
9. 组会 PPT 中形成的叙事框架。
10. weekly logs 中关于 failure mode 的反思。

这些资产的共同特点是：它们不是一次性代码，而是后续论文、实验、实车复现和组会汇报可以反复使用的基础。

---

## 42. 最终给新 Project 的一句话委托

如果只能给新 ChatGPT Project 一句话，那就是：

> 请不要把这个项目当作普通 MPPI demo；它已经发展成一个以 MuJoCo live simulation 和 ROS 实车 bridge 为双平台、以 LaserScan 局部感知为输入、以 scan_guard 为硬安全、以 local_obstacle_layer 为障碍表达、以 Memory-Augmented MPPI 为当前研究改进点的移动机器人局部避障系统。接下来请帮助我小步稳定 live sim、做 memory on/off 消融、保留实车安全边界，并把所有修改都建立在当前代码和实验现象之上。
