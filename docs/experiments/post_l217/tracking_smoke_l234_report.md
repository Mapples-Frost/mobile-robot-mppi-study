# L234 大尺度 Tracking 单种子 MuJoCo smoke 审计报告

**实验日期：** 2026-07-20
**审计日期：** 2026-07-21
**实验目录：** `results/research_platform/rl/tracking_smoke_l234_seed923301001_attempt2`
**定位：** development/pipeline qualification；不是论文正式结果

## 1. 实验目的

本轮只检查新建 Tracking 链路是否能够端到端运行：三条大尺度路径能否加载，四个 2×2 核心方法臂能否调用 MuJoCo、ICODE、Actor、HSS、MPPI 与安全链，并完整输出 Tracking 指标和逐拍日志。

四个方法臂为：

| 方法 | Value-aligned ICODE | role-aware adaptive HSS |
|---|---:|---:|
| Simple combination | 否 | 否 |
| Value fixed | 是 | 否 |
| Ordinary adaptive | 否 | 是 |
| Full proposed | 是 | 是 |

实验采用 development seed `923301001`、`nominal_seen` 物理域、每次决策 30 条 rollout、1 次 MPPI iteration、最多 1200 个控制步。没有读取或使用 Tracking sealed seeds。

## 2. 完整性与来源审计

| 审计项 | 结果 |
|---|---:|
| 计划回合 | 12 |
| 完成回合 | 12 |
| 重复 method×scene×domain×seed 单元 | 0 |
| `config_resolved.yaml` | 12/12 |
| `trajectory.csv` | 12/12 |
| `metrics.json` | 12/12 |
| 回合 `provenance.json` | 12/12 |
| `qualification` | 全部为 1 |
| MuJoCo 版本 | 全部为 3.2.3 |
| Git SHA | 全部为 `a9be4e933c3064707c04b0a61cf9930339de8d5e` |
| rollout 预算 | 全部为 30 |
| MPPI iterations | 全部为 1 |

顶层 provenance 保存了 manifest SHA、Actor checkpoint SHA、ordinary/value ICODE ensemble SHA、可靠性校准文件 SHA、场景来源、随机化 schedule seed 和方法顺序。`progress.csv` 有 236 列；仅 `time_to_goal_s` 和 `minimum_dynamic_obstacle_center_distance` 为空，分别因为没有成功到达以及本轮没有动态障碍物，属于预期缺失。

## 3. 逐回合结果

| 方法 | 场景 | 终止原因 | 步数 | 路径完成度 | CTE RMSE (m) | CTE P95 (m) | 最小 footprint 边界余量 (m) |
|---|---|---|---:|---:|---:|---:|---:|
| Full | S-Chicane | max_steps | 1200 | 0.1834 | 0.1631 | 0.2826 | 0.2582 |
| HSS | S-Chicane | max_steps | 1200 | 0.1842 | 0.1895 | 0.1996 | 0.3666 |
| Value | S-Chicane | boundary_violation | 948 | 0.1701 | 0.2759 | 0.4756 | -0.0038 |
| Simple | S-Chicane | boundary_violation | 872 | 0.1714 | 0.1872 | 0.4208 | -0.0017 |
| Full | Hairpin | max_steps | 1200 | 0.1121 | 0.0655 | 0.1813 | 0.3228 |
| HSS | Hairpin | max_steps | 1200 | 0.1118 | 0.0981 | 0.1236 | 0.3063 |
| Value | Hairpin | max_steps | 1200 | 0.1118 | 0.3109 | 0.3289 | 0.2510 |
| Simple | Hairpin | max_steps | 1200 | 0.1118 | 0.3114 | 0.3316 | 0.2631 |
| Full | Infinity | max_steps | 1200 | 0.1091 | 0.0853 | 0.1342 | 0.2946 |
| HSS | Infinity | max_steps | 1200 | 0.1089 | 0.0790 | 0.1485 | 0.2933 |
| Value | Infinity | boundary_violation | 533 | 0.0269 | 0.4224 | 0.5808 | -0.0002 |
| Simple | Infinity | boundary_violation | 527 | 0.0284 | 0.4233 | 0.5880 | -0.0045 |

全部 12 个回合的 `success=False` 和 `boundary_safe_success=False`。因此，本轮不能支持任何“Full 优于基线”或“Tracking 已成功”的论文结论。

## 4. 方法级描述性汇总

这里只给出单 seed 的描述统计，不计算置信区间或显著性。

| 方法 | 成功率 | 平均完成度 | 平均 CTE RMSE (m) | 平均 CTE P95 (m) | 三场景最小边界余量 (m) | 平均 safety interventions | 平均 planner time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full proposed | 0/3 | 0.1348 | 0.1046 | 0.1994 | 0.2582 | 771.3 | 258.2 |
| Ordinary adaptive | 0/3 | 0.1350 | 0.1222 | 0.1572 | 0.2933 | 763.0 | 235.1 |
| Simple combination | 0/3 | 0.1039 | 0.3073 | 0.4468 | -0.0045 | 340.7 | 305.2 |
| Value fixed | 0/3 | 0.1029 | 0.3364 | 0.4618 | -0.0038 | 343.7 | 321.7 |

自适应 HSS 两个方法臂在该 smoke 中保持了更低的横向误差和正边界余量，而固定 HSS 两臂在 S-Chicane 与 Infinity 出现了边界越界。这只能说明机制和指标具有可观测响应，不能在单 seed、全部任务未完成的条件下解释为有效性证据。

## 5. 失败模式与根因

### 5.1 不是从起点就“不会走”

Hairpin 的前 100 步中，Simple、HSS 和 Full 的平均 proposed speed 分别约为 0.340、0.440 和 0.414 m/s，说明路径预览、先验和 MPPI 在无近障碍阶段能够产生正常前进动作。

### 5.2 主要失败发生在首个中心线附近障碍物

Hairpin 四个方法最终都停留在约 `(7.93, 2.67–2.95)`，紧邻位于 `(8.50, 3.10)` 的首个 box；路径完成度冻结在约 0.112。S-Chicane 和 Infinity 的自适应方法也在首个障碍物附近长期停滞。

自适应方法在 1200 步中约有 752–779 次（依场景而异）安全干预，平均 applied speed 随长时间停滞下降到约 0.05–0.06 m/s。固定 HSS 方法部分回合没有被安全层长期限速，但发生了边界越界。

### 5.3 几何可行不等于当前控制预算可解

场景几何审计只证明在 footprint 和走廊边界之间存在静态可行通道；最困难障碍物的推荐安全旁路余量仅约 0.024–0.042 m。它没有证明仅用 `K=30`、一次 refinement 的局部 MPPI 能稳定找到绕行轨迹。

因此，本轮首要矛盾是“中心线近障碍 + 很小旁路余量 + 低采样预算”造成的局部绕障能力不足，而不是 ICODE 或 RL+ICODE 核心耦合被否定。

## 6. Gate 结论

- **Pipeline integrity：通过。** 四方法、三场景、MuJoCo、Tracking 指标和 provenance 全部贯通。
- **Terminal/path success：未通过。** 0/12 成功，不能进入 72 回合 Development Matrix。
- **数据科学定位：负向 development evidence。** 必须保留，不得覆盖或当作正式确认结果。
- **下一步：L235 首障碍绕行能力探针。** 只在 development seed 上增加所有方法公平共享的 MPPI rollout/refinement 预算，先判断是否属于采样覆盖不足；核心 ICODE、Actor、HSS、cost、安全链和地图不变。

## 7. 进一步分析建议

1. 在每个场景只跑 Full proposed 的首障碍探针，采用 `K=100`、2 iterations、700 步，三场景并行。
2. 预先冻结通过阈值：零碰撞、零边界越界，并且路径进度明确超过各场景首障碍弧长。
3. 若通过，再对四方法做同预算短资格矩阵；若失败，保留结果并区分“采样不足”“安全冻结”“局部目标拓扑不足”。
4. 在首障碍 Gate 通过前，不运行 72 回合矩阵，也不接触 sealed Tracking seeds。
