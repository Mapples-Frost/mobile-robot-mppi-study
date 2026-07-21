# L246 Tracking 失败根因只读审计协议

日期：2026-07-21

状态：预注册只读 development analysis；不产生新仿真回合

## 1. 研究问题

L244 已证明 BC-anchored Path/Residual-Conditioned Actor 能增加 Hairpin 和三场景平均路径推进，
但 Hairpin、S-Chicane、Infinity 在 700 步内均未完成，且后两者各有一步 footprint boundary
violation。L245 证明简单增加 path boundary buffer 不能解决问题。

本审计只回答以下问题：

1. 700-step 上限在几何与速度约束下是否足以完成每条路径；
2. 每条轨迹未完成的主要原因属于 step budget、路径投影、Actor proposal、安全仲裁、边界代价、
   local minimum、terminal logic 或 planner latency 中的哪一类；
3. Infinity 中央交叉是否发生 branch identity 或 progress continuity 异常；
4. 下一轮只允许改变的单一工程变量应是什么。

## 2. 冻结输入

只读取以下三个已经完成的 L244 `full_proposed` 回合：

```text
results/research_platform/rl/tracking_l244_bc_anchor_hairpin_seed923301001/
results/research_platform/rl/tracking_l244_bc_anchor_s_chicane_seed923301001/
results/research_platform/rl/tracking_l244_bc_anchor_infinity_seed923301001/
```

冻结条件：

- development seed：`923301001`；
- physics domain：`nominal_seen`；
- MuJoCo：3.2.3；
- Actor checkpoint SHA256：`e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c`；
- 总 rollout 预算 K=100，50 candidates × 2 iterations；
- episode 上限：700 steps；
- control dt、action limits、路径、障碍、cost、scan_guard 和安全链均保持原样。

不得读取 sealed seed，不得重跑已完成回合，不得修改控制参数、模型或地图。

## 3. 预先冻结的计算定义

### 3.1 路径与时间预算

对每个场景：

- 路径总长：逐拍 `path_arc_length + remaining_path_length` 的中位数；
- 控制周期：相邻 `time` 差的中位数；
- 最大线速度：resolved config 的 `action_space.upper[0]`；
- 理论最少步数：`ceil(path_length / (v_max * dt))`；
- 实际平均前进速度：`mean(max(applied_v, 0))`，包含被安全层降为零的时刻；
- applied-velocity 预计步数：`ceil(path_length / (mean_forward_applied_v * dt))`；
- empirical-progress 预计步数：`ceil(observed_steps / final_path_completion_ratio)`；
- 700-step 余量分别报告为 `700 - estimated_steps`。负值表示预算不足。

三个估计不得互相替代：理论值只回答物理下界，applied-velocity 值反映执行速度，empirical-progress
值同时包含路径投影、绕行、横向偏差和控制停滞。

### 3.2 停滞与事件

- sustained progress stall：连续 50 steps（约 5 s）内 `path_arc_length` 增量不超过 0.05 m；
- safety-dense window：50-step 滚动窗口内 `safety_override` 比例最高的窗口；
- first dense safety segment：首个安全覆盖率不低于 0.5 的 50-step 窗口；
- 最大横向误差：`abs(signed_cross_track_error)` 最大的时刻；
- 最小边界余量：`minimum_footprint_boundary_margin` 最小的时刻；
- planner deadline：使用 resolved config/metrics 已冻结的 deadline，并报告 p50/p95/p99 和 miss rate；
- Actor 使用：报告 `reliability_proposal_authority`、`paper_guided_elite_count` 和 fallback fraction。

停滞分类：

- safety intervention：停滞窗口 safety fraction >= 0.5；
- repeated turning：停滞窗口 mean `abs(applied_omega)` >= 0.3 rad/s；
- local minimum：满足停滞，但不满足以上两项；
- no sustained stall：未出现定义内停滞，未完成优先检查 step budget。

### 3.3 Infinity 路径投影连续性

仅对 Infinity 额外计算：

- `branch_id` 的唯一值与相邻变化次数；
- `path_progress_ratio` 单步回退超过 0.002 的事件；
- 单步向前跳变超过 0.03 的事件；
- `center_crossing_count` 的变化位置；
- 中央区域中 branch 变化是否伴随非连续 progress 跳变。

上述阈值在读取完整 outcome 前冻结，不根据结果调整。

## 4. 必须生成的产物

1. 三场景路径预算表；
2. 三场景关键事件表；
3. 每场景一张同步时间序列图，包含：
   - path completion；
   - proposed/executed/applied `v`；
   - proposed/executed/applied `omega`；
   - cross-track error 与 footprint boundary margin；
   - safety intervention；
   - Actor proposal authority 与 guided elites；
   - derived stall indicator；
   - planner time；
4. Infinity branch/progress continuity 图与事件表；
5. 根因分类报告，明确 evidence、inference 和 recommendation；
6. 分析脚本、配置/输入哈希与 Windows 桌面副本。

## 5. 根因判定优先级

1. 若 `theoretical_min_steps > 700`，则 step budget 被判定为必要根因；
2. 若理论预算充足但 applied/empirical 预计步数大于 700，继续区分安全压制、转向停滞与 local minimum；
3. 若 Infinity 出现预注册阈值以上 progress 回退/跳跃并与 branch 变化重合，标记 path projection；
4. boundary violation 发生在候选被选中后，且相邻时段并无安全压制时，标记 candidate-level boundary cost；
5. planner latency 只在 p99 超过 deadline 且 miss rate 非零时列为实时性根因；
6. terminal logic 只在 path completion 接近终点（>=0.95）但未 success 时成立。

允许多根因并存，但必须区分主根因与次根因。

## 6. 本轮停止条件

完成只读审计、报告和图表后停止，不自动进入 L247，不重新训练 Actor，不运行新 seed。
下一项修复必须是由审计证据支持的单变量 development 改动，并在运行前另行预注册。

