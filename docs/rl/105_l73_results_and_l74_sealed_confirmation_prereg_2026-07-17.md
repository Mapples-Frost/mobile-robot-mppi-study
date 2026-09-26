# L73 部署选择结果与 L74 封存确认预注册

日期：2026-07-17  
状态：L73 已完成；L74 规则在首次打开封存种子前锁定。

## 1. L73 结果

L73 使用从未参与 L72 训练/验证的 episode seeds `22170801`–`22170803`，对每个训练块的冻结 BC 和 5k–30k 六个 SAC checkpoints 进行同场景、同 seed 配对闭环评估。

- 3 个模型块；
- 3 个静态几何场景；
- 3 个新开发 seeds；
- 7 个候选；
- 共 189/189 episodes；
- episode 矩阵完整，无 NaN/Inf；
- 冻结 BC correction gate alpha 严格为 0。

L73 的预注册门槛要求至少 2/3 模型块选择非零 checkpoint，实际为 3/3。三个块均选择 30k：

| 模型块 | RL seed | 选中 checkpoint | 配对成功增益 | 配对成功损失 | 新增碰撞 | 平均终点距离改善 | correction gate alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20260751 | 30k | +1 | 0 | 0 | +0.3888 m | 0.3485 |
| 1 | 20260752 | 30k | +1 | 0 | 0 | +0.2705 m | 0.2770 |
| 2 | 20260753 | 30k | +1 | 0 | 0 | +0.0020 m | 0.2850 |

第三块依据成功增益通过；其平均距离改善小于 0.005 m，不将其描述为显著距离改善。L73 是开发集上的 checkpoint selection，不是独立最终确认。

## 2. L74 研究问题

在完全不再选择 checkpoint 的条件下，L73 锁定的三个 30k SAC 策略能否在五个封存 seeds 上，相对对应冻结 BC 保持跨模型块的非劣与任务改进？

## 3. 锁定设计

- 配置：`configs/rl/static_risk_balanced_sealed_confirmation_l74.yaml`。
- 固定候选：每块只比较 step-0 冻结 BC 与 step-30k SAC。
- 模型块：L73 的三组 RL/ICODE/BC 配对，不更换 checkpoint。
- 场景：`clean_single_obstacle`、`narrow_corridor`、`u_trap_long_board`。
- 物理域：与 L73 相同的固定 high-dynamic MuJoCo plant。
- 封存 seeds：`22170811`–`22170815`。
- 每块：3 场景 × 5 seeds × 2 方法 = 30 episodes。
- 总计：90 episodes。
- 调度：每块独立随机打乱，schedule seed 为 `2026071774 + block_index`。

## 4. 主要门槛

每个模型块对 15 组同场景同 seed 结果进行配对。30k SAC 在该块合格，当且仅当：

1. 相对 BC 的成功损失为 0；
2. 新增碰撞为 0；
3. 平均终点距离不恶化；
4. 至少新增 1 次成功，或平均终点距离改善至少 0.005 m；
5. correction gate alpha 均值至少为 0.01。

L74 总门槛：至少 2/3 独立模型块合格。任何缺失 episode、重复组合、非有限指标、旧 seed 混入或冻结 BC correction 非零均 fail closed。

## 5. 次要报告指标

在不改变主要门槛的前提下，报告：

- 45 个配对 episode 的成功 gains/losses；
- 分场景成功率和终点距离；
- collision、minimum clearance；
- mean/max planner time；
- control jerk、spin/stuck steps；
- correction gate alpha。

episode 级汇总是描述性结果；同一训练块内 episode 并非新的模型训练重复。模型块数量仅为 3，因此不以普通 episode 独立性假设夸大统计显著性。

## 6. 停止规则

- 若 L74 通过：将该静态高动态条件作为正向确认结果，随后转向动态障碍/OOD 与实车迁移，不再用这些封存 seeds 调参。
- 若 L74 失败：如实保留结果，不回到 5k–25k 重选 checkpoint，不修改门槛，不补挑 seeds。

## 7. 解释边界

L74 最多支持“风险均衡回放提高冻结 BC 上 SAC 修正的静态场景跨模型块稳定性”。它不证明全局最优、理论稳定、所有环境优越或实车有效；动态障碍、感知噪声和真实平台需要独立验证。
