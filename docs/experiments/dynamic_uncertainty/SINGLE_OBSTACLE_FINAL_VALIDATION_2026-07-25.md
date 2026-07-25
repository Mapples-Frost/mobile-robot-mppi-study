# 单动态障碍 Actor–MPPI 最终开发验证报告

日期：2026-07-25  
权威仓库：`D:\Projects\mobile-robot-mppi-study`  
结论状态：**单障碍 development 阶段完成；三障碍暂停；尚未宣称 sealed / publication-confirmatory 泛化。**

## 1. 最终结论

保留方案为 V5 Amendment 6 的 update-250 Candidate，加上同周期 guided-cost 过滤、既有概率风险约束、因果 LaserScan 动态逃逸和 Pareto forward commit。

它通过了两层证据：

1. 24 对全新 development seeds（730100228–251）的冻结、配对、臂顺序平衡矩阵，11 项预注册门全部通过。
2. 全新 seed730100255 的实时确认：Candidate 到达目标、无碰撞、300/300 个有效决策均为 600 rollouts、同周期过滤实际触发、LaserScan 时间逃逸实际触发 17 周期，Planner P95 为 83.54 ms。

所以，单障碍在当前授权的 development 范围内已经形成可保留的明确正向结果。它不是“所有指标全面碾压”：Candidate 的成功率和碰撞率更好，但成功回合通常略慢；统计事件数也仍偏小，不能包装成已完成正式泛化验证。

## 2. 昨晚单障碍到底做了什么

### 2.1 先否定了不稳定的早期结果

最初的 24 对扩展矩阵（seeds 730100204–227）暴露了 Candidate-only 碰撞、丢失 Source 成功回合和终距退化，因此原先的小样本正向结论被撤回，没有包装成成功。

### 2.2 Amendment1 只修复安全，不满足总门槛

第一轮有界修复引入 Pareto forward commit，消除了新增 Candidate 碰撞和丢失 Source 成功，但总步数与批次非退化门槛仍未同时通过，因此继续保留为失败开发结果。

### 2.3 Amendment2 才形成完整通过

Amendment2 使用未打开的新 seeds 730100228–251，在执行前冻结协议、checkpoint、代码哈希、三批八对的臂顺序以及所有判定门槛。执行中未调参、未替换 seed。

最终 24 对矩阵：

| 指标 | Source | Candidate | 变化 |
|---|---:|---:|---:|
| 成功 | 20/24 | 22/24 | +2 回合（+8.33 个百分点） |
| 碰撞 | 3/24 | 1/24 | −2 回合（−8.33 个百分点） |
| Candidate-only 碰撞 | — | 0 | 通过 |
| 丢失 Source 成功 | — | 0 | 通过 |
| 挽救 Source 失败 | — | 2 | seeds 730100240、730100245 |
| outcome-aware efficiency | 7741 | 7615 | 改善 126 penalty-adjusted steps |
| 平均终点距离 | 0.773262 m | 0.523574 m | 改善 0.249688 m |
| 同周期过滤 | 开启且触发 | 开启且触发 | 两臂均有机制证据 |
| rollout 预算 | 600/决策 | 600/决策 | 不变 |

三批中 batch2、batch3 非退化，batch1 有退化；因此结果是“总体 development gate 通过”，不是每个子批都占优。

## 3. 正式统计解释

24 对配对结果的正式检验如下：

| 指标 | 统计结果 | 正确解释 |
|---|---|---|
| 成功率 | exact McNemar p=0.500 | 方向正向，但只有 2 个不一致对，证据量不足以拒绝零假设 |
| 碰撞率 | exact McNemar p=0.500 | 方向正向，但事件数同样偏少 |
| 终点距离 | mean Δ=−0.249688 m；Wilcoxon p=0.855；bootstrap 95% CI [−0.8305, 0.0811] m | pooled mean 改善主要受挽救失败回合影响；中位数没有改善 |
| 原始步数 | mean Δ=+10.25；Wilcoxon p=0.024 | Candidate 在多数成功/完整回合上更慢，不能声称无条件效率更高 |
| 预注册 outcome-aware efficiency | 7741→7615；Wilcoxon p=0.182 | 把失败计为 400 步后总体改善，但正式显著性不足 |

因此最稳妥的论文级表述是：**该 Candidate 在冻结 development 矩阵中满足联合安全/完成/非退化工程门槛，并显示更少碰撞与更多成功的方向性改善；尚不具备统计上确认“总体优于 Source”的证据。**

## 4. 全新实时确认

全新 development seed730100255 在执行前冻结，Candidate-first，未用默认 checkpoint，显式绑定：

- Candidate SHA256：`714694F8D179F02B90C14E56C9E3E4F29A0101B70AD53F3CC795EB72DEA7A1B5`
- Source SHA256：`6E346E5E4FCC1ECA029EE4495CA0FFD0C978C5CE11089AB8EBFAAEE89236C63D`

结果：

| 指标 | Candidate | Source |
|---|---:|---:|
| 到达目标 | 是 | 是 |
| 碰撞 | 否 | 否 |
| 步数 | 300 | 316 |
| 最小净空 | 0.395698 m | 0.395698 m |
| 最终目标距离 | 0.299778 m | 0.285260 m |
| Planner P95 | **83.537 ms** | 78.644 ms |
| 有效决策 | 300 | 316 |
| rollout | 300/300 均为 600 | 316/316 均为 600 |
| 同周期过滤活跃比例 | 0.5200 | 0.6329 |
| LaserScan temporal emergency | 17 周期 | 17 周期 |
| vetted emergency action | 17 周期 | 17 周期 |
| 在线 tracker / forecast | 每个决策有效 | 每个决策有效 |

Candidate 比 Source 少 16 步，同时满足 100 ms P95 门槛。

## 5. 实时性边界必须如实保留

实时性并非所有运行都稳定低于 100 ms：

- 旧 24 对矩阵的 Candidate episode-level P95 最大值为 115.15 ms。
- seed730100254 的首次无 profile 运行，Candidate 虽成功且无碰撞，但 P95=121.43 ms，未过实时门。
- 相同 seed 的后续行为完全复现，P95 分别为 106.69 ms 和 99.26 ms，说明计时对 Windows 当时负载敏感。
- seed730100299 的优化确认 P95=86.86 ms；全新 seed730100255 为 83.54 ms。

因此可以确认“已有新的、含动态逃逸触发的 <100 ms 正向实时运行”，但不能声称“整个 24-seed 分布都严格 <100 ms”。

## 6. 因果性与完整性

- 单动态障碍：1。
- 每决策预算：固定 600 rollouts，无新增候选预算。
- 控制输入使用在线 LaserScan 多目标 tracker 和概率 forecast；seed730100255 中每个有效决策的 tracker 与 forecast 均有效。
- `exact_ground_truth_pose_for_control=false`、`simulator_truth_for_control=false`。
- 24 对 Amendment2：220/220 原始文件重新计算 SHA256，0 缺失、0 不匹配；11/11 gate 通过。
- seed730100255：9 个原始文件已写入独立 integrity audit；stderr 为空。
- 156 项定向回归通过。
- 相关 Python 文件编译通过。
- `git diff --check` 通过。
- 三障碍进程未恢复。

## 7. 关键证据

- 冻结协议：`configs/research/dynamic_actor_v5a6_samecycle_expanded_development_amendment2.yaml`
- 24 对矩阵：`research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2`
- 正式统计：`research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/formal_statistics.json`
- 矩阵完整性：`research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/integrity_audit.json`
- 新 seed 协议：`configs/research/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255.yaml`
- 新 seed 结果：`research_artifacts/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255/result.json`
- 新 seed 完整性：`research_artifacts/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255/integrity_audit.json`
- 验证报告：`research_artifacts/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255/validation_report.md`

## 8. 停止点

当前停止点是：**保留单障碍 Candidate，冻结现有 development 证据，不开启 sealed seeds，不恢复三障碍。**

如果下一阶段要形成论文级确认，需要另行预注册 sealed / held-out 协议，并把实时性作为分布级门槛，而不是依赖单个通过 seed。

