# L214 七方法最终 Point-goal Benchmark 正式结果

日期：2026-07-19  
状态：正式预注册实验完成；210/210 episodes 通过完整性审计  
预注册：`docs/rl/212_final_paper_benchmark_prereg_2026-07-19.md`  
Qualification：`docs/rl/213_final_benchmark_qualification_results_2026-07-19.md`

## 1. 先给结论

这轮结果是**有价值但不全面正向的正式结果**：

1. reliability-adaptive HSS 在冻结的十个新 seed 上得到确认：相对普通 ICODE + 固定 30% Actor guidance，平均 final goal distance 改善 0.41975 m，seed-cluster 95% CI 为 [0.02697, 0.79717]；jerk 和 planner time 也严格改善；
2. Full Proposed 相对 Simple Combination 将 success 从 30.0% 提高到 50.0%，final distance 改善 0.52833 m，但这两个 95% CI 均略跨 0，不能写成确认性显著改善；jerk 与 planner time 的改善则具有严格有利区间；
3. value-aligned ICODE 单因素在本轮没有复现 L211 的强主效应：success、final distance、jerk 和 compute 的 CI 均跨 0；
4. 所有七种方法共 210 个 episode 均为 0 collision，且每个 MPPI arm 的模型 rollout 预算保持 100；
5. Traditional MPPI 在该 clean point-goal benchmark 上仍然最好：63.3% success、0.3883 m final distance、4.51 ms mean planner time；
6. 因此，本轮**支持 HSS 机制和“学习模块在未见组合失配中有潜力”的有限主张，但不支持 Full Proposed 全局优于强 Traditional MPPI，也不支持两个耦合机制都已被最终确认**。

这是冻结配置、未参与开发的 seeds 101--110 上得到的结果。没有删除 seed、修改阈值、重新训练或根据中间结果改变分析。

## 2. 实验设计与统计单位

- 七方法：Traditional MPPI、ICODE-MPPI、RL-driven MPPI、Simple Combination、Value Fixed、Ordinary Adaptive、Full Proposed；
- 场景：`clean_single_obstacle`；
- 物理域：`nominal_seen`、`long_delay_seen`、`combined_unseen`；
- 独立实验单位：seed，共 10 个；
- 重复分层：每个 seed 内的三个 physics domains；
- 完整区组：10 seeds × 3 domains = 30 blocks；
- 每个 block 内七方法顺序随机化；
- 总 episode 数：30 × 7 = 210；
- 置信区间：按 seed cluster 重采样 10,000 次的 percentile bootstrap；
- timestep 和同 seed 的多个 physics domain 均未被当成独立样本；
- qualification seed 99 没有进入正式数据。

由于使用预注册的配对、seed-cluster bootstrap，主要效应解释不依赖 seed-level 差异服从正态分布，也没有在看到结果后更换检验。报告以效应量和区间为主，不进行显著性检验购物。

## 3. 七方法总体结果

| 方法 | Success | Collision | Final distance (m) | Control jerk | Mean planner time (ms) |
|---|---:|---:|---:|---:|---:|
| Traditional MPPI | **63.3%** | 0% | **0.3883** | 0.06419 | **4.51** |
| ICODE-MPPI | 33.3% | 0% | 0.4312 | **0.05716** | 85.96 |
| RL-driven MPPI | 46.7% | 0% | 1.2528 | 0.09058 | 27.19 |
| Simple Combination | 30.0% | 0% | 1.9098 | 0.08986 | 166.84 |
| Value Fixed | 33.3% | 0% | 1.8314 | 0.09023 | 170.11 |
| Ordinary Adaptive HSS | 43.3% | 0% | 1.4901 | 0.08397 | 160.42 |
| Full Proposed | 50.0% | 0% | 1.3815 | 0.08321 | 159.78 |

这个表最重要的含义不是“Full 排名第二”，而是：

- 强 Traditional MPPI 在简单、低感知歧义的 point-goal 任务上具有明显优势；
- ICODE-only 的 final distance 与 jerk 很好，但 success 较低，说明当前 success 判据、终端行为或 300-step 截断与平均距离提供了不同信息；
- RL guidance 与两个 learned dynamics ensemble 带来明显计算开销；
- Full 能修复 Simple Combination 的部分失败，但尚未恢复到 Traditional 的总体水平。

## 4. 冻结的核心消融

下表中的 effect 统一为正值更好。

| 对比 | Success effect | Final-distance improvement | Jerk improvement | Planner-time improvement |
|---|---:|---:|---:|---:|
| Value Fixed vs Simple | +3.33 pp, CI [-16.67, 23.33] | +0.07841 m, CI [-0.38297, 0.53368] | -0.000365, CI [-0.001225, 0.000655] | -3.263 ms, CI [-8.120, 1.708] |
| Adaptive HSS vs Simple | +13.33 pp, CI [-6.67, 33.33] | **+0.41975 m, CI [0.02697, 0.79717]** | **+0.005894, CI [0.003508, 0.008420]** | **+6.428 ms, CI [0.189, 12.135]** |
| Full vs Simple | +20.00 pp, CI [-3.33, 43.33] | +0.52833 m, CI [-0.03354, 1.09678] | **+0.006651, CI [0.004913, 0.008416]** | **+7.065 ms, CI [2.286, 11.157]** |

配对 Cohen's \(d_z\)：

- Adaptive HSS final distance：0.639；jerk：1.389；planner time：0.632；
- Full vs Simple final distance：0.542；success：0.474；jerk：2.238；planner time：0.933；
- Value Fixed final distance：0.102；success：0.101。

### 4.1 可以确认的机制

Adaptive HSS 在 fixed rollout budget 下，降低了 final distance、jerk 和平均规划耗时。这是本轮最稳健的正向结论。

### 4.2 尚未确认的机制

Value alignment 单独使用时只有很小的 success/final-distance 点估计改善，区间很宽且跨 0；jerk 与计算时间点估计略差。因此 L211 的价值对齐强效应没有在更大的十-seed正式队列中稳定复现。

### 4.3 Full Proposed 的准确表述

Full 相对 Simple 的 success 和 final-distance 点估计有实际意义，但区间略跨 0；jerk 与计算时间严格改善。可以写成“改善趋势并确认效率/平滑性收益”，不能写成“正式确认全面提高任务完成率”。

## 5. 2×2 因子效应

因子名称在这里恢复为论文语义：

- value-alignment factor：ordinary vs value-aligned ICODE；
- adaptive-HSS factor：fixed 30% vs reliability-adaptive sampling。

| 指标 | Value alignment main effect | Adaptive HSS main effect | Interaction |
|---|---:|---:|---:|
| Success | +5.0 pp, CI [-8.33, 18.33] | +15.0 pp, CI [-1.67, 31.67] | +3.33 pp, CI [-26.67, 30.00] |
| Final distance | +0.0935 m favorable, CI [-0.2275, 0.4207] | **+0.4348 m favorable, CI [0.0831, 0.8021]** | +0.0302 m favorable, CI [-0.5670, 0.6124] |
| Control jerk | +0.000196 favorable, CI [-0.000705, 0.001148] | **+0.006455 favorable, CI [0.004568, 0.008189]** | +0.001123 favorable, CI [-0.001781, 0.003390] |
| Planner time | -1.313 ms, CI [-4.245, 2.095] | **+8.378 ms, CI [4.222, 12.481]** | +3.900 ms, CI [-2.422, 10.465] |

没有任何 interaction 区间严格排除 0。因此不能宣称超加性 synergy；“两个模块相互校准”的机制描述仍可保留，但目前最终证据主要由 HSS 驱动。

## 6. 分物理域结果

### 6.1 Success rate

| 方法 | Nominal | Long delay | Combined unseen |
|---|---:|---:|---:|
| Traditional | 50% | 70% | 70% |
| ICODE | 10% | 10% | **80%** |
| RL-driven | 30% | 60% | 50% |
| Simple I+RL | 20% | 50% | 20% |
| Value-aligned | 40% | 30% | 30% |
| Adaptive HSS | **70%** | 40% | 20% |
| Full Proposed | 30% | 40% | **80%** |

### 6.2 Final goal distance

| 方法 | Nominal | Long delay | Combined unseen |
|---|---:|---:|---:|
| Traditional | **0.402** | **0.381** | 0.381 |
| ICODE | 0.491 | 0.466 | **0.337** |
| RL-driven | 1.696 | 0.995 | 1.067 |
| Simple I+RL | 2.103 | 1.442 | 2.184 |
| Value-aligned | 1.662 | 1.929 | 1.904 |
| Adaptive HSS | 0.915 | 1.684 | 1.872 |
| Full Proposed | 1.877 | 1.637 | 0.631 |

Combined-unseen 同时改变质量、惯量、执行延迟、扭矩限制与接触摩擦。Full 在该域达到 80% success，相对 Simple 的 20% 是值得后续独立复现的信号；但 Full final distance 0.631 m 仍差于 ICODE 0.337 m 和 Traditional 0.381 m。该分域观察在本报告中作为预先计划的描述性外部有效性结果，不额外进行选择性显著性检验。

## 7. “复杂场景”边界

`combined_unseen` 是复杂动力学失配，不等于复杂障碍导航。本轮场景仍是 clean single-obstacle。

此前独立 `lab_complex` 补充包含 5 seeds × 3 physics domains × 4 arms = 60 episodes，在 180-step 截断下四个方法均为 0/15 success：

| Arm | Success | Final distance (m) | Jerk | Planner time (ms) |
|---|---:|---:|---:|---:|
| Ordinary fixed | 0/15 | 2.2321 | 0.10857 | 188.38 |
| Value fixed | 0/15 | 2.1537 | 0.10949 | 189.21 |
| Ordinary adaptive | 0/15 | 2.2410 | 0.10366 | 177.55 |
| Full Proposed | 0/15 | 2.2025 | 0.10628 | 178.44 |

该结果仅支持 HSS 降低计算量且没有碰撞回归，不支持复杂导航成功率提升。复杂障碍下的主要瓶颈仍可能是全局/局部路径拓扑和 RL prior 的场景能力，而不是残差动力学精度。

## 8. 与论文主张的关系

### 当前可以写入论文

1. 在相同 rollout budget 下，ICODE reliability-adaptive HSS 能提高 simple-combination 的进度、平滑性和在线效率；
2. Full Proposed 相对直接模块拼接具有明确的 jerk/compute 收益，并在 success/final distance 上呈有实际意义但不确定的改善；
3. Full 在 combined-unseen dynamics 上出现 80% success，说明跨层适应在复合失配下值得进一步研究；
4. 强 Traditional MPPI 在简单任务上仍具优势，学习模块不是所有场景都应启用；
5. 所有正式方法保持 0 collision，并保留安全链。

### 当前不能写

1. Full Proposed 全面优于 Traditional MPPI；
2. value-consistent ICODE 已在最终十-seed实验中稳定改善闭环控制；
3. ICODE 与 RL 存在超加性 synergy；
4. 复杂障碍导航成功率已经改善；
5. 具备稳定性、收敛性、收缩性或实车泛化保证。

## 9. 投稿判断与下一实验

这轮数据可作为论文中的一张关键“边界清晰的正式消融表”，但不能单独承担整篇论文的中心胜出结论。下一步不应回头调 L214，而应新建独立预注册：

1. **Sample-efficiency / short-horizon benchmark**：在 K=25/50/100/200 或更短 horizon 下比较 Traditional、RL-driven、Simple、HSS 和 Full，检验学习引导是否只在计算受限时有价值；
2. **复杂障碍 benchmark**：先确保统一路径可行性与至少一个强方法可完成，再比较 static corridor、U-trap、lab complex 和动态障碍；
3. **动力学失配定向复现**：对 combined-unseen 的 80% success 使用全新 seeds 做单独确认；
4. **真实小车 offline/shadow**：验证 reliability 是否与真实 prediction-execution error 单调关联；
5. value alignment 若继续作为主贡献，必须用未见任务重新训练/评估并得到独立正向证据，否则降为探索性模块。

## 10. 完整性、异常与溯源

- `progress.csv` SHA256：`d9c24e52286a438f6719d0839cf7152383315bd5f9c41a964d1c2d0e8546bd46`；
- `provenance.json` SHA256：`b157e1467a85dc841eb4cff301e108ef4cdb73e0a15ca42ef13cb307889d0fbf`；
- 仿真 Git SHA：`bcf9b32`；
- 210 行均为 formal、0 行 qualification；
- 30 个 block 均包含七个不同 arm；
- 每个 arm 恰有 30 episodes；
- 每个运行目录包含 resolved config、trajectory、metrics 和 provenance；
- 统计使用固定 seed 20260719 与 10,000 cluster bootstrap samples。

所有 episode 完成后，原 runner 的后处理曾因新字段 `benchmark_arm` 与旧 helper 的 `factorial_arm` 不一致而退出。原始 210 个 episode、每臂 CSV、schedule 和 provenance 已在退出前完整写入。修复仅新增只读分析器并从不变的 `progress.csv` 重新生成统计；没有重跑、覆盖或筛选 episode。

## 11. 结果路径

仓库内完整生成结果：

```text
results/research_platform/rl/final_paper_point_goal_l214/
```

关键文件：

- `progress.csv`：210 个正式 episode；
- `paired_comparisons.json`：十个预先规定的配对比较；
- `factorial_contrasts.json`：冻结 2×2 因子效应；
- `descriptive_summary.csv`：总体与分域描述性汇总；
- `analysis_audit.json`：完整性与输入哈希；
- `figures/fig_l214_method_overview.{pdf,png}`；
- `figures/fig_l214_core_ablation.{pdf,png}`；
- `runs/`：210 个完整运行目录。

Windows 桌面交付包包含上述完整原始结果、报告、冻结配置、测试记录和图表。Git 仅提交紧凑证据、脚本和报告；大型逐步轨迹仍遵守仓库 `.gitignore`。
