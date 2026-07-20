# L217 复杂场景封存实验科研交付报告

日期：2026-07-20

实验状态：一次性封存确认实验已完成；420/420 回合完整

Git SHA：`9ec885007a56b3c3689422a45e450262fb9a4dc5`
正式 manifest SHA-256：`b4e93d2b383b8cc96e5c6b673dd039f70c0617b51f38b6f8af1909422cf674d2`

## 1. 结论先行

本轮结果是**有实质正向证据、同时存在明确代价和机制缺口**的结果，不能表述为“所有耦合机制已经被完整证明”。

在 10 个从未用于开发的封存 seeds、3 个静态复杂障碍场景、2 个 MuJoCo 物理域下：

- `Full Proposed` 完成 60/60，碰撞 0/60；
- `Simple Combination`、`Value-aligned`、`Role-aware HSS` 也均完成 60/60，碰撞均为 0/60；
- `RL-driven MPPI` 完成 59/60；
- `Traditional MPPI` 与 `ICODE-MPPI` 在这组 600 步上限的复杂 point-goal 任务中均为 0/60；
- Full 相对 Simple 平均减少 17.42 个控制步，seed-cluster bootstrap 95% CI 为 `[9.08, 26.05]`；
- Full 相对 Simple 平均缩短轨迹 0.143 m，95% CI 为 `[0.092, 0.191]` m；
- Full 相对 Simple 平均减少卡滞 8.82 步，95% CI 为 `[2.57, 15.07]`；
- 代价是平均最小间隙减少 0.0168 m、control jerk 增加 0.00581、平均 planner 时间增加约 6.40 ms。

因此，本轮**强力支持“RL prior 是复杂导航可达性的关键来源，role-aware HSS 可以在等 rollout 预算下进一步提高完成效率”**。但它只弱支持 value-aligned ICODE 的独立控制收益，而且没有执行完整的 residual-conditioned policy context 与 reliability-weighted terminal value。完整论文主张仍需新一轮、全新封存种子的机制补齐确认。

## 2. 研究问题与冻结设计

本轮正式比较七种方法：

1. Traditional MPPI；
2. ICODE-MPPI；
3. RL-driven MPPI；
4. Simple Combination；
5. Value-aligned；
6. Role-aware HSS；
7. Full Proposed。

实验矩阵为：

```text
7 methods × 3 scenes × 2 physics domains × 10 sealed seeds
= 420 episodes
= 60 randomized complete blocks
```

场景为 `lab_complex`、`narrow_corridor`、`u_trap_long_board`。物理域为训练/校准范围内的 `nominal_seen` 与组合未见失配 `combined_unseen`。每个控制时刻的模型 rollout 预算固定为 100；RL variants 使用 2 轮、每轮 50 条候选。最大回合长度为 600 control steps。

独立统计单位是 seed。scene 和 physics domain 是同一 seed 内的重复分层，不能当作 60 个独立样本。所有区间均按 seed cluster 重采样 10,000 次；bootstrap seed 固定为 `20260724`。

## 3. 数据完整性与可追溯性

自动完整性审计结果：

| 审计项 | 结果 |
|---|---:|
| 正式回合 | 420/420 |
| 唯一 method-scene-domain-seed 单元 | 420 |
| 重复单元 | 0 |
| 完整随机区组 | 60/60 |
| qualification 行 | 0 |
| 独立封存 seeds | 10 |
| 每回合四项原始文件 | 420/420 完整 |
| resolved-config 哈希 | 420/420 一致 |
| 外部 checkpoint/calibration 哈希 | 9/9 一致 |
| Git SHA | 三个 shard 及 420 个回合一致 |

每个回合均保留：

```text
config_resolved.yaml
trajectory.csv
metrics.json
provenance.json
```

三个并行 shard 分别包含 168、126、126 回合，种子集合互斥；合并后恰好覆盖 seeds 78006--78015。没有重跑、删失败、筛 seed 或混入 qualification seed 99。

## 4. 七方法总体结果

| 方法 | 成功率 | 碰撞率 | 平均步数 | 最终距离/m | 轨迹长度/m | 最小间隙/m | 卡滞步数 | Jerk | Planner mean/ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Traditional MPPI | 0.0% | 0.0% | 600.0 | 3.3989 | 0.9955 | 0.1592 | 481.45 | 0.17017 | 4.79 |
| ICODE-MPPI | 0.0% | 0.0% | 600.0 | 3.3898 | 1.0011 | 0.1632 | 480.98 | 0.16268 | 97.70 |
| RL-driven MPPI | 98.3% | 0.0% | 429.15 | 0.2983 | 6.9269 | 0.3635 | 85.63 | 0.07616 | 31.77 |
| Simple Combination | 100% | 0.0% | 429.35 | 0.2983 | 6.8506 | 0.3585 | 90.27 | 0.07443 | 187.26 |
| Value-aligned | 100% | 0.0% | 439.10 | 0.2979 | 6.8346 | 0.3599 | 101.35 | 0.07375 | 186.96 |
| Role-aware HSS | 100% | 0.0% | 417.70 | 0.2976 | 6.7360 | 0.3362 | 83.65 | 0.07937 | 196.60 |
| Full Proposed | 100% | 0.0% | **411.93** | 0.2978 | **6.7072** | 0.3417 | **81.45** | 0.08024 | 193.65 |

Traditional/ICODE 的低轨迹长度不是优势，而是车辆在起点附近卡住后未完成任务的结果。成功率和步数必须先于轨迹长度解释。

## 5. 主要比较：Full Proposed vs Simple Combination

以下“有利效应”为正表示 Full 更好。

| 指标 | Simple | Full | 有利效应 | 95% seed-cluster CI | 判断 |
|---|---:|---:|---:|---:|---|
| 成功率 | 100% | 100% | 0 pp | `[0, 0]` | 均达到上限，无法证明优越 |
| 碰撞率 | 0% | 0% | 0 pp | `[0, 0]` | 均为零，不是安全保证 |
| 执行步数 | 429.35 | 411.93 | **+17.42** | **`[9.08, 26.05]`** | 明确效率改善 |
| 最终距离/m | 0.29830 | 0.29781 | +0.00049 | `[0.00007, 0.00088]` | 数值显著但实际量级很小 |
| 轨迹长度/m | 6.8506 | 6.7072 | **+0.1434** | **`[0.0924, 0.1912]`** | 明确缩短 |
| 卡滞步数 | 90.27 | 81.45 | **+8.82** | **`[2.57, 15.07]`** | 明确减少 |
| 原地旋转步数 | 6.28 | 5.28 | +1.00 | `[-0.28, 2.25]` | 方向有利但区间跨零 |
| 平均最小间隙/m | 0.3585 | 0.3417 | **-0.0168** | **`[-0.0275, -0.0038]`** | Full 更靠近障碍物 |
| Control jerk | 0.07443 | 0.08024 | **-0.00581** | **`[-0.00682, -0.00446]`** | Full 更不平滑 |
| Planner mean/ms | 187.26 | 193.65 | -6.40 | `[-13.91, -0.12]` | 本次并发负载下更慢 |
| 每拍 rollout | 100 | 100 | 0 | `[0, 0]` | 预算严格相同 |

核心正向结果不是成功率——因为四个组合方法都到达了 100% 上限——而是**在同样成功和碰撞表现、同样 rollout 预算下，Full 更快、路径更短、卡滞更少**。

## 6. 2×2 核心消融

四个核心臂分别为 Simple、Value-only、HSS-only、Full。对 executed steps 的冻结 factorial 统计为：

| 效应 | 原始对比（after − before）/steps | 95% CI | 解释 |
|---|---:|---:|---|
| Value alignment 主效应 | +1.99 | `[-4.03, 8.21]` | 没有独立效率证据 |
| Role-aware HSS 主效应 | **−19.41** | **`[-25.68, -13.56]`** | 明确减少执行步数 |
| Value × HSS 交互 | −15.52 | `[-33.62, 3.40]` | 方向有利，但区间跨零 |

HSS 主效应还缩短轨迹约 0.121 m，但同时降低平均最小间隙约 0.0203 m、增加 jerk 约 0.00571。Value alignment 单独使用时平均步数由 429.35 增至 439.10，卡滞由 90.27 增至 101.35；因此不能声称 value-consistent residual 已经在本轮产生独立的闭环优势。

## 7. Seen 与 Unseen 物理域

Full 在两个物理域和三个场景的 60 个单元中均成功。相对 Simple 的平均步数变化为：

| 场景 | Seen: Simple − Full | Unseen: Simple − Full |
|---|---:|---:|
| Lab complex | +13.1 | +18.9 |
| Narrow corridor | +38.0 | +47.2 |
| Long-board U-trap | −10.6 | −2.1 |

这说明效率收益具有场景异质性：Lab 和 narrow corridor 明显受益，U-trap 中 Full 反而略慢。总体正向均值不是“每个场景都赢”。后续应把场景结构作为预先指定的 moderator，而不是只报汇总平均。

## 8. 机制激活审计：必须修正的论文边界

对逐拍诊断量的回合级汇总发现：

- `Full Proposed` 的 `reliability_hss_enabled_fraction=1.0`；
- Full 的平均 HSS authority 为 0.908，平均 guided fraction 为 0.562；Simple 的 guided fraction 固定为 0.30；
- Full 的平均 ICODE dynamics confidence 仅为 0.0476，因此 `policy_rescue` 路由确实在低模型置信度下保留了 Actor 候选；
- `residual_policy_context_enabled_fraction=0.0`，七种方法均为零；
- `residual_policy_authority_enabled_fraction=0.0`，七种方法均为零；
- Full 的 `terminal_value_authority_mean=1.0`，并未随很低的 dynamics confidence 衰减；
- 因而本轮没有激活 residual-conditioned policy context，也没有激活 reliability-weighted terminal-value authority。

所以 L217 的严格方法名称应是：

> Value-aligned ICODE + generic offline RL prior + role-aware reliability HSS + MPPI

而不是完整的：

> Value-Consistent ICODE + Residual-Conditioned RL Prior + Reliability-Weighted Value/HSS + MPPI

这不是原始数据造假或运行失败，而是**冻结配置与论文目标机制之间的覆盖缺口**。L217 可以作为强基线和 HSS 确认证据，但不能单独作为最终完整方法的唯一主表。

## 9. 对论文方向的证据分级

### 得到强支持

1. 在这些复杂 point-goal 场景中，RL prior 显著改变了可达性；
2. ICODE 与 RL prior 可以在不破坏 scan guard、安全仲裁和等 rollout 预算的前提下联合运行；
3. role-aware HSS 在封存 seeds 上减少了执行步数和路径长度；
4. 该收益在 seen 与 combined-unseen 物理域中均可观察到；
5. 完整 raw trajectory、配置和哈希链支持可复核性。

### 仅得到弱支持或混合证据

1. Full 相比 Simple 的成功率没有提高，因为二者均为 100%；
2. value-aligned ICODE 的独立控制收益未被确认；
3. Value × HSS interaction 方向有利但 95% CI 跨零；
4. Full 更靠近障碍、jerk 更大，存在效率—安全裕度—平滑性权衡；
5. 并发 CPU benchmark 的绝对规划时延不适合直接作为实车实时性结论。

### 本轮没有验证

1. residual-conditioned RL observation/context；
2. reliability-weighted terminal value；
3. 动态障碍物；
4. 多起终点、多方向和大尺度绕行的 route diversity；
5. 实车跨地面泛化；
6. 原始 ICODE 理论中的稳定性、收缩性或收敛性保证。

## 10. 计算时延的解释限制

三个 shard 同时在同一台 CPU 上运行，ICODE ensemble 与 RL 方法会竞争 CPU 资源。因此配对顺序和相同并发环境仍让方法间比较具有参考价值，但 `193.65 ms` 等绝对数字受到系统负载影响。本轮不能直接声称满足某个实车实时频率。

论文若要主张实时性，应追加：单进程、固定 CPU affinity、预热、固定线程数、独立重复的 sequential latency profile；若改用 CUDA，应重新报告设备、驱动、PyTorch/MuJoCo 版本和数据传输开销，不能把 GPU 数据与本轮 CPU 数据混在同一时延表中。

## 11. 下一轮建议

保持论文大方向不变，建议按以下顺序推进：

1. 在 development seeds 上真正启用并测试 `residual_context` 输入，逐拍审计 `residual_policy_context_enabled_fraction > 0`；
2. 让 terminal-value authority 显式读取 dynamics/critic reliability，并设置可验证的激活诊断；
3. 保留已经确认有效的 `policy_rescue` HSS，不在 L217 seeds 上继续调参；
4. 将已批准的大地图、蛇形路线、大 U 型障碍与圆柱阵列实现为新的 MuJoCo 场景；
5. 加入多起终点方向与动态障碍，预先检查每个地图存在安全可行路线；
6. development Gate 通过后，再提交新的 preregistration，并使用从未运行过的全新 sealed seeds；
7. 另做单进程 CPU/GPU latency benchmark；
8. 最后进入实车 offline、shadow mode 和低速安全验证。

## 12. 可复现命令

```bash
python3 experiments/rl/merge_sealed_benchmark_shards.py \
  --shards \
  results/research_platform/rl/complex_navigation_sealed_l217/shard_0_of_3 \
  results/research_platform/rl/complex_navigation_sealed_l217/shard_1_of_3 \
  results/research_platform/rl/complex_navigation_sealed_l217/shard_2_of_3 \
  --output-dir \
  results/research_platform/rl/complex_navigation_sealed_l217/merged_analysis

python3 experiments/rl/analyze_final_paper_benchmark.py \
  --result-dir results/research_platform/rl/complex_navigation_sealed_l217/merged_analysis \
  --seeds 78006,78007,78008,78009,78010,78011,78012,78013,78014,78015 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260724

python3 experiments/rl/summarize_l217_complex_navigation.py \
  --result-dir results/research_platform/rl/complex_navigation_sealed_l217/merged_analysis

MPLBACKEND=Agg python3 experiments/rl/plot_l217_complex_navigation.py \
  --result-dir results/research_platform/rl/complex_navigation_sealed_l217/merged_analysis
```

## 13. 最终科研判断

L217 是一批真实、完整、可复核且有正向贡献的数据。它足以支撑论文中的一个重要实证结论：**在复杂静态导航和未见组合物理失配下，role-aware reliability HSS 能在保持 100% 成功与零碰撞的同时，减少平均完成步数和路径长度。**

但它还不是完整论文主方法的最终决定性数据，因为 residual-conditioned policy 与 reliability-weighted terminal value 没有被实际激活，value-alignment 的独立闭环收益也未被确认。正确做法不是修改或筛掉 L217，而是保留它作为正式基线证据，并用新的、机制覆盖完整的封存实验补齐主张。
