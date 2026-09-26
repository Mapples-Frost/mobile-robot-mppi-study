# Full Proposed 2×2 因子实验：独立确认结果

日期：2026-07-19  
冻结实现：`cba00cbf3ab65f2350159277da6bb0f4cdb583de`  
预注册：`docs/rl/180_full_proposed_factorial_prereg_2026-07-19.md`  
单次修订：`docs/rl/181_completion_preserving_hss_amendment_2026-07-19.md`

## 1. 本轮回答的问题

本轮不再检验“ICODE 是否能降低预测误差”这一已完成问题，而是检验冻结方案的
两个新增因素在相同 MPPI 预算下是否具有闭环价值：

1. **value-aligned ICODE**：critic-informed、competence-gated 的价值对齐残差
   动力学；
2. **reliability-adaptive HSS**：根据 ICODE ensemble reliability 在
   0/30/60% 间调节 persistent Actor 候选比例，并在终段保留 30% 完成性下限。

四个随机区组实验臂为：

| 实验臂 | ICODE | Actor Hybrid Sampling |
|---|---|---|
| `ordinary_fixed` | ordinary ensemble | 固定 30% |
| `value_fixed` | value-aligned ensemble | 固定 30% |
| `ordinary_adaptive` | ordinary ensemble | 自适应 0/30/60%，含终段下限 |
| `full_proposed` | value-aligned ensemble | 自适应 0/30/60%，含终段下限 |

所有实验臂均使用同一个冻结 SAC Actor、target critic、terminal-value weight、
\(K=100\)、2 次 MPPI refinement、36-step horizon、LaserScan 感知和安全仲裁。
因此 `ordinary_fixed` 已经是“ICODE + RL prior/value”的简单组合，而不是弱
nominal MPPI 对照。

## 2. 独立确认设计

- 场景：`clean_single_obstacle`；
- 种子：51–55，开发阶段未运行；
- 物理域：nominal seen、long-delay seen、combined unseen；
- 每个 seed × physics block 包含全部四个实验臂；
- 随机区组顺序由 seed `20260736` 冻结；
- episode 上限：300 control steps；
- 统计独立单位：seed，不把控制 timestep 当作样本；
- 区组内配对后进行 seed-cluster bootstrap，10,000 次重采样；
- 总计：5 seeds × 3 domains × 4 arms = 60 episodes。

## 3. 确认集主结果

### 3.1 各实验臂绝对结果

| 实验臂 | 成功 | 碰撞 | 最终距离均值 (m) | jerk 均值 | planner 均值 (ms) |
|---|---:|---:|---:|---:|---:|
| ordinary fixed | 9/15 | 0/15 | 0.47896 | 0.09955 | 186.28 |
| value fixed | 14/15 | 0/15 | 0.30794 | 0.10054 | 186.78 |
| ordinary adaptive | 13/15 | 0/15 | 0.31899 | 0.09833 | 183.44 |
| **full proposed** | **14/15** | **0/15** | **0.29553** | 0.10068 | 187.73 |

### 3.2 Full Proposed 相对简单组合

下表的 effect 已统一为“正值更好”：

| 指标 | favorable effect | seed-cluster 95% CI | 判定 |
|---|---:|---:|---|
| success | **+0.3333** | **[+0.1333, +0.5333]** | 严格有利 |
| final distance | **+0.18343 m** | **[+0.14505, +0.22166]** | 严格有利 |
| collision | 0 | [0, 0] | 不增加 |
| jerk | −0.00113 | [−0.00426, +0.00201] | 均值增加 1.13%，满足预注册 5% 非劣界 |
| planner time | −1.45 ms | [−4.75, +2.53] | 无确定差异 |

因此独立确认 Gate 通过：

1. success 与 final distance 方向均有利；
2. 两个主要进度指标的 95% CI 均严格有利；
3. 碰撞没有增加；
4. jerk 的相对变化在预注册工程非劣界内；
5. 三个物理域均已报告；
6. 每拍 rollout 预算严格相同。

### 3.3 分物理域结果

| 实验臂 | 物理域 | 成功 | 最终距离均值 (m) |
|---|---|---:|---:|
| ordinary fixed | nominal seen | 3/5 | 0.45857 |
| ordinary fixed | long-delay seen | 2/5 | 0.58999 |
| ordinary fixed | combined unseen | 4/5 | 0.38832 |
| full proposed | nominal seen | 4/5 | 0.31088 |
| full proposed | long-delay seen | 5/5 | 0.28973 |
| full proposed | combined unseen | 5/5 | 0.28598 |

完整方法相对简单组合的优势不是由单一物理域驱动；seen、delay 和 unseen 三个
域的成功率或最终距离方向均有利。

## 4. 消融与交互效应

### 4.1 两个模块分别有效

`value_fixed` 相对 `ordinary_fixed`：

- success：+0.3333，95% CI [+0.1333, +0.5333]；
- final distance：+0.17102 m，95% CI [+0.14693, +0.20265]；
- planner time：−0.50 ms，95% CI [−0.87, −0.03]，即有小幅额外开销。

`ordinary_adaptive` 相对 `ordinary_fixed`：

- success：+0.2667，95% CI [+0.1333, +0.3333]；
- final distance：+0.15997 m，95% CI [+0.10000, +0.21993]；
- planner time：+2.83 ms，95% CI [+1.05, +4.66]，即严格节省计算时间。

因此确认集分别支持：

1. value-aligned ICODE 改善闭环任务完成；
2. reliability-adaptive HSS 改善闭环任务完成并降低平均规划耗时。

### 4.2 不支持超加性协同

成功率交互项为 −0.2667，95% CI [−0.4667, −0.0667]。两个模块单独已经修复
了部分相同失败，完整组合受到成功率上限效应影响，没有获得两项改善的简单
相加。其他主要交互项也没有形成有利的严格区间。

因此论文可以说：

> 完整方案显著优于普通 ICODE 与固定 Actor guidance 的简单组合，并且两个
> 机制各自具有可复现的主效果。

论文不得说：

> ICODE 与 RL 已经被证明具有超加性 synergy。

当前更准确的叙事是**相互校准、功能互补且可消融**，而不是“组合收益大于两个
模块单独收益之和”。

## 5. `lab_complex` 鲁棒性补充

严格沿用相同四臂、种子和三个物理域，只把场景换为 `lab_complex`，并使用
预注册的 180-step 截断。共运行 60 个额外 episode。

| 实验臂 | 成功 | 碰撞 | 最终距离均值 (m) | jerk 均值 | planner 均值 (ms) |
|---|---:|---:|---:|---:|---:|
| ordinary fixed | 0/15 | 0/15 | 2.23214 | 0.10857 | 188.38 |
| value fixed | 0/15 | 0/15 | 2.15369 | 0.10949 | 189.21 |
| ordinary adaptive | 0/15 | 0/15 | 2.24097 | 0.10366 | 177.55 |
| full proposed | 0/15 | 0/15 | 2.20246 | 0.10628 | 178.44 |

该补充按预注册不是成功率主实验。180 steps 内所有方法均未到达目标，说明复杂
导航几何而不是动力学学习成为主导瓶颈。重要的边界结果是：

- 所有方法仍为 0 碰撞；
- Full Proposed 相对 ordinary fixed 的最终距离方向略有利，但 CI 跨 0；
- Full Proposed 的 jerk 方向有利，但 CI 跨 0；
- Full Proposed 平均规划时间节省 9.94 ms，95% CI
  [+8.53, +10.88] ms；
- 自适应 HSS 在 ordinary/value-aligned ICODE 下均节省约 10.8 ms，并在
  value-aligned 条件下严格降低 jerk。

所以复杂场景补充支持“方法没有破坏安全并可降低计算量”，但不支持
“180-step lab_complex 导航成功率提高”的结论。

## 6. 结果路径与复现

开发集：

```text
results/research_platform/rl/full_proposed_remediation_seed{48,49,50}_l208/
results/research_platform/rl/full_proposed_remediation_development_l209/
```

确认集：

```text
results/research_platform/rl/full_proposed_confirmation_seed{51..55}_l210/
results/research_platform/rl/full_proposed_confirmation_l211/
```

复杂场景补充：

```text
results/research_platform/rl/full_proposed_lab_seed{51..55}_l212/
results/research_platform/rl/full_proposed_lab_supplement_l213/
```

每个 shard 保存逐 episode `progress.csv`、逐方法结果目录、trajectory、
metrics 和 provenance。合并目录保存完整 2×2 结果、配对比较和 factorial
contrast。大体积结果遵循 `.gitignore`，不提交到 Git。

## 7. 结论边界

本轮结果支持：

1. Full Proposed 在测试的 MuJoCo clean dynamics task 上优于强简单组合；
2. value-aligned ICODE 和 reliability-adaptive HSS 各自具有闭环主效果；
3. 结果覆盖 seen、delay 和 unseen 物理域；
4. 等 rollout 预算下不增加碰撞，jerk 满足非劣界；
5. 在复杂几何补充中自适应 HSS 仍能降低计算耗时。

本轮结果不支持：

1. 超加性 ICODE × RL synergy；
2. 形式化稳定性、收缩性或收敛性保证；
3. 复杂场景的全局导航成功率改善；
4. 实车泛化；
5. 原始 ICODE 理论的完整复现。

