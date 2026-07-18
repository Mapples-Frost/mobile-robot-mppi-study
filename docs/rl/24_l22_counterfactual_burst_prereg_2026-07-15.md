# L22：短时门控 RL 修正序列反事实实验预注册

日期：2026-07-15  
状态：冻结于生成任何 L22 episode 数据之前  
前置证据：L21 的 124 个单步修正分支全部为 neutral；L20 完整闭环中 β=2 相比 BC 有 5 次成功增益和 1 次成功损失。

## 1. 研究问题

L21 已否定“一个孤立 correction 足以形成可学习风险标签”的假设。本轮检验：

> 在完全相同的 MuJoCo、LaserScan、MPPI 历史与随机数流下，将一次修正扩展为固定长度的门控 SAC correction burst，是否会产生可重复的短时收益与风险信号？

本轮只构造和审计数据，不预先承诺训练风险模型。只有预注册的数据充分性 Gate 全部通过，后续才允许训练独立 risk ensemble。

## 2. 干预长度的先验选择

在查看任何 L22 新 seed 前，只使用已完成的 L20 calibration 日志统计 β=2 的连续接受段：

| quantity | value |
|---|---:|
| closed-loop episodes | 36 |
| total steps | 9,856 |
| accepted steps | 5,654（57.37%） |
| accepted-run median | 2 steps |
| accepted-run q75 | 5 steps |
| accepted-run q90 | 11 steps |
| accepted-run maximum | 65 steps |

因此固定 `intervention_steps = 10`。以当前控制周期约 0.1 s 计，它代表约 1 s 的策略影响，接近既有策略 accepted-run 的 90% 分位，但不是依赖极端最长段选择。L22 运行后不得根据结果更改该值；其他长度只能作为新编号实验重新预注册。

## 3. 配对反事实设计

每个样本从 frozen-BC reference trajectory 的同一状态出发：

```text
Baseline branch:
  40 steps frozen BC

Candidate branch:
  first 10 steps: deterministic SAC proposal
                  -> target twin-critic consensus LCB, beta = 2
                  -> accepted: SAC correction
                  -> rejected: frozen BC action
  next 30 steps: frozen BC
```

约束如下：

1. branch 第一步必须被 β=2 Gate 接受，否则该候选点只计入 coverage，不生成样本；
2. 后续 9 步按候选分支各自 observation 重新计算 SAC proposal 和 Gate；不强迫 Gate 接受；
3. baseline 始终使用其各自 observation 下的 frozen BC；
4. 两分支使用相同 seed、相同历史 action replay 和相同 MPPI 随机数流；
5. MuJoCo plant、sensors、LaserScan、local obstacle layer、scan_guard 和安全仲裁保持开启；
6. Memory、ICODE、OOD 与动态障碍不在本轮同时启用，避免混杂。

这是一种配对、重复测量式模拟实验。branch row 嵌套在 `(training checkpoint, episode seed)` 中，不能把同一 episode 的多个 branch points 当成独立科研重复。

## 4. 固定模型与场景

- scene：`u_trap_long_board`；
- candidate checkpoints：训练种子 `20260721/20260722/20260723` 各自的 `step_000020000.pt`；
- reference policy：每个 checkpoint 内冻结且哈希校验的 BC actor；
- candidate proposal：同一 checkpoint 的 deterministic SAC actor；
- Gate：target twin-critic consensus LCB，`beta = 2`；
- branch steps：`10, 30, 50, 70, 90, 110`；
- branch horizon：40 control steps；
- intervention：前 10 步 gated SAC burst。

## 5. 数据划分与停止规则

本轮使用全新的 development episode seeds：

```text
train      = 20281401--20281408
validation = 20281409--20281410
```

保留且不得打开：

```text
sealed risk-model test = 20281311--20281315
sealed closed-loop selection = 20281101--20281115
```

三个 checkpoint 均运行同一 train/validation seed schedule。正式收集必须完成预定的 30 个 reference episodes（3 checkpoints × 10 seeds）；不得在看到标签数后提前停止或追加同分布 seed。若数据不足，结论是本设计未通过，后续扩展必须另行预注册。

## 6. 结果与标签

连续结果沿用 L21：

\[
\Delta R_{40}=R_{40}^{\mathrm{burst}}-R_{40}^{\mathrm{BC}},
\]

\[
\Delta d_{40}=d_{40}^{\mathrm{BC}}-d_{40}^{\mathrm{burst}}.
\]

正的 \(\Delta d_{40}\) 表示 burst 分支更接近目标。三分类阈值不因 L21 结果而改变：

```text
harmful:
  collision_regression
  OR success_loss
  OR (return_delta <= -0.5 AND distance_improvement <= -0.03 m)

beneficial:
  not harmful
  AND (success_gain
       OR (return_delta >= +0.5 AND distance_improvement >= +0.03 m))

neutral:
  otherwise
```

每个 candidate branch 额外记录：burst 请求长度、实际执行长度、Gate 接受/拒绝步数、接受比例与实际 correction 幅值。

## 7. 数据质量与充分性 Gate

数据质量必须同时满足：

1. 所有 history replay observation error ≤ `1e-10`；
2. 无重复 `(training_seed, episode_seed, branch_step)`；
3. CSV/NPZ 行数、schema、split、checkpoint contract 一致；
4. 所有数值有限，无缺失值；
5. candidate 的首步 Gate 必须接受；
6. `0 < accepted burst steps <= 10` 且计数守恒。

只有以下条件也全部满足，才允许训练风险模型：

- train accepted branches ≥ 60；
- validation accepted branches ≥ 12；
- train harmful ≥ 8 且 beneficial ≥ 8；
- validation harmful ≥ 2 且 beneficial ≥ 2；
- 三个 training checkpoints 均贡献 train 和 validation 数据。

不允许通过降低阈值、复制少数类、随机拆分同一 episode、打开 sealed test 或把嵌套 branch rows 当独立重复来绕过 Gate。

## 8. 预期解释边界

- 若出现两类且跨 checkpoint/episode 分布：支持 sequence-level risk learning 的下一阶段；
- 若连续目标拉开但标签仍不足：说明 10-step burst 有效应，但不足以支持预注册分类器；
- 若仍几乎全部 neutral：说明问题不只是干预长度，需检查 actor correction 幅度、任务敏感性或改用 trajectory-level 预测目标；
- 即使通过，也只能声称在当前静态 U-trap 仿真和冻结策略上的反事实可分性，不能直接外推到动态障碍、OOD 或实车；
- 本实验不改变 ICODE 结论，也不把 SAC critic 解释为真实安全概率。

