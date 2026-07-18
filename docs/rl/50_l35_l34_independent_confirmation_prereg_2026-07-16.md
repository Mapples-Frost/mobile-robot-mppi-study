# L35：L34 有界 RL 修正的独立确认实验预注册

日期：2026-07-16  
状态：在读取任何 L35 闭环结果前冻结。

## 1. 目的与证据边界

L34 在三个独立训练 seed 的开发集上，将 pooled success 从 `4/18` 提高到
`10/18`，并将 collision 从 `12/18` 降低到 `7/18`。但是同一批固定 validation
episode 同时参与了 checkpoint 选择，因此它们只能支持开发结论，不能作为独立确认。

L35 只检验以下问题：

> 在不再训练、不再选 checkpoint、不再调整混合系数的前提下，L34 的训练后模型
> 是否能在全新闭环 episode seed 上稳定优于各自的 step-0 模型？

本实验仍固定 nominal rollout dynamics，因而不能用来声称 ICODE 与 RL 已产生协同作用。
只有 L35 通过后，才进入 Traditional/RL × Nominal/ICODE 的正交因子实验。

## 2. 冻结因素

- 三个训练 seed：`20260731`、`20260732`、`20260733`；
- checkpoint：每个训练 run 的 `initial.pt` 与开发阶段已经选定的 `best.pt`；
- best step：分别为 `30k`、`30k`、`15k`；
- 推理混合：`fixed alpha = 0.25`；
- rollout dynamics：`nominal`；
- 场景：held-out diagonal 与 held-out offset 两种动态障碍运动路径；
- MuJoCo 物理域：`combined_unseen`；
- 每个场景 5 个全新 episode seed：
  `20460731`–`20460735`；
- temporal scan guard、scan guard、local obstacle layer 与最终安全仲裁保持开启；
- memory 维持关闭；planner 仍只能使用 LaserScan 派生的局部障碍物。

两个场景、五个 episode seed、三个训练 seed、两个 checkpoint role 形成
`2 × 5 × 3 × 2 = 60` 个闭环 episode。相同场景和 episode seed 在 initial/best
之间配对。训练 seed 是独立模型重复；控制 step 不是独立样本。

## 3. 主要终点与分析

主要终点按以下顺序报告：

1. success；
2. collision；
3. final goal distance。

同时完整保留轨迹长度、最小间隙、控制 jerk、规划耗时和安全仲裁次数，但这些指标
在本轮属于次要描述性终点，不能替代主要 Gate。

二元终点使用相同模型、场景、episode seed 的配对差异，报告 gain/loss 计数和 exact
paired sign test。连续终点报告配对均值差。置信区间使用分层 bootstrap：先重采样训练
seed，再在模型内重采样场景—episode 单元，以避免把同一模型的多个 episode 错当成
完全独立的模型重复。

## 4. 预注册通过条件

L35 只有同时满足以下条件才通过：

1. 60 个 episode 齐全、无重复 key、无 NaN/Inf，配置和 checkpoint 哈希完整；
2. 至少 `2/3` 个训练 seed 的 best success 高于各自 initial；
3. 至少 `2/3` 个训练 seed 的 best collision 不高于各自 initial；
4. 汇总 30 个配对单元后，success gains 减 losses 至少为 `3`；
5. collision regressions 减 improvements 不大于 `0`；
6. 三个训练 seed 等权后的平均 final-goal-distance improvement 至少为 `0.10 m`。

Gate 不依赖事后挑选 p 值。本轮 exact test 与 bootstrap CI 用于量化证据强度和不确定性，
不会被替换为更有利的检验。

## 5. 失败处理

若 L35 失败：

- 不进入 2×2 因子实验；
- 不解封旧 L29–L32 的 confirmation seeds；
- 不在这五个新 seed 上搜索新的 alpha、奖励或 checkpoint；
- 将结果保留为“有界 RL 修正在开发集有效但独立复现不足”，下一假设必须使用新的
  开发集提出并重新预注册。

若 L35 通过：冻结三套 best checkpoint，并预注册以下 2×2 因子实验：

```text
traditional prior × nominal dynamics
traditional prior × ICODE dynamics
bounded RL prior × nominal dynamics
bounded RL prior × ICODE dynamics
```

## 6. 运行入口

```bash
python3 experiments/rl/run_l34_independent_confirmation.py \
  --config configs/rl/l34_independent_confirmation_l35.yaml \
  --output-dir results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1

python3 experiments/rl/summarize_l34_independent_confirmation.py \
  --config configs/rl/l34_independent_confirmation_l35.yaml \
  --input-dir results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1
```

