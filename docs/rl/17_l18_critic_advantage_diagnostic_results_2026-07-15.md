# L18：SAC critic 相对 BC 优势诊断结果

日期：2026-07-15  
性质：validation 诊断，不是 sealed final test  
预注册：[`16_critic_advantage_diagnostic_prereg_2026-07-15.md`](16_critic_advantage_diagnostic_prereg_2026-07-15.md)

## 1. 结论先行

L18 的代码与数据质量 Gate 通过，但 online/target critic 的零阈值优势门控均未通过预注册性能 Gate：

- 三个独立 training seeds、每个 15 个配对 validation episodes；
- 对应 BC 共成功 `42/45`；
- 完整 correction、online hard gate、target hard gate 均为 `36/45`；
- 四个条件均为 `0/45` collision；
- online gate 平均接受 `77.65%` 控制步，target gate 平均接受 `76.64%`，因此两者都不是全开或全关；
- 但 online/target gate 各自仍损失了 `7` 个 BC 原本成功的 episode，只各挽救 `1` 个 BC 原本失败的 episode；
- 两个 hard gate 均未改善完整 correction 的 pooled success；
- 因此本轮不进入新 final test，也不部署 critic advantage gate。

准确解释是：critic advantage 含有一定相对排序信息，但 `A_min >= 0` 不是足以保护 BC 的可靠判据。它不是“置信概率”，也没有校准成安全阈值。

## 2. 方法与边界

对冻结 BC action 与完整 correction action，分别计算两个 critic 的配对差值：

\[
A_i(s)=Q_i(s,a_{\mathrm{corr}})-Q_i(s,a_{\mathrm{BC}}),
\qquad
A_{\min}(s)=\min(A_1(s),A_2(s)).
\]

每个控制步同时记录 online critics 与 target critics。诊断条件为：

| condition | correction 规则 |
|---|---|
| `bc` | L17 `initial.pt`，deterministic correction 精确为零 |
| `correction_none` | L17 20k correction 完整生效 |
| `correction_online_hard` | online `A_min >= 0` 时生效，否则恢复 BC latent action |
| `correction_target_hard` | target `A_min >= 0` 时生效，否则恢复 BC latent action |

门控发生在 frozen-BC correction 层，不是外层 GoalWarmStart/OOD gate。ICODE、Memory、动态障碍和 RL 再训练均关闭。

## 3. 配对 validation 结果

### 3.1 跨训练种子汇总

| condition | success | BC success losses | success gains | collision | mean return delta | mean goal-distance improvement | accept fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| full correction | 36/45 | 8 | 2 | 0/45 | -4.458 | -0.0940 m | 1.000 |
| online hard | 36/45 | 7 | 1 | 0/45 | -4.355 | -0.0861 m | 0.777 |
| target hard | 36/45 | 7 | 1 | 0/45 | -4.530 | -0.0764 m | 0.766 |

`goal-distance improvement = BC distance - candidate distance`，因此负数表示候选更差。pooled episode 仅用于描述；独立训练重复仍只有 3 个。

### 3.2 每个 training seed 的成功数

| training seed | BC | full correction | online hard | target hard |
|---:|---:|---:|---:|---:|
| 20260721 | 15/15 | 12/15 | 13/15 | 11/15 |
| 20260722 | 14/15 | 10/15 | 9/15 | 11/15 |
| 20260723 | 13/15 | 14/15 | 14/15 | 14/15 |
| pooled | **42/45** | **36/45** | **36/45** | **36/45** |

seed `20260723` 单独看是正向的，但另外两个独立训练种子明显退化，不能选择性报告 seed `20260723`。

## 4. critic 到底学到了什么

结果不是简单的“critic 完全随机”：

- full correction 的 pooled online advantage 与 paired return delta 的 Pearson 相关为 `0.602`，Spearman 为 `0.367`；
- online hard 的 pooled Spearman 为 `0.310`；
- target hard 的 pooled Spearman 为 `0.315`；
- 预注册的“至少 2/3 training seeds 相关方向为正”检查通过。

但它也不是可直接使用的可靠门控：

- online hard 损失的 7 个 BC-success episodes，其 episode mean online conservative advantage 全部仍为正，范围约 `0.0040--0.0180`；
- target hard 损失的 7 个 BC-success episodes，其 episode mean online conservative advantage 也全部为正；
- 门控在这些失败 episode 中仍接受约 `48%--74%` 的控制步；
- training seed `20260723` 的 gate 内相关甚至转为负方向；
- 零阈值门控只是重新分配了成功/失败 episode，没有提高 pooled success。

因此当前 critic 更像一个带噪声的相对排序器，而不是已经校准的“是否允许 RL 介入”判别器。actor 训练目标本身会推动 correction 获得正的 critic advantage，所以“advantage 大于零”并不是独立证据。

## 5. 预注册判定

| 判据 | online hard | target hard |
|---|---:|---:|
| 至少 2/3 training seeds 相关方向为正 | 通过 | 通过 |
| 不损失任何 BC-success episode | **失败** | **失败** |
| 不增加 collision | 通过 | 通过 |
| gate 非全开/全关 | 通过 | 通过 |
| 至少有 success gain 或距离收益 | 通过 | 通过 |
| 最终 eligible | **否** | **否** |

按预注册规则，只要 BC non-inferiority 失败，就不能进入新 final test。不能在同一 validation 集上扫描阈值直到结果好看，然后再把该结果称为确认性证据。

## 6. 数据质量与复现检查

| training seed | episode rows | paired rows | control-step rows | duplicate keys | missing critical fields | NaN/Inf | BC raw correction max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | 60 | 45 | 16,262 | 0 | 0 | 0 | 0.0 |
| 20260722 | 60 | 45 | 17,673 | 0 | 0 | 0 | 0.0 |
| 20260723 | 60 | 45 | 15,851 | 0 | 0 | 0 | 0.0 |

其他合同检查：

- `correction_none` 精确复现 L17 validation 的 `12/15、10/15、14/15`；
- BC 精确复现 L17 step-zero 的 `15/15、14/15、13/15`；
- 标准 `ExperimentRunner` 的 seed `20280701` target-hard 结果与诊断 runner 一致：success 为 true，final distance 为 `0.2996659448 m`，accept fraction 为 `0.8787878788`；
- 并行执行只用于缩短数据采集时间，本轮不比较 planner wall-clock；
- `scan_guard`、安全仲裁、MuJoCo plant、MPPI cost、ROS bridge 均未修改。

聚合文件：

```text
results/research_platform/rl/
  l18_advantage_validation_multiseed_20260715_v1/
    aggregate.json
    paired_episodes_all_training_seeds.csv
```

每个 training seed 的目录还包含 `config_snapshot.json`、`summary.json`、`episodes.csv`、`paired_episodes.csv` 和逐控制步 `steps.csv`。

## 7. 下一步决策

L18 应冻结为“critic 有弱排序信息、零阈值门控不安全”的诊断结果。下一轮不能直接扩大 correction，也不能立即把 ICODE、OOD 或动态障碍混入。

合理的下一步是单独研究 **校准与悲观化**：将 validation 再严格划分为 calibration 与 selection，使用与 actor 训练不同的数据估计 conservative margin，或训练独立 critic ensemble / cost critic 来判断 correction 风险。只有新门控在多个 training seeds 上满足 BC non-inferiority，才申请新的未见 final seeds。

本轮不支持以下说法：

- 不支持“critic advantage gate 已解决 L17 退化”；
- 不支持“正 advantage 等于可靠置信度”；
- 不支持“继续增加训练步数必然改善”；
- 不支持“RL 已可进入 ICODE+OOD 主实验”。

## 8. 复现命令

单训练种子配对诊断：

```bash
.venv/bin/python experiments/rl/run_correction_advantage_diagnostic.py \
  --config configs/rl/sac_mppi_utrap_critic_advantage_l18.yaml \
  --scene-config configs/research/mujoco_u_trap_long_board.yaml \
  --baseline-checkpoint RUN/checkpoints/initial.pt \
  --candidate-checkpoint RUN/checkpoints/step_000020000.pt \
  --training-seed 20260721 \
  --seeds 20280701,20280702,20280703,20280704,20280705,20280706,20280707,20280708,20280709,20280710,20280711,20280712,20280713,20280714,20280715 \
  --output-dir results/research_platform/rl/L18_RUN
```

多训练种子质量审计与聚合：

```bash
.venv/bin/python experiments/rl/summarize_correction_advantage_diagnostic.py \
  --run-dir L18_SEED_1 \
  --run-dir L18_SEED_2 \
  --run-dir L18_SEED_3 \
  --output-dir results/research_platform/rl/l18_advantage_validation_multiseed_20260715_v1
```
