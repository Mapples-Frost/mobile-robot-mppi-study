# L17 v3：保守 SAC correction 的结果与数据质量报告

## 1. 结论先行

L17 v3 的工程安全目标通过，性能目标未通过：

- 三个 training seeds 的 fail-closed selector 均保留 step 0；
- learned correction 在 validation 中确实改变了行为，但频繁损失原本成功的回合；
- selected checkpoint 因而全部等价于对应 BC；
- 新 sealed final test 上 BC 与 selected 逐回合任务指标完全一致；
- pooled success 为 `54/60`，collision 为 `0/60`；
- 当前没有证据支持“RL correction 优于 BC”。

这是一项有效的负结果：框架现在能在 RL 没有提供可靠收益时自动拒绝它，而不是因为某个平均 return 较高就部署退化策略。

## 2. 实际实现的研究机制

### 2.1 Correction 正则

冻结 BC base actor，只训练有界 SAC correction，并在 actor loss 中加入：

\[
\mathcal L_{\mathrm{corr}}=
\frac{1}{B d_a}\sum_{i,j}
\left(\frac{\Delta a_{ij}}{\Delta a^{\max}_{j}}\right)^2,
\qquad
\mathcal L_{\mathrm{actor}}=
\mathcal L_{\mathrm{SAC}}+0.25\mathcal L_{\mathrm{corr}}.
\]

训练日志记录原始 penalty、加权 penalty、unit correction 和实际 applied correction。

### 2.2 Paired fail-closed selector

每个 checkpoint 与同一个 training seed 的 step 0 在完全相同的 15 个 scene/seed 行上比较。任何 reference success loss、新 collision 或平均终点距离增加都会拒绝候选；没有至少 1 个 success gain 或 `0.005 m` 平均距离改善也不会更新 best。

selector 的 step-zero reference、阈值、best rank 和 best step 进入 checkpoint 与 resume contract。缺失 reference state 时拒绝续训。

### 2.3 Seed 与 validation contract 修复

本轮审计发现并修复了两个会破坏复现性的边界：

1. `MppiPriorEnv.reset(seed)` 过去没有用 episode seed 重置 MPPI sampling RNG；
2. training initial-state jitter 过去会隐式进入 validation，而标准 evaluator 使用精确 benchmark 起点。

最终 v3 contract 为：training 保留起点扰动；validation 使用零起点扰动；episode seed 同时控制 plant、sensors 和 MPPI sampling。修复后，trainer 内部 validation 与标准 checkpoint evaluator 的 success、终点距离和轨迹指标一致。

## 3. 数据文件与质量检查

### 3.1 文件格式与来源

主要数据是 UTF-8 CSV：

- `updates.csv`：每个 training seed 18,001 行、27 列，约 6.86--6.89 MB；
- `validation_episodes.csv`：每个 seed 75 行、24 列，约 19 KB；
- final `episodes.csv`：每个方法/seed 20 个未见回合；
- checkpoint：包含 agent、normalizer、优化器、selector state、配置与 provenance。

CSV 是纯文本实验表格，使用 Python 标准库逐行读取；检查了行列数、字段完整性、重复行、复合主键和数值聚合。

### 3.2 完整性

三个 training seeds 均满足：

- `updates.csv` 无完全重复行；
- `validation_episodes.csv` 无重复 `(global_step, scene, seed)`；
- correction penalty、weighted penalty、unit/applied correction 字段无缺失；
- loss 与 correction 指标均为有限值；
- v2/v3 对应 20k actor SHA-256 一致，证明 validation 起点合同的改变没有暗中改变训练权重。

20k actor SHA-256 前缀：

| training seed | SHA-256 前缀 | v2 = v3 |
|---:|---|---:|
| 20260721 | `01ebd5dea6ca9bdd` | true |
| 20260722 | `571926db5e505d89` | true |
| 20260723 | `7667a978e04e28c9` | true |

## 4. Validation 结果

所有 checkpoint collision 均为 `0/15`。核心结果如下：

| training seed | step 0 success | step 15k success | step 20k success | step 0 distance (m) | step 15k distance (m) | step 20k distance (m) | selected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | 15/15 | 9/15 | 12/15 | 0.2930 | 0.5693 | 0.4290 | 0 |
| 20260722 | 14/15 | 6/15 | 10/15 | 0.3417 | 0.7313 | 0.5344 | 0 |
| 20260723 | 13/15 | 12/15 | 14/15 | 0.3866 | 0.4414 | 0.3398 | 0 |

seed `20260723` 的 20k checkpoint 表面上从 13 个成功增加到 14 个，平均距离也下降，但它同时让 1 个原本成功的 paired episode 失败；fail-closed 规则因此拒绝它。这正是逐回合配对规则与只看平均成功率的区别。

确定性 validation correction 诊断：

| training seed | step 15k applied abs mean | step 15k max | step 20k applied abs mean | step 20k max |
|---:|---:|---:|---:|---:|
| 20260721 | 0.04458 | 0.14151 | 0.02516 | 0.12052 |
| 20260722 | 0.03698 | 0.16852 | 0.02492 | 0.15443 |
| 20260723 | 0.04159 | 0.16031 | 0.02778 | 0.12473 |

RL 不是“没有学到任何东西”，而是学到的 correction 没有可靠转化为控制收益。

## 5. Sealed final test

final seeds `40301--40320` 仅在三个 selected checkpoint 和 validation/evaluator 一致性全部冻结后打开。每个 training seed 的 BC 与 L17 v3 selected 任务行逐字段一致。

| training seed | BC success | selected success | collision | final distance (m) | minimum clearance (m) |
|---:|---:|---:|---:|---:|---:|
| 20260721 | 18/20 | 18/20 | 0/20 | 0.3611 | 0.3235 |
| 20260722 | 20/20 | 20/20 | 0/20 | 0.2942 | 0.3176 |
| 20260723 | 16/20 | 16/20 | 0/20 | 0.4043 | 0.3335 |
| pooled | **54/60** | **54/60** | **0/60** | **0.3532** | **0.3249** |

并行运行会造成 CPU 竞争，因此本表不使用 planner wall-clock time 作方法比较。两种 checkpoint 的 Python 推理封装不同，计时差异也不能在本轮被解释为控制性能差异。

聚合结果位于：

```text
results/research_platform/rl/
  bc_vs_l17_v3_conservative_correction_finaltest_20260714_v1/
```

## 6. 研究解释与限制

本轮支持：zero correction 精确恢复 BC；selector 能阻止退化 RL 部署；checkpoint、resume、validation 与标准 evaluator 合同可审计；安全链保持零碰撞。

本轮不支持：SAC correction 优于 BC；当前 penalty weight 已最优；RL 已能进入 OOD/动态障碍主实验；RL、ICODE 与 Memory 联合方法已经验证。60 个 final episodes 也不能被当作 60 个独立训练重复，独立训练重复仍只有 3 个。

## 7. 下一步建议

L17 应作为“安全退化负对照”冻结。下一轮不宜直接扩大 correction 上限或继续堆场景。优先检验 critic 是否高估偏离 BC 的动作，以及 correction 是否应只在相对 BC 的优势超过保守阈值时生效。可考虑 advantage-filtered correction 或显式 trust region，但必须单独消融，并继续保持 ICODE、Memory 和动态障碍关闭。

## 8. 可复现入口

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_conservative_correction_l17_v3.yaml \
  --seed 20260721 --steps 20000 \
  --initialize-actor-from \
    results/research_platform/rl/bc_l13_seed20260721_20260714/checkpoints/best.pt \
  --output-dir \
    results/research_platform/rl/conservative_correction_l17_v3_seed20260721_20k_continuous_20260714_v1
```

```bash
.venv/bin/python experiments/rl/evaluate_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_conservative_correction_l17_v3.yaml \
  --scene-config configs/research/mujoco_u_trap_long_board.yaml \
  --checkpoint RUN_DIR/checkpoints/best.pt \
  --seeds 40301,40302,40303,40304,40305,40306,40307,40308,40309,40310,40311,40312,40313,40314,40315,40316,40317,40318,40319,40320 \
  --pose-source ground_truth --twist-source ground_truth \
  --output-dir EVAL_DIR
```
