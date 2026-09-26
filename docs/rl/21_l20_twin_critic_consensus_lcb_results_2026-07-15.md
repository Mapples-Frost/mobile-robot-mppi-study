# L20：Twin-Critic 共识 LCB 门控结果与数据审计

日期：2026-07-15  
性质：calibration 阶段负结果，但相对 L19 显著缩小失败范围；selection 未开启  
预注册：[`20_l20_twin_critic_consensus_lcb_prereg_2026-07-15.md`](20_l20_twin_critic_consensus_lcb_prereg_2026-07-15.md)

## 1. 结论先行

L20 工程、回归与数据质量 Gate 全部通过，但六个预注册 beta 均未满足“零 paired BC-success loss”的部署条件，最终保持：

```json
{
  "selected_mode": "bc_fallback",
  "selected_beta": null,
  "selection_reason": "no_preregistered_beta_eligible",
  "selection_seeds_opened": false
}
```

不过，L20 比 L19 更接近目标。`beta=2` 在三个独立训练模型上获得：

- BC：`27/36` success；
- LCB beta 2：`31/36` success；
- 5 个 success gains；
- 1 个 paired BC-success loss；
- 0 collision；
- 三个 training seeds 的 mean paired return delta 均为正；
- pooled mean return delta `+3.761`；
- mean goal-distance improvement `+0.0442 m`；
- mean gate acceptance `0.577`。

因此，尺度不变的 critic 共识惩罚确实比 raw Q margin 更有效地保留收益并过滤有害修正，但现有同一对 target critics 仍不能把所有有害 correction 与有益 correction 分开。严格规则不允许用 5 次救回抵消 1 次原本成功任务的损失，所以不能打开 selection seeds。

## 2. 门控公式

\[
\Delta Q_i=Q_i^{\mathrm{target}}(s,a_{\mathrm{corr}})
-Q_i^{\mathrm{target}}(s,a_{\mathrm{BC}}),
\]

\[
S_\beta=
\frac{\Delta Q_1+\Delta Q_2}{2}
-\beta\frac{|\Delta Q_1-\Delta Q_2|}{2}.
\]

仅当 (S_\beta\ge0) 时使用 RL correction。beta 越大，对两个 critic 的一致性要求越高。beta 1 精确等价于 L18 的 `min(Delta Q1, Delta Q2) >= 0`。

该分数对 Q 的正比例缩放保持决策不变，但仍不是统计置信下界，也没有安全理论保证。

## 3. Pooled calibration 结果

| beta | success | BC-success losses | success gains | collision | mean return delta | distance improvement | accept fraction | eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC | 27/36 | — | — | 0/36 | — | — | — | reference |
| 1.0 | 28/36 | 2 | 3 | 0/36 | +0.996 | -0.0164 m | 0.766 | no |
| 1.5 | 28/36 | 3 | 4 | 0/36 | +0.863 | -0.0127 m | 0.672 | no |
| 2.0 | **31/36** | **1** | **5** | 0/36 | **+3.761** | **+0.0442 m** | 0.577 | **no** |
| 3.0 | 28/36 | 2 | 3 | 0/36 | +0.946 | -0.0126 m | 0.430 | no |
| 5.0 | **31/36** | 2 | **6** | 0/36 | **+3.840** | **+0.0758 m** | 0.295 | no |
| 8.0 | 30/36 | 1 | 4 | 0/36 | +2.810 | +0.0288 m | 0.183 | no |

`success = BC success - paired losses + paired gains`。所有 beta 都至少破坏一条 BC 原本成功的轨迹，因此均违反预注册规则。

## 4. beta 2 的独立训练模型结果

| training seed | BC success | candidate success | loss | gain | mean return delta | distance improvement | accept fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | 11/12 | 12/12 | 0 | 1 | +2.806 | +0.0030 m | 0.615 |
| 20260722 | 8/12 | 9/12 | **1** | 2 | +2.849 | +0.0323 m | 0.499 |
| 20260723 | 8/12 | 10/12 | 0 | 2 | +5.630 | +0.0972 m | 0.617 |

与 L19 的 raw margin 不同，beta 2 没有出现某个训练模型 mean return 退化，接受率也更接近。这支持“无量纲共识惩罚改善了跨模型一致性”，但不能升级为部署结论，因为 seed `20260722` 仍有一条 paired success regression。

## 5. 关键反例与正例

beta 2 的唯一 BC-success loss：

| training seed | episode seed | result | return delta | distance improvement | accept fraction |
|---:|---:|---|---:|---:|---:|
| 20260722 | 20281201 | BC success -> LCB failure | -35.180 | -0.6707 m | 0.350 |

同一 beta 共救回 5 条 BC 失败轨迹：

```text
20260721 / 20281205
20260722 / 20281202
20260722 / 20281212
20260723 / 20281210
20260723 / 20281212
```

这提供了下一阶段的明确监督目标：独立风险信号必须优先拒绝 `20260722/20281201` 一类高风险 correction，同时尽量保留上述五类有益 correction。L20 数据可以用于诊断和设计，但不能再作为 L21 的最终验证集。

## 6. CSV/JSON 数据质量审计

每个 training seed 目录包含 `config_snapshot.json`、`summary.json`、`episodes.csv`、`paired_episodes.csv` 和 `steps.csv`。

| training seed | episode rows | paired rows | control-step rows | CSV columns (episode/paired/step) |
|---:|---:|---:|---:|---:|
| 20260721 | 84 | 72 | 22,440 | 29 / 40 / 45 |
| 20260722 | 84 | 72 | 24,304 | 29 / 40 / 45 |
| 20260723 | 84 | 72 | 23,337 | 29 / 40 / 45 |
| total | **252** | **216** | **70,081** | — |

质量检查结果：

- condition/seed、condition/seed/step 复合键无重复；
- 所有 CSV 单元格无缺失；
- 所有数值无 NaN/Inf；
- 三个 run 的 beta 网格、condition mapping 和 calibration seeds 完全一致；
- LCB mean、half-disagreement、score、beta 与 gate alpha 字段完整；
- BC raw correction 最大值精确为 `0.0`；
- 约 58.7 MB step CSV 均为 UTF-8 明文，可复核到每个控制步。

## 7. 工程交付

新增或扩展：

- `src/mobile_robot_mppi/rl/sac.py`：LCB 共识分数和 fail-closed action filtering；
- `src/mobile_robot_mppi/rl/prior.py`：兼容 `lcb` 开关与 beta 配置；
- `src/mobile_robot_mppi/rl/calibration.py`：跨训练种子 beta selector；
- `src/mobile_robot_mppi/evaluation/metrics.py`：标准评估链记录 LCB 指标；
- `experiments/rl/run_correction_advantage_lcb_grid.py`：单模型配对闭环网格；
- `experiments/rl/select_correction_advantage_lcb.py`：跨模型审计与冻结决策；
- `configs/rl/sac_mppi_utrap_consensus_lcb_l20.yaml`；
- `tests/rl/test_critic_consensus_calibration.py` 及 SAC/prior 回归测试。

默认 `correction_advantage_gate_mode: none` 时行为不变；`hard` 保留 L18/L19 兼容路径；`lcb` 是显式 opt-in。MPPI、MuJoCo plant、LaserScan、scan_guard、local obstacle layer、Memory 和硬件 bridge 均未改写或绕过。

## 8. 科研解释与停止决定

L20 排除了：

> 只使用正在训练 actor 的同一对 twin critics，并增加 disagreement penalty，就足以形成零回归部署门控。

但数据同时支持：

> critic 共识包含有用信息；它将 L19 最接近候选的 3 次 paired success loss 缩小到 1 次，并把三个训练模型的 mean return delta 全部变为正。

按预注册停止条件，本轮不继续扩大 beta 网格。下一步应建立**与 actor 优化目标解耦的独立风险估计器或独立 critic ensemble**，并采用新的数据划分。L20 的唯一失败 episode 与五个成功救回 episode 是设计诊断依据，而不是未来最终测试样本。

## 9. 结果位置

```text
results/research_platform/rl/
  l20_lcb_calibration_seed20260721_20281201_20281212_20260715_v1/
  l20_lcb_calibration_seed20260722_20281201_20281212_20260715_v1/
  l20_lcb_calibration_seed20260723_20281201_20281212_20260715_v1/
  l20_lcb_calibration_multiseed_20260715_v1/
    selection.json
    paired_calibration_rows.csv
```

## 10. 复现命令

```bash
.venv/bin/python experiments/rl/run_correction_advantage_lcb_grid.py \
  --config configs/rl/sac_mppi_utrap_consensus_lcb_l20.yaml \
  --scene-config configs/research/mujoco_u_trap_long_board.yaml \
  --baseline-checkpoint RUN/checkpoints/initial.pt \
  --candidate-checkpoint RUN/checkpoints/step_000020000.pt \
  --training-seed 20260721 \
  --seeds 20281201,20281202,20281203,20281204,20281205,20281206,20281207,20281208,20281209,20281210,20281211,20281212 \
  --betas 1,1.5,2,3,5,8 \
  --output-dir results/research_platform/rl/L20_RUN

.venv/bin/python experiments/rl/select_correction_advantage_lcb.py \
  --run-dir L20_SEED_1 --run-dir L20_SEED_2 --run-dir L20_SEED_3 \
  --output-dir results/research_platform/rl/l20_lcb_calibration_multiseed_20260715_v1
```
