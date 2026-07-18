# L21：单步反事实风险数据结果与充分性审计

日期：2026-07-15  
性质：数据采集链路通过，风险模型训练 Gate 未通过  
预注册：[`22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md`](22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md)

## 1. 结论先行

L21 成功建立了可逐位复现的 MuJoCo/MPPI 反事实分支数据链，但单步 correction 干预在 20-step horizon 内没有产生足够大的风险/收益标签：

```json
{
  "quality_passed": true,
  "train_samples": 101,
  "validation_samples": 23,
  "train_harmful": 0,
  "train_beneficial": 0,
  "validation_harmful": 0,
  "validation_beneficial": 0,
  "train_risk_model": false,
  "sealed_test_seeds_opened": false
}
```

因此本轮没有训练神经网络，也没有通过复制少数类、随机拆 branch rows、降低标签阈值或打开 test seeds 来制造“可训练数据”。

## 2. 实际干预

每个样本从 frozen BC reference trajectory 的同一状态出发：

```text
Baseline branch:
  当前一步 BC -> 后续 19 步 BC

Candidate branch:
  当前一步 beta-2 已接受的 SAC correction -> 后续 19 步 BC
```

两条分支使用相同 episode seed、相同历史 actions 和相同 MPPI random-number stream。当前一步之外，后续策略类别相同，因此测量的是孤立 correction 的短时局部效应。

## 3. 确定性重放结果

开发前置检查连续重放 80 步：

| quantity | maximum error |
|---|---:|
| latent action | 0.0 |
| encoded observation | 0.0 |
| reward | 0.0 |
| goal distance | 0.0 |
| termination flags | identical |

正式 124 个 accepted branches 的所有历史 replay：

```text
max_replay_observation_error = 0.0
```

这证明数据不是通过只复制 MuJoCo `qpos/qvel` 得到的近似快照；plant、sensors、MPPI、history、safety 和 controller state 均通过同 seed action replay 完整恢复。

## 4. 数据规模与覆盖

固定 branch steps：`10, 30, 50, 70, 90, 110`。  
固定 horizon：20 control steps。  
pre-gate：target twin-critic LCB beta 2。

| training seed | considered branches | beta-2 accepted | train | validation | replay max error |
|---:|---:|---:|---:|---:|---:|
| 20260721 | 60 | 42 | 35 | 7 | 0.0 |
| 20260722 | 60 | 29 | 23 | 6 | 0.0 |
| 20260723 | 60 | 53 | 43 | 10 | 0.0 |
| total | **180** | **124** | **101** | **23** | **0.0** |

accepted coverage 为 `124/180 = 68.9%`。三个独立训练 checkpoint 均同时贡献 train 和 validation 样本。

## 5. 连续结果分布

### Train

| target | min | q10 | median | q90 | max |
|---|---:|---:|---:|---:|---:|
| return delta | -0.07977 | -0.02236 | -0.00008 | +0.01641 | +0.10435 |
| distance improvement | -0.03809 m | -0.01106 m | ~0 | +0.00835 m | +0.05163 m |

### Validation

| target | min | q10 | median | q90 | max |
|---|---:|---:|---:|---:|---:|
| return delta | -0.03599 | -0.02285 | -0.00548 | +0.00628 | +0.10589 |
| distance improvement | -0.01770 m | -0.00906 m | -0.00318 m | +0.00202 m | +0.05243 m |

所有分支：

- collision regression = 0；
- success loss/gain = 0；
- safety override delta = 0；
- 没有样本同时越过预注册 harmful 条件 `return <= -0.5` 与 `distance <= -0.03 m`；
- 没有样本同时越过 beneficial 条件 `return >= +0.5` 与 `distance >= +0.03 m`。

结果不是标签代码故障：连续 target 已保存并显示单步 correction 的真实差异量级远小于预注册阈值。

## 6. 数据格式与质量

每个 training seed 输出：

- `samples.npz`：`features [N,67]`、labels、group IDs 和连续 targets；
- `samples.csv`：展平的 67 维 feature、两个 branch 摘要与标签；
- `metadata.json`：checkpoint、git SHA、split、schema、label contract；
- `reference_episodes.csv`：reference 与 replay coverage；
- `config_snapshot.json`。

特征结构：

```text
48 normalized observation
 2 frozen BC latent action
 2 candidate latent action
 2 correction
12 critic/correction diagnostics
 1 branch step
-----------------------------
67 dimensions
```

审计结果：

- duplicate `(training_seed, episode_seed, branch_step)` keys = 0；
- missing cells = 0；
- CSV/NPZ NaN/Inf = 0；
- NPZ 与 CSV 行数逐 run 一致；
- feature/schema/label/split/branch contracts 跨三个 checkpoint 一致；
- train/validation episode seeds 完全隔离；
- sealed test seeds `20281311--20281315` 未运行。

## 7. 为什么不能直接降低标签阈值

看到 return delta 只有约 ±0.1 后，把阈值从 ±0.5 改为 ±0.01 会产生一些正负类别，但这不能证明这些微小局部差异能够预测 L20 的完整 episode success regression。它还会把 MPPI 数值扰动和短时小偏差包装成“风险”。

L20 的门控在完整闭环中会连续多次接受 correction；唯一 paired success loss 是一段 correction 序列的累积结果。L21 表明 frozen BC 能在单步 correction 后迅速恢复，因此单步干预不是当前 sequence-level failure 的正确作用单位。

## 8. 下一步方法决策

下一步不训练单步风险模型。应预注册短时 correction burst counterfactual：

```text
Baseline:  K 步 BC         -> 后续 BC
Candidate: K 步 gated SAC -> 后续 BC
```

其中 K 必须在看新数据前固定，并使用新的 train/validation episode seeds。该设计更接近 L20 门控在真实闭环中的持续作用，同时仍能通过相同历史 replay 保持分支起点和随机数配对。

只有 burst 数据同时出现足够的 harmful 与 beneficial episode groups，才训练独立 risk ensemble。L21 数据可用于验证 replay 工程和连续目标量级，不作为未来 sealed model test。

## 9. 工程交付

- `src/mobile_robot_mppi/rl/sac.py`：公开、验证过的 frozen BC action 接口；
- `src/mobile_robot_mppi/rl/risk_dataset.py`：标签、无泄漏 feature schema、split 和充分性 Gate；
- `experiments/rl/collect_correction_counterfactual_branches.py`：确定性 reference/replay/branch collector；
- `experiments/rl/audit_correction_counterfactual_dataset.py`：跨训练 checkpoint 合并与 fail-closed 审计；
- `configs/rl/correction_counterfactual_branches_l21.yaml`；
- `tests/rl/test_risk_dataset.py` 及 SAC 回归测试。

原 MPPI、MuJoCo plant、LaserScan、local obstacle layer、scan_guard、Memory、ICODE 和硬件 bridge 均未被绕过或改写。

## 10. 结果位置

```text
results/research_platform/rl/
  l21_counterfactual_seed20260721_train20281301_08_val20281309_10_20260715_v1/
  l21_counterfactual_seed20260722_train20281301_08_val20281309_10_20260715_v1/
  l21_counterfactual_seed20260723_train20281301_08_val20281309_10_20260715_v1/
  l21_counterfactual_multiseed_20260715_v1/
    audit.json
    samples_all_training_seeds.npz
    samples_all_training_seeds.csv
    reference_episodes_all_training_seeds.csv
```
