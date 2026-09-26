# L22：短时门控 RL 修正序列反事实结果与数据审计

日期：2026-07-15  
性质：预注册数据采集完成；数据质量通过；风险分类模型训练 Gate 未通过  
预注册：[`24_l22_counterfactual_burst_prereg_2026-07-15.md`](24_l22_counterfactual_burst_prereg_2026-07-15.md)

## 1. 结论先行

L22 将 L21 的单步 correction 扩展为 10-step gated SAC burst，成功放大了短时轨迹差异，但仍未产生预注册三分类所需的 harmful/beneficial 标签：

```json
{
  "quality_passed": true,
  "burst_contract_passed": true,
  "samples": 120,
  "train_samples": 96,
  "validation_samples": 24,
  "train_harmful": 0,
  "train_beneficial": 0,
  "validation_harmful": 0,
  "validation_beneficial": 0,
  "train_risk_model": false,
  "sealed_test_seeds_opened": false
}
```

因此本轮没有训练分类器，也没有事后降低阈值。L22 支持“correction 的作用具有序列累积性和状态依赖性”，但不支持“当前 40-step 标签可以直接训练可靠的 harmful/beneficial Gate”。

## 2. 实际执行的干预

```text
Baseline:
  40 steps frozen BC

Candidate:
  first 10 steps = SAC proposal -> target twin-critic LCB beta 2
                   accepted -> correction
                   rejected -> frozen BC
  next 30 steps  = frozen BC
```

branch 第一步必须被 Gate 接受，后续 9 步依据候选分支自己的 observation 动态重算，并非复制第一步动作或强制连续接受。

## 3. 数据规模与覆盖

固定 branch points：`10, 30, 50, 70, 90, 110`。  
固定 train seeds：`20281401--20281408`。  
固定 validation seeds：`20281409--20281410`。

| training checkpoint seed | considered | accepted samples | train | validation |
|---:|---:|---:|---:|---:|
| 20260721 | 60 | 37 | 31 | 6 |
| 20260722 | 60 | 36 | 29 | 7 |
| 20260723 | 60 | 47 | 36 | 11 |
| total | **180** | **120** | **96** | **24** |

首步接受 coverage 为 `120/180 = 66.7%`。候选窗口中实际接受情况：

| split | mean accept fraction | accepted steps median | min–max |
|---|---:|---:|---:|
| train | 72.19% | 8.0 / 10 | 1–10 |
| validation | 68.33% | 7.5 / 10 | 2–10 |

这证明 L22 测量的是实际短时 gated policy，而不是仅执行一次 correction 后把名称改为 burst。

## 4. 连续结果

### Train

| target | min | q10 | median | q90 | max |
|---|---:|---:|---:|---:|---:|
| return delta | -0.1635 | -0.0500 | -0.0004 | +0.0602 | +0.3707 |
| distance improvement | -0.0801 m | -0.0249 m | -0.0002 m | +0.0286 m | +0.1846 m |

### Validation

| target | min | q10 | median | q90 | max |
|---|---:|---:|---:|---:|---:|
| return delta | -0.2037 | -0.0925 | -0.0068 | +0.0177 | +0.1422 |
| distance improvement | -0.0941 m | -0.0429 m | -0.0055 m | +0.0074 m | +0.0713 m |

与 L21 单步干预相比，最大绝对距离效应从约 5 cm 扩大到约 18.5 cm，说明序列累积效应真实存在。但 return 从未达到预注册的 `+0.5/-0.5`，因此即使有 3 cm 以上的距离变化，联合标签仍为 neutral。

没有出现 collision regression、success loss 或 success gain。这里的“没有”只适用于 40-step 局部 horizon，不能外推为完整 episode 安全结论。

## 5. 探索性 EDA：为什么全是 neutral

以下分析在预注册标签判定完成后进行，只用于解释和设计下一轮，不改变 L22 结论。

### 5.1 Return 与距离几乎重复

| split | Pearson correlation | sign agreement |
|---|---:|---:|
| train | 0.9868 | 92 / 96 |
| validation | 0.9962 | 23 / 24 |

当前 40-step reward delta 几乎由目标距离变化主导。因此标签中的 return 与 distance 并非两条近似独立的证据；`|return delta| >= 0.5` 成为实际决定性门槛，而本轮观测范围仅为 `[-0.204, +0.371]`。

这不授权事后删除 return 条件。它说明下一轮必须在采数前重新定义与 horizon 尺度一致的连续 utility 或完整 episode outcome。

### 5.2 已存在方向性信号，但验证覆盖不足

若仅作探索性计数，不作为 L22 标签：

| split | distance ≥ +3 cm | distance ≤ -3 cm | unique checkpoint/episode groups |
|---|---:|---:|---:|
| train | 10 | 7 | 10 positive / 7 negative |
| validation | 1 | 4 | 1 positive / 4 negative |

三个 checkpoint 在 train 都出现正、负距离事件；validation 的负事件也覆盖三个 checkpoint，但正事件只来自 checkpoint `20260721`。因此当前 development validation 尚不足以证明“正向收益可跨 checkpoint 泛化”。

### 5.3 接受更多并不等于更好

| relation | train correlation | validation correlation |
|---|---:|---:|
| accepted steps vs. distance improvement | +0.196 | -0.316 |
| correction magnitude vs. distance improvement | +0.106 | -0.222 |

相关性弱且跨 split 不稳定。不能用“多接受一些 correction”作为简单 Gate；效果取决于 observation、场景位置和动作方向。这正是状态条件风险/收益估计可能有意义的地方，但当前数据规模还不能证明该估计器可学。

## 6. 数据质量与可复现性

- history replay 最大 observation error：`0.0`；
- burst 接受/拒绝计数守恒：通过；
- duplicate `(training_seed, episode_seed, branch_step)`：0；
- CSV/NPZ missing、NaN、Inf：0；
- 三个 checkpoint 的 config、feature schema、label contract 一致；
- train/validation episode split 隔离；
- sealed risk test `20281311--20281315` 未运行；
- sealed closed-loop selection `20281101--20281115` 未运行。

独立研究重复仍是 checkpoint/episode group，不是 120 个 branch rows。开发集共有 24 个 train groups 和 6 个 validation groups；分支点是组内重复测量。

## 7. 科研判断

本轮对总体方向的证据是“部分支持、但仍不足以形成论文结论”：

1. 支持：单步影响很小，10-step sequence 能显著扩大正负轨迹偏移；
2. 支持：相同 accepted count 可能产生相反效果，状态相关调控比固定 correction 强度更合理；
3. 不支持：当前 40-step、三分类、联合阈值设计不能提供风险模型监督信号；
4. 警告：validation 平均效果在三个 checkpoint 上均略为负，不能只展示 train 的 +18.5 cm 最佳样本；
5. 未验证：独立 Gate 能否预测收益、能否消除 L20 的 paired success loss、能否在动态障碍或 OOD 中泛化。

因此研究方向不应放弃，但风险学习的目标应从任意阈值的单步分类，转向与真实闭环决策一致的 sequence-level continuous utility 或 macro-policy outcome。

## 8. 下一阶段建议（尚未实施）

建议单独预注册 L23：

1. 将 Gate 的作用单位明确为短时 macro-policy，而不是孤立 action；
2. 使用连续 `distance/return/clearance` utility 与不确定性区间，避免先把小数据硬切成三类；
3. 增加全新的 development episode groups，保证正负效应在每个 checkpoint 的验证覆盖；
4. 模型结构、loss、阈值冻结后才允许打开 `20281311--20281315`；
5. 只有 sealed test 通过，才进入完整闭环 selection，最后打开 `20281101--20281115`；
6. 仍保留 BC fallback 和 scan_guard 最高安全优先级。

不能把 L22 的探索性 `±3 cm` 计数直接包装成新标签，也不能立即把 validation 用于模型选择后再称为独立验证。

## 9. 工程交付

- `configs/rl/correction_counterfactual_burst_l22.yaml`；
- `experiments/rl/collect_correction_counterfactual_branches.py`：可配置 1-step / burst intervention；
- `experiments/rl/audit_correction_counterfactual_dataset.py`：burst contract 与旧 L21 兼容审计；
- `src/mobile_robot_mppi/rl/risk_dataset.py`：sealed split 和 burst accounting 合同；
- `tests/rl/test_risk_dataset.py`：解析、隔离和计数回归；
- `results/research_platform/rl/l22_burst_multiseed_20260715_v1/audit.json`：机器可读结果。

原 MPPI、MuJoCo、LaserScan、local obstacle layer、scan_guard、ICODE、Memory 和硬件 bridge 均未被绕过或改写。

