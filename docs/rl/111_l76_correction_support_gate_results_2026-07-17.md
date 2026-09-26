# L76：SAC 修正支持度门控结果

日期：2026-07-17  
性质：预注册新 seed 开发验证；未使用封存 seed

## 1. 判定

L76 预注册 Gate **未通过**。逐 step observation-support 衰减在新 seed 上没有复现
L76a 校准集的净成功增益，因此 L76 封存 seed 保持关闭。

这轮失败没有否定“冻结 BC + 有界 SAC correction”的整体结构，但否定了：

> 单独使用 normalizer 最大标准化偏离分数，就能在当前任务中稳定判断是否采用
> SAC correction。

## 2. 完整性

- 3 个独立训练模型块；
- 3 个静态几何场景；
- 5 个全新开发 seed；
- 3 个严格配对条件；
- 期望/实际 episode：`135/135`；
- 重复 episode 键：0；
- 受保护或封存 seed：0；
- NaN/Inf：0；
- collision regression：0；
- 冻结 BC correction gate alpha：严格为 0。

## 3. 结果

相对冻结 BC：

| 方法 | success gain | success loss | 净变化 | 平均最终距离改善 |
|---|---:|---:|---:|---:|
| 原 SAC-LCB | 6 | 7 | -1 | +0.074 m |
| support SAC-LCB | 3 | 4 | -1 | -0.111 m |

support gate 按模型块：

| block | gain | loss | 净变化 | 平均最终距离改善 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 3 | -3 | -0.625 m |
| 1 | 2 | 1 | +1 | +0.137 m |
| 2 | 1 | 0 | +1 | +0.156 m |

按场景：

| 场景 | gain | loss | 净变化 | 平均最终距离改善 |
|---|---:|---:|---:|---:|
| single obstacle | 2 | 2 | 0 | -0.005 m |
| narrow corridor | 1 | 2 | -1 | -0.329 m |
| U-trap | 0 | 0 | 0 | +0.002 m |

平均 support confidence 为 `0.809`。有效 correction gate alpha 相对原 SAC-LCB
只降低 `13.8%`，未达到预注册的 15%；更重要的是性能 Gate 同时失败，不能通过
放宽该工程阈值补救。

## 4. 解释

正向信息是：

- U-trap 新增损失被降到 0；
- 三块中两块净 `+1`；
- 没有碰撞回归；
- 冻结 BC 脚手架的结构性退化属性成立。

但 block 0 出现 `-3`，且窄通道净变化为负。OOD 分数在成功与失败修正 step 上有
大量重叠，不能单独作为策略可靠性的充分统计量。L76 的离线 best-of-three oracle
在 45 组中成功 37 组，且三种条件都被选择多次，说明仍存在可选择空间，但当前
部署时信号不足。

## 5. 科研决策

不继续扫描 OOD 阈值。旧实验已经表明固定限幅在另一动态 benchmark 上也不能稳定
确认，因此下一步优先检查 checkpoint 选择是否过拟合：

- L72 训练内规则选择 block 0/1 的 25k、block 2 的 30k；
- 后续仅用 3 个部署 seed 的 L73 把三块都改选为 30k；
- L74–L76 的不稳定主要集中在 30k block 0。

下一轮将恢复**事先已有的 L72 训练内选择规则**，在全新 seed 上直接验证
`25k/25k/30k`，不再增加门控元素。若该规则也失败，则停止部署层补丁，回到 SAC
训练目标与 critic 校准。

## 6. 产物

- 配置：`configs/rl/correction_support_gate_l76.yaml`
- 初始预注册：`docs/rl/109_l76_correction_support_gate_prereg_2026-07-17.md`
- 阈值修订与 L76a：
  `docs/rl/110_l76_prereg_amendment_and_l76a_calibration_2026-07-17.md`
- 结果：
  `results/research_platform/rl/l76_correction_support_gate_20260717_v1/summary/l76_summary.json`
- episode 配对：
  `results/research_platform/rl/l76_correction_support_gate_20260717_v1/summary/l76_paired_effects.csv`
