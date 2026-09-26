# L79：保守训练 checkpoint 的新 seed 部署结果

日期：2026-07-17  
结论：development deployment Gate failed；sealed seeds 未打开

## 1. 完整性

- 预期/实际 episode：`90 / 90`；
- 重复 episode key：0；
- 旧 development seed 泄漏：0；
- 封存 seed 使用：0；
- collision regression：0；
- frozen base correction：精确为 0；
- artifact integrity：通过。

结果目录：

`results/research_platform/rl/l79_conservative_schedule_deployment_20260717_v1/summary/`

## 2. 主要结果

| 聚合 | success gain/loss | net gain | mean final-distance improvement | mean clearance change |
|---|---:|---:|---:|---:|
| overall | `1 / 2` | `-1` | `+0.07992 m` | `+0.01227 m` |
| block 0, 25k | `0 / 0` | `0` | `+0.00242 m` | `+0.01082 m` |
| block 1, step 0 | `0 / 0` | `0` | `0.00000 m` | `0.00000 m` |
| block 2, 25k | `1 / 2` | `-1` | `+0.23735 m` | `+0.02600 m` |

按场景：

| scene | success gain/loss | net gain | mean final-distance improvement |
|---|---:|---:|---:|
| clean single obstacle | `1 / 0` | `+1` | `+0.15512 m` |
| narrow corridor | `0 / 1` | `-1` | `+0.23443 m` |
| U-trap | `0 / 1` | `-1` | `-0.14978 m` |

三个 success-discordant pair 都来自 block 2：

- clean seed `22201102`：BC 失败、candidate 成功，最终距离改善 `+2.28787 m`；
- corridor seed `22201103`：BC 成功、candidate 失败，最终距离变化 `-0.04358 m`；
- U-trap seed `22201102`：BC 成功、candidate 失败，最终距离变化 `-2.25981 m`。

## 3. 判定

预注册 Gate 未通过：

- 需要 gain ≥ 3，实际 1；
- 允许 loss ≤ 1，实际 2；
- 需要 net gain ≥ 2，实际 -1；
- 需要至少 2/3 合格 blocks，实际 0/3；
- corridor 需要 net gain ≥ 2，实际 -1；
- U-trap success loss 必须为 0，实际 1。

因此没有运行 `22201111`–`22201115` 封存 seeds，也没有事后修改阈值。

## 4. 科学解释

L79 不是“RL 完全无效”：连续指标总体改善、没有新增碰撞，且 clean scene 出现一次真实成功增益。
但 block 2 同时在不同几何场景间交换成功与失败，说明 L78 的普通 minibatch 平均 actor objective
仍允许跨场景补偿：一个场景的大收益可以掩盖另一个场景的稀有损失。

这个证据不支持继续增加部署 gate。下一轮必须修改训练目标，使 actor 直接关注最差场景组，而不是在
部署阶段再猜测何时关闭 correction。

