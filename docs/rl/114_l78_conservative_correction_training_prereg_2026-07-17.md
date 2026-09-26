# L78：保守 SAC 修正训练预注册

日期：2026-07-17  
性质：development training；在任何 L78 训练 transition 产生前冻结

## 1. 背景与可证伪问题

L72 的风险平衡 replay 在训练内部验证集上通过 3/3 模型块，但 L74 的封存确认、L76 的
correction-support gate 和 L77 的训练内 checkpoint 规则复验均未得到跨 seed 的稳定正收益。
L75 进一步说明，直接撤掉 BC 先验直到车辆停滞会破坏已有的轨迹脚手架。

因此 L78 不再扫描部署阈值，也不修改 MPPI cost。它检验以下训练层假设：

> 先让 critic 充分积累覆盖三个静态几何场景的经验，再以更小学习率和更强 BC 修正惩罚更新
> residual actor，能否减少 SAC 对冻结 BC 先验的跨 seed 破坏，同时保留至少两个模型块的
> 非零、非劣修正？

如果只有 0/3 或 1/3 模型块选择非零 checkpoint，该假设在当前训练预算下被判为不支持；
不得用单个好 seed 宣称方法有效。

## 2. 固定训练处理

对照是 L72 `sac_mppi_static_risk_balanced_l72`。L78 只改变一个预先定义的复合训练日程：

| 训练量 | L72 | L78 |
|---|---:|---:|
| actor learning rate | `1e-4` | `5e-5` |
| normalized correction penalty | `0.25` | `1.00` |
| first actor update | `5,000` | `10,000` |

这个复合处理用于先筛查“保守训练”是否值得继续；若后续新 seed 部署验证通过，必须再做
actor learning rate、correction penalty、actor delay 的拆分消融，不能把三者的因果贡献混为一谈。

以下项目与 L72 保持不变：

- frozen BC correction 的动作参数化与 correction scale；
- twin critic、target network、熵正则和 reward；
- 30,000 environment steps、2,000 step warm-up/update start；
- `scene_outcome_balanced` replay 和 0.5 success fraction；
- single obstacle、narrow corridor、U-trap 三个静态场景；
- 三个独立 ICODE checkpoint、高动态 MuJoCo plant、command delay；
- MPPI 样本数、horizon、cost、LaserScan、local obstacle layer、scan guard 和安全仲裁；
- memory 关闭，progress gate 和 correction-support gate 关闭；
- checkpoint selection 的 initial-noninferiority 规则。

## 3. 实验单位、随机化与数据边界

独立实验单位是完整训练模型块，不是 gradient update、transition 或 validation episode。

| block | SAC seed | frozen BC checkpoint | ICODE checkpoint |
|---:|---:|---|---|
| 0 | `20260774` | `bc_l13_seed20260721_20260714/checkpoints/best.pt` | L57 seed `20261201` |
| 1 | `20260775` | `bc_l13_seed20260722_20260714/checkpoints/best.pt` | L57 seed `20261202` |
| 2 | `20260776` | `bc_l13_seed20260723_20260714/checkpoints/best.pt` | L57 seed `20261203` |

训练内 checkpoint selection 使用固定 validation seeds `22300801`–`22300805`，并在三个场景上
配对。它们只用于模型选择，不是最终检验样本。L75–L77 的 episode seeds 不得复用。

若训练 Gate 通过，下一阶段才可打开一次新的 deployment development 集 `22201101`–`22201105`。
封存集 `22201111`–`22201115` 继续保持不可见，直到 deployment development Gate 通过。

## 4. 训练完整性 Gate

每个模型块必须满足：

1. 恰好完成 30,000 environment steps；
2. interrupted episode 为 0；
3. 0、5k、10k、15k、20k、25k、30k 的验证记录完整；
4. 三个场景均写入 replay，且 replay strategy 为 `scene_outcome_balanced`；
5. 配置快照、run metadata、validation CSV、update CSV、initial/best/periodic checkpoints 齐全；
6. 所有数值有限，checkpoint 可重新加载。

训练阶段继续的最低 Gate：

- 至少 2/3 模型块由预注册的 initial-noninferiority 规则选择非零 checkpoint；
- 被选择 checkpoint 相对 block 自己的 step-0 frozen BC：
  - success loss = 0；
  - collision regression = 0；
  - 平均最终目标距离不恶化；
  - 至少 1 个 success gain，或平均最终目标距离改善至少 0.005 m。

只要少于 2/3 模型块选择非零，L78 停止在训练阶段，不运行新 deployment seeds。

## 5. 后续部署 Gate（只冻结规则，不提前观察数据）

若训练 Gate 通过，L79 将：

- 每个 block 只使用 L78 训练内规则选出的一个 checkpoint；
- 与该 block 的 frozen BC step-0 严格配对；
- 使用 3 scenes × 5 seeds × 2 conditions × 3 blocks = 90 episodes；
- block 内运行顺序随机化；
- 禁止重新搜索 checkpoint、门控阈值或场景权重。

继续到封存确认至少需要：

- 至少 2/3 blocks 合格；
- overall success gains ≥ 3、losses ≤ 1、net gain ≥ 2；
- collision regressions = 0；
- overall mean final goal distance 不恶化；
- narrow corridor net gain ≥ 2；
- single obstacle 与 U-trap 各自 success loss = 0；
- SAC correction 的实际激活量非零。

## 6. 解释边界

L78 即使通过，也只能说明这个复合保守训练日程在当前静态高动力学 MuJoCo benchmark 上值得进入
新的配对部署验证。它不证明每个训练因素各自有效，不证明 RL 普遍优于传统 MPPI，也不构成动态
障碍或实车结论。

若 L78 失败，下一步不再增加部署 gate。应转向 critic 校准/算法替换，或把 RL 降级为探索与数据采集
模块；ICODE、传统 MPPI 与安全链的已验证结论不受该负结果影响。

