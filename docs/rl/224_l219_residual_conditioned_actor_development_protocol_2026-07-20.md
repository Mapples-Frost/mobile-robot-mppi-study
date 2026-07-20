# L219 Residual-conditioned Actor 跨地图开发协议

## 目的

L218 开发筛查和完整预算探针均表明，原 L205 Actor 在六张 6.5 m × 6.5 m 新地图上处于支持域外。HSS 正确将 proposal authority 降为零，从而保护 ICODE-MPPI；但这只能验证安全退化，不能验证 RL 对复杂地图的积极贡献。

L219 在不改变论文核心机制的前提下，仅扩展 Actor 的开发训练分布：

```text
Value-Consistent ICODE
    -> residual/innovation context
Residual-Conditioned RL Actor
    -> MPPI proposal
role-aware reliability HSS
    -> proposal/value authority
MPPI -> scan_guard -> MuJoCo
```

## 冻结边界

- ICODE checkpoint、控制仿射残差结构和价值一致性训练结果不变；
- MPPI rollout、cost、LaserScan、local obstacle layer、scan_guard 和 arbitration 不变；
- HSS 及 proposal authority 修复不变；
- 只在 development training/validation seeds 上训练和选择 Actor；
- 不读取、使用或筛选后续 sealed seeds；
- 所有训练失败和候选 checkpoint 均保留；
- 本阶段结果不得称为跨拓扑泛化，因为训练与验证共享六类几何，只隔离随机种子和物理域回合。

## 训练设计

- 初始化：L204 residual-conditioned direct Actor 的 `best.pt`；
- 新训练：三组独立 seed，分别训练 30,000 environment steps；
- 训练场景：L218 六场景全部纳入；
- 物理域：seen 与 unseen 均纳入，防止只学习单一响应；
- replay：scene-balanced；场景×物理域覆盖单独审计，不使用会因长回合而永久锁死梯度更新的 all-groups latch；
- observation normalizer：仅用新的训练流在线更新；
- validation：固定且与训练 seed 隔离；
- checkpoint 选择：验证集控制指标，不读取后续封存数据。

## Development Gate

候选 Actor 必须先在 L218 development seeds 上满足：

1. 六场景均无碰撞率恶化；
2. Actor support/proposal authority 不再全为零；
3. 至少一张原失败地图的到达率或路径完成度显著提高；
4. 三张原成功地图不得出现系统性成功率回退；
5. Full Proposed 优于同预算 Simple combination，并报告相对 ICODE-MPPI 的真实增益或等价性；
6. 如果 Actor 被 HSS 全程拒绝，或激活后造成碰撞/系统性退化，则 Gate 失败，不进入 sealed benchmark。

只有通过 Gate 后，才会固定 checkpoint SHA-256、代码 Git SHA、配置、方法矩阵和全新 sealed seeds。

## 训练 Gate 修正记录

首轮三组 30k 运行完整结束且无异常，但训练摘要显示 `update_records = 0`。原因是 `replay_require_all_scenes` 在当前 pool 中实际要求训练开始前随机访问全部场景×物理域 group；长回合使每组训练只产生约 30 个 episode，未能覆盖所有 group，因此所有梯度更新被阻断。

这三组输出作为负向工程结果完整保留，不参与 Actor 候选比较。修正仅关闭该 all-groups latch，保持 scene-balanced replay、warmup、训练场景、物理域、奖励、网络、ICODE、HSS 和 MPPI 不变。修正后的运行使用新版本号，禁止覆盖首轮目录。

## V2 checkpoint 选择规则

V2 三组训练各完成 30,000 步和 29,001 次梯度更新。候选只能使用固定 validation seeds 产生的 30 个场景—物理域回合；development evaluation seeds 91001--91003 不参与选择。

固定字典序如下：

1. 最小化碰撞数；
2. 最大化成功数；
3. 最大化平均路径完成度；
4. 最小化平均横向 RMSE；
5. 最小化平均目标距离；
6. 最大化平均 return；
7. seed 和 step 只用于确定性打破完全相同的排序。

按该规则选中 seed 20262193 的 step 10,000 checkpoint，SHA-256 为：

```text
bf26a67ebac313930d63760db931e5d50704cdd9181923afd7f66d16159356b4
```

该选择仍属于 development validation。它必须在 seeds 91001--91003 的等预算 MPPI 闭环比较中通过 Gate，才能进入预注册 sealed benchmark。
