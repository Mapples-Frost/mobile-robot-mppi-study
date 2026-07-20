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
- replay：scene-balanced，且所有场景进入 replay 后才允许均衡更新；
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
