# L47 结果：横向误差 primary 失败，但效率与平滑性形成新假设

日期：2026-07-16  
结论等级：development；原 primary Gate 未通过；L47 sealed seed 未开启。

## 原 primary

270/270 episode 完整。structure-preserving delay-aware ICODE 相对同样 delay-aware nominal 的平均 cross-track RMSE
变化为 `-2.20%`（恶化），绝对差 `-0.00012 m`，95% CI `[-0.00801, +0.00734] m`。success 均为 90/90，
collision 均为 0。原横向误差 Gate 明确失败。

## 探索性次要指标

以下分析在 primary 失败后进行，因此只用于生成新假设：

| 指标（nominal - ICODE） | Estimate | Hierarchical bootstrap 95% CI |
|---|---:|---:|
| Path length reduction | 0.0681 m | [0.0323, 0.1091] m |
| Control jerk reduction | 0.00223 | [0.00150, 0.00305] |
| Applied-control jerk reduction | 0.00158 | [0.00096, 0.00218] |
| Step reduction | 2.69 | [-0.26, 5.49] |

路径长度、control jerk 和 applied jerk 在 3/3 model block 上方向一致，且分层区间严格大于零。它们不能事后替代失败的 L47 primary，
但足以支持一个全新、可证伪的独立确认实验。

## 决策

L48 使用完全未出现过的 10 个 episode seed，并在运行前把路径长度与 jerk 冻结为 primary endpoints；不再调模型、路径、物理域或 Gate。

