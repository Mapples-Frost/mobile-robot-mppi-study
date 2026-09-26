# L48 独立确认结果：平滑性复现，路径长度总 Gate 未通过

日期：2026-07-16  
结论等级：independent confirmation；总 Gate 未通过。

360/360 episode 完整，无保护或封存 seed 泄漏。ICODE 与 nominal 均保持零碰撞，ICODE 净增加 1 次成功。

| 预注册终点 | Estimate | Hierarchical bootstrap 95% CI | 结论 |
|---|---:|---:|---|
| Path-length reduction | 0.0409 m | [-0.0382, 0.1107] m | 未确认 |
| Control-jerk reduction | 0.00123 | [0.00026, 0.00222] | 确认 |
| Applied-jerk reduction | 0.00123 | [0.00057, 0.00194] | 确认 |

control jerk 与 applied jerk 在 3/3 training-seed model block 上方向一致；路径缩短只在 2/3 block 为正，故按照“全部 Gate 条款同时满足”
的规则，总 Gate 判定失败。cross-track RMSE 相对增幅为 4.93%，仍位于预注册的 5% 实际非劣界限内；completion ratio 差值为 -0.0013。

这构成 residual 对控制平滑性的独立正证据，但尚不足以支持“稳定缩短路径”的声明。

下一步不调 MPPI cost，也不删除表现较差的 training seed。按照 ICODE-MPPI 的 iterative-training 思路，用 L48 新采集的 nominal on-policy
applied-control 轨迹扩充训练分布，以降低模型 seed 方差，然后在完全新 episode seed 上再次确认。

