# L59 固定高动态 plant 独立确认结果（2026-07-16）

## 1. 完整性结论

L59 完成 240/240 episodes。10 个 seeds 均来自 L58 预先封存的 21860771–21860780；metadata
与 observed seeds 完全一致，未使用 protected development seeds，无缺失、额外 key 或非有限指标。

## 2. Primary result

相对 nominal MPPI，ICODE-MPPI 的 paired cross-track RMSE：

- nominal/ICODE absolute mean：0.05758/0.04117 m；
- mean relative reduction：27.06%；
- mean absolute improvement：0.01641 m；
- hierarchical-bootstrap 95% CI：[0.01384, 0.01877] m；
- 3/3 independent ICODE training blocks 为正；
- 4/4 scenes 为正。

未见 reverse-S 路径：

- relative reduction：34.26%；
- mean absolute improvement：0.02086 m；
- hierarchical-bootstrap 95% CI：[0.01938, 0.02228] m；
- 3/3 model blocks 为正。

## 3. Safety、smoothness 与计算开销

- net success gain：0；
- net collision increase：0；
- nominal 与 ICODE 均为 120/120 success、0 collision；
- completion ratio difference：+0.000053；
- relative applied-control jerk change：-3.46%；
- ICODE mean planner compute time：37.42 ms，低于冻结的 50 ms Gate。

所有 L59 预注册 Gate 均通过。L58 development 的 26.62% tracking reduction 与 L59 confirmation
的 27.06% 基本一致，且置信区间均严格大于 0。

## 4. 当前可支持与不可支持的结论

当前结果支持：在同一固定、隐藏、低摩擦高惯量 MuJoCo differential-drive plant 上，以 nominal
dynamic-unicycle 为预测基线，训练于 episode-disjoint applied-control 数据的 control-affine ICODE
residual 能改善 H36 prediction，并在全新 sealed episode seeds 上稳定降低 MPPI path-tracking error，
且未观察到成功率、碰撞率或控制 jerk 退化。

当前结果仍不能支持：

- 对不同质量、摩擦、执行器或延迟 plant 的跨域泛化；
- 对 wheel-odometry drift、传感噪声或感知延迟的鲁棒性；
- 对静态/动态障碍场景的控制收益；
- 实车收益；
- 原始 ICODE 理论中的稳定性、收缩性或收敛性保证；
- RL prior 或 cross-layer gate 的收益。

因此 L59 是动力学残差分支的强确认里程碑，但不是整篇 ICRA story 的最终证据。下一阶段应冻结
该 ICODE checkpoint family，转向 plant variation、odometry/perception factor 与 RL sampling-prior 的
正交实验，避免继续在同一 fixed-plant benchmark 上重复堆 seed。
