# L75 在线进度能力门控预注册

日期：2026-07-17  
状态：代码单元测试通过；任何 L75 效果数据产生前冻结。

## 1. 前置证据

L74 的封存确认否定了“静态障碍场景中始终开放 RL 且零成功回退”的强假设。其分场景结果为：

- 窄通道：RL 净增 3 次成功，平均终点距离改善约 0.605 m；
- 单障碍：总体中性；
- U-trap：冻结 BC 为 15/15，常开 RL 为 13/15。

L74 seeds `22170811`–`22170815` 已消费，本轮严禁用于阈值选择、模型选择或效果判断。

## 2. 可证伪假设

> 在局部几何复杂但机器人仍持续接近目标时，冻结 BC/传统 MPPI 已具备足够能力，应抑制 RL；只有机器人在固定时间窗内缺乏目标进度时，才开放现有 SAC correction，可保留窄通道解困增益并减少饱和简单场景中的负迁移。

本轮不重新训练 SAC、BC 或 ICODE。唯一方法变化是在线 RL activation。

## 3. 在线信号

门控输入仅为：

1. 现有 LaserScan 几何复杂度；
2. 普通定位与目标参考链计算的 goal distance；
3. 时间戳；
4. 现有 target-critic LCB correction filter。

不使用：

- 场景名称；
- MuJoCo 障碍真值；
- future outcome；
- oracle dynamics；
- L74 配对标签。

在长度 2.0 s、最小覆盖 1.8 s 的滑动窗内定义目标进度：

\[
\Delta d_t=d_{\mathrm{old}}-d_t.
\]

停滞激活为：

\[
\alpha_{\mathrm{stagnation}}=
\begin{cases}
1, & \Delta d_t\le 0.04\ \mathrm{m},\\
\dfrac{0.20-\Delta d_t}{0.20-0.04},
&0.04<\Delta d_t<0.20\ \mathrm{m},\\
0, & \Delta d_t\ge 0.20\ \mathrm{m}.
\end{cases}
\]

最终外层贡献：

\[
\alpha_{\mathrm{RL}}
=
\alpha_{\mathrm{complexity}}
\alpha_{\mathrm{stagnation}}
\alpha_{\mathrm{near-goal}}.
\]

激活后保持 1.0 s 防止逐帧抖动。观察窗未就绪、scan 无效或证据不足时退回 BC/传统 MPPI。现有 near-goal fallback、critic LCB、scan guard、local obstacle layer 与 safety arbitration 均保持。

## 4. 条件

每个模型块运行三种条件：

1. `complexity_bc_icode`：冻结 BC，使用 `initial.pt`；
2. `gated_lcb_icode`：L74 的复杂度常开方案，使用锁定 30k SAC；
3. `progress_gated_lcb_icode`：新增进度能力门控，使用同一 30k SAC。

三个模型块分别对应 SAC seeds `20260751`–`20260753` 和 ICODE seeds `20261201`–`20261203`。所有条件的 MuJoCo plant、ICODE、MPPI \(K=100\)、场景、感知、安全链与控制上限保持相同。

## 5. 阻断、随机化与样本

- 场景：single obstacle、narrow corridor、U-trap；
- 新开发 seeds：`22200801`–`22200805`；
- 新封存 seeds：`22200811`–`22200815`；
- 独立重复单位：3 个训练模型块；
- episode 是块内配对重复测量，不作为独立训练重复；
- 每块内按固定 schedule seed 随机打乱条件—场景—seed；
- 总计：3 blocks × 3 scenes × 5 seeds × 3 conditions = 135 episodes。

## 6. 主要开发门槛

相对冻结 BC，progress-gated SAC 必须同时满足：

1. 至少 2/3 模型块净成功变化非负；
2. 45 个配对中 success gains 至少 2；
3. success losses 至多 1；
4. 总净成功增益至少 1；
5. U-trap success losses 为 0；
6. narrow corridor 净成功增益至少 2；
7. 新增碰撞为 0；
8. 平均终点距离不恶化；
9. 相对 always-on complexity SAC，平均外层 gate alpha 至少下降 50%；
10. 窄通道平均 stagnation activation 至少 0.02；
11. 冻结 BC correction gate alpha 严格为 0。

全部条件必须通过。缺失、重复、非有限指标、旧 seeds 或封存 seeds 混入均 fail closed。

## 7. Oracle 上界

仅在结果汇总时离线计算 `episode oracle`：对同一块、场景和 seed，在 BC、always-on SAC 与 progress-gated SAC 中按成功、碰撞、终点距离选择最好结果。它只衡量“选择机制还有多少上界空间”，不作为可部署方法，也不用于修改 L75 门槛。

## 8. 停止规则

- L75 未通过：不打开 `22200811`–`22200815`，保留负结果并诊断门槛失败来源；
- L75 通过：先冻结代码、阈值与 checkpoint，再使用封存 seeds 一次性确认；
- 不允许根据 L75 结果重新解释 L74；
- 不允许用增加 SAC 训练步数同时改变 activation 逻辑。
