# L254 MPPI 更新抵消短时诊断协议

日期：2026-07-21
状态：预注册 development diagnostic

## 目的

L251-L253 已依次排除边界时序误判、确定性候选缺失和常值 warm start 三项解释。
L253 中 ICODE 与 Full 仍在安全、净空充分且存在可行候选时输出近零控制；Full 的
有效样本数约为 2，且 RL proposal authority 已接近 0。L254 不修改任何控制行为，
只记录优化器内部的候选和更新结果，用于区分：

1. 最佳候选具有推进动作，但加权更新发生多模态抵消；
2. 最佳候选本身就是停车，说明局部 cost/importance correction 选择了停车；
3. 加权序列代价高于最佳候选，说明需要 candidate-backed update safeguard。

## 冻结设置

- 继承 L253 的全部方法、地图、checkpoint、path rollout prior 与安全设置；
- 场景：`tracking_grand_s_chicane_l234`；
- 方法：`icode_mppi`、`full_proposed`；
- development seed：`923301001`；物理域：`nominal_seen` MuJoCo；
- 每决策 100 rollouts；仅运行 350 步；
- 唯一新增配置：`optimizer_diagnostics_enabled=true`；
- 不改变候选、权重、动作、cost、终止条件或 rollout 预算。

## 诊断字段与判定

逐周期记录最佳可行候选代价与第一动作、最终选中序列代价与第一动作、代价差、第一
动作范数抵消比例及初始 proposal 第一动作。

对最终选中 `(v, omega)` 范数小于 `1e-3` 的停滞周期：

- 若至少 50% 周期满足最佳动作范数大于 `0.05` 且 selected cost 比 best cost 高至少
  `1.0`，支持“加权更新抵消/劣化”解释；
- 若最佳动作也接近零，或 selected cost 不高于 best cost，则转向 cost、importance
  correction 与局部 target 表示审计；
- 两方法分别判断，不以单方法结果外推另一方法。

本轮是诊断，不作论文效果结论，不使用 sealed seeds。无论结果如何均只保留原始数据和
简短根因状态，不画图、不写详细报告；不得调参、筛 seed 或删除失败。

