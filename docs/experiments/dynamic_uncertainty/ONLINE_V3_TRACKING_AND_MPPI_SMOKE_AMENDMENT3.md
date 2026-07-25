# Online V3 Tracking and MPPI Smoke — Amendment 3

日期：2026-07-23  
发生阶段：任何闭环控制结果产生之前的 seed registry 合规复核  
原因：原 seed 位于 development 数值区间，但不属于登记允许的 smoke 子集

## 发现

`dynamic_uncertainty_splits.yaml` 将 `730100001–730100300` 定义为
development 区间，但同时规定 smoke 只能使用登记的五个 seed：

`730100001, 730100002, 730100003, 730100004, 730100005`

原预注册的 `730100101–730100103` 属于 development，却不满足更严格的
smoke 子集规则。它们尚未产生任何闭环控制结果。

## 修订

- 从登记 smoke 子集中选用：`730100001, 730100003, 730100005`；
- episode 时域仍为 `[0, 40] s`；
- 起点：`(-4.6, -1.8)`；
- 终点：`(4.2, -1.8)`；
- 三个 seed 的有效路线穿越次数分别为 `5, 3, 4`。

路线选择仅使用冻结的障碍物真值轨迹做预先几何资格检查，没有运行控制器，
也没有查看 risk-disabled / risk-enabled 结果。

## 未修改内容

- 冻结 V3 generator、Change-Aware IMM 和 Collision Risk V1；
- noise profile、过程参数与 schedule seed；
- paired arms、MPPI 参数与全部 smoke Gate；
- 独立单位仍是完整 episode。

后续 smoke 统一使用
`mujoco_v3_probabilistic_crossing_smoke_amendment3.yaml`。前三版配置与说明
保留为可追溯的设计修订链。

