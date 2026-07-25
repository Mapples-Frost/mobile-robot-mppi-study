# Online V3 Tracking and MPPI Smoke — Amendment 5

日期：2026-07-23  
发生阶段：Amendment 4 development smoke 失败后的预注册允许诊断  
修改范围：在线检测/关联与离线审计；冻结预测器和风险数学不变

## Amendment 4 结果

9 个 Gate 中 7 个通过。失败项为：

- 最低 tracker forecast availability：`0.610 < 0.900`；
- 错误计算的最大 measurement RMSE：`0.872 m > 0.120 m`。

与此同时，risk-enabled 没有产生相对 risk-disabled 的新增碰撞，三对 seed
均产生行为差异，中位最小 clearance 提升 `0.138 m`，planner P95
低于 `40 ms`。

## 坐标系诊断

tracker、forecast 与 MPPI 都在轮速里程计坐标系中工作。Amendment 4 的
离线审计却直接用 MuJoCo 世界坐标减 tracker 测量，把长期里程计漂移错误地
归为障碍物检测误差。

使用每步已记录控制做确定性重放：

- 六个轨迹的重放位置最大误差：`0.0 m`；
- 将障碍物真值投影到在线里程计坐标系后，六个 RMSE 为
  `0.0229–0.0312 m`。

因此 RMSE 失败是审计坐标系错误，不是错误关联。修订后的 runner 记录每步
在线 observation pose，并在同一坐标系内评分。真值只在 episode 后用于
坐标投影，仍不进入控制器。

## 远距离检测诊断

risk-enabled 小车保守等待时，障碍物距离约 `4.4–6.6 m`，直径 `0.5 m`
的圆柱在 2° LaserScan 中经常只占 1–2 根 beam。原 detector 固定要求
3 根连续 beam，导致 cluster 消失。

修订为：

- tracker 尚未初始化：仍要求至少 3 根 beam；
- tracker 已初始化：允许 1 根 beam 形成候选；
- 单束候选仍必须落入冻结 IMM 一步预测的 `0.90 m` gate；
- 场景仍是预注册的单动态目标，没有引入真值身份标签。

用 Amendment 4 的完整已记录轨迹做离线诊断，单束跟踪对两个长 episode
均达到 forecast availability `1.0`，RMSE 分别约 `0.087 m` 和
`0.061 m`，低于原 Gate `0.12 m`。这些是 development 诊断值，不作为
Amendment 5 的最终结果。

## 未修改内容

V3、Change-Aware IMM、Collision Risk V1、seed、路线、风险权重、paired
设计和全部 Gate 均保持不变。Amendment 5 将重新运行全部六个完整 episode。

