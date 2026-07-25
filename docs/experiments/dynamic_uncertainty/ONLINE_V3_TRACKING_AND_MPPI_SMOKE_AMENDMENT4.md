# Online V3 Tracking and MPPI Smoke — Amendment 4

日期：2026-07-23  
发生阶段：第一次启动的首个 episode 中断后；完整 episode 数为 0  
原因：在线 forecast 缺失时抛异常，不是可持续运行的 fail-closed 行为

## 发现

首个随机运行顺序为 risk-enabled / seed 730100005。障碍物暂时超过原
`6.0 m` LaserScan 量程，tracker 按冻结契约在连续未观测超过 `1.5 s`
后停止发布 forecast。Collision Risk V1 正确拒绝缺失输入，但在线运行层
把这种拒绝表现为异常，导致 episode 直接中断，没有产生闭环结果。

这不是概率公式失效。它表示：“没有可靠概率输入时，不允许继续按正常速度
规划”。对于持续运行系统，工程上应把它实现为安全停车，而不是进程崩溃。

## 修订

1. 保留 Collision Risk V1 默认 `raise` 行为，既有资格测试语义不变；
2. 本 smoke 显式设置缺失 forecast 行为为 `stop`：
   输出零速度序列，等待重新观测后恢复正常 MPPI；
3. LaserScan 最大量程由 `6.0 m` 改为 `10.0 m`。冻结 V3 waypoint 与路线
   端点的最大几何距离约 `8.68 m`，因此 10 m 覆盖完整实验场景；这不提供
   mode、seed 或未来轨迹，只增加真实当前扫描的可观测范围。

## 未修改内容

- V3 generator、Change-Aware IMM、Collision Risk V1 数学定义；
- seed、路线、paired arms、MPPI 风险权重与全部 Gate；
- 控制器仍然只能使用当前/历史 LaserScan 和里程计；
- simulator truth 仍只允许在 episode 完成后做审计。

中断目录被保留为
`mujoco_v3_probabilistic_crossing_smoke_amendment3_interrupted_launch`。
后续完整运行统一使用 Amendment 4。

