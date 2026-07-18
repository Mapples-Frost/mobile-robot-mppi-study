# L93 route × dynamic crossing headroom：未通过，停止协方差门控

日期：2026-07-18  
结论等级：预注册 development headroom gate；主 Gate 未通过。

## 1. 结果

在同一条 5.5 m 直线路线上，比较 clean 与 4 种 moving-cylinder crossing，完成
120/120 个冻结 ICODE-MPPI 闭环 episode。context oracle 相对最强全局固定 `baseline`：

- success difference：`0.00`，95% CI `[-0.12,+0.12]`；
- collision difference：`0.00`，95% CI `[-0.12,+0.12]`；
- elapsed time：`-0.40 s`，95% CI `[-2.236,+1.296] s`；
- final distance：`-0.0167 m`，95% CI `[-0.3396,+0.2768] m`。

所有主要区间跨零，主 Gate 未通过。fast/large crossing 在 selection block 中三个固定
协方差均为 100% collision，说明在这些条件下不存在“换一个采样协方差就能解决”的
headroom。

## 2. 解释与边界

本结果否定的是用协方差 fallback 处理当前动态穿越障碍的实现方式，不否定 L89 已独立
通过的 clean route contextual bandit。当前 MPPI 把 LaserScan 障碍作为当前局部几何，
没有在 rollout 中预测动态障碍未来位置；因此采样分布即使改变，也没有新的未来运动信息。

继续训练 scan-risk gate 只会学习选择多个同样失败的候选，故按预注册停止规则终止该支线。
动态障碍保留为 stress-test 和明确局限；若未来将其升级为核心方法，需要独立的 obstacle
motion prediction，而不能包装成 RL covariance 的功劳。

## 3. 下一决策

研究主线回到 L89 的已证实能力：RL 根据路线几何改变 MPPI 探索。下一 Gate 检验它能否
在 `K=50/100/200/400` 下提供计算—性能优势，尤其能否用 50 samples 达到或超过固定
协方差 100 samples。这比继续堆动态模块更符合“story 少、技术作用清楚”的要求。

原始结果：
`results/research_platform/rl/l93_route_dynamic_covariance_headroom_20260718_v1/summary.json`。
