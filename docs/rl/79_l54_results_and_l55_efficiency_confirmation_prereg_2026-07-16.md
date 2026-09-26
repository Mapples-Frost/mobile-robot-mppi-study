# L54 结果与 L55 路径效率独立确认预注册（2026-07-16）

## 1. L54 结论：严格 Gate 未通过

L54 完成 720/720 episodes，所有完整性检查通过：无缺失/重复 key、无 sealed 或
protected seed 泄漏、matched 域跨层门与 nominal 逐步一致、oracle 分支逐步一致、
冷启动违规为 0、执行上下文不匹配为 0。

但是，L54 的预注册主要 Gate **未通过**：

- issued jerk reduction = 0.000267，95% CI [-0.000416, 0.000898]；
- applied jerk reduction = 0.000250，95% CI [-0.000443, 0.000883]；
- oracle-retention margin 的 95% CI 下界为 -0.000259；
- 相对 cross-track RMSE 增幅为 5.87%，略高于 5% 门槛。

因此，不再把“稳定降低 jerk”作为当前跨层门的已支持结论。L51 的 jerk 正结果没有在
L54 新 seeds 上稳定复现，必须保留这一否定性证据。

## 2. L54 中形成的新开发假设

L54 的预注册次要指标显示了更稳定的路径效率信号：

- 平均路径长度缩短 0.0403 m；
- 层级 bootstrap 95% CI [0.00210, 0.09787]；
- 3/3 ICODE 模型块方向为正；
- 三条路径的平均缩短均为正：double-turn 0.0124 m、gentle-S 0.00776 m、
  slalom 0.1007 m；
- 成功 episode 净增 3，碰撞净增 0；
- 绝对 cross-track RMSE 平均改善 0.00139 m，但区间跨 0；
- long-delay 平均 alpha 为 0.531–0.566，matched 为 0。

该信号主要来自 long-delay，matched 域因 fail-closed 与 nominal 完全一致。由于路径长度
在 L54 中只是次要指标，本结果只能生成新假设，不能回头修改 L54 的主要结论。

## 3. L55 的独立确认问题

L55 预注册检验：在全新、此前未运行的 seeds 上，冻结的跨层 ICODE 门能否稳定缩短
路径，同时保持成功、碰撞、完成率与跟踪误差非劣。

比较条件、3 个 checkpoint、3 条路径、2 个物理域和全部控制/感知/安全配置均与 L54
一致。新 confirmation seeds 为 21760731–21760740，在本文件生成前未用于训练、校准
或闭环测试。共 720 episodes；运行命令必须显式使用 `--allow-sealed-confirmation`。

## 4. 预注册主要门槛

跨层门相对 nominal 必须同时满足：

1. path-length reduction 的层级 bootstrap 95% CI 下界 > 0；
2. path-length reduction 在 3/3 模型块中为正；
3. 三条路径的平均 path-length reduction 均为正；
4. cross-track improvement 的 95% CI 下界不低于 -0.005 m；
5. long-delay mean alpha ≥ 0.50，matched mean alpha ≤ 0.01；
6. 成功数不减少，碰撞数不增加；
7. completion ratio difference ≥ -0.01；
8. 平均 planner compute ≤ 50 ms；
9. 全部 artifact、identity、cold-start、context 和 seed 审计通过。

5 mm 的 cross-track 非劣界相对于 0.32 m 轮距和 0.25 m 终点容差较小，并与 L54
开发区间下界 -0.00482 m 对齐。该界限在 L55 数据生成前冻结。

## 5. 次要指标与解释纪律

Jerk、成功增益和各场景效应继续报告，但 jerk 不再是 L55 成败门槛。若 L55 通过，
可声称“跨层门在指定高延迟路径跟踪域中改善路径效率且满足冻结非劣约束”；仍不可声称
普遍改善所有控制质量、自动检测任意 OOD、已经实车验证或已经证明 RL+ICODE 完整体系。

