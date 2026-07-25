# Probabilistic Collision Risk V1 资格报告

日期：2026-07-23  
平台：Windows native  
分支：`codex/change-aware-probabilistic-mppi`  
结论：**PASS，可以进入在线障碍关联与风险感知 MPPI development smoke。**

## 1. 本阶段完成内容

- 新增严格的 `GaussianMixtureObstacleForecast` 数据契约；
- 新增二维混合高斯圆碰撞风险的保守切平面上界；
- 输出单步上界、整时域 union bound、未裁剪概率质量与硬阈值状态；
- 将概率风险作为可选端口接入通用 MPPI；
- 端口默认关闭，历史控制器行为不变；
- 端口打开但预测缺失、时间戳错位或步长错位时 fail closed；
- 增加输入、数值、因果、单调性、Monte Carlo 和 MPPI 排序测试；
- 生成正式资格产物、环境清单、源码哈希与完整性审计。

## 2. 概率含义

该风险量不是把高斯均值周围画一个固定圆，也不是只判断预测均值是否碰撞。

对每个 IMM 模式，它将机器人指向障碍均值的方向作为最接近碰撞圆的方向，把二维
高斯投影到这个方向后计算一维概率。因为整个碰撞圆包含于对应切平面事件中，该概率
是单步圆碰撞概率的保守上界。均值已经进入组合半径时直接返回 1。

多个 IMM 模式按模式概率加权。多个未来时刻不假设相互独立，而是使用 union bound：
把各时刻风险相加并在 1 处裁剪。这可能偏保守，但不会凭空假设未来误差彼此独立。

冻结的组合半径包含：

- 机器人半径：0.25 m；
- 障碍物自身半径；
- 显式安全裕量：0.10 m。

这个 0.10 m 裕量用于回应 predictor held-out 中模式切换阶段仍偏自信的问题。

## 3. 正式 Gate 结果

13/13 项全部通过。

| 检查 | 结果 |
|---|---:|
| 合法输出有限且位于 `[0,1]` | PASS |
| 均值在组合半径内返回 1 | PASS |
| 更近的外部均值风险更高 | PASS |
| 更大的径向协方差风险更高 | PASS |
| 混合权重线性 | PASS，误差 0 |
| 模式置换不变 | PASS，误差 0 |
| 穿越轨迹风险高于绕行 | PASS |
| union bound 支配所有单步风险 | PASS |
| MPPI 端口关闭时代价完全一致 | PASS，最大差值 0 |
| MPPI 端口打开后惩罚穿越 | PASS |
| Monte Carlo 上界违反 | PASS，0/4 |
| 参考批量计算小于 25 ms | PASS |
| 输入契约拒绝非法数据 | PASS |

合成穿越路径的未裁剪风险质量为 3.0000，绕行路径为 0.001976。使用 development
smoke 参数时，MPPI 穿越候选代价为 10244.37，绕行候选为 39.53。

## 4. Monte Carlo 检查

四组预定义各向同性/各向异性二维高斯均通过：

| Case | 解析上界 | Monte Carlo 圆碰撞率 | 上界违反 |
|---|---:|---:|---:|
| 1 | 0.15896 | 0.14619 | 否 |
| 2 | 0.11665 | 0.10214 | 否 |
| 3 | 0.06662 | 0.02984 | 否 |
| 4 | 0.05636 | 0.04209 | 否 |

Monte Carlo 仅用于实现审计；保守性本身来自碰撞圆盘属于切平面事件这一集合关系。

## 5. 计算开销

Windows CPU，`K=600` 个 MPPI 候选、`H=36` 步、`M=4` 个 IMM 模式，30 次重复：

- minimum：10.999 ms；
- median：12.045 ms；
- P90：13.216 ms；
- maximum：13.731 ms；
- 冻结 Gate：maximum ≤ 25 ms。

这是风险模块自身的批量时间，不等于整个 MPPI 控制周期时间。闭环阶段仍需报告完整
控制器端到端 `compute_ms`。

## 6. 回归与完整性

- 新风险接口测试：11/11 通过；
- 风险 + 通用 MPPI + 边界代价相关回归：45/45 通过；
- 全仓库：1064 passed，5 failed；
- 5 个失败与此前记录的历史 RL checkpoint/config 失败完全相同，没有新增失败；
- 正式资格产物完整性：通过；
- 被审计文件 SHA-256 不匹配：0；
- provenance 包含风险模块、MPPI 接入代码、冻结配置和预注册文档。

历史失败：

1. `test_l77_uses_preexisting_l72_checkpoint_choices_and_fresh_seeds`
2. `test_l35_design_is_frozen_disjoint_and_balanced`
3. `test_l71_candidate_entries_bind_every_saved_training_checkpoint`
4. `test_l73_uses_fresh_deployment_seeds_and_l72_checkpoints`
5. `test_l74_locks_selected_checkpoint_and_binds_sealed_seeds`

## 7. 结论边界与下一步

本次通过证明风险数学接口和 MPPI 代价端口值得进入闭环开发，但尚未证明：

- LaserScan 中的多个点能稳定关联成同一个动态障碍；
- 在线 Change-Aware IMM 能在 MuJoCo 控制循环中按 10 Hz 实时更新；
- 风险权重和硬阈值能取得合理的安全—效率权衡；
- 机器人能在反复穿越必经路线的无规律障碍前正确等待并择机通过。

下一阶段应先实现因果的单目标检测/关联与 10 Hz 预测对齐，然后在冻结的起点—终点
走廊上进行 risk-disabled 与 risk-enabled 的成对 development smoke。该阶段仍不使用
held-out 或 sealed seed。

## 8. 可复核产物

- 配置：`configs/research/probabilistic_collision_risk_v1.yaml`
- 预注册：`docs/experiments/dynamic_uncertainty/PROBABILISTIC_COLLISION_RISK_V1_PREREGISTRATION.md`
- 正式结果：`research_artifacts/probabilistic_collision_risk_v1_qualification/analysis/summary.json`
- 性能明细：`research_artifacts/probabilistic_collision_risk_v1_qualification/analysis/runtime_benchmark.csv`
- 完整性审计：`research_artifacts/probabilistic_collision_risk_v1_qualification/integrity_audit.json`

