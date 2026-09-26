# L236 attempt 1 配置注入审计

## 状态

该次运行完成了三场景，但**不属于 L236 treatment 的有效实验**。

## 发现

`run_final_paper_benchmark.py` 只从 `final_benchmark.planner_overrides` 向逐回合 planner 注入资格实验覆盖项。首版 manifest 将边界参数写在顶层 `planner`，导致三个 `config_resolved.yaml` 均不存在 `path_boundary_*` 字段。

三个回合因此与 L235 数值完全一致：

| 场景 | 步数 | 路径完成度 | 终止原因 |
|---|---:|---:|---|
| Hairpin | 138 | 0.0904 | boundary_violation |
| S-Chicane | 369 | 0.1751 | boundary_violation |
| Infinity | 42 | 0.0256 | boundary_violation |

## 处理

- 原始 attempt 1 目录和日志全部保留；
- 不将其解释为 boundary-aware cost 的负向结果；
- manifest 改用已有、受审计的 `final_benchmark.planner_overrides` 注入接口；
- attempt 2 启动后，必须先检查 resolved config 中四个 `path_boundary_*` 字段，再等待全部场景完成；
- seed、地图、方法、K、iterations、checkpoints、其他 cost 和安全链保持不变。
