# L244 BC-Anchored Path Actor 的 Tracking Development Gate

日期：2026-07-21
状态：预注册 development protocol，尚未读取 L244 outcome

## 1. 研究问题与固定机制

本 Gate 检验 L243 通过独立 validation 冻结选中的 BC-anchored Actor，能否把离线路径条件能力
转化为三个 L234 Tracking 场景中的闭环增益。核心机制保持：

> Value-Consistent ICODE + Path/Residual-Conditioned RL Prior + role-aware
> Reliability-Weighted Value/HSS + MPPI

不得改变 ICODE、MPPI cost、MuJoCo 地图、LaserScan、scan_guard、安全仲裁或 L239 的
`W=4.0D` 几何条件。

## 2. 冻结模型

```text
path: results/research_platform/rl/l243_path_preview_bc_anchor_seed20262332_30k_v1/checkpoints/step_000030000.pt
sha256: e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c
train seed: 20262332
selected global step: 30000
validation: 0/30 collisions, 6/30 successes
```

选择严格来自 L243 固定 validation 字典序；L234、L239 和 L242 的闭环 outcome 未参与模型选择。

## 3. 实验设计

- development seed：`923301001`；
- physics：`nominal_seen`；
- scenes：L239 `W=4.0D` 的 Hairpin、S-Chicane、Infinity；
- 新运行 arms：`icode_mppi`、`full_proposed`；
- 只读历史对照：L242 同 seed、同场景、同预算的 selected-initial Full；
- rollout budget：每次决策 100 条 rollout；Full 为 50 candidates × 2 iterations，ICODE 为
  一次 100-sample batch；
- 最大 700 control steps；
- qualification=`1`，不得使用 sealed seeds。

L244 Full 与 L242 Full 的唯一 Actor 因素分别为 L243 BC-selected checkpoint 与 L241
selected-initial checkpoint。新跑 ICODE 用于核验当前执行环境和等预算参考；L242 只读 Full 不重跑，
避免生成数值等价的重复证据。

## 4. 读取前完整性 Gate

读取方法效果前必须确认：

1. 三个场景各有 `icode_mppi` 与 `full_proposed`，共 6 个唯一 MuJoCo 回合；
2. MuJoCo 3.2.3、seed、qualification、K、iterations、max steps 均写入 resolved config；
3. coupled Actor SHA、Git SHA、manifest SHA、ICODE/value checkpoints 与 calibration 来源完整；
4. 每回合存在 `config_resolved.yaml`、`trajectory.csv`、`metrics.json`、`provenance.json`；
5. L242 只读对照的地图、物理、cost、安全链和 rollout 预算与 L244 一致；
6. 不存在重复 experiment key、Traceback、Exception、NaN 或 Inf。

## 5. 预注册判据

训练 Gate 已单独通过；闭环 Gate 只有同时满足以下条件才通过：

1. L244 Full 三场景均零碰撞；
2. 相对 L242 selected-initial Full，平均 path completion 提升至少 `0.02`；
3. 任一场景相对 L242 Full 的 completion 回退不超过 `0.02`；
4. boundary violation 总数不高于 L242 Full；
5. proposal authority 或 guided-elite contribution 非平凡；
6. 同时报告与 ICODE、L242 Full 的成功、完成度、cross-track、boundary、安全干预和 planner time。

若未通过，必须保留所有负向结果并停止 Tracking sealed 注册；不得筛 seed、删除失败、放松
scan_guard 或根据本轮 outcome 回头改选 checkpoint。本轮是单 development seed 的机制 Gate，
即使通过也只允许进入多 seed development 验证，不能直接作为论文确认性结论。
