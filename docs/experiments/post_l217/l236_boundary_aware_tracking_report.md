# L236 footprint-aware corridor constraint 结果

## 结果定位

L236 attempt 2 的 treatment 已在三个 `config_resolved.yaml` 中确认生效。三回合均为 MuJoCo 3.2.3、development seed `923301001`、`nominal_seen`、`K=100`、2 iterations、Git SHA `d60f6f27815f571f9979e406037c1477af5402ec`，并完整保存逐回合工件。

L236 Gate **未通过**：

| 场景 | 步数 | 终止原因 | 碰撞 | 越界步数 | 最小边界余量 (m) | 完成度 | Gate 阈值 |
|---|---:|---|---:|---:|---:|---:|---:|
| Hairpin | 700 | max_steps | 0 | 0 | 0.0084 | 0.1093 | ≥0.145 |
| S-Chicane | 231 | boundary_violation | 0 | 1 | -0.0070 | 0.1765 | ≥0.205 |
| Infinity | 321 | boundary_violation | 0 | 1 | -0.0018 | 0.1047 | ≥0.135 |

## 解释

与 treatment 未注入的 attempt 1 相比，Hairpin 不再早期越界，并完整运行到 700 步；S-Chicane 和 Infinity 的越界也推迟到更靠后的路径位置。说明 boundary-aware rollout cost 的方向正确，但当前走廊给首障碍旁路留下的几何余量太小，模型预测到真实执行之间数毫米级偏差即可触发越界。

三场景仍未越过首障碍，因此不能进入四方法矩阵。继续单纯提高 penalty 或 K 会把“约束可行性”和“采样能力”混在一起。

## 下一步

按照既定 Geometry Qualification 计划，L237 只测试 `W=3.5D`：机器人碰撞直径 `D=0.5 m`，走廊全宽设为 `1.75 m`（半宽 `0.875 m`）。路径中心线、障碍物、核心方法、cost 权重、K、iterations、安全链和 seed 不变。

该变量直接增加首障碍旁路余量，用于回答当前失败是否由走廊几何过紧造成。结果无论正负均保留；不使用 sealed seeds。

