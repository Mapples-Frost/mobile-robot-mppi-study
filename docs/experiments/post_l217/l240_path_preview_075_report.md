# L240 Tracking Path-Preview Qualification（0.75 m/s）结果

## 结果定位

L240 的三个 MuJoCo 3.2.3 development 回合均完整结束并通过来源核验。实验固定为 `full_proposed`、`nominal_seen`、seed `923301001`、`K=100`（50 candidates × 2 iterations）、700 步上限、`W=4.0D`、5 cm boundary buffer、Git SHA `02ba0f6`。唯一 treatment `path_preview_speed_mps=0.75` 在三个 resolved config 中均已生效，逐回合工件齐全。

L240 Gate **未通过**：

| 场景 | 步数 | 终止原因 | 碰撞 | 越界步数 | 最小 footprint 余量 (m) | 完成度 | Gate 阈值 |
|---|---:|---|---:|---:|---:|---:|---:|
| Hairpin | 700 | max_steps | 0 | 0 | 0.0007 | 0.1085 | ≥0.145 |
| S-Chicane | 275 | boundary_violation | 0 | 1 | -0.0158 | 0.1789 | ≥0.205 |
| Infinity | 700 | max_steps | 0 | 0 | 0.0328 | 0.1050 | ≥0.135 |

## 失败分类

延长共享 MPPI path preview 后，Hairpin 和 Infinity 均在零碰撞、零越界下运行到 700 步，但路径完成度仍停留在首障碍附近；S-Chicane 更早越界。因此，失败不是单纯因为固定 horizon 看得不够远。

机制日志给出了更直接的限制：

| 场景 | RL proposal authority 均值 | fallback fraction | guided elite 总数 |
|---|---:|---:|---:|
| Hairpin | 0.1018 | 0.8982 | 8 |
| S-Chicane | 0.0472 | 0.9528 | 0 |
| Infinity | 0.1151 | 0.8849 | 3 |

当前 L219 Actor 的 residual context 已激活，但它没有接收多点路径预览输入；在这些新 Tracking 几何中，Actor proposal 大部分时间被 HSS 正确回退，Full proposed 因而主要退化为 Gaussian ICODE-MPPI。继续改变走廊、buffer 或 preview cost 只能调共享 baseline，不能检验论文要求的“残差感知 RL prior 如何根据未来路径主动提出绕行候选”。

## 决策

停止对 L234 场景继续做 planner 超参数搜索。转入已经在 L222 协议中预注册、但尚未实际执行的 Path-Conditioned Residual Actor Gate：给 residual-conditioned Actor 增加四个车体坐标系多点路径预览点，训练三个独立候选，并只用隔离的 validation seeds 选择 checkpoint。

该步骤不改变 Value-Consistent ICODE、HSS、MPPI、地图、安全链或论文核心方向；它补齐的是原计划中的 `path geometry -> residual-conditioned RL prior` 可观测输入。L236–L240 全部负向结果保留。
