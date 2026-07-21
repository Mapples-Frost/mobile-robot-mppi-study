# L247 Tracking 公平回合预算 Development Protocol

日期：2026-07-21  
状态：预注册 development protocol；尚未读取任何 L247 outcome

## 1. 研究问题

L246 只读审计证明，固定 `700` control steps 对 Hairpin 和 Infinity 在物理上不可能完成：
两条路径即使始终以最大线速度 `0.65 m/s` 行驶，理论下界仍分别为 816 和 750 步。

本轮只回答：

> 当回合上限由冻结路径长度公平决定后，L244 的失败中有多少属于不合理的时间截断，
> 又有多少仍属于边界、局部最小值或 proposal 能力问题？

本轮不检验新算法，不声称论文主方法获得提升。

## 2. 唯一 treatment

唯一改变是 `experiment.max_steps`。对场景 \(s\) 冻结：

$$
N_s = \left\lceil
\frac{1.25 L_s}{0.30\,\mathrm{m/s}\times0.1\,\mathrm{s}}
\right\rceil.
$$

| 场景 | L246 路径长度 | 冻结 max steps |
|---|---:|---:|
| Hairpin | 53.023 m | 2210 |
| S-Chicane | 33.706 m | 1405 |
| Infinity | 48.715 m | 2030 |

同一场景的全部 arms 使用完全相同的上限。runner 在启用 `max_steps_by_scene` 后采用
fail-closed 语义：任一选中场景缺少显式预算即终止，不允许静默回退到 700。

## 3. 保持冻结的条件

- 核心机制：Value-Consistent ICODE + Path/Residual-Conditioned RL Prior +
  role-aware Reliability-Weighted Value/HSS + MPPI；
- Actor：L243 validation-only 选中的 BC-anchored Actor，SHA256
  `e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c`；
- scenes：L239 `W=4.0D` Hairpin、S-Chicane、Infinity；
- development seed：`923301001`；
- physics：`nominal_seen`；
- arms：`icode_mppi`、`full_proposed`；
- rollout budget：每次决策 100 条；Full 为 50 candidates × 2 iterations，
  ICODE 为 100 candidates × 1 iteration；
- L244 的 MPPI cost、boundary cost、地图、障碍、LaserScan、scan_guard 和安全仲裁全部不变；
- qualification=`1`，不得使用 sealed seeds。

## 4. 完整性 Gate

读取效果前必须确认：

1. 三场景 × 两 arms，共 6 个唯一 MuJoCo 回合；
2. resolved config 和 `progress.csv` 中的预算分别精确为 2210/1405/2030；
3. seed、qualification、MuJoCo、Git、manifest、Actor、ICODE/value checkpoint 与 calibration provenance 完整；
4. 每回合具有 `config_resolved.yaml`、`trajectory.csv`、`metrics.json`、`provenance.json`；
5. 不存在重复实验键、Traceback、Exception、NaN 或 Inf；
6. L244/L245/L246 的历史负向结果保持只读且不覆盖。

## 5. 预注册判据

L247 是实验设计修复 Gate，而不是性能 Gate。满足以下条件即允许进入 L248：

1. 6 个回合全部通过完整性 Gate；
2. Hairpin 和 Infinity 不再由于旧的 700-step 上限终止；若失败，必须记录新的真实终止原因；
3. 所有碰撞、boundary violation、停滞和 max-step 失败完整保留；
4. 报告每个 arm/scene 的 success、completion、termination、boundary、collision、cross-track、
   safety burden、Actor authority 和 planner time；
5. 不依据本轮 outcome 修改预算公式或挑选 seed。

L247 通过只说明 success denominator 合法，不说明 Full 优于 ICODE。随后按既定顺序进入 L248，
把 footprint 边界约束前移到所有 MPPI arms 的候选评价阶段。

## 6. 禁止事项

- 不重新训练 Actor 或 ICODE；
- 不修改网络结构、cost 权重、地图、障碍或安全链；
- 不放松 boundary termination；
- 不使用 sealed seeds；
- 不筛 seed、不删除失败、不把 qualification 写成论文确认性结果；
- 不因运行中间效果回头调参。
