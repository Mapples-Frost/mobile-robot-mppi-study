# L245 BC Actor Boundary Buffer 单变量实验报告

日期：2026-07-21

性质：development / qualification，单一 development seed；严格保留的负向结果

## 1. 完整性

- 完成 `full_proposed` × Hairpin/S-Chicane/Infinity 共 3 个唯一 MuJoCo 回合；
- physics=`nominal_seen`，seed=`923301001`，qualification=`1`；
- Git SHA：`24e5d02`；MuJoCo 3.2.3；
- selected Actor SHA256：
  `e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c`；
- K=100，50 candidates × 2 iterations，最大 700 步；
- 三个 resolved config 均确认 `path_boundary_buffer=0.075`；
- 所有逐回合工件完整，日志无 Traceback、Exception、NaN 或 Inf。

L245 相对 L244 的唯一变化是公共 `path_boundary_buffer: 0.05 -> 0.075 m`。Actor、
ICODE、HSS、其余 MPPI cost、地图、物理与安全链均保持不变。

## 2. 结果

| scene | completion | Δ vs L244 | Δ vs L242 | collision | boundary steps | min boundary margin (m) | cross-track RMSE | safety interventions | proposal authority | guided elites | planner mean (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hairpin | 0.2589 | **-0.0833** | +0.1512 | 0 | 0 | 0.0031 | 0.4278 | 73 | 0.1554 | 30 | 393.3 |
| S-Chicane | 0.1816 | -0.0061 | +0.0058 | 0 | **1** | **-0.0097** | 0.4607 | 37 | 0.0953 | 25 | 409.0 |
| Infinity | 0.1061 | +0.0005 | +0.0036 | 0 | **1** | **-0.0067** | 0.3340 | 92 | 0.1054 | 20 | 400.1 |

汇总：

- 三场景平均 completion：0.1822；
- 相对 L242 Full 平均增益：+0.0535；
- 相对 L244 Full，Hairpin 回退 -0.0833，超过允许的 -0.02；
- collision：0/3；boundary violation：2 步；
- 最小 footprint boundary margin：-0.0097 m；
- proposal authority 平均 0.1187，guided elites 合计 75，RL prior 仍被非平凡使用。

## 3. Gate 判定

**L245 Gate 失败。**

它没有满足：

1. 三场景零 boundary violation；
2. 最小 footprint boundary margin 非负；
3. 相对 L244 任一场景回退不超过 0.02。

L245 仍保持零碰撞以及相对 L242 的平均 completion 增益，但这不足以越过预注册安全 Gate。
增加 buffer 没有产生预期的单调安全改善，并以明显降低 Hairpin 推进为代价。因此不能继续在同一
development seed 上扫描更多 buffer，也不能注册当前 Tracking sealed benchmark。

## 4. 对研究主线的限定结论

L243/L244 的证据仍说明 BC anchor 能让 residual-conditioned Actor 在特定复杂转弯上产生真实的
闭环推进增益；L245 则说明该增益不能仅靠扩大 MPPI boundary buffer 转化为稳定的跨场景安全收益。
这不否定 ICODE 或 RL+ICODE 的核心方向，但否定了“当前 Actor + 简单 buffer 调整已经足够”的说法。

后续若重新开启 Tracking 开发，应先形成新的机制假设，例如显式的 executed-footprint risk-aware
proposal arbitration，而不是继续调 buffer。本轮在此停止，不读取 sealed seeds。

## 5. 工件

- 原始结果：`results/research_platform/rl/tracking_l245_buffer075_*_seed923301001/`；
- 数值表：`docs/experiments/post_l217/tables/l245_boundary_buffer_probe_summary.csv`；
- 图：`docs/experiments/post_l217/figures/fig_l245_boundary_buffer_probe.pdf`；
- 预注册协议：`docs/experiments/post_l217/l245_bc_actor_boundary_buffer_protocol.md`。
