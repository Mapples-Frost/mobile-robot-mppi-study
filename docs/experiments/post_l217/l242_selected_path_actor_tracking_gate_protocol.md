# L242 冻结选中 Path Actor 的 Tracking Development Gate

日期：2026-07-21  
状态：预注册 development protocol，尚未读取 L242 outcome

## 1. 问题与不变项

本 Gate 检查 L241 按冻结 validation 规则选中的 checkpoint 在三个 L234 Tracking
场景中是否产生闭环增益。核心机制保持：

> Value-Consistent ICODE + Path/Residual-Conditioned RL Prior + role-aware
> Reliability-Weighted Value/HSS + MPPI

不得改变 ICODE、MPPI cost、MuJoCo 地图、LaserScan、scan_guard 或安全仲裁。

## 2. 冻结 checkpoint

```text
path: results/research_platform/rl/l241_path_preview_actor_seed20262222_30k_v1/checkpoints/initial.pt
sha256: 150a2d7ba1e4ab125fcb94e614810acb33e86f549b3eaa310e6120cd02440982
train seed: 20262222
selected global step: 0
```

step 0 的含义必须透明报告：这是扩展观测后的初始化策略，不是训练后策略；新增
path-preview 权重列为零初始化。

## 3. 设计

- development seed：`923301001`；
- physics：`nominal_seen`；
- scenes：L239 的 Hairpin、S-Chicane、Infinity，走廊宽度 `4.0D`；
- 预算：K=100（50 samples × 2 iterations），最大 700 control steps；
- 新运行 arms：`icode_mppi`、`full_proposed`；
- 旧 Actor Full 对照：只读复用 L239 同 seed、同场景、同预算结果；
- qualification=1；不使用任何 sealed seed。

复用 L239 旧 Full 是为避免用相同配置重复生成一份数值等价结果。新 Full 与旧
Full 的唯一方法因素是 coupled Actor checkpoint；ICODE-MPPI 提供无 RL prior
的同预算参考。

## 4. 读取前 Gate

完成后先核验：

1. 2 arms × 3 scenes = 6 个唯一回合；
2. MuJoCo 3.2.3、seed、qualification、K、iterations、max steps 与 resolved config；
3. checkpoint SHA、Git SHA、逐回合 trajectory/metrics/provenance；
4. 新旧 Full 的地图、物理、cost、safety 和 rollout 预算一致。

## 5. 预注册判据

所有方法必须零碰撞。相对 L239 旧 Actor Full，冻结选中模型只有同时满足以下条件
才称为 development 正向：

1. 三场景平均完成度提升不小于 0.02；
2. 任一场景提升不以另一场景超过 0.02 的回退为代价；
3. proposal authority 或 guided-elite contribution 非平凡；
4. boundary violation 数不增加；
5. 平均 planner time 必须同时报告。

若不通过，结论限定为“当前训练/选择未产生 Tracking 闭环增益”。不得改写为正式
论文结论、不得筛 seed 或删除失败。后续修复只能在新的 development 协议中处理。
