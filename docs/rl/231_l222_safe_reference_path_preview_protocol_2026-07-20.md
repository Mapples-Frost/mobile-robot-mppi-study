# L222 安全可行参考路径与多点 Path-Preview 开发协议

日期：2026-07-20

状态：预注册 development protocol，禁止作为正式确认性结果

前置证据：L221 Gate 与根因审计均已完成，尚未运行 L222 outcome 数据

## 1. 不变的论文核心

核心机制保持：

> Value-Consistent ICODE + Residual-Conditioned RL Prior + role-aware
> Reliability-Weighted Value/HSS + MPPI

本阶段不得关闭或削弱 `scan_guard`，不得改变 MuJoCo 障碍 geometry，不得
使用 simulator obstacle truth 作为在线 planner 输入，不得使用 sealed seeds。
Memory 继续关闭。

## 2. L222 的两个修复因素

### 因素 A：安全可行任务参考

三个旧失败场景保留完全相同的 MuJoCo 障碍物，只新增版本化 polyline。
离线 Gate 要求以 0.38 m safety inflation 审计时最小净空不小于 0.03 m。
在线时仍只使用外部 reference 和 LaserScan 障碍链；这与实车接收全局规划
路径的接口一致。

### 因素 B：共享多点 path-preview

启用配置：

```yaml
path_preview_enabled: true
path_preview_speed_mps: 0.45
path_preview_heading_weight: 0.25
```

第 h 个 rollout 状态对应路径弧长上的第 h 个未来参考姿态。原有
`goal_running_weight=1.0` 和 `goal_terminal_weight=25.0` 不改，只把目标从
单个静态 lookahead 点替换为按时间推进的路径姿态。该 cost 对
ICODE-MPPI、Simple combination 和 Full proposed 完全相同。

Residual-conditioned Actor 的后续候选在旧输入末尾增加四个车体坐标系路径
预览点，距离固定为 `[0.4, 0.8, 1.2, 1.6] m`，缩放为 2.0 m。旧 Actor
列原样复制，新输入列零初始化；critic、replay、optimizer 和随机数状态不
从旧 checkpoint 导入。

## 3. 分阶段 Gate

### Gate A：软件和几何契约

必须满足：

1. 三条新路径的 0.38 m 膨胀净空均不小于 0.03 m；
2. path preview 插值、角度、batch shape、NaN/Inf 测试通过；
3. `path_preview_enabled=false` 与旧 MPPI 数值回归一致；
4. PointGoal、旧 checkpoint、ROS bridge 和 scan_guard 不受影响；
5. 所有 arms 的 resolved config 中 path-preview cost 完全一致。

### Gate B：单种子工程探针

只用 development seed 91001，在三个旧失败场景的新安全参考上先运行
ICODE-MPPI。预算固定为 K=30、1 iteration、1400 steps。通过条件：零
碰撞，且至少两个场景成功；若不满足，只能在 development 协议下定位共享
planner 表示，不能查看或使用 sealed seeds。

### Gate C：Path-preview Actor 训练

使用训练 seeds `20262221--20262223` 的三个候选，validation seed 基准为
`20262821--20262823`；训练/验证均不得使用 development seeds 91001--91003。
每个候选 30k environment steps，按冻结的 validation 安全、成功、路径完成
度、碰撞规则选择一个 checkpoint。训练失败、NaN、零 update 或负向候选均
保留。

### Gate D：六地图三-seed 随机完整区组

方法：`ICODE-MPPI / Simple combination / Full proposed`。

场景：三个安全参考困难地图 + Giant U + Opposed U + Cylinder forest。

seeds：91001、91002、91003。

每个 seed 内按固定 schedule seed 随机化方法顺序；seed 是独立统计单位，
六地图是重复 strata。所有方法使用相同 K、iterations、horizon、cost、
LaserScan、MuJoCo 物理和安全链。

## 4. 冻结通过条件

1. 所有方法均零碰撞；
2. Full 在三个旧失败地图中至少两个达到 3/3 成功，第三个平均完成度不低于
   0.90；
3. Full 在原成功地图的成功数不低于 ICODE；
4. Full 相对 ICODE 的六地图 seed-cluster 平均成功率不为负，完成度不低于
   -0.02；
5. Full 明确优于等预算 Simple（成功率或完成度至少一项严格为正）；
6. residual context、counterfactual authority、HSS proposal authority 至少在
   一个场景非平凡；
7. 同时报告规划耗时，不能用成功率掩盖不可部署的计算代价。

不通过则保留完整负向结果，禁止注册 sealed benchmark。通过后才允许创建
新的、从未使用的 sealed seed manifest，并一次性运行。

## 5. 防止结果美化

- 不筛 seed、不删除失败、不用 qualification 充当正式确认；
- L218/L219/L220/L221 文件与数据只读保留；
- 新参考、新 Actor 和新 benchmark 使用 `l222` 独立名称；
- 任何进一步参数变化必须先写新的开发协议，再读取其 outcome；
- 所有图同时提供原始 CSV、配置、Git SHA、checkpoint SHA 和生成脚本。
