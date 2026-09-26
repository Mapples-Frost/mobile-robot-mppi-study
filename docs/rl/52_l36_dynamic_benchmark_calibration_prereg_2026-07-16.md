# L36：动态 benchmark 可辨识性校准预注册

日期：2026-07-16  
状态：在运行 L36 前冻结；本阶段不读取任何 RL checkpoint 输出。

## 1. 触发原因

L35 独立确认中，diagonal 场景 30/30 次碰撞，offset 场景 30/30 次不碰撞。该完全分离
使 collision 出现地板/天花板效应，无法区分“策略略有改善”与“场景结局已由几何和运动
时序锁死”。L35 仍按预注册判定失败，不能事后删除或替换；L36 是新的 benchmark
development 阶段。

## 2. 校准原则

L36 只运行以下控制器：

```text
traditional GoalWarmStart MPPI × nominal rollout dynamics
```

明确关闭 RL 和 memory，不加载 RL checkpoint，也不加载 ICODE checkpoint。这样场景选择
只由传统 baseline 的可辨识性决定，不会为了放大本文方法而选择有利场景。

真实 plant 固定为 `combined_unseen` MuJoCo 物理域；LaserScan、scan guard、local obstacle
layer、temporal safety 和最终仲裁保持开启。planner 不能读取动态障碍的真值位置或未来轨迹。

## 3. 因子与样本

候选场景以单个横穿圆柱为基础，系统改变：

- 穿越方向：反对角、陡斜、中央斜、浅斜、水平、偏置；
- 半径：0.14–0.26 m；
- 周期：12–18 s；
- 相位、周期与端点继续做 seed 驱动随机化。

共 8 个候选场景，每个使用 8 个全新 development episode seed
`20560731`–`20560738`，总计 64 个闭环 episode。运行顺序由 seed `2026071602`
随机化。

## 4. 预注册选择规则

一个候选只有同时至少产生 1 次 success 和 1 次 collision，才是 eligible。若 eligible
候选不少于 3 个，则按如下固定规则选择 easy/moderate/hard 三个场景：

1. 目标 collision rates 固定为 0.25、0.50、0.75；
2. 依次选择与目标 rate 绝对差最小的尚未选候选；
3. 若并列，优先 final-goal-distance 方差更大者；
4. 若仍并列，按配置中的候选顺序选择。

L36 Gate 只有同时满足以下条件才通过：

- 64 个 episode 完整、无重复、无 NaN/Inf；
- 至少 3 个 eligible 候选；
- 三个已选候选的 collision-rate span 至少为 0.25；
- 每个已选候选均至少有 1 次 success 和 1 次 collision。

若 Gate 失败，不运行 RL 对比，而是基于本轮传统 baseline 结果提出新的场景因子范围并
重新预注册。若 Gate 通过，三个场景只用于后续 development；最终 confirmation 仍必须
使用独立的新 seed 和未参与选择的运动参数。

## 5. 运行入口

```bash
python3 experiments/rl/run_dynamic_benchmark_calibration.py \
  --config configs/rl/dynamic_benchmark_calibration_l36.yaml \
  --output-dir results/research_platform/rl/l36_dynamic_benchmark_calibration_20260716_v1

python3 experiments/rl/summarize_dynamic_benchmark_calibration.py \
  --config configs/rl/dynamic_benchmark_calibration_l36.yaml \
  --input-dir results/research_platform/rl/l36_dynamic_benchmark_calibration_20260716_v1
```

