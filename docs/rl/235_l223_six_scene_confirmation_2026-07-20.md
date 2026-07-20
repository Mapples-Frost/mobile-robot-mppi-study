# L223 六场景 MuJoCo 统一确认报告

日期：2026-07-20

实验性质：开发集平台可行性确认（不是封存正式统计结论）

## 1. 结论

在同一 Git 版本、同一 ICODE-MPPI 配置、同一开发 seed 和同一 MuJoCo 后端下，六张扩展地图均安全到达目标：

- 成功：6/6；
- 碰撞：0/6；
- 终止原因：全部为 `goal_reached`；
- 最低观测净空：0.117 m；
- 平均路径完成度：约 96.95%；
- 平均横向跟踪 RMSE：约 0.134 m。

因此，L218/L222 新设计的六张地图已经通过本轮开发集平台可行性 Gate。按照当前任务约定，本轮到此停止，不继续启动 Actor、RL 或 sealed benchmark。

## 2. 六场景结果

| 场景 | 成功 | 碰撞 | 步数 | 路径完成度 | 最小净空 (m) | 横向 RMSE (m) | 安全干预次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Giant U | 是 | 否 | 356 | 96.83% | 0.313 | 0.188 | 24 |
| Opposed U | 是 | 否 | 452 | 96.80% | 0.224 | 0.184 | 97 |
| Cylinder forest | 是 | 否 | 289 | 96.35% | 0.183 | 0.154 | 33 |
| Cylinder spiral | 是 | 否 | 368 | 96.04% | 0.117 | 0.062 | 110 |
| Nested U | 是 | 否 | 632 | 96.82% | 0.143 | 0.071 | 305 |
| Serpentine | 是 | 否 | 1297 | 98.94% | 0.192 | 0.145 | 444 |

说明：到达判据允许机器人在目标容差内终止，因此“路径完成度”不必等于 100%。安全干预包含减速等保护行为，并不等同于碰撞或硬急停。

## 3. 相对 L222 的明显正向变化

三张困难地图使用相同 ICODE-MPPI、seed 91001 和 1400 步预算进行对照。L223 仅对仿真平台的共享 `scan_guard` 做有界校准，没有关闭安全链、没有缩小机器人碰撞几何、没有改变 ICODE、MPPI 代价或地图障碍：

| 场景 | L222 严格 guard | L223 校准 guard | 变化 |
|---|---|---|---|
| Serpentine | 1400 步未到达，完成度约 31.7% | 1297 步到达，完成度 98.9% | 从失败变为成功 |
| Nested U | 1400 步未到达，完成度约 44.8% | 632 步到达，完成度 96.8% | 从失败变为成功 |
| Cylinder spiral | 1215 步到达 | 368 步到达 | 同为成功，执行步数下降约 69.7% |

这说明此前困难地图失败的主因是仿真安全膨胀与通道几何不匹配，而不是 ICODE-MPPI 本身无法沿参考路径行驶。经有界校准后，六图共享平台具备后续公平算法比较的基础条件。

## 4. 冻结执行条件与来源

- Git SHA：`b77ec38489995e8114838de1483a52c71b3c01e5`
- 分支：`codex/l218-expanded-mujoco-scenes`
- MuJoCo：3.2.3
- plant backend：`mujoco_diff_drive`
- prediction mode：`icode_residual`
- RL：关闭
- seed：91001（development）
- MPPI：`K=30`，1 iteration
- 路径表示：共享 polyline path preview
- L223 manifest：`configs/research/expanded_navigation_relaxed_guard_l223.yaml`
- 仿真 `scan_guard`：near-body/side 0.32 m，hard/front stop 0.34 m；机器人碰撞半径仍为 0.25 m
- 实车 ROS bridge、实车 `scan_guard` 和硬件安全链：未修改

六个回合均通过 Git SHA、配置哈希、MuJoCo 版本、seed、预测模式、预算和逐回合工件审计。统一机器可读审计位于：

- `docs/rl/artifacts/l223/six_scene_confirmation/gate_a_audit.json`
- `docs/rl/artifacts/l223/six_scene_confirmation/gate_a_episode_audit.csv`
- `docs/rl/artifacts/l223/six_scene_confirmation/gate_a_progress_windows.csv`

## 5. 可复现命令

```bash
python3 experiments/rl/analyze_l222_gate_a.py \
  --run-glob='results/research_platform/rl/l223_gate_b_icode_*_seed91001' \
  --run-glob='results/research_platform/rl/l223_six_scene_icode_*_seed91001' \
  --expected-git-sha=b77ec38489995e8114838de1483a52c71b3c01e5 \
  --expected-manifest=configs/research/expanded_navigation_relaxed_guard_l223.yaml \
  --expected-scenes=l218_giant_u,l218_opposed_u,l218_cylinder_forest,l222_serpentine_safe,l222_nested_u_safe,l222_cylinder_spiral_safe \
  --required-successes=6 \
  --output-dir=docs/rl/artifacts/l223/six_scene_confirmation
```

论文图由 `experiments/rl/plot_l223_six_scene_confirmation.py` 从上述审计和原始逐步轨迹直接生成，同时输出 PDF 矢量图和 300 dpi PNG。

## 6. 科研解释边界

这是一项明确的正向工程与开发集结果，但必须准确限定其含义：

1. 它证明六张地图在真实 MuJoCo 刚体/接触仿真链中对当前 ICODE-MPPI 平台可行；
2. 它证明有界仿真安全校准修复了此前的几何不可行性；
3. 它不证明 Full Proposed 或 RL 优于 ICODE-MPPI，因为本轮 RL 关闭；
4. 它只有一个开发 seed，不能替代多 seed sealed benchmark 和统计推断；
5. 本轮结果可作为后续算法消融的基础设施资格验证，不应单独当作论文核心性能结论。

## 7. 交付内容

Windows 桌面交付包包含本报告、审计表、论文图、六个完整原始结果目录、配置、协议、运行日志和 SHA-256 文件清单。仓库中同步保存小型审计工件、图、报告以及完整原始结果快照，便于后续追溯。

## 8. 最终回归验证

```text
python -m py_compile experiments/rl/analyze_l222_gate_a.py \
  experiments/rl/plot_l223_six_scene_confirmation.py
结果：通过

PYTHONPATH=.:src pytest tests/experiments/test_final_paper_benchmark.py \
  tests/platform/test_generic_mppi.py \
  tests/platform/test_l222_safe_references.py -q
结果：50 passed in 0.51s
```
