# Exploratory Data Analysis Report: L234 Tracking smoke

**Generated:** 2026-07-21  
**Source:** `results/research_platform/rl/tracking_smoke_l234_seed923301001_attempt2`

## Executive Summary

该目录包含 12 个 MuJoCo 3.2.3 development qualification 回合。实验设计矩阵和逐回合工件完整，数据不存在重复实验单元；但 12 个回合全部未完成路径。因此该数据集适合验证数据链和定位失败模式，不适合估计方法有效性或形成论文结论。

## Basic Information

- 格式：CSV、JSON、YAML
- 主要表：`progress.csv`
- 表结构：12 行 × 236 列
- 实验单元：method × scene × physics_domain × seed
- 独立统计单位：seed；本轮仅 1 个 development seed
- 逐拍轨迹：12 个 CSV，均存在且可解析
- 结果文件：12 个 metrics JSON、12 个 resolved config、12 个 run provenance

## Quality Assessment

- 完成度：12/12 计划单元存在。
- 唯一性：0 个重复实验键。
- 预期缺失：`time_to_goal_s` 12 个空值，因为没有回合成功；`minimum_dynamic_obstacle_center_distance` 12 个空值，因为没有动态障碍物。
- 关键 Tracking 字段：CTE、P95、积分偏差、切向航向误差、footprint 边界余量、越界步数、路径完成度、障碍通过/恢复事件均存在。
- provenance：顶层记录 manifest/checkpoint/calibration hashes；每个回合记录 config hash 与 Git SHA。
- 运行环境：12/12 metrics 均声明 `backend=mujoco_diff_drive`、`mujoco_version=3.2.3`。

## Statistical Summary

所有方法成功率均为 0。自适应 HSS 方法的平均 CTE RMSE 为 0.1046–0.1222 m，固定 HSS 方法为 0.3073–0.3364 m；但所有回合都在完成整条路径前终止，所以这些数字受到停止位置和暴露时长严重混杂，不能进行因果比较。

## Key Findings

1. 数据链达到科研审计所需的基本完整性。
2. 固定 HSS 方法在两个场景发生边界越界；自适应 HSS 方法保持正边界余量但在障碍物前长期停滞。
3. 首个中心线附近障碍物是共同失败位置；无近障碍初始阶段的速度命令正常。
4. `K=30`、1 iteration 的 smoke 预算不足以证明这些几何通道对当前局部 MPPI 可解。
5. 样本量只有一个 development seed，禁止置信区间、显著性检验和正式方法排序。

## Recommendations

- 先做首障碍绕行能力 qualification，而不是直接增加完整路线运行时间。
- 只改变所有方法共享的 rollout/refinement 预算；先保持核心方法、cost、安全链和场景不变。
- 通过预注册的开发 Gate 后，才进入多 seed 和双物理域矩阵。
- sealed Tracking seeds 保持未使用。

## Analysis Software

- Python 3 / CSV parser（结构和逐拍统计）
- PowerShell CSV/JSON audit（完整性、唯一性、provenance）
- MuJoCo 3.2.3（物理后端，由每回合 metrics 记录）

