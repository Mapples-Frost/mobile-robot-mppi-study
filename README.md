# Mobile Robot MPPI Study

Learning and implementing MPPI for mobile robot navigation and obstacle avoidance, with notes, code, and experiment logs.

---

## 项目简介

这个仓库用于记录我在“移动机器人导航与避障”方向上的前期学习与实现过程。当前阶段先不急着上实体机器人、ROS 或正式仿真平台，而是先在 **WSL + PyCharm + Python** 环境下，搭建一个**最小二维算法实验台**，把环境、运动学、碰撞检测、轨迹生成、控制采样和代价评估这些基础问题看清楚、跑通、能解释、能逐步扩展。

项目的核心主线是：

- 先完成二维环境与机器人运动学基础
- 再逐步实现 vanilla MPPI（Model Predictive Path Integral，模型预测路径积分控制）原型
- 后续再整理实验、阅读论文、尝试第一版改进方向

---

## 当前阶段

### 当前正在做
- [x] 搭建二维环境与障碍物表示
- [x] 实现机器人状态表示与运动学更新
- [x] 实现轨迹 rollout 与碰撞检测
- [x] 实现多候选控制比较
- [x] 从单个控制采样推进到控制序列采样
- [x] 理解并实现 weighted update（加权更新）原型
- [x] 串起 receding horizon（滚动时域）主循环
- [x] 跑通 vanilla MPPI baseline
- [ ] 开始系统整理实验结果与论文笔记

### 当前暂时不做
- [ ] 实体小车联调
- [ ] ROS 节点系统开发
- [ ] Gazebo / Webots / MATLAB 平台迁移
- [ ] RL-guided MPPI / Residual-MPPI 等改进版实现

---

## 当前已经完成的内容

### 1. 二维环境与可视化
- `src/envs/simple_env.py`
- `src/envs/trajectory_demo.py`

已经可以画出：
- 起点
- 终点
- 圆形障碍物
- 轨迹与机器人朝向

### 2. 机器人运动学
- `src/models/unicycle_model.py`

已经实现：
- `step(state, control, dt)` 单步状态更新
- `rollout(...)` 多步轨迹展开

当前默认状态与控制表示为：
- `state = (x, y, theta)`
- `control = (v, omega)`

### 3. 碰撞检测
- `src/envs/collision_demo.py`

已经实现：
- 单点碰撞检测
- 整条轨迹碰撞检测
- 基于圆形机器人与圆形障碍物的最简碰撞判断

### 4. 从控制选择到采样控制
- 多候选控制比较
- 围绕 `nominal_control` 的高斯采样
- 从“单个控制采样”推进到“控制序列采样”
- rollout → trajectory → cost 的整条链路理解与实现

### 5. MPPI 原型理解推进
目前已经进入 vanilla MPPI 的核心骨架前半段：

- 控制序列初始化
- 噪声采样
- rollout
- cost evaluation
- weighted update

下一步重点是把这些模块真正串进 receding horizon 主循环。

---

## 项目结构

```text
mobile-robot-mppi-study/
├── .venv/
├── docs/
│   ├── paper_notes/         # 论文笔记
│   ├── roadmap.md           # 项目路线图
│   └── weekly_logs/         # 学习记录 / 周志
├── experiments/
│   ├── configs/             # 实验配置
│   ├── logs/                # 实验日志
│   └── scripts/             # 批量实验脚本
├── reports/
│   ├── milestone_reports/   # 阶段总结
│   └── weekly_reports/      # 周报
├── results/
│   ├── figures/             # 图片结果
│   ├── gifs/                # 动图结果
│   └── tables/              # 表格结果
├── src/
│   ├── costs/               # 代价函数
│   ├── envs/                # 环境相关代码
│   ├── models/              # 运动学 / 动力学模型
│   ├── planners/            # 规划与控制器原型
│   └── utils/               # 工具函数
├── .gitignore
└── README.md
