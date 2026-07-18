# L69 静态复杂几何 RL 因果贡献消融预注册

日期：2026-07-16  
性质：development eligibility；结果产生前冻结。

## 研究问题

既往 L25/L26 表明带复杂度门控的学习 prior 在低采样预算下远好于传统 MPPI，
但没有加入“使用相同复杂度门控、只保留冻结 BC 教师动作”的对照。因此，既往收益
可能来自教师 BC 或外层门控，而不一定来自 SAC 强化学习修正。

L69 只回答一个可证伪问题：

> 在相同 ICODE、相同 MuJoCo plant、相同 LaserScan 门控、相同 MPPI `K=100`
> 和同一个冻结 BC base actor 下，SAC correction 是否带来额外闭环收益？

## 固定因素与四个条件

固定因素包括：L56 高动态 plant、ground-truth pose/twist、三个已通过 L57--L68 的
ICODE checkpoint、静态障碍几何、scan guard、安全仲裁、memory off 和全部 MPPI cost。

四个条件为：

1. `traditional_icode`：GoalWarmStart + ICODE；
2. `frozen_bc_icode`：始终使用 checkpoint 内不可训练的 BC base actor；
3. `complexity_bc_icode`：LaserScan complexity gate 在 GoalWarmStart 与 BC base 间混合；
4. `gated_lcb_icode`：相同 outer gate，但允许 target-twin LCB 接受 SAC correction。

主要比较是 4 对 3。两者使用同一个 correction checkpoint；新加入的 `base` 模式将
latent action 精确替换为 checkpoint 内冻结 base actor，不用人为设置巨大阈值，也不
允许 critic 决定 BC 对照。

## 设计与独立重复

实验矩阵为：

```text
3 independent RL/ICODE model blocks
× 4 scenes
× 1 fixed physics domain
× 5 fresh development episode seeds
× 4 conditions
= 240 episodes
```

场景包括 clean control、single obstacle、narrow corridor 和 U-trap。U-trap 属于 BC/RL
训练几何；另两个阻塞场景属于跨几何测试。训练模型 block 是方法层独立重复，episode
seed 是 block 内重复测量，控制 step 不是独立样本。

## 事前 Gate

只有同时满足以下条件才允许建立 L70 sealed confirmation：

- 完整唯一矩阵，无 protected/sealed seed 泄漏；
- clean 场景中两个 complexity 条件逐 step 精确退化为 traditional ICODE；
- `complexity_bc_icode` 的 correction gate alpha 恒为 0；
- 阻塞场景中 SAC correction 接受比例至少 5%；
- SAC 相对 gated BC 净增加至少 3 次成功，至少 2/3 model blocks 和 2/3 scenes 为正；
- SAC 相对 gated BC 不增加碰撞；
- SAC 相对 traditional ICODE 净增加至少 12 次成功且不增加碰撞；
- combined learned planner 平均计算时间不超过 50 ms。

本轮不以事后 final-distance 改善替换失败的 success Gate，不在 development seeds 上调整
复杂度阈值、LCB beta、K、reward、checkpoint 或场景标签。若 Gate 失败，sealed seeds
保持关闭，并依据预先记录的对照定位是 BC、outer gate 还是 SAC correction 失效。

## 解释边界

通过只能说明当前静态几何、固定 MuJoCo plant 和有界动作空间中的 RL correction 贡献；
不能外推到动态障碍、任意 OOD、实车或稳定性保证。失败也只否定当前 checkpoint/训练
协议，不等于否定 RL sampling prior 的一般可行性。

