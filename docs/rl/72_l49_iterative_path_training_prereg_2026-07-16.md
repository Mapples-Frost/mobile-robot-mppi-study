# L49 task-specific iterative ICODE training 预注册

日期：2026-07-16  
状态：数据构建与训练前冻结。

## 数据来源

只使用 L48 `block_0 / traditional_nominal` 的真实 MuJoCo 轨迹，不读取 ICODE 轨迹作为监督，不按控制结果筛选 episode。
每条 transition 使用已记录的 interval-average applied control，重算 nominal derivative 与 residual target。

episode-level split：

- train：21360731–21360736；
- validation：21360737；
- test：21360738；
- unseen：21360739–21360740。

同一 episode 不跨 split；统计量只由 train 计算。三个路径与两个延迟域在各 split 中均保留。

## 训练

继续使用 structure mask `[0,0,0,1,1]`、control-affine ICODE、RK4、H36 与 L46 loss。因数据量扩大，batch size 预先设为 512，
maximum epochs 100、early stopping 15。training seeds 为 20261101、20261102、20261103。

三个 checkpoint 必须先通过 test/unseen H36 离线准入，之后才能进入新闭环实验。L48 数据只用于训练/开发；后续闭环必须使用全新 seed。

