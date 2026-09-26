# 原文与源码定位

这些链接固定到本次下载的具体提交，不随默认分支变化。代码证据与原实验配置证据应分开看；找到类与函数不代表原配置已经找回。

|模块|作者源码定位|本次用途|
|---|---|---|
|被引用SAC的默认超参数|[Haarnoja 2018补充材料，附录D表1](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf)|batch256、buffer10^6、critic256×2、Adam3e-4、tau0.005；Bøhn第4节列明自己的例外|
|SAC动作缩放、回报缩放、两套经验池|[sac.py, 470行起](https://github.com/eivindeb/stable-baselines/blob/1539282c8a11417b96e13040e5c4649be251dca2/stable_baselines/sac/sac.py#L470)|保留原作者训练循环；连续动作进入策略经验池，阶段成本进入终端值经验池|
|终端模型32步样本|[sac.py, 372行起](https://github.com/eivindeb/stable-baselines/blob/1539282c8a11417b96e13040e5c4649be251dca2/stable_baselines/sac/sac.py#L372)|原作者实现；采样边界修正见本地runtime.py|
|终端Bellman目标|[sac.py, 262行起](https://github.com/eivindeb/stable-baselines/blob/1539282c8a11417b96e13040e5c4649be251dca2/stable_baselines/sac/sac.py#L262)|阶段成本＋按实际步数折扣的自举值|
|AHMPCPolicy及poly模型|[policies.py, 301行起](https://github.com/eivindeb/stable-baselines/blob/1539282c8a11417b96e13040e5c4649be251dca2/stable_baselines/sac/policies.py#L301)|策略和终端函数；poly是线性项、逐维平方项、偏置|
|可变时域MPC|[controllers.py, 676行起](https://github.com/eivindeb/gym-letMPC/blob/3f0572e4761f6797327e48d77e2abc343ab0239c/gym_let_mpc/controllers.py#L676)|在固定最大时域内屏蔽超出H的阶段|
|车辆避障与感知预测误差|[controllers.py, 773行起](https://github.com/eivindeb/gym-letMPC/blob/3f0572e4761f6797327e48d77e2abc343ab0239c/gym_let_mpc/controllers.py#L773)|TTAHMPC的障碍生成、软硬约束、预测噪声|
|MPC目标折扣与终端折扣|[controller.py, 1090行起](https://github.com/eivindeb/do-mpc/blob/3f4fea1d262082083ac280b5206a7ca915a6676d/do_mpc/controller.py#L1090)|保留作者修改的do-mpc目标组装|
|缺失配置、外部数据目录线索|[let_mpc.py, 540行起](https://github.com/eivindeb/gym-letMPC/blob/3f0572e4761f6797327e48d77e2abc343ab0239c/gym_let_mpc/let_mpc.py#L540)|明确引用cart_pendulum_horizon.json、unicycle_ca_horizon.json及lmpc-horizon/datasets|

## 本地实现入口

- `configure.py`：按论文公式10/11构造两个任务；补设项见README。
- `runtime.py`：导入固定源码及四项兼容/语义修正，作者工作树保持原样。
- `run.py`：训练、保存、冻结评估和终端值移除消融。
- `suite.py`：26组主队列；`extra_jobs.py`仅提前运行其中指定的尾部任务，不增加实验组数。
- `validate.py`：独立数值核验；`check_replay.py`：模型重载及冻结场景回放核验。
- `report.py`：汇总成本、约束违反、H和场景轨迹；数据来自本次运行。
- `setup.sh`和`requirements-legacy.txt`：独立Python3.7、TF1运行环境。
- `retrieve.ps1`、`audit_history.py`、`provenance.py`：来源检索、失败记录与文件散列。

原论文见 [DOI](https://doi.org/10.1016/j.ifacol.2021.08.563) 和 [作者公开PDF](https://torarnj.folk.ntnu.no/lahmpc_nmpc2021_eeb.pdf)。PDF第4–5页包含学习及实验协议；第5–6页包含主要结果。报告没有从低分辨率图中猜测原始数据。

进一步定位与运行入口：`DIAGNOSIS.md`、`paper_suite.py`（引用链校正设置）、`diagnosis_report.py --group paper_defaults`（完整报告）。batch64＋自动熵系数的旧组作为诊断保留。
