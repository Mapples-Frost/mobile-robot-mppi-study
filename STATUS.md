# 接管状态（2026-09-26）

当前结论：尚不能宣称复现成功。部署及证据迁移进行中，服务存活以服务器 state/heartbeat.json、systemctl 和实际 PID 为准。

已直接核查：D 盘目标分支 codex/change-aware-probabilistic-mppi，HEAD a9aea0c；有两处既有受跟踪修改和大量未跟踪历史文件。D 盘 Bøhn 原型只有 vehicle/cartpole 环境与 cartpole MPC，无可见正式训练结果。原始 git status/diff/branch/log/remote 和25份相关文件清单保存在 docs/bohn2021_takeover。

WSL 工作区 HEAD cad9f76，独立脏工作树。具有作者固定来源、原方法/改进方法、多 seed 模型、固定H网格以及约32GB原始证据。旧报告结论只作线索，迁移后以哈希复核结果为准。

最新 latency_tree_2026-09-26：文件显示 vehicle seeds 0/1/2、pendulum seed0 四个完成标记；pendulum seed1 仅 started/threshold_reference，seed2 未启动。旧训练 PID1694525不存在，Windows 接续 PID26188、7108不存在；状态 active:true 已失真。保留全部状态与失败/中断证据。

下一步：完整迁移及哈希审计；原 Python3.7/TF1 环境移植；两任务一步工程 smoke；登记跨主机恢复规则；恢复有信息价值的下一项工作。由于浅树选择使用实测时间，禁止将 WSL/AWS 计时当同质样本直接混用。

API smoke 已通过 gpt-5.5/xhigh，未切换模型。密钥在仓库外受限文件，本文不含凭据。GitHub 现有远程可写，分块外部恢复备份由独立程序校验 SHA256。
