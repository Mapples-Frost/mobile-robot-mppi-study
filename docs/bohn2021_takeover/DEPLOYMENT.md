# Bøhn 无人值守科研服务：接管与部署记录

记录日期：2026-09-26。**研究继续进行，尚未达到复现成功验收。**

服务器：ubuntu@18.236.70.13，工作目录 `/data/openai-agent/mobile-robot-mppi-study`。`bohn-research.service` 已 enabled + active，主进程归 systemd（PPID1）管理。已实测 SIGKILL 后自动重启，以及 SSH 断开后重新连接仍有新心跳。重启恢复由 enabled 服务和独立路径挂载单元配置；本次没有重启整台 EC2。

API：仅 `gpt-5.5`，接口实际接受 `reasoning.effort=xhigh`，文本及函数调用 smoke 都通过。服务不设置其他模型回退。凭据位于仓库外 `.secrets`，文件模式600，未写入本报告。

## 审计发现

- 指定D盘版本是早期原型；另一个WSL工作树才有完整作者复现资产。两者均保留，没有重置既有未提交改动。
- 已迁移33,481,974,323字节历史Bøhn资料，共64,871个文件清单。递归核验12,213个文件哈希，其中含726个冻结输入，未发现不一致；这不等于对全部历史实验重新运行或重新统计。
- 最新浅树训练4个作业完成（车辆0/1/2、倒立摆0）；倒立摆1中断、2未启动。旧active标记对应的进程已经不存在。
- 服务器重新核验车辆seed1训练内复选10条件、160回合、12,640步原始记录：4个树候选中3个全程H25，最终选择fixed H25。不能据此宣称学到了有效自适应策略；该策略不满足原定逐seed回合内自适应门槛。未打开本轮validation或sealed test结果。
- 旧Python3.7.16 / TensorFlow1.15.5 / CasADi3.5.5 / NumPy1.18.5已迁移并实际运行；vehicle及pendulum各一步工程MPC smoke通过。这仅验证可运行性，不是复现成功或正式速度证据。

## 长期执行

控制器使用SQLite、JSONL、逐实验registry和stdout/stderr。一次仅执行一个实验，每轮至多6次模型调用并强制保存后续状态，单实验最多4小时；3次相同失败阻止原样重试，累计失败进入诊断。48次API/UTC日、报告token阈值150万/日；预算耗尽自动等待，不换模型。内存上限3100MiB、CPU180%、数值库单线程、日志上限和磁盘余量保护均已配置。

旧科学源码和绝对路径通过bind mount兼容；不改写已冻结源码。新的诊断/改进另立脚本和协议。跨WSL/AWS的实测时间不能直接混合参与同一次排名。当前下一步是训练候选退化与fit/select差异诊断，随后按证据选择有针对性的实验。

## 备份与恢复

代码分支：[codex/bohn-aws-20260926](https://github.com/Mapples-Frost/mobile-robot-mppi-study/tree/codex/bohn-aws-20260926)。原始成果分块归档：[GitHub evidence release](https://github.com/Mapples-Frost/mobile-robot-mppi-study/releases/tag/bohn-aws-evidence-20260926)。

首个恢复探针已经实际下载、验SHA256、解包验证。多个正式原始资料分块也已上传并验证GitHub端SHA256；**全部历史资料的分块备份仍在补齐，不应写成全部完成。** 未上传部分仍保留于原WSL副本。新资料优先按修改时间归档，每轮最多16个包，避免备份长期挤占研究；SQLite索引支持重启后增量续传。原运行环境另有539MB左右的完整tar包，包含环境内链接，等待同一备份链上传。

恢复工具：`scripts/research_service/restore.py`。下载release所有清单和对应tar.gz后，验证SHA256并按时间顺序解包至新的工作根；环境包位于restored research_artifacts/deployment，需再解包到runtime，重建历史路径挂载与secret后重新核验源码/模型。受保护secret不进入公开备份。

## 只读检查命令

```sh
sudo systemctl status bohn-research
cat /data/openai-agent/state/heartbeat.json
cat /data/openai-agent/state/research_state.json
cat /data/openai-agent/state/backup_status.json
journalctl -u bohn-research --since '1 hour ago'
```

核心证据：同目录 `server_evidence_audit.json`、`server_smoke.json`、`vehicle_seed1_raw_diagnosis.json`、`survival_test.json`、`backup_restore_probe.json`。实时研究状态以服务器文件为准，此报告是部署时快照。未更改AWS释放任务、IAM、实例规格或/data挂载布局。
