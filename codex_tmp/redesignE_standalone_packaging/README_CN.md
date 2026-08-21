# Redesign E 四臂独立测试数据包

本目录是自包含的 Redesign E 四臂 matched-case 测试包。测试编号为
`redesignE_independent_test_v1`，包含 B00、B01、B10、B11 四个 arms，
每臂 50 个相同物理场景，共 200 个 episodes。分析不依赖其他实验的目录、
汇总表或统计结果。

## 测试范围

- 地图：`chapter3_redesignE`
- Arms：B00、B01、B10、B11
- Case：50 个 matched cases
- 显示编号：795100001–795100050
- 样本量：每臂 50，共 200 episodes
- 主要 endpoint：`metrics.success AND NOT metrics.collision`

目录中的显示编号用于连续索引；每个 episode 的原始 RNG seed 保存在
`case_identity.json`。

## 目录

- `INDEPENDENT_TEST_CONTRACT.md`：独立测试契约。
- `raw/redesignE_four_arm_exact/`：四臂原始 evidence。
- `audit/COPY_VERIFICATION.json`：逐文件 SHA-256 完整性 Gate。
- `analysis/REDESIGNE_ANALYSIS_CN.md`：中文独立分析报告。
- `analysis/INDEPENDENT_ANALYSIS_GATE.json`：独立分析 Gate。
- `analysis/analyze_redesignE_standalone.py`：自包含分析入口。
- `analysis/RUN_COMMAND.txt`：准确复现命令。

## 完整性状态

- 文件：1,408/1,408
- 字节：1,934,788,074
- 缺失文件：0
- SHA-256 不一致：0
- 额外文件：0
- 四臂：B00/B01/B10/B11 各 50 cases

原始 evidence 只读保留；统计输出均写入独立的 `analysis/` 目录。
