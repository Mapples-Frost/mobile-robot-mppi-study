# Redesign E 四臂独立测试路径索引

## 测试包

- 工程根目录：
  `D:\Projects\mobile-robot-mppi-study-single-v6`
- 独立测试根目录：
  `D:\Projects\mobile-robot-mppi-study-single-v6\research_artifacts\redesignE_four_arm_exact_standalone_v1`
- 测试契约：
  `D:\Projects\mobile-robot-mppi-study-single-v6\research_artifacts\redesignE_four_arm_exact_standalone_v1\INDEPENDENT_TEST_CONTRACT.md`
- 顶层说明：
  `D:\Projects\mobile-robot-mppi-study-single-v6\research_artifacts\redesignE_four_arm_exact_standalone_v1\README_CN.md`

## 原始 evidence

- 原始 evidence 根目录：
  `D:\Projects\mobile-robot-mppi-study-single-v6\research_artifacts\redesignE_four_arm_exact_standalone_v1\raw\redesignE_four_arm_exact`
- Arms：上述目录下的 `B00`、`B01`、`B10`、`B11`
- episode 目录格式：
  `...\raw\redesignE_four_arm_exact\<ARM>\seed<DISPLAY_ALIAS>\`

每个 episode 含 `case_identity.json`、`config_resolved.yaml`、
`config_sha256.txt`、`metrics.json`、`preflight.json`、`provenance.json` 和
`trajectory.csv.gz`。

## 独立分析

- 中文报告：`analysis\REDESIGNE_ANALYSIS_CN.md`
- 独立分析 Gate：`analysis\INDEPENDENT_ANALYSIS_GATE.json`
- episode 索引：`analysis\REDESIGNE_EPISODE_INDEX.csv`
- 四臂汇总：`analysis\REDESIGNE_ARM_SUMMARY.csv`
- 二元配对：`analysis\REDESIGNE_PAIRED_BINARY.csv`
- 连续配对：`analysis\REDESIGNE_PAIRED_CONTINUOUS.csv`
- 分析入口：`analysis\analyze_redesignE_standalone.py`
- 统计核心：`analysis\redesignE_analysis_core.py`
- 复现命令：`analysis\RUN_COMMAND.txt`

## 完整性

- 复制验证：`audit\COPY_VERIFICATION.json`
- 原始 evidence SHA-256：`audit\SOURCE_RAW_DATA.sha256`
- 测试协议：`audit\TEST_PROTOCOL.json`
- 非原始文件 SHA-256：`audit\PACKAGE_NONRAW.sha256`
