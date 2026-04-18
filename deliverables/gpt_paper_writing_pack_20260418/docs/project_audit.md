# 项目审计结论

## 1. 当前主线最可能对应的文件

- 数据：
  - `data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv`
  - `data_external/vcor_selected/selected_manifest.csv`
- 提示：
  - `prompts/current/vcor_balanced_primary_prompts.csv`
  - `prompts/current/vcor_balanced_auxiliary_prompts.csv`
- 推理：
  - `scripts/inference/run_multimodel_batch.py`
  - `scripts/parsing/parse_restructured_car_color_outputs.py`
- 旧分析主线：
  - `analysis/current/vcor_balanced_primary/`
  - `analysis/current/vcor_balanced_auxiliary/`

这些文件说明仓库里已经存在“最终平衡评测集 + C0-C4 + A1/A2 + 三模型”的核心实验能力。

## 2. 与当前论文主线冲突或会造成误用的部分

- Stanford-only 作为并列主流程的配置、结果和 viewer
- `robustness`、`RPE`、`LDI` 这类旧包装
- `results_summary/current/` 里围绕旧叙事组织的总结文件
- `external_review_pack_current/` 之类对外整理包
- 旧版 `run_multimodel_stanford_cars_pipeline_v2.py`

这些内容会把注意力从“最终平衡评测集上的主实验”重新拉回旧对照线或旧理论包装。

## 3. 可删除、归档或下线的内容类型

- Stanford-only 主流程入口
- 决策边界 / 阈值 / 语言主导率包装输出
- 旧版综合总结脚本与对外 review 包
- 历史中间结果和旧 viewer

## 4. 原仓库缺失的关键能力

- 论文专用目录与统一入口脚本
- 单独可跑的 `C0` 基线脚本
- 以 Wilson CI 和 paired exact McNemar 为核心的比例分析
- 直接生成 Table 1 / Figure 2 / Table 3 / 数据集分布图
- 面向论文主线的复现实验文档

## 5. 本次重构后的保留原则

- 保留稳定的底层推理、解析和数据构建能力
- 新增论文专用配置、入口脚本、结果目录和文档
- 将 Stanford-only 降级为 appendix-only sanity check
- 不再把旧理论包装放在默认结果导出中
