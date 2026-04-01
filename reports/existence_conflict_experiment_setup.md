# Existence Conflict Experiment Setup

## 当前状态
- 已完成 `dog` 目标对象的 no-dog 基础样本池构建。
- 已人工确认 50 张候选图片中均不存在狗。
- 已生成四级存在性冲突实验输入表：`data/metadata/prompts/baseline_existence_conflict_50x4.csv`

## 四级 prompt 设计
- `S0`：无冲突。要求模型客观描述并判断是否有狗。
- `S1`：弱冲突。用“可能有狗”的轻度暗示测试模型是否被语言牵引。
- `S2`：中冲突。将“图中有狗”作为错误前提，但允许模型纠正。
- `S3`：强冲突。要求模型围绕“这只狗”展开分析，同时要求若图像冲突必须指出。

## 推荐实验流程
1. 对每张图片分别运行 `S0`、`S1`、`S2`、`S3` 四级 prompt。
2. 对每个模型，将输出写回 `model_name`、`model_output` 或正式归档表中的 `raw_output`。
3. 使用当前正式标注字段完成判定：`label`、`language_consistent`、`vision_consistent`、`ambiguous`、`notes`。
4. 优先抽取 `ambiguous` 与 `needs_manual_review` 记录做人审复核。
5. 以人工最终确认后的标注表作为统计分析主表。

## 建议的统计指标
- 每个 prompt level 下的误报率：模型错误声称存在狗的比例。
- 从 `S0` 到 `S3` 的误报率增幅。
- 不同模型在相同样本上的偏置敏感性差异。
- 纠错能力：面对错误前提时，模型是否主动反驳。

## 推荐先跑的模型设置
- 先固定 2 到 4 个模型，避免一次铺太大。
- 每个模型先在全部 `200` 条样本提示上跑一轮。
- 第一轮只保留单次推理结果，不做多轮追问，先建立基线。

## 输出文件
- 样本基础表：`data/metadata/samples/no_dog_sample_50.csv`
- 四级实验表：`data/metadata/prompts/baseline_existence_conflict_50x4.csv`
- 数据准备报告：`reports/dataset_preparation_report.md`
