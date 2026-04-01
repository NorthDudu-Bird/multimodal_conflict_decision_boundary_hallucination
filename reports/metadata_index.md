# Metadata Index

本页用于集中索引当前项目中与 metadata 相关的 CSV 文件，便于论文写作、结果复核与后续分析调用。

## 1. samples
- `data/metadata/samples/no_dog_all.csv`：COCO val2017 中筛出的 no-dog 完整清单。
- `data/metadata/samples/no_dog_filtered_candidates.csv`：在 no-dog 清单上进一步按尺寸与标注复杂度过滤后的候选集。
- `data/metadata/samples/no_dog_sample_50.csv`：最终人工确认的 50 张正式 no-dog 样本。
- `data/metadata/samples/no_dog_stress_subset_10.csv`：按随机种子 42 从 50 张正式样本中固定抽出的 10 张 stress 子集。

## 2. prompts
- `data/metadata/prompts/baseline_existence_conflict_50x4.csv`：baseline 四级存在性冲突 prompt 表，共 50 x 4 = 200 条。
- `data/metadata/prompts/stress_existence_conflict_10x4.csv`：增强版 stress conflict prompt 表，共 10 x 4 = 40 条。

## 3. outputs_raw
- `data/metadata/outputs_raw/qwen2vl7b_baseline_runtime.csv`：Qwen2-VL-7B baseline 批量推理的续跑中间文件。
- `data/metadata/outputs_raw/qwen2vl7b_baseline_raw.csv`：Qwen2-VL-7B baseline 正式 raw 结果表。
- `data/metadata/outputs_raw/qwen2vl7b_smoke_raw.csv`：Qwen2-VL-7B 冒烟测试 raw 结果文件，当前包含一次中途停止后保留的 8 条 smoke 结果，尚未形成完整 20 条正式冒烟结果。
- `data/metadata/outputs_raw/qwen2vl7b_stress_raw.csv`：Qwen2-VL-7B stress 实验 raw 结果的标准命名槽位，当前尚未生成实际文件。

## 4. outputs_labeled
- `data/metadata/outputs_labeled/qwen2vl7b_baseline_prelabeled.csv`：baseline 结果的规则型自动预标注版本。
- `data/metadata/outputs_labeled/qwen2vl7b_baseline_manual_review.csv`：baseline 结果的重点人工复核清单。
- `data/metadata/outputs_labeled/qwen2vl7b_baseline_final_labeled.csv`：baseline 结果的最终人工确认模板和主标注表。

## 5. analysis
- `data/metadata/analysis/`：后续建议存放统计汇总、按 level 分析、decision boundary 分析结果。目前目录已建立，尚无分析 CSV。

## 6. baseline 与 stress 的区分原则
- baseline：文件名包含 `baseline`，对应主实验四级存在性冲突设置。
- stress：文件名包含 `stress`，对应增强版高压文本前提测试设置。
- 若同一模型同时有两套结果，应保持 `<modeltag>_baseline_*` 与 `<modeltag>_stress_*` 成对出现，避免混淆。

## 7. 推荐使用顺序
1. 从 `samples/` 读取样本与子集。
2. 从 `prompts/` 读取待推理输入表。
3. 将模型输出写入 `outputs_raw/`。
4. 将自动预标注与人工标注结果写入 `outputs_labeled/`。
5. 将统计汇总与分析结果写入 `analysis/`。

