# metadata 目录说明

## 1. 目录结构说明
当前 `data/metadata/` 已重构为以下结构：

```text
data/
├── metadata/
│   ├── samples/
│   ├── prompts/
│   ├── outputs_raw/
│   ├── outputs_labeled/
│   ├── analysis/
│   └── README.md
```

## 2. 各子目录用途
- `samples/`：样本池、筛选候选集、正式样本子集、stress 子集等基础样本表。
- `prompts/`：实验输入表，仅保存 prompt 扩展后的待推理记录，不直接保存模型输出。
- `outputs_raw/`：模型原始输出及推理阶段的中间续跑文件。
- `outputs_labeled/`：自动预标注、人工复核清单、最终人工确认版标签表。
- `analysis/`：后续统计分析文件目录，建议保存 `summary`、`by_level`、`decision_boundary` 等结果。

## 3. 当前已有文件清单
- `samples/no_dog_all.csv`
- `samples/no_dog_filtered_candidates.csv`
- `samples/no_dog_sample_50.csv`
- `samples/no_dog_stress_subset_10.csv`
- `prompts/baseline_existence_conflict_50x4.csv`
- `prompts/stress_existence_conflict_10x4.csv`
- `outputs_raw/qwen2vl7b_baseline_runtime.csv`
- `outputs_raw/qwen2vl7b_baseline_raw.csv`
- `outputs_raw/qwen2vl7b_smoke_raw.csv`：当前文件已存在，包含 8 条 smoke 中间结果，尚未完成完整 20 条冒烟集。
- `outputs_labeled/qwen2vl7b_baseline_prelabeled.csv`
- `outputs_labeled/qwen2vl7b_baseline_manual_review.csv`
- `outputs_labeled/qwen2vl7b_baseline_final_labeled.csv`

## 4. baseline 与 stress 文件如何区分
- baseline 文件统一包含 `baseline`，例如：`baseline_existence_conflict_50x4.csv`、`qwen2vl7b_baseline_raw.csv`。
- stress 文件统一包含 `stress`，例如：`stress_existence_conflict_10x4.csv`、后续的 `qwen2vl7b_stress_raw.csv`。
- 样本级文件若是增强版子集，则命名为 `no_dog_stress_subset_10.csv`。

## 5. raw / prelabeled / final / analysis 的区别
- `raw`：模型原始输出结果，不做人工判断，是最接近推理原貌的记录。
- `prelabeled`：基于规则脚本的机器初判版本，适合做第一轮快速筛查。
- `manual_review`：从 prelabel 中筛出的重点人工复核清单。
- `final_labeled`：人工最终确认版，后续正式统计分析应优先使用这一版。
- `analysis`：由 `final_labeled` 或其他结果表进一步计算得到的统计与分析文件。
- `runtime`：批量推理脚本用于断点续跑的技术中间文件，通常不直接作为论文主表引用，但建议保留。

## 6. 后续新增第二模型时建议命名规则
建议统一采用：

```text
<modeltag>_<scenario>_<stage>.csv
```

其中：
- `<modeltag>`：模型短名，全部小写，尽量不用斜杠和空格，例如 `qwen2vl7b`、`internvl28b`、`minicpmv26`。
- `<scenario>`：实验场景，例如 `baseline`、`stress`。
- `<stage>`：处理阶段，例如 `runtime`、`raw`、`prelabeled`、`manual_review`、`final_labeled`。

示例：
- `internvl28b_baseline_raw.csv`
- `internvl28b_baseline_final_labeled.csv`
- `minicpmv26_stress_raw.csv`
- `minicpmv26_stress_final_labeled.csv`

## 7. 兼容性说明
原先位于 `data/metadata/` 根目录下的 CSV 已迁移到新子目录结构。项目中的主要脚本已经更新为“新路径优先、旧路径 fallback”的读取方式，因此后续建议统一使用新路径，不再继续在根目录堆放新的实验 CSV。

