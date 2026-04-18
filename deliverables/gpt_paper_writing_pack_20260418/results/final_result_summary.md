# 当前结果摘要

## 数据集

- 正式评测集：`data/balanced_eval_set/final_manifest.csv`
- 总样本数：300
- 颜色分布：`red / blue / green / yellow / black / white` 各 50
- 来源构成：StanfordCars 93，VCoR 207

## C0 基线

- `Qwen2-VL-7B-Instruct`：忠实率 100.00%，冲突一致率 0.00%
- `LLaVA-1.5-7B`：忠实率 100.00%，冲突一致率 0.00%
- `InternVL2-8B`：忠实率 100.00%，冲突一致率 0.00%

## C0-C4 主实验

- `Qwen2-VL-7B-Instruct`：`C0-C2` 冲突一致率均为 0.00%，`C3` 为 0.33%，`C4` 为 0.33%
- `LLaVA-1.5-7B`：`C1` 为 0.33%，`C2` 为 1.00%，`C3` 为 9.00%，`C4` 为 3.33%
- `InternVL2-8B`：`C0-C4` 冲突一致率均为 0.00%

主结论稳定：在清晰视觉证据下，三模型整体表现出显著视觉优先性；有限的语言主导偏差主要集中在 `LLaVA-1.5-7B` 的强误导提示条件，尤其是 `C3`。

## A1/A2 辅助实验

- `Qwen2-VL-7B-Instruct`：`A1` 服从率 55.67%，`A2` 服从率 90.67%
- `LLaVA-1.5-7B`：`A1` 服从率 85.33%，`A2` 服从率 100.00%
- `InternVL2-8B`：`A1` 服从率 73.67%，`A2` 服从率 100.00%

辅助实验说明，在受限答案空间下，模型对错误前提的服从性会明显提高，因此 A1/A2 适合作为补充证据，而不是正文主实验替代。

## 关键文件

- Table 1：`results/main/table1_main_metrics.csv`
- Figure 2：`results/main/figure2_conflict_aligned_rates.png`
- Table 3：`results/auxiliary/table3_aux_metrics.csv`
- 数据集分布图：`results/appendix/dataset_distribution.png`
- 主实验统计检验：`results/main/main_exact_tests.csv`
- 辅助实验统计检验：`results/auxiliary/aux_exact_tests.csv`

## 当前重跑状态

- 已完成：`python scripts/build_dataset.py`
- 已完成：`python scripts/run_baseline_c0.py`
- 已完成：`python scripts/run_main_c0_c4.py`
- 已完成：`python scripts/run_aux_a1_a2.py`
- 已完成：`python scripts/analyze_results.py`
- 已完成：`python scripts/make_figures.py`

本次重构后的新目录下，`baseline / main / auxiliary / appendix` 四类论文输出都已重新生成并核对完成。
