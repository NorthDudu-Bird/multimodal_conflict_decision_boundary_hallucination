# 复现实验说明

## 1. 环境

- Python 依赖见 `requirements.txt`
- 论文主线配置文件：`configs/paper_mainline.yaml`
- 本地模型目录默认使用：
  - `models/qwen2_vl_7b`
  - `models/llava_1_5_7b_hf`
  - `models/internvl2_8b`

## 2. 数据准备

构建最终平衡评测集：

```bash
python scripts/build_dataset.py
```

构建后请使用：

- `data/balanced_eval_set/final_manifest.csv`
- `data/metadata/balanced_eval_set/balanced_eval_set_summary.json`
- `data/metadata/balanced_eval_set/dataset_distribution.csv`

## 3. 运行基线与主实验

运行 `C0` 基线：

```bash
python scripts/run_baseline_c0.py
```

运行 `C0-C4` 主实验：

```bash
python scripts/run_main_c0_c4.py
```

运行 `A1/A2` 辅助实验：

```bash
python scripts/run_aux_a1_a2.py
```

如需只跑部分模型：

```bash
python scripts/run_main_c0_c4.py --models qwen2vl7b llava15_7b
```

## 4. 生成论文表格与图

```bash
python scripts/make_figures.py
```

主要输出位置：

- `results/baseline/`
- `results/main/`
- `results/auxiliary/`
- `results/appendix/`

## 5. 关键结果文件

- Table 1：`results/main/table1_main_metrics.csv`
- Figure 2：`results/main/figure2_conflict_aligned_rates.png`
- Table 3：`results/auxiliary/table3_aux_metrics.csv`
- 数据集分布图：`results/appendix/dataset_distribution.png`
- 主实验统计检验：`results/main/main_exact_tests.csv`
- 辅助实验统计检验：`results/auxiliary/aux_exact_tests.csv`
- 代表性案例：
  - `results/main/main_representative_cases.csv`
  - `results/auxiliary/aux_representative_cases.csv`

## 6. 已废弃或不应再默认使用的内容

以下内容不再属于当前论文默认主流程：

- Stanford-only 控制主流程脚本
- `robustness` 命名的旧总结文件
- `LDI / RPE / decision boundary / threshold` 叙事
- `results_summary/current/` 一类旧式汇总目录
- `external_review_pack_current/` 之类的对外打包目录

这些内容如果仍保留，只作为归档或 appendix 参考，不再作为正文入口。
