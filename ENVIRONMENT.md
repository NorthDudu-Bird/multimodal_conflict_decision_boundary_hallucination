# Environment Guide

## 推荐环境

- OS：Windows 11
- Python：`3.11`
- PyTorch：CUDA 版本
- 最近完成全量重跑的显卡：RTX 4080 Laptop GPU

## 安装

```powershell
conda create -n cv_proj python=3.11 -y
conda activate cv_proj
pip install -r requirements.txt
```

## 本地模型目录

请将三套模型权重放在以下目录：

- `models/qwen2_vl_7b`
- `models/llava_1_5_7b_hf`
- `models/internvl2_8b`

## 当前唯一有效的配置入口

- `configs/paper_mainline.yaml`

所有论文主线脚本默认都从这份配置读取路径和模型设置。旧的 `current` 配置树已不再属于 GitHub 默认工作流。

## 当前唯一有效的执行入口

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/build_dataset.py
python scripts/run_baseline_c0.py --skip-build
python scripts/run_main_c0_c4.py --skip-build
python scripts/run_aux_a1_a2.py --skip-build
python scripts/run_robustness_c3_prompt_variants.py --skip-build
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

## 说明

- 当前仓库不再提供旧的 VCoR-balanced 预览流水线。
- Stanford-only 不再作为并列主实验，只保留为 appendix source sanity check 的一部分。
- `logs/` 中的内容仅用于运行日志、候选预览和复现快照，不属于 Git 跟踪的论文资产。
