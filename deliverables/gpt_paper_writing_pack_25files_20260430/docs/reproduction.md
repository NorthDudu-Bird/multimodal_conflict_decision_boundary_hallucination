# 复现实验说明

本文档对应当前唯一官方论文流程：平衡评测集、`C0`、`C0-C4`、`A1/A2`、`C3` prompt 变体控制、parser audit、附录来源 sanity check，以及最终复现一致性校验。

## 1. 环境

- 配置文件：`configs/paper_mainline.yaml`
- Python 依赖：`requirements.txt`
- 模型目录：
  - `models/qwen2_vl_7b`
  - `models/llava_1_5_7b_hf`
  - `models/internvl2_8b`

## 2. 先创建锁定结果快照

最终提交前，先把当前锁定结果复制到忽略目录 `logs/reproducibility_snapshot/latest/`，供重跑后逐项比对。

```powershell
$repo = Resolve-Path .
$snapshot = Join-Path $repo "logs\\reproducibility_snapshot\\latest"
if (Test-Path $snapshot) { Remove-Item -LiteralPath $snapshot -Recurse -Force }

$targets = @(
  "data\\balanced_eval_set",
  "data\\metadata\\balanced_eval_set",
  "prompts\\c0_c4",
  "prompts\\a1_a2",
  "prompts\\robustness",
  "results\\baseline",
  "results\\main",
  "results\\auxiliary",
  "results\\robustness",
  "results\\parser",
  "results\\appendix",
  "results\\final_result_summary.md"
)

foreach ($rel in $targets) {
  $src = Join-Path $repo $rel
  if (Test-Path $src) {
    $dst = Join-Path $snapshot $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
  }
}
```

## 3. 清空待重建结果

只清空当前生成结果，不清理数据源和模型权重：

```powershell
$repo = Resolve-Path .
$generated = @(
  "results\\baseline",
  "results\\main",
  "results\\auxiliary",
  "results\\robustness",
  "results\\parser",
  "results\\appendix"
)

foreach ($rel in $generated) {
  $path = Join-Path $repo $rel
  if (Test-Path $path) { Remove-Item -LiteralPath $path -Recurse -Force }
}

$variantCsv = Join-Path $repo "prompts\\robustness\\c3_prompt_variants.csv"
if (Test-Path $variantCsv) { Remove-Item -LiteralPath $variantCsv -Force }
```

## 4. 全量重跑顺序

严格按以下顺序执行，不加 `--limit`，三模型全跑：

```bash
python scripts/build_dataset.py
python scripts/run_baseline_c0.py --skip-build
python scripts/run_main_c0_c4.py --skip-build
python scripts/run_aux_a1_a2.py --skip-build
python scripts/run_robustness_c3_prompt_variants.py --skip-build
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

## 5. 关键输出

- 数据集：
  - `data/balanced_eval_set/final_manifest.csv`
  - `data/metadata/balanced_eval_set/balanced_eval_set_summary.json`
- 主实验：
  - `results/main/table1_main_metrics.csv`
  - `results/main/main_key_tests.csv`
  - `results/main/main_stats_summary.md`
  - `results/main/figure2_conflict_aligned_rates.png`
- 辅助实验：
  - `results/auxiliary/table3_aux_metrics.csv`
- 鲁棒性：
  - `results/robustness/prompt_variant_metrics.csv`
  - `results/robustness/prompt_variant_exact_tests.csv`
  - `results/robustness/prompt_variant_summary.md`
- 解析审查：
  - `results/parser/label_mapping_audit.md`
  - `results/parser/ambiguous_outputs_sample.csv`
- 附录 sanity check：
  - `results/appendix/stanford_core_sanity_check.csv`
  - `results/appendix/stanford_core_sanity_check.md`
- 总摘要：
  - `results/final_result_summary.md`
- 复现比对：
  - `results/reproducibility_comparison.csv`
  - `results/reproducibility_audit.md`

## 6. 论文级派生补强

以下脚本只读取已有 canonical parsed/metrics outputs，不调用模型推理，也不改变主结果口径：

```bash
python scripts/generate_paired_flip_analysis.py
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_visual_clarity_audit.py
python scripts/build_writing_pack.py --pack-date 20260430
```

对应新增输出：

- same-image paired analysis：
  - `results/main/paired_transition_counts.csv`
  - `results/main/paired_flip_metrics.csv`
  - `results/main/paired_flip_summary.md`
- prompt wording boundary control：
  - `results/robustness/prompt_boundary_metrics.csv`
  - `results/robustness/prompt_boundary_summary.md`
- visual clarity audit infrastructure：
  - `results/audit/visual_clarity_audit_manifest.csv`
  - `results/audit/visual_clarity_audit_readme.md`
- 写作包：
  - `deliverables/gpt_paper_writing_pack_25files_20260430.zip`

## 7. 验收标准

`python scripts/verify_reproducibility.py` 必须通过，且至少满足：

- `C0` 三模型仍为完美视觉忠实
- `results/main/main_key_tests.csv` 仍为 12 行
- `results/robustness/prompt_variant_metrics.csv` 仍为 9 行
- `results/parser/ambiguous_outputs_sample.csv` 仍为 27 行
- `results/appendix/stanford_core_sanity_check.csv` 仍为 12 行
- 最终口径仍为：`LLaVA-1.5-7B` 的现象是有限且模板敏感的语言偏差，而不是稳定的跨 wording 规律
- 派生补强应满足：LLaVA `C3` paired flips 为 `27/300`，`C4` 为 `10/300`；prompt boundary 中 LLaVA Original C3/C3-v2/C3-v3 分别为 `27/300`、`5/300`、`0/300`；visual clarity audit manifest 为 `54` 行且所有 image paths 存在。

## 8. 不再使用的旧入口

以下旧路径不再属于当前 GitHub 默认工作流：

- 旧 `current` 配置树
- 旧 `current` prompt 树
- 旧分析目录
- 旧输出目录
- 旧预览与 review 目录
- 旧可视化预览页面
- 旧一键流水线
