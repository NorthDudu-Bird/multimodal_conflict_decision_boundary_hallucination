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

## 6. Derived Strengthening And Phase 2 Diagnostics

The scripts below are split into two classes. The first class reads existing canonical parsed/metrics outputs and does not call model inference. The second class runs Phase 2 local diagnostic inference in separate directories without overwriting canonical C0-C4/A1-A2 outputs.

### 6.1 Derived-only strengthening

```bash
python scripts/generate_paired_flip_analysis.py
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_visual_clarity_audit.py
python scripts/generate_color_split_analysis.py
python scripts/generate_visual_clarity_completed_audit.py
python scripts/generate_phase2_synthesis.py
```

Key outputs:

- same-image paired analysis: `results/main/paired_flip_summary.md`
- prompt wording boundary control: `results/robustness/prompt_boundary_summary.md`
- per-color split: `results/color_split/color_split_summary.md`
- completed visual clarity audit: `results/audit/visual_clarity_audit_manifest_completed.csv`, `results/audit/visual_clarity_audit_summary.md`
- failure taxonomy and casebook: `results/case_analysis/*`
- gatekeeping protocol: `docs/gatekeeping_protocol.md`, `results/gatekeeping/*`

### 6.2 Phase 2 local diagnostic inference

Run smoke tests before full local diagnostic inference:

```bash
python scripts/run_phase2_diagnostics.py --family factorization --limit 12
python scripts/run_phase2_diagnostics.py --family format_control --limit 12
python scripts/run_phase2_diagnostics.py --family multiturn --limit 12
```

Then run the three full modules:

```bash
python scripts/run_phase2_diagnostics.py --family factorization
python scripts/run_phase2_diagnostics.py --family format_control
python scripts/run_phase2_diagnostics.py --family multiturn
```

Key outputs:

- factorization: `results/factorization/factorized_prompt_metrics.csv`, `results/factorization/factorized_prompt_summary.md`
- answer-format control: `results/format_control/format_control_metrics.csv`, `results/format_control/format_control_summary.md`
- multi-turn extension: `results/multiturn/multiturn_metrics.csv`, `results/multiturn/multiturn_summary.md`

## 7. Acceptance Criteria

`python scripts/verify_reproducibility.py` checks the locked canonical artifacts. After Phase 2 writing-summary updates, `results/final_result_summary.md` may intentionally differ from the old snapshot because it contains a new Phase 2 addendum. This is a writing-summary update, not a change to canonical parsed outputs or metrics.

Required checks:

- `C0` remains perfectly visually faithful for all three models.
- `results/main/main_key_tests.csv` remains 12 rows.
- `results/robustness/prompt_variant_metrics.csv` remains 9 rows.
- `results/parser/ambiguous_outputs_sample.csv` remains 27 rows.
- `results/appendix/stanford_core_sanity_check.csv` remains 12 rows.
- LLaVA `C3` paired flips remain `27/300`; `C4` remains `10/300`.
- Prompt boundary LLaVA Original C3/C3-v2/C3-v3 remains `27/300`, `5/300`, and `0/300`.
- Phase 2 row-count checks: color split main metrics `90` rows; color paired metrics `72`; completed audit manifest `84`; factorization combined `9000`; format-control combined `9900`; multi-turn combined `5400`; gatekeeping table `8`.
- Completed visual clarity audit has no missing image paths.
- Final writing pack: `deliverables/gpt_paper_writing_pack_20files_final.zip` plus `deliverables/gpt_paper_writing_pack_20files_manifest.md`.

## 8. Deprecated Entrypoints

The following old paths are no longer part of the current official workflow:

- old `current` config tree
- old `current` prompt tree
- old analysis directories
- old output directories
- old preview/review directories
- old visualization preview page
- old one-command pipeline
- old 25-file writing packs as final delivery entrypoints
