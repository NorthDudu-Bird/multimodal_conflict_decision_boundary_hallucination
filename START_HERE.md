# Start Here

如果你只关心当前论文定版版本，请按下面顺序看：

1. [README.md](README.md)
2. [docs/reproduction.md](docs/reproduction.md)
3. [results/final_result_summary.md](results/final_result_summary.md)
4. [results/main/table1_main_metrics.md](results/main/table1_main_metrics.md)
5. [results/robustness/prompt_variant_summary.md](results/robustness/prompt_variant_summary.md)
6. [results/parser/label_mapping_audit.md](results/parser/label_mapping_audit.md)
7. [results/appendix/stanford_core_sanity_check.md](results/appendix/stanford_core_sanity_check.md)

唯一官方脚本入口：

```bash
python scripts/build_dataset.py
python scripts/run_baseline_c0.py
python scripts/run_main_c0_c4.py
python scripts/run_aux_a1_a2.py
python scripts/run_robustness_c3_prompt_variants.py
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

不要再使用旧的 `current` 配置树、旧 `current` prompt 树、旧可视化预览页面或 Stanford-only 并列主流程。
