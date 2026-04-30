# Start Here

如果你只关心当前论文定版版本，请按下面顺序看：

1. [README.md](README.md)
2. [docs/reproduction.md](docs/reproduction.md)
3. [docs/strengthening_master_plan.md](docs/strengthening_master_plan.md)
4. [results/final_result_summary.md](results/final_result_summary.md)
5. [results/main/main_results_paper_ready.md](results/main/main_results_paper_ready.md)
6. [results/main/paired_flip_summary.md](results/main/paired_flip_summary.md)
7. [results/robustness/prompt_boundary_summary.md](results/robustness/prompt_boundary_summary.md)
8. [results/auxiliary/aux_role_note.md](results/auxiliary/aux_role_note.md)
9. [results/threats_to_validity_summary.md](results/threats_to_validity_summary.md)

唯一官方实验重建入口：

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

论文级派生补强入口：

```bash
python scripts/generate_paired_flip_analysis.py
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_visual_clarity_audit.py
python scripts/build_writing_pack.py --pack-date 20260430
```

不要再使用旧的 `current` 配置树、旧 `current` prompt 树、旧可视化预览页面或 Stanford-only 并列主流程。
