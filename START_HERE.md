# Start Here

如果你只关心当前论文定版版本，请按下面顺序看：

1. [README.md](README.md)
2. [docs/reproduction.md](docs/reproduction.md)
3. [docs/final_writing_interface_note.md](docs/final_writing_interface_note.md)
4. [docs/validity_control_plan.md](docs/validity_control_plan.md)
5. [docs/integrated_experiment_system_plan.md](docs/integrated_experiment_system_plan.md)
6. [docs/final_writing_pack_note.md](docs/final_writing_pack_note.md)
7. [results/integrated_experiment_summary.md](results/integrated_experiment_summary.md)
8. [results/final_result_summary.md](results/final_result_summary.md)
9. [results/main/main_results_paper_ready.md](results/main/main_results_paper_ready.md)
10. [results/main/paired_flip_summary.md](results/main/paired_flip_summary.md)
11. [results/robustness/prompt_boundary_summary.md](results/robustness/prompt_boundary_summary.md)
12. [results/auxiliary/aux_role_note.md](results/auxiliary/aux_role_note.md)
13. [results/threats_to_validity_summary.md](results/threats_to_validity_summary.md)

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
python scripts/generate_visual_clarity_completed_audit.py
python scripts/generate_color_split_analysis.py
python scripts/run_controlled_diagnostics.py --family factorization
python scripts/run_controlled_diagnostics.py --family format_control
python scripts/run_controlled_diagnostics.py --family multiturn
python scripts/generate_integrated_synthesis.py
python scripts/build_final_writing_pack.py
```

不要再使用旧的 `current` 配置树、旧 `current` prompt 树、旧可视化预览页面或 Stanford-only 并列主流程。

为后续 `nature-*` skills 接入，优先使用 `docs/final_writing_interface_note.md`、
`results/integrated_experiment_summary.md` 和 `deliverables/gpt_paper_writing_pack_20files_final.zip`。
`results/final_result_summary.md` 是写作型整合摘要，不作为阻断式复现门禁的实验锁定表。
