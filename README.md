# Balanced Car-Color Conflict Mainline

This repository is now organized around one paper mainline only:

- final balanced evaluation set
- C0 baseline
- C0-C4 main experiment
- A1/A2 auxiliary experiment
- three-model comparison: `LLaVA-1.5-7B`, `Qwen2-VL-7B-Instruct`, `InternVL2-8B`

Start here:

- `docs/experiment_plan.md`
- `docs/reproduction.md`
- `docs/project_audit.md`
- `results/main/table1_main_metrics.md`
- `results/main/figure2_conflict_aligned_rates.png`
- `results/auxiliary/table3_aux_metrics.md`
- `results/appendix/dataset_distribution.png`

Official paper-facing entrypoints:

- `python scripts/build_dataset.py`
- `python scripts/run_baseline_c0.py`
- `python scripts/run_main_c0_c4.py`
- `python scripts/run_aux_a1_a2.py`
- `python scripts/make_figures.py`

Official paper-facing assets:

- `data/balanced_eval_set/final_manifest.csv`
- `prompts/c0_c4/c0_baseline_prompts.csv`
- `prompts/c0_c4/main_c0_c4_prompts.csv`
- `prompts/a1_a2/a1_a2_prompts.csv`
- `results/baseline/`
- `results/main/`
- `results/auxiliary/`
- `results/appendix/`

Deprecated or down-scoped material has either been archived or removed from the default workflow. The old Stanford-only control is appendix-only and is no longer a primary entrypoint.
