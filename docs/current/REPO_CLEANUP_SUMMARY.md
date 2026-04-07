# Repo Cleanup Summary

## What Changed

This cleanup makes the `strict-colors` multimodel workflow the single obvious current mainline and moves older `v2`, older `S0-S7`, old viewers, old results, and old docs into clearer archive locations.

## Final Mainline

The current mainline is:

- `strict-colors` Stanford Cars
- three-model matched workflow: `Qwen2-VL-7B / LLaVA-1.5-7B / InternVL2-8B`
- strict faithful definition: `parsed_label == true_color`
- canonical config: `configs/current/restructured_experiment_strict_colors.yaml`

## Directory Reorganization

Added and activated:

- `configs/current/`
- `configs/archived/`
- `docs/current/`
- `docs/archived/`
- `prompts/current/`
- `prompts/archived/`
- `analysis/current/`
- `analysis/archived/`
- `outputs/current/`
- `outputs/archived/`
- `reports/current/`
- `reports/archived/`
- `annotation/current/`
- `annotation/archived/`

## Script Reorganization

Current scripts are now grouped by function:

- `scripts/data_prep/`
- `scripts/data_prep/bootstrap/`
- `scripts/inference/`
- `scripts/parsing/`
- `scripts/analysis/`
- `scripts/utils/`

Most-used current files:

- `scripts/data_prep/prepare_stanford_cars_multimodel_v2.py`
- `scripts/inference/run_multimodel_stanford_cars_pipeline_v2.py`
- `scripts/parsing/parse_restructured_car_color_outputs.py`
- `scripts/analysis/analyze_multimodel_car_color_results.py`

## Archived Material

Moved but kept:

- old `v2` prompts to `prompts/archived/`
- old `v2` analyses to `analysis/archived/`
- old `v2` outputs to `outputs/archived/`
- old viewers / previews / setup reports to `reports/archived/`
- old md docs to `docs/archived/`
- old annotation copies to `annotation/archived/`
- old logs to `archives/logs_snapshot_2026-04-07/`
- older legacy material remains in `archives/`

## Removed Or Cleared

- `__pycache__` directories under the active script tree
- root `logs/` contents, leaving the directory available for future fresh runs
- non-mainline clutter from the repository root was moved out of the way

## Recommended Starting Order

1. `RUN_THIS_FIRST.md`
2. `PROJECT_INDEX.md`
3. `docs/current/WHAT_TO_READ_FOR_PAPER.md`
4. `scripts/inference/run_multimodel_stanford_cars_pipeline_v2.py` for reruns
