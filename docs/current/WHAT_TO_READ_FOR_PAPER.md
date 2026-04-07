# What To Read For Paper

If you want to continue writing the paper now, read in this order.

## Read These First

1. `docs/current/results_ready_summary_v3.md`
2. `docs/current/method_ready_text_v3.md`
3. `analysis/current/primary_v3/analysis_summary.md`
4. `reports/current/strict_colors_multimodel_results_viewer.html`

The canonical preview entry in `reports/current/strict_colors_multimodel_results_viewer.html` now points to the latest v3 expanded rerun. The previous pre-v3 preview was preserved at `reports/archived/strict_colors_multimodel_results_viewer_pre_v3.html`.

## For Method

- Method-ready text: `docs/current/method_ready_text_v3.md`
- Experiment status: `docs/current/EXPERIMENT_STATUS_V3.md`
- Final manifest: `data/processed/stanford_cars/final_primary_manifest_v4_expanded.csv`
- Color distribution: `analysis/current/color_distribution_v4_expanded.md`

## For Results

Primary:

- `analysis/current/primary_v3/model_condition_metrics.csv`
- `analysis/current/primary_v3/summary_metrics.csv`
- `analysis/current/primary_v3/analysis_summary.md`

Auxiliary:

- `analysis/current/auxiliary_v3/model_condition_metrics.csv`
- `analysis/current/auxiliary_v3/summary_metrics.csv`
- `analysis/current/auxiliary_v3/analysis_summary.md`
- `analysis/current/auxiliary_v3/answer_space_compliance_metrics.csv`
- `analysis/current/auxiliary_v3/answer_space_behavior_breakdown.csv`

Cross-model:

- `analysis/current/cross_model_v3/README.md`
- `analysis/current/cross_model_v3/`

## For Figures

- `analysis/current/primary_v3/plots/`
- `analysis/current/auxiliary_v3/plots/`
- `analysis/current/cross_model_v3/`

Fastest overview:

- `reports/current/strict_colors_multimodel_results_viewer.html`

Direct v3 copy:

- `reports/current/strict_colors_multimodel_results_viewer_v3.html`

## For Data Screening

- Final manifest: `data/processed/stanford_cars/final_primary_manifest_v4_expanded.csv`
- Full exclusions: `data/processed/stanford_cars/excluded_records_v4_expanded.csv`
- Manual-review exclusions: `data/processed/stanford_cars/excluded_manual_review_v3.csv`

## If You Need To Rerun

Full workflow:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/inference/run_multimodel_stanford_cars_pipeline_v2.py --config configs/current/restructured_experiment_strict_colors_v3.yaml --truth-source reviewed
```

Data and prompt tables only:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/data_prep/prepare_stanford_cars_multimodel_v2.py --config configs/current/restructured_experiment_strict_colors_v3.yaml --truth-source reviewed
```

## What Not To Read First

Unless you are tracing history, do not start with:

- `docs/archived/`
- `analysis/archived/`
- `outputs/archived/`
- `reports/archived/`
- `archives/`
