# Start Here

If you only want the current active experiment, use the merged main files below and ignore the rest.

## 1. Final Chinese summary

- `results_summary/current/vcor_balanced_rerun/paper_ready_results_summary.md`

## 2. Main preview HTML

- `reports/current/vcor_balanced_multimodel_results_viewer.html`

This is the default viewer now.
It mixes Stanford Cars and VCoR in one place and shows `source_dataset` and `source_split` directly for each image.

## 3. Main official manifest

- `data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv`

This is the default manifest now.
You do not need separate dataset files just to know where an image came from.

## 4. Active main config

- `configs/current/restructured_experiment_vcor_balanced.yaml`

## 5. Main result tables

- `analysis/current/vcor_balanced_primary/`
- `analysis/current/vcor_balanced_auxiliary/`

## 6. Optional control-only materials

- `reports/current/stanford_core_multimodel_results_viewer.html`
- `configs/current/restructured_experiment_stanford_core_vcor_robustness.yaml`
- `analysis/current/stanford_core_primary/`
- `analysis/current/stanford_core_auxiliary/`
- `data/processed/stanford_cars/primary_core_stanford_only.csv`

Keep these only for robustness checks or historical alignment.

## Ignore for now

- `clean/`
- `archives/` if it shows up again elsewhere
- old `strict_colors` and `v3` material already moved out
- `vendor/`, `models/`, `logs/`, `annotation/` unless you are debugging
