# Environment Reproduction Guide

## Current Target Environment

This repository is currently organized around the VCoR-balanced multimodel rerun, with a Stanford-only control line kept for robustness checks.

Verified local setup:

- OS: Windows
- Python: `3.11`
- PyTorch: CUDA-enabled build
- Main GPU used in recent reruns: RTX 4080 Laptop GPU

## Recommended Setup

```powershell
conda create -n cv_proj python=3.11 -y
conda activate cv_proj
pip install -r requirements.txt
```

## Model Weights

Place or download the model weights under:

- `models/qwen2_vl_7b`
- `models/llava_1_5_7b_hf`
- `models/internvl2_8b`

## Current Mainline Commands

Prepare data and prompt tables:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/data_prep/prepare_stanford_cars_multimodel_v2.py --config configs/current/restructured_experiment_vcor_balanced.yaml --truth-source reviewed
```

Run the full VCoR-balanced multimodel pipeline:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/inference/run_multimodel_vcor_balanced_pipeline.py --config configs/current/restructured_experiment_vcor_balanced.yaml
```

Run the Stanford-only robustness control:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/inference/run_multimodel_vcor_balanced_pipeline.py --config configs/current/restructured_experiment_stanford_core_vcor_robustness.yaml --core-only
```

Regenerate the current results viewer:

```powershell
python scripts/analysis/generate_multimodel_results_viewer.py --viewer-mode vcor_balanced_rerun --primary-csv analysis/current/vcor_balanced_primary/primary_combined_parsed_results.csv --auxiliary-csv analysis/current/vcor_balanced_auxiliary/auxiliary_combined_parsed_results.csv --manifest-csv data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv --output-html reports/current/vcor_balanced_multimodel_results_viewer.html
```

## Notes

- The active analysis uses exact-match faithful scoring and keeps only the six-color clean subset in the primary analysis.
- `source_dataset` distinguishes Stanford Cars core samples from VCoR supplement samples in the merged manifest and viewer.
- Historical local cleanup archives may still exist under `clean/`, but they are not part of the current GitHub snapshot.
