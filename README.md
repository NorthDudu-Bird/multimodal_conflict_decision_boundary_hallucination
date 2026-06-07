# Multimodal Conflict Decision Boundary Hallucination

Code, prompts, evaluation manifests, and structured outputs for studying how
misleading text affects visual-language model judgments in a controlled car-color
classification task.

This public repository is limited to research code and reproducibility artifacts.
Manuscripts, submission files, reviewer material, private notes, model weights, and
third-party source datasets are intentionally excluded.

## Repository Layout

| Path | Contents |
| --- | --- |
| `configs/` | Experiment configuration files |
| `data/annotations/` | Annotation and adjudication tables |
| `data/balanced_eval_set/` | Final evaluation manifests |
| `data/metadata/` | Dataset preparation and labeling metadata |
| `data/processed/` | Small processed evaluation assets required by the pipeline |
| `data_external/` | Publicly shareable external-dataset metadata only |
| `prompts/` | Baseline, conflict, robustness, and diagnostic prompts |
| `scripts/data_prep/` | Dataset preparation utilities |
| `scripts/inference/` | Model loading and batch inference |
| `scripts/parsing/` | Output parsing |
| `scripts/*.py` | Analysis, audit, and figure-generation entry points |
| `results/` | Raw model outputs, parsed tables, statistics, audits, and plots |

## Environment

The verified environment uses Python 3.11 and an NVIDIA CUDA stack. Install the
pinned dependencies in a fresh environment:

```powershell
conda create -n vlm-conflict python=3.11 -y
conda activate vlm-conflict
pip install -r requirements.txt
```

See `ENVIRONMENT.md` for model-directory conventions and the full execution order.

## Data and Models

Model weights and original third-party datasets are not included. Place local model
checkpoints under:

```text
models/qwen2_vl_7b/
models/llava_1_5_7b_hf/
models/internvl2_8b/
```

The repository contains the evaluation manifests and processed assets needed to
identify the analyzed samples. Users remain responsible for obtaining source
datasets and model checkpoints under their respective licenses.

## Reproduce the Analysis

The main configuration is `configs/experiment.yaml`. To rebuild the dataset and
run the primary experiment:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/build_dataset.py
python scripts/run_baseline_c0.py --skip-build
python scripts/run_main_c0_c4.py --skip-build
python scripts/run_aux_a1_a2.py --skip-build
python scripts/run_robustness_c3_prompt_variants.py --skip-build
```

To regenerate analyses and validation artifacts from the stored outputs:

```powershell
python scripts/analyze_results.py
python scripts/generate_paired_flip_analysis.py
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_color_split_analysis.py
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

Controlled diagnostic families can be regenerated with:

```powershell
python scripts/run_controlled_diagnostics.py --family factorization
python scripts/run_controlled_diagnostics.py --family format_control
python scripts/run_controlled_diagnostics.py --family multiturn
```

## Key Reproducibility Artifacts

- `data/balanced_eval_set/final_manifest.csv`
- `results/main/main_condition_metrics.csv`
- `results/main/main_key_tests.csv`
- `results/main/paired_flip_metrics.csv`
- `results/robustness/prompt_boundary_metrics.csv`
- `results/parser/label_mapping_audit.md`
- `results/reproducibility_audit.md`
- `docs/reproduction.md`

## Privacy and Repository Hygiene

Do not commit manuscripts, submission packages, reviewer correspondence, personal
notes, credentials, local absolute paths, model weights, or restricted source data.
The ignore rules cover common document, archive, secret, cache, model, and temporary
output formats; review every staged change before pushing.
