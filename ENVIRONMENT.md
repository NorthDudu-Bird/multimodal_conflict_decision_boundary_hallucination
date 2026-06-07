# Environment Guide

## Verified Setup

- Windows 11
- Python 3.11
- NVIDIA GPU with a CUDA-enabled PyTorch build
- Last full rerun hardware: RTX 4080 Laptop GPU

Create the environment:

```powershell
conda create -n vlm-conflict python=3.11 -y
conda activate vlm-conflict
pip install -r requirements.txt
```

## Local Model Directories

Download model checkpoints separately and place them in:

- `models/qwen2_vl_7b`
- `models/llava_1_5_7b_hf`
- `models/internvl2_8b`

Model weights are local-only and must not be committed.

## Configuration

The canonical configuration is:

```text
configs/experiment.yaml
```

It defines dataset, prompt, model, output, and analysis paths used by the primary
pipeline.

## Execution Order

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/build_dataset.py
python scripts/run_baseline_c0.py --skip-build
python scripts/run_main_c0_c4.py --skip-build
python scripts/run_aux_a1_a2.py --skip-build
python scripts/run_robustness_c3_prompt_variants.py --skip-build
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

Original source datasets, model checkpoints, caches, and runtime logs are excluded
from version control.
