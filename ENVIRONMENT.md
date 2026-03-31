# Environment Reproduction Guide

This project has been verified locally with the following environment for the Qwen2-VL experiment pipeline.

## Verified runtime
- OS: Windows 10/11
- Python: `3.11.2`
- PyTorch: `2.6.0+cu126`
- torchvision: `0.21.0+cu126`
- CUDA available: `True`
- GPU used in verification: `NVIDIA GeForce RTX 4080 Laptop GPU (12 GB)`
- Recommended model loading mode on 12 GB VRAM: `4-bit NF4`

## Recommended setup
1. Create the environment:

```powershell
conda create -n cv_proj python=3.11.2 -y
conda activate cv_proj
```

2. Install the Python packages:

```powershell
pip install -r requirements.txt
```

3. Download the model weights to the project-local path expected by the scripts:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-VL-7B-Instruct', local_dir='models/qwen2_vl_7b')"
```

4. Run the environment check:

```powershell
python scripts/check_vlm_env.py
```

5. Run the 20-row smoke test:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_qwen2vl_smoke_test.py
```

## Full experiment command
```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_qwen2vl_batch.py --batch-size 1 --max-new-tokens 96 --temperature 0.0
```

## Notes
- The smoke test and batch script expect the model directory to be `models/qwen2_vl_7b/`.
- On a 12 GB GPU, keep `batch_size=1` unless you have verified larger values locally.
- The verified run in this repo used `Qwen2-VL-7B-Instruct` with `bitsandbytes` 4-bit quantization.
- In my local session, `cv_proj` itself was not writable, so I temporarily used a project-local dependency overlay. That overlay is not required for a clean reproduction. A fresh environment can install everything directly from `requirements.txt`.
