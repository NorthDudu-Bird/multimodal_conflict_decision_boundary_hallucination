# Model Checkpoints and Hardware

## Checkpoints Used

- `qwen2vl7b`
  Model name: `Qwen2-VL-7B-Instruct`
  Checkpoint: `Qwen/Qwen2-VL-7B-Instruct`
  Precision: `bfloat16`
- `llava15_7b`
  Model name: `LLaVA-1.5-7B`
  Checkpoint: `llava-hf/llava-1.5-7b-hf`
  Precision: `float16`
- `internvl2_8b`
  Model name: `InternVL2-8B`
  Checkpoint: `OpenGVLab/InternVL2-8B`
  Precision: `bfloat16`

These are the actual checkpoints recorded in the per-run metadata files under each output directory.

## Hardware and Runtime

- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- GPU memory: `12282 MiB`
- Driver: `577.00`
- Runtime policy: one model loaded at a time
- Batch size: `1`
- Device mapping: `auto`
- Effective device recorded in metadata: `cuda:0+cpu_offload`

Because the available VRAM was not sufficient to comfortably hold all three full-precision models with larger batches, the pipeline runs them sequentially and uses CPU offload where needed. After each stage, results are saved to disk before the next model is loaded.

## Why Quantization Was Not Used

This run intentionally avoids:

- 4-bit quantization
- 8-bit quantization
- AWQ
- GPTQ
- bitsandbytes lightweight loading

The reason is experimental comparability. The goal of the v2 study is to compare model behavior under a common prompt and parsing framework without introducing quantization as a hidden source of variation. When memory pressure arose, we reduced throughput rather than changing model fidelity: batch size stayed at `1`, models were run sequentially, and CPU offload was used instead of low-bit weights.

## Run Metadata Files

The main metadata files for the reviewed-truth primary runs are:

- `outputs/qwen2vl7b_v2_primary/run_metadata.json`
- `outputs/llava15_7b_v2_primary/run_metadata.json`
- `outputs/internvl2_8b_v2_primary/run_metadata.json`

Equivalent metadata files are also written for the smoke and auxiliary runs.
