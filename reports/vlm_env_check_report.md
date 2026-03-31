# VLM Environment Check Report

## 基础环境
- Python: `3.11.2`
- 平台: `Windows-10-10.0.26200-SP0`
- Torch: `2.6.0+cu126`
- Torch 来源: `D:\anaconda3\envs\cv_proj\Lib\site-packages\torch\__init__.py`
- CUDA 可用: `True`
- CUDA 版本: `12.6`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- 显存(GB): `11.99`
- BF16 支持: `True`
- 本地依赖覆盖层: `D:\multimodal_conflict_decision_boundary_hallucination\vendor\cv_proj_sitepackages`

## 依赖检查
- 已安装 `torch`: `2.6.0+cu126`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\torch\__init__.py`
- 已安装 `torchvision`: `0.21.0+cu126`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\torchvision\__init__.py`
- 已安装 `transformers`: `4.51.3`，来源 `D:\multimodal_conflict_decision_boundary_hallucination\vendor\cv_proj_sitepackages\transformers\__init__.py`
- 已安装 `accelerate`: `1.8.1`，来源 `D:\multimodal_conflict_decision_boundary_hallucination\vendor\cv_proj_sitepackages\accelerate\__init__.py`
- 已安装 `safetensors`: `0.6.2`，来源 `D:\multimodal_conflict_decision_boundary_hallucination\vendor\cv_proj_sitepackages\safetensors\__init__.py`
- 已安装 `pillow`: `11.1.0`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\PIL\__init__.py`
- 已安装 `pandas`: `2.2.2`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\pandas\__init__.py`
- 已安装 `tqdm`: `4.67.1`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\tqdm\__init__.py`
- 已安装 `bitsandbytes`: `0.49.2`，来源 `D:\multimodal_conflict_decision_boundary_hallucination\vendor\cv_proj_sitepackages\bitsandbytes\__init__.py`
- 已安装 `sentencepiece`: `0.2.1`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\sentencepiece\__init__.py`
- 已安装 `qwen_vl_utils`: `unknown`，来源 `D:\multimodal_conflict_decision_boundary_hallucination\vendor\cv_proj_sitepackages\qwen_vl_utils\__init__.py`
- 已安装 `huggingface_hub`: `0.34.4`，来源 `D:\anaconda3\envs\cv_proj\Lib\site-packages\huggingface_hub\__init__.py`

## 7B 可运行性评估
- 结论: `更适合先降级到 2B 或增加量化/卸载方案`
- 显存约 11.99 GB，7B 在本机上风险较高。
- bitsandbytes 可导入，具备尝试 4-bit/8-bit 量化的基础条件。

## 建议
- 若继续部署 Qwen2-VL-7B-Instruct，优先尝试 4-bit 量化。
- 若 Windows 下 bitsandbytes 无法稳定工作，可改为 CPU offload 或退到 Qwen2-VL-2B-Instruct。
